"""CPU tests of production packing/collation and loss, without the GPU launcher imports."""

import __future__

import ast
import dataclasses
import enum
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar

import numpy as np
import pytest
import torch


def production_namespace():
    # Follow test_reward_packing_loss: execute the real definitions with CPU
    # dependencies, without importing vLLM/DeepSpeed/Ray on the test host.
    ns = dict(
        torch=torch,
        math=math,
        dist=torch.distributed,
        np=np,
        enum=enum,
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        fields=dataclasses.fields,
        replace=dataclasses.replace,
        Generic=Generic,
        T=TypeVar("T"),
    )
    definitions = {
        "data_types.py": {"CollatedBatchData"},
        "rl_utils.py": {
            "PackedSequences",
            "pack_sequences",
            "pad_packed_sequences",
            "reset_position_ids",
            "masked_mean",
        },
        "data_loader.py": {"collate_fn", "prepare_collated_data_for_workers"},
        "grpo_utils.py": {
            "GRPOLossType",
            "DPPODivergenceType",
            "compute_dppo_mask",
            "compute_dppo_ratio",
            "compute_dppo_policy_loss",
            "compute_policy_gradient_weights",
            "PolicyDriftAccumulator",
            "_fixed_histogram",
            "_histogram_quantile",
            "_top_fraction_mass_share",
            "compute_binary_divergence",
            "compute_grpo_loss",
            "deepspeed_gradient_reduction_divisor",
            "tiled_grpo_loss_scale",
            "TiledGRPOLMHeadLoss",
            "tiled_grpo_lm_head_loss",
            "_sequence_id_counts",
            "_sequence_loss_weights",
            "_count_sequences",
            "sequence_weighted_mean",
            "compute_metrics_from_loss_stats",
        },
    }
    for filename, names in definitions.items():
        path = Path(__file__).parents[1] / "open_instruct" / filename
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("POLICY_") for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", __future__.annotations.compiler_flag),
            ns,
        )
        if filename == "data_types.py":
            ns["data_types"] = SimpleNamespace(CollatedBatchData=ns["CollatedBatchData"])
    return ns


@pytest.fixture
def prod():
    return production_namespace()


def make_packs(prod, count):
    if count == 0:
        return prod["PackedSequences"]([], [], [], [], position_ids=[], advantages=[], vllm_logprobs=[])
    packed = prod["pack_sequences"](
        queries=[[1]] * count,
        responses=[[2, 3]] * count,
        masks=[[1, 1]] * count,
        pack_length=3,
        pad_token_id=0,
        vllm_logprobs=[[-1.0, -1.5]] * count,
        rollout_sample_ids=list(range(count)),
        model_steps=[7] * count,
    )
    packed.advantages = [m.float() * 0.1 for m in packed.response_masks]
    packed.rewards = [m.float() * 0.2 for m in packed.response_masks]
    return packed


@pytest.mark.parametrize("count", [0, 1, 5, 6, 7, 73])
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_every_real_token_survives_balanced_collation(prod, count, microbatch):
    packed = make_packs(prod, count)
    original = {
        f.name: [v.clone() for v in getattr(packed, f.name)]
        for f in dataclasses.fields(packed)
        if getattr(packed, f.name) is not None and f.name != "original_responses"
    }
    workers = prod["prepare_collated_data_for_workers"](packed, 6, microbatch, 0, pin_memory=False)
    assert len(workers) == 6
    assert len({len(w.query_responses) for w in workers}) == 1
    assert len({tuple(t.shape[0] for t in w.query_responses) for w in workers}) == 1
    seen = []
    for worker in workers:
        for i, mask in enumerate(worker.response_masks):
            valid = mask.bool()
            seen.extend(worker.rollout_sample_ids[i][valid].tolist())
            dummy_rows = ~valid.any(dim=1)
            if dummy_rows.any():
                # Model inputs retain a valid document, but no supervision or identity.
                assert worker.attention_masks[i][dummy_rows].any()
                for name in ("response_masks", "prompt_masks", "advantages", "vllm_logprobs", "rewards", "dones"):
                    assert not getattr(worker, name)[i][dummy_rows].any()
                for name in ("rollout_sample_ids", "model_steps"):
                    assert (getattr(worker, name)[i][dummy_rows] == -1).all()
    assert sorted(seen) == sorted(list(range(count)) * 2)
    assert len(packed.original_responses) == count
    for name, values in original.items():
        assert len(getattr(packed, name)) == len(values)
        for old, new in zip(values, getattr(packed, name), strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0, equal_nan=True)


def test_optional_fields_and_noop(prod):
    packed = make_packs(prod, 6)
    assert prod["pad_packed_sequences"](packed, 6) is packed
    packed = make_packs(prod, 1)
    packed.prompt_masks = packed.rollout_sample_ids = packed.model_steps = None
    packed.rewards = packed.dones = None
    workers = prod["prepare_collated_data_for_workers"](packed, 6, 1, 0, pin_memory=False)
    assert all(w.rewards is None and w.dones is None for w in workers)
    assert sum(m.sum().item() for w in workers for m in w.response_masks) == 2
    with pytest.raises(ValueError, match="positive"):
        prod["pad_packed_sequences"](packed, 0)


@pytest.mark.parametrize("mode", ["token", "sequence"])
@pytest.mark.parametrize("count", [1, 7, 73])
def test_tiled_dppo_padded_distribution_preserves_loss_and_gradients(prod, mode, count):
    packed = make_packs(prod, count)
    torch.manual_seed(4)
    weight = torch.randn(4, 3) * 0.1

    def evaluate(world_size):
        workers = prod["prepare_collated_data_for_workers"](packed, world_size, 1, 0, pin_memory=False)
        head = torch.nn.Linear(3, 4, bias=False)
        head.weight.data.copy_(weight)
        hidden = torch.randn(1, 2, 3, generator=torch.Generator().manual_seed(5))
        parameter = torch.nn.Parameter(hidden.clone())
        total_loss = 0.0
        denominator = count * (2 if mode == "token" else 1)
        for worker in workers:
            for i, mask in enumerate(worker.response_masks):
                mask = mask[:, 1:].bool()
                ids = worker.rollout_sample_ids[i][:, 1:]
                local_count = mask.sum() if mode == "token" else prod["_count_sequences"](mask, ids)
                outputs = prod["tiled_grpo_lm_head_loss"](
                    lm_head=head,
                    hidden_states=parameter,
                    selected_token_ids=worker.query_responses[i][:, 1:],
                    response_mask=mask,
                    advantages=worker.advantages[i][:, 1:],
                    old_logprobs=worker.vllm_logprobs[i][:, 1:],
                    ref_logprobs=None,
                    temperature=1.0,
                    beta=0.0,
                    clip_lower=0.2,
                    clip_higher=0.2,
                    shards=2,
                    loss_scale=local_count.float() / denominator,
                    loss_fn="dppo",
                    dppo_divergence_threshold=100.0,
                    loss_denominator=mode,
                    rollout_sample_ids=ids,
                )
                assert all(torch.isfinite(o).all() for o in outputs)
                if not mask.any():
                    assert all((o == 0).all() for o in outputs)
                outputs[0].backward()
                total_loss += outputs[0].item()
        return total_loss, head.weight.grad, parameter.grad

    reference = evaluate(1)
    padded = evaluate(6)
    for expected, actual in zip(reference, padded, strict=True):
        torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected), rtol=2e-5, atol=1e-6)
    assert reference[1].abs().sum() > 0


def test_zero_token_packs_do_not_affect_weighted_metrics(prod):
    result = prod["compute_metrics_from_loss_stats"](
        {"val/ratio": torch.tensor([2.0, 0.0]), "loss/policy_avg": torch.tensor([3.0, 0.0])}, torch.tensor([10.0, 0.0])
    )
    assert result["val/ratio"] == 2.0
    assert result["loss/policy_avg"] == 3.0
