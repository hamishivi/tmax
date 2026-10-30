"""CPU tensor integration: masked groups -> real packing -> real DPPO loss."""

import __future__

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_production(filename, names, namespace):
    path = Path(__file__).parents[1] / "open_instruct" / filename
    functions = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec", __future__.annotations.compiler_flag),
        namespace,
    )


class RewardPackingLossTest(unittest.TestCase):
    def test_small_groups_preserve_token_advantages_and_gradients(self):
        namespace = dict(
            np=np,
            torch=torch,
            PackedSequences=SimpleNamespace,
            GRPOLossType=SimpleNamespace(dapo="dapo", cispo="cispo", dppo="dppo", tvpo="tvpo"),
        )
        load_production("data_loader.py", {"compute_group_advantages"}, namespace)
        load_production("rl_utils.py", {"pack_sequences", "reset_position_ids"}, namespace)
        load_production("grpo_utils.py", {"compute_grpo_loss", "compute_dppo_policy_loss"}, namespace)
        scores = np.array([0, 999, 0.5, 1, 1, 999, 999, 0])
        valid = np.array([True, False, True, True, True, False, False, True])
        kept = np.flatnonzero(valid).tolist()
        advantages = namespace["compute_group_advantages"](scores, 4, "centered", valid)[valid]
        expected_advantages = [-0.5, 0, 0.5, 0.5, -0.5]
        np.testing.assert_allclose(advantages, expected_advantages)
        responses = [[20] * (i % 3 + 1) for i in kept]
        for pack_length in (5, 100):
            for min_batches in (1, 3):
                packed = namespace["pack_sequences"](
                    queries=[[10]] * len(kept),
                    responses=responses,
                    masks=[[1] * len(r) for r in responses],
                    pack_length=pack_length,
                    pad_token_id=0,
                    vllm_logprobs=[[0.0] * len(r) for r in responses],
                    rollout_sample_ids=kept,
                    min_num_batches=min_batches,
                )
                lookup = torch.tensor(np.r_[0, advantages], dtype=torch.float64)
                parameters = torch.full((8,), 0.1, dtype=torch.float64, requires_grad=True)
                loss = torch.tensor(0.0, dtype=torch.float64)
                seen = set()
                for mask, ids in zip(packed.response_masks, packed.rollout_sample_ids):
                    token_advantages = lookup[mask]
                    token_logprobs = parameters[ids]
                    _, _, token_loss, _ = namespace["compute_grpo_loss"](
                        token_logprobs,
                        token_logprobs.exp(),
                        token_advantages,
                        None,
                        SimpleNamespace(loss_fn="dppo", dppo_ratio_cap=10.0),
                    )
                    loss = loss + token_loss[mask > 0].sum()
                    seen.update(ids[mask > 0].tolist())
                loss.backward()
                expected_grad = np.zeros(8)
                for i, adv, response in zip(kept, expected_advantages, responses):
                    expected_grad[i] = -adv * np.exp(0.1) * len(response)
                np.testing.assert_allclose(parameters.grad.numpy(), expected_grad)
                self.assertEqual(seen, set(kept))
                self.assertEqual({i // 4 for i in seen}, {0, 1})
