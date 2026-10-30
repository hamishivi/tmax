"""Exercise production reference checkpoint validation with CPU tensors."""

import ast
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from open_instruct import model_utils


def save_reference():
    source = Path(__file__).parents[1] / "open_instruct/grpo_fast.py"
    definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "gather_and_write_ref_policy"
    )
    namespace = {"torch": torch, "os": os, "logger": Mock()}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[definition.name]


def test_save_uses_complete_gathered_state(tmp_path):
    state = {"weight": torch.ones(2, 3), "buffer": torch.arange(3)}
    gather = Mock(return_value=state)
    save_reference()(SimpleNamespace(_zero3_consolidated_16bit_state_dict=gather), str(tmp_path), 0, 3)
    gather.assert_called_once_with()
    loaded = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    for name, expected in state.items():
        torch.testing.assert_close(loaded[name], expected)


def test_save_rejects_empty_partition_and_does_not_write(tmp_path):
    engine = SimpleNamespace(_zero3_consolidated_16bit_state_dict=lambda: {"weight": torch.empty(0)})
    with pytest.raises(RuntimeError, match="numel=0"):
        save_reference()(engine, str(tmp_path), 0, 3)
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_save_reports_write_failure(tmp_path, monkeypatch):
    engine = SimpleNamespace(_zero3_consolidated_16bit_state_dict=lambda: {"weight": torch.ones(2)})
    monkeypatch.setattr(torch, "save", Mock(side_effect=OSError("write failed")))
    with pytest.raises(RuntimeError, match="write failed"):
        save_reference()(engine, str(tmp_path), 0, 3)


@pytest.fixture
def partitioned_model(monkeypatch):
    model = torch.nn.Linear(2, 1)
    model.register_buffer("cache", torch.ones(1))
    for parameter in model.parameters():
        parameter.ds_status = object()
    monkeypatch.setattr(model_utils, "GatheredParameters", lambda *args, **kwargs: nullcontext())
    return model


def test_load_restores_valid_parameters_and_buffers(partitioned_model, tmp_path):
    state = {name: torch.full_like(value, 3) for name, value in partitioned_model.state_dict().items()}
    path = tmp_path / "reference.bin"
    torch.save(state, path)
    model_utils.maybe_load_checkpoint(partitioned_model, str(path), torch.device("cpu"), 0)
    for name, value in partitioned_model.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_bad_shape_is_rejected_before_any_mutation(partitioned_model, tmp_path):
    before = {name: value.clone() for name, value in partitioned_model.state_dict().items()}
    state = {name: torch.full_like(value, 7) for name, value in before.items()}
    state["bias"] = torch.ones(5)
    path = tmp_path / "reference.bin"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="rejected before load"):
        model_utils.maybe_load_checkpoint(partitioned_model, str(path), torch.device("cpu"), 0)
    for name, value in partitioned_model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_failure_after_mutation_is_fatal_even_with_fallback(partitioned_model, tmp_path, monkeypatch):
    path = tmp_path / "reference.bin"
    torch.save(partitioned_model.state_dict(), path)

    def broken_load(state):
        with torch.no_grad():
            partitioned_model.weight.fill_(9)
        raise RuntimeError("load hook failed")

    monkeypatch.setattr(partitioned_model, "load_state_dict", broken_load)
    with pytest.raises(model_utils.ReferencePolicyCorruption, match="AFTER mutation"):
        model_utils.maybe_load_checkpoint(partitioned_model, str(path), torch.device("cpu"), 0, throw_on_error=False)
