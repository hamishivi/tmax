"""Local RL datasets must load before tool auto-discovery inspects their columns."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import datasets
import pytest


@pytest.fixture
def discover_tools():
    source = Path(__file__).parents[1] / "open_instruct/grpo_fast.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_discover_tools_from_datasets"
    )
    namespace = {
        "os": os,
        "datasets": datasets,
        "_hf_offline_mode_enabled": lambda: False,
        "TOOLS_COLUMN_KEY": "tools",
        "ENV_CONFIG_KEY": "env_config",
        "logger": SimpleNamespace(warning=Mock()),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
@pytest.mark.parametrize(
    "env_config", [{"env_name": "sandbox"}, {"env_configs": [{"env_name": "sandbox"}]}, [{"env_name": "sandbox"}]]
)
def test_discovers_local_dataset_tools(tmp_path, discover_tools, suffix, env_config):
    dataset = datasets.Dataset.from_list([{"tools": ["bash"], "env_config": env_config}])
    path = tmp_path / f"prompts{suffix}"
    if suffix == ".jsonl":
        dataset.to_json(str(path))
    else:
        dataset.to_parquet(str(path))

    assert discover_tools([str(path), "1.0"], ["train"]) == {"bash", "sandbox"}


def test_hub_dataset_keeps_requested_split(monkeypatch, discover_tools):
    dataset = datasets.Dataset.from_list([{"tools": ["bash"]}])
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(datasets, "load_dataset", loader)

    assert discover_tools(["example/tasks", "1.0"], ["validation"]) == {"bash"}
    loader.assert_called_once_with("example/tasks", split="validation")
