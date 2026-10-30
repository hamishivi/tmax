"""Sandbox connection settings must reach Ray actors in reduced environments."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def runtime_env_builder():
    source = Path(__file__).parents[1] / "open_instruct/grpo_fast.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ray_runtime_env_from_environment"
    )
    namespace = {
        "os": os,
        "Any": Any,
        "EXCLUDED_ENV_VARS": {"CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"},
        "logger": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def test_minimal_environment_preserves_sandbox_connections(monkeypatch, runtime_env_builder):
    settings = {
        "SANDFLEET_URL": "http://controller.example:8765",
        "SANDFLEET_POOL": "training-smoke",
        "SANDFLEET_CLIENT_TOKEN": "example-client-token",
        "SANDFLEET_REGISTRY": "/data/runtime/controller.json",
        "SWERL_APPTAINER_SIF_DIR": "/data/sif-pool",
        "SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP": "/data/verifier-timeouts.json",
        "HTTP_PROXY": "http://proxy.example:8080",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "ALL_PROXY": "http://proxy.example:8080",
        "NO_PROXY": "controller.example,.workers.example",
        "http_proxy": "http://proxy.example:8080",
        "https_proxy": "http://proxy.example:8080",
        "all_proxy": "http://proxy.example:8080",
        "no_proxy": "controller.example,.workers.example",
    }
    monkeypatch.setattr(
        os,
        "environ",
        {
            **settings,
            "OPEN_INSTRUCT_RAY_RUNTIME_ENV": "minimal",
            "CUDA_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "0",
            "SANDFLEET_TOKEN": "example-admin-token",
            "UNRELATED_SECRET": "example-secret",
        },
    )
    assert runtime_env_builder()["env_vars"] == settings


@pytest.mark.parametrize("mode", ["full", "unexpected"])
def test_full_environment_excludes_device_assignments(monkeypatch, runtime_env_builder, mode):
    monkeypatch.setattr(
        os, "environ", {"OPEN_INSTRUCT_RAY_RUNTIME_ENV": mode, "SANDFLEET_POOL": "test", "CUDA_VISIBLE_DEVICES": "0"}
    )
    assert runtime_env_builder()["env_vars"] == {"OPEN_INSTRUCT_RAY_RUNTIME_ENV": mode, "SANDFLEET_POOL": "test"}


def test_disabled_environment_skips_runtime_override(monkeypatch, runtime_env_builder):
    monkeypatch.setattr(os, "environ", {"OPEN_INSTRUCT_RAY_RUNTIME_ENV": "off"})
    assert runtime_env_builder() is None
