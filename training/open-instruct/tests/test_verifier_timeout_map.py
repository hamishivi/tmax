"""Per-task verifier timeout budgets.

The scalar `test_timeout` is wrong in two directions at once: its 600s floor inflates the
budgets of tasks that declared less, and the single value truncates the ~4900 CalibForge /
SETA / LiteCoder tasks that declared more. A truncated verifier returns 0 rather than
raising, so every case here is one that is silent in production.
"""

import asyncio
import json
import os
import tempfile

import pytest

from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv


def _env(**kw):
    return SWERLVanilluxSandboxEnv(backend="sandfleet", image="x.sif", **kw)


def _map_file(obj):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(obj, fh)
        return fh.name


def test_no_map_keeps_scalar_and_floor():
    """The three TMAX arms declare no per-task budgets and must be byte-identical."""
    env = _env(test_timeout=600)
    assert env._verifier_timeout_map == {}
    assert env._effective_test_timeout() == 600
    # The floor on the scalar path is unchanged.
    assert _env(test_timeout=60)._effective_test_timeout() == 600


def test_declared_budget_below_floor_is_honoured():
    """1186 CalibForge and 991 SETA tasks declare < 600s. The floor overrode all of them."""
    path = _map_file({"t1": 120.0})
    try:
        env = _env(test_timeout=600, verifier_timeout_map=path)
        env._task_test_timeout = env._verifier_timeout_map["t1"]
        assert env._effective_test_timeout() == 120.0
    finally:
        os.unlink(path)


def test_declared_budget_above_scalar_is_honoured():
    """The case that loses reward: a task asking 1800s was being killed at 600s."""
    path = _map_file({"t1": 1800.0})
    try:
        env = _env(test_timeout=600, verifier_timeout_map=path)
        env._task_test_timeout = env._verifier_timeout_map["t1"]
        assert env._effective_test_timeout() == 1800.0
    finally:
        os.unlink(path)


def test_unreadable_map_raises():
    with pytest.raises(RuntimeError, match="VERIFIER_TIMEOUT_MAP unreadable"):
        _env(verifier_timeout_map="/nonexistent/vt.json")


def test_env_var_is_honoured(monkeypatch):
    path = _map_file({"t1": 900.0})
    try:
        monkeypatch.setenv("SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP", path)
        assert _env()._verifier_timeout_map == {"t1": 900.0}
    finally:
        os.unlink(path)


@pytest.mark.parametrize("bad", [0, -1, -0.5, float("inf"), float("nan")])
def test_non_positive_or_non_finite_is_rejected_at_construction(bad):
    """Validated up front: a bad entry found lazily would crash one actor hours in."""
    path = _map_file({"ok": 300.0, "bad": bad})
    try:
        with pytest.raises(RuntimeError, match="non-positive or non-finite"):
            _env(verifier_timeout_map=path)
    finally:
        os.unlink(path)


def test_non_numeric_is_rejected_at_construction():
    path = _map_file({"t1": "fast"})
    try:
        with pytest.raises(RuntimeError, match="non-numeric"):
            _env(verifier_timeout_map=path)
    finally:
        os.unlink(path)


def test_absent_task_in_supplied_map_is_an_error():
    """An incomplete map must not fall through to the scalar.

    Falling through would silently reimpose the 600s cap on exactly the tasks whose
    declaration went missing -- the failure the map exists to remove -- and it would be
    invisible, because a truncated verifier scores 0 instead of raising.
    """
    path = _map_file({"t1": 300.0})
    try:
        env = _env(pwd="/", verifier_timeout_map=path)
        with pytest.raises(RuntimeError, match="absent from the supplied verifier timeout map"):
            asyncio.run(env.reset(task_id="not_in_map", image="/tmp/none.sif"))
    finally:
        os.unlink(path)


def test_budget_does_not_disturb_the_agent_bash_timeout():
    """Verifier budget and per-bash-call agent budget are separate knobs."""
    path = _map_file({"t1": 1800.0})
    try:
        env = _env(timeout=120, test_timeout=600, verifier_timeout_map=path)
        env._task_test_timeout = env._verifier_timeout_map["t1"]
        assert env._timeout == 120
        assert env._effective_test_timeout() == 1800.0
    finally:
        os.unlink(path)
