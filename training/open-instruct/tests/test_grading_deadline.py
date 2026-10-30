"""Regression tests for independent agent, rollout, and verifier deadlines."""

import __future__

import ast
import asyncio
import dataclasses
import json
import os
import tempfile
from pathlib import Path

import pytest

from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv

# Execute the production definitions without importing the GPU inference stack.
_source = Path(__file__).parents[1] / "open_instruct/vllm_utils.py"
_definitions = [
    node
    for node in ast.parse(_source.read_text()).body
    if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in {"PoolSetup", "_is_submission_call"}
]
_namespace = {"dataclasses": dataclasses, "__name__": __name__}
exec(
    compile(
        ast.Module(body=_definitions, type_ignores=[]), str(_source), "exec", __future__.annotations.compiler_flag
    ),
    _namespace,
)
PoolSetup = _namespace["PoolSetup"]
_is_submission_call = _namespace["_is_submission_call"]

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _pool_setup(deadline=3720.0, markers=(SUBMIT,)):
    return PoolSetup(
        acquired={},
        actor_map={},
        allowed_tools={"bash"},
        tool_response_roles={},
        tool_call_format_error_messages={},
        active_env_names=["swerl"],
        text_env_names=[],
        grading_deadlines={"bash": deadline},
        submit_markers={"bash": markers},
    )


def _call(command):
    return EnvCall(id="1", name="bash", args={"command": command})


# --- which calls get the grading deadline -----------------------------------------


def test_submission_call_is_recognised():
    assert _is_submission_call(_call(f"echo {SUBMIT}"), _pool_setup()) is True


def test_ordinary_call_is_not():
    assert _is_submission_call(_call("pytest -q"), _pool_setup()) is False


def test_env_without_grading_contract_never_matches():
    """Non-grading envs must be untouched by all of this."""
    plain = _pool_setup()
    plain.grading_deadlines.clear()
    plain.submit_markers.clear()
    assert _is_submission_call(_call(f"echo {SUBMIT}"), plain) is False


def test_marker_in_any_argument_counts():
    tc = EnvCall(id="1", name="bash", args={"command": "sh x.sh", "stdin": SUBMIT})
    assert _is_submission_call(tc, _pool_setup()) is True


# --- the env's side of the contract -----------------------------------------------


def _env(**kw):
    return SWERLVanilluxSandboxEnv(backend="sandfleet", image="x.sif", **kw)


def _map_file(obj):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(obj, fh)
        return fh.name


def test_grading_deadline_is_budget_plus_bounded_overhead():
    env = _env(test_timeout=600)
    assert env.get_grading_deadline() == 600 + env._GRADING_OVERHEAD_S
    assert env.get_submit_markers() == (SUBMIT,)


def test_grading_deadline_tracks_the_selected_task():
    path = _map_file({"t1": 3600.0})
    try:
        env = _env(verifier_timeout_map=path)
        env._task_test_timeout = env._verifier_timeout_map["t1"]
        assert env.get_grading_deadline() == 3600 + env._GRADING_OVERHEAD_S
    finally:
        os.unlink(path)


def test_ceiling_clamps_and_is_reported():
    """CalibForge's 12000s tasks are outside this cutover; a ceiling must be visible."""
    path = _map_file({"slow": 12000.0, "ok": 900.0})
    try:
        env = _env(verifier_timeout_map=path, max_verifier_timeout_s=3600)
        env._task_test_timeout = env._verifier_timeout_map["slow"]
        assert env._effective_test_timeout() == 3600
        assert env._verifier_budget_was_clamped() is True
        env._task_test_timeout = env._verifier_timeout_map["ok"]
        assert env._effective_test_timeout() == 900
        assert env._verifier_budget_was_clamped() is False
    finally:
        os.unlink(path)


def test_negative_ceiling_rejected():
    with pytest.raises(RuntimeError, match="must be >= 0"):
        _env(max_verifier_timeout_s=-1)


def test_grading_results_are_tagged(monkeypatch):
    """The tag is what makes an outer cancellation attributable later."""
    env = _env(test_timeout=600)
    monkeypatch.setattr(env, "_grade", lambda: StepResult(result="ok", reward=1.0, done=True))
    out = env._run_tests()
    assert out.metadata["grading"] is True
    assert out.metadata["verifier_timeout_s"] == 600
    assert out.metadata["verifier_timeout_declared"] is False
    assert out.reward == 1.0


# --- the real wait_for path, scaled -------------------------------------------------


class _Stub:
    """Stands in for the remote env actor: sleeps, then returns a grading result."""

    def __init__(self, duration):
        self.duration = duration
        self.completed = False

    async def run(self):
        await asyncio.sleep(self.duration)
        self.completed = True
        return {"reward": 1.0, "metadata": {"grading": True}}


async def _dispatch(stub, step_timeout):
    """Mirror of the production dispatch: shield so the remote keeps running, and drain."""
    fut = asyncio.ensure_future(stub.run())
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout=step_timeout), False
    except asyncio.TimeoutError:
        drained = await asyncio.wait_for(asyncio.shield(fut), timeout=5)
        return drained, True


def test_grader_outlasting_the_ordinary_tool_timeout_returns_its_reward():
    """0.30s grader against a 0.05s ordinary cap and a 1.0s grading deadline.

    This is the requirement in miniature: under the ordinary cap the reward only survives
    via the drain, and under the grading deadline it is returned directly.
    """
    ordinary_cap, grading_deadline = 0.05, 1.0

    slow = _Stub(0.30)
    result, timed_out = asyncio.run(_dispatch(slow, grading_deadline))
    assert timed_out is False, "grading deadline must not fire for a grader inside budget"
    assert result["reward"] == 1.0

    slow2 = _Stub(0.30)
    result2, timed_out2 = asyncio.run(_dispatch(slow2, ordinary_cap))
    assert timed_out2 is True, "the ordinary cap is what used to truncate grading"
    assert result2["reward"] == 1.0, "drain still recovers it, but only within 60s"


def test_selected_verifier_deadline_still_terminates_grading():
    """The deadline is a real bound, not a removal of bounds."""
    hung = _Stub(5.0)

    async def go():
        fut = asyncio.ensure_future(hung.run())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.05)
        fut.cancel()

    asyncio.run(go())
    assert hung.completed is False


def test_ordinary_commands_remain_capped():
    ordinary = _Stub(1.0)

    async def go():
        fut = asyncio.ensure_future(ordinary.run())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.05)
        fut.cancel()

    asyncio.run(go())
    assert ordinary.completed is False


def test_grading_deadline_ignores_remaining_rollout_time():
    """An agent that submits with 1s of its budget left must still be graded.

    Reproduces the selection in process_request: grading takes its own deadline whole,
    ordinary calls take min(tool_cap, remaining).
    """
    tool_cap, remaining, grading_deadline = 120.0, 1.0, 3720.0

    ordinary = min(tool_cap, max(0.001, remaining))
    assert ordinary == 1.0

    ps = _pool_setup(deadline=grading_deadline)
    chosen = ps.grading_deadlines["bash"] if _is_submission_call(_call(f"echo {SUBMIT}"), ps) else ordinary
    assert chosen == 3720.0
