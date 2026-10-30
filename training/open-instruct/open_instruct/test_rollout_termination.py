"""Regression tests for terminal rollouts and unavailable sandboxes."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from open_instruct.data_types import EnvConfig, EnvConfigEntry
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.vllm_utils import PoolSetup, SamplingConfig, process_request

INSTANCE_NOT_STARTED = "Instance not started. Call start() first."
TOOL = "code"
MAX_STEPS = 12
TOOL_CALL_TIMEOUT_S = 0.05


# --------------------------------------------------------------------- fakes


class _Logprobs:
    token_logprobs = [-0.1]


class _Choice:
    token_ids = [11]
    text = "<tool>bash</tool>"
    logprobs = _Logprobs()
    finish_reason = "tool_call"


class _ApiResponse:
    choices = [_Choice()]


class _Completions:
    async def create(self, **_kwargs):
        return _ApiResponse()


class _Client:
    completions = _Completions()


class _ParseResult:
    had_tool_call = True

    def __init__(self):
        # a fresh EnvCall each generation, as the real parser would produce
        self.tool_calls = [EnvCall(id="", name=TOOL, args={"command": "server &"})]


class _Parser:
    def parse_tool_calls(self, _text):
        return _ParseResult()


class _Tokenizer:
    eos_token_id = 2


class _ModelConfig:
    max_model_len = 4096


class _Engine:
    model_config = _ModelConfig()
    tokenizer = _Tokenizer()


class _Remote:
    """Stand-in for a Ray `.remote()` handle whose result is awaited."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **kw):
        return self._fn(*a, **kw)


class _WedgedEnv:
    """A sandbox that hangs once, then is permanently dead.

    Models the real sequence: a command wedges the container (the tool call
    exceeds `tool_call_timeout`), after which every subsequent command raises
    `RuntimeError("Instance not started")` from backends.py.
    """

    def __init__(self):
        self.calls = 0
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(TOOL_CALL_TIMEOUT_S * 20)  # wedge -> wait_for times out
            return StepResult(result="never reached")
        raise RuntimeError(INSTANCE_NOT_STARTED)

    async def _get_metrics(self):
        return {}


class _Pool:
    def __init__(self):
        self.released = []
        self.release = _Remote(self._release)

    def _release(self, actor):
        self.released.append(actor)


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _Actor:
    def __init__(self, pool):
        self.client = _Client()
        self.tool_parser = _Parser()
        self.llm_engine = _Engine()
        self.model_name = "fake"
        self.server_port = 0
        self.tool_call_timeout = TOOL_CALL_TIMEOUT_S
        self.per_turn_max_tokens = None
        self.seed_per_turn = False
        self.mask_tool_use = False
        self.pools = {TOOL: pool}
        self.active_tasks = {}
        self.completion_queue = _Queue()
        # split_request_id("train_0_1") -> {"base_id": "train_0", "request_index": 1}
        self.request_metadata = {
            "train_0": {
                "prompt_token_ids": [1, 2, 3],
                "active_tools": [TOOL],
                "env_config": EnvConfig(
                    max_steps=MAX_STEPS, env_configs={TOOL: EnvConfigEntry(env_name=TOOL, is_text_env=False)}
                ),
                "original_sampling_params": SamplingConfig(n=1),
            }
        }


# ----------------------------------------------------------------- the test


class TestWedgedSandboxTerminates(unittest.TestCase):
    def _run(self):
        pool, env = _Pool(), _WedgedEnv()
        actor = _Actor(pool)

        setup = PoolSetup(
            acquired={TOOL: (pool, env)},
            actor_map={TOOL: env},
            allowed_tools={TOOL},
            tool_response_roles={TOOL: "tool"},
            tool_call_format_error_messages={},
            active_env_names=[TOOL],
            text_env_names=[],
        )

        async def _no_health(_port):
            return None

        async def _pools(*_a, **_kw):
            return setup

        # ray.cancel(step_ref, force=True) on the timeout path goes through Ray's
        # auto-init hook; with no cluster up it blocks for minutes trying to start
        # one. It is already wrapped in contextlib.suppress(Exception) in the
        # tree, so a no-op is faithful. Harmless on trees that don't call it.
        with (
            patch("open_instruct.vllm_utils._check_health", _no_health),
            patch("open_instruct.vllm_utils._acquire_and_reset_pools", _pools),
            patch("ray.cancel", lambda *a, **kw: None),
            patch("open_instruct.vllm_utils.process_tool_tokens", lambda *a, **kw: ([9], [0.0], [0], 0)),
        ):
            asyncio.run(process_request(actor, "train_0_1", SamplingConfig(max_tokens=64)))

        self.assertTrue(actor.completion_queue.items, "rollout never enqueued a completion")
        state = actor.completion_queue.items[0]["request_output"].outputs[0].rollout_state
        return state, env, pool

    def test_a_wedged_sandbox_ends_before_max_steps(self):
        state, env, _ = self._run()
        self.assertLess(
            state["step_count"],
            MAX_STEPS,
            f"rollout ground to max_steps ({MAX_STEPS}) against a dead sandbox; env was called {env.calls}x",
        )

    def test_b_no_run_of_identical_dead_sandbox_errors(self):
        state, _, _ = self._run()
        occurrences = state["tool_error"].count(INSTANCE_NOT_STARTED)
        self.assertLessEqual(
            occurrences,
            1,
            f"rollout re-issued steps against a dead sandbox {occurrences}x; "
            "the first dead-sandbox error must be terminal",
        )

    def test_c_timeout_flag_implies_loop_exited(self):
        state, _, _ = self._run()
        if state["timeout"]:
            self.assertLess(
                state["step_count"],
                MAX_STEPS,
                "rollout.timeout was set but the loop kept stepping — the flag is written and never read in-loop",
            )

    def test_d_lease_released_exactly_once_on_terminal_path(self):
        """A repair that terminates the rollout must not strand the lease.

        Release happens in `process_request`'s `finally`, so this holds on both
        trees today — it is here to fail loudly if a future termination fix
        returns early past the teardown.
        """
        _, env, pool = self._run()
        self.assertEqual(
            len(pool.released), 1, f"expected exactly one lease release on the terminal path, got {len(pool.released)}"
        )
        self.assertIs(pool.released[0], env, "released a different actor than the one acquired")


class _SlowHealthyEnv:
    """A sandbox that never fails and never wedges — it is merely slow.

    Deliberately a different failure mode from `_WedgedEnv`: every step succeeds
    well inside `tool_call_timeout`, so no per-call timeout fires, no backend
    dies, and none of the brakes exercised by A-C are involved. The only thing
    that can stop this rollout is the independent wall-clock deadline.
    """

    STEP_S = 0.15

    def __init__(self):
        self.calls = 0
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        self.calls += 1
        await asyncio.sleep(self.STEP_S)
        return StepResult(result="ok", reward=0.0)

    async def _get_metrics(self):
        return {}


class _RecordingCompletions:
    def __init__(self):
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return _ApiResponse()


class _HealthyEnv:
    def __init__(self):
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        return StepResult(result="ok", reward=0.0)

    async def _get_metrics(self):
        return {}


class TestPerTurnSampling(unittest.TestCase):
    def _run(self, seed_per_turn: bool):
        pool, env = _Pool(), _HealthyEnv()
        actor = _Actor(pool)
        actor.seed_per_turn = seed_per_turn
        completions = _RecordingCompletions()
        actor.client = type("Client", (), {"completions": completions})()
        actor.request_metadata["train_0"]["env_config"].max_steps = 2
        setup = PoolSetup(
            acquired={TOOL: (pool, env)},
            actor_map={TOOL: env},
            allowed_tools={TOOL},
            tool_response_roles={TOOL: "tool"},
            tool_call_format_error_messages={},
            active_env_names=[TOOL],
            text_env_names=[],
        )

        async def _no_health(_port):
            return None

        async def _pools(*_a, **_kw):
            return setup

        with (
            patch("open_instruct.vllm_utils._check_health", _no_health),
            patch("open_instruct.vllm_utils._acquire_and_reset_pools", _pools),
            patch("open_instruct.vllm_utils.process_tool_tokens", lambda *a, **kw: ([9], [0.0], [0], 0)),
        ):
            asyncio.run(process_request(actor, "train_0_1", SamplingConfig(max_tokens=64, seed=123)))

        return completions.requests

    def test_generation_turns_do_not_reset_the_rollout_seed_by_default(self):
        completions = self._run(seed_per_turn=False)

        self.assertEqual(len(completions.requests), 2)
        self.assertTrue(all("seed" not in request for request in completions.requests))

    def test_opt_in_per_turn_seeding_increments_the_seed(self):
        completions = self._run(seed_per_turn=True)

        self.assertEqual([request["seed"] for request in completions], [123, 124])


class TestWallClockDeadline(unittest.TestCase):
    """Assertion E — the independent backstop brake.

    A-C cover the wedge path. None of them can exercise `ROLLOUT_TIMEOUT_S`:
    the wedge resolves within `max_steps` long before any plausible wall clock
    expires. Since the agreed repair restores the deadline, something has to
    prove it actually fires, or it ships untested.
    """

    DEADLINE_S = 0.5

    def test_e_slow_rollout_stopped_by_wall_clock(self):
        pool, env = _Pool(), _SlowHealthyEnv()
        actor = _Actor(pool)

        setup = PoolSetup(
            acquired={TOOL: (pool, env)},
            actor_map={TOOL: env},
            allowed_tools={TOOL},
            tool_response_roles={TOOL: "tool"},
            tool_call_format_error_messages={},
            active_env_names=[TOOL],
            text_env_names=[],
        )

        async def _no_health(_port):
            return None

        async def _pools(*_a, **_kw):
            return setup

        # create=True: the symbol is absent on a tree whose wall clock was
        # removed, and the point of this test is to be red there rather than to
        # error out collecting.
        with (
            patch("open_instruct.vllm_utils._check_health", _no_health),
            patch("open_instruct.vllm_utils._acquire_and_reset_pools", _pools),
            patch("ray.cancel", lambda *a, **kw: None),
            patch("open_instruct.vllm_utils.ROLLOUT_TIMEOUT_S", self.DEADLINE_S, create=True),
            patch("open_instruct.vllm_utils.process_tool_tokens", lambda *a, **kw: ([9], [0.0], [0], 0)),
        ):
            asyncio.run(process_request(actor, "train_0_1", SamplingConfig(max_tokens=64)))

        state = actor.completion_queue.items[0]["request_output"].outputs[0].rollout_state
        budget = int(self.DEADLINE_S / _SlowHealthyEnv.STEP_S) + 2
        self.assertLess(
            state["step_count"],
            MAX_STEPS,
            f"rollout ran to max_steps ({MAX_STEPS}) with no wall-clock backstop; "
            f"deadline was {self.DEADLINE_S}s at {_SlowHealthyEnv.STEP_S}s/step",
        )
        self.assertLessEqual(
            state["step_count"],
            budget,
            f"rollout overran its {self.DEADLINE_S}s deadline by more than one step "
            f"(took {state['step_count']} steps, budget {budget})",
        )
        self.assertTrue(state["timeout"], "wall-clock expiry must mark the rollout as timed out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
