"""Exercise the production timeout/shield/drain statements without GPU imports."""

import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class StepDrainTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "open_instruct/vllm_utils.py"
        tree = ast.parse(path.read_text())
        dispatch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "step_ref" for child in node.body
            )
        )
        future = next(
            node
            for node in dispatch.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "step_future"
        )
        wait = next(
            node
            for node in dispatch.body
            if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "step_result"
        )
        timeout = next(node for node in dispatch.handlers if ast.unparse(node.type) == "asyncio.TimeoutError")
        drain = next(node for node in timeout.body if isinstance(node, ast.Try))
        drain_start = next(
            i
            for i, node in enumerate(timeout.body)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "drained_result"
        )
        drain_body = timeout.body[drain_start : timeout.body.index(drain) + 1]
        wrapper = ast.parse("async def dispatch(step_ref):\n    pass\n").body[0]
        wrapper.body = [
            future,
            dispatch.body[dispatch.body.index(future) + 1],
            ast.Try(
                body=[wait],
                handlers=[ast.ExceptHandler(type=timeout.type, name=None, body=drain_body)],
                orelse=[],
                finalbody=[],
            ),
        ]
        self.rollout = SimpleNamespace(info={})
        self.unsafe = set()

        async def bounded_wait(awaitable, timeout):
            return await asyncio.wait_for(awaitable, timeout=0.03 if timeout == 60 else timeout)

        namespace = {
            "asyncio": SimpleNamespace(
                ensure_future=asyncio.ensure_future,
                shield=asyncio.shield,
                wait_for=bounded_wait,
                TimeoutError=asyncio.TimeoutError,
            ),
            "StepResult": object,
            "step_timeout": 0.001,
            "rollout": self.rollout,
            "logger": Mock(),
            "unsafe_env_names": self.unsafe,
            "pool_setup": SimpleNamespace(acquired={"sandbox": (None, "actor")}),
            "target": "actor",
        }
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        self.dispatch = namespace["dispatch"]

    async def test_timeout_waits_for_completion_without_cancelling_remote_work(self):
        completed = []

        async def remote_step():
            await asyncio.sleep(0.02)
            completed.append(True)
            return SimpleNamespace(metadata={}, reward=0.0)

        task = asyncio.create_task(remote_step())
        await self.dispatch(task)
        self.assertEqual(completed, [True])
        self.assertFalse(task.cancelled())
        self.assertFalse(self.unsafe)

    async def test_drain_failure_marks_unscored_and_prevents_pool_reuse(self):
        async def remote_step():
            await asyncio.sleep(0.02)
            raise ConnectionError("worker lost")

        await self.dispatch(asyncio.create_task(remote_step()))
        self.assertTrue(self.rollout.info["infrastructure_failure"])
        self.assertEqual(self.unsafe, {"sandbox"})

    async def test_drain_preserves_worker_loss_metadata(self):
        async def remote_step():
            await asyncio.sleep(0.02)
            return SimpleNamespace(metadata={"sandbox_lost": True}, reward=0.0)

        await self.dispatch(asyncio.create_task(remote_step()))
        self.assertTrue(self.rollout.info["infrastructure_failure"])

    async def test_cancellation_during_drain_quarantines_actor(self):
        remote = asyncio.create_task(asyncio.sleep(1))
        waiter = asyncio.create_task(self.dispatch(remote))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(self.unsafe, {"sandbox"})
        remote.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await remote

    async def test_unresponsive_step_is_quarantined_after_bounded_drain(self):
        remote = asyncio.create_task(asyncio.sleep(1))
        await self.dispatch(remote)
        self.assertTrue(self.rollout.info["infrastructure_failure"])
        self.assertEqual(self.unsafe, {"sandbox"})
        remote.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await remote
