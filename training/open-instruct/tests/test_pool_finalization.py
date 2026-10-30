"""Exercise the production release/discard loop's error and timeout handling."""

import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


class PoolFinalizationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "open_instruct/vllm_utils.py"
        tree = ast.parse(path.read_text())
        release_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and "pooled_env_name" in ast.unparse(node.target)
        )
        wrapper = ast.parse("async def finalize():\n    pass\n").body[0]
        wrapper.body = [release_loop]
        self.pool = SimpleNamespace(
            discard=SimpleNamespace(remote=AsyncMock()), release=SimpleNamespace(remote=Mock())
        )
        self.dead = {"sandbox"}
        self.timeouts = []

        async def bounded_wait(awaitable, timeout):
            self.timeouts.append(timeout)
            return await asyncio.wait_for(awaitable, timeout=0.01)

        namespace = {
            "asyncio": SimpleNamespace(wait_for=bounded_wait),
            "pool_setup": SimpleNamespace(acquired={"sandbox": (self.pool, "actor")}),
            "dead_env_names": self.dead,
        }
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        self.finalize = namespace["finalize"]

    async def test_replacement_setup_failure_reaches_caller(self):
        self.pool.discard.remote.side_effect = RuntimeError("replacement setup failed")
        with self.assertRaisesRegex(RuntimeError, "replacement setup failed"):
            await self.finalize()
        self.pool.release.remote.assert_not_called()

    async def test_replacement_hang_is_bounded(self):
        async def hanging_discard(*args):
            await asyncio.Event().wait()

        self.pool.discard.remote.side_effect = hanging_discard
        with self.assertRaises(asyncio.TimeoutError):
            await self.finalize()
        self.assertEqual(self.timeouts, [60])
        self.pool.release.remote.assert_not_called()

    async def test_healthy_release_remains_fire_and_forget(self):
        self.dead.clear()
        await self.finalize()
        self.pool.release.remote.assert_called_once_with("actor")
        self.pool.discard.remote.assert_not_called()
        self.assertEqual(self.timeouts, [])
