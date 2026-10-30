"""Exercise complete production pool lifecycle methods without GPU imports."""

import ast
import asyncio
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


class ActorDiedError(Exception):
    pass


def load_pool():
    path = Path(__file__).parents[1] / "open_instruct/environments/pool.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    for method in cls.body:
        method.decorator_list = []
    namespace = {
        "asyncio": asyncio,
        "contextlib": contextlib,
        "logger": Mock(),
        "ray": SimpleNamespace(kill=Mock()),
        "DEFAULT_ACQUIRE_TIMEOUT_S": 0.1,
        "_actor_key": lambda actor: actor.name,
        "_actor_reusable_after_error": lambda error: not isinstance(error, ActorDiedError),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["EnvironmentPool"], namespace["ray"]


class PoolReplenishmentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        cls, self.ray = load_pool()
        self.pool = cls.__new__(cls)
        self.old = SimpleNamespace(name="old")
        self.new = SimpleNamespace(name="new")
        self.pool._actors = [self.old]
        self.pool._available = asyncio.Queue()
        self.pool._actor_host_leases = {}
        self.pool._close_on_release = False
        self.pool._acquire_timeout = 0.1
        self.pool._create_actor_batch = Mock(return_value=[self.new])

    async def test_discard_replenishes_for_next_rollout(self):
        await self.pool.discard(self.old, "unconfirmed command completion")
        self.assertIs(await self.pool.acquire(), self.new)
        self.assertEqual(self.pool.size(), 1)
        self.ray.kill.assert_called_once_with(self.old, no_restart=True)

    async def test_duplicate_discards_create_only_one_replacement(self):
        await asyncio.gather(self.pool.discard(self.old), self.pool.discard(self.old))
        self.pool._create_actor_batch.assert_called_once_with(1, 2)
        self.assertEqual(self.pool._available.qsize(), 1)
        self.assertEqual(self.pool.size(), 1)

    async def test_reset_failure_replaces_exactly_once(self):
        self.pool._available.put_nowait(self.old)
        self.pool._reset_actor = AsyncMock(side_effect=ActorDiedError("dead"))
        with self.assertRaises(ActorDiedError):
            await self.pool.acquire_reset({})
        self.pool._create_actor_batch.assert_called_once_with(1, 2)
        self.assertIs(await self.pool.acquire(), self.new)

    async def test_failed_close_replenishes(self):
        self.pool._close_on_release = True
        self.old.close = SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("close failed")))
        await self.pool.release(self.old)
        self.assertIs(await self.pool.acquire(), self.new)

    async def test_replacement_failure_is_not_silenced(self):
        self.pool._create_actor_batch.side_effect = RuntimeError("setup failed")
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            await self.pool.discard(self.old)
        self.assertEqual(self.pool.size(), 0)
        self.assertTrue(self.pool._available.empty())
