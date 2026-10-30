"""Test production client routing without loading the training stack."""

import ast
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class ScopedCancelTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "open_instruct/environments/sandfleet_backend.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SandfleetBackend")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_command")
        namespace = {"uuid": uuid, "ExecutionResult": SimpleNamespace, "_API_VERSION": "v1"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        self.backend = type("Backend", (), {"run_command": namespace["run_command"]})()
        self.backend._timeout = 60
        self.backend._agent_url = "http://agent"
        self.backend._lease_id = "lease"
        self.backend._lease_token = "token"
        self.backend._agent_request = Mock(return_value={"stdout": "", "stderr": "", "exit_code": 0})
        self.backend._request = Mock(return_value={"state": "running", "interrupted": True})

    def test_success_never_cancels_background_services(self):
        self.backend.run_command("server &")
        first = self.backend._agent_request.call_args.kwargs["payload"]["command_id"]
        self.backend.run_command("echo next")
        second = self.backend._agent_request.call_args.kwargs["payload"]["command_id"]
        self.assertNotEqual(first, second)
        self.backend._request.assert_not_called()

    def test_interrupt_cancels_only_failed_command(self):
        self.backend._agent_request.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.backend.run_command("sleep 60")
        command_id = self.backend._agent_request.call_args.kwargs["payload"]["command_id"]
        self.assertEqual(self.backend._request.call_args.kwargs["payload"], {"command_id": command_id})
        self.assertTrue(self.backend._request.call_args.args[1].endswith("/exec-cancel"))

    def test_failed_cancel_is_not_silently_accepted(self):
        self.backend._agent_request.side_effect = ConnectionError("lost response")
        self.backend._request.return_value = {"state": "running", "interrupted": False}
        with self.assertRaisesRegex(ConnectionError, "lost response") as raised:
            self.backend.run_command("sleep 60")
        self.assertIn("could not interrupt", raised.exception.__notes__[0])

    def test_lost_worker_error_survives_unreachable_cancel_route(self):
        class LostWorkerError(RuntimeError):
            pass

        original = LostWorkerError("worker disappeared")
        self.backend._agent_request.side_effect = original
        self.backend._request.side_effect = ConnectionError("agent unreachable")
        with self.assertRaises(LostWorkerError) as raised:
            self.backend.run_command("sleep 60")
        self.assertIs(raised.exception, original)
        self.assertIn("agent unreachable", original.__notes__[0])
