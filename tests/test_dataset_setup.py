"""Dataset reset setup contracts, without model or container dependencies."""

import ast
import os
import shlex
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from test_vanillux_shell_parity import SETSID, ShellSession

SOURCE = (
    Path(__file__).resolve().parents[1] / "training/open-instruct/open_instruct/environments/swerl_vanillux_sandbox.py"
)


def harness():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SWERLVanilluxSandboxEnv")
    names = {"_apply_task_setup", "_do_reset", "_run_tests"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(
        shlex=shlex,
        time=time,
        Any=object,
        StepResult=SimpleNamespace,
        TaskSetupError=RuntimeError,
        logger=Mock(),
        _BASH_WRAPPER_PATH_QUOTED="/wrapper",
        _BASH_CWD_PATH="/cwd",
        _BASH_ENV_PATH="/env",
        _SETUP_CWD_PATH="/setup-cwd",
        _SETUP_ENV_PATH="/setup-env",
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), ns)
    return type("Harness", (), {name: ns[name] for name in names})()


class DatasetSetupTest(unittest.TestCase):
    def setUp(self):
        self.env = harness()
        self.env._backend = Mock()
        self.env._backend.run_command.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
        self.env._timeout = 120
        self.env._task_id = "task"
        self.env._has_task_setup = False
        self.env._pwd_map = {}
        self.env._verifier_timeout_map = {}
        self.env._task_test_timeout = None

    def test_absent_and_empty_do_nothing(self):
        for command in (None, "", "  \n"):
            self.env._apply_task_setup(command)
        self.env._backend.run_command.assert_not_called()
        self.assertFalse(self.env._has_task_setup)

    def test_command_is_one_quoted_argument_with_deadline(self):
        command = "cd '/path with spaces' && export TEST='a; b'"
        self.env._apply_task_setup(command)
        invocation = self.env._backend.run_command.call_args_list[0]
        self.assertEqual(shlex.split(invocation.args[0]), ["bash", "/wrapper", command])
        self.assertEqual(invocation.kwargs["timeout"], 120)
        self.assertTrue(self.env._has_task_setup)

    def test_failure_and_timeout_fail_reset_without_snapshot(self):
        for code in (1, 124):
            self.env._backend.reset_mock()
            self.env._backend.run_command.return_value = SimpleNamespace(exit_code=code, stdout="", stderr="failed")
            with self.assertRaisesRegex(RuntimeError, f"exit={code}"):
                self.env._apply_task_setup("false")
            self.assertEqual(self.env._backend.run_command.call_count, 1)
            self.assertFalse(self.env._has_task_setup)

    def test_snapshot_failure_is_fatal(self):
        self.env._backend.run_command.side_effect = [
            SimpleNamespace(exit_code=0),
            SimpleNamespace(exit_code=1, stdout="", stderr="disk full"),
        ]
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            self.env._apply_task_setup("true")
        self.assertFalse(self.env._has_task_setup)

    def test_reset_clears_previous_setup_and_tests_before_validation(self):
        self.env._has_task_setup = True
        self.env._tests_dir = "previous tests"
        with self.assertRaisesRegex(ValueError, "must be a string"):
            self.env._do_reset(setup_command=["invalid"])
        self.assertFalse(self.env._has_task_setup)
        self.assertIsNone(self.env._tests_dir)

    def test_each_reset_uses_current_row_after_loading(self):
        ns = self.env._do_reset.__func__.__globals__
        ns.update(os=os, VANILLUX_CALL_LIMIT=100, TIMING_LOGS=False, render_instance=lambda text: text)
        self.env._backend_type = "sandfleet"
        ns["prefer_local_sif"] = lambda image: image
        self.env._backend_kwargs = {}
        self.env._tool_definitions = []
        events = []
        self.env._prepare_vanillux_runtime = lambda: events.append("runtime")
        self.env._load_task_data = lambda path: events.append("files")
        apply = self.env._apply_task_setup

        def setup(command):
            events.append(command)
            apply(command)

        self.env._apply_task_setup = setup
        with tempfile.TemporaryDirectory() as directory:
            self.env._task_data_dir = directory
            (Path(directory) / "first").mkdir()
            self.env._do_reset("first", image="fixture", setup_command="export DATASET_ENV=ready")
            self.assertEqual(events, ["files", "runtime", "export DATASET_ENV=ready"])
            self.assertTrue(self.env._has_task_setup)
            events.clear()
            self.env._do_reset("second", image="fixture")
            self.assertEqual(events, ["runtime", None])
            self.assertFalse(self.env._has_task_setup)
            self.assertIsNone(self.env._tests_dir)


class SetupShellTest(unittest.TestCase):
    def test_setup_state_survives_for_agent_and_verifier(self):
        if not SETSID:
            self.skipTest("Linux setsid required")
        with tempfile.TemporaryDirectory() as directory:
            session = ShellSession(Path(directory) / "sandbox", training=True)
            env = harness()
            # Bind actual production setup method to the fixture state paths.
            globals_ = env._apply_task_setup.__func__.__globals__
            globals_.update(
                _BASH_WRAPPER_PATH_QUOTED=shlex.quote(str(session.wrapper_path)),
                _BASH_CWD_PATH=str(session.cwd_path),
                _BASH_ENV_PATH=str(session.env_path),
                _SETUP_CWD_PATH=str(session.root / "setup-cwd"),
                _SETUP_ENV_PATH=str(session.root / "setup-env"),
            )

            def run(command, timeout=4):
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=session.initial_cwd,
                    env=session.env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

            env._backend = SimpleNamespace(run_command=run)
            env._timeout = 4
            env._task_id = "fixture"
            env._has_task_setup = False
            (session.destination / "seed").write_text("available")
            env._apply_task_setup(
                f"cd {shlex.quote(str(session.destination))} && test -f seed && export DATASET_ENV=ready"
            )
            agent = session.run('printf "%s:%s" "$PWD" "$DATASET_ENV"')
            self.assertEqual(agent.stdout, f"{session.destination}:ready")
            session.run("cd / && export DATASET_ENV=changed")
            # Exercise the real verifier method, replacing only fixture I/O.
            globals_.update(uuid=uuid, truncate_observation=lambda text: text)
            test_script = session.root / "test.sh"
            test_script.write_text('printf "%s:%s" "$PWD" "$DATASET_ENV"')

            def verifier_run(command, timeout=4):
                if command == "test -f /tests/test.sh && echo EXISTS":
                    return SimpleNamespace(stdout="EXISTS")
                return run(command.replace("/tests/test.sh", shlex.quote(str(test_script))), timeout)

            env._backend.run_command = verifier_run
            env._tests_dir = "fixture"
            env._test_timeout = 4
            env._upload_directory = Mock()
            env._parse_reward = lambda: 1.0
            result = env._run_tests()
            self.assertIn(f"{session.destination}:ready", result.result)
            self.assertEqual(result.reward, 1.0)
