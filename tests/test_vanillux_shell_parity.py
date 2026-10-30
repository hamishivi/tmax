"""Run the actual evaluation/training shell wrappers without the ML dependencies.

Run with ``python -m unittest discover -s tests -p 'test_vanillux_shell_parity.py'``.
The training wrapper requires the real Linux ``setsid`` utility, so these tests
skip on platforms without it. No container runtime or model service is needed.
This verifies shell behavior, not Apptainer/Sandfleet isolation or timeouts.

Only the definitions under test are compiled from their source AST. In
particular, the shell program is not copied into the tests: a regression in the
deployed wrapper is exercised without importing torch, vLLM, Harbor or LiteLLM.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCE = REPO_ROOT / "Vanillux2Agent/agent.py"
TRAINING_SOURCE = REPO_ROOT / "training/open-instruct/open_instruct/environments/swerl_vanillux_sandbox.py"
BASH = shutil.which("bash")
SETSID = shutil.which("setsid")


def _load_evaluation_wrapper(state_dir: Path):
    tree = ast.parse(EVALUATION_SOURCE.read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Vanillux2Agent")
    method = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "_wrap_command")
    namespace = {"_STATE_DIR": str(state_dir)}
    exec(  # noqa: S102 - execute the checked-in method, without heavyweight imports
        compile(ast.Module(body=[method], type_ignores=[]), str(EVALUATION_SOURCE), "exec"), namespace
    )
    return lambda command: namespace["_wrap_command"](SimpleNamespace(persistent_bash=True), command)


def _load_training_wrapper(cwd_path: Path, env_path: Path) -> str:
    tree = ast.parse(TRAINING_SOURCE.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_BASH_WRAPPER" for target in node.targets)
    )
    namespace = {"shlex": shlex, "_BASH_CWD_PATH": str(cwd_path), "_BASH_ENV_PATH": str(env_path)}
    wrapper = eval(compile(ast.Expression(definition.value), str(TRAINING_SOURCE), "eval"), namespace)
    # Keep output files under the owned fixture even when a broken wrapper
    # times out before reaching cleanup. Only filesystem destinations change.
    return wrapper.replace("/tmp/.swerl_vanillux_stdout.", str(cwd_path.parent / ".stdout.")).replace(
        "/tmp/.swerl_vanillux_stderr.", str(cwd_path.parent / ".stderr.")
    )


class ShellSession:
    """Fresh backend shells sharing only the wrapper's persisted state files."""

    def __init__(self, root: Path, training: bool):
        self.root = root
        self.root.mkdir()
        self.initial_cwd = root / "initial"
        self.initial_cwd.mkdir()
        self.destination = root / "destination with 'quotes'"
        self.destination.mkdir()
        self.state_dir = root / "state"
        self.state_dir.mkdir()
        self.cwd_path = self.state_dir / "cwd"
        self.env_path = self.state_dir / "env"
        self.env = {
            "HOME": str(root),
            "PATH": os.defpath,
            # Avoid imposing a production memory cap on the test runner. The
            # cap is a training resource policy, not persisted-shell behavior.
            "SWERL_SANDBOX_ULIMIT_AS_KB": "unlimited",
        }
        if SETSID:
            self.env["PATH"] = str(Path(SETSID).parent) + os.pathsep + self.env["PATH"]
        subprocess.run(
            [BASH, "-c", f"pwd > {shlex.quote(str(self.cwd_path))}; export -p > {shlex.quote(str(self.env_path))}"],
            cwd=self.initial_cwd,
            env=self.env,
            check=True,
            capture_output=True,
        )
        self.training = training
        if training:
            self.wrapper_path = root / "wrapper.sh"
            self.wrapper_path.write_text(_load_training_wrapper(self.cwd_path, self.env_path))
        else:
            self.wrap_command = _load_evaluation_wrapper(self.state_dir)
        self.pid_files: list[Path] = []
        self.process_group_files: list[Path] = []

    def run(
        self,
        command: str,
        timeout: float = 4,
        command_timeout: float | None = None,
        completion_path: Path | None = None,
    ) -> subprocess.CompletedProcess:
        argv = [BASH, str(self.wrapper_path), command] if self.training else [BASH, "-c", self.wrap_command(command)]
        if completion_path is not None:
            assert self.training
            argv.append(str(completion_path))
        if command_timeout is not None:
            argv = [shutil.which("timeout"), "--kill-after=3s", f"{command_timeout}s", *argv]
        process = subprocess.Popen(
            argv,
            cwd=self.initial_cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The group belongs exclusively to this test invocation. Detached
            # services are separately tracked by exact PIDs written by our
            # own commands; never match arbitrary system process names.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            self.cleanup()
            process.communicate(timeout=2)
            raise
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    def cleanup(self):
        for pid_file in self.process_group_files:
            if pid_file.exists():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.killpg(int(pid_file.read_text()), signal.SIGKILL)
        for pid_file in self.pid_files:
            if pid_file.exists():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)


@unittest.skipUnless(BASH and SETSID, "Requires Bash and real Linux setsid; run on Linux for shell parity")
class TestVanilluxShellParity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tmax-shell-parity-")
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.sessions = [ShellSession(root / "evaluation", False), ShellSession(root / "training", True)]
        for session in self.sessions:
            self.addCleanup(session.cleanup)

    def assert_success(self, session: ShellSession, command: str, stdout: str):
        result = session.run(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, stdout)
        self.assertEqual(result.stderr, "")

    def test_cwd_export_and_relative_paths_survive_fresh_calls(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(
                    session,
                    f"cd {shlex.quote(str(session.destination))}; export TMAX_PARITY_VALUE=kept; printf content > relative-file",
                    "",
                )
                self.assert_success(
                    session,
                    'pwd; printf "%s\\n" "$TMAX_PARITY_VALUE"; cat relative-file',
                    f"{session.destination}\nkept\ncontent",
                )

    def test_multiline_quotes_and_exported_values_round_trip(self):
        value = "one 'quote' \"double\" $dollar `literal` \\slash\nsecond line"
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, f"export TMAX_PARITY_VALUE={shlex.quote(value)}\ntrue", "")
                self.assert_success(session, 'printf "%s" "$TMAX_PARITY_VALUE"', value)

    def test_nonzero_status_still_saves_state(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run(
                    f"cd {shlex.quote(str(session.destination))}; export TMAX_PARITY_VALUE=changed; false"
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assert_success(
                    session, 'pwd; printf "%s" "$TMAX_PARITY_VALUE"', f"{session.destination}\nchanged"
                )

    def test_new_export_can_be_unset(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, "export TMAX_PARITY_VALUE=temporary", "")
                self.assert_success(session, "unset TMAX_PARITY_VALUE", "")
                self.assert_success(session, 'printf "%s" "${TMAX_PARITY_VALUE-unset}"', "unset")

    def test_keeps_fresh_fakeroot_connection_across_calls(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                session.env["FAKEROOTKEY"] = "first-connection"
                self.assert_success(
                    session, 'export TMAX_PARITY_VALUE=kept; printf "%s" "$FAKEROOTKEY"', "first-connection"
                )
                self.assertNotIn("FAKEROOTKEY", session.env_path.read_text())
                session.env["FAKEROOTKEY"] = "next-connection"
                self.assert_success(
                    session, 'printf "%s:%s" "$FAKEROOTKEY" "$TMAX_PARITY_VALUE"', "next-connection:kept"
                )
                del session.env["FAKEROOTKEY"]
                self.assert_success(session, 'printf "%s" "${FAKEROOTKEY-unset}"', "unset")

    def test_unexported_variables_do_not_persist(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, "TMAX_PARITY_LOCAL=temporary", "")
                self.assert_success(session, 'printf "%s" "${TMAX_PARITY_LOCAL-unset}"', "unset")

    def test_exported_readonly_attribute_survives(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, "export TMAX_PARITY_READONLY=locked; readonly TMAX_PARITY_READONLY", "")
                result = session.run("TMAX_PARITY_READONLY=changed")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("readonly", result.stderr)
                self.assert_success(session, 'printf "%s" "$TMAX_PARITY_READONLY"', "locked")

    def test_changing_positional_arguments_does_not_break_state_saving(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, "set -- replacement args; export TMAX_PARITY_VALUE=kept", "")
                self.assert_success(session, 'printf "%s" "$TMAX_PARITY_VALUE"', "kept")

    def test_stdout_stderr_and_exit_status(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run("printf 'out\\n'; printf 'err\\n' >&2; bash -c 'exit 7'")
                self.assertEqual((result.stdout, result.stderr, result.returncode), ("out\n", "err\n", 7))

    def test_explicit_exit_does_not_commit_partial_state(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run(
                    f"cd {shlex.quote(str(session.destination))}; export TMAX_PARITY_VALUE=discarded; exit 7"
                )
                self.assertEqual(result.returncode, 7)
                self.assert_success(
                    session, 'pwd; printf "%s" "${TMAX_PARITY_VALUE-unset}"', f"{session.initial_cwd}\nunset"
                )

    def test_errexit_does_not_commit_partial_state(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run("set -e; export TMAX_PARITY_VALUE=discarded; false")
                self.assertEqual(result.returncode, 1)
                self.assert_success(session, 'printf "%s" "${TMAX_PARITY_VALUE-unset}"', "unset")

    def test_syntax_error_does_not_commit_partial_state(self):
        statuses = []
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run("export TMAX_PARITY_VALUE=discarded\nif")
                self.assertNotEqual(result.returncode, 0)
                statuses.append(result.returncode)
                self.assertIn("syntax error", result.stderr)
                self.assert_success(session, 'printf "%s" "${TMAX_PARITY_VALUE-unset}"', "unset")
        self.assertEqual(statuses[0], statuses[1])

    def test_command_sigint_is_not_ignored(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                result = session.run("kill -INT $$; printf survived")
                # Popen represents a directly signalled shell as -SIGINT;
                # the training transport's wait exposes Bash's 128+SIGINT.
                status = result.returncode if result.returncode >= 0 else 128 - result.returncode
                self.assertEqual(status, 130, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_actual_runtime_setup_preserves_backend_initial_cwd_and_exports(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                session.cwd_path.unlink()
                session.env_path.unlink()
                session.env["TMAX_INITIAL_VALUE"] = "provided-by-backend"
                session.env["FAKEROOTKEY"] = "setup-connection"

                class LocalBackend:
                    def __init__(self, fixture):
                        self.fixture = fixture

                    def run_command(self, command):
                        # Remap only the runtime's three filesystem roots to
                        # this test sandbox. Execute the actual init command.
                        for directory in ("/workspace", "/root", "/app"):
                            command = command.replace(
                                directory, shlex.quote(str(self.fixture.root / ("sandbox-" + directory[1:])))
                            )
                        result = subprocess.run(
                            [BASH, "-c", command],
                            cwd=self.fixture.initial_cwd,
                            env=self.fixture.env,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

                    def write_file(self, path, content):
                        Path(path).write_text(content)

                    async def exec(self, command, timeout_sec):
                        result = self.run_command(command)
                        return SimpleNamespace(
                            return_code=result.exit_code, stdout=result.stdout, stderr=result.stderr
                        )

                source = TRAINING_SOURCE if session.training else EVALUATION_SOURCE
                class_name = "SWERLVanilluxSandboxEnv" if session.training else "Vanillux2Agent"
                method_name = "_prepare_vanillux_runtime" if session.training else "setup"
                tree = ast.parse(source.read_text())
                agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
                method = next(
                    node
                    for node in agent.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
                )
                wrapper_path = session.root / "wrapper.sh"
                namespace = {
                    "shlex": shlex,
                    "BaseEnvironment": object,
                    "_STATE_DIR": str(session.state_dir),
                    "_BASH_CWD_PATH": str(session.cwd_path),
                    "_BASH_ENV_PATH": str(session.env_path),
                    "_BASH_WRAPPER_PATH": str(wrapper_path),
                    "_BASH_WRAPPER_PATH_QUOTED": shlex.quote(str(wrapper_path)),
                    "_BASH_WRAPPER": _load_training_wrapper(session.cwd_path, session.env_path),
                }
                exec(  # noqa: S102 - execute the actual checked-in setup method
                    compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace
                )
                backend = LocalBackend(session)
                instance = SimpleNamespace(_backend=backend, persistent_bash=True)
                if session.training:
                    namespace[method_name](instance)
                else:
                    asyncio.run(namespace[method_name](instance, backend))
                self.assertNotIn("FAKEROOTKEY", session.env_path.read_text())
                session.env["FAKEROOTKEY"] = "after-setup"
                self.assert_success(session, 'printf "%s" "$FAKEROOTKEY"', "after-setup")
                self.assert_success(
                    session, 'pwd; printf "%s" "$TMAX_INITIAL_VALUE"', f"{session.initial_cwd}\nprovided-by-backend"
                )

    def test_missing_state_keeps_backend_working_directory(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                session.cwd_path.unlink()
                session.env_path.unlink()
                result = session.run("pwd")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"{session.initial_cwd}\n")

    def test_deleted_saved_directory_falls_back_to_backend_working_directory(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                self.assert_success(session, f"cd {shlex.quote(str(session.destination))}", "")
                session.destination.rmdir()
                self.assert_success(session, "pwd", f"{session.initial_cwd}\n")

    def test_saved_bash_env_does_not_run_before_the_command_shell_starts(self):
        for session in self.sessions:
            with self.subTest(training=session.training):
                startup = session.root / "bash startup.sh"
                startup.write_text("printf 'unexpected-startup\\n'\n")
                self.assert_success(session, f"export BASH_ENV={shlex.quote(str(startup))}", "")
                self.assert_success(session, "printf expected", "expected")

    def test_training_background_service_returns_and_survives_next_call(self):
        # Evaluation transports manage background lifetimes themselves. This
        # training-only assertion protects its Apptainer pipe-detachment fix.
        session = self.sessions[1]
        pid_file = session.root / "owned-service.pid"
        session.pid_files.append(pid_file)
        started = time.monotonic()
        self.assert_success(
            session, f"sleep 30 & printf '%s\\n' \"$!\" > {shlex.quote(str(pid_file))}; printf started", "started"
        )
        self.assertLess(time.monotonic() - started, 3)
        pid = int(pid_file.read_text())
        self.assert_success(session, f"kill -0 {pid} && printf alive", "alive")

    @unittest.skipUnless(shutil.which("timeout"), "Requires GNU timeout")
    def test_user_exit_124_and_inner_timeout_have_completion_records(self):
        session = self.sessions[1]
        for index, command in enumerate(("exit 124", "timeout 0.1s sleep 30")):
            with self.subTest(command=command):
                completion_path = session.root / f"completion-{index}"
                result = session.run(command, completion_path=completion_path)
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertEqual(completion_path.read_text(), "124\n")
                self.assert_success(session, "printf next-call", "next-call")

    @unittest.skipUnless(shutil.which("timeout"), "Requires GNU timeout")
    def test_training_timeout_terminates_detached_command_and_preserves_partial_output(self):
        session = self.sessions[1]
        pid_file = session.root / "owned-command-group.pid"
        completion_path = session.root / "timeout-completion"
        session.process_group_files.append(pid_file)
        result = session.run(
            f"printf '%s\\n' \"$$\" > {shlex.quote(str(pid_file))}; "
            "export TMAX_PARITY_VALUE=discarded; printf before-timeout; sleep 30",
            command_timeout=0.3,
            completion_path=completion_path,
        )
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(result.stdout, "before-timeout")
        self.assertFalse(completion_path.exists())
        pid = int(pid_file.read_text())
        self.assert_process_stopped(pid)
        self.assert_success(session, 'printf "%s" "${TMAX_PARITY_VALUE-unset}"', "unset")

    def assert_process_stopped(self, pid):
        stat_file = Path(f"/proc/{pid}/stat")
        # An exited orphan may briefly remain as a zombie until PID 1 reaps it.
        # It is no longer running and must not be mistaken for a leaked shell.
        if stat_file.exists():
            self.assertEqual(stat_file.read_text().rsplit(")", 1)[1].split()[0], "Z")
        else:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    @unittest.skipUnless(shutil.which("timeout"), "Requires GNU timeout")
    def test_training_timeout_kills_command_and_child_that_ignore_term(self):
        session = self.sessions[1]
        group_file = session.root / "owned-stubborn-command-group.pid"
        child_file = session.root / "owned-stubborn-child.pid"
        session.process_group_files.append(group_file)
        session.pid_files.append(child_file)
        started = time.monotonic()
        result = session.run(
            f"printf '%s\\n' \"$$\" > {shlex.quote(str(group_file))}; "
            "trap '' TERM; sleep 30 & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(str(child_file))}; wait",
            command_timeout=0.3,
        )
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        self.assert_process_stopped(int(group_file.read_text()))
        self.assert_process_stopped(int(child_file.read_text()))


if __name__ == "__main__":
    unittest.main()
