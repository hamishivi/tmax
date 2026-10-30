"""Execute the Harbor agent's setup/command methods without GPU dependencies."""

import asyncio
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_vanillux_protocol_parity import AGENT_PATH, load_definitions

NS = {"BaseEnvironment": object, "shlex": shlex, "uuid": uuid}
# Only the return object is replaced; all shell/timeout logic is production code.
NS["_CommandResult"] = lambda stdout, stderr, return_code, timed_out: SimpleNamespace(
    stdout=stdout, stderr=stderr, return_code=return_code, timed_out=timed_out
)
load_definitions(AGENT_PATH, {"_execute_bash", "setup", "_wrap_command"}, NS, "Vanillux2Agent")


def result(code=0, stdout="", stderr=""):
    return SimpleNamespace(return_code=code, stdout=stdout, stderr=stderr)


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(command_timeout=120, _wrap_command=lambda command: command)

    def execute(self, *results):
        environment = SimpleNamespace(exec=AsyncMock(side_effect=results))
        output = asyncio.run(NS["_execute_bash"](self.agent, "exit 124", environment))
        return output, environment

    def test_ordinary_124_clears_false_timeout_prefix_and_removes_marker(self):
        output, environment = self.execute(
            result(124, "out", "Command timed out after 120s.\nuser stderr"), result(stdout="124\n")
        )
        self.assertFalse(output.timed_out)
        self.assertEqual(output.stderr, "user stderr")
        commands = [c.kwargs["command"] for c in environment.exec.call_args_list]
        marker = commands[0].split(" > ")[1].splitlines()[0]
        self.assertIn(f"cat {marker}; rm -f {marker}", commands[1])

    def test_deadline_without_completion_record(self):
        output, _ = self.execute(result(124, "partial", "timeout"), result())
        self.assertTrue(output.timed_out)
        self.assertEqual(output.stdout, "partial")

    def test_explicit_deadline_flag_is_authoritative_for_all_exit_codes(self):
        for code in (0, 124, 137, 143):
            for timed_out in (True, False):
                with self.subTest(code=code, timed_out=timed_out):
                    response = result(code, "partial", "stderr")
                    response.timed_out = timed_out
                    environment = SimpleNamespace(
                        preserves_command_timeout=True, exec=AsyncMock(return_value=response)
                    )
                    output = asyncio.run(NS["_execute_bash"](self.agent, "sleep 10", environment))
                    self.assertEqual(output.timed_out, timed_out)
                    self.assertEqual(output.return_code, code)
                    environment.exec.assert_awaited_once_with(command="sleep 10", timeout_sec=120)

    def test_explicit_deadline_contract_cannot_silently_fall_back(self):
        for value in (None, "false", 0):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "omitted timed_out"):
                response = result(143)
                response.timed_out = value
                environment = SimpleNamespace(preserves_command_timeout=True, exec=AsyncMock(return_value=response))
                asyncio.run(NS["_execute_bash"](self.agent, "sleep 10", environment))

    def test_probe_failure_is_infrastructure_error(self):
        with self.assertRaisesRegex(RuntimeError, "probe exit_code=1"):
            self.execute(result(124), result(1))

    def test_backend_loss_propagates_without_reclassifying_as_timeout(self):
        for responses in [(RuntimeError("worker lost"),), (result(124), RuntimeError("worker lost"))]:
            with self.subTest(responses=responses), self.assertRaisesRegex(RuntimeError, "worker lost"):
                self.execute(*responses)

    def test_setup_failure_is_not_silently_ignored(self):
        NS["_STATE_DIR"] = "/tmp/test-only-state"
        for code in (1, 124):
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, "Failed to initialize"):
                asyncio.run(
                    NS["setup"](
                        SimpleNamespace(persistent_bash=True),
                        SimpleNamespace(exec=AsyncMock(return_value=result(code, stderr="cannot write state"))),
                    )
                )

    def test_nonpersistent_setup_does_not_run(self):
        environment = SimpleNamespace(exec=AsyncMock())
        asyncio.run(NS["setup"](SimpleNamespace(persistent_bash=False), environment))
        environment.exec.assert_not_called()


@unittest.skipUnless(shutil.which("bash"), "Requires Bash")
class RealCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vanillux-exec-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        NS["_STATE_DIR"] = str(self.state)
        self.agent = SimpleNamespace(persistent_bash=True, command_timeout=2)
        self.agent._wrap_command = lambda command: NS["_wrap_command"](self.agent, command)
        self.environment = SimpleNamespace(exec=self.exec)
        asyncio.run(NS["setup"](self.agent, self.environment))

    async def exec(self, command, timeout_sec):
        # Model the backend contract: kill/drain this command's process group
        # before returning a deadline result. No existing processes are touched.
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=self.root,
            env={"PATH": os.defpath, "HOME": str(self.root)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            return result(proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=2)
            return result(124, stdout, stderr)

    def run_command(self, command):
        return asyncio.run(NS["_execute_bash"](self.agent, command, self.environment))

    def test_explicit_exit_124_is_not_a_deadline(self):
        output = self.run_command("printf deliberate; exit 124")
        self.assertEqual(output.return_code, 124)
        self.assertFalse(output.timed_out)
        self.assertEqual(output.stdout, "deliberate")

    @unittest.skipUnless(shutil.which("timeout"), "Requires timeout")
    def test_inner_timeout_is_not_backend_deadline(self):
        output = self.run_command(f"{shlex.quote(shutil.which('timeout'))} 0.1s sleep 10")
        self.assertEqual(output.return_code, 124)
        self.assertFalse(output.timed_out)

    def test_backend_deadline_preserves_partial_output(self):
        self.agent.command_timeout = 0.1
        output = self.run_command("printf before; sleep 10")
        self.assertEqual(output.return_code, 124)
        self.assertTrue(output.timed_out)
        self.assertEqual(output.stdout, "before")

    def test_persistent_state_and_fakeroot_filter_survive_command_envelope(self):
        directory = self.root / "directory with 'quotes'"
        directory.mkdir()
        output = self.run_command(f"cd {shlex.quote(str(directory))}; export VALUE=kept FAKEROOTKEY=stale; false")
        self.assertEqual(output.return_code, 1)
        output = self.run_command('pwd; printf "%s:%s" "$VALUE" "${FAKEROOTKEY-unset}"')
        self.assertEqual(output.stdout, f"{directory}\nkept:unset")

    def test_nonpersistent_commands_also_distinguish_exit_124(self):
        self.agent.persistent_bash = False
        output = self.run_command("exit 124")
        self.assertFalse(output.timed_out)


if __name__ == "__main__":
    unittest.main()
