"""Exercise the agent wrapper through the installed, unmodified Harbor method."""

import ast
import asyncio
import contextlib
import importlib.util
import logging
import os
import shlex
import shutil
import signal
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

try:
    from harbor.environments import daytona as daytona_module
except ImportError:
    daytona_module = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("timeout"), "Requires Linux and GNU timeout")
class BackgroundCaptureTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if daytona_module is None:
            raise unittest.SkipTest("Requires Harbor")
        module = daytona_module
        source = Path(module.__file__)
        tree = ast.parse(source.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DaytonaEnvironment")
        fn = next(n for n in klass.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_sandbox_exec")
        ns = dict(module.__dict__)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
        cls.stock_execute = staticmethod(ns["_sandbox_exec"])
        spec = importlib.util.spec_from_file_location("capture", ROOT / "Vanillux2Agent/daytona_capture.py")
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        cls.capture = staticmethod(capture.execute_daytona_command)

    async def execute(self, environment, command, timeout_sec, cwd=None, env=None):
        if env:
            command = (
                "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " bash -c " + shlex.quote(command)
            )
        if cwd:
            command = "cd " + shlex.quote(cwd) + " && " + command
        out, err, code, timed_out = await self.capture(environment, command, timeout_sec)
        return SimpleNamespace(stdout=out, stderr=err, return_code=code, timed_out=timed_out)

    async def asyncSetUp(self):
        self.processes = []
        self.deleted = []
        self.sessions = {}

        async def create_session(session):
            pass

        async def delete_session(session):
            self.deleted.append(session)
            # SDK deletion is not relied upon for descendant cleanup.

        async def start(session, request, timeout):
            proc = await asyncio.create_subprocess_shell(
                request.command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True
            )
            self.sessions[session] = proc
            self.processes.append(proc)
            return SimpleNamespace(cmd_id=str(len(self.processes) - 1))

        async def poll(session, command):
            proc = self.processes[int(command)]
            out, err = await proc.communicate()
            return SimpleNamespace(return_code=proc.returncode, stdout=out.decode(), stderr=err.decode())

        self.env = SimpleNamespace(
            _sandbox=SimpleNamespace(
                process=SimpleNamespace(
                    create_session=create_session, delete_session=delete_session, execute_session_command=start
                )
            ),
            _poll_response=poll,
            logger=logging.getLogger(__name__),
        )

        async def stock_exec(**kwargs):
            return await self.stock_execute(self.env, **kwargs)

        self.env.exec = stock_exec

    async def test_background_and_and_list_return_before_child_exit(self):
        for command in ["sleep 2 & echo ok", "cd /tmp && sleep 2 & echo ok"]:
            result = await asyncio.wait_for(self.execute(self.env, command, timeout_sec=1), 1)
            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(self.deleted, [])

    async def test_foreground_timeout_preserves_partial_output(self):
        result = await self.execute(self.env, "echo partial; sleep 10", timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        self.assertEqual(result.stdout, "partial\n")
        self.assertTrue(result.timed_out)

    async def test_exit_status_stderr_cwd_and_environment(self):
        result = await self.execute(
            self.env,
            'pwd; echo "$PROBE"; echo error >&2; exit 124',
            cwd="/tmp",
            env={"PROBE": 'spaces and "quotes"'},
            timeout_sec=1,
        )
        self.assertEqual(result.return_code, 124)
        self.assertIn('spaces and "quotes"', result.stdout)
        self.assertEqual(result.stderr, "error\n")

    async def test_poll_failure_propagates(self):
        async def fail(*args):
            raise RuntimeError("transport failure")

        self.env._poll_response = fail
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            await self.execute(self.env, "true", timeout_sec=1)
        self.assertEqual(self.deleted, [])
        for proc in self.processes:
            await proc.communicate()

    @staticmethod
    def kill_session(session_id):
        # Test teardown removes services intentionally preserved by execution,
        # including GNU timeout's separate group inside the test session.
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                if int(fields[3]) == session_id:
                    os.kill(int(stat.parent.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass

    async def asyncTearDown(self):
        for proc in self.processes:
            with contextlib.suppress(ProcessLookupError):
                self.kill_session(proc.pid)
            await proc.wait()

    async def test_explicit_timeout_codes_preserve_background_services(self):
        for code in (124, 137):
            result = await self.execute(self.env, f"sleep 30 & echo $!; exit {code}", timeout_sec=1)
            self.assertEqual(result.return_code, code)
            self.assertFalse(result.timed_out)
            self.assertEqual(result.stderr, "")
            os.kill(int(result.stdout.strip()), 0)
        self.assertEqual(self.deleted, [])

    async def test_timeout_cleans_resistant_grandchild_but_preserves_prior_service(self):
        earlier = await self.execute(self.env, "sleep 30 & echo $!", timeout_sec=1)
        result = await self.execute(self.env, "bash -c 'trap \"\" TERM; echo $$; sleep 30' & sleep 30", timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        self.assertEqual(self.deleted, [])
        pid = int(result.stdout.strip())
        for _ in range(100):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("TERM-resistant grandchild survived command-group cleanup")
        os.kill(int(earlier.stdout.strip()), 0)

    async def test_escalated_deadline_is_not_explicit_exit_137(self):
        result = await self.execute(self.env, "bash -c 'trap \"\" TERM; echo partial; sleep 30'", timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        self.assertEqual(result.stdout, "partial\n")
        self.assertTrue(result.timed_out)
        self.assertEqual(self.deleted, [])

    async def test_sdk_final_record_newline_is_removed_without_changing_stderr(self):
        original_poll = self.env._poll_response

        async def framed_poll(*args):
            result = await original_poll(*args)
            result.stderr += "\n"
            return result

        self.env._poll_response = framed_poll
        result = await self.execute(self.env, "printf error >&2", timeout_sec=1)
        self.assertEqual(result.stderr, "error")
        self.assertEqual(result.return_code, 0)

    async def test_missing_completion_evidence_is_infrastructure_error(self):
        async def corrupt(*args):
            return SimpleNamespace(stdout="", stderr="", return_code=124)

        self.env._poll_response = corrupt
        with self.assertRaisesRegex(RuntimeError, "Missing Daytona command completion"):
            await self.execute(self.env, "true", timeout_sec=1)
        for proc in self.processes:
            await proc.communicate()

    async def test_broad_pkill_preserves_capture_and_returns_partial_output(self):
        # Generated at runtime so only this test command's inner shell matches.
        for kill_flag in ("", "-9 "):
            marker = "capture_probe_" + uuid.uuid4().hex
            result = await self.execute(
                self.env, f"echo partial; pkill {kill_flag}-f {marker}; echo unreachable", timeout_sec=2
            )
            self.assertEqual(result.return_code, 137 if kill_flag else 143)
            self.assertFalse(result.timed_out)
            self.assertEqual(result.stdout, "partial\n")
            recovered = await self.execute(self.env, "echo recovered", timeout_sec=2)
            self.assertEqual(recovered.stdout, "recovered\n")

    async def test_command_text_quoting_unicode_and_multiline_are_preserved(self):
        result = await self.execute(self.env, "printf '%s\\n' \"spaces ' \\\" café\"\nprintf tail", timeout_sec=2)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout, "spaces ' \" café\ntail")

    async def test_decoder_preserves_trailing_backslash_newlines(self):
        command = "printf '%s' preserved" + "\\\n\n"
        result = await self.execute(self.env, command, timeout_sec=2)
        self.assertEqual(result.stdout, "preserved")
        self.assertEqual(result.return_code, 0)

    async def test_decoder_failure_is_not_success(self):
        stock_exec = self.env.exec

        async def failed_decoder(**kwargs):
            kwargs["command"] = "base64() { return 42; }; export -f base64\n" + kwargs["command"]
            return await stock_exec(**kwargs)

        self.env.exec = failed_decoder
        result = await self.execute(self.env, "echo must-not-run", timeout_sec=2)
        self.assertEqual(result.return_code, 125)
        self.assertEqual(result.stdout, "")
        self.assertFalse(result.timed_out)
