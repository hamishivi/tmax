"""Environment interaction module for agent testing in Apptainer containers with interactive shell (PTY-backed, promptless)."""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import queue
import re
import shutil
import subprocess
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # Strip ANSI escapes


class InteractiveContainerEnvironment:
    """Manages interaction with a pre-built Apptainer container using an interactive shell over a PTY."""

    def __init__(
        self,
        container_sif_path: str,
        initial_test_path: str,
        final_test_path: str,
        def_path: str,
        max_actions: int = 50,
        verbose: bool = True,
        read_timeout: float = 30.0,
    ):
        # Resolve all incoming paths to absolute paths
        self.sif_path = Path(container_sif_path).expanduser().resolve()
        self.initial_test_path = Path(initial_test_path).expanduser().resolve()
        self.final_test_path = Path(final_test_path).expanduser().resolve()
        self.def_path = Path(def_path).expanduser().resolve()

        self.max_actions = max_actions
        self.verbose = verbose
        self.read_timeout = read_timeout

        self.temp_dir: Optional[Path] = None
        self.action_history: List[Dict[str, str]] = []
        self.instance_name: Optional[str] = None

        self.shell_process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None

        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Unique marker to delimit command completion and carry exit code
        self._marker = f"__CMD_DONE__{uuid.uuid4().hex}__"

    # ----------------------------
    # Low-level PTY I/O utilities
    # ----------------------------
    def _reader_loop(self) -> None:
        """Background thread to read from PTY master and push text into a queue (no selector)."""
        fd = self.master_fd
        if fd is None:
            return
        # Non-blocking
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while (
                not self._stop_event.is_set()
                and self.shell_process
                and self.shell_process.poll() is None
            ):
                try:
                    data = os.read(fd, 16384)
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        self.output_queue.put_nowait(text)
                        continue
                except BlockingIOError:
                    pass
                except OSError as e:
                    if getattr(e, "errno", None) in (errno.EBADF, errno.EIO):
                        break
                    raise
                time.sleep(0.005)
        finally:
            try:
                while True:
                    data = os.read(fd, 16384)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    self.output_queue.put_nowait(text)
            except Exception:
                pass

    def _drain_queue(self) -> str:
        chunks: List[str] = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(chunks)

    def _read_until_marker(self, timeout: Optional[float] = None) -> Tuple[str, Optional[int]]:
        """
        Read buffered output until we see our unique marker line, e.g. '__CMD_DONE__...__:0'
        Returns (output_without_marker, exit_code or None on timeout)
        """
        if timeout is None:
            timeout = self.read_timeout

        deadline = time.time() + timeout
        buf = []

        marker_match = None
        while time.time() < deadline:
            # pull whatever we have
            chunk = self._drain_queue()
            if chunk:
                buf.append(chunk)
                joined = "".join(buf)
                # try to find the last marker occurrence (in case command prints similar text)
                for line in joined.splitlines():
                    if self._marker in line:
                        # marker format: {marker}:{exit_code}
                        if ":" in line:
                            parts = line.rsplit(":", 1)
                            if len(parts) == 2 and parts[0].endswith(self._marker):
                                try:
                                    code = int(parts[1].strip())
                                except ValueError:
                                    code = None
                                marker_match = (joined, code)
                if marker_match:
                    full_out, code = marker_match
                    first_marker_index = full_out.find(self._marker)
                    cleaned = full_out[:first_marker_index]
                    return cleaned, code
            time.sleep(0.002)

        # timeout: return whatever we accumulated
        return "".join(buf), None

    # ----------------------------
    # Shell lifecycle
    # ----------------------------
    def _start_shell(self) -> bool:
        """Start an interactive Apptainer shell session on a PTY."""
        if self.shell_process:
            return True

        # Create PTY pair
        self.master_fd, self.slave_fd = pty.openpty()

        # Make the slave a proper TTY with sane settings
        try:
            attrs = termios.tcgetattr(self.slave_fd)
            # disable echo (we'll still see command output, avoids double-echo)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        # Compose apptainer shell command
        cmd = [
            "apptainer", "shell",
            "--cleanenv",
            "--pwd", "/home/user",
            f"instance://{self.instance_name}",
        ]

        try:
            # Launch with PTY endpoints
            self.shell_process = subprocess.Popen(
                cmd,
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                close_fds=True,
                start_new_session=True,  # new process group
            )

            # Start reader thread
            self._stop_event.clear()
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            # Give shell a brief moment to start up
            time.sleep(0.05)
            if self.shell_process.poll() is not None:
                if self.verbose:
                    print("Apptainer shell exited early with code:", self.shell_process.returncode)
                leftover = self._drain_queue()
                if leftover:
                    print("Shell start output:\n", leftover)
                return False

            # Initialize shell: make it predictable
            init_script = (
                "set -o pipefail 2>/dev/null; "
                "export PS1='[$PWD]$ '; "
                "export HOME=/home/user; "
                "cd \"$HOME\" 2>/dev/null || true; "
                f"printf '{self._marker}:0\\n'"
            )
            os.write(self.master_fd, (init_script + "\n").encode("utf-8"))
            _, code = self._read_until_marker(timeout=10.0)
            if code is None:
                if self.verbose:
                    print("Shell init timed out.")
                return False

            if self.verbose:
                init_out = self._drain_queue()
                if init_out:
                    print(f"Shell started. Initial output:\n{init_out}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to start shell: {e}")
            return False

    def _stop_shell(self):
        """Stop the interactive shell session and close PTY."""
        try:
            # signal reader to stop before we close fds
            self._stop_event.set()
            if self.reader_thread:
                try:
                    self.reader_thread.join(timeout=1.0)
                except Exception:
                    pass
                self.reader_thread = None

            if self.shell_process and self.shell_process.poll() is None:
                try:
                    os.write(self.master_fd, b"exit\n")
                except Exception:
                    pass
                try:
                    self.shell_process.wait(timeout=2)
                except Exception:
                    self.shell_process.terminate()
                    try:
                        self.shell_process.wait(timeout=2)
                    except Exception:
                        self.shell_process.kill()
        finally:
            self.shell_process = None

            # Close PTY fds
            for fd in (self.master_fd, self.slave_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
            self.master_fd = None
            self.slave_fd = None

    def _stop_instance(self) -> None:
        """Stop the Apptainer instance if running."""
        if self.instance_name:
            subprocess.run(
                ["apptainer", "instance", "stop", self.instance_name],
                capture_output=True
            )
            self.instance_name = None

    # ----------------------------
    # Public API
    # ----------------------------
    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def initialize(self, run_initial_tests: bool = True) -> bool:
        """Initialize the container environment and validate initial state."""
        if self.verbose:
            print(f"🔧 Initializing container environment with {self.sif_path.name}...")

        if not self.sif_path.exists():
            if self.verbose:
                print(f"⚠️  SIF not found at {self.sif_path}, attempting to build...")
            if self.def_path.exists():
                self.build_container()
            else:
                print("❌ Neither SIF nor def file exists")
                return False

        # Create temporary directory for test files (on host)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_env_")).resolve()

        # Start a long-lived Apptainer instance
        self.instance_name = f"agent_{uuid.uuid4().hex[:8]}"
        start_cmd = [
            "apptainer", "instance", "start",
            "--fakeroot",
            "--userns",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ]
        if self.verbose:
            print(f"🔧 Starting instance with command: {' '.join(start_cmd)}")
        start_proc = subprocess.run(start_cmd, capture_output=True, text=True)
        if start_proc.returncode != 0:
            if self.verbose:
                print(f"❌ Instance start failed: {start_proc.stdout + start_proc.stderr}")
            return False
        else:
            if self.verbose:
                print(f"✅ Instance started: {start_proc.stdout + start_proc.stderr}")

        # Start the interactive shell
        if not self._start_shell():
            if self.verbose:
                print("❌ Failed to start interactive shell")
            self._stop_instance()
            return False
        else:
            if self.verbose:
                print("✅ Interactive shell started")

        # Run initial tests if requested
        if run_initial_tests:
            if not self.run_initial_tests():
                if self.verbose:
                    print("❌ Initial state tests failed")
                self._stop_shell()
                self._stop_instance()
                return False

        if self.verbose:
            print("✅ Container environment ready")
        self.exec("cd /home/user")
        return True

    def exec(self, command: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """
        Execute a command in the interactive shell.

        Returns:
            (success, output_without_marker_and_ansi)
        """
        # Check if shell exists
        if not self.shell_process:
            if not self._start_shell():
                return False, "Failed to start shell"

        # Check if shell process is still alive
        if self.shell_process.poll() is not None:
            if self.verbose:
                print(f"⚠️  Shell process died (exit code: {self.shell_process.returncode}), restarting...")
            self.shell_process = None
            if not self._start_shell():
                return False, "Shell process died and restart failed"

        # Check if reader thread is still alive
        if not self.reader_thread or not self.reader_thread.is_alive():
            if self.verbose:
                print("⚠️  Reader thread died, restarting shell...")
            self._stop_shell()
            if not self._start_shell():
                return False, "Reader thread died and shell restart failed"

        # Clear any stale output
        _ = self._drain_queue()

        # Wrap the command to always emit our marker with the exit code
        # Use a subshell to ensure we capture the correct `$?` across pipelines
        # Special-case heredocs: avoid grouping with braces so the terminator can be on its own line
        if "<<" in command:
            wrapped = f"{command}\ncode=$?; printf '{self._marker}:%s\\n' \"$code\""
        else:
            wrapped = f"{{ {command}; }}; code=$?; printf '{self._marker}:%s\\n' \"$code\""
        
        try:
            os.write(self.master_fd, (wrapped + "\n").encode("utf-8"))
        except Exception as e:
            return False, f"Command write failed: {e}"

        time.sleep(0.01)
        if self.shell_process.poll() is not None:
            return False, f"Shell died immediately after command (exit code: {self.shell_process.returncode})"

        raw_out, code = self._read_until_marker(timeout=timeout)
        
        # Handle timeout
        if code is None:
            if self.verbose:
                print(f"⚠️  Command timed out after {timeout or self.read_timeout}s")
            return False, f"Command timed out. Partial output:\n{raw_out[:500]}"
        
        # Clean output: strip ANSI, strip echoed lines (PTY has no echo, but some programs add it)
        cleaned = ANSI_RE.sub("", raw_out)
        cleaned = cleaned.replace("\r", "")

        success = (code == 0)
        return success, cleaned

    def run_initial_tests(self) -> bool:
        """Run initial state validation tests."""
        if self.verbose:
            print("🧪 Running initial state tests...")

        with open(self.initial_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_initial.py"

        # Write test file with retry logic
        max_retries = 1
        retry_delay = 0.1

        for attempt in range(max_retries):
            marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
            write_cmd = (
                f"cat <<'{marker}' > {test_path_in_container}\n"
                f"{test_file_text}\n"
                f"{marker}\n"
            )

            success, output = self.exec(write_cmd)
            if success:
                break

            if self.verbose:
                print(f"⚠️  Write attempt {attempt + 1}/{max_retries} failed: {output[:100]}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if self.verbose:
                    print(f"❌ Failed to write test file after {max_retries} attempts: {output}")
                return False

        # Run pytest on the file inside the instance
        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")

        # Clean up the test file from the instance
        self.exec(f"rm -f {test_path_in_container}")

        if not test_success:
            if self.verbose:
                print(f"Initial test output:\n{test_output}")
        else:
            if self.verbose:
                print("✅ Initial state tests passed")
        
        return test_success

    def run_final_tests(self) -> Tuple[bool, str]:
        """Run final state validation tests inside the instance."""
        if self.verbose:
            print("🧪 Running final state tests...")

        with open(self.final_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_final.py"

        # Write test file with retry logic
        max_retries = 1
        retry_delay = 0.1

        for attempt in range(max_retries):
            marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
            write_cmd = (
                f"cat <<'{marker}' > {test_path_in_container}\n"
                f"{test_file_text}\n"
                f"{marker}\n"
            )
            ok, write_out = self.exec(write_cmd)

            if ok:
                break

            if self.verbose:
                print(f"⚠️  Write attempt {attempt + 1}/{max_retries} failed: {write_out[:100]}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if self.verbose:
                    print(f"❌ Failed to write final test file after {max_retries} attempts: {write_out}")
                return False, write_out

        # Run pytest inside the instance
        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")

        # Clean up test file
        self.exec(f"rm -f {test_path_in_container}")

        if self.verbose:
            if test_success:
                print("✅ Final state tests passed!")
            else:
                print("❌ Final state tests failed")
                print(test_output)

        return test_success, test_output

    def cleanup(self):
        """Clean up temporary files and processes."""
        self._stop_shell()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._stop_instance()
        if self.verbose:
            print("🧹 Cleaned up temporary files and processes")

    def build_container(self):
        """Rebuild sif from def."""
        with open(self.def_path, "r") as f:
            def_text = f.read()

        if "chmod 755 /home/user" not in def_text:
            section_headers = [line for line in def_text.split("\n") if line.strip().startswith("%")]
            post_idx = [i for i, line in enumerate(section_headers) if "post" in line.lower()]
            if post_idx:
                idx = post_idx[0]
                if idx + 1 < len(section_headers):
                    next_header = section_headers[idx + 1]
                    def_text = def_text.replace(
                        next_header, "    chmod 755 /home/user\n" + next_header
                    )
                else:
                    def_text = def_text.rstrip() + "\n    chmod 755 /home/user\n"
                with open(self.def_path, "w") as f:
                    f.write(def_text)

        build_rc = subprocess.run(
            ["apptainer", "build", str(self.sif_path), str(self.def_path)],
            capture_output=True,
            text=True,
        )
        if build_rc.returncode != 0:
            print(f"Apptainer build failed: {build_rc.stdout + build_rc.stderr}")
            return False
        return build_rc.returncode == 0

    def get_prompt(self) -> str:
        """Get the current shell prompt showing the working directory."""
        success, output = self.exec("pwd")
        if success and output:
            current_dir = output.strip().splitlines()[-1]
            return f"({self.sif_path.name}) {current_dir} $ "
        return f"({self.sif_path.name}) $ "


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = parser.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"
    final_test_path = task_path / "test_final_state.py"
    container_sif_path = task_path / "container.sif"

    # sample_task.json is optional; guard it to avoid crashes
    task_description = ""
    truth = ""
    if Path("sample_task.json").exists():
        with open("sample_task.json", "r") as f:
            task_data = json.load(f)
        task_description = task_data.get("description", "")
        truth = task_data.get("truth", "")

    env = InteractiveContainerEnvironment(
        container_sif_path=container_sif_path,
        initial_test_path=initial_test_path,
        final_test_path=final_test_path,
        def_path=def_path,
        verbose=True,
    )
    if not container_sif_path.exists():
        env.build_container()

    if not env.initialize(run_initial_tests=True):
        raise SystemExit(1)

    try:
        print("\nStarting interactive session with the container...")
        print("Type 'exit' or 'quit' to finish.")
        if task_description:
            print(f"Task description: {task_description}")

        while True:
            try:
                prompt = env.get_prompt()
                command = input(prompt)
                if command.lower() in ["exit", "quit"]:
                    break
                if not command.strip():
                    continue
                success, output = env.exec(command)
                if output:
                    print(output)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive session.")
                break
    finally:
        env.run_final_tests()
        env.cleanup()
