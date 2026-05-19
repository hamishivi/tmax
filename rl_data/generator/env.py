"""Environment interaction module for agent testing in Apptainer containers with interactive shell (PTY-backed, promptless)."""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import queue
import re
import shlex
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


def _fakeroot_flags() -> List[str]:
    """Return the apptainer fakeroot flag list, honouring an env opt-out.

    Some upstream Docker base images (notably TerminalTraj's older Fedora /
    glibc<2.33 layers, but also some Alpine/Ubuntu variants) ship a bundled
    ``/.singularity.d/libs/fakeroot`` whose embedded glibc is incompatible
    with the host kernel/userspace, causing solve-time
    ``apptainer instance start --fakeroot`` and ``apptainer exec --fakeroot``
    to crash with::

        FATAL:   exec /.singularity.d/libs/fakeroot failed: a shared library
                 is likely missing in the image

    Setting ``APPTAINER_IGNORE_FAKEROOT_COMMAND=1`` adds
    ``--ignore-fakeroot-command`` to every fakeroot invocation, which makes
    apptainer skip the bundled binary and fall back to its built-in
    user-namespace fakeroot emulation.  Default off so unaffected datasets
    (ET / OT / TermiGen / our 10k) preserve current behaviour.
    """
    if os.environ.get("APPTAINER_IGNORE_FAKEROOT_COMMAND", "0") == "1":
        return ["--fakeroot", "--ignore-fakeroot-command"]
    return ["--fakeroot"]


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
        #: Time to wait for the first shell prompt / init marker. Under heavy
        #: concurrent Apptainer load (many workers × many solutions), 10s is often
        #: too short and causes spurious "Shell init timed out".
        shell_init_timeout: float = 120.0,
        shell_init_attempts: int = 3,
        #: Directory containing pre-built base SIFs (``base_{domain}.sif``).
        #: When set, the env uses a shared base SIF + task delta instead of a
        #: per-task SIF (saves ~200MB per task on disk).
        base_sifs_dir: Optional[str] = None,
    ):
        # Resolve all incoming paths to absolute paths
        self.sif_path = Path(container_sif_path).expanduser().resolve()
        self.initial_test_path = Path(initial_test_path).expanduser().resolve()
        self.final_test_path = Path(final_test_path).expanduser().resolve()
        self.def_path = Path(def_path).expanduser().resolve()
        self.base_sifs_dir = Path(base_sifs_dir).resolve() if base_sifs_dir else None

        self.max_actions = max_actions
        self.verbose = verbose
        self.read_timeout = read_timeout
        self.shell_init_timeout = shell_init_timeout
        self.shell_init_attempts = max(1, shell_init_attempts)

        self.temp_dir: Optional[Path] = None
        #: Host copy of image /home/user, bind-mounted over /home/user in the instance
        self._writable_home_path: Optional[Path] = None
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
    def _start_shell_once(self) -> bool:
        """Single attempt: PTY + apptainer shell + wait for init marker."""
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

            # Under concurrent instance load, the shell needs longer than a few ms
            time.sleep(0.2)
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
            _, code = self._read_until_marker(timeout=self.shell_init_timeout)
            if code is None:
                if self.verbose:
                    print(
                        f"Shell init timed out after {self.shell_init_timeout}s "
                        "(try lowering --workers / concurrent solutions, or raise --shell-init-timeout)."
                    )
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

    def _start_shell(self) -> bool:
        """Start an interactive Apptainer shell session on a PTY (with retries)."""
        if self.shell_process:
            return True

        for attempt in range(self.shell_init_attempts):
            if attempt > 0:
                self._stop_shell()
                delay = min(4.0, 0.5 * (2 ** (attempt - 1)))
                if self.verbose:
                    print(
                        f"Retrying shell start ({attempt + 1}/{self.shell_init_attempts}) "
                        f"after {delay:.1f}s..."
                    )
                time.sleep(delay)

            if self._start_shell_once():
                return True

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
            # Drop any stale PTY output so the next shell start doesn't confuse readers
            try:
                while True:
                    self.output_queue.get_nowait()
            except queue.Empty:
                pass

    def _recover_shell(self) -> None:
        """Recover the shell after a command timeout by killing the running command and re-syncing.

        Without this, a timed-out command keeps running in the shell. The next
        exec() sends a new command into the still-busy shell, corrupting its
        state. Every subsequent command then also times out, cascading into
        total failure.
        """
        try:
            # Send Ctrl+C twice to interrupt running command (handles subshells)
            os.write(self.master_fd, b'\x03')
            time.sleep(0.3)
            os.write(self.master_fd, b'\x03')
            time.sleep(0.3)

            # Drain any output from the killed command
            self._drain_queue()

            # Re-sync: send a fresh marker and verify the shell responds
            sync_cmd = f"printf '{self._marker}:0\\n'\n"
            os.write(self.master_fd, sync_cmd.encode("utf-8"))
            _, sync_code = self._read_until_marker(timeout=5.0)

            if sync_code is None:
                if self.verbose:
                    print("⚠️  Shell unresponsive after Ctrl+C, restarting...")
                self._stop_shell()
                self._start_shell()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Recovery failed ({e}), restarting shell...")
            try:
                self._stop_shell()
                self._start_shell()
            except Exception:
                pass

    def _stop_instance(self) -> None:
        """Stop the Apptainer instance if running."""
        if self.instance_name:
            subprocess.run(
                ["apptainer", "instance", "stop", self.instance_name],
                capture_output=True
            )
            self.instance_name = None

    def _resolve_runtime_sif(self) -> Optional[Path]:
        """Decide which SIF to use: per-task SIF, or a shared base SIF from ``base_sifs_dir``.

        Resolution order at the base-SIF stage:
            1. ``task.json["base_image"]`` if set (v2 tasks set this to
               ``"intricate"`` for non-legacy verifier_kind / fixture_kind).
            2. ``task.json["domain"]`` (the legacy resolution path).
            3. The default ``software_engineering`` fallback baked into
               ``parse_def_to_delta`` and ``_resolve_base``.
        """
        if self.sif_path.exists():
            return self.sif_path

        if self.base_sifs_dir and self.def_path.exists():
            from rl_data.generator.apptainer_def_gen import parse_def_to_delta, _resolve_base
            task_json = self.def_path.parent / "task.json"
            domain_hint: Optional[str] = None
            base_image_hint: Optional[str] = None
            if task_json.exists():
                try:
                    import json as _json
                    _meta = _json.loads(task_json.read_text())
                    domain_hint = _meta.get("domain")
                    base_image_hint = _meta.get("base_image")
                except Exception:
                    pass

            # base_image_hint takes precedence over domain_hint when set —
            # this is how v2 tasks route to base_intricate.sif without
            # touching their domain field (which is preserved for analysis).
            resolved_hint = base_image_hint or domain_hint
            base_domain, _ = parse_def_to_delta(self.def_path.read_text(), domain_hint=resolved_hint)
            base_sif = _resolve_base(base_domain, base_sifs_dir=self.base_sifs_dir)
            if base_sif.exists():
                return base_sif
        return None

    def _get_setup_script(self) -> Optional[str]:
        """Return the task-specific setup script content, or None.

        Prefers ``setup.sh`` next to the def; falls back to parsing the ``.def``.
        """
        setup_sh = self.def_path.parent / "setup.sh"
        if setup_sh.exists():
            return setup_sh.read_text(encoding="utf-8")
        if self.def_path.exists():
            from rl_data.generator.apptainer_def_gen import parse_def_to_delta
            task_json = self.def_path.parent / "task.json"
            domain_hint = None
            if task_json.exists():
                try:
                    import json as _json
                    domain_hint = _json.loads(task_json.read_text()).get("domain")
                except Exception:
                    pass
            _, setup_body = parse_def_to_delta(self.def_path.read_text(), domain_hint=domain_hint)
            return setup_body
        return None

    def _materialize_writable_home_user(self, sif_for_materialize: Path) -> Tuple[bool, str]:
        """Copy the image's ``/home/user`` tree to the host.

        With ``--fakeroot`` + ``--writable-tmpfs``, ``/home/user`` often lives on
        ``fuse-overlayfs``. Creating regular files there can fail with
        ``OSError: [Errno 22] Invalid argument``. Binding a normal host directory
        over ``/home/user`` makes task outputs and tests reliable.
        """
        assert self.temp_dir is not None
        dest = self.temp_dir / "_writable_home_user"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        inner_mount = "/mnt/_agent_home_materialize"
        copy_cmd = [
            "apptainer", "exec",
            *_fakeroot_flags(), "--userns", "--writable-tmpfs", "--cleanenv",
            # See _start_instance for the --no-home rationale (R2E Gym bug).
            "--no-home",
            "--bind", f"{dest}:{inner_mount}",
            str(sif_for_materialize),
            "/bin/sh", "-c", f"cp -a /home/user/. {inner_mount}/",
        ]
        proc = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            err = (proc.stdout or "") + (proc.stderr or "")
            return False, f"materialize /home/user failed (exit {proc.returncode}): {err.strip() or 'no output'}"

        self._writable_home_path = dest
        if self.verbose:
            print(f"✅ Materialized writable /home/user at {dest}")
        return True, ""

    # Wall-clock cap for running a per-task `_delta_setup.sh` inside the
    # base SIF.  Some skill_tax tasks legitimately take >>600s here (large
    # data fixture generation, slow compiles, optional rustup re-install on
    # non-software_engineering bases, etc.).  Bumping to 30 min covers the
    # long tail without letting truly-stuck deltas park forever.
    _DELTA_SETUP_TIMEOUT_S = 1800

    def _apply_setup_delta(self, sif_for_exec: Path, setup_script: str) -> Tuple[bool, str]:
        """Run the task-specific setup script inside the container.

        The script runs with ``/home/user`` bind-mounted from the host so file
        creation goes to the real filesystem (not fuse-overlayfs).
        """
        assert self.temp_dir is not None and self._writable_home_path is not None
        script_path = self.temp_dir / "_delta_setup.sh"
        script_path.write_text(setup_script, encoding="utf-8")
        script_path.chmod(0o755)

        exec_cmd = [
            "apptainer", "exec",
            *_fakeroot_flags(), "--userns", "--writable-tmpfs", "--cleanenv",
            # See _start_instance for the --no-home rationale (R2E Gym bug).
            "--no-home",
            "--pwd", "/home/user",
            "--bind", f"{self._writable_home_path}:/home/user",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            str(sif_for_exec),
            "/bin/bash", str(script_path),
        ]
        proc = subprocess.run(
            exec_cmd, capture_output=True, text=True,
            timeout=self._DELTA_SETUP_TIMEOUT_S,
        )
        if proc.returncode != 0:
            err = (proc.stdout or "") + (proc.stderr or "")
            return False, f"setup delta failed (exit {proc.returncode}): {err.strip()[-500:] or '(no output)'}"
        return True, ""

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

        runtime_sif = self._resolve_runtime_sif()
        using_base_sif = runtime_sif is not None and runtime_sif != self.sif_path

        if runtime_sif is None:
            if self.def_path.exists():
                if self.verbose:
                    print(f"⚠️  No SIF found, building from def...")
                self.build_container()
                runtime_sif = self.sif_path
            else:
                print("❌ Neither SIF nor base SIF nor def file exists")
                return False

        # Create temporary directory for test files (on host)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_env_")).resolve()

        ok_mat, mat_msg = self._materialize_writable_home_user(runtime_sif)
        if not ok_mat:
            if self.verbose:
                print(f"❌ {mat_msg}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            return False

        if using_base_sif:
            setup_script = self._get_setup_script()
            if setup_script and setup_script.strip():
                ok_delta, delta_msg = self._apply_setup_delta(runtime_sif, setup_script)
                if not ok_delta:
                    if self.verbose:
                        print(f"❌ {delta_msg}")
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    self.temp_dir = None
                    return False

        # Start a long-lived Apptainer instance
        self.instance_name = f"agent_{uuid.uuid4().hex[:8]}"
        start_cmd = [
            "apptainer", "instance", "start",
            *_fakeroot_flags(),
            "--userns",
            "--writable-tmpfs",
            # Suppress Apptainer's default $HOME auto-mount. Under --fakeroot
            # the host user (UID `osey`) is mapped to UID 0 inside the
            # container, so Apptainer's "mount $HOME -> $HOME" default ends
            # up bind-mounting /gpfs/home/osey over /root, hiding whatever
            # the base image stored there. Most adapters did not notice
            # because their images keep nothing important under /root, but
            # R2E Gym (hamishivi/agent-task-r2e-gym) bakes its uv-managed
            # Python at /root/.local/share/uv/python/cpython-3.10.16/...,
            # which /testbed/.venv/bin/python3 follows transitively. Once
            # /root is shadowed, every `pytest` invocation through the venv
            # fails with `bad interpreter: /testbed/.venv/bin/python3:
            # No such file or directory`, the harness's final-state pytest
            # wrapper crashes before reaching the per-task verifier, and the
            # task is recorded as a failure. The harness already explicitly
            # bind-mounts a writable /home/user from the host, so suppressing
            # the implicit $HOME mount is safe across every adapter.
            "--no-home",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--bind", f"{self._writable_home_path}:/home/user",
            "--cleanenv",
            str(runtime_sif),
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
            self._stop_shell()
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
        self.exec("export PYTHONPATH=/home/user/.local/lib/python3/dist-packages:${PYTHONPATH:-}")
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

        command = command.strip()

        # Wrap the command to always emit our marker with the exit code.
        #
        # Three cases:
        #   1. Heredoc (`<<`): can't group with braces, terminator must be on its
        #      own line.  Use newline-then-marker.
        #   2. Backgrounded command (trailing `&`): grouping with `{ cmd & ; }`
        #      is a bash syntax error (`unexpected token ;` after `&`).  We
        #      run it on its own line; the shell prompt returns immediately
        #      after the spawn, so the marker prints right after with
        #      $? = 0 (== "spawn succeeded").  This is the only case where the
        #      reported exit code is the *spawn* status, not the command's
        #      eventual exit -- the alternative (waiting on a daemon to exit)
        #      would deadlock the agent loop, so spawn-success is correct.
        #   3. Everything else: group in `{ ... ; }` so we capture `$?`
        #      correctly across pipelines and compound statements.
        _stripped = command.rstrip()
        _is_background = (
            _stripped.endswith("&")
            and not _stripped.endswith("&&")  # `cmd1 && cmd2` is NOT a background
        )
        if "<<" in command or _is_background:
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
        
        # Handle timeout: kill the running command and recover the shell
        if code is None:
            if self.verbose:
                print(f"⚠️  Command timed out after {timeout or self.read_timeout}s")
            self._recover_shell()
            return False, f"Command timed out. Partial output:\n{raw_out[:500]}"
        
        # Clean output: strip ANSI, strip echoed lines (PTY has no echo, but some programs add it)
        cleaned = ANSI_RE.sub("", raw_out)
        cleaned = cleaned.replace("\r", "")

        success = (code == 0)
        return success, cleaned

    def _write_file_to_container(self, content: str, container_path: str) -> Tuple[bool, str]:
        """Write a file into the container via the bound temp directory.

        Avoids sending large content through the PTY (which can overflow the
        kernel buffer or corrupt shell state via heredocs).

        Note: ``cp`` from the bind mount into ``/home/user`` can fail with EINVAL on
        some Apptainer/overlay stacks. Prefer :meth:`_write_test_on_bind_mount` for
        pytest files (run pytest on the bind-mounted path; no copy to ``/home/user``).
        """
        host_path = self.temp_dir / f"_transfer_{uuid.uuid4().hex[:8]}"
        host_path.write_text(content, encoding="utf-8")
        return self.exec(f"cp {host_path} {container_path}")

    def _write_test_on_bind_mount(self, content: str, basename: str) -> Tuple[bool, str, Optional[Path]]:
        """Write test content under ``temp_dir`` (bind-mounted); same path in container.

        Avoids ``cp … /home/user/…`` which may return EINVAL on some setups.
        """
        if not self.temp_dir:
            return False, "temp_dir not initialized", None
        try:
            root = self.temp_dir.resolve()
            path = (root / basename).resolve()
            path.relative_to(root)
            path.write_text(content, encoding="utf-8")
            return True, "", path
        except (ValueError, OSError) as e:
            return False, str(e), None

    def run_initial_tests(self) -> bool:
        """Run initial state validation tests."""
        if self.verbose:
            print("🧪 Running initial state tests...")

        with open(self.initial_test_path, "r") as f:
            test_file_text = f.read()

        ok, err, test_path = self._write_test_on_bind_mount(test_file_text, "pytest_initial_state.py")
        if not ok or test_path is None:
            if self.verbose:
                print(f"❌ Failed to write test file: {err}")
            return False

        q = shlex.quote(str(test_path))
        rootdir = shlex.quote(str(self.temp_dir))
        test_success, test_output = self.exec(
            f"pytest -q --rootdir={rootdir} --override-ini='collect_ignore_glob=[]' {q}"
        )
        self.exec(f"rm -f {q}")

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

        ok, write_out, test_path = self._write_test_on_bind_mount(test_file_text, "pytest_final_state.py")
        if not ok or test_path is None:
            if self.verbose:
                print(f"❌ Failed to write final test file: {write_out}")
            return False, write_out

        q = shlex.quote(str(test_path))
        rootdir = shlex.quote(str(self.temp_dir))
        test_success, test_output = self.exec(
            f"pytest -q --rootdir={rootdir} --override-ini='collect_ignore_glob=[]' {q}"
        )
        self.exec(f"rm -f {q}")

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
        self._writable_home_path = None
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
