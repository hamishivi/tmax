"""SWERL sandbox environment with the Vanillux solver harness.

Matches the shell and tool-call behavior of ``Vanillux2Agent``.

Key properties (matching the reference solver):

* Single ``bash`` tool — no ``str_replace_editor`` and no separate ``submit``.
  The agent submits by issuing ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
  through ``bash``.
* ``system_template`` and ``instance_template`` come from
  :mod:`open_instruct.environments.vanillux_prompts.yaml`, which is vendored
  from mini-swe-agent v2.2.x (matching the reference solver's
  ``vanillux_prompts.yaml``).
* Tool-result observations use mini-swe-agent's head/tail truncation strategy
  with a ``too_long_hint`` rather than a simple half/half cut, and every
  observation ends with ``(exit_code=N)``.
"""

import asyncio
import contextlib
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml
from huggingface_hub import snapshot_download
from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .apptainer_images import prefer_local_sif
from .backends import SandboxBackend, SandboxLostError, SandboxOOMError, create_backend
from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .swerl_sandbox import LAST_STEP_WARNING, SUBMIT_MARKER, TIMING_LOG_THRESHOLD_S, TIMING_LOGS

logger = logger_utils.setup_logger(__name__)

VANILLUX_CALL_LIMIT = int(os.getenv("VANILLUX_CALL_LIMIT", "100"))

_BASH_WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
_BASH_WRAPPER_PATH_QUOTED = shlex.quote(_BASH_WRAPPER_PATH)
_BASH_CWD_PATH = "/tmp/.swerl_vanillux_cwd"
_BASH_ENV_PATH = "/tmp/.swerl_vanillux_env"

_SETUP_CWD_PATH = "/tmp/.swerl_setup_cwd"
_SETUP_ENV_PATH = "/tmp/.swerl_setup_env"


class TaskSetupError(RuntimeError):
    """Dataset-provided sandbox setup failed."""


_BASH_WRAPPER = f"""#!/bin/bash
# Default address-space limit: 2 GiB, configurable in KiB.
ulimit -v "${{SWERL_SANDBOX_ULIMIT_AS_KB:-2097152}}" 2>/dev/null || true
_stdout="$(mktemp /tmp/.swerl_vanillux_stdout.XXXXXX)"
_stderr="$(mktemp /tmp/.swerl_vanillux_stderr.XXXXXX)"
_command_pid=""
_collect_output() {{
    cat "$_stdout"
    cat "$_stderr" >&2
    rm -f "$_stdout" "$_stderr"
}}
_terminate_command() {{
    trap '' HUP INT TERM
    if [ -n "$_command_pid" ]; then
        # Cancel the detached command group, including children that ignore TERM.
        kill -TERM -- "-$_command_pid" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$_command_pid" 2>/dev/null || true
        wait "$_command_pid" 2>/dev/null || true
    fi
    _collect_output
    exit "$1"
}}
trap '_terminate_command 129' HUP
trap '_terminate_command 130' INT
trap '_terminate_command 143' TERM
# Files prevent background services from holding Apptainer's output pipes open.
# Restore, execute, and save in the same shell, as in Vanillux2Agent._wrap_command.
(
# Background shells inherit ignored SIGINT/SIGQUIT; reset them before exec.
trap - HUP INT QUIT TERM
exec setsid bash -c 'cd "$(cat {shlex.quote(_BASH_CWD_PATH)})" 2>/dev/null || true
. {shlex.quote(_BASH_ENV_PATH)} 2>/dev/null || true
'"$1"'
_vanillux2_ec=$?
pwd > {shlex.quote(_BASH_CWD_PATH)}
(unset FAKEROOTKEY; export -p) > {shlex.quote(_BASH_ENV_PATH)}
exit $_vanillux2_ec'
) >"$_stdout" 2>"$_stderr" </dev/null &
_command_pid=$!
wait "$_command_pid"
_exit_code=$?
# Record ordinary exit 124; cancellation exits without writing this marker.
if [ "$_exit_code" -eq 124 ] && [ -n "${{2:-}}" ]; then
    printf '%s\\n' "$_exit_code" > "$2"
fi
trap - HUP INT TERM
_collect_output
exit $_exit_code
"""

_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}


with (Path(__file__).resolve().parent / "vanillux_prompts.yaml").open(encoding="utf-8") as _fh:
    _VANILLUX_PROMPTS: dict[str, Any] = yaml.safe_load(_fh)
INSTANCE_TEMPLATE: str = _VANILLUX_PROMPTS["instance_template"]
FORMAT_ERROR_TEMPLATE: str = _VANILLUX_PROMPTS["format_error_template"]
_OBS_MAX_CHARS: int = int(_VANILLUX_PROMPTS["observation"].get("max_chars", 10000))
_OBS_HEAD_CHARS: int = int(_VANILLUX_PROMPTS["observation"].get("head_chars", 5000))
_OBS_TAIL_CHARS: int = int(_VANILLUX_PROMPTS["observation"].get("tail_chars", 5000))
_OBS_TOO_LONG_HINT: str = _VANILLUX_PROMPTS["observation"].get("too_long_hint", "Output truncated.")
TOOL_CALL_FORMAT_ERROR_MESSAGE = (
    "Format error: Your last response did not include a valid `bash` tool call.\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    "end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "via the `bash` tool, with no other content in the command.\n"
)

_COMPOSE_PROVIDER_RE = re.compile(
    r"\x1b\[4m>>>> Executing external compose provider "
    r'"[^"]*docker-compose"\. Please see podman-compose\(1\) for how to disable '
    r"this message\. <<<<\n\n\x1b\[0m"
)
_DOCKER_EXEC_ERROR_RE = re.compile(
    r"(?ms)^Error: executing [^\n]*(?:docker-compose|docker compose)" r".*?: exit status \d+\s*$"
)


def render_instance(task: str) -> str:
    """Render the vanillux instance template with the task description.

    The vendored template uses ``{{task}}`` as the only Jinja variable; we
    substitute literally rather than pulling in Jinja2 to keep deps light.
    """
    return INSTANCE_TEMPLATE.replace("{{task}}", task)


def truncate_observation(output: str) -> str:
    """Apply mini-swe-agent's head/tail truncation to long tool outputs."""
    if len(output) <= _OBS_MAX_CHARS:
        return output
    elided = len(output) - _OBS_HEAD_CHARS - _OBS_TAIL_CHARS
    return (
        f"{_OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({_OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:_OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({_OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-_OBS_TAIL_CHARS:]}"
    )


def format_error_message(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


class SWERLVanilluxSandboxEnv(RLEnvironment):
    """SWERL sandbox with the Vanillux (mini-swe-agent style) bash-only harness."""

    config_name = "swerl_vanillux_sandbox"
    _tool_definitions = (_BASH_TOOL,)
    _RESET_RETRY_BASE_DELAY_S = 1.0
    _RESET_RETRY_MAX_DELAY_S = 16.0
    _RESET_RETRY_JITTER_S = 2.0
    _MIN_TEST_TIMEOUT_S = 600
    # Bounded allowance for the transport and cleanup that share the submission step with
    # the verifier: tests upload, chmod, existence probe, exit-status read-back, reward
    # parse. Fixed rather than proportional -- it is the same work whether the verifier
    # asked for 60s or 3600s.
    _GRADING_OVERHEAD_S = 120.0

    def __init__(
        self,
        backend: str = "docker",
        image: str = "python:3.12-slim",
        task_data_dir: str = "",
        task_data_hf_repo: str = "",
        test_timeout: int = 600,
        timeout: int = 120,
        last_step_warning: bool = False,
        append_turns_remaining: bool = False,
        pwd: str = "",
        pwd_map: str = "",
        verifier_timeout_map: str = "",
        max_verifier_timeout_s: float = 0.0,
        **backend_kwargs: Any,
    ):
        backend_kwargs["image"] = image
        self._backend_type = backend
        # ApptainerBackend defaults pwd="/workspace", which does not exist in the
        # swerl-tmax task images: every exec dies with "failed to set working
        # directory: chdir /workspace" before the command that would create it can
        # run. The knob was dropped from the config when the vmvm backend was removed
        # (61b5a85d), leaving no way to set it from tool_configs. Only forward it to
        # backends that accept it -- DockerBackend has no pwd parameter.
        if pwd:
            if backend in ("apptainer", "sandfleet"):
                backend_kwargs["pwd"] = pwd
            else:
                logger.warning("Ignoring pwd=%r: the %r backend does not accept it.", pwd, backend)
        # A single pwd is only correct for a corpus whose images share one working
        # directory. swerl-tmax does; LiteCoder and CalibForge do not -- their images
        # declare /app, /root, /app/repo, /builds and others, and 601/602 LiteCoder
        # verifiers abort with "No working directory set" before running a test if they
        # are started from the wrong one. The failure is silent: no reward artifact is
        # produced, so it is indistinguishable from a dead image.
        #
        # SWERL_SANDBOX_PWD_MAP points at {task_id: pwd} and is applied per reset. The
        # scalar pwd above stays the default for tasks absent from the map, so existing
        # runs are unaffected.
        self._pwd_map: dict[str, str] = {}
        _map_path = pwd_map or os.getenv("SWERL_SANDBOX_PWD_MAP", "")
        if _map_path:
            try:
                with open(_map_path) as _fh:
                    self._pwd_map = json.load(_fh)
                logger.info("Loaded per-task pwd map: %d entries from %s", len(self._pwd_map), _map_path)
            except Exception as exc:  # a missing map must be loud, not a silent fallback
                raise RuntimeError(f"SWERL_SANDBOX_PWD_MAP unreadable: {_map_path}: {exc}") from exc
        self._timeout = timeout
        backend_kwargs.pop("penalty", None)
        backend_kwargs.pop("call_name", None)
        self._backend_kwargs = backend_kwargs
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._task_id: str | None = None
        self._task_data_dir = task_data_dir
        self._has_task_setup = False
        self._task_data_hf_repo = task_data_hf_repo
        self._test_timeout = max(test_timeout, self._MIN_TEST_TIMEOUT_S)
        # A supplied map uses each task's declared verifier budget without the
        # scalar floor. Agent-command and image-build timeouts are independent.
        # Validate the complete map before selecting tasks.
        self._verifier_timeout_map: dict[str, float] = {}
        _vt_path = verifier_timeout_map or os.getenv("SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP", "")
        if _vt_path:
            try:
                with open(_vt_path) as _fh:
                    _raw = json.load(_fh)
            except Exception as exc:  # loud, never a silent fallback -- as for pwd_map
                raise RuntimeError(f"SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP unreadable: {_vt_path}: {exc}") from exc
            # Validate the WHOLE map at construction. A bad entry discovered lazily at
            # reset would surface hours into a run, on one task, as a crashed actor.
            for _tid, _v in _raw.items():
                try:
                    _f = float(_v)
                except (TypeError, ValueError):
                    raise RuntimeError(
                        f"verifier timeout map {_vt_path}: task {_tid!r} has non-numeric budget {_v!r}"
                    ) from None
                if not math.isfinite(_f) or _f <= 0:
                    raise RuntimeError(
                        f"verifier timeout map {_vt_path}: task {_tid!r} has non-positive or non-finite budget {_v!r}"
                    )
                self._verifier_timeout_map[_tid] = _f
            logger.info(
                "Loaded per-task verifier timeout map: %d entries from %s (range %g-%gs)",
                len(self._verifier_timeout_map),
                _vt_path,
                min(self._verifier_timeout_map.values()),
                max(self._verifier_timeout_map.values()),
            )
        # A ceiling on how long ONE task may hold an actor. 0 disables it. This is a
        # capacity decision, not a grading one: a 12000s CalibForge verifier would occupy
        # a rollout actor for over three hours. Clamping is logged and recorded per
        # episode -- an unannounced clamp would be the same silent truncation this whole
        # change exists to remove, just with a different number.
        self._max_verifier_timeout_s = float(
            max_verifier_timeout_s or os.getenv("SWERL_SANDBOX_MAX_VERIFIER_TIMEOUT_S", "") or 0.0
        )
        if self._max_verifier_timeout_s < 0:
            raise RuntimeError(f"max_verifier_timeout_s must be >= 0, got {max_verifier_timeout_s!r}")
        if self._max_verifier_timeout_s and self._verifier_timeout_map:
            _over = {t: v for t, v in self._verifier_timeout_map.items() if v > self._max_verifier_timeout_s}
            if _over:
                logger.warning(
                    "%d task(s) declare a verifier budget above the %gs ceiling and will be "
                    "clamped (largest %gs). Their verifiers may be cut short.",
                    len(_over),
                    self._max_verifier_timeout_s,
                    max(_over.values()),
                )
        # Resolved per reset; None means "use the scalar self._test_timeout".
        self._task_test_timeout: float | None = None
        self._last_step_warning = last_step_warning
        self._append_turns_remaining = append_turns_remaining
        self._max_steps: int | None = None
        self._instruction = ""
        self._tests_dir: str | None = None

    @staticmethod
    def resolve_task_data_dir(task_data_hf_repo: str) -> str:
        """Download and extract task data once per machine."""
        repo_dir = snapshot_download(task_data_hf_repo, repo_type="dataset")
        tarball = os.path.join(repo_dir, "task-data.tar.gz")
        if os.path.isfile(tarball):
            extract_dir = tarball + ".extracted"
            complete_file = os.path.join(extract_dir, ".extraction_complete")
            lock_dir = extract_dir + ".lock"
            while not os.path.isfile(complete_file):
                try:
                    os.mkdir(lock_dir)
                except FileExistsError:
                    time.sleep(1)
                    continue
                try:
                    if os.path.isfile(complete_file):
                        break
                    logger.info(f"Extracting {tarball} to {extract_dir}...")
                    if os.path.isdir(extract_dir):
                        shutil.rmtree(extract_dir)
                    os.makedirs(extract_dir, exist_ok=True)
                    subprocess.run(["tar", "-xzf", tarball, "-C", extract_dir], check=True)
                    with open(complete_file, "w", encoding="utf-8") as f:
                        f.write("ok\n")
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.rmdir(lock_dir)
            return extract_dir
        return repo_dir

    async def setup(self) -> None:
        """Download task data from HuggingFace if configured."""
        if self._task_data_hf_repo and (not self._task_data_dir or not os.path.isdir(self._task_data_dir)):
            if self._task_data_dir:
                logger.info(
                    f"Task data directory {self._task_data_dir} is not available on this actor; "
                    f"downloading from {self._task_data_hf_repo}..."
                )
            else:
                logger.info(f"Downloading task data from {self._task_data_hf_repo}...")
            self._task_data_dir = self.resolve_task_data_dir(self._task_data_hf_repo)
            logger.info(f"Task data at {self._task_data_dir}")

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    def get_tool_call_format_error_message(self) -> str:
        return TOOL_CALL_FORMAT_ERROR_MESSAGE

    def get_grading_deadline(self) -> float | None:
        """How long the caller must wait for a submission step on THIS task.

        The verifier's own budget plus a bounded allowance for the work that surrounds it
        inside the same step: uploading the tests directory, chmod, the existence probe,
        reading back the exit-status file and parsing the reward. That surrounding work is
        transport and cleanup, not grading, so it gets a fixed allowance rather than a
        share of the task's budget.

        Only meaningful once a task is selected. Before reset the scalar is the honest
        answer, and it is what the caller would have used anyway.
        """
        return self._effective_test_timeout() + self._GRADING_OVERHEAD_S

    def get_submit_markers(self) -> tuple[str, ...]:
        return (SUBMIT_MARKER,)

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        last_error = None
        max_attempts = 5 if self._backend_type == "docker" else 3
        for attempt in range(max_attempts):
            try:
                return self._do_reset(task_id, **kwargs)
            except Exception as e:
                last_error = e
                if self._backend is not None:
                    with contextlib.suppress(Exception):
                        self._backend.close()
                    self._backend = None
                if attempt + 1 == max_attempts:
                    logger.warning(
                        f"SWERLVanilluxSandboxEnv.reset attempt {attempt + 1} failed: {e}. No attempts remain."
                    )
                    break
                delay = self._reset_retry_delay(attempt + 1)
                logger.warning(
                    "SWERLVanilluxSandboxEnv.reset attempt %s failed: %s. Retrying in %.2fs...", attempt + 1, e, delay
                )
                await asyncio.sleep(delay)
        if last_error is not None:
            raise RuntimeError(f"Reset failed after {max_attempts} attempts: {last_error}") from last_error
        raise RuntimeError("Reset failed without capturing an error.")

    @classmethod
    def _reset_retry_delay(cls, attempt: int) -> float:
        backoff = min(cls._RESET_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), cls._RESET_RETRY_MAX_DELAY_S)
        return backoff + random.uniform(0.0, cls._RESET_RETRY_JITTER_S)

    def _do_reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        reset_start_time = time.perf_counter()
        phase_start_time = reset_start_time
        timings: dict[str, float] = {}

        def record_phase(name: str) -> None:
            nonlocal phase_start_time
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + (now - phase_start_time)
            phase_start_time = now

        self._has_task_setup = False
        self._tests_dir = None
        setup_command = kwargs.get("setup_command")
        if setup_command is not None and not isinstance(setup_command, str):
            raise ValueError("env_config.setup_command must be a string or null")
        resolved_image = kwargs.get("image")
        if not resolved_image and self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
            image_file = os.path.join(task_dir, "image.txt")
            if os.path.isfile(image_file):
                with open(image_file, encoding="utf-8") as f:
                    image_tag = f.read().strip()
                if image_tag:
                    resolved_image = image_tag
        if not resolved_image:
            instance_details = {
                "task_id": task_id,
                "backend_type": self._backend_type,
                "env_name": kwargs.get("env_name"),
                "instance_name": kwargs.get("instance_name"),
                "instance_id": kwargs.get("instance_id"),
                "episode_id": kwargs.get("episode_id"),
            }
            logger.error(
                "Missing explicit image for SWERLVanilluxSandboxEnv reset: %s",
                {k: v for k, v in instance_details.items() if v is not None},
            )
            raise ValueError(
                "SWERLVanilluxSandboxEnv requires an explicit image per task. "
                "Set env_config.image or provide image.txt in task data."
            )
        if self._backend_type in ("apptainer", "sandfleet"):
            # Both run apptainer on the sandbox host; a plain docker ref forces a
            # per-lease pull + OCI->SIF conversion there (exit=255 under
            # restricted egress). Resolve to the prebuilt local pool when
            # configured.
            resolved_image = prefer_local_sif(resolved_image)
        self._backend_kwargs["image"] = resolved_image
        if self._backend_type == "docker" and kwargs.get("docker_host"):
            self._backend_kwargs["docker_host"] = kwargs["docker_host"]
        # Resolve cwd for the selected image. A supplied map must cover this
        # task; otherwise fail reset rather than grade in an unintended cwd.
        task_pwd = None
        if self._pwd_map:
            task_pwd = self._pwd_map.get(task_id or "")
            if not task_pwd:
                raise RuntimeError(
                    f"task_id {task_id!r} is absent from the supplied pwd map "
                    f"({len(self._pwd_map)} entries). Refusing to fall back to pwd="
                    f"{self._backend_kwargs.get('pwd')!r}: an incomplete map silently "
                    f"grades tasks from the wrong working directory."
                )
            self._backend_kwargs["pwd"] = task_pwd

        # Same rule as the pwd map, for the same reason: an INCOMPLETE map must not fall
        # through to the scalar. Falling through would silently reinstate the 600s cap for
        # exactly the tasks whose declaration went missing -- the failure the map exists to
        # remove -- and it would be invisible, because a truncated verifier scores 0 rather
        # than raising. Tasks with no declaration get their fallback recorded explicitly in
        # the map at build time, so "absent" here always means a staging defect.
        self._task_test_timeout = None
        if self._verifier_timeout_map:
            _budget = self._verifier_timeout_map.get(task_id or "")
            if _budget is None:
                raise RuntimeError(
                    f"task_id {task_id!r} is absent from the supplied verifier timeout map "
                    f"({len(self._verifier_timeout_map)} entries). Refusing to fall back to "
                    f"{self._test_timeout}s: an incomplete map silently truncates verifiers "
                    f"to a budget they did not declare, which scores 0 without erroring."
                )
            self._task_test_timeout = _budget
        record_phase("resolve_image")

        if (
            self._backend is not None
            and self._backend_type == "docker"
            and getattr(self._backend, "_docker_host", None) != self._backend_kwargs.get("docker_host")
        ):
            self._backend.close()
            record_phase("close")
            self._backend = None

        # A pooled backend is reused across tasks and its pwd was fixed at construction,
        # so a changed pwd must force a rebuild. Without this the second task to land on
        # a given actor silently runs in the previous task's directory.
        if self._backend is not None and task_pwd and getattr(self._backend, "_pwd", None) != task_pwd:
            self._backend.close()
            record_phase("close_pwd_change")
            self._backend = None

        if self._backend is not None:
            self._backend._image = self._backend_kwargs.get("image", self._backend._image)
            self._backend.restart()
            record_phase("restart")
        else:
            bkwargs = dict(self._backend_kwargs)
            bkwargs.setdefault("timeout", self._timeout)
            self._backend = create_backend(self._backend_type, **bkwargs)
            record_phase("create_backend")
            self._backend.start()
            record_phase("start")

        self._step_count = 0
        self._task_id = task_id
        self._max_steps = kwargs.get("max_steps", VANILLUX_CALL_LIMIT)
        mkdir_result = self._backend.run_command("mkdir -p /workspace /output /logs/verifier")
        # fail the reset loudly when directory
        # initialization fails instead of nominally succeeding on a torn sandbox.
        if mkdir_result.exit_code != 0:
            raise RuntimeError(
                "Failed to initialize Vanillux sandbox directories "
                f"(exit={mkdir_result.exit_code}): {mkdir_result.stderr or mkdir_result.stdout}"
            )
        record_phase("mkdir")

        self._instruction = ""
        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
        if task_dir and os.path.isdir(task_dir):
            self._load_task_data(task_dir)
        record_phase("load_task_data")

        self._prepare_vanillux_runtime()
        record_phase("prepare_vanillux")

        self._apply_task_setup(setup_command)
        record_phase("task_setup")

        reset_total_s = time.perf_counter() - reset_start_time
        if TIMING_LOGS and reset_total_s >= TIMING_LOG_THRESHOLD_S:
            logger.info(
                "SWERLVanilluxSandboxEnv.reset timing task_id=%s image=%s total=%.3fs phases=%s",
                task_id,
                resolved_image,
                reset_total_s,
                {key: round(value, 3) for key, value in timings.items()},
            )

        if self._instruction:
            observation = render_instance(self._instruction)
        elif task_id:
            observation = render_instance(f"[Task: {task_id}] Sandbox ready.")
        else:
            observation = render_instance("Sandbox ready.")

        return (
            StepResult(result=observation, metadata={"task_id": task_id, "backend": self._backend_type}),
            list(self._tool_definitions),
        )

    def _load_task_data(self, task_dir: str) -> None:
        """Load instruction and seed files into the container. Tests are deferred to submit."""
        assert self._backend is not None

        instruction_file = os.path.join(task_dir, "instruction.md")
        if os.path.isfile(instruction_file):
            with open(instruction_file, encoding="utf-8") as f:
                self._instruction = f.read().strip()

        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            self._upload_directory(seeds_dir, "/workspace")

        self._tests_dir = os.path.join(task_dir, "tests") if os.path.isdir(os.path.join(task_dir, "tests")) else None

    def _upload_directory(self, host_dir: str, container_dir: str) -> None:
        assert self._backend is not None
        prefix = container_dir.lstrip("/")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            dir_info = tarfile.TarInfo(name=prefix)
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            dir_info.uid = 0
            dir_info.gid = 0
            dir_info.uname = "root"
            dir_info.gname = "root"
            tar.addfile(dir_info)

            for root, dirs, files in os.walk(host_dir):
                rel_root = os.path.relpath(root, host_dir)
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root != "." else d
                    info = tarfile.TarInfo(name=f"{prefix}/{rel_dir}")
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    tar.addfile(info)
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_file = os.path.join(rel_root, fname) if rel_root != "." else fname
                    # force deterministic root
                    # ownership on every member; tar.add() preserved uploader
                    # uid/gid, drifting file ownership inside task containers.
                    info = tar.gettarinfo(src_path, arcname=f"{prefix}/{rel_file}")
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    with open(src_path, "rb") as f:
                        tar.addfile(info, f)

        self._backend.put_archive("/", tar_stream.getvalue())

    def _apply_task_setup(self, command: str | None) -> None:
        """Run optional dataset setup in the agent shell, bounded by its timeout.

        Snapshot setup state for verification; later agent exports/cd must not
        change the verifier's starting environment. Tests remain hidden until submit.
        """
        if not command or not command.strip():
            return
        assert self._backend is not None
        started = time.perf_counter()
        result = self._backend.run_command(
            f"bash {_BASH_WRAPPER_PATH_QUOTED} {shlex.quote(command)}", timeout=self._timeout
        )
        if result.exit_code != 0:
            raise TaskSetupError(
                f"Task setup failed for {self._task_id!r} (exit={result.exit_code}, "
                f"elapsed={time.perf_counter() - started:.2f}s, timeout={self._timeout}s): "
                f"{(result.stderr or result.stdout or '')[-2000:]}"
            )
        snapshot = self._backend.run_command(
            f"cp {shlex.quote(_BASH_CWD_PATH)} {shlex.quote(_SETUP_CWD_PATH)} && "
            f"cp {shlex.quote(_BASH_ENV_PATH)} {shlex.quote(_SETUP_ENV_PATH)}"
        )
        if snapshot.exit_code != 0:
            raise TaskSetupError(f"Could not save task setup state: {snapshot.stderr or snapshot.stdout}")
        self._has_task_setup = True
        logger.info("Task setup completed: task_id=%s elapsed=%.3fs", self._task_id, time.perf_counter() - started)

    def _prepare_vanillux_runtime(self) -> None:
        assert self._backend is not None
        init_result = self._backend.run_command(
            "mkdir -p /workspace /root && "
            # Capture backend state before setup changes cwd.
            f"pwd > {shlex.quote(_BASH_CWD_PATH)} && "
            f"(unset FAKEROOTKEY; export -p) > {shlex.quote(_BASH_ENV_PATH)} && "
            "cd /workspace && "
            '{ [ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; }; }'
        )
        # a reset must not report success over an
        # unusable sandbox — the false-success class behind the poison-pill incident.
        if init_result.exit_code != 0:
            raise RuntimeError(
                "Failed to initialize Vanillux runtime directories "
                f"(exit={init_result.exit_code}): {init_result.stderr or init_result.stdout}"
            )
        self._backend.write_file(_BASH_WRAPPER_PATH, _BASH_WRAPPER)
        chmod_result = self._backend.run_command(f"chmod +x {_BASH_WRAPPER_PATH_QUOTED}")
        if chmod_result.exit_code != 0:
            raise RuntimeError(
                "Failed to install Vanillux bash wrapper "
                f"(exit={chmod_result.exit_code}): {chmod_result.stderr or chmod_result.stdout}"
            )

    async def step(self, call: EnvCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name == "bash":
            try:
                return self._with_last_step_warning(self._execute_bash(call.args))
            except SandboxOOMError as e:
                logger.warning(f"[{self._task_id}] sandbox OOM: {e}")
                return StepResult(
                    result=("Sandbox container was killed by the OOM reaper. Ending episode with reward 0."),
                    reward=0.0,
                    done=True,
                    metadata={"oom_killed": True, "task_id": self._task_id},
                )
            except SandboxLostError as e:
                logger.warning(f"[{self._task_id}] sandbox worker lost: {e}")
                with contextlib.suppress(Exception):
                    self._backend.close()
                self._backend = None
                return StepResult(
                    result="Sandbox worker was lost. Ending this rollout so it can be resampled.",
                    reward=0.0,
                    done=True,
                    metadata={
                        "sandbox_lost": True,
                        "infrastructure_failure": True,
                        "error": str(e),
                        "task_id": self._task_id,
                    },
                )
            except RuntimeError as e:
                # Backend loss ends the episode but does not establish a timeout,
                # including when it prevents checking an exit-124 completion marker.
                if "Instance not started" not in str(e):
                    raise
                logger.warning(f"[{self._task_id}] sandbox backend unavailable: {e}")
                return StepResult(
                    result="Sandbox backend is no longer running. Ending episode with reward 0.",
                    reward=0.0,
                    done=True,
                    metadata={
                        "backend_unavailable": True,
                        "infrastructure_failure": True,
                        "error": str(e),
                        "task_id": self._task_id,
                    },
                )
        return self._with_last_step_warning(
            StepResult(result=format_error_message(f"Unknown tool '{call.name}'. The only available tool is `bash`."))
        )

    def _with_last_step_warning(self, result: StepResult) -> StepResult:
        if (
            self._last_step_warning
            and not result.done
            and self._max_steps is not None
            and self._step_count == self._max_steps - 1
        ):
            result.result = f"{result.result}\n\n{LAST_STEP_WARNING}"
        return result

    def _turns_remaining_message(self) -> str | None:
        if not self._append_turns_remaining or self._max_steps is None:
            return None
        turns_remaining = max(self._max_steps - self._step_count, 0)
        if turns_remaining == 1:
            return "One turn remaining. Please submit your work"
        return f"Turns remaining: {turns_remaining}"

    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "") if isinstance(args, dict) else None
        if not isinstance(command, str):
            return StepResult(result=TOOL_CALL_FORMAT_ERROR_MESSAGE, metadata={"exit_code": 1})

        command = command.strip()
        exit_status_path = f"/tmp/.swerl_vanillux_exit.{uuid.uuid4().hex}"
        result = self._backend.run_command(
            f"bash {_BASH_WRAPPER_PATH_QUOTED} {shlex.quote(command)} {shlex.quote(exit_status_path)}"
        )
        stderr = result.stderr
        timed_out = False
        # Explicit backend flag wins over the exit code; a SIGKILLed/OOM
        # child (137) must never be misread as a deadline.
        if result.timed_out is not None:
            timed_out = result.timed_out
        elif result.exit_code == 124:
            # An ordinary exit 124 writes a marker; a backend deadline does not.
            quoted_status_path = shlex.quote(exit_status_path)
            completion = self._backend.run_command(
                f"if [ -f {quoted_status_path} ]; then cat {quoted_status_path}; rm -f {quoted_status_path}; fi"
            )
            if completion.exit_code != 0:
                error = f"Could not inspect command completion status (probe exit_code={completion.exit_code})."
                return StepResult(
                    result=error,
                    reward=0.0,
                    done=True,
                    metadata={"infrastructure_failure": True, "error": error, "task_id": self._task_id},
                )
            timed_out = completion.stdout.strip() != "124"
            # Backends label all 124s as timeouts, including ordinary exits.
            if not timed_out and stderr:
                stderr = stderr.removeprefix(f"Command timed out after {self._timeout}s.\n")

        output = result.stdout or ""
        if stderr:
            output += f"\n{stderr}" if output else stderr
        output = _COMPOSE_PROVIDER_RE.sub("", output)
        output = _DOCKER_EXEC_ERROR_RE.sub("", output).rstrip()
        truncated = truncate_observation(output) if output else "(no output)"
        observation = f"{truncated}\n\n(exit_code={result.exit_code})"
        if timed_out:
            return StepResult(
                result=observation,
                reward=0.0,
                done=True,
                metadata={"exit_code": result.exit_code, "timeout": True, "task_id": self._task_id},
            )
        if SUBMIT_MARKER in command or SUBMIT_MARKER in observation:
            return self._run_tests()
        turns_remaining_message = self._turns_remaining_message()
        if turns_remaining_message is not None:
            observation = f"{observation}\n{turns_remaining_message}"
        return StepResult(result=observation, metadata={"exit_code": result.exit_code})

    def _effective_test_timeout(self) -> float:
        """Use the declared per-task budget or the scalar default and optional ceiling."""
        budget = self._test_timeout if self._task_test_timeout is None else self._task_test_timeout
        if self._max_verifier_timeout_s:
            budget = min(budget, self._max_verifier_timeout_s)
        return budget

    def _verifier_budget_was_clamped(self) -> bool:
        return bool(
            self._max_verifier_timeout_s
            and self._task_test_timeout is not None
            and self._task_test_timeout > self._max_verifier_timeout_s
        )

    def _tests_destination(self) -> str:
        """Where THIS task's verifier expects to find the uploaded tests.

        Default /tests, which is what every absolute-path grader requires. Only a verifier
        that actually references the relative `../tests/` gets the sibling-of-pwd location
        it implies -- so a grader using absolute /tests is never moved, whatever its
        working directory.
        """
        default = "/tests"
        if not self._tests_dir:
            return default
        script = os.path.join(self._tests_dir, "test.sh")
        try:
            with open(script, errors="replace") as fh:
                body = fh.read()
        except OSError:
            return default
        if "../tests/" not in body:
            return default
        pwd = (self._backend_kwargs or {}).get("pwd") or "/"
        parent = os.path.dirname(pwd.rstrip("/")) or "/"
        dest = os.path.join(parent, "tests")
        if dest != default:
            logger.info("[%s] verifier uses ../tests/ with pwd=%s; uploading tests to %s", self._task_id, pwd, dest)
        return dest

    def _run_tests(self) -> StepResult:
        """Grade, tagging every outcome as a grading step.

        The tag is what lets the rollout loop tell a verifier apart from an ordinary bash
        call after the fact -- including the case where the caller, not the verifier,
        ended it. Without it an outer cancellation is indistinguishable from an agent
        command that ran long, and the two need opposite fixes.
        """
        result = self._grade()
        meta = dict(result.metadata or {})
        meta.update(
            grading=True,
            verifier_timeout_s=self._effective_test_timeout(),
            verifier_timeout_declared=self._task_test_timeout is not None,
            verifier_timeout_clamped=self._verifier_budget_was_clamped(),
        )
        result.metadata = meta
        return result

    def _grade(self) -> StepResult:
        """Upload tests (deferred from reset to prevent peeking), then run."""
        assert self._backend is not None

        if self._tests_dir is None:
            raise RuntimeError(f"No test data directory for task {self._task_id}")

        # Absolute-path graders keep /tests. Relative ../tests references place
        # tests beside the task cwd, independent of its nesting depth.
        dest = self._tests_destination()
        check_stdout = ""
        for attempt in range(2):
            try:
                self._upload_directory(self._tests_dir, dest)
            except Exception as e:
                logger.warning(f"[{self._task_id}] tests upload attempt {attempt + 1} raised: {e}")
                if attempt == 0:
                    continue
                raise
            self._backend.run_command(f"chmod +x {shlex.quote(dest)}/test.sh 2>/dev/null || true")
            check = self._backend.run_command(f"test -f {shlex.quote(dest)}/test.sh && echo EXISTS")
            check_stdout = check.stdout.strip()
            if check_stdout == "EXISTS":
                break
            logger.warning(f"[{self._task_id}] {dest}/test.sh missing after upload attempt {attempt + 1}; retrying")
        if check_stdout != "EXISTS":
            ls = self._backend.run_command(f"ls -la {shlex.quote(dest)} 2>&1 || true")
            raise RuntimeError(
                f"No test.sh found in test data for task {self._task_id}. {dest} listing: {ls.stdout!r}"
            )

        status_path = f"/tmp/.vanillux_verifier_exit.{uuid.uuid4().hex}"
        verifier_command = (
            f"bash {shlex.quote(dest)}/test.sh; status=$?; printf '%s' \"$status\" "
            f'> {shlex.quote(status_path)}; exit "$status"'
        )
        if self._has_task_setup:
            verifier_command = (
                f'. {shlex.quote(_SETUP_ENV_PATH)} && cd "$(cat {shlex.quote(_SETUP_CWD_PATH)})" && {verifier_command}'
            )
        verifier_budget = self._effective_test_timeout()
        result = self._backend.run_command(verifier_command, timeout=verifier_budget)
        completion = self._backend.run_command(
            f"if [ -f {shlex.quote(status_path)} ]; then cat {shlex.quote(status_path)}; "
            f"rm -f {shlex.quote(status_path)}; fi"
        )
        if completion.exit_code != 0:
            raise RuntimeError("Could not inspect verifier completion status")
        reward = self._parse_reward()
        verifier_timed_out = (
            result.timed_out
            if result.timed_out is not None
            else (result.exit_code == 124 and not completion.stdout.strip())
        )
        if verifier_timed_out:
            # This path had no log line at all, so "how often do verifiers time out?"
            # could only be answered by grepping for a string that is never written --
            # an absence that reads as zero. Anything asked of these runs' grading needs
            # this to be countable.
            logger.warning(
                "[%s] verifier terminated at its %gs budget (%s)",
                self._task_id,
                verifier_budget,
                "declared" if self._task_test_timeout is not None else "corpus scalar",
            )
            return StepResult(
                result="Verifier exceeded its time budget.",
                reward=0.0 if reward is None else reward,
                done=True,
                # Record the budget that was actually enforced. Without it a terminated
                # verifier is indistinguishable from one the scalar cap truncated, which
                # is precisely the question being asked of these runs.
                metadata={
                    "timeout": True,
                    "exit_code": 124,
                    "task_id": self._task_id,
                    "verifier_timeout_s": verifier_budget,
                    "verifier_timeout_declared": self._task_test_timeout is not None,
                },
            )
        if reward is None:
            return StepResult(
                result="Verifier did not produce a finite reward in [0, 1].",
                reward=0.0,
                done=True,
                metadata={
                    "invalid_reward": True,
                    "error": "Missing or invalid verifier reward",
                    "task_id": self._task_id,
                },
            )

        stdout = truncate_observation(result.stdout) if result.stdout else ""
        stderr = truncate_observation(result.stderr) if result.stderr else ""
        observation = (
            f"Test execution complete.\n"
            f"Exit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}\n"
            f"Reward: {reward}"
        )

        return StepResult(result=observation, reward=reward, done=True)

    def _parse_reward(self) -> float | None:
        """Return a valid score, or None when there is no usable measurement."""
        assert self._backend is not None

        try:
            reward_result = self._backend.run_command("cat /logs/verifier/reward.txt")
            if reward_result.exit_code == 0 and reward_result.stdout.strip():
                reward = float(reward_result.stdout.strip())
                return reward if math.isfinite(reward) and 0.0 <= reward <= 1.0 else None
        except (ValueError, TypeError):
            pass

        return None

    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    def state(self) -> State:
        return State(episode_id=self._task_id, step_count=self._step_count)

    async def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    async def shutdown(self) -> None:
        await self.close()


@dataclass
class SWERLVanilluxSandboxEnvConfig(BaseEnvConfig):
    """Configuration for SWERLVanilluxSandboxEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = SWERLVanilluxSandboxEnv
    backend: str = "docker"
    image: str = "python:3.12-slim"
    mem_limit: str = "4g"
    penalty: float = -0.05
    task_data_dir: str = ""
    task_data_hf_repo: str = ""
    test_timeout: int = 600
    timeout: int = 120
    last_step_warning: bool = False
    append_turns_remaining: bool = False
    # Working directory inside the sandbox. Empty keeps each backend's own default;
    # set "/" for the swerl-tmax images, which have no /workspace.
    pwd: str = ""
    # {task_id: seconds} from each task's [verifier] timeout_sec. Empty keeps the single
    # `test_timeout` above (with its 600s floor) for every task, which is what the three
    # TMAX arms use -- their corpora declare no per-task budgets.
    verifier_timeout_map: str = ""
