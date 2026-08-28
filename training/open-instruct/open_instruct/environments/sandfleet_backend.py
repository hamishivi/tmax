"""Remote sandbox backend for the Sandfleet Slurm service."""

from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from docker import errors as docker_errors
from docker import utils as docker_utils

from open_instruct.environments.backend_base import ExecutionResult, SandboxBackend, SandboxLostError, SandboxOOMError

_API_VERSION = "v1"
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MIB = 1024 * 1024
_SANDBOX_CPUS = 1


def _memory_limit_mb(mem_limit: str | int) -> int:
    try:
        memory_bytes = docker_utils.parse_bytes(mem_limit)
    except (TypeError, ValueError, docker_errors.DockerException) as error:
        raise ValueError(f"Invalid mem_limit {mem_limit!r}") from error
    memory_mb = (memory_bytes + _MIB - 1) // _MIB
    if memory_mb < 64:
        raise ValueError("mem_limit must be at least 64 MiB for Sandfleet")
    return memory_mb


class SandfleetBackend(SandboxBackend):
    """Lease a sandbox from Sandfleet and expose the normal backend contract."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 1800,
        mem_limit: str | int = "4g",
        pwd: str = "/workspace",
        extra_start_flags: tuple[str, ...] = (),
        sandfleet_url: str | None = None,
        sandfleet_pool: str | None = None,
        sandfleet_token: str | None = None,
        sandfleet_token_env: str = "SANDFLEET_CLIENT_TOKEN",
        sandfleet_request_timeout: int = 60,
        sandfleet_acquire_timeout: int = 900,
    ):
        self._image = image
        self._timeout = timeout
        self._pwd = pwd
        self._extra_start_flags = tuple(extra_start_flags)
        self._url = (sandfleet_url or os.getenv("SANDFLEET_URL") or "").rstrip("/")
        self._pool = sandfleet_pool
        self._resources = (
            None if sandfleet_pool is not None else {"cpus": _SANDBOX_CPUS, "memory_mb": _memory_limit_mb(mem_limit)}
        )
        self._token = sandfleet_token or os.getenv(sandfleet_token_env) or ""
        self._request_timeout = sandfleet_request_timeout
        self._acquire_timeout = sandfleet_acquire_timeout

        self._lease_id: str | None = None
        self._lease_token: str | None = None
        self._agent_url: str | None = None
        self._worker_metadata: dict[str, Any] = {}
        self._sandfleet_status: dict[str, Any] = {}
        self._name: str | None = None
        self._lease_ttl_seconds = 0
        self._renew_stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._renewal_error: Exception | None = None

    def _ensure_configured(self) -> None:
        if not self._url:
            raise RuntimeError("Sandfleet service URL is unset; set SANDFLEET_URL or sandfleet_url")
        if not self._token:
            raise RuntimeError("Sandfleet client token is unset")

    @staticmethod
    def _decode_error(error: HTTPError) -> tuple[str, str]:
        raw = error.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        detail = payload.get("error", {})
        return detail.get("type", "RuntimeError"), detail.get(
            "message", f"Sandfleet request failed with HTTP {error.code}"
        )

    @staticmethod
    def _raise_remote_error(error_type: str, message: str, *, cause: Exception) -> None:
        if error_type == "SandboxOOMError":
            raise SandboxOOMError(message) from cause
        if error_type == "SandboxLostError":
            raise SandboxLostError(message) from cause
        if error_type == "FileNotFoundError":
            raise FileNotFoundError(message) from cause
        if error_type == "IsADirectoryError":
            raise IsADirectoryError(message) from cause
        if error_type == "ValueError":
            raise ValueError(message) from cause
        raise RuntimeError(f"Sandfleet {error_type}: {message}") from cause

    def _request(
        self,
        base_url: str,
        path: str,
        *,
        token: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if body is not None and len(body) > _MAX_REQUEST_BYTES:
            raise ValueError(f"Sandfleet request exceeds {_MAX_REQUEST_BYTES} bytes")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(base_url.rstrip("/") + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout or self._request_timeout) as response:  # noqa: S310
                raw = response.read()
                return json.loads(raw) if raw else {}
        except HTTPError as error:
            error_type, message = self._decode_error(error)
            self._raise_remote_error(error_type, message, cause=error)
        except URLError as error:
            raise ConnectionError(f"Could not reach Sandfleet endpoint {base_url!r}: {error.reason}") from error
        raise AssertionError("unreachable")

    def _agent_request(
        self,
        operation: str,
        *,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._lease_id is None or self._lease_token is None or self._agent_url is None:
            raise RuntimeError("Sandfleet lease is not active. Call start() first.")
        if self._renewal_error is not None:
            raise SandboxLostError(f"Sandfleet lease renewal failed: {self._renewal_error}") from self._renewal_error
        try:
            return self._request(
                self._agent_url,
                f"/{_API_VERSION}/leases/{self._lease_id}/{operation}",
                token=self._lease_token,
                method=method,
                payload=payload,
                timeout=timeout,
            )
        except ConnectionError as error:
            reason = str(error)
            try:
                status = self._request(
                    self._url,
                    f"/{_API_VERSION}/leases/{self._lease_id}",
                    token=self._token,
                    timeout=self._request_timeout,
                )
                reason = status.get("lost_reason") or reason
            except Exception as controller_error:
                error.add_note(f"Controller could not explain the lost agent connection: {controller_error}")
            raise SandboxLostError(reason) from error

    def _acquire(self) -> dict[str, Any]:
        deadline = time.monotonic() + self._acquire_timeout
        selection = {"pool": self._pool} if self._resources is None else {"resources": self._resources}
        result = self._request(
            self._url,
            f"/{_API_VERSION}/lease-requests",
            token=self._token,
            method="POST",
            payload={**selection, "timeout_seconds": self._acquire_timeout},
        )
        request_id = str(result["request_id"])
        delay = max(0.0, float(result["poll_after_seconds"]))
        try:
            while result["status"] == "pending":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for Sandfleet selection {selection!r}")
                base_delay = max(delay, float(result["poll_after_seconds"]))
                time.sleep(min(base_delay * random.uniform(0.9, 1.1), remaining))
                delay = min(10.0, max(0.1, base_delay * 1.5))
                result = self._request(self._url, f"/{_API_VERSION}/lease-requests/{request_id}", token=self._token)
            if result["status"] != "assigned":
                raise TimeoutError(result["error"] or f"Could not acquire from Sandfleet selection {selection!r}")
            return result["lease"]
        except Exception as error:
            try:
                self._request(
                    self._url, f"/{_API_VERSION}/lease-requests/{request_id}", token=self._token, method="DELETE"
                )
            except Exception as cleanup_error:
                error.add_note(f"Could not cancel Sandfleet lease request {request_id}: {cleanup_error}")
            raise

    def _renew_once(self) -> None:
        lease_id = self._lease_id
        if lease_id is None:
            return
        self._request(
            self._url, f"/{_API_VERSION}/leases/{lease_id}/renew", token=self._token, method="POST", payload={}
        )

    def _renew_loop(self) -> None:
        interval = max(1.0, self._lease_ttl_seconds / 3)
        while not self._renew_stop.wait(interval * random.uniform(0.9, 1.1)):
            try:
                self._renew_once()
            except ConnectionError:
                continue
            except Exception as error:
                self._renewal_error = error
                return

    def _start_renewal(self, ttl_seconds: int) -> None:
        self._lease_ttl_seconds = ttl_seconds
        self._renewal_error = None
        self._renew_stop.clear()
        if ttl_seconds <= 0:
            return
        self._renew_thread = threading.Thread(
            target=self._renew_loop, name=f"sandfleet-renew-{str(self._lease_id)[:8]}", daemon=True
        )
        self._renew_thread.start()

    def _start_payload(self) -> dict[str, Any]:
        return {
            "image": self._image,
            "timeout": self._timeout,
            "pwd": self._pwd,
            "extra_start_flags": list(self._extra_start_flags),
        }

    def start(self) -> None:
        self._ensure_configured()
        if self._lease_id is not None:
            self.close()
        lease = self._acquire()
        self._lease_id = lease["lease_id"]
        self._lease_token = lease["lease_token"]
        self._agent_url = lease["agent_url"]
        self._start_renewal(int(lease["ttl_seconds"]))
        try:
            result = self._agent_request("start", payload=self._start_payload(), timeout=max(1800, self._timeout + 60))
        except Exception as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"Sandfleet lease release also failed: {close_error}")
            raise
        self._name = result["instance_name"]
        self._worker_metadata = result["cgroup"]
        self._sandfleet_status = result

    def restart(self) -> None:
        if self._lease_id is None:
            self.start()
            return
        result = self._agent_request("restart", payload=self._start_payload(), timeout=max(1800, self._timeout + 60))
        self._name = result["instance_name"]
        self._worker_metadata = result["cgroup"]
        self._sandfleet_status = result

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        effective_timeout = self._timeout if timeout is None else timeout
        result = self._agent_request(
            "exec", payload={"command": command, "timeout": timeout}, timeout=effective_timeout + 30
        )
        return ExecutionResult(
            stdout=str(result["stdout"]), stderr=str(result["stderr"]), exit_code=int(result["exit_code"])
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._agent_request(
            "write-file", payload={"path": path, "content_b64": base64.b64encode(content).decode("ascii")}, timeout=120
        )

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        result = self._agent_request("read-file", payload={"path": path}, timeout=120)
        content = base64.b64decode(result["content_b64"].encode("ascii"), validate=True)
        return content if binary else content.decode("utf-8", errors="replace")

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        self._agent_request(
            "put-archive",
            payload={"root": root, "content_b64": base64.b64encode(tar_bytes).decode("ascii")},
            timeout=300,
        )

    def close(self) -> None:
        lease_id = self._lease_id
        self._renew_stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=2)
            self._renew_thread = None
        if lease_id is not None:
            self._request(
                self._url,
                f"/{_API_VERSION}/leases/{lease_id}",
                token=self._token,
                method="DELETE",
                timeout=max(120, self._request_timeout),
            )
        self._lease_id = None
        self._lease_token = None
        self._agent_url = None
        self._name = None
        self._worker_metadata = {}
        self._sandfleet_status = {}
        self._lease_ttl_seconds = 0
        self._renewal_error = None
