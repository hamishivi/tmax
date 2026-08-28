"""Remote sandbox backend for the Sandfleet Slurm service."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from open_instruct.environments.backends import ExecutionResult, SandboxBackend, SandboxLostError, SandboxOOMError

_API_VERSION = "v1"
_MAX_REQUEST_BYTES = 64 * 1024 * 1024


class SandfleetBackend(SandboxBackend):
    """Lease a sandbox from Sandfleet and expose the normal backend contract."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 1800,
        pwd: str = "/workspace",
        extra_start_flags: tuple[str, ...] = (),
        sandfleet_url: str | None = None,
        sandfleet_pool: str = "default",
        sandfleet_token: str | None = None,
        sandfleet_token_env: str = "SANDFLEET_TOKEN",
        sandfleet_request_timeout: int = 60,
        sandfleet_acquire_timeout: int = 900,
    ):
        self._image = image
        self._timeout = timeout
        self._pwd = pwd
        self._extra_start_flags = tuple(extra_start_flags)
        self._url = (sandfleet_url or os.getenv("SANDFLEET_URL") or "").rstrip("/")
        self._pool = sandfleet_pool
        self._token = sandfleet_token or os.getenv(sandfleet_token_env) or ""
        self._request_timeout = sandfleet_request_timeout
        self._acquire_timeout = sandfleet_acquire_timeout

        self._lease_id: str | None = None
        self._lease_token: str | None = None
        self._agent_url: str | None = None
        self._worker_metadata: dict[str, Any] = {}
        self._sandfleet_status: dict[str, Any] = {}
        self._name: str | None = None

    def _ensure_configured(self) -> None:
        if not self._url:
            raise RuntimeError("Sandfleet service URL is unset; set SANDFLEET_URL or sandfleet_url")
        if not self._token:
            raise RuntimeError("Sandfleet service token is unset")

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
            with contextlib.suppress(Exception):
                status = self._request(
                    self._url,
                    f"/{_API_VERSION}/leases/{self._lease_id}",
                    token=self._token,
                    timeout=self._request_timeout,
                )
                reason = status.get("lost_reason") or reason
            raise SandboxLostError(reason) from error

    def _acquire(self) -> dict[str, Any]:
        deadline = time.monotonic() + self._acquire_timeout
        result = self._request(
            self._url,
            f"/{_API_VERSION}/lease-requests",
            token=self._token,
            method="POST",
            payload={"pool": self._pool, "timeout_seconds": self._acquire_timeout},
        )
        request_id = str(result["request_id"])
        try:
            while result.get("status") == "pending":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for Sandfleet pool {self._pool!r}")
                time.sleep(min(float(result.get("poll_after_seconds", 2)), remaining))
                result = self._request(
                    self._url,
                    f"/{_API_VERSION}/lease-requests/{request_id}",
                    token=self._token,
                )
            if result.get("status") != "assigned" or "lease" not in result:
                raise TimeoutError(result.get("error") or f"Could not acquire from Sandfleet pool {self._pool!r}")
            return result["lease"]
        except Exception:
            with contextlib.suppress(Exception):
                self._request(
                    self._url,
                    f"/{_API_VERSION}/lease-requests/{request_id}",
                    token=self._token,
                    method="DELETE",
                )
            raise

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
        try:
            result = self._agent_request(
                "start",
                payload=self._start_payload(),
                timeout=max(1800, self._timeout + 60),
            )
        except Exception:
            self.close()
            raise
        self._name = result.get("instance_name")
        self._worker_metadata = result.get("cgroup", {})
        self._sandfleet_status = result

    def restart(self) -> None:
        if self._lease_id is None:
            self.start()
            return
        result = self._agent_request(
            "restart",
            payload=self._start_payload(),
            timeout=max(1800, self._timeout + 60),
        )
        self._name = result.get("instance_name")
        self._worker_metadata = result.get("cgroup", {})
        self._sandfleet_status = result

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        effective_timeout = self._timeout if timeout is None else timeout
        result = self._agent_request(
            "exec",
            payload={"command": command, "timeout": timeout},
            timeout=effective_timeout + 30,
        )
        return ExecutionResult(
            stdout=str(result["stdout"]),
            stderr=str(result["stderr"]),
            exit_code=int(result["exit_code"]),
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._agent_request(
            "write-file",
            payload={"path": path, "content_b64": base64.b64encode(content).decode("ascii")},
            timeout=120,
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
        lease_id, self._lease_id = self._lease_id, None
        self._lease_token = None
        self._agent_url = None
        self._name = None
        self._worker_metadata = {}
        self._sandfleet_status = {}
        if lease_id is None or not self._url or not self._token:
            return
        with contextlib.suppress(Exception):
            self._request(
                self._url,
                f"/{_API_VERSION}/leases/{lease_id}",
                token=self._token,
                method="DELETE",
                timeout=max(120, self._request_timeout),
            )
