"""Tests for the remote Sandfleet sandbox backend."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from open_instruct.environments.backends import ExecutionResult, SandboxLostError, SandboxOOMError, create_backend
from open_instruct.environments.sandfleet_backend import SandfleetBackend


class _ServiceHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, dict]] = []
    base_url = ""
    oom = False
    lost = False
    release_error = False

    def log_message(self, *_args):
        pass

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        payload = self._body()
        self.__class__.requests.append(("POST", self.path, payload))
        if self.path == "/v1/lease-requests":
            self._send(
                202,
                {
                    "request_id": "request-1",
                    "status": "assigned",
                    "poll_after_seconds": 2,
                    "lease": {
                        "lease_id": "lease-1",
                        "lease_token": "lease-token",
                        "agent_url": self.base_url,
                        "ttl_seconds": 3600,
                    },
                },
            )
        elif self.path == "/v1/leases/lease-1/renew":
            self._send(200, {"lease_id": "lease-1", "ttl_seconds": 3600})
        elif self.path.endswith("/start"):
            self._send(200, {"instance_name": "first", "cgroup": {"path": "/cg/step-1"}})
        elif self.path.endswith("/restart"):
            self._send(200, {"instance_name": "second", "cgroup": {"path": "/cg/step-1"}})
        elif self.path.endswith("/exec"):
            if self.lost:
                self._send(500, {"error": {"type": "SandboxLostError", "message": "worker preempted"}})
            elif self.oom:
                self._send(500, {"error": {"type": "SandboxOOMError", "message": "memory exhausted"}})
            else:
                self._send(200, {"stdout": "hello\n", "stderr": "", "exit_code": 0})
        elif self.path.endswith("/write-file") or self.path.endswith("/put-archive"):
            self._send(200, {"ok": True})
        elif self.path.endswith("/read-file"):
            self._send(200, {"content_b64": base64.b64encode(b"contents").decode()})
        else:
            self._send(404, {"error": {"type": "KeyError", "message": self.path}})

    def do_DELETE(self):  # noqa: N802
        payload = self._body()
        self.__class__.requests.append(("DELETE", self.path, payload))
        if self.release_error:
            self._send(500, {"error": {"type": "RuntimeError", "message": "release failed"}})
        else:
            self._send(200, {"released": True})


@pytest.fixture
def service():
    _ServiceHandler.requests = []
    _ServiceHandler.oom = False
    _ServiceHandler.lost = False
    _ServiceHandler.release_error = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ServiceHandler)
    _ServiceHandler.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _ServiceHandler.base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(autouse=True)
def client_token(monkeypatch):
    monkeypatch.setenv("SANDFLEET_CLIENT_TOKEN", "client-token")


def test_full_backend_contract_and_persistent_remote_lease(service):
    backend = SandfleetBackend(image="/shared/first.sif", sandfleet_url=service, sandfleet_pool="rollouts")
    backend.start()
    assert backend._name == "first"
    assert backend._worker_metadata["path"] == "/cg/step-1"
    assert backend.run_command("echo hello") == ExecutionResult(stdout="hello\n", stderr="", exit_code=0)
    backend.write_file("/workspace/data", b"contents")
    assert backend.read_file("/workspace/data", binary=True) == b"contents"
    backend.put_archive("/workspace", b"archive")

    backend._image = "/shared/second.sif"
    backend.restart()
    assert backend._name == "second"
    assert backend._lease_id == "lease-1"
    backend._renew_once()
    backend.close()

    paths = [path for _, path, _ in _ServiceHandler.requests]
    assert paths.count("/v1/lease-requests") == 1
    assert "/v1/leases/lease-1/restart" in paths
    assert "/v1/leases/lease-1/renew" in paths
    assert paths[-1] == "/v1/leases/lease-1"


def test_remote_oom_preserves_backend_exception(service):
    backend = SandfleetBackend(image="/shared/image.sif", sandfleet_url=service)
    backend.start()
    _ServiceHandler.oom = True
    try:
        with pytest.raises(SandboxOOMError, match="memory exhausted"):
            backend.run_command("allocate everything")
    finally:
        backend.close()


def test_lost_worker_is_a_distinct_retryable_infrastructure_error(service):
    backend = SandfleetBackend(image="/shared/image.sif", sandfleet_url=service)
    backend.start()
    _ServiceHandler.lost = True
    try:
        with pytest.raises(SandboxLostError, match="worker preempted"):
            backend.run_command("echo never-runs")
    finally:
        backend.close()


def test_named_pool_ignores_local_memory_setting(service):
    backend = create_backend(
        "sandfleet",
        image="/shared/image.sif",
        mem_limit="12g",
        sandfleet_url=service,
        sandfleet_pool="small",
        sandfleet_acquire_timeout=123,
    )
    assert isinstance(backend, SandfleetBackend)
    assert backend._pool == "small"
    assert backend._resources is None
    assert backend._acquire_timeout == 123


@pytest.mark.parametrize(("mem_limit", "memory_mb"), [("4g", 4096), ("1.5g", 1536), (67108865, 65)])
def test_backend_requests_two_cpus_and_existing_memory_limit(service, mem_limit, memory_mb):
    backend = create_backend("sandfleet", image="/shared/image.sif", mem_limit=mem_limit, sandfleet_url=service)
    backend.start()
    try:
        request = next(
            payload
            for method, path, payload in _ServiceHandler.requests
            if method == "POST" and path == "/v1/lease-requests"
        )
        assert request == {"resources": {"cpus": 2, "memory_mb": memory_mb}, "timeout_seconds": 900}
    finally:
        backend.close()


@pytest.mark.parametrize("mem_limit", ["garbage", "63m", 0])
def test_backend_rejects_invalid_or_too_small_memory(mem_limit):
    with pytest.raises(ValueError, match="mem_limit"):
        SandfleetBackend(sandfleet_url="http://controller", mem_limit=mem_limit)


def test_client_token_comes_from_environment(monkeypatch):
    monkeypatch.setenv("SANDFLEET_CLIENT_TOKEN", "different-client-token")
    monkeypatch.setenv("SANDFLEET_TOKEN", "admin-token")
    backend = SandfleetBackend(sandfleet_url="http://controller")
    assert backend._token == "different-client-token"


def test_client_role_does_not_fall_back_to_admin(monkeypatch):
    monkeypatch.delenv("SANDFLEET_CLIENT_TOKEN", raising=False)
    monkeypatch.setenv("SANDFLEET_TOKEN", "admin-token")
    backend = SandfleetBackend(sandfleet_url="http://controller")
    assert backend._token == ""


def test_failed_release_keeps_lease_releasable(service):
    backend = SandfleetBackend(sandfleet_url=service)
    backend.start()
    _ServiceHandler.release_error = True
    with pytest.raises(RuntimeError, match="release failed"):
        backend.close()
    assert backend._lease_id == "lease-1"
