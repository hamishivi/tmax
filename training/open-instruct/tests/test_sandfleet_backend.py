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
                    "lease": {
                        "lease_id": "lease-1",
                        "lease_token": "lease-token",
                        "agent_url": self.base_url,
                    },
                },
            )
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
        self._send(200, {"released": True})


@pytest.fixture
def service():
    _ServiceHandler.requests = []
    _ServiceHandler.oom = False
    _ServiceHandler.lost = False
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


def test_full_backend_contract_and_persistent_remote_lease(service):
    backend = SandfleetBackend(
        image="/shared/first.sif",
        sandfleet_url=service,
        sandfleet_pool="rollouts",
        sandfleet_token="service-token",
    )
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
    backend.close()

    paths = [path for _, path, _ in _ServiceHandler.requests]
    assert paths.count("/v1/lease-requests") == 1
    assert "/v1/leases/lease-1/restart" in paths
    assert paths[-1] == "/v1/leases/lease-1"


def test_remote_oom_preserves_backend_exception(service):
    backend = SandfleetBackend(
        image="/shared/image.sif",
        sandfleet_url=service,
        sandfleet_token="service-token",
    )
    backend.start()
    _ServiceHandler.oom = True
    try:
        with pytest.raises(SandboxOOMError, match="memory exhausted"):
            backend.run_command("allocate everything")
    finally:
        backend.close()


def test_lost_worker_is_a_distinct_retryable_infrastructure_error(service):
    backend = SandfleetBackend(
        image="/shared/image.sif",
        sandfleet_url=service,
        sandfleet_token="service-token",
    )
    backend.start()
    _ServiceHandler.lost = True
    try:
        with pytest.raises(SandboxLostError, match="worker preempted"):
            backend.run_command("echo never-runs")
    finally:
        backend.close()


def test_factory_ignores_local_slurm_and_memory_settings(service):
    backend = create_backend(
        "sandfleet",
        image="/shared/image.sif",
        mem_limit="4g",
        srun_cpus_per_task=8,
        sandfleet_url=service,
        sandfleet_pool="small",
        sandfleet_token="token",
        sandfleet_acquire_timeout=123,
    )
    assert isinstance(backend, SandfleetBackend)
    assert backend._pool == "small"
    assert backend._acquire_timeout == 123
