"""The actor's local inference requests must bypass environment proxies."""

import ast
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import backoff
import openai


def test_local_inference_bypasses_proxy(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {
                    "id": "test",
                    "object": "text_completion",
                    "created": 0,
                    "model": "test",
                    "choices": [{"index": 0, "text": "ok", "finish_reason": "stop", "logprobs": None}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    for key in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
        monkeypatch.setenv(key.upper(), "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")

    source = Path(__file__).parents[1] / "open_instruct/vllm_utils.py"
    tree = ast.parse(source.read_text())
    health = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_check_health"
    )
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMRayActor")
    init = next(
        node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "_init_openai_client"
    )
    namespace = {
        "aiohttp": aiohttp,
        "backoff": backoff,
        "asyncio": asyncio,
        "openai": openai,
        "logger": SimpleNamespace(info=lambda *args: None),
        "VLLM_HEALTH_CHECK_TIMEOUT_S": 2,
        "VLLM_OPENAI_TIMEOUT_S": 2,
        "VLLM_OPENAI_MAX_RETRIES": 0,
    }
    exec(compile(ast.Module(body=[health, init], type_ignores=[]), str(source), "exec"), namespace)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    instance = SimpleNamespace(
        server_port=server.server_port,
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="test"))),
    )

    async def complete():
        async with instance.client:
            result = await instance.client.completions.create(model=instance.model_name, prompt="test")
            assert result.choices[0].text == "ok"

    try:
        namespace["_init_openai_client"](instance)
        asyncio.run(complete())
        assert requests == ["/health", "/v1/completions"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
