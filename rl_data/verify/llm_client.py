"""Direct provider client for resumable rubric, repair, and rollout jobs.

Provider-qualified LiteLLM models select credentials; gateway aliases are rejected.
"""
from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import litellm

litellm.suppress_debug_info = True
DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"


class ProviderAuthError(RuntimeError):
    """Missing credentials or provider/model access; do not retry a whole corpus."""


def provider(model: str) -> str:
    name, sep, model_id = model.partition("/")
    if not sep or name not in {"gemini", "openai", "anthropic"} or not model_id:
        raise ValueError("Use gemini/<model>, openai/<model>, or anthropic/<model>.")
    if "genai" in model_id or "metagen" in model_id.lower():
        raise ValueError("MetaGen aliases cannot be used with direct providers.")
    return name


def load_api_key(model: str | None = None) -> str:
    name = provider(model or os.getenv("VERIFY_MODEL", DEFAULT_MODEL))
    variables = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }[name]
    for var in variables:
        if os.getenv(var):
            return os.environ[var].strip()
    filename = os.getenv(f"{name.upper()}_API_KEY_FILE")
    if not filename:
        filename = f"~/.config/tmax-evals/{name}-api-key"
    path = Path(filename).expanduser()
    if path.is_file():
        if path.stat().st_mode & 0o077:
            raise ProviderAuthError(f"Credential file {path} must have mode 600.")
        key = path.read_text().strip()
        if key:
            return key
    raise ProviderAuthError(f"Set {variables[0]} or {name.upper()}_API_KEY_FILE.")


def output_limit_kwarg(model: str, tokens: int) -> dict[str, int]:
    name = provider(model)
    model_id = model.split("/", 1)[1]
    if name == "openai" and model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": tokens}
    return {"max_tokens": tokens}


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._min_interval
        if delay:
            time.sleep(delay)


def complete(
    messages: Sequence[dict[str, Any]], *, model: str | None = None,
    api_key: str | None = None, max_output_tokens: int = 4096,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None, timeout: int = 600,
    max_attempts: int = 5, limiter: RateLimiter | None = None,
) -> Any:
    resolved = model or os.getenv("VERIFY_MODEL", DEFAULT_MODEL)
    provider(resolved)
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    kwargs: dict[str, Any] = {
        "model": resolved, "messages": list(messages),
        "api_key": api_key or load_api_key(resolved), "timeout": timeout,
        "num_retries": 0, **output_limit_kwarg(resolved, max_output_tokens),
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools:
        kwargs["tools"] = tools
    for attempt in range(max_attempts):
        if limiter:
            limiter.wait()
        try:
            return litellm.completion(**kwargs)
        except (litellm.AuthenticationError, litellm.PermissionDeniedError, litellm.NotFoundError) as exc:
            raise ProviderAuthError(f"Provider access failed for {resolved} ({type(exc).__name__}).") from None
        except (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError, litellm.InternalServerError) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Provider request failed after {max_attempts} attempts ({type(exc).__name__}).") from None
            time.sleep(min(2 ** (attempt + 1), 60) + random.uniform(0, 1))
        # Invalid requests and content refusals are per-item evidence, not transient
        # failures. Let the existing task checkpoint code record them once.
