"""Vanillux2Agent - direct LiteLLM agent using the vanillux prompt harness.

This is the Harbor-agent version of ``rl_data.generator.vanillux_solver``:
it uses the same mini-SWE-agent-derived prompts, bash tool schema, submit
marker, format-error recovery, and output truncation, but executes commands
through Harbor's active environment and calls the model directly with LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from rl_data.generator.sample_solutions import (
    SUBMIT_MARKER,
    TOOL_SCHEMAS,
    _extract_tool_call,
)
from rl_data.generator.vanillux_solver import (
    _format_error_message,
    _render_instance,
    _SYSTEM_TEMPLATE,
    _truncate_observation,
)

os.environ.setdefault("OPENAI_API_KEY", "dummy")

logger = logging.getLogger(__name__)

ABORT_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.ContextWindowExceededError,
    litellm.exceptions.UnsupportedParamsError,
    litellm.exceptions.PermissionDeniedError,
)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
_STATE_DIR = "/tmp/.vanillux2"


class Vanillux2Agent(BaseAgent):
    """Bash-tool Harbor agent with vanillux prompts and direct API calls."""

    @staticmethod
    def name() -> str:
        return "vanillux2-agent"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        cost_limit: float = 0.0,
        api_base: str | None = None,
        command_timeout: int = 120,
        persistent_bash: bool = True,
        max_format_errors: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit
        self.api_base = api_base
        self.command_timeout = command_timeout
        self.persistent_bash = persistent_bash
        self.max_format_errors = max_format_errors
        self.cost: float = 0.0

    async def setup(self, environment: BaseEnvironment) -> None:
        if not self.persistent_bash:
            return
        await environment.exec(
            command=(
                f"mkdir -p {_STATE_DIR} && "
                f"pwd > {_STATE_DIR}/cwd && "
                f"export -p > {_STATE_DIR}/env"
            ),
            timeout_sec=10,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or "anthropic/claude-haiku-4-5"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": _render_instance(instruction.strip())},
        ]

        timing_log: list[dict[str, Any]] = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        format_errors = 0

        try:
            for step in range(self.max_steps):
                logger.info("Step %s/%s", step + 1, self.max_steps)

                if self.cost_limit > 0 and self.cost >= self.cost_limit:
                    logger.warning(
                        "Cost limit reached: $%.2f >= $%.2f",
                        self.cost,
                        self.cost_limit,
                    )
                    break

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, messages)
                except litellm.exceptions.ContextWindowExceededError:
                    logger.warning("Context window exceeded; stopping current run")
                    break
                llm_time = time.monotonic() - t0

                self._accumulate_usage(response, usage_totals)
                try:
                    self.cost += litellm.completion_cost(response, model=model)
                except Exception:
                    pass

                msg = response.choices[0].message.model_dump()
                messages.append(msg)

                action = _extract_tool_call(msg)
                if action["type"] == "no_tool_call":
                    format_errors += 1
                    self._append_format_error(messages, action.get("tool_call_id"))
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "format_error": True,
                        }
                    )
                    if format_errors >= self.max_format_errors:
                        logger.warning("Stopping after %s format errors", format_errors)
                        break
                    continue

                format_errors = 0
                command = action.get("command") or ""
                tool_call_id = action.get("tool_call_id") or ""

                t1 = time.monotonic()
                result = await self._execute_bash(command, environment)
                exec_time = time.monotonic() - t1

                tool_content = self._format_tool_result(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )

                timing_log.append(
                    {
                        "step": step + 1,
                        "llm_s": round(llm_time, 1),
                        "bash_s": round(exec_time, 1),
                        "return_code": result.return_code,
                        "cmd": command[:200],
                    }
                )

                if action["type"] == "done" or SUBMIT_MARKER in tool_content:
                    break
        finally:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(messages, indent=2, default=str) + "\n"
            )
            (self.logs_dir / "timing.json").write_text(
                json.dumps(timing_log, indent=2) + "\n"
            )
            (self.logs_dir / "usage.json").write_text(
                json.dumps(
                    {
                        **usage_totals,
                        "cost_usd": self.cost,
                        "max_steps": self.max_steps,
                    },
                    indent=2,
                )
                + "\n"
            )
            context.cost_usd = self.cost
            context.n_input_tokens = usage_totals["prompt_tokens"]
            context.n_output_tokens = usage_totals["completion_tokens"]

    async def _query_with_retry(
        self, model: str, messages: list[dict[str, Any]]
    ) -> Any:
        api_base = (
            self.api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    api_base=api_base,
                )
            except ABORT_EXCEPTIONS:
                raise
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.error("Max retries reached: %s: %s", type(exc).__name__, exc)
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Retry %s/%s after %s: %s (waiting %.0fs)",
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _accumulate_usage(response: Any, usage_totals: dict[str, int]) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for key in usage_totals:
            usage_totals[key] += getattr(usage, key, 0) or 0

    @staticmethod
    def _append_format_error(
        messages: list[dict[str, Any]], tool_call_id: str | None
    ) -> None:
        content = _format_error_message(
            "Your last response did not include a valid `bash` tool call."
        )
        if tool_call_id:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )
            return
        messages.append({"role": "user", "content": content})

    def _wrap_command(self, command: str) -> str:
        if not self.persistent_bash:
            return command
        return (
            f'cd "$(cat {_STATE_DIR}/cwd)" 2>/dev/null || true\n'
            f". {_STATE_DIR}/env 2>/dev/null || true\n"
            f"{command}\n"
            "_vanillux2_ec=$?\n"
            f"pwd > {_STATE_DIR}/cwd\n"
            f"export -p > {_STATE_DIR}/env\n"
            "exit $_vanillux2_ec"
        )

    async def _execute_bash(
        self, command: str, environment: BaseEnvironment
    ) -> Any:
        return await environment.exec(
            command=self._wrap_command(command),
            timeout_sec=self.command_timeout,
        )

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr
        truncated = _truncate_observation(output) if output else "(no output)"
        return f"{truncated}\n\n(exit_code={result.return_code})"
