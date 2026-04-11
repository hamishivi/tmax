"""TassieAgent — a Harbor BaseAgent with a simple bash-only tool loop.

Implements Harbor's BaseAgent interface and drives an LLM through
sandbox tasks using only bash execution. Inspired by mini-SWE-agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import os
import litellm
os.environ.setdefault("OPENAI_API_KEY", "dummy")
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

# Exceptions that should immediately abort the run (no retry).
ABORT_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.ContextWindowExceededError,
    litellm.exceptions.UnsupportedParamsError,
    litellm.exceptions.PermissionDeniedError,
)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

MAX_OUTPUT_CHARS = 10_000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    n_elided = len(text) - limit
    return f"{text[:half]}\n\n... [{n_elided} characters elided] ...\n\n{text[-half:]}"


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_STATE_DIR = "/tmp/.tassie"

SYSTEM_PROMPT_STATELESS = """\
You are a helpful coding assistant. You have access to a bash terminal.
Use it to explore the codebase, understand the problem, implement a solution, and verify it works.

IMPORTANT RULES:
- Every response must include a THOUGHT section explaining your reasoning, followed by exactly one bash command.
- Directory or environment variable changes are not persistent. Every command runs in a new subshell. \
Use `cd /path && <command>` to run commands in a specific directory.
- Edit files using bash commands like `sed`, `cat > file << 'EOF'`, etc.
- Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.
- Interactive commands are not possible. Use `yes`/`no`, etc. as appropriate.
- Output may be truncated. Use `head`/`tail`/`grep` to filter large outputs.
- When you are confident your solution is correct, submit by running: \
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- After submitting you cannot continue working on the task.
"""

SYSTEM_PROMPT_PERSISTENT = """\
You are a helpful coding assistant. You have access to a persistent bash terminal.
Use it to explore the codebase, understand the problem, implement a solution, and verify it works.

IMPORTANT RULES:
- Every response must include a THOUGHT section explaining your reasoning, followed by exactly one bash command.
- Your working directory and environment variables persist between commands. \
You can `cd` into a directory and subsequent commands will run there. \
You can `export` variables and they will be available in later commands.
- Edit files using bash commands like `sed`, `cat > file << 'EOF'`, etc.
- Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.
- Interactive commands are not possible. Use `yes`/`no`, etc. as appropriate.
- Output may be truncated. Use `head`/`tail`/`grep` to filter large outputs.
- When you are confident your solution is correct, submit by running: \
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- After submitting you cannot continue working on the task.
"""

BASH_TOOL_STATELESS = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Each command runs in a new subshell.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    },
}

BASH_TOOL_PERSISTENT = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    },
}


class TassieAgent(BaseAgent):

    @staticmethod
    def name() -> str:
        return "tassie-agent"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 30,
        cost_limit: float = 0.0,
        persistent_bash: bool = False,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.cost_limit = cost_limit
        self.cost: float = 0.0
        self.persistent_bash = persistent_bash
        self.api_base = api_base

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.persistent_bash:
            await environment.exec(
                f"mkdir -p {_STATE_DIR} && pwd > {_STATE_DIR}/cwd && export -p > {_STATE_DIR}/env"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or "anthropic/claude-haiku-4-5"
        system_prompt = SYSTEM_PROMPT_PERSISTENT if self.persistent_bash else SYSTEM_PROMPT_STATELESS
        bash_tool = BASH_TOOL_PERSISTENT if self.persistent_bash else BASH_TOOL_STATELESS
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        timing_log: list[dict[str, Any]] = []

        try:
            for step in range(self.max_steps):
                logger.info(f"Step {step + 1}/{self.max_steps}")

                if self.cost_limit > 0 and self.cost >= self.cost_limit:
                    logger.warning(f"Cost limit reached: ${self.cost:.2f} >= ${self.cost_limit:.2f}")
                    break

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, messages, bash_tool)
                except litellm.exceptions.ContextWindowExceededError:
                    logger.warning("Context window exceeded — stopping and submitting current work")
                    break
                llm_time = time.monotonic() - t0

                try:
                    step_cost = litellm.completion_cost(response, model=model)
                except Exception:
                    step_cost = 0.0
                self.cost += step_cost

                msg = response.choices[0].message.model_dump()
                n_tokens = response.usage.completion_tokens if response.usage else 0
                n_prompt = response.usage.prompt_tokens if response.usage else 0

                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    timing_log.append({"step": step + 1, "llm_s": round(llm_time, 1), "completion_tokens": n_tokens, "prompt_tokens": n_prompt, "stop": True})
                    break

                done = False
                for tc in tool_calls:
                    func = tc["function"]
                    args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
                    command = args.get("command", "")

                    t1 = time.monotonic()
                    output = await self._execute_bash(command, environment)
                    exec_time = time.monotonic() - t1

                    timing_log.append({
                        "step": step + 1,
                        "llm_s": round(llm_time, 1),
                        "bash_s": round(exec_time, 1),
                        "completion_tokens": n_tokens,
                        "prompt_tokens": n_prompt,
                        "cmd": command[:80],
                    })

                    # Check for submit marker
                    if SUBMIT_MARKER in output:
                        done = True

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _truncate(output)})

                if done:
                    break
        finally:
            # Always save trajectory and timing, even on error
            (self.logs_dir / "trajectory.json").write_text(json.dumps(messages, indent=2, default=str))
            (self.logs_dir / "timing.json").write_text(json.dumps(timing_log, indent=2))

    async def _query_with_retry(self, model: str, messages: list[dict[str, Any]], bash_tool: dict[str, Any]) -> Any:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                temperature = 1.0 if "gpt-5" in model else 0.7
                return await litellm.acompletion(
                    model=model, messages=messages, tools=[bash_tool], temperature=temperature,
                    top_p=0.95,
                    api_base=self.api_base,
                )
            except ABORT_EXCEPTIONS as e:
                logger.error(f"Aborting: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"Max retries reached: {type(e).__name__}: {e}")
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Retry {attempt}/{MAX_RETRIES} after {type(e).__name__}: {e} (waiting {delay:.0f}s)")
                await asyncio.sleep(delay)

    @staticmethod
    def _wrap_command(command: str) -> str:
        """Wrap a command to restore and save shell state (cwd + env vars)."""
        return (
            f'cd "$(cat {_STATE_DIR}/cwd)" 2>/dev/null\n'
            f". {_STATE_DIR}/env 2>/dev/null\n"
            f"{command}\n"
            f"_tassie_ec=$?\n"
            f"pwd > {_STATE_DIR}/cwd\n"
            f"export -p > {_STATE_DIR}/env\n"
            f"exit $_tassie_ec"
        )

    async def _execute_bash(self, command: str, env: BaseEnvironment) -> str:
        if self.persistent_bash:
            command = self._wrap_command(command)
        result = await env.exec(command=command)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}"
        # Prefix with the command to match SFT training data format
        output = f"{command}\n{output}" if output else command
        return output or "(no output)"
