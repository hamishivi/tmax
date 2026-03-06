"""TassieAgent — a Harbor BaseAgent with a simple bash-only tool loop.

Implements Harbor's BaseAgent interface and drives an LLM through
sandbox tasks using only bash execution. Inspired by mini-SWE-agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

SYSTEM_PROMPT = """\
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

BASH_TOOL = {
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
        cost_limit: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.cost_limit = cost_limit
        self.cost: float = 0.0

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or "anthropic/claude-sonnet-4-20250514"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        try:
            for step in range(self.max_steps):
                logger.info(f"Step {step + 1}/{self.max_steps}")

                if self.cost_limit > 0 and self.cost >= self.cost_limit:
                    logger.warning(f"Cost limit reached: ${self.cost:.2f} >= ${self.cost_limit:.2f}")
                    break

                response = await self._query_with_retry(model, messages)

                try:
                    step_cost = litellm.completion_cost(response, model=model)
                except Exception:
                    step_cost = 0.0
                self.cost += step_cost
                logger.info(f"Step cost: ${step_cost:.4f}, total: ${self.cost:.4f}")

                msg = response.choices[0].message.model_dump()
                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    break

                done = False
                for tc in tool_calls:
                    func = tc["function"]
                    args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
                    command = args.get("command", "")

                    output = await self._execute_bash(command, environment)

                    # Check for submit marker
                    if output.startswith(SUBMIT_MARKER):
                        done = True

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _truncate(output)})

                if done:
                    break
        finally:
            # Always save trajectory, even on error
            (self.logs_dir / "trajectory.json").write_text(json.dumps(messages, indent=2, default=str))

    async def _query_with_retry(self, model: str, messages: list[dict[str, Any]]) -> Any:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await litellm.acompletion(
                    model=model, messages=messages, tools=[BASH_TOOL], temperature=0.7, top_p=0.95,
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

    async def _execute_bash(self, command: str, env: BaseEnvironment) -> str:
        result = await env.exec(command=command)
        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.return_code != 0:
            output = f"Exit code {result.return_code}\n{output}"
        return output or "(no output)"
