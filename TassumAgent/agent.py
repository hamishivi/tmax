"""TassumAgent — TassieAgent with proactive context summarisation.

When free tokens (max_input_tokens - current context) drop below
proactive_summarization_threshold, the agent:

1. Unwinds recent messages until there's room for summarisation
2. Runs a 3-step summarisation (summarise → Q&A → replace)
3. Falls back to simple summary if that fails
4. Ultimate fallback: continues with just system prompt + task description

Enable with enable_summarize=True in agent config.
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

ABORT_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.UnsupportedParamsError,
    litellm.exceptions.PermissionDeniedError,
)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

MAX_OUTPUT_CHARS = 10_000

# Token budget reserved for the summarisation calls themselves.
_SUMMARISATION_BUDGET = 4000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    n_elided = len(text) - limit
    return f"{text[:half]}\n\n... [{n_elided} characters elided] ...\n\n{text[-half:]}"


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_STATE_DIR = "/tmp/.tassum"

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

# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------

SUMMARISE_PROMPT = """\
You are summarising a coding agent's conversation history to compress context.

Below is the conversation so far between the agent and its bash tool. \
Produce a concise summary that preserves:
1. What the task is asking for.
2. What has been tried so far (commands run, files read/edited, key outputs).
3. What worked and what didn't.
4. The current state: working directory, files modified, current approach.
5. What the agent should do next.

Be specific — include file paths, function names, error messages, and test results. \
Do NOT include raw command outputs verbatim; summarise the key information.

Conversation to summarise:
{conversation}"""

SUMMARISE_QA_PROMPT = """\
Given this summary of work done so far, list any open questions or \
important details that might be lost. Answer each briefly.

Summary:
{summary}"""

FALLBACK_SUMMARISE_PROMPT = """\
Briefly summarise what has been done so far on this task. \
Include: files modified, current approach, key findings, and next steps. \
Be concise (under 500 words).

Task description:
{task}

Conversation ({n_messages} messages, summarised):
{conversation}"""


class TassumAgent(BaseAgent):

    @staticmethod
    def name() -> str:
        return "tassum-agent"

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
        enable_summarize: bool = False,
        max_input_tokens: int = 32768,
        proactive_summarization_threshold: int = 8000,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.cost_limit = cost_limit
        self.cost: float = 0.0
        self.persistent_bash = persistent_bash
        self.api_base = api_base
        self.enable_summarize = enable_summarize
        self.max_input_tokens = max_input_tokens
        self.proactive_summarization_threshold = proactive_summarization_threshold
        self._summarisation_count = 0

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.persistent_bash:
            await environment.exec(
                f"mkdir -p {_STATE_DIR} && pwd > {_STATE_DIR}/cwd && export -p > {_STATE_DIR}/env"
            )

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token estimate: chars / 4."""
        total = 0
        for m in messages:
            content = m.get("content") or ""
            total += len(content)
            for tc in m.get("tool_calls") or []:
                func = tc.get("function", {})
                total += len(func.get("arguments", ""))
        return total // 4

    def _free_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self.max_input_tokens - self._estimate_tokens(messages)

    def _needs_summarisation(self, messages: list[dict[str, Any]]) -> bool:
        return self._free_tokens(messages) < self.proactive_summarization_threshold

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _unwind_for_budget(
        self, messages: list[dict[str, Any]], budget: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Step 1 — Unwind: remove messages from the end (keeping at least
        the first message) until we have `budget` free tokens for the
        summarisation calls.

        Returns (to_summarise, kept_tail). The caller should also preserve
        messages[:2] (system + task).
        """
        # Always keep system + task (messages[:2])
        if len(messages) <= 2:
            return [], []

        body = messages[2:]
        kept_tail: list[dict[str, Any]] = []

        # Remove from end until we have enough room
        while body and self._estimate_tokens(messages[:2] + body) > (self.max_input_tokens - budget):
            kept_tail.insert(0, body.pop())

        return body, kept_tail

    async def _do_summarise(
        self, model: str, messages: list[dict[str, Any]], instruction: str
    ) -> list[dict[str, Any]]:
        """Run the full 4-step summarisation recovery.

        Step 1: Unwind messages to make room for summarisation calls.
        Step 2: Standard 3-step summarisation (summarise → Q&A → replace).
        Step 3: Fallback simple summary if step 2 fails.
        Step 4: Ultimate fallback — system + task only.
        """
        self._summarisation_count += 1
        logger.info(
            f"Summarisation #{self._summarisation_count}: "
            f"{len(messages)} messages, ~{self._estimate_tokens(messages)} tokens"
        )

        # Step 1: Unwind
        to_summarise, kept_tail = self._unwind_for_budget(messages, _SUMMARISATION_BUDGET)
        preserved_start = messages[:2]  # system + task

        if not to_summarise:
            # Nothing to summarise — ultimate fallback
            logger.warning("Nothing to summarise after unwind — using system + task only")
            return preserved_start + kept_tail

        conversation_text = self._render_messages(to_summarise)

        # Step 2: Standard 3-step summarisation
        try:
            summary_response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "user", "content": SUMMARISE_PROMPT.format(conversation=conversation_text)},
                ],
                temperature=0.3,
                api_base=self.api_base,
            )
            summary = summary_response.choices[0].message.content

            qa_response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "user", "content": SUMMARISE_QA_PROMPT.format(summary=summary)},
                ],
                temperature=0.3,
                api_base=self.api_base,
            )
            qa = qa_response.choices[0].message.content

            summary_content = (
                f"[CONVERSATION SUMMARY — #{self._summarisation_count}, "
                f"{len(to_summarise)} messages compressed]\n\n"
                f"{summary}\n\n"
                f"--- Additional Details ---\n{qa}"
            )

            compressed = preserved_start + [
                {"role": "user", "content": summary_content},
                {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing with the task."},
            ] + kept_tail

            logger.info(
                f"Standard summarisation done: {len(messages)} → {len(compressed)} messages, "
                f"~{self._estimate_tokens(compressed)} tokens"
            )
            return compressed

        except Exception as e:
            logger.warning(f"Standard summarisation failed: {e} — trying fallback")

        # Step 3: Fallback — simpler single-call summary
        try:
            # Use a very compressed render for the fallback
            short_conversation = self._render_messages(to_summarise, max_per_message=300)
            fallback_response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "user", "content": FALLBACK_SUMMARISE_PROMPT.format(
                        task=instruction[:2000],
                        n_messages=len(to_summarise),
                        conversation=short_conversation[:8000],
                    )},
                ],
                temperature=0.3,
                api_base=self.api_base,
            )
            fallback_summary = fallback_response.choices[0].message.content

            compressed = preserved_start + [
                {"role": "user", "content": f"[FALLBACK SUMMARY — #{self._summarisation_count}]\n\n{fallback_summary}"},
                {"role": "assistant", "content": "Understood. Continuing with the task."},
            ] + kept_tail

            logger.info(
                f"Fallback summarisation done: {len(messages)} → {len(compressed)} messages, "
                f"~{self._estimate_tokens(compressed)} tokens"
            )
            return compressed

        except Exception as e:
            logger.warning(f"Fallback summarisation also failed: {e} — ultimate fallback")

        # Step 4: Ultimate fallback — just system + task + kept tail
        logger.warning("Ultimate fallback: continuing with system + task only")
        return preserved_start + kept_tail

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]], max_per_message: int = 1000) -> str:
        """Render messages to readable text for the summariser."""
        parts = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []

            if role == "assistant" and tool_calls:
                cmds = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    cmd = args.get("command", args) if isinstance(args, dict) else args
                    cmds.append(str(cmd))
                thought = content[:max_per_message // 2]
                if len(content) > max_per_message // 2:
                    thought += "..."
                parts.append(f"[ASSISTANT] {thought}\n[COMMANDS] {'; '.join(cmds)}")
            elif role == "tool":
                truncated = content[:max_per_message]
                if len(content) > max_per_message:
                    truncated += "..."
                parts.append(f"[TOOL OUTPUT] {truncated}")
            else:
                truncated = content[:max_per_message]
                if len(content) > max_per_message:
                    truncated += "..."
                parts.append(f"[{role.upper()}] {truncated}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

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

                # Proactive summarisation check
                if self.enable_summarize and len(messages) > 2 and self._needs_summarisation(messages):
                    logger.info("Proactive summarisation triggered")
                    messages = await self._do_summarise(model, messages, instruction)

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, messages, bash_tool)
                except litellm.exceptions.ContextWindowExceededError:
                    if self.enable_summarize:
                        logger.warning("Context window exceeded — attempting emergency summarisation")
                        messages = await self._do_summarise(model, messages, instruction)
                        try:
                            response = await self._query_with_retry(model, messages, bash_tool)
                        except Exception:
                            logger.warning("Still failing after emergency summarisation — stopping")
                            break
                    else:
                        logger.warning("Context window exceeded — stopping (summarisation disabled)")
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
                    timing_log.append({
                        "step": step + 1, "llm_s": round(llm_time, 1),
                        "completion_tokens": n_tokens, "prompt_tokens": n_prompt,
                        "stop": True,
                    })
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

                    if SUBMIT_MARKER in output:
                        done = True

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _truncate(output)})

                if done:
                    break
        finally:
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
        return (
            f'cd "$(cat {_STATE_DIR}/cwd)" 2>/dev/null\n'
            f". {_STATE_DIR}/env 2>/dev/null\n"
            f"{command}\n"
            f"_tassum_ec=$?\n"
            f"pwd > {_STATE_DIR}/cwd\n"
            f"export -p > {_STATE_DIR}/env\n"
            f"exit $_tassum_ec"
        )

    async def _execute_bash(self, command: str, env: BaseEnvironment) -> str:
        if self.persistent_bash:
            command = self._wrap_command(command)
        result = await env.exec(command=command)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}"
        output = f"{command}\n{output}" if output else command
        return output or "(no output)"
