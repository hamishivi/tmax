"""TassieAgent — a Harbor BaseAgent with a simple tool-use loop.

Implements Harbor's BaseAgent interface and drives an LLM through
sandbox tasks using execute_bash, str_replace_editor, and submit tools.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful coding assistant. You have access to a bash terminal and a file editor. \
Use the tools to explore the codebase, understand the problem, implement a solution, and \
verify it works. When you are confident your solution is correct, use the submit tool.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": (
                "Execute a bash command in the terminal.\n"
                "* Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.\n"
                "* Interactive: Not possible. Use `yes`/`no`, etc. as appropriate.\n"
                "* Output: May be truncated. Use `head`/`tail`/`grep` to filter."
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
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "Custom editing tool for viewing, creating, and editing files.\n"
                "* State is persistent across command calls and discussions.\n"
                "* `view` for reading files/directories, `create` for new files,\n"
                "  `str_replace` for editing, `insert` for adding lines,\n"
                "  `undo_edit` to revert the last edit to a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The editor command to run.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for `create`. The full content of the new file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for `str_replace`. The exact string to replace (must appear exactly once).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Optional for `str_replace` (omit to delete old_str), required for `insert`.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for `insert`. Line number after which to insert `new_str`.",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional for `view`. Two-element [start, end] line range.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your solution and run the test suite. Only call when you believe the task is complete.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


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
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self._undo_stack: dict[str, str] = {}  # path -> previous content

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

        for step in range(self.max_steps):
            logger.info(f"Step {step + 1}/{self.max_steps}")

            response = await litellm.acompletion(
                model=model, messages=messages, tools=TOOLS, temperature=0.0,
            )
            msg = response.choices[0].message.model_dump()
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                break

            done = False
            for tc in tool_calls:
                func = tc["function"]
                name = func["name"]
                args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]

                result = await self._dispatch(name, args, environment)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

                if name == "submit":
                    done = True

            if done:
                break

        # Save trajectory
        (self.logs_dir / "trajectory.json").write_text(json.dumps(messages, indent=2, default=str))

    async def _dispatch(
        self, name: str, args: dict[str, Any], env: BaseEnvironment
    ) -> str:
        if name == "execute_bash":
            return await self._execute_bash(args, env)
        elif name == "str_replace_editor":
            return await self._str_replace_editor(args, env)
        elif name == "submit":
            return await self._submit(env)
        return f"Unknown tool: {name}"

    async def _execute_bash(self, args: dict[str, Any], env: BaseEnvironment) -> str:
        command = args.get("command", "")
        result = await env.exec(command=command)
        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.exit_code != 0:
            output = f"Exit code {result.exit_code}\n{output}"
        return output or "(no output)"

    async def _str_replace_editor(self, args: dict[str, Any], env: BaseEnvironment) -> str:
        cmd = args.get("command", "")
        path = args.get("path", "")

        if cmd == "view":
            view_range = args.get("view_range")
            if view_range and len(view_range) == 2:
                start, end = view_range
                result = await env.exec(command=f"sed -n '{start},{end}p' {path} | cat -n")
            else:
                result = await env.exec(command=f"cat -n {path}")
            return result.stdout or f"ERROR: Could not read {path}\n{result.stderr}"

        elif cmd == "create":
            file_text = args.get("file_text", "")
            escaped = file_text.replace("'", "'\\''")
            await env.exec(command=f"mkdir -p $(dirname {path})")
            await env.exec(command=f"cat > {path} << 'TASSIE_EOF'\n{escaped}\nTASSIE_EOF")
            return f"File created at {path}."

        elif cmd == "str_replace":
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            # Read, replace, write back
            result = await env.exec(command=f"cat {path}")
            if result.exit_code != 0:
                return f"ERROR reading {path}: {result.stderr}"
            content = result.stdout
            if old_str not in content:
                return "ERROR: old_str not found in file."
            if content.count(old_str) > 1:
                return f"ERROR: old_str found {content.count(old_str)} times, must be unique."
            # Save for undo
            self._undo_stack[path] = content
            new_content = content.replace(old_str, new_str, 1)
            escaped = new_content.replace("'", "'\\''")
            await env.exec(command=f"cat > {path} << 'TASSIE_EOF'\n{escaped}\nTASSIE_EOF")
            return "Replacement applied."

        elif cmd == "insert":
            line_num = args.get("insert_line", 0)
            text = args.get("new_str", "")
            # Save for undo
            result = await env.exec(command=f"cat {path}")
            if result.exit_code != 0:
                return f"ERROR reading {path}: {result.stderr}"
            self._undo_stack[path] = result.stdout
            # Insert after line_num (line_num+1 for sed's 'i' which inserts before)
            escaped = text.replace("'", "'\\''")
            insert_at = line_num + 1
            await env.exec(command=f"sed -i '{insert_at}i\\{escaped}' {path}")
            return f"Inserted after line {line_num}."

        elif cmd == "undo_edit":
            if path not in self._undo_stack:
                return "ERROR: No edit to undo for this file."
            prev = self._undo_stack.pop(path)
            escaped = prev.replace("'", "'\\''")
            await env.exec(command=f"cat > {path} << 'TASSIE_EOF'\n{escaped}\nTASSIE_EOF")
            return f"Undo applied for {path}."

        return f"Unknown subcommand: {cmd}"

    async def _submit(self, env: BaseEnvironment) -> str:
        await env.exec(command="mkdir -p /logs/verifier")
        result = await env.exec(command="bash /tests/test.sh", timeout_sec=120)
        try:
            reward_result = await env.exec(command="cat /logs/verifier/reward.txt")
            reward = float(reward_result.stdout.strip())
        except Exception:
            reward = 0.0
        output = result.stdout or "(no test output)"
        return f"Test output:\n{output}\nReward: {reward}"
