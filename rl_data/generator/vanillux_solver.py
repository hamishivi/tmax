"""VanilluxAgent harness ported to our Apptainer environment.

This module mirrors :func:`rl_data.generator.sample_solutions.run_n_solutions`
but exposes the **swe-agent-flavoured** tool surface that VanilluxAgent uses
on Daytona: ``bash``, ``str_replace_editor``, and ``submit``. It drives our
existing :class:`rl_data.generator.env.InteractiveContainerEnvironment` so the
RL solution-sampling pipeline can produce trajectories that match the
deployment harness without rebuilding the env stack.

What's the same as upstream Vanillux
------------------------------------
* The 3-tool set (bash + str_replace_editor + submit) — the exact subset the
  swe-agent paper specifies as the "vanilla" setup.
* Submission via an explicit ``submit`` action (no echo-marker convention).
* Per-instance call cap (``max_actions``) — the same idea as Vanillux's
  ``per_instance_call_limit``.

What's deliberately different
-----------------------------
* ``str_replace_editor`` is implemented natively in Python here (read via
  base64, edit in-process, write back via base64) instead of relying on
  upstream swe-agent's installed editor command. This avoids needing
  swe-agent itself inside our SIFs and keeps the implementation auditable.
* No history processors / prompt caching — those are anthropic-specific and
  caused issues with Gemini in the on-Daytona Vanillux.
* No CLI shim — this is a Python function called directly by
  ``rl_data.generate_solutions`` via the ``--harness vanillux`` switch.

ATIF-v1.5 trajectory dump
-------------------------
Each run writes (in addition to the OpenAI-style ``messages`` list returned
in the summary) a top-level ATIF-v1.5 dict, so
:mod:`scripts.analysis.analyze_tb2_eval` can parse the results uniformly with
its existing ``_atif_to_steps`` helper.
"""
from __future__ import annotations

import base64
import json
import shlex
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rl_data import DEFAULT_MODEL, chat_completion_batch_with_tools
from rl_data.generator.env import InteractiveContainerEnvironment as ContainerEnvironment
from rl_data.generator.sample_solutions import (
    CommandDebugLogger,
    MAX_OUTPUT_LENGTH,
    SUBMIT_MARKER,
    _truncate,
)

# ---------------------------------------------------------------------------
# Tool schema (swe-agent-flavoured)
# ---------------------------------------------------------------------------

VANILLUX_SYSTEM_PROMPT = """\
You are a helpful coding assistant working in a Linux terminal. Use the
tools provided to explore the environment, understand the problem,
implement a solution, and verify it works. When you are confident your
solution is complete, call the `submit` tool to terminate.

You have three tools:

  * `bash(command)` — run a bash command in a persistent shell. Working
    directory and environment variables persist between calls.
  * `str_replace_editor` — surgical edits to files. Subcommands:
      * `view`            : show file contents (numbered) or directory listing
      * `create`          : create or overwrite a file with `file_text`
      * `str_replace`     : replace one *unique* occurrence of `old_str`
                           with `new_str` in `path`
      * `insert`          : insert `new_str` at `insert_line` in `path`
      * `undo_edit`       : revert the last edit on `path`
    Prefer this over heredoc-based file rewrites with bash; it is more
    reliable on long files and never duplicates content.
  * `submit` — terminate the run. Call this when your solution is complete.

Tips:
  * Use `view` to inspect a file before editing. For `str_replace`, the
    `old_str` must match exactly once in the file.
  * Long bash commands: wrap with `timeout`.
  * Do NOT call `bash` to read or write files when `str_replace_editor`
    can do it — the editor is much more precise.
"""

VANILLUX_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in a persistent shell.",
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
                "Surgical edits to files. Subcommand selected by the "
                "`command` field; see system prompt for the full list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the target file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Contents to write (used by `create`).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to replace (used by `str_replace`).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement / inserted text (used by `str_replace`, `insert`).",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "0-indexed line number for `insert` (0 = top of file).",
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
            "description": "Submit your solution; ends the run.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# str_replace_editor helpers
# ---------------------------------------------------------------------------


def _b64_read(env: ContainerEnvironment, path: str) -> Tuple[bool, bytes, str]:
    """Read a file's raw bytes via base64. Handles arbitrary content."""
    cmd = f"base64 {shlex.quote(path)} 2>&1 | tr -d '\\n'"
    success, out = env.exec(cmd)
    if not success:
        return False, b"", out or "(no output)"
    try:
        return True, base64.b64decode(out.strip()), ""
    except Exception as exc:
        return False, b"", f"base64 decode failed: {exc}; raw[:200]={out[:200]}"


def _b64_write(env: ContainerEnvironment, path: str, data: bytes) -> Tuple[bool, str]:
    """Write raw bytes to a file via base64. Idempotently mkdirs the parent."""
    encoded = base64.b64encode(data).decode("ascii")
    cmd = (
        f"mkdir -p \"$(dirname {shlex.quote(path)})\" 2>/dev/null; "
        f"echo {encoded} | base64 -d > {shlex.quote(path)}"
    )
    return env.exec(cmd)


def _editor_view(env: ContainerEnvironment, path: str) -> Tuple[bool, str]:
    success, out = env.exec(
        f"if [ -d {shlex.quote(path)} ]; then ls -la {shlex.quote(path)}; "
        f"else cat -n {shlex.quote(path)}; fi 2>&1"
    )
    return success, (out or "(no output)")


def _editor_create(env: ContainerEnvironment, path: str, file_text: str) -> Tuple[bool, str]:
    ok, out = _b64_write(env, path, file_text.encode("utf-8"))
    return (True, f"created {path}") if ok else (False, f"create failed: {out}")


def _editor_str_replace(
    env: ContainerEnvironment, path: str, old_str: str, new_str: str,
    undo_stack: List[Tuple[str, bytes]],
) -> Tuple[bool, str]:
    ok, content_bytes, err = _b64_read(env, path)
    if not ok:
        return False, f"could not read {path}: {err}"
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False, f"file {path} is not valid UTF-8; cannot str_replace"
    if old_str == "":
        return False, "old_str must not be empty"
    if old_str not in content:
        return False, f"old_str not found in {path}"
    n = content.count(old_str)
    if n > 1:
        return False, f"old_str matches {n} times in {path}; must be unique"
    undo_stack.append((path, content_bytes))
    new_content = content.replace(old_str, new_str, 1)
    ok, out = _b64_write(env, path, new_content.encode("utf-8"))
    return (True, f"replaced 1 occurrence in {path}") if ok else (False, f"write failed: {out}")


def _editor_insert(
    env: ContainerEnvironment, path: str, insert_line: int, new_str: str,
    undo_stack: List[Tuple[str, bytes]],
) -> Tuple[bool, str]:
    ok, content_bytes, err = _b64_read(env, path)
    if not ok:
        return False, f"could not read {path}: {err}"
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False, f"file {path} is not valid UTF-8; cannot insert"
    lines = content.splitlines(keepends=True)
    insert_line = max(0, min(insert_line, len(lines)))
    undo_stack.append((path, content_bytes))
    suffix = "" if new_str.endswith("\n") else "\n"
    lines.insert(insert_line, new_str + suffix)
    new_content = "".join(lines)
    ok, out = _b64_write(env, path, new_content.encode("utf-8"))
    return (
        (True, f"inserted at line {insert_line} in {path}") if ok
        else (False, f"write failed: {out}")
    )


def _editor_undo(
    env: ContainerEnvironment, path: str, undo_stack: List[Tuple[str, bytes]],
) -> Tuple[bool, str]:
    for i in range(len(undo_stack) - 1, -1, -1):
        if undo_stack[i][0] == path:
            _, content_bytes = undo_stack.pop(i)
            ok, out = _b64_write(env, path, content_bytes)
            return (
                (True, f"undid last edit on {path}") if ok
                else (False, f"undo write failed: {out}")
            )
    return False, f"no recent edits to undo for {path}"


def _editor_dispatch(
    env: ContainerEnvironment, args: Dict[str, Any], undo_stack: List[Tuple[str, bytes]],
) -> Tuple[bool, str]:
    cmd = args.get("command", "")
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return False, f"str_replace_editor: 'path' is required (got {path!r})"
    if cmd == "view":
        return _editor_view(env, path)
    if cmd == "create":
        return _editor_create(env, path, args.get("file_text", ""))
    if cmd == "str_replace":
        return _editor_str_replace(
            env, path, args.get("old_str", ""), args.get("new_str", ""), undo_stack,
        )
    if cmd == "insert":
        return _editor_insert(
            env, path, int(args.get("insert_line", 0) or 0),
            args.get("new_str", ""), undo_stack,
        )
    if cmd == "undo_edit":
        return _editor_undo(env, path, undo_stack)
    return False, f"unknown str_replace_editor command: {cmd!r}"


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------


def _extract_vanillux_action(response_msg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a tool-calling response message into a normalized action dict.

    Returns a dict with keys:
      * ``type``: ``"bash"`` / ``"str_replace_editor"`` / ``"submit"`` /
                  ``"no_tool_call"`` / ``"unknown"``.
      * ``args``: parsed JSON args (empty dict on failure).
      * ``tool_call_id``: id needed for the tool-response message.
      * ``raw_action``: best-effort string summary (used in trajectory dump).
    """
    tool_calls = response_msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "args": {}, "tool_call_id": None, "raw_action": ""}

    tc = tool_calls[0]
    func = tc.get("function", {}) or {}
    name = func.get("name", "")
    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    else:
        args = args_raw or {}

    if name in ("bash", "str_replace_editor", "submit"):
        if name == "bash":
            raw = (args.get("command") or "")[:200]
        elif name == "str_replace_editor":
            raw = f"{args.get('command', '?')} {args.get('path', '?')}"[:200]
        else:
            raw = "submit"
        return {"type": name, "args": args, "tool_call_id": tc.get("id"), "raw_action": raw}
    return {"type": "unknown", "args": args, "tool_call_id": tc.get("id"), "raw_action": name[:80]}


# ---------------------------------------------------------------------------
# ATIF-v1.5 trajectory dump
# ---------------------------------------------------------------------------


def _atif_step(step_id: int, source: str, message: str = "",
               tool_calls: Optional[List[Dict[str, Any]]] = None,
               observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "step_id": step_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "message": message,
    }
    if tool_calls:
        s["tool_calls"] = tool_calls
    if observation is not None:
        s["observation"] = observation
    return s


def _build_atif_trajectory(
    *, session_id: str, model_name: str, system_prompt: str,
    instruction: str, action_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    steps.append(_atif_step(1, "system", system_prompt))
    steps.append(_atif_step(2, "user", instruction))
    next_id = 3
    for entry in action_log:
        action_type = entry.get("type", "unknown")
        raw_action = entry.get("raw_action", "")
        observation = entry.get("observation", "")
        steps.append(_atif_step(
            next_id, "agent",
            tool_calls=[{
                "tool_call_id": f"call_{next_id}_1",
                "function_name": action_type,
                "arguments": {"raw_action": raw_action},
            }],
            observation={"results": [{"content": observation}]},
        ))
        next_id += 1
    return {
        "schema_version": "ATIF-v1.5",
        "session_id": session_id,
        "agent": {
            "name": "vanillux-apptainer",
            "version": "0.1.0",
            "model_name": model_name,
            "extra": {"original_format": "vanillux-bash-editor-submit"},
        },
        "steps": steps,
        "notes": "Generated by rl_data.generator.vanillux_solver",
        "final_metrics": {},
    }


# ---------------------------------------------------------------------------
# Main entry point — same shape as run_n_solutions for drop-in replacement
# ---------------------------------------------------------------------------


def run_n_solutions_vanillux(
    num_solutions: int,
    container_sif_path: str,
    initial_test_path: str,
    final_test_path: str,
    def_path: str,
    task_path: str,
    max_actions: int = 60,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 65536,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    num_pool_workers: int = 128,
    run_initial_tests: bool = True,
    command_timeout: float = 120.0,
    shell_init_timeout: float = 120.0,
    shell_init_attempts: int = 3,
    log_commands: bool = False,
    command_log_dir: Optional[str] = None,
    base_sifs_dir: Optional[str] = None,
    max_timeouts_per_solution: int = 2,
) -> Dict[str, Any]:
    """Vanillux-flavoured equivalent of ``run_n_solutions``.

    Identical signature so :func:`rl_data.generate_solutions.process_task` can
    pick between bash and vanillux harnesses by a single dispatch.
    """
    task_data = json.loads(Path(task_path).read_text(encoding="utf-8"))
    task_description: str = task_data.get("description", "").strip()
    print(f"[vanillux] running {num_solutions} solutions for task")

    results: List[Dict[str, Any]] = []
    num_success = 0
    usage_accum: List[Dict[str, int]] = [
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
         "reasoning_tokens": 0}
        for _ in range(num_solutions)
    ]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    messages: List[List[Dict[str, Any]]] = [
        [
            {"role": "system", "content": VANILLUX_SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ]
        for _ in range(num_solutions)
    ]
    # Per-solution undo stacks for str_replace_editor.
    undo_stacks: List[List[Tuple[str, bytes]]] = [[] for _ in range(num_solutions)]
    # Per-solution structured action log (used for ATIF dump).
    action_logs: List[List[Dict[str, Any]]] = [[] for _ in range(num_solutions)]

    envs: List[ContainerEnvironment] = []
    cmd_logger: Optional[CommandDebugLogger] = None
    if log_commands:
        if command_log_dir:
            log_root = Path(command_log_dir).expanduser().resolve()
        elif save_dir:
            log_root = Path(save_dir).expanduser().resolve() / "debug_commands"
        else:
            log_root = None
        if log_root is not None:
            cmd_logger = CommandDebugLogger(log_root, num_solutions, str(Path(task_path).resolve()))
        elif verbose:
            print("⚠️  log_commands=True but no command_log_dir / save_dir; debug logs disabled.")

    try:
        start_time = time.time()

        def _init_env(i: int) -> ContainerEnvironment:
            env = ContainerEnvironment(
                container_sif_path=container_sif_path,
                initial_test_path=initial_test_path,
                final_test_path=final_test_path,
                def_path=def_path,
                max_actions=max_actions,
                verbose=verbose,
                read_timeout=command_timeout,
                shell_init_timeout=shell_init_timeout,
                shell_init_attempts=shell_init_attempts,
                base_sifs_dir=base_sifs_dir,
            )
            ok = env.initialize(run_initial_tests=False)
            if not ok:
                raise RuntimeError(f"Failed to initialize environment #{i}")
            return env

        with ThreadPoolExecutor(max_workers=num_pool_workers) as executor:
            envs = list(executor.map(_init_env, range(num_solutions)))
        print(f"[vanillux] environments initialized in {time.time() - start_time:.1f} seconds")

        if run_initial_tests and not envs[0].run_initial_tests():
            raise AssertionError("Initial state tests failed for env")

        is_done: List[bool] = [False] * num_solutions
        not_done_idx: List[int] = list(range(num_solutions))
        timeout_counts: List[int] = [0] * num_solutions
        num_steps = 0

        while not all(is_done):
            if not not_done_idx:
                break
            prompt_messages = [messages[i] for i in not_done_idx]
            print(f"[vanillux] generating turn {num_steps} for task {task_path}")
            t0 = time.time()
            responses_raw = chat_completion_batch_with_tools(
                prompt_messages,
                tools=VANILLUX_TOOL_SCHEMAS,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=len(prompt_messages),
            )
            print(f"[vanillux] turn {num_steps} LLM took {time.time() - t0:.1f}s")

            response_msgs: List[Dict[str, Any]] = []
            for local_i, r in enumerate(responses_raw):
                if r is None:
                    response_msgs.append({})
                    continue
                response_msgs.append(r.choices[0].message.model_dump())
                sol_idx = not_done_idx[local_i]
                if hasattr(r, "usage") and r.usage is not None:
                    u = r.usage
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
                        usage_accum[sol_idx][k] += getattr(u, k, 0) or 0

            actions = [_extract_vanillux_action(msg) for msg in response_msgs]

            to_mark_done: List[int] = []
            # (sol_idx, action_dict) pairs to dispatch in parallel.
            to_dispatch: List[Tuple[int, Dict[str, Any]]] = []

            for i, n in enumerate(not_done_idx):
                msg = response_msgs[i]
                act = actions[i]
                if not msg:
                    messages[n].append({
                        "role": "assistant",
                        "content": "I encountered an error. Let me try again.",
                    })
                    continue
                messages[n].append(msg)

                if act["type"] == "submit":
                    is_done[n] = True
                    to_mark_done.append(n)
                    if act["tool_call_id"]:
                        messages[n].append({
                            "role": "tool",
                            "tool_call_id": act["tool_call_id"],
                            "content": "Submitted.",
                        })
                    action_logs[n].append({
                        "type": "submit", "raw_action": "submit",
                        "observation": "Submitted.",
                    })
                elif act["type"] in ("bash", "str_replace_editor"):
                    to_dispatch.append((n, act))
                elif act["type"] == "no_tool_call":
                    # Model emitted text without a tool call — terminate this solution.
                    is_done[n] = True
                    to_mark_done.append(n)
                    action_logs[n].append({
                        "type": "no_tool_call", "raw_action": "(text only)",
                        "observation": "(no tool call; terminated)",
                    })
                else:
                    # Unknown tool name — surface as a tool message, do not terminate.
                    err = f"unknown tool: {act.get('raw_action', '?')}"
                    if act["tool_call_id"]:
                        messages[n].append({
                            "role": "tool",
                            "tool_call_id": act["tool_call_id"],
                            "content": err,
                        })

            t0 = time.time()
            if to_dispatch:
                def _exec_one(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], bool, str]:
                    idx, action = item
                    if action["type"] == "bash":
                        cmd = (action["args"].get("command") or "").strip()
                        success, output = envs[idx].exec(cmd)
                        if cmd_logger:
                            cmd_logger.log(idx, num_steps, cmd, success, output or "")
                        return idx, action, success, output or ""
                    # str_replace_editor — explicit (success, message) tuple.
                    success, output = _editor_dispatch(
                        envs[idx], action["args"], undo_stacks[idx],
                    )
                    if cmd_logger:
                        editor_label = (
                            f"str_replace_editor {action['args'].get('command', '?')} "
                            f"{action['args'].get('path', '?')}"
                        )
                        cmd_logger.log(idx, num_steps, editor_label, success, output, note="editor")
                    return idx, action, success, output

                with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
                    exec_results = list(pool.map(_exec_one, to_dispatch))

                for idx, action, success, output in exec_results:
                    truncated = _truncate(output) if output else "(no output)"
                    if action["type"] == "bash":
                        result_back = (
                            f"{truncated}\n\n(exit_code={'0' if success else '1'})"
                        )
                        if "Command timed out" in output:
                            timeout_counts[idx] += 1
                            if timeout_counts[idx] >= max_timeouts_per_solution:
                                is_done[idx] = True
                                if idx not in to_mark_done:
                                    to_mark_done.append(idx)
                                if verbose:
                                    print(
                                        f"⏹️  [vanillux] sol {idx} aborted after "
                                        f"{timeout_counts[idx]} timeouts"
                                    )
                        # Belt-and-braces: legacy SUBMIT_MARKER also terminates.
                        if SUBMIT_MARKER in output:
                            is_done[idx] = True
                            if idx not in to_mark_done:
                                to_mark_done.append(idx)
                    else:
                        result_back = truncated

                    messages[idx].append({
                        "role": "tool",
                        "tool_call_id": action["tool_call_id"] or "",
                        "content": result_back,
                    })

                    raw_action = action.get("raw_action", "")
                    action_logs[idx].append({
                        "type": action["type"],
                        "raw_action": raw_action,
                        "observation": output,
                    })
            print(f"[vanillux] turn {num_steps} dispatch took {time.time() - t0:.1f}s")

            if to_mark_done:
                done_set = set(to_mark_done)
                not_done_idx = [idx for idx in not_done_idx if idx not in done_set]

            num_steps += 1
            if num_steps >= max_actions:
                if verbose:
                    print(f"[vanillux] reached max_actions={max_actions}; stopping")
                is_done = [True] * num_solutions
                not_done_idx = []
                break

        # ── Final tests ─────────────────────────────────────────────────────
        t0 = time.time()

        def _run_final(i: int) -> Tuple[bool, str]:
            return envs[i].run_final_tests()

        with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
            finals = list(pool.map(_run_final, range(num_solutions)))

        atif_dir: Optional[Path] = None
        if save_dir:
            atif_dir = Path(save_dir) / "atif"
            atif_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_solutions):
            success, output = finals[i]
            if success:
                num_success += 1
            results.append({
                "success": success,
                "messages": messages[i],
                "output": output,
                "reward": 1 if success else 0,
                "usage": usage_accum[i],
            })
            if atif_dir is not None:
                traj = _build_atif_trajectory(
                    session_id=str(uuid.uuid4()),
                    model_name=model,
                    system_prompt=VANILLUX_SYSTEM_PROMPT,
                    instruction=task_description,
                    action_log=action_logs[i],
                )
                (atif_dir / f"sol_{i:03d}.atif.json").write_text(json.dumps(traj, indent=2))
        print(f"[vanillux] final tests in {time.time() - t0:.1f}s")

    finally:
        for env in envs:
            try:
                env.cleanup()
            except Exception:
                pass

    n = num_solutions
    c = num_success
    pass_at_k: Dict[int, float] = {}
    for k in range(1, n + 1):
        if c == 0:
            pass_at_k[k] = 0.0
        else:
            pass_at_k[k] = float(1.0 - (comb(n - c, k) / comb(n, k)))

    total_usage = {
        k: sum(u[k] for u in usage_accum)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
    }

    summary: Dict[str, Any] = {
        "num_runs": num_solutions,
        "num_success": num_success,
        "pass_at_k": pass_at_k,
        "usage": total_usage,
        "results": results,
        "harness": "vanillux",
    }
    return summary
