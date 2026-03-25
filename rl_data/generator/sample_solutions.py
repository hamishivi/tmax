"""Solution sampling & verification using tool-calling format.

Runs N parallel solution attempts inside Apptainer containers, driving an LLM
agent that uses the same bash tool-calling harness as tmax's SFT training data.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from rl_data import chat_completion_batch_with_tools, DEFAULT_MODEL
from rl_data.generator.env import InteractiveContainerEnvironment as ContainerEnvironment

MAX_OUTPUT_LENGTH = 50_000
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

HARNESS_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "sft" / "preprocessing" / "config"
SYSTEM_PROMPT = (HARNESS_CONFIG_DIR / "system_prompt.txt").read_text().strip()
TOOL_SCHEMAS = json.loads((HARNESS_CONFIG_DIR / "tool_schemas.json").read_text())

# Max characters per log entry (full stdout/stderr from container); avoids huge files.
_MAX_CMD_DEBUG_CHARS = 512_000


class CommandDebugLogger:
    """Append-only per-solution command/output logs for debugging (thread-safe per env index)."""

    def __init__(self, base_dir: Path, num_envs: int, task_path: str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks = [threading.Lock() for _ in range(num_envs)]
        readme = self.base_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "Per-solution bash command logs from generate_solutions / run_n_solutions.\n"
                f"task.json: {task_path}\n"
                "Files: env_0000.log, env_0001.log, ... (one parallel solution attempt each).\n"
                "Each block: timestamp, turn, success, command, raw PTY output.\n",
                encoding="utf-8",
            )

    def log(
        self,
        env_idx: int,
        turn: int,
        command: str,
        success: bool,
        output: str,
        *,
        note: str = "",
    ) -> None:
        if env_idx < 0 or env_idx >= len(self._locks):
            return
        path = self.base_dir / f"env_{env_idx:04d}.log"
        body = output or ""
        if len(body) > _MAX_CMD_DEBUG_CHARS:
            tail = len(body) - _MAX_CMD_DEBUG_CHARS
            body = body[:_MAX_CMD_DEBUG_CHARS] + f"\n... [{tail} characters truncated for log]\n"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        extra = f"  note={note}" if note else ""
        block = (
            f"\n{'=' * 80}\n"
            f"time={ts}  solution={env_idx}  turn={turn}  success={success}{extra}\n"
            f"$ {command}\n"
            f"{'-' * 80}\n"
            f"{body}\n"
        )
        with self._locks[env_idx]:
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(block)


def _truncate(text: str, limit: int = MAX_OUTPUT_LENGTH) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    n_elided = len(text) - limit
    return f"{text[:half]}\n\n... [{n_elided} characters elided] ...\n\n{text[-half:]}"


def _extract_tool_call(response_msg: dict) -> Dict[str, Optional[str]]:
    """Parse a tool-calling response message.

    Returns dict with:
      type: "command" | "done" | "no_tool_call"
      command: the bash command string (if type=="command")
      tool_call_id: the id needed for the tool response message
    """
    tool_calls = response_msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "command": None, "tool_call_id": None}

    tc = tool_calls[0]
    func = tc.get("function", {})
    func_name = func.get("name", "")

    if func_name != "bash":
        return {"type": "no_tool_call", "command": None, "tool_call_id": tc.get("id")}

    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return {"type": "no_tool_call", "command": None, "tool_call_id": tc.get("id")}
    else:
        args = args_raw

    command = args.get("command", "").strip()
    tool_call_id = tc.get("id")

    if SUBMIT_MARKER in command:
        return {"type": "done", "command": command, "tool_call_id": tool_call_id}

    return {"type": "command", "command": command, "tool_call_id": tool_call_id}


def run_n_solutions(
    num_solutions: int,
    container_sif_path: str,
    initial_test_path: str,
    final_test_path: str,
    def_path: str,
    task_path: str,
    max_actions: int = 16,
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
    """Produce n interactive solutions for the given task using tool-calling format."""

    task_data = json.loads(Path(task_path).read_text(encoding="utf-8"))
    task_description: str = task_data.get("description", "").strip()
    print(f"running {num_solutions} solutions for task")
    results: List[Dict[str, Any]] = []
    num_success = 0

    # Per-solution token usage accumulators
    usage_accum: List[Dict[str, int]] = [
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
         "reasoning_tokens": 0}
        for _ in range(num_solutions)
    ]

    out_dir: Optional[Path] = None
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    messages: List[List[Dict[str, Any]]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ]
        for _ in range(num_solutions)
    ]

    envs: List[ContainerEnvironment] = []
    cmd_logger: Optional[CommandDebugLogger] = None
    if log_commands:
        if command_log_dir:
            log_root = Path(command_log_dir).expanduser().resolve()
        elif save_dir:
            log_root = (Path(save_dir).expanduser().resolve() / "debug_commands")
        else:
            log_root = None
        if log_root is not None:
            cmd_logger = CommandDebugLogger(log_root, num_solutions, str(Path(task_path).resolve()))
        elif verbose:
            print("⚠️  log_commands=True but no command_log_dir and no save_dir; command debug logs disabled.")

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
        end_time = time.time()
        print(f"environments initialized in {end_time - start_time:.1f} seconds")

        if run_initial_tests:
            if not envs[0].run_initial_tests():
                raise AssertionError("Initial state tests failed for env")

        is_done: List[bool] = [False] * num_solutions
        not_done_idx: List[int] = list(range(num_solutions))
        timeout_counts: List[int] = [0] * num_solutions
        num_steps = 0

        while not all(is_done):
            if not not_done_idx:
                break

            prompt_messages = [messages[i] for i in not_done_idx]
            print(f"generating solutions...for task {task_path} turn {num_steps}")
            start_time = time.time()
            responses_raw = chat_completion_batch_with_tools(
                prompt_messages,
                tools=TOOL_SCHEMAS,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=len(prompt_messages),
            )
            end_time = time.time()
            print(f"solutions generated in {end_time - start_time:.1f} seconds")

            response_msgs: List[dict] = []
            for local_i, r in enumerate(responses_raw):
                if r is None:
                    response_msgs.append({})
                else:
                    response_msgs.append(r.choices[0].message.model_dump())
                    sol_idx = not_done_idx[local_i]
                    if hasattr(r, "usage") and r.usage is not None:
                        u = r.usage
                        usage_accum[sol_idx]["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                        usage_accum[sol_idx]["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                        usage_accum[sol_idx]["total_tokens"] += getattr(u, "total_tokens", 0) or 0
                        usage_accum[sol_idx]["reasoning_tokens"] += getattr(u, "reasoning_tokens", 0) or 0

            actions = [_extract_tool_call(msg) for msg in response_msgs]

            to_mark_done: List[int] = []
            to_exec: List[tuple[int, str, str]] = []

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

                if act["type"] == "done":
                    is_done[n] = True
                    to_mark_done.append(n)
                    if act["tool_call_id"] and act["command"]:
                        success, output = envs[n].exec(act["command"])
                        if cmd_logger:
                            cmd_logger.log(
                                n, num_steps, act["command"], success, output or "", note="submit"
                            )
                        messages[n].append({
                            "role": "tool",
                            "tool_call_id": act["tool_call_id"],
                            "content": _truncate(output) if output else "(no output)",
                        })

                elif act["type"] == "command":
                    command = act["command"] or ""
                    tool_call_id = act["tool_call_id"] or ""
                    to_exec.append((n, command, tool_call_id))

                else:
                    pass

            start_time = time.time()
            if to_exec:
                def _exec_one(item: tuple[int, str, str]) -> tuple[int, bool, str, str]:
                    idx, cmd, tc_id = item
                    success, output = envs[idx].exec(cmd)
                    if cmd_logger:
                        cmd_logger.log(idx, num_steps, cmd, success, output or "")
                    return idx, success, output, tc_id

                with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
                    exec_results: list[tuple[int, bool, str, str]] = list(pool.map(_exec_one, to_exec))

                for idx, success, output, tc_id in exec_results:
                    truncated = _truncate(output) if output else "(no output)"

                    if success:
                        result_back = f"{truncated}\n\n(exit_code=0)"
                    else:
                        result_back = f"{truncated}\n\n(exit_code=1)"

                    if "Command timed out" in output:
                        timeout_counts[idx] += 1
                        if timeout_counts[idx] >= max_timeouts_per_solution:
                            is_done[idx] = True
                            if idx not in to_mark_done:
                                to_mark_done.append(idx)
                            if verbose:
                                print(f"⏹️  Solution {idx} aborted after {timeout_counts[idx]} timeouts")

                    if SUBMIT_MARKER in output:
                        is_done[idx] = True
                        if idx not in to_mark_done:
                            to_mark_done.append(idx)

                    messages[idx].append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_back,
                    })

            end_time = time.time()
            print(f"commands executed in {end_time - start_time:.1f} seconds")

            if to_mark_done:
                done_set = set(to_mark_done)
                not_done_idx = [idx for idx in not_done_idx if idx not in done_set]

            num_steps += 1
            if num_steps >= max_actions:
                is_done = [True] * num_solutions
                not_done_idx = []
                break

        start_time = time.time()

        def _run_final(i: int) -> tuple[bool, str]:
            return envs[i].run_final_tests()

        with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
            finals: list[tuple[bool, str]] = list(pool.map(_run_final, range(num_solutions)))

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
        end_time = time.time()
        print(f"final tests executed in {end_time - start_time:.1f} seconds")

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
            p = 0.0
        else:
            p = 1.0 - (comb(n - c, k) / comb(n, k))
        pass_at_k[k] = float(p)

    total_usage = {
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage_accum),
        "completion_tokens": sum(u["completion_tokens"] for u in usage_accum),
        "total_tokens": sum(u["total_tokens"] for u in usage_accum),
        "reasoning_tokens": sum(u["reasoning_tokens"] for u in usage_accum),
    }

    summary: Dict[str, Any] = {
        "num_runs": num_solutions,
        "num_success": num_success,
        "pass_at_k": pass_at_k,
        "usage": total_usage,
        "results": results,
    }

    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--task-dir", type=str, default="tasks")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)

    args = ap.parse_args()
    n = args.n
    task_dir = args.task_dir
    task_path = os.path.join(task_dir, "task.json")
    container_sif_path = os.path.join(task_dir, "container.sif")
    initial_test_path = os.path.join(task_dir, "test_initial_state.py")
    final_test_path = os.path.join(task_dir, "test_final_state.py")
    def_path_str = os.path.join(task_dir, "container.def")

    max_actions = 16

    summary = run_n_solutions(
        n,
        container_sif_path=container_sif_path,
        initial_test_path=initial_test_path,
        final_test_path=final_test_path,
        def_path=def_path_str,
        task_path=task_path,
        max_actions=max_actions,
        model=args.model,
        temperature=0.7,
        save_dir=task_dir,
        verbose=True,
        run_initial_tests=True,
    )

    print(json.dumps({
        "num_runs": summary["num_runs"],
        "num_success": summary["num_success"],
        "pass_at_k": summary["pass_at_k"],
    }, indent=4))
