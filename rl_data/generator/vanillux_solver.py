"""Vanillux harness — clean mini-swe-agent-style bash-tool harness.

Background
----------
The original "vanillux" plan was to faithfully reproduce VanilluxAgent (a
Harbor TB-2.0 agent built on upstream SWE-agent's full toolbox: bash +
str_replace_editor + submit + review_on_submit_m). After looking at the
upstream project's own messaging — "we now recommend `mini-swe-agent`
instead of SWE-agent: same performance, much more simple & flexible"
(SWE-agent README, 2026; mini-swe-agent scores >74% on SWE-bench
verified) — we pivoted to vendoring **mini-swe-agent's prompts** instead.

mini-swe-agent's design realises that almost all of full SWE-agent's value
sits in the prompts (system + instance templates, recommended workflow,
truncation hints, format-error recovery). The multi-tool surface (editor,
submit, reviewer) adds ~600 lines of internal state machinery for
relatively little win — bash alone is sufficient for >74% on SWE-bench
verified, beating SWE-agent's own reported ~65%.

So our "vanillux" harness becomes:

    (legacy bash harness)  +  (vanillux prompts: vendored from mini-swe-agent)
                           +  (higher action budget, 64 default vs. 16)
                           +  (same single ``bash`` tool, same submit marker)

Concretely, the bash vs. vanillux A/B test that ``run_generate_solutions_*``
runs is now a clean **prompt-richness + budget** comparison rather than a
confounded prompt+tool+budget comparison. Both harnesses share the same
underlying sandbox (Apptainer instance, shared SIF), the same OpenAI-style
tool-calling format, and the same trajectory schema, so analysis across
harnesses works unchanged.

Why this is dramatically simpler than the previous sweagent-CLI approach:

* **No SIF dependencies** — works on every base SIF (legacy 9 domains AND
  ``base_intricate``), not just one. The agent loop runs in our solver
  process; only bash actions cross into the container.
* **No vendored state machinery** — we reuse our existing tool-calling
  infrastructure (``chat_completion_batch_with_tools``, ``BASH_TOOL``).
* **No subprocess parsing** — trajectory is built directly in Python.
* **Easy to customise** — edit ``vanillux_prompts.yaml`` to tweak the
  workflow recommendations or formatting rules.

Schema parity
-------------
Returns the exact same summary dict shape as
:func:`rl_data.generator.sample_solutions.run_n_solutions` so
:func:`rl_data.generate_solutions.process_task` and the downstream
aggregator (``compute_pass_at_k_for_dir``) stay drop-in.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from rl_data import chat_completion_batch_with_tools, DEFAULT_MODEL
from rl_data.generator.env import InteractiveContainerEnvironment as ContainerEnvironment
from rl_data.generator.sample_solutions import (
    CommandDebugLogger,
    SUBMIT_MARKER,
    TOOL_SCHEMAS,
    _extract_tool_call,
    _truncate,
)


# ---------------------------------------------------------------------------
# Vanillux prompt loading (vendored from mini-swe-agent)
# ---------------------------------------------------------------------------

_PROMPTS_PATH = Path(__file__).resolve().parent / "vanillux_prompts.yaml"


def _load_vanillux_prompts() -> Dict[str, Any]:
    """Load the vendored vanillux prompts. Cached at module level."""
    with _PROMPTS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_VANILLUX_PROMPTS: Dict[str, Any] = _load_vanillux_prompts()
_SYSTEM_TEMPLATE: str = _VANILLUX_PROMPTS["system_template"]
_INSTANCE_TEMPLATE: str = _VANILLUX_PROMPTS["instance_template"]
_OBS_CFG: Dict[str, Any] = _VANILLUX_PROMPTS["observation"]
_FORMAT_ERROR_TEMPLATE: str = _VANILLUX_PROMPTS["format_error_template"]


def _render_instance(task: str) -> str:
    """Render the vanillux instance template with the task description.

    The vendored template uses ``{{task}}`` as the only Jinja variable; we
    substitute literally rather than pulling in Jinja2 to keep deps light.
    """
    return _INSTANCE_TEMPLATE.replace("{{task}}", task)


def _truncate_observation(output: str) -> str:
    """Apply mini-swe-agent's head/tail truncation to long tool outputs."""
    max_chars = int(_OBS_CFG.get("max_chars", 10000))
    if len(output) <= max_chars:
        return output
    head_n = int(_OBS_CFG.get("head_chars", 5000))
    tail_n = int(_OBS_CFG.get("tail_chars", 5000))
    elided = len(output) - head_n - tail_n
    hint = _OBS_CFG.get("too_long_hint", "Output truncated.")
    return (
        f"{hint}\n\n"
        f"---- HEAD ({head_n} chars) ----\n"
        f"{output[:head_n]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({tail_n} chars) ----\n"
        f"{output[-tail_n:]}"
    )


def _format_error_message(error: str) -> str:
    return _FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


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
    max_actions: int = 64,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    num_pool_workers: int = 16,
    run_initial_tests: bool = True,
    command_timeout: float = 120.0,
    shell_init_timeout: float = 120.0,
    shell_init_attempts: int = 3,
    log_commands: bool = False,
    command_log_dir: Optional[str] = None,
    base_sifs_dir: Optional[str] = None,
    max_timeouts_per_solution: int = 2,
) -> Dict[str, Any]:
    """Vanillux harness: bash-only tool calling with mini-swe-agent prompts.

    Identical signature to :func:`rl_data.generator.sample_solutions.run_n_solutions`
    so :func:`rl_data.generate_solutions.process_task` can dispatch on
    ``cfg.harness`` without other changes.

    Differences from the legacy bash harness (``run_n_solutions``):

    * ``system_prompt`` and the user-side instance template come from the
      vendored ``vanillux_prompts.yaml`` (mini-swe-agent v2 templates).
    * ``max_actions`` defaults to **64** (legacy bash defaults to 16). A
      mini-swe-agent-style "Recommended Workflow" benefits from a larger
      budget; 64 lines up with the per-instance call limit upstream
      VanilluxAgent runs with.
    * Tool result observations are truncated using mini-swe-agent's
      head/tail strategy (``_truncate_observation``) when they exceed
      10 000 chars, instead of our legacy hard cut.
    """
    task_data = json.loads(Path(task_path).read_text(encoding="utf-8"))
    task_description: str = task_data.get("description", "").strip()
    print(f"[vanillux] running {num_solutions} solutions for task")

    results: List[Dict[str, Any]] = []
    num_success = 0

    usage_accum: List[Dict[str, int]] = [
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        for _ in range(num_solutions)
    ]

    out_dir: Optional[Path] = None
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Vanillux/mini-swe-agent-style system prompt + instance template,
    # rendered once and shared across all N parallel solution attempts.
    rendered_instance = _render_instance(task_description)
    messages: List[List[Dict[str, Any]]] = [
        [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": rendered_instance},
        ]
        for _ in range(num_solutions)
    ]

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
        # ── Initialize N parallel Apptainer instances ────────────────────
        t0 = time.time()

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

        with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
            envs = list(pool.map(_init_env, range(num_solutions)))
        print(f"[vanillux] envs initialised in {time.time() - t0:.1f}s")

        if run_initial_tests and not envs[0].run_initial_tests():
            raise AssertionError("Initial state tests failed for env")

        # ── Agent loop ────────────────────────────────────────────────────
        is_done: List[bool] = [False] * num_solutions
        not_done_idx: List[int] = list(range(num_solutions))
        timeout_counts: List[int] = [0] * num_solutions
        num_steps = 0

        while not all(is_done):
            if not not_done_idx:
                break

            prompt_messages = [messages[i] for i in not_done_idx]
            print(f"[vanillux] generating solutions for {Path(task_path).name} turn {num_steps}")
            t_gen = time.time()
            responses_raw = chat_completion_batch_with_tools(
                prompt_messages,
                tools=TOOL_SCHEMAS,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=len(prompt_messages),
            )
            print(f"[vanillux] generated in {time.time() - t_gen:.1f}s")

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
            to_exec: List[Tuple[int, str, str]] = []
            to_format_error: List[Tuple[int, str]] = []

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
                    if act.get("tool_call_id") and act.get("command"):
                        success, output = envs[n].exec(act["command"])
                        if cmd_logger:
                            cmd_logger.log(
                                n, num_steps, act["command"], success, output or "", note="submit"
                            )
                        messages[n].append({
                            "role": "tool",
                            "tool_call_id": act["tool_call_id"],
                            "content": _truncate_observation(output) if output else "(no output)",
                        })

                elif act["type"] == "command":
                    command = act.get("command") or ""
                    tool_call_id = act.get("tool_call_id") or ""
                    to_exec.append((n, command, tool_call_id))

                elif act["type"] == "no_tool_call":
                    # mini-swe-agent's format-error recovery: tell the model
                    # exactly what went wrong and let it retry. This avoids
                    # the legacy "silently drop the turn" failure mode.
                    err = "Your last response did not include a `bash` tool call."
                    to_format_error.append((n, err))

            t_exec = time.time()
            if to_exec:
                def _exec_one(item: Tuple[int, str, str]) -> Tuple[int, bool, str, str]:
                    idx, cmd, tc_id = item
                    success, output = envs[idx].exec(cmd)
                    if cmd_logger:
                        cmd_logger.log(idx, num_steps, cmd, success, output or "")
                    return idx, success, output, tc_id

                with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
                    exec_results = list(pool.map(_exec_one, to_exec))

                for idx, success, output, tc_id in exec_results:
                    truncated = _truncate_observation(output) if output else "(no output)"
                    if success:
                        result_back = f"{truncated}\n\n(exit_code=0)"
                    else:
                        result_back = f"{truncated}\n\n(exit_code=1)"

                    if "Command timed out" in (output or ""):
                        timeout_counts[idx] += 1
                        if timeout_counts[idx] >= max_timeouts_per_solution:
                            is_done[idx] = True
                            if idx not in to_mark_done:
                                to_mark_done.append(idx)
                            if verbose:
                                print(f"⏹️  [vanillux] solution {idx} aborted after {timeout_counts[idx]} timeouts")

                    if SUBMIT_MARKER in (output or ""):
                        is_done[idx] = True
                        if idx not in to_mark_done:
                            to_mark_done.append(idx)

                    messages[idx].append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_back,
                    })

            for idx, err in to_format_error:
                messages[idx].append({
                    "role": "user",
                    "content": _format_error_message(err),
                })

            print(f"[vanillux] commands executed in {time.time() - t_exec:.1f}s")

            if to_mark_done:
                done_set = set(to_mark_done)
                not_done_idx = [idx for idx in not_done_idx if idx not in done_set]

            num_steps += 1
            if num_steps >= max_actions:
                if verbose:
                    print(f"[vanillux] hit max_actions={max_actions}; terminating remaining solutions.")
                is_done = [True] * num_solutions
                not_done_idx = []
                break

        # ── Final tests ──────────────────────────────────────────────────
        t0 = time.time()

        def _run_final(i: int) -> Tuple[bool, str]:
            return envs[i].run_final_tests()

        with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:
            finals = list(pool.map(_run_final, range(num_solutions)))

        for i in range(num_solutions):
            ok, output = finals[i]
            if ok:
                num_success += 1
            results.append({
                "success": ok,
                "messages": messages[i],
                "output": output,
                "reward": 1 if ok else 0,
                "usage": usage_accum[i],
            })
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
        key: sum(u[key] for u in usage_accum)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
    }

    return {
        "num_runs": num_solutions,
        "num_success": num_success,
        "pass_at_k": pass_at_k,
        "usage": total_usage,
        "results": results,
        "harness": "vanillux",
    }
