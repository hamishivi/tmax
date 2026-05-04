#!/usr/bin/env python3
"""Analyze a Harbor TassieAgent eval run (Terminal-Bench 2.0 or similar).

Walks a job directory (jobs/<name>/<trial>/...) and emits:

    {out_dir}/per_trial.jsonl    one row per trial, all extracted fields
    {out_dir}/summary.json       aggregate statistics, machine-readable
    {out_dir}/failures.md        auto-generated narrative of every failure

Reusable across (model, harness) combinations: pass --label to tag the run.

Usage
-----
    uv run python scripts/analysis/analyze_tb2_eval.py \\
        --job-dir jobs/tb2_gemini \\
        --harbor-cache ~/.cache/harbor \\
        --label "TassieAgent + gemini-3-flash-preview" \\
        --out scripts/analysis/out/tb2_gemini_tassieagent

It is intentionally dependency-free (stdlib only) so it can run on any node.

Vocabulary
----------
- "trial"  : one (task, attempt) pair, lives at jobs/<name>/<trial_name>/
- "turn"   : one TassieAgent step (1 LLM call + 1 bash exec)
- "submitted" : the agent emitted ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
- "hit_max_steps" : the agent stopped because it ran out of steps, not submission
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# SWE-agent / VanilluxAgent submits by running a `submit` action; the literal
# token appears as the final word (or alone) in the bash command. We match
# whole-word at end-of-string or before a non-word char to avoid false hits
# from words like "submitted" or paths containing "submit".
_SUBMIT_RE_SWE = re.compile(r"(?:^|[\s|;&])submit\b\s*$|(?:^|[\s|;&])submit_and_validate\b")


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    trial_name: str
    task_name: str
    # Outcome
    reward: float
    passed: bool
    exception_type: str | None
    failure_mode: str
    # Agent loop
    n_steps: int
    max_steps: int
    hit_max_steps: bool
    submitted: bool
    final_assistant_text: str
    # Tools used (bash command verb histogram, top-level only)
    tool_hist: dict[str, int]
    # Token & timing
    prompt_tokens_total: int
    completion_tokens_total: int
    peak_prompt_tokens: int
    llm_total_s: float
    bash_total_s: float
    agent_execution_s: float
    verifier_s: float
    total_s: float
    # Verifier (test-level)
    n_tests: int
    n_tests_passed: int
    n_tests_failed: int
    failed_test_names: list[str]
    failed_test_excerpts: list[str]  # one short excerpt per failed test
    # Task metadata (from task.toml)
    task_difficulty: str | None
    task_category: str | None
    task_tags: list[str]
    task_agent_timeout_s: float | None
    task_env_image: str | None
    instruction_excerpt: str
    # Misc
    model_name: str | None
    agent_name: str | None


def _safe_load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _bash_verb(cmd: str) -> str:
    """Extract the leading "tool" of a bash command for histogram bucketing.

    We look for the first command in the line, ignoring leading whitespace,
    comments, and shell prefixes like timeout/sudo/env. For heredoc cat
    constructs we report ``cat-write``. We strip path components so
    ``/usr/bin/python3`` is normalised to ``python3``.
    """
    if not cmd:
        return "(empty)"
    # Heredoc file-write: cat > /path << 'EOF' ... EOF
    if re.search(r"^\s*cat\s*>", cmd) or re.search(r"^\s*cat\s*>>", cmd):
        return "cat-write"
    # Tear off trailing pipe/&&; just look at first segment.
    first = re.split(r"[|&;]", cmd, maxsplit=1)[0].strip()
    if not first:
        return "(empty)"
    # Strip leading 'sudo', 'timeout N', 'env VAR=val' wrappers iteratively.
    tokens = first.split()
    while tokens and tokens[0] in {"sudo", "env"}:
        tokens.pop(0)
        # 'env' may have multiple VAR=val tokens.
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens.pop(0)
    if tokens and tokens[0] == "timeout":
        # `timeout 10 cmd ...` or `timeout --foreground 10 cmd`
        tokens.pop(0)
        # Skip any flag args
        while tokens and tokens[0].startswith("-"):
            tokens.pop(0)
        if tokens:
            tokens.pop(0)  # the duration arg
    if not tokens:
        return "(empty)"
    head = tokens[0]
    head = head.split("/")[-1]  # /usr/bin/python3 -> python3
    head = head.lstrip("'\"")
    # Strip Python -m form: "python3 -m pip" -> "python3-m"
    return head[:32]


def _classify_failure(
    *,
    passed: bool,
    exception_type: str | None,
    submitted: bool,
    hit_max_steps: bool,
    n_tests: int,
    n_tests_failed: int,
    failed_test_excerpts: list[str],
) -> str:
    if passed:
        return "pass"
    if exception_type == "AgentTimeoutError":
        return "agent_timeout"
    if exception_type:
        return f"other_error:{exception_type}"
    if not submitted and hit_max_steps:
        return "no_submit_max_steps"
    if not submitted:
        return "no_submit_early_stop"
    # Submitted but verifier failed.
    if n_tests == 0:
        return "submitted_verifier_no_tests"
    # Bucket by failure character, looking at the first excerpt.
    txt = (failed_test_excerpts[0] if failed_test_excerpts else "").lower()
    if "filenotfounderror" in txt or "no such file" in txt or "does not exist" in txt:
        return "submitted_missing_artifact"
    if (
        "header" in txt
        or "csv structure" in txt
        or "schema" in txt
        or "format" in txt
        or "unexpected" in txt and "header" in txt
    ):
        return "submitted_wrong_format"
    if "expected" in txt and ("got" in txt or "actual" in txt):
        return "submitted_wrong_value"
    if "modulenotfounderror" in txt or "importerror" in txt:
        return "submitted_missing_dependency"
    return "submitted_verifier_failed_other"


def _ctrf_failures(ctrf: dict | None) -> tuple[int, int, list[str], list[str]]:
    """Return (n_tests, n_failed, failed_names, failed_excerpts) from ctrf.json."""
    if not ctrf:
        return 0, 0, [], []
    summary = ctrf.get("results", {}).get("summary", {})
    tests = ctrf.get("results", {}).get("tests", []) or []
    n_tests = summary.get("tests", len(tests))
    failed_names: list[str] = []
    failed_excerpts: list[str] = []
    for t in tests:
        if t.get("status") != "passed":
            failed_names.append(t.get("name", "?"))
            trace = (t.get("trace") or t.get("message") or "").strip()
            # Keep the last non-empty assertion-ish line, plus 1-2 lines of context.
            assertion_lines = [ln for ln in trace.splitlines() if ln.startswith("E ")]
            if assertion_lines:
                excerpt = "\n".join(assertion_lines[:3])
            else:
                # Fallback: tail of trace.
                excerpt = "\n".join(trace.splitlines()[-3:])
            failed_excerpts.append(excerpt[:600])
    n_failed = summary.get("failed", len(failed_excerpts))
    return n_tests, n_failed, failed_names, failed_excerpts


def _final_assistant_text(trajectory: Any) -> str:
    """Return the last non-empty assistant text from either trajectory shape.

    Supports two shapes:
      (a) TassieAgent: list of OpenAI-style messages [{role, content, ...}, ...]
      (b) VanilluxAgent / swe-agent ATIF-v1.5: {"steps": [{source, message, ...}, ...]}
    """
    if not trajectory:
        return ""
    # ATIF: dict with "steps" list.
    if isinstance(trajectory, dict) and isinstance(trajectory.get("steps"), list):
        for step in reversed(trajectory["steps"]):
            if step.get("source") != "agent":
                continue
            msg = step.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:800]
        return ""
    # OpenAI-style list of messages.
    if isinstance(trajectory, list):
        for msg in reversed(trajectory):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:800]
            if isinstance(content, list):
                chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
                joined = "\n".join(c for c in chunks if c)
                if joined.strip():
                    return joined.strip()[:800]
    return ""


def _atif_to_steps(trajectory: dict) -> list[dict]:
    """Project an ATIF-v1.5 trajectory into TassieAgent-style step records.

    Returns a list of dicts with the same shape `agent/timing.json` uses:
        {step, cmd, llm_s, bash_s, prompt_tokens, completion_tokens}

    SWE-agent doesn't expose per-step latency or token counts in the ATIF dump,
    so the timing/token fields are zero. The cmd is truncated to 80 chars to
    match TassieAgent's preview convention (preserves submission detection).
    """
    out: list[dict] = []
    steps = trajectory.get("steps") or []
    for s in steps:
        if s.get("source") != "agent":
            continue
        tcs = s.get("tool_calls") or []
        if not tcs:
            continue
        # SWE-agent emits one tool call per step ("swe_agent_action").
        args = tcs[0].get("arguments") or {}
        raw = args.get("raw_action") or args.get("command") or ""
        if not isinstance(raw, str):
            raw = json.dumps(raw)[:200]
        out.append({
            "step": s.get("step_id", len(out) + 1),
            "cmd": raw[:80],
            "llm_s": 0.0,
            "bash_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })
    return out


def _build_task_index(harbor_cache: Path) -> dict[str, Path]:
    """Map task_name -> path to its cached task dir.

    Harbor stores tasks at ~/.cache/harbor/tasks/<random_id>/<task_name>/.
    We just walk one level deep.
    """
    index: dict[str, Path] = {}
    tasks_root = harbor_cache / "tasks"
    if not tasks_root.exists():
        return index
    for hash_dir in tasks_root.iterdir():
        if not hash_dir.is_dir():
            continue
        for task_dir in hash_dir.iterdir():
            if task_dir.is_dir() and (task_dir / "task.toml").exists():
                index[task_dir.name] = task_dir
    return index


def _read_task_meta(task_dir: Path | None) -> tuple[dict, str]:
    """Return (meta_dict, instruction_excerpt) for a cached task dir."""
    if not task_dir:
        return {}, ""
    meta: dict[str, Any] = {}
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            meta = tomllib.loads(toml_path.read_text())
        except Exception:
            meta = {}
    instr = ""
    instr_path = task_dir / "instruction.md"
    if instr_path.exists():
        try:
            instr = instr_path.read_text().strip()[:1000]
        except Exception:
            instr = ""
    return meta, instr


def _duration_s(ts: dict | None) -> float:
    if not ts:
        return 0.0
    a, b = ts.get("started_at"), ts.get("finished_at")
    if not a or not b:
        return 0.0
    # ISO-8601, possibly with trailing Z.
    from datetime import datetime
    def _p(s: str) -> float:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    return max(0.0, _p(b) - _p(a))


def extract_trial(trial_dir: Path, task_index: dict[str, Path]) -> TrialRecord | None:
    result = _safe_load_json(trial_dir / "result.json")
    if not result:
        return None

    task_name = result.get("task_name") or trial_dir.name.split("__")[0]
    cfg = result.get("config", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    max_steps = int(agent_cfg.get("kwargs", {}).get("max_steps", 50) or 50)

    # Reward / status
    vr = result.get("verifier_result") or {}
    reward = float((vr.get("rewards") or {}).get("reward", 0.0) or 0.0)
    passed = reward >= 1.0
    exc_info = result.get("exception_info") or None
    exc_type = (exc_info or {}).get("exception_type")

    # Timings
    agent_execution_s = _duration_s(result.get("agent_execution"))
    verifier_s = _duration_s(result.get("verifier"))
    total_s = _duration_s({"started_at": result.get("started_at"), "finished_at": result.get("finished_at")})

    # Two harness layouts:
    #   - TassieAgent : agent/timing.json (one row per step) + agent/trajectory.json (OpenAI msgs)
    #   - VanilluxAgent (swe-agent ATIF-v1.5): only agent/trajectory.json with a "steps" list,
    #     no separate timing.json. We synthesize step rows from the trajectory.
    timing = _safe_load_json(trial_dir / "agent" / "timing.json")
    trajectory = _safe_load_json(trial_dir / "agent" / "trajectory.json")
    is_atif = isinstance(trajectory, dict) and isinstance(trajectory.get("steps"), list)
    if not timing and is_atif:
        timing = _atif_to_steps(trajectory)
        # In SWE-agent / VanilluxAgent the budget is on LLM *calls*, not steps.
        # If the run dir name encodes the cap (e.g. "..._calls100"), trust that;
        # otherwise fall back to the configured max_steps.
        m = re.search(r"_calls(\d+)\b", trial_dir.parent.name)
        if m:
            max_steps = int(m.group(1))
    timing = timing or []

    n_steps = len(timing)
    hit_max_steps = (n_steps >= max_steps) and (exc_type is None)
    llm_total_s = sum(float(s.get("llm_s", 0) or 0) for s in timing)
    bash_total_s = sum(float(s.get("bash_s", 0) or 0) for s in timing)
    prompt_tokens_total = sum(int(s.get("prompt_tokens", 0) or 0) for s in timing)
    completion_tokens_total = sum(int(s.get("completion_tokens", 0) or 0) for s in timing)
    peak_prompt_tokens = max((int(s.get("prompt_tokens", 0) or 0) for s in timing), default=0)

    # Tool histogram + submission detection from cmd previews.
    # Two submission conventions:
    #   - TassieAgent: literal SUBMIT_MARKER (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
    #   - SWE-agent  : standalone `submit` token at end of bash command chain
    # We DO NOT scan agent/trajectory.json for the TassieAgent marker because the
    # system prompt itself documents the marker, which would false-positive every
    # trial.
    tool_hist: collections.Counter[str] = collections.Counter()
    submitted = False
    for s in timing:
        cmd = s.get("cmd") or ""
        tool_hist[_bash_verb(cmd)] += 1
        if SUBMIT_MARKER in cmd:
            submitted = True
        elif _SUBMIT_RE_SWE.search(cmd):
            submitted = True

    final_text = _final_assistant_text(trajectory)

    # Verifier
    ctrf = _safe_load_json(trial_dir / "verifier" / "ctrf.json")
    n_tests, n_tests_failed, failed_names, failed_excerpts = _ctrf_failures(ctrf)
    n_tests_passed = max(0, n_tests - n_tests_failed)

    # Task metadata
    task_dir = task_index.get(task_name)
    meta, instr = _read_task_meta(task_dir)
    md = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    agent_meta = meta.get("agent", {}) if isinstance(meta, dict) else {}
    env_meta = meta.get("environment", {}) if isinstance(meta, dict) else {}

    failure_mode = _classify_failure(
        passed=passed,
        exception_type=exc_type,
        submitted=submitted,
        hit_max_steps=hit_max_steps,
        n_tests=n_tests,
        n_tests_failed=n_tests_failed,
        failed_test_excerpts=failed_excerpts,
    )

    agent_info = result.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}

    return TrialRecord(
        trial_name=result.get("trial_name") or trial_dir.name,
        task_name=task_name,
        reward=reward,
        passed=passed,
        exception_type=exc_type,
        failure_mode=failure_mode,
        n_steps=n_steps,
        max_steps=max_steps,
        hit_max_steps=hit_max_steps,
        submitted=submitted,
        final_assistant_text=final_text,
        tool_hist=dict(tool_hist),
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_total=completion_tokens_total,
        peak_prompt_tokens=peak_prompt_tokens,
        llm_total_s=round(llm_total_s, 2),
        bash_total_s=round(bash_total_s, 2),
        agent_execution_s=round(agent_execution_s, 2),
        verifier_s=round(verifier_s, 2),
        total_s=round(total_s, 2),
        n_tests=n_tests,
        n_tests_passed=n_tests_passed,
        n_tests_failed=n_tests_failed,
        failed_test_names=failed_names,
        failed_test_excerpts=failed_excerpts,
        task_difficulty=md.get("difficulty"),
        task_category=md.get("category"),
        task_tags=list(md.get("tags") or []),
        task_agent_timeout_s=float(agent_meta.get("timeout_sec")) if agent_meta.get("timeout_sec") is not None else None,
        task_env_image=env_meta.get("docker_image"),
        instruction_excerpt=instr,
        model_name=model_info.get("name"),
        agent_name=agent_info.get("name"),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "p25": round(s[max(0, int(0.25 * len(s)) - 1)], 2),
        "p75": round(s[min(len(s) - 1, int(0.75 * len(s)))], 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _group_stats(records: list[TrialRecord], key: str) -> dict[str, dict]:
    groups: dict[str, list[TrialRecord]] = collections.defaultdict(list)
    for r in records:
        v = getattr(r, key) if hasattr(r, key) else None
        if isinstance(v, list):
            for item in v:
                groups[str(item)].append(r)
        else:
            groups[str(v)].append(r)
    out: dict[str, dict] = {}
    for k, recs in sorted(groups.items()):
        passed = sum(1 for r in recs if r.passed)
        out[k] = {
            "n": len(recs),
            "n_pass": passed,
            "pass_rate": round(passed / len(recs), 3) if recs else 0.0,
            "mean_turns": round(statistics.fmean([r.n_steps for r in recs]), 2),
        }
    return out


def aggregate(records: list[TrialRecord], label: str) -> dict[str, Any]:
    n = len(records)
    n_pass = sum(1 for r in records if r.passed)
    n_fail = n - n_pass
    n_timeout = sum(1 for r in records if r.exception_type == "AgentTimeoutError")
    n_other_err = sum(1 for r in records if r.exception_type and r.exception_type != "AgentTimeoutError")

    passed = [r for r in records if r.passed]
    failed = [r for r in records if not r.passed]

    failure_mode_counts = collections.Counter(r.failure_mode for r in records)

    # Tool histograms: pass vs fail vs all.
    def merge_hist(recs: list[TrialRecord]) -> dict[str, int]:
        c: collections.Counter[str] = collections.Counter()
        for r in recs:
            c.update(r.tool_hist)
        return dict(c.most_common())

    return {
        "label": label,
        "n_trials": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "n_agent_timeout": n_timeout,
        "n_other_error": n_other_err,
        "failure_modes": dict(failure_mode_counts.most_common()),
        # Turns
        "turns_all": _stats([r.n_steps for r in records]),
        "turns_pass": _stats([r.n_steps for r in passed]),
        "turns_fail": _stats([r.n_steps for r in failed]),
        # Tokens
        "prompt_tokens_all": _stats([r.prompt_tokens_total for r in records]),
        "completion_tokens_all": _stats([r.completion_tokens_total for r in records]),
        "peak_prompt_tokens_all": _stats([r.peak_prompt_tokens for r in records]),
        # Wall-clock
        "agent_execution_s_all": _stats([r.agent_execution_s for r in records]),
        "agent_execution_s_pass": _stats([r.agent_execution_s for r in passed]),
        "agent_execution_s_fail": _stats([r.agent_execution_s for r in failed]),
        # Submission discipline
        "n_submitted": sum(1 for r in records if r.submitted),
        "n_hit_max_steps": sum(1 for r in records if r.hit_max_steps),
        "n_submitted_but_failed": sum(1 for r in records if r.submitted and not r.passed),
        # Tools
        "tool_hist_all": merge_hist(records),
        "tool_hist_pass": merge_hist(passed),
        "tool_hist_fail": merge_hist(failed),
        # Groupings (overview only — full per-trial in JSONL)
        "by_difficulty": _group_stats(records, "task_difficulty"),
        "by_category": _group_stats(records, "task_category"),
        "by_failure_mode": {
            mode: {"n": cnt, "trials": sorted([r.trial_name for r in records if r.failure_mode == mode])}
            for mode, cnt in failure_mode_counts.most_common()
        },
    }


# ---------------------------------------------------------------------------
# Failure narrative (markdown)
# ---------------------------------------------------------------------------


def render_failures_md(records: list[TrialRecord], label: str) -> str:
    failed = sorted([r for r in records if not r.passed], key=lambda r: (r.failure_mode, r.task_name))
    lines: list[str] = []
    lines.append(f"# Failure-by-failure detail — {label}")
    lines.append("")
    lines.append(
        "Auto-generated. One block per failed trial, grouped by failure mode. "
        "Excerpts truncated. See `per_trial.jsonl` for raw fields."
    )
    lines.append("")

    by_mode: dict[str, list[TrialRecord]] = collections.defaultdict(list)
    for r in failed:
        by_mode[r.failure_mode].append(r)

    for mode in sorted(by_mode):
        recs = by_mode[mode]
        lines.append(f"## `{mode}` — {len(recs)} trials")
        lines.append("")
        for r in recs:
            lines.append(f"### {r.task_name} ({r.trial_name})")
            lines.append("")
            meta_bits: list[str] = []
            if r.task_difficulty:
                meta_bits.append(f"difficulty=`{r.task_difficulty}`")
            if r.task_category:
                meta_bits.append(f"category=`{r.task_category}`")
            if r.task_tags:
                meta_bits.append(f"tags=`{', '.join(r.task_tags[:6])}`")
            meta_bits.append(f"steps={r.n_steps}/{r.max_steps}")
            meta_bits.append(f"submitted={r.submitted}")
            meta_bits.append(f"hit_max_steps={r.hit_max_steps}")
            meta_bits.append(f"agent_s={r.agent_execution_s}")
            if r.task_agent_timeout_s is not None:
                meta_bits.append(f"task_timeout_s={r.task_agent_timeout_s}")
            lines.append("- " + " | ".join(meta_bits))
            if r.exception_type:
                lines.append(f"- **Exception**: `{r.exception_type}`")
            if r.instruction_excerpt:
                instr = textwrap.shorten(
                    r.instruction_excerpt.replace("\n", " "), width=400, placeholder=" …"
                )
                lines.append(f"- **Instruction (excerpt)**: {instr}")
            if r.failed_test_names:
                lines.append("- **Failed tests**:")
                for name, ex in zip(r.failed_test_names[:3], r.failed_test_excerpts[:3]):
                    ex_short = ex.replace("\n", " ⏎ ")[:300]
                    lines.append(f"  - `{name}` — {ex_short}")
            if r.final_assistant_text:
                tail = textwrap.shorten(
                    r.final_assistant_text.replace("\n", " "), width=400, placeholder=" …"
                )
                lines.append(f"- **Last assistant turn**: {tail}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-dir", required=True, type=Path, help="Path to harbor job dir (jobs/<name>)")
    p.add_argument(
        "--harbor-cache",
        type=Path,
        default=Path.home() / ".cache" / "harbor",
        help="Harbor task cache root (for task.toml / instruction.md). Default: ~/.cache/harbor",
    )
    p.add_argument("--label", default="", help="Free-form label for this run, e.g. \"TassieAgent + gemini-3-flash-preview\".")
    p.add_argument("--out", required=True, type=Path, help="Output directory; created if missing.")
    args = p.parse_args(argv)

    job_dir: Path = args.job_dir
    if not job_dir.exists():
        print(f"job-dir not found: {job_dir}", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    task_index = _build_task_index(args.harbor_cache)
    print(f"Indexed {len(task_index)} cached tasks from {args.harbor_cache}")

    trials = sorted([d for d in job_dir.iterdir() if d.is_dir()])
    records: list[TrialRecord] = []
    for d in trials:
        if not (d / "result.json").exists():
            continue
        rec = extract_trial(d, task_index)
        if rec:
            records.append(rec)
    print(f"Extracted {len(records)} trial records from {job_dir}")

    label = args.label or job_dir.name

    # 1. per_trial.jsonl
    jsonl_path = out_dir / "per_trial.jsonl"
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=str) + "\n")
    print(f"Wrote {jsonl_path}")

    # 2. summary.json
    summary = aggregate(records, label=label)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {summary_path}")

    # 3. failures.md
    failures_md = render_failures_md(records, label=label)
    failures_path = out_dir / "failures.md"
    failures_path.write_text(failures_md)
    print(f"Wrote {failures_path}")

    # 4. Console one-liner.
    print()
    print(
        f"  pass_rate={summary['pass_rate']:.3f}  "
        f"({summary['n_pass']}/{summary['n_trials']})  "
        f"timeouts={summary['n_agent_timeout']}  "
        f"other_err={summary['n_other_error']}  "
        f"submitted_but_failed={summary['n_submitted_but_failed']}  "
        f"hit_max_steps={summary['n_hit_max_steps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
