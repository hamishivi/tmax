"""Analyze generated tasks and solutions — summary tables and plots."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Complexity shortening ------------------------------------------------

_TASK_COMPLEXITY_ORDER = ["short", "moderate", "complex"]
_CMD_COMPLEXITY_ORDER = ["bash-only", "bash+code", "bash+code+services"]

_CMD_COMPLEXITY_MAP = {
    "bash-only": "bash-only",
    "bash and code": "bash+code",
    "bash, code, and system services": "bash+code+services",
}


def _shorten_task_complexity(raw: str) -> str:
    m = re.match(r"(short|moderate|complex)\b", raw, re.IGNORECASE)
    return m.group(1).lower() if m else raw


def _shorten_cmd_complexity(raw: str) -> str:
    prefix = raw.split("(")[0].strip()
    return _CMD_COMPLEXITY_MAP.get(prefix, prefix)


def discover_models(tasks_dir: Path) -> List[str]:
    """Return sorted list of model slugs found across all task solution dirs."""
    slugs: set[str] = set()
    for task_path in tasks_dir.iterdir():
        if not task_path.name.startswith("task_"):
            continue
        solutions_dir = task_path / "solutions"
        if not solutions_dir.exists():
            continue
        for f in solutions_dir.glob("*_summary.json"):
            if f.name == "summary.json":
                continue
            slug = f.name.removesuffix("_summary.json")
            slugs.add(slug)
    return sorted(slugs)


_TOK_PER_WORD = 1.3  # rough whitespace-word → BPE-token ratio

# (input_$/1M_tok, output_$/1M_tok) — text only
# slug → (price_in, price_out).  Slug is model_id with "/" replaced by "_".
# Source: https://ai.google.dev/gemini-api/docs/gemini-3
_PRICING: Dict[str, tuple] = {
    "gemini_gemini-3.1-pro-preview":        (2.00, 12.00),
    "gemini_gemini-3-pro-preview":          (2.00, 12.00),
    "gemini_gemini-3-flash-preview":        (0.50,  3.00),
    "gemini_gemini-3.1-flash-lite-preview": (0.25,  1.50),
    "gemini_gemini-2.5-pro":                (1.25, 10.00),
    "gemini_gemini-2.5-flash":              (0.15,  0.60),
    "gemini_gemini-2.0-flash":              (0.10,  0.40),
}

# Task generation always uses 3.1 Pro
_TASK_GEN_MODEL = "gemini_gemini-3.1-pro-preview"
_TASK_GEN_AVG_INPUT_WORDS = 800
_TASK_GEN_AVG_OUTPUT_WORDS = 1200


def _estimate_cost(input_tokens: int, output_tokens: int,
                   model_slug: str) -> float:
    """Return estimated USD cost given token counts and model slug."""
    pi, po = _PRICING.get(model_slug, (1.0, 5.0))
    return (input_tokens * pi + output_tokens * po) / 1e6


def _count_words_in_messages(
    messages: List[Dict[str, Any]],
) -> tuple:
    """Return (input_words, output_words) across all messages."""
    inp, out = 0, 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        wc = len(content.split()) if content else 0
        tc_wc = 0
        for tc in (msg.get("tool_calls") or []):
            args = tc.get("function", {}).get("arguments", "")
            if args:
                tc_wc += len(args.split())
        if role == "assistant":
            out += wc + tc_wc
        else:
            inp += wc + tc_wc
    return inp, out


def _load_summary(summary_path: Path, record: Dict[str, Any]) -> None:
    """Populate *record* with metrics from a model summary file.

    If the summary contains actual ``usage`` data (prompt_tokens,
    completion_tokens) captured from the API, those are used directly.
    Otherwise we fall back to word-count estimation.
    """
    with open(summary_path) as f:
        sol = json.load(f)
    record["num_runs"] = sol.get("num_runs", 0)
    record["num_success"] = sol.get("num_success", 0)
    pass_at_k = sol.get("pass_at_k", {})
    record["pass@1"] = pass_at_k.get("1", pass_at_k.get(1, None))
    record["pass@8"] = pass_at_k.get("8", pass_at_k.get(8, None))

    turns_per_run = []
    input_words_per_run = []
    output_words_per_run = []
    for r in sol.get("results", []):
        msgs = r.get("messages", [])
        n_turns = sum(1 for m in msgs if m.get("role") == "tool")
        turns_per_run.append(n_turns)
        iw, ow = _count_words_in_messages(msgs)
        input_words_per_run.append(iw)
        output_words_per_run.append(ow)

    n = len(turns_per_run)
    total_in_w = sum(input_words_per_run)
    total_out_w = sum(output_words_per_run)

    record["avg_turns"] = sum(turns_per_run) / n if n else 0
    record["total_words"] = total_in_w + total_out_w
    record["avg_words"] = record["total_words"] / n if n else 0

    # Prefer actual API usage when available; fall back to word-count estimate.
    top_usage = sol.get("usage")
    has_actual = (
        top_usage
        and (top_usage.get("prompt_tokens", 0) or top_usage.get("completion_tokens", 0))
    )

    if has_actual:
        record["total_input_tokens"] = top_usage.get("prompt_tokens", 0)
        record["total_output_tokens"] = top_usage.get("completion_tokens", 0)
        record["reasoning_tokens"] = top_usage.get("reasoning_tokens", 0)
        record["tokens_source"] = "api"
    else:
        # Try summing per-result usage (newer format may have per-result but no top-level)
        per_result_in = 0
        per_result_out = 0
        found_any = False
        for r in sol.get("results", []):
            ru = r.get("usage")
            if ru and (ru.get("prompt_tokens", 0) or ru.get("completion_tokens", 0)):
                per_result_in += ru.get("prompt_tokens", 0)
                per_result_out += ru.get("completion_tokens", 0)
                found_any = True
        if found_any:
            record["total_input_tokens"] = per_result_in
            record["total_output_tokens"] = per_result_out
            record["reasoning_tokens"] = sum(
                (r.get("usage") or {}).get("reasoning_tokens", 0)
                for r in sol.get("results", [])
            )
            record["tokens_source"] = "api"
        else:
            record["total_input_tokens"] = int(total_in_w * _TOK_PER_WORD)
            record["total_output_tokens"] = int(total_out_w * _TOK_PER_WORD)
            record["reasoning_tokens"] = 0
            record["tokens_source"] = "estimated"

    record["total_tokens_est"] = (
        record["total_input_tokens"] + record["total_output_tokens"]
    )
    record["avg_tokens_est"] = record["total_tokens_est"] // n if n else 0
    record["has_solutions"] = True


def load_tasks(
    tasks_dir: Path, model_slug: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan *tasks_dir* for task directories and load metadata + solution summaries.

    If *model_slug* is given (e.g. ``"gemini_gemini-3-flash-preview"``), only
    that model's ``<slug>_summary.json`` is loaded.  Otherwise the first
    ``*_summary.json`` found is used (backwards-compatible).
    """
    records = []
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.name.startswith("task_"):
            continue
        task_json = task_path / "task.json"
        if not task_json.exists():
            continue

        with open(task_json) as f:
            task_data = json.load(f)

        raw_tc = task_data.get(
            "task_complexity", task_data.get("complexity", "unknown")
        )
        raw_cc = task_data.get("command_complexity", "unknown")

        record: Dict[str, Any] = {
            "name": task_data.get("name", task_path.name),
            "domain": task_data.get("domain", task_data.get("category", "unknown")),
            "skill_type": task_data.get("skill_type", "unknown"),
            "primitive_skills": task_data.get("primitive_skills", []),
            "task_complexity": _shorten_task_complexity(raw_tc),
            "command_complexity": _shorten_cmd_complexity(raw_cc),
            "scenario": task_data.get("scenario", "unknown"),
            "dir": str(task_path),
        }

        _NO_SOLUTION = dict(
            num_runs=0, num_success=0, avg_turns=0,
            total_words=0, avg_words=0,
            total_input_tokens=0, total_output_tokens=0,
            reasoning_tokens=0,
            total_tokens_est=0, avg_tokens_est=0,
            tokens_source="none",
            has_solutions=False,
            **{"pass@1": None, "pass@8": None},
        )

        solutions_dir = task_path / "solutions"
        if model_slug:
            summary_file = solutions_dir / f"{model_slug}_summary.json"
            if summary_file.exists():
                _load_summary(summary_file, record)
            else:
                record.update(_NO_SOLUTION)
        else:
            summary_files = (
                list(solutions_dir.glob("*_summary.json"))
                if solutions_dir.exists()
                else []
            )
            summary_files = [f for f in summary_files if f.name != "summary.json"]
            if summary_files:
                _load_summary(summary_files[0], record)
            else:
                record.update(_NO_SOLUTION)

        records.append(record)
    return records


def _fmt_count(n: int) -> str:
    """Human-friendly large number: 1234 → '1,234', 1234567 → '1.23M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def print_summary(
    records: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
    max_rows: int = 50,
) -> None:
    """Print model stats banner, optionally per-task rows, and aggregate table."""
    solved = [r for r in records if r["has_solutions"]]

    # ── Model-level stats banner ──────────────────────────────────────
    total_runs = sum(r["num_runs"] for r in records)
    total_words = sum(r["total_words"] for r in records)
    total_in_tok = sum(r["total_input_tokens"] for r in records)
    total_out_tok = sum(r["total_output_tokens"] for r in records)
    total_tokens = total_in_tok + total_out_tok
    avg_words = int(total_words / total_runs) if total_runs else 0
    avg_tokens = int(total_tokens / total_runs) if total_runs else 0

    title = model_name or "all models"
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")
    print(
        f"  Tasks: {len(records)}  "
        f"│  With solutions: {len(solved)}  "
        f"│  Total runs: {total_runs}"
    )
    if solved:
        avg_p1 = (
            sum(r["pass@1"] for r in solved if r["pass@1"] is not None)
            / len(solved)
        )
        avg_p8 = (
            sum(r["pass@8"] for r in solved if r["pass@8"] is not None)
            / len(solved)
        )
        avg_turns = sum(r["avg_turns"] for r in solved) / len(solved)
        print(
            f"  Mean p@1: {avg_p1:.2f}  "
            f"│  Mean p@8: {avg_p8:.2f}  "
            f"│  Avg turns: {avg_turns:.1f}"
        )
    print(
        f"  Total words: {_fmt_count(total_words)}  "
        f"│  Avg words/run: {_fmt_count(avg_words)}  "
        f"│  Avg tokens/run: {_fmt_count(avg_tokens)}"
    )
    # Determine token source label
    sources = {r.get("tokens_source", "none") for r in records if r["has_solutions"]}
    if sources == {"api"}:
        tok_label = "actual"
    elif "api" in sources:
        tok_label = "actual+est"
    else:
        tok_label = "est"

    total_reasoning = sum(r.get("reasoning_tokens", 0) for r in records)
    reasoning_info = ""
    if total_reasoning:
        reasoning_info = f"  (incl. {_fmt_count(total_reasoning)} reasoning)"

    print(
        f"  Input tokens: {_fmt_count(total_in_tok)}  "
        f"│  Output tokens: {_fmt_count(total_out_tok)}  "
        f"│  Total tokens: {_fmt_count(total_tokens)}  ({tok_label})"
    )
    if reasoning_info:
        print(f"  {reasoning_info}")

    # Cost estimation
    slug = model_slug
    if slug and slug in _PRICING:
        pi, po = _PRICING[slug]
        sol_cost = _estimate_cost(total_in_tok, total_out_tok, slug)

        # Task generation cost (always 3.1 Pro)
        n_tasks = len(records)
        tg_in = int(n_tasks * _TASK_GEN_AVG_INPUT_WORDS * _TOK_PER_WORD)
        tg_out = int(n_tasks * _TASK_GEN_AVG_OUTPUT_WORDS * _TOK_PER_WORD)
        tg_cost = _estimate_cost(tg_in, tg_out, _TASK_GEN_MODEL)

        cost_per_run = sol_cost / total_runs if total_runs else 0
        print(
            f"  Solution cost (est): ${sol_cost:,.2f}  "
            f"(${pi:.2f}/${po:.2f} per 1M tok)  "
            f"│  $/run: ${cost_per_run:,.4f}"
        )
        print(
            f"  Task gen cost (est): ${tg_cost:,.2f}  "
            f"(3.1-pro, {n_tasks} tasks)  "
            f"│  Total pipeline: ${tg_cost + sol_cost:,.2f}"
        )
    elif slug:
        print(f"  Cost: pricing not available for {slug}")

    print(f"{'─'*80}")

    # ── Per-task rows (skip if too many) ──────────────────────────────
    show_rows = max_rows == 0 or len(records) <= max_rows
    if show_rows:
        header = (
            f"{'Task':<30} {'Domain':<24} {'Skill Type':<20} "
            f"{'Task Cplx':<12} {'Cmd Cplx':<20} "
            f"{'Runs':>5} {'Pass':>5} "
            f"{'p@1':>6} {'p@8':>6} {'Turns':>6}"
        )
        print(header)
        print("-" * len(header))
        for r in records:
            p1 = f"{r['pass@1']:.2f}" if r["pass@1"] is not None else "-"
            p8 = f"{r['pass@8']:.2f}" if r["pass@8"] is not None else "-"
            turns = (
                f"{r['avg_turns']:>6.1f}" if r["has_solutions"] else f"{'-':>6}"
            )
            print(
                f"{r['name']:<30} {r['domain']:<24} {r['skill_type']:<20} "
                f"{r['task_complexity']:<12} {r['command_complexity']:<20} "
                f"{r['num_runs']:>5} {r['num_success']:>5} "
                f"{p1:>6} {p8:>6} {turns}"
            )
        print()
    else:
        print(
            f"  ({len(records)} tasks — per-task rows hidden; "
            f"use --max-rows 0 to show all)\n"
        )

    # ── Aggregate breakdown ───────────────────────────────────────────
    if not solved:
        return
    _print_aggregate(solved, "domain", "Domain", model_slug=slug)
    _print_aggregate(solved, "task_complexity", "Task Complexity",
                     key_order=_TASK_COMPLEXITY_ORDER, model_slug=slug)
    _print_aggregate(solved, "command_complexity", "Cmd Complexity",
                     key_order=_CMD_COMPLEXITY_ORDER, model_slug=slug)


def _print_aggregate(
    solved: List[Dict[str, Any]],
    field: str,
    label: str,
    key_order: Optional[List[str]] = None,
    model_slug: Optional[str] = None,
) -> None:
    """Print a small aggregate table grouped by *field*."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in solved:
        buckets[r[field]].append(r)

    keys = key_order if key_order else sorted(buckets.keys())
    has_cost = model_slug and model_slug in _PRICING

    hdr = (
        f"  {'':2}{label:<24} {'n':>4} {'p@1':>6} {'p@8':>6} "
        f"{'Turns':>6} {'Tokens':>10}"
    )
    if has_cost:
        hdr += f" {'Cost':>10}"
    print(hdr)
    print(f"  {'':2}{'-'*(len(hdr)-4)}")
    for k in keys:
        recs = buckets.get(k, [])
        if not recs:
            line = (f"  {'':2}{k:<24} {'0':>4} {'-':>6} {'-':>6} "
                    f"{'-':>6} {'-':>10}")
            if has_cost:
                line += f" {'-':>10}"
            print(line)
            continue
        n = len(recs)
        mp1 = sum(r["pass@1"] for r in recs if r["pass@1"] is not None) / n
        mp8 = sum(r["pass@8"] for r in recs if r["pass@8"] is not None) / n
        mt = sum(r["avg_turns"] for r in recs) / n
        ti = sum(r["total_input_tokens"] for r in recs)
        to = sum(r["total_output_tokens"] for r in recs)
        tt = ti + to
        line = (
            f"  {'':2}{k:<24} {n:>4} {mp1:>6.2f} {mp8:>6.2f} "
            f"{mt:>6.1f} {_fmt_count(tt):>10}"
        )
        if has_cost:
            c = _estimate_cost(ti, to, model_slug)
            line += f" {'${:,.2f}'.format(c):>10}"
        print(line)
    print()


def plot_distributions(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate pie charts for all metadata axes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = [
        ("domain", "Domain Distribution", "dist_domain.png"),
        ("skill_type", "Skill Type Distribution", "dist_skill_type.png"),
        ("task_complexity", "Task Complexity Distribution", "dist_task_complexity.png"),
        (
            "command_complexity",
            "Command Complexity Distribution",
            "dist_command_complexity.png",
        ),
        ("scenario", "Scenario Distribution", "dist_scenario.png"),
    ]

    for field, title, fname in axes:
        counts = Counter(r[field] for r in records)
        labels = list(counts.keys())
        sizes = list(counts.values())

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, _texts, _autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.85,
            textprops={"fontsize": 9},
        )
        ax.legend(
            wedges,
            [f"{lb} ({ct})" for lb, ct in zip(labels, sizes)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


def _bar_chart(
    records: List[Dict[str, Any]],
    field: str,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    color: str = "steelblue",
    expected_keys: Optional[List[str]] = None,
) -> None:
    """Helper: grouped bar chart of *metric* averaged by *field*.

    If *expected_keys* is given, all listed categories are shown (in that
    order) even when no data exists for some of them.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if val is not None:
            buckets[r[field]].append(val)
    if not buckets and not expected_keys:
        return

    if expected_keys:
        keys = expected_keys
    else:
        keys = sorted(buckets.keys())

    means = [
        (sum(buckets[k]) / len(buckets[k])) if buckets.get(k) else 0
        for k in keys
    ]
    counts = [len(buckets.get(k, [])) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    bars = ax.bar(range(len(keys)), means, color=color)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if metric.startswith("pass"):
        ax.set_ylim(0, 1.05)
    for bar, val, n in zip(bars, means, counts):
        if n == 0:
            label = "n=0"
        elif metric.startswith("pass"):
            label = f"{val:.2f}\n(n={n})"
        else:
            label = f"{val:.1f}\n(n={n})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / fname}")


def plot_quality(
    records: List[Dict[str, Any]],
    out_dir: Path,
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
) -> None:
    """Generate quality analysis plots (bar charts + pass@k curve)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = [r for r in records if r["has_solutions"] and r["pass@1"] is not None]
    if not solved:
        print("  No solution data available for quality plots.")
        return

    tag = f" [{model_name}]" if model_name else ""
    all_domains = sorted({r["domain"] for r in records})

    # -- pass@1 charts --
    _bar_chart(
        solved, "domain", "pass@1", "Mean pass@1",
        f"Pass@1 by Domain{tag}", "quality_pass1_by_domain.png",
        out_dir, color="steelblue", expected_keys=all_domains,
    )
    _bar_chart(
        solved, "task_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Task Complexity{tag}", "quality_pass1_by_task_complexity.png",
        out_dir, color="darkorange", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "command_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Command Complexity{tag}", "quality_pass1_by_command_complexity.png",
        out_dir, color="mediumpurple", expected_keys=_CMD_COMPLEXITY_ORDER,
    )

    # -- pass@8 (pass-at-any) charts --
    max_k_key = "pass@8"
    _bar_chart(
        solved, "domain", max_k_key, "Mean pass@8",
        f"Pass@8 by Domain{tag}", "quality_pass8_by_domain.png",
        out_dir, color="royalblue", expected_keys=all_domains,
    )
    _bar_chart(
        solved, "task_complexity", max_k_key, "Mean pass@8",
        f"Pass@8 by Task Complexity{tag}", "quality_pass8_by_task_complexity.png",
        out_dir, color="coral", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "command_complexity", max_k_key, "Mean pass@8",
        f"Pass@8 by Command Complexity{tag}", "quality_pass8_by_command_complexity.png",
        out_dir, color="orchid", expected_keys=_CMD_COMPLEXITY_ORDER,
    )

    # -- turns charts --
    _bar_chart(
        solved, "task_complexity", "avg_turns", "Avg Turns",
        f"Average Turns by Task Complexity{tag}",
        "quality_turns_by_task_complexity.png",
        out_dir, color="seagreen", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "domain", "avg_turns", "Avg Turns",
        f"Average Turns by Domain{tag}", "quality_turns_by_domain.png",
        out_dir, color="teal", expected_keys=all_domains,
    )

    # --- Pass@k curve (averaged across tasks) ---
    all_pass_at_k: Dict[int, List[float]] = defaultdict(list)
    slug = model_slug
    for r in solved:
        task_dir = Path(r["dir"])
        if slug:
            sf = task_dir / "solutions" / f"{slug}_summary.json"
            if not sf.exists():
                continue
            summary_files = [sf]
        else:
            summary_files = list((task_dir / "solutions").glob("*_summary.json"))
            summary_files = [f for f in summary_files if f.name != "summary.json"]
        if not summary_files:
            continue
        with open(summary_files[0]) as f:
            sol = json.load(f)
        for k_str, v in sol.get("pass_at_k", {}).items():
            all_pass_at_k[int(k_str)].append(v)

    if all_pass_at_k:
        ks = sorted(all_pass_at_k.keys())
        means = [sum(all_pass_at_k[k]) / len(all_pass_at_k[k]) for k in ks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, means, "o-", color="crimson", linewidth=2, markersize=5)
        ax.set_xlabel("k")
        ax.set_ylabel("Mean pass@k")
        ax.set_title(
            f"Pass@k Curve (averaged across tasks){tag}", fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_pass_at_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / 'quality_pass_at_k.png'}")


def _analyze_model(
    tasks_dir: Path,
    plots_base: Path,
    model_slug: str,
    max_rows: int = 50,
) -> None:
    """Run the full per-model analysis (table + quality plots)."""
    display = model_slug.replace("_", "/", 1)

    records = load_tasks(tasks_dir, model_slug=model_slug)

    model_dir = plots_base / model_slug
    print_summary(records, model_name=display, model_slug=model_slug,
                  max_rows=max_rows)

    print(f"Generating quality plots for {display}...")
    plot_quality(records, model_dir, model_name=display, model_slug=model_slug)

    print(f"Done. Model plots saved to {model_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze generated RL tasks and solutions."
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing task_* subdirectories",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: <tasks-dir>/analysis)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to analyze, e.g. 'gemini/gemini-3-flash-preview'. "
            "Omit to auto-discover and analyze all models."
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help=(
            "Max per-task rows to print (default 50). "
            "Set to 0 to show all rows regardless of count."
        ),
    )
    args = ap.parse_args()

    tasks_dir = args.tasks_dir
    plots_dir = args.plots_dir or (tasks_dir / "analysis")

    print(f"Scanning {tasks_dir}...")

    # Distribution plots use all tasks (model-independent)
    all_records = load_tasks(tasks_dir)
    if not all_records:
        print("No tasks found.")
        return

    print("Generating distribution plots...")
    plot_distributions(all_records, plots_dir)

    # Determine which model(s) to analyze
    if args.model:
        slugs = [args.model.replace("/", "_")]
    else:
        slugs = discover_models(tasks_dir)
        if not slugs:
            print("No model summaries found — nothing to analyze.")
            return
        print(f"Discovered {len(slugs)} model(s): {', '.join(slugs)}")

    for slug in slugs:
        _analyze_model(tasks_dir, plots_dir, slug, max_rows=args.max_rows)

    print(f"\nDone. All plots saved under {plots_dir}/")


if __name__ == "__main__":
    main()
