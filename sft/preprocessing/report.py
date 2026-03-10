"""Rich terminal reporting for the conversion pipeline.

Prints a nicely formatted summary after conversion completes:
  - Per-source overview table (input / kept / dropped / yield %)
  - Drop-reason breakdown per source
  - JSON extraction strategy distribution
  - Warning flag counts
  - Qualitative examples (a sampled converted trace shown in condensed form)
"""

from __future__ import annotations

import io
import json
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI helpers (no external dependency)
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]], *, align: list[str] | None = None) -> str:
    """Render a simple ASCII table.  *align* is ``'l'`` or ``'r'`` per column."""
    if not rows:
        return "  (no data)\n"
    ncols = len(headers)
    if align is None:
        align = ["l"] * ncols
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str], sep: str = " | ") -> str:
        parts = []
        for i, cell in enumerate(cells):
            if align[i] == "r":
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  " + sep.join(parts)

    lines = [
        _fmt_row(headers),
        "  " + "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    width = 72
    print()
    print(_c("=" * width, _BOLD))
    print(_c(f"  {title}", _BOLD))
    print(_c("=" * width, _BOLD))
    print()


def _print_section(title: str) -> None:
    print(_c(f"  --- {title} ---", _CYAN))
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_report(
    report: dict,
    examples: list[dict] | None = None,
    dropped_examples: list[dict] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Print a rich terminal summary of the conversion report.

    Parameters
    ----------
    report : dict
        The report dict returned by :func:`pipeline.run_pipeline`.
    examples : list[dict] | None
        Optional list of sampled converted traces to display as qualitative
        examples.  Each dict should have ``source``, ``messages``, ``metadata``.
    dropped_examples : list[dict] | None
        Optional list of sampled *dropped* traces for diagnosis.  Each dict
        has ``source``, ``drop_reason``, ``trial_name``, ``messages``,
        ``warnings``, ``num_raw_messages``, ``raw_first_user``,
        ``raw_last_assistant``.
    save_path : str | Path | None
        If given, also write the report (ANSI codes stripped) to this file.
    """
    # Tee stdout to a buffer so we can save a plain-text copy
    _original_stdout = sys.stdout
    _capture = io.StringIO()

    class _Tee:
        def write(self, s: str) -> int:
            _original_stdout.write(s)
            _capture.write(s)
            return len(s)
        def flush(self) -> None:
            _original_stdout.flush()

    sys.stdout = _Tee()  # type: ignore[assignment]
    try:
        _print_report_body(report, examples, dropped_examples)
    finally:
        sys.stdout = _original_stdout

    if save_path is not None:
        ansi_re = re.compile(r"\033\[[0-9;]*m")
        plain = ansi_re.sub("", _capture.getvalue())
        Path(save_path).write_text(plain)


def _print_report_body(
    report: dict,
    examples: list[dict] | None,
    dropped_examples: list[dict] | None,
) -> None:
    """Inner report rendering (all print() calls go through the tee)."""
    _print_header("Terminus-2 -> SWE-Agent Conversion Report")

    agg = report["aggregate"]
    per_source = report["per_source"]

    # ---- Overview table ----
    _print_section("Per-Source Overview")

    headers = ["Source", "Input", "Kept", "Dropped", "Yield %", "Time"]
    align = ["l", "r", "r", "r", "r", "r"]
    rows = []
    for label, s in per_source.items():
        inp = s["input_traces"]
        out = s["output_traces"]
        drop = s["dropped"]
        pct = f"{out / inp * 100:.1f}%" if inp > 0 else "-"
        elapsed = f"{s['elapsed_seconds']:.0f}s"
        short = _short_label(label)
        rows.append([short, str(inp), str(out), str(drop), pct, elapsed])

    # Totals row
    total_in = agg["total_input_traces"]
    total_out = agg["total_output_traces"]
    total_drop = agg["total_dropped"]
    total_pct = f"{total_out / total_in * 100:.1f}%" if total_in > 0 else "-"
    total_time = f"{sum(s['elapsed_seconds'] for s in per_source.values()):.0f}s"
    rows.append(["TOTAL", str(total_in), str(total_out), str(total_drop), total_pct, total_time])

    print(_table(headers, rows, align=align))

    # ---- Drop reasons ----
    _print_section("Drop Reasons (aggregate)")

    if agg["drop_reasons"]:
        dr_headers = ["Reason", "Count", "% of dropped"]
        dr_rows = []
        for reason, count in sorted(agg["drop_reasons"].items(), key=lambda x: -x[1]):
            pct = f"{count / total_drop * 100:.1f}%" if total_drop > 0 else "-"
            dr_rows.append([reason, str(count), pct])
        print(_table(dr_headers, dr_rows, align=["l", "r", "r"]))
    else:
        print("  No traces dropped.\n")

    # ---- Per-source drop breakdown ----
    sources_with_drops = {k: v for k, v in per_source.items() if v["dropped"] > 0}
    if len(sources_with_drops) > 1:
        _print_section("Drop Reasons (per source)")
        for label, s in sources_with_drops.items():
            if not s["drop_reasons"]:
                continue
            print(f"  {_c(_short_label(label), _BOLD)}")
            for reason, count in sorted(s["drop_reasons"].items(), key=lambda x: -x[1]):
                bar = _bar(count, s["dropped"], width=20)
                print(f"    {reason:<30s} {count:>6d}  {bar}")
            print()

    # ---- JSON extraction strategies ----
    _print_section("JSON Extraction Strategy Distribution")

    strat = agg["json_strategy_distribution"]
    strat_total = sum(strat.values())
    strat_labels = {
        1: "Strategy 1 (direct parse)",
        2: "Strategy 2 (brace-match)",
        3: "Strategy 3 (error-fix)",
        0: "Failed (all strategies)",
    }
    if strat_total > 0:
        for key in [1, 2, 3, 0]:
            count = strat.get(key, 0)
            pct = count / strat_total * 100
            bar = _bar(count, strat_total, width=30)
            color = _GREEN if key == 1 else (_YELLOW if key in (2, 3) else _RED)
            print(f"  {strat_labels[key]:<35s} {_c(f'{count:>7d}', color)}  ({pct:5.1f}%)  {bar}")
        print()

    # ---- Turn count distribution ----
    turn_stats = agg.get("per_source_turn_stats", {})
    if turn_stats:
        _print_section("Turn Count Distribution (kept traces)")
        th = ["Source", "Min", "Median", "Mean", "P95", "Max"]
        tr = []
        for lbl, ts in turn_stats.items():
            if ts:
                tr.append([
                    _short_label(lbl),
                    str(ts.get("min", "-")),
                    str(ts.get("median", "-")),
                    str(ts.get("mean", "-")),
                    str(ts.get("p95", "-")),
                    str(ts.get("max", "-")),
                ])
        if tr:
            print(_table(th, tr, align=["l", "r", "r", "r", "r", "r"]))

    # ---- Warnings ----
    _print_section("Warning Flags (kept traces)")

    if agg["warning_counts"]:
        wh = ["Flag", "Count"]
        wr = [[flag, str(count)] for flag, count in
              sorted(agg["warning_counts"].items(), key=lambda x: -x[1])]
        print(_table(wh, wr, align=["l", "r"]))
    else:
        print("  No warnings.\n")

    # ---- Qualitative examples (kept) ----
    if examples:
        _print_section(f"Qualitative Examples ({len(examples)} sampled kept traces)")
        for idx, ex in enumerate(examples):
            _print_example(idx + 1, ex)

    # ---- Dropped trace examples ----
    if dropped_examples:
        _print_section(f"Dropped Trace Examples ({len(dropped_examples)} sampled)")
        _print_dropped_examples(dropped_examples)

    # ---- Footer ----
    print(_c("-" * 72, _DIM))
    print(
        _c(f"  Summary: {total_out:,} traces kept out of {total_in:,} "
           f"({total_out / total_in * 100:.1f}% yield)", _BOLD)
        if total_in > 0 else ""
    )
    print()


def _print_example(num: int, ex: dict) -> None:
    """Print a condensed view of a single converted trace."""
    source = ex.get("source", "?")
    meta = ex.get("metadata", {})
    messages = ex.get("messages", [])

    print(f"  {_c(f'Example {num}', _BOLD)}  source={_short_label(source)}  "
          f"trial={meta.get('trial_name', '?')}  turns={meta.get('num_turns', '?')}")
    print()

    for msg in messages:
        role = msg.get("role", "?")

        if role == "system":
            print(f"    {_c('[system]', _DIM)}  {_truncate(msg.get('content', ''), 80)}")

        elif role == "user":
            content = msg.get("content", "")
            print(f"    {_c('[user]', _CYAN)}    {_truncate(content, 100)}")

        elif role == "assistant":
            reasoning = msg.get("reasoning_content", "")
            tool_calls = msg.get("tool_calls", [])
            if reasoning:
                print(f"    {_c('[think]', _YELLOW)}   {_truncate(reasoning, 100)}")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", {})
                    if name == "bash":
                        cmd = args.get("command", "")
                        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
                            print(f"    {_c('[submit]', _GREEN)}")
                        else:
                            print(f"    {_c('[bash]', _GREEN)}    {_truncate(cmd, 100)}")
            elif not reasoning:
                content = msg.get("content", "")
                if content:
                    print(f"    {_c('[asst]', _DIM)}    {_truncate(content, 100)}")

        elif role == "tool":
            content = msg.get("content", "")
            print(f"    {_c('[output]', _DIM)}  {_truncate(content, 100)}")

    print()
    print(f"    {_c('---', _DIM)}")
    print()


def _print_dropped_examples(dropped_examples: list[dict]) -> None:
    """Print dropped trace examples grouped by drop reason."""
    by_reason: dict[str, list[dict]] = {}
    for ex in dropped_examples:
        by_reason.setdefault(ex["drop_reason"], []).append(ex)

    for reason, examples in sorted(by_reason.items()):
        print(f"  {_c(f'Drop reason: {reason}', _RED + _BOLD)}  ({len(examples)} sampled)")
        print()
        for ex in examples:
            source = _short_label(ex.get("source", "?"))
            trial = ex.get("trial_name", "?")
            n_raw = ex.get("num_raw_messages", 0)
            n_converted = len(ex.get("messages", []))
            warnings = ex.get("warnings", [])

            print(f"    {_c('source', _DIM)}={source}  "
                  f"{_c('trial', _DIM)}={trial}  "
                  f"{_c('raw_msgs', _DIM)}={n_raw}  "
                  f"{_c('converted_msgs', _DIM)}={n_converted}")

            if warnings:
                print(f"    {_c('warnings:', _YELLOW)}  {', '.join(warnings[:5])}"
                      f"{'...' if len(warnings) > 5 else ''}")

            first_user = ex.get("raw_first_user", "")
            if first_user:
                print(f"    {_c('[first user]', _CYAN)}  {_truncate(first_user, 120)}")

            last_asst = ex.get("raw_last_assistant", "")
            if last_asst:
                print(f"    {_c('[last asst]', _YELLOW)}  {_truncate(last_asst, 120)}")

            if n_converted > 0:
                msgs = ex.get("messages", [])
                last_msg = msgs[-1]
                role = last_msg.get("role", "?")
                content = last_msg.get("content", "")
                tool_calls = last_msg.get("tool_calls", [])
                if tool_calls:
                    fn = tool_calls[0].get("function", {})
                    cmd = fn.get("arguments", {}).get("command", "")
                    print(f"    {_c('[last converted: ' + role + ']', _DIM)}  "
                          f"bash: {_truncate(cmd, 100)}")
                elif content:
                    print(f"    {_c('[last converted: ' + role + ']', _DIM)}  "
                          f"{_truncate(content, 100)}")

            print()
        print(f"    {_c('---', _DIM)}")
        print()


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _short_label(label: str) -> str:
    """Shorten ``nvidia/Nemotron-Terminal-Corpus/skill_based_easy`` to
    ``Nemotron/skill_based_easy`` for display."""
    parts = label.split("/")
    if len(parts) >= 3:
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) == 2:
        return parts[-1]
    return label


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " \\n ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _bar(value: int, total: int, *, width: int = 20) -> str:
    if total == 0:
        return ""
    filled = round(value / total * width)
    return _c("█" * filled, _GREEN) + _c("░" * (width - filled), _DIM)
