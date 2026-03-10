"""Quality filters for converted traces.

Each filter returns a simple verdict; the pipeline applies them in bulk via
``datasets.Dataset.filter`` / ``datasets.Dataset.map``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilterVerdict:
    keep: bool = True
    drop_reason: str | None = None
    warning_flags: list[str] = field(default_factory=list)


def apply_mandatory_filters(row: dict) -> FilterVerdict:
    """Filters that cause the trace to be **dropped**."""
    v = FilterVerdict()

    if not row.get("_conversion_ok", False):
        v.keep = False
        v.drop_reason = "conversion_failed"
        return v

    meta = row.get("metadata", {})

    if meta.get("json_extraction_failed", False):
        v.keep = False
        v.drop_reason = "json_extraction_failed"
        return v

    if meta.get("num_turns", 0) < 1:
        v.keep = False
        v.drop_reason = "too_few_turns"
        return v

    if not meta.get("has_task_complete", False):
        v.keep = False
        v.drop_reason = "no_task_complete"
        return v

    return v


def apply_warning_flags(row: dict) -> list[str]:
    """Flags that **do not** drop the trace but are recorded for review."""
    flags: list[str] = []
    warnings = row.get("warnings", [])

    for w in warnings:
        if "TASK_DELIM not found" in w:
            flags.append("task_delim_missing")
        if "Prose outside JSON" in w:
            flags.append("prose_outside_json")
        if "has commands but no following user message" in w:
            flags.append("missing_tool_result")

    return flags


def apply_optional_filters(
    row: dict,
    *,
    max_turns: int = 999,
    drop_trivial_only: bool = False,
) -> FilterVerdict:
    """Configurable quality heuristics.  Each can be toggled independently."""
    v = FilterVerdict()
    meta = row.get("metadata", {})

    if meta.get("num_turns", 0) > max_turns:
        v.keep = False
        v.drop_reason = f"exceeds_{max_turns}_turns"
        return v

    if drop_trivial_only:
        messages = row.get("messages", [])
        commands = _extract_all_commands(messages)
        if commands and all(_is_trivial(c) for c in commands):
            v.keep = False
            v.drop_reason = "trivial_only"
            return v

    return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRIVIAL_PREFIXES = ("echo ", "ls", "pwd", "whoami", "date", "hostname")


def _extract_all_commands(messages: list[dict]) -> list[str]:
    cmds: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name") == "bash":
                cmd = fn.get("arguments", {}).get("command", "")
                if cmd:
                    cmds.append(cmd)
    return cmds


def _is_trivial(command: str) -> bool:
    for line in command.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not any(stripped.startswith(p) for p in _TRIVIAL_PREFIXES):
            return False
    return True
