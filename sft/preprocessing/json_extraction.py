"""Robust JSON extraction from Terminus-2 assistant messages.

Implements a 5-strategy cascade:
  Strategy 1 – direct ``json.loads`` (content is pure JSON).
  Strategy 2 – brace-matching to locate outermost ``{…}`` in mixed prose+JSON.
  Strategy 3 – fix common LLM JSON errors (trailing commas) then retry parse.
  Strategy 4 – strip ``<think>…</think>`` tags then apply strategies 1–3 to
               the text *after* the closing ``</think>``.
  Strategy 5 – extract ``commands``, ``task_complete``, and ``plan`` fields
               from inside a ``<think>`` block via regex when the model merged
               its reasoning prose directly into the JSON body (no opening
               ``{`` / ``"analysis"`` key).
"""

from __future__ import annotations

import json
import re

_TERMINUS_KEYS = frozenset({"analysis", "plan", "commands", "task_complete"})
_JSON_DECODER = json.JSONDecoder()


def extract_json_from_content(content: str) -> tuple[dict | None, str, int]:
    """Extract a JSON object from an assistant message.

    Returns
    -------
    parsed : dict | None
        The parsed JSON dict, or ``None`` if all strategies fail.
    prose : str
        Any text surrounding the JSON blob (empty when content is pure JSON).
        ``<think>`` tags are **always stripped** so that downstream reasoning
        fields are not double-wrapped.
    strategy : int
        Which strategy succeeded (1–5) or 0 on failure.
    """
    content = content.strip()

    # ------------------------------------------------------------------
    # Fast path: no <think> tags → original strategies 1-3
    # ------------------------------------------------------------------
    if not content.startswith("<think>"):
        return _extract_from_text(content)

    # ------------------------------------------------------------------
    # <think>-wrapped content
    # ------------------------------------------------------------------
    think_end = content.find("</think>")
    if think_end < 0:
        inner = content[7:].strip()
        return _extract_think_body(inner)

    think_prose = content[7:think_end].strip()
    after_think = content[think_end + 8 :].strip()

    # Strategy 4: JSON lives *after* </think>
    if after_think:
        parsed, _extra_prose, strategy = _extract_from_text(after_think)
        if parsed is not None:
            return parsed, think_prose, strategy

    # Strategy 5: JSON fields embedded inside <think> (no outer { )
    return _extract_think_body(think_prose)


# ---------------------------------------------------------------------------
# Original strategies (1-3) on plain text
# ---------------------------------------------------------------------------

def _extract_from_text(text: str) -> tuple[dict | None, str, int]:
    """Strategies 1-3 applied to *text* (which must NOT contain think tags)."""
    text = text.strip()

    # Strategy 1: direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "", 1
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first { and parse with C-speed raw_decode
    start_idx = text.find("{")
    if start_idx == -1:
        return None, text, 0

    try:
        parsed, raw_end = _JSON_DECODER.raw_decode(text, start_idx)
        if isinstance(parsed, dict):
            prose = _surrounding_prose(text, start_idx, raw_end - 1)
            return parsed, prose, 2
    except json.JSONDecodeError:
        pass

    # Strategy 3: fall back to manual brace matching + common error fixes
    # (handles trailing commas and other LLM JSON mistakes that raw_decode rejects)
    end_idx = _find_matching_brace(text, start_idx)
    if end_idx is None:
        return None, text, 0

    json_str = text[start_idx : end_idx + 1]
    prose = _surrounding_prose(text, start_idx, end_idx)

    fixed = _fix_common_json_errors(json_str)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed, prose, 3
    except json.JSONDecodeError:
        pass

    return None, text, 0


# ---------------------------------------------------------------------------
# Strategy 5: regex field extraction from merged prose+JSON
# ---------------------------------------------------------------------------

def _extract_think_body(think_text: str) -> tuple[dict | None, str, int]:
    """Try to recover a Terminus-2 JSON dict from *think_text* where the
    model merged its reasoning prose directly into the JSON value body.

    The typical layout is::

        [long analysis prose]…some escaped quote\\",
          "plan": "…",
          "commands": [{…}, …],
          "task_complete": false
        }

    There is no opening ``{`` or ``"analysis"`` key.  We reconstruct the
    dict by extracting ``commands``, ``task_complete``, and ``plan`` via
    targeted parsing and use the leading prose as the analysis value.
    """
    if '"task_complete"' not in think_text and '"commands"' not in think_text:
        return None, think_text, 0

    result: dict = {}

    tc_match = re.search(r'"task_complete"\s*:\s*(true|false)', think_text)
    if tc_match:
        result["task_complete"] = tc_match.group(1) == "true"

    commands = _extract_json_array(think_text, "commands")
    if commands is not None:
        result["commands"] = commands

    plan = _extract_json_string(think_text, "plan")
    if plan is not None:
        result["plan"] = plan

    if not result:
        return None, think_text, 0

    # Everything before the first extracted key is the analysis/reasoning.
    first_key_pos = len(think_text)
    for key in ("plan", "commands", "task_complete"):
        needle = f'"{key}"'
        pos = think_text.find(needle)
        if 0 <= pos < first_key_pos:
            first_key_pos = pos

    prefix = think_text[:first_key_pos]
    # Strip trailing separator left from the missing analysis value close
    prefix = re.sub(r'["\s,]+$', "", prefix)
    result["analysis"] = prefix.strip()

    return result, prefix.strip(), 5


def _extract_json_array(text: str, key: str) -> list | None:
    """Find ``"<key>": [...]`` and parse the array value."""
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\[')
    m = pattern.search(text)
    if m is None:
        return None
    bracket_start = m.end() - 1
    end = _find_matching_bracket(text, bracket_start)
    if end is None:
        return None
    array_str = text[bracket_start : end + 1]
    try:
        return json.loads(array_str)
    except json.JSONDecodeError:
        fixed = _fix_common_json_errors(array_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _extract_json_string(text: str, key: str) -> str | None:
    """Find ``"<key>": "…"`` and return the unescaped string value."""
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*"')
    m = pattern.search(text)
    if m is None:
        return None
    start = m.end()
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            try:
                return text[start:i].encode().decode("unicode_escape", errors="replace")
            except Exception:
                return text[start:i]
        i += 1
    return None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _find_matching_brace(text: str, start: int) -> int | None:
    """Return the index of the ``}`` that closes the ``{`` at *start*."""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _find_matching_bracket(text: str, start: int) -> int | None:
    """Return the index of the ``]`` that closes the ``[`` at *start*."""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def _surrounding_prose(content: str, json_start: int, json_end: int) -> str:
    before = content[:json_start].strip()
    after = content[json_end + 1 :].strip()
    parts = [p for p in (before, after) if p]
    return "\n".join(parts)


def _fix_common_json_errors(json_str: str) -> str:
    """Best-effort fixes for common LLM JSON generation mistakes."""
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json_str
