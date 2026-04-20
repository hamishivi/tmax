"""LLM-based classifier that fits external tasks into OUR taxonomy.

For each external task we need a comparable label in four axes:

* ``domain`` — one of 9 stable values defined in :mod:`rl_data.comparison.core`
* ``skill_type`` — one of ~29 values (the second-level tag used in our dataset)
* ``task_complexity`` — short / moderate / complex
* ``command_complexity`` — bash-only / bash+code / bash+code+services

The classifier writes results back into each task's ``task.json`` under
``classified_domain``, ``classified_task_complexity``,
``classified_command_complexity`` (leaving native fields untouched).  It is
idempotent: tasks that already carry these keys are skipped unless
``--force`` is passed.

Usage:

    python -m rl_data.comparison.taxonomy_classifier \\
        --tasks-dir rl_data/output/tasks_endless_terminals \\
        --model gemini/gemini-3-flash-preview

Runs in parallel via ``rl_data.chat_completion_batch``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_data import chat_completion_batch
from rl_data.comparison.core import (
    COMMAND_COMPLEXITY_ORDER,
    DOMAINS_ORDER,
    SKILL_TYPES_ORDER,
    TASK_COMPLEXITY_ORDER,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a data curator assigning concise taxonomy labels to terminal-style
coding tasks. Return ONLY a single minified JSON object matching the schema.
No prose, no markdown, no code fences."""


USER_PROMPT_TEMPLATE = """Classify the task below into our taxonomy.

# Allowed values
- domain: one of {domains}
- skill_type: one of {skills}  (second-level specialisation; pick the single most representative of what the task asks the agent to do)
- task_complexity: one of {tcs}  (short = 1-3 steps, moderate = 4-8 steps across setup+impl+verify, complex = many subtasks or non-trivial algorithms)
- command_complexity: one of {ccs}  (bash-only = only shell; bash+code = also writes/runs Python/C/etc.; bash+code+services = additionally starts services / daemons / databases)

# Schema
{{
  "domain": "<one value>",
  "skill_type": "<one value>",
  "task_complexity": "<one value>",
  "command_complexity": "<one value>",
  "rationale": "<<=20 words>"
}}

# Task instruction
{description}

# Environment summary (apt + pip + services)
{env_summary}

# Verifier summary (truncated)
{verifier_summary}

Return the JSON object and nothing else.
"""


_APT_RE = re.compile(r"apt(?:-get)?\s+(?:-\S+\s+)*install\s+([^\n;&|]+)")
_PIP_RE = re.compile(r"pip3?\s+install\s+([^\n;&|]+)")
_SERVICE_RE = re.compile(
    r"\b(systemctl|service)\s+(start|restart|enable)\b|"
    r"\b(nginx|redis-server|postgres(?:ql)?|mysqld|mongod)\b"
)


def _extract_env_summary(container_def: str) -> str:
    if not container_def:
        return "(no container.def)"
    apt_pkgs: set[str] = set()
    pip_pkgs: set[str] = set()
    for m in _APT_RE.findall(container_def):
        for tok in m.replace("\\\n", " ").split():
            if tok and not tok.startswith("-") and tok not in ("&&", "||"):
                apt_pkgs.add(tok)
    for m in _PIP_RE.findall(container_def):
        for tok in m.replace("\\\n", " ").split():
            if tok and not tok.startswith("-") and tok not in ("&&", "||"):
                pip_pkgs.add(tok.split("==")[0].split(">")[0].split("<")[0])
    services = sorted({m[2] for m in _SERVICE_RE.findall(container_def) if m[2]})
    parts = []
    if apt_pkgs:
        parts.append(f"apt: {', '.join(sorted(apt_pkgs)[:15])}")
    if pip_pkgs:
        parts.append(f"pip: {', '.join(sorted(pip_pkgs)[:15])}")
    if services:
        parts.append(f"services: {', '.join(services)}")
    return " | ".join(parts) if parts else "(apt-only base image)"


def _truncate(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _build_prompt(task_dir: Path) -> Optional[List[Dict[str, str]]]:
    task_json = task_dir / "task.json"
    if not task_json.exists():
        return None
    try:
        tj = json.loads(task_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    description = tj.get("description", "")
    if not description.strip():
        return None

    cdef = ""
    cdef_path = task_dir / "container.def"
    if cdef_path.exists():
        try:
            cdef = cdef_path.read_text()
        except OSError:
            cdef = ""

    verifier = ""
    v_path = task_dir / "test_final_state.py"
    if v_path.exists():
        try:
            verifier = v_path.read_text()
        except OSError:
            verifier = ""

    user_msg = USER_PROMPT_TEMPLATE.format(
        domains=", ".join(DOMAINS_ORDER),
        skills=", ".join(SKILL_TYPES_ORDER),
        tcs=", ".join(TASK_COMPLEXITY_ORDER),
        ccs=", ".join(COMMAND_COMPLEXITY_ORDER),
        description=_truncate(description, 2000),
        env_summary=_extract_env_summary(cdef),
        verifier_summary=_truncate(verifier, 800),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _parse_response(raw: Any) -> Optional[Dict[str, str]]:
    """Extract the JSON object from a litellm completion response."""
    if raw is None:
        return None
    try:
        text = raw.choices[0].message.content or ""
    except Exception:
        return None
    text = text.strip()
    # Strip leading/trailing code fences if the model added any.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None

    d = str(obj.get("domain", "")).strip()
    s = str(obj.get("skill_type", "")).strip()
    t = str(obj.get("task_complexity", "")).strip()
    c = str(obj.get("command_complexity", "")).strip()

    if d not in DOMAINS_ORDER:
        return None

    # skill_type: case-fuzzy match against our vocab (model sometimes
    # returns "web security" or "systems" lowercase).
    if s not in SKILL_TYPES_ORDER:
        lower_map = {v.lower(): v for v in SKILL_TYPES_ORDER}
        s_norm = lower_map.get(s.lower().strip())
        if s_norm is None:
            # Don't fail the whole task on skill_type alone; mark as unknown
            # so we still preserve the domain + complexities.
            s_norm = ""
        s = s_norm

    if t not in TASK_COMPLEXITY_ORDER:
        tl = t.lower().split()[0] if t else ""
        t = tl if tl in TASK_COMPLEXITY_ORDER else None
        if t is None:
            return None
    if c not in COMMAND_COMPLEXITY_ORDER:
        alias = {
            "bash only": "bash-only",
            "bash and code": "bash+code",
            "bash, code, and system services": "bash+code+services",
            "bash+code+service": "bash+code+services",
        }
        c = alias.get(c.lower(), None)
        if c is None:
            return None

    return {
        "classified_domain": d,
        "classified_skill_type": s,  # may be "" if the model returned a value outside our vocab
        "classified_task_complexity": t,
        "classified_command_complexity": c,
        "classifier_rationale": str(obj.get("rationale", ""))[:200],
    }


def classify_tasks_dir(
    tasks_dir: Path,
    *,
    model: str,
    force: bool = False,
    max_concurrency: int = 32,
    temperature: float = 0.0,
    limit: int = 0,
) -> dict:
    """Classify every task under ``tasks_dir`` in a single batched call.

    Returns a stats dict: ``{total, classified, skipped, failed}``.
    """
    task_dirs = sorted(
        p for p in tasks_dir.iterdir()
        if p.is_dir() and (p.name.startswith("task_") or (p / "task.json").exists())
    )

    # Skip tasks that already carry classified_* keys (unless --force).
    need: List[Path] = []
    skipped = 0
    for td in task_dirs:
        if limit and len(need) >= limit:
            break
        tj_path = td / "task.json"
        if not tj_path.exists():
            skipped += 1
            continue
        if not force:
            try:
                tj = json.loads(tj_path.read_text())
            except (OSError, json.JSONDecodeError):
                tj = {}
            if tj.get("classified_domain") in DOMAINS_ORDER:
                skipped += 1
                continue
        need.append(td)

    logger.info("taxonomy: %d tasks total, %d already classified, %d to classify",
                len(task_dirs), skipped, len(need))

    if not need:
        return {"total": len(task_dirs), "classified": 0, "skipped": skipped, "failed": 0}

    prompts: List[Optional[List[Dict[str, str]]]] = [_build_prompt(td) for td in need]
    valid_idx = [i for i, p in enumerate(prompts) if p is not None]
    if not valid_idx:
        return {"total": len(task_dirs), "classified": 0, "skipped": skipped,
                "failed": len(need)}

    batched = [prompts[i] for i in valid_idx]
    logger.info("taxonomy: dispatching %d LLM calls via %s (max_concurrency=%d)",
                len(batched), model, max_concurrency)

    responses = chat_completion_batch(
        batched,
        model=model,
        temperature=temperature,
        max_tokens=256,
        max_concurrency=max_concurrency,
        show_progress=True,
    )

    classified = 0
    failed = len(need) - len(valid_idx)
    for pos, resp in zip(valid_idx, responses):
        td = need[pos]
        parsed = _parse_response(resp)
        if not parsed:
            failed += 1
            continue
        tj_path = td / "task.json"
        try:
            tj = json.loads(tj_path.read_text())
        except (OSError, json.JSONDecodeError):
            failed += 1
            continue
        tj.update(parsed)
        tj_path.write_text(json.dumps(tj, indent=2))
        classified += 1

    return {
        "total": len(task_dirs),
        "classified": classified,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gemini/gemini-3-flash-preview")
    ap.add_argument("--force", action="store_true",
                    help="Re-classify tasks that already have classified_* fields")
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Classify at most N unclassified tasks (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    stats = classify_tasks_dir(
        args.tasks_dir,
        model=args.model,
        force=args.force,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
        limit=args.limit,
    )
    logger.info("Done. %s", stats)


if __name__ == "__main__":
    main()
