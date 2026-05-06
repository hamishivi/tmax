"""Convert per-task agent-loop trajectories (the `*_summary.json` files
produced by ``rl_data.generate_solutions``) into the SFT parquet format used
by the rest of this preprocessing pipeline.

This is the missing-link converter between the **rl_data** harness output
(one `solutions/<MODEL_TAG>[_<HARNESS>]_summary.json` per task, each
containing N trajectories under ``results``) and the **sft trainer** input
(one parquet per source with ``messages / source / metadata`` columns).

The ``HARNESS`` suffix is OMITTED for the legacy bash harness (so existing
``<MODEL_TAG>_summary.json`` files keep working unchanged) and INCLUDED for
non-bash harnesses (currently just ``vanillux`` →
``<MODEL_TAG>_vanillux_summary.json``). Pass ``--harness vanillux`` when
converting trajectories from a vanillux solve run.

Schema parity is critical: the parquet emitted here must match the existing
``tmax-sft-full-20260409`` configs (Sera / Nemotron / OpenThoughts) so the
trainer in ``sft/scripts/run_sft_*.sh`` ingests it without modification.
The output columns are identical to what `preprocessing.pipeline` produces.

Quick start::

    python -m preprocessing.convert_trajectories \
        --tasks-dir rl_data/output/tasks_skill_tax_20260324_1k \
        --model-tag hosted_vllm_Qwen_Qwen3.5-27B \
        --output-dir sft/output/preprocessing/skill_tax_20260324_1k \
        --name skill_tax_20260324_1k_all

Add ``--filter-success`` to keep only trajectories whose harness verifier
returned True (i.e. ``result.success == True``); useful for the "rejection
sampling" SFT variant alongside the all-trajectories baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

# Keep in sync with rl_data.generator.sample_solutions.SUBMIT_MARKER -- the
# string the agent must echo to declare task complete.
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Per the existing parquet schema (tmax-sft-full-20260409), every message
# struct in the list must carry ALL of these keys, even when empty, so HF
# Datasets infers a single unified type across rows.  Optional ones get
# defaults: empty string for content/reasoning_content, empty list for
# tool_call_ids/tool_calls.
_MESSAGE_KEYS = ("content", "reasoning_content", "role", "tool_call_ids", "tool_calls")

# Same idea for metadata: every row must have the same 13 fields, including
# the json_strategy_counts struct (zero-filled for our path since we don't
# go through the JSON-extraction code in convert.py).
_DEFAULT_JSON_STRATEGY_COUNTS = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}


# OpenAI-style tools schema we expose to the model.  The rl_data harness
# offers exactly one tool to the agent (`bash`); embed its full JSON-Schema
# spec on every row so SFT trainers can drop it into
# ``tokenizer.apply_chat_template(..., tools=json.loads(row["tools"]))``
# without any extra metadata.
#
# Stored as a JSON string column (not a typed struct) because the
# `parameters` sub-object is a dynamic JSON-Schema document and forcing
# it into Arrow types would either freeze the schema or require lossy
# stringification of just that field -- a single string column round-
# trips losslessly via ``json.loads`` and matches the convention used by
# datasets like glaiveai/glaive-function-calling-v2 and
# Salesforce/xlam-function-calling-60k.
_BASH_TOOL: dict[str, Any] = {
    "function": {
        "description": "Execute a bash command. Each command runs in a new subshell.",
        "name": "bash",
        "parameters": {
            "properties": {
                "command": {
                    "description": "The bash command to execute.",
                    "type": "string",
                },
            },
            "required": ["command"],
            "type": "object",
        },
    },
    "type": "function",
}

_DEFAULT_TOOLS_JSON = json.dumps([_BASH_TOOL], ensure_ascii=False)


def _model_tag_to_canonical(tag: str) -> str:
    """`hosted_vllm_Qwen_Qwen3.5-27B` -> `Qwen/Qwen3.5-27B` (and similar).

    The harness writes summaries under `solutions/<MODEL_TAG>_summary.json`
    where MODEL_TAG is litellm's model string with `/` replaced by `_`.
    Reverse that for the metadata's ``source_model`` field, which is the
    canonical HF / API model id.
    """
    if tag.startswith("hosted_vllm_"):
        tag = tag[len("hosted_vllm_"):]
    if "_" in tag:
        org, name = tag.split("_", 1)
        return f"{org}/{name}"
    return tag


def _normalise_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Strip LiteLLM-specific noise from a raw `result.messages[i]` dict and
    return one that matches the tmax-sft parquet schema (5 keys always).

    Input quirks we handle (all observed in the gemini-3-flash + Qwen3.5
    summaries that this converter operates on):

      * Assistant messages carry LiteLLM-side junk: ``function_call``,
        ``images``, ``thinking_blocks``, ``provider_specific_fields`` -- we
        drop all of these.
      * Assistant ``content`` may be ``None`` when there's a tool call; we
        coerce to ``""`` because the parquet schema is non-nullable string.
      * Each ``tool_calls[i]`` carries ``index`` + ``provider_specific_fields``
        that we drop, and ``function.arguments`` is a JSON-encoded **string**
        whereas the parquet schema is ``struct<command: string>``; we
        ``json.loads`` it (with a defensive fallback if the model emits
        garbage JSON).
      * Tool messages have a singular ``tool_call_id`` (string) but the
        parquet schema uses plural ``tool_call_ids`` (list[string]); we
        rewrap as a one-element list to match.
    """
    role = msg.get("role", "")
    content = msg.get("content")
    if content is None:
        content = ""

    out: dict[str, Any] = {
        "content": str(content),
        "reasoning_content": "",
        "role": role,
        "tool_call_ids": [],
        "tool_calls": [],
    }

    if role == "assistant":
        # Pull thinking text into the canonical field if present.  Gemini
        # encodes it as a list of dicts under `thinking_blocks`; we
        # concatenate the `.thinking` text payloads (other variants
        # observed: a top-level `reasoning_content` already in the message).
        if "reasoning_content" in msg and isinstance(msg["reasoning_content"], str):
            out["reasoning_content"] = msg["reasoning_content"]
        else:
            blocks = msg.get("thinking_blocks") or []
            if isinstance(blocks, list) and blocks:
                pieces = []
                for blk in blocks:
                    if not isinstance(blk, dict):
                        continue
                    t = blk.get("thinking") or blk.get("text") or ""
                    if isinstance(t, str) and t:
                        pieces.append(t)
                if pieces:
                    out["reasoning_content"] = "\n\n".join(pieces)

        # Normalise tool calls to the parquet's struct shape.
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            name = func.get("name") or ""
            raw_args = func.get("arguments")
            # `arguments` is a JSON-encoded string for OpenAI-style tool
            # calls; the parquet schema expects struct<command: string>.
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    parsed = {"command": raw_args}
            elif isinstance(raw_args, dict):
                parsed = raw_args
            else:
                parsed = {}

            command = parsed.get("command", "") if isinstance(parsed, dict) else ""
            if not isinstance(command, str):
                # Defensive: if the model emitted a non-string command,
                # JSON-encode it back so the schema stays string-typed.
                command = json.dumps(command, ensure_ascii=False)
            elif not command and isinstance(parsed, dict) and parsed:
                # No `command` key but something else -> dump for traceability.
                command = json.dumps(parsed, ensure_ascii=False)

            out["tool_calls"].append({
                "function": {
                    "arguments": {"command": command},
                    "name": name,
                },
                "id": str(tc.get("id") or ""),
                "type": "function",
            })

    elif role == "tool":
        # Singular tool_call_id -> plural tool_call_ids list.
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            out["tool_call_ids"] = [str(tool_call_id)]
        else:
            ids = msg.get("tool_call_ids") or []
            out["tool_call_ids"] = [str(x) for x in ids]

    return out


def _has_submit(messages: list[dict[str, Any]]) -> bool:
    """True if any assistant tool-call command emits SUBMIT_MARKER.

    We check on the *normalised* messages (post-`_normalise_message`) so the
    `function.arguments.command` field is already a clean string.
    """
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments") or {}
            cmd = args.get("command", "")
            if isinstance(cmd, str) and SUBMIT_MARKER in cmd:
                return True
    return False


def _count_turns(messages: list[dict[str, Any]]) -> int:
    """Number of assistant turns -- matches `turn_index` semantics in
    `convert.py` / `convert_sera.py` (one turn per emitted assistant msg).
    """
    return sum(1 for m in messages if m.get("role") == "assistant")


def _row_for_result(
    *,
    result: dict[str, Any],
    result_index: int,
    task_meta: dict[str, Any],
    task_name: str,
    source_label: str,
    source_model: str,
    summary_mtime_iso: str,
    tools_json: str = _DEFAULT_TOOLS_JSON,
) -> dict[str, Any] | None:
    """Convert one trajectory (one entry in `summary['results']`) into the
    parquet row shape.  Returns None when the trajectory has no messages
    (which would be a corrupt summary).
    """
    raw_msgs = result.get("messages") or []
    if not raw_msgs:
        return None

    messages = [_normalise_message(m) for m in raw_msgs if isinstance(m, dict)]
    if not messages:
        return None

    has_task_complete = _has_submit(messages)
    num_turns = _count_turns(messages)

    metadata = {
        "date": summary_mtime_iso,
        "enable_thinking": False,  # our local-vLLM runs all set enable_thinking=false
        "episode": f"sol-{result_index}",
        "has_ctrl_c": False,        # n/a for the rl_data harness loop
        "has_task_complete": has_task_complete,
        "json_extraction_failed": False,  # n/a, we don't go through json_extraction
        "json_strategy_counts": dict(_DEFAULT_JSON_STRATEGY_COUNTS),
        "num_turns": num_turns,
        "num_warnings": 0,
        "run_id": f"{task_name}:{result_index}",
        "source_model": source_model,
        "task": task_name,
        "trial_name": f"{task_name}__sol{result_index}",
    }

    return {
        "messages": messages,
        "tools": tools_json,
        "source": source_label,
        "metadata": metadata,
        # Carry the success flag separately for filtering BEFORE we drop
        # this column in the final parquet.
        "_success": bool(result.get("success", False)),
        # Rich extras we might want in the report later -- also dropped
        # before parquet write.
        "_reward": result.get("reward", 0),
        "_usage": result.get("usage", {}),
        # Plus the task fields, in case a downstream wants to attach them
        # (we don't pour them into metadata to keep schema parity strict).
        "_task_meta": task_meta,
    }


def _summary_basename(model_tag: str, harness: str) -> str:
    """Mirror ``rl_data.generate_solutions._summary_basename``.

    Bash keeps the legacy ``<MODEL_TAG>_summary.json`` filename; non-bash
    harnesses get a harness suffix so they can coexist on disk with bash
    summaries on the same task (no overwriting, no cp-r dance).
    """
    suffix = "" if harness == "bash" else f"_{harness}"
    return f"{model_tag}{suffix}_summary.json"


def _scan_tasks(
    tasks_dir: Path,
    model_tag: str,
    harness: str = "bash",
) -> list[tuple[Path, Path]]:
    """Return [(task_dir, summary_path), ...] for every task with a summary
    matching the given (model_tag, harness) pair.

    Sorted by task_dir.name so output is deterministic across runs.
    """
    summary_name = _summary_basename(model_tag, harness)
    pairs: list[tuple[Path, Path]] = []
    for child in sorted(tasks_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child.name.startswith("task_") or (child / "task.json").exists()):
            continue
        sp = child / "solutions" / summary_name
        if sp.exists():
            pairs.append((child, sp))
    return pairs


def _load_task_meta(task_dir: Path) -> dict[str, Any]:
    """Best-effort load of task.json.  Returns {} on any error."""
    p = task_dir / "task.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", p, exc)
        return {}


def convert(
    *,
    tasks_dir: Path,
    model_tag: str,
    output_dir: Path,
    name: str,
    filter_success: bool = False,
    source_label: str | None = None,
    tools_json: str = _DEFAULT_TOOLS_JSON,
    harness: str = "bash",
) -> dict[str, Any]:
    """Walk the trajectories under ``tasks_dir`` and write a parquet to
    ``output_dir/<name>.parquet`` plus a ``<name>.report.json`` summary.

    ``harness`` selects which summary filename to look for under each
    ``<task>/solutions/`` dir; it must match what ``generate_solutions.py``
    was invoked with at solve time. ``"bash"`` (default) reads the legacy
    ``<MODEL_TAG>_summary.json``; ``"vanillux"`` reads
    ``<MODEL_TAG>_vanillux_summary.json``.

    Returns the report dict (also written to disk).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_model = _model_tag_to_canonical(model_tag)
    if source_label is None:
        # Mirrors the source-string convention of tmax-sft-full-20260409
        # (e.g. "open-thoughts/OpenThoughts-Agent-v1-SFT") -- we use a
        # synthetic identifier so HF Datasets viewers can filter by it.
        source_label = f"tmax-rl-trajectories/{tasks_dir.name}/{canonical_model}"

    summary_basename = _summary_basename(model_tag, harness)
    pairs = _scan_tasks(tasks_dir, model_tag, harness=harness)
    logger.info(
        "Scanning %s for %s -> %d task(s) with a summary",
        tasks_dir, summary_basename, len(pairs),
    )

    rows: list[dict[str, Any]] = []
    n_traj_total = 0
    n_traj_kept = 0
    n_traj_dropped_filter = 0
    n_traj_dropped_empty = 0
    turn_counts: list[int] = []
    completion_tokens_sum = 0
    prompt_tokens_sum = 0
    n_success = 0

    for task_dir, summary_path in pairs:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            logger.warning("Skipping %s (parse error: %s)", summary_path, exc)
            continue

        task_meta = _load_task_meta(task_dir)
        task_name = task_meta.get("name") or task_dir.name
        summary_mtime_iso = (
            datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc)
            .isoformat(timespec="seconds")
        )

        results = summary.get("results") or []
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            n_traj_total += 1
            if result.get("success") is True:
                n_success += 1

            row = _row_for_result(
                result=result,
                result_index=i,
                task_meta=task_meta,
                task_name=task_name,
                source_label=source_label,
                source_model=canonical_model,
                summary_mtime_iso=summary_mtime_iso,
                tools_json=tools_json,
            )
            if row is None:
                n_traj_dropped_empty += 1
                continue

            if filter_success and not row["_success"]:
                n_traj_dropped_filter += 1
                continue

            turn_counts.append(row["metadata"]["num_turns"])
            usage = row["_usage"] if isinstance(row["_usage"], dict) else {}
            completion_tokens_sum += int(usage.get("completion_tokens", 0) or 0)
            prompt_tokens_sum += int(usage.get("prompt_tokens", 0) or 0)
            rows.append(row)
            n_traj_kept += 1

    # Strip private (`_`-prefixed) helper columns before writing parquet,
    # so the output schema is exactly {messages, source, metadata}.
    parquet_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in rows
    ]
    if parquet_rows:
        ds = Dataset.from_list(parquet_rows)
        out_path = output_dir / f"{name}.parquet"
        ds.to_parquet(str(out_path))
    else:
        out_path = output_dir / f"{name}.parquet"
        logger.warning("No rows to write -- skipping parquet creation at %s", out_path)

    turn_counts_sorted = sorted(turn_counts)
    n = len(turn_counts_sorted)
    turn_stats: dict[str, Any] = {}
    if n > 0:
        turn_stats = {
            "min": turn_counts_sorted[0],
            "p50": turn_counts_sorted[n // 2],
            "p95": turn_counts_sorted[min(n - 1, int(n * 0.95))],
            "max": turn_counts_sorted[-1],
            "mean": round(sum(turn_counts_sorted) / n, 2),
        }

    try:
        tools_summary = [
            (t.get("function") or {}).get("name", "<unnamed>")
            for t in json.loads(tools_json)
        ]
    except Exception:
        tools_summary = []

    report = {
        "tasks_dir": str(tasks_dir),
        "model_tag": model_tag,
        "model_canonical": canonical_model,
        "harness": harness,
        "summary_basename": summary_basename,
        "source_label": source_label,
        "output_parquet": str(out_path),
        "filter_success": filter_success,
        "n_tasks_scanned": len(pairs),
        "n_trajectories_total": n_traj_total,
        "n_trajectories_kept": n_traj_kept,
        "n_trajectories_dropped_filter_success": n_traj_dropped_filter,
        "n_trajectories_dropped_empty_messages": n_traj_dropped_empty,
        "n_successful_trajectories": n_success,
        "success_rate_overall": round(n_success / n_traj_total, 4) if n_traj_total else 0.0,
        "turn_stats": turn_stats,
        "total_prompt_tokens": prompt_tokens_sum,
        "total_completion_tokens": completion_tokens_sum,
        "tools": tools_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_path = output_dir / f"{name}.report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        "Wrote %d rows to %s (kept %d / %d trajectories; success-only=%s)",
        n_traj_kept, out_path, n_traj_kept, n_traj_total, filter_success,
    )
    return report


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--tasks-dir",
        required=True,
        type=Path,
        help="Path to the tasks directory (e.g. rl_data/output/tasks_skill_tax_20260324_1k).",
    )
    p.add_argument(
        "--model-tag",
        required=True,
        help="Model tag used in the summary filename, with '/' replaced by '_' "
             "(e.g. hosted_vllm_Qwen_Qwen3.5-27B, gemini_gemini-3-flash-preview).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write <name>.parquet and <name>.report.json.",
    )
    p.add_argument(
        "--name",
        required=True,
        help="Stem for the parquet/report files; also becomes the HF dataset 'config' "
             "name when uploaded by sft/scripts/upload_data_to_hf.sh.",
    )
    p.add_argument(
        "--filter-success",
        action="store_true",
        help="Keep only trajectories where the harness verifier returned success=True.",
    )
    p.add_argument(
        "--harness",
        default="bash",
        choices=("bash", "vanillux"),
        help="Solve-time harness used by rl_data.generate_solutions. Selects "
             "which summary file to look for under each <task>/solutions/: "
             "'bash' (default) -> <MODEL_TAG>_summary.json, "
             "'vanillux' -> <MODEL_TAG>_vanillux_summary.json. Must match the "
             "--harness passed to the solve script.",
    )
    p.add_argument(
        "--source-label",
        default=None,
        help="Override the auto-generated 'source' field on every row "
             "(default: tmax-rl-trajectories/<tasks_dir.name>/<canonical_model>).",
    )
    p.add_argument(
        "--tools-json",
        default=None,
        help="Override the OpenAI-style tools spec stored on every row, as a "
             "JSON string (must already be a list of tool definitions). "
             "Defaults to the single-tool bash spec the rl_data harness uses. "
             "Use @path/to/tools.json to load from a file.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.tools_json is None:
        tools_json = _DEFAULT_TOOLS_JSON
    elif args.tools_json.startswith("@"):
        tools_json = Path(args.tools_json[1:]).read_text()
    else:
        tools_json = args.tools_json
    # Validate it parses to a list (fail loud rather than silently writing
    # garbage into every row).
    parsed = json.loads(tools_json)
    if not isinstance(parsed, list):
        p.error("--tools-json must be a JSON array of tool definitions; got %s" % type(parsed).__name__)

    t0 = time.time()
    report = convert(
        tasks_dir=args.tasks_dir,
        model_tag=args.model_tag,
        output_dir=args.output_dir,
        name=args.name,
        filter_success=args.filter_success,
        source_label=args.source_label,
        tools_json=tools_json,
        harness=args.harness,
    )
    elapsed = time.time() - t0
    logger.info("Done in %.1fs.  Report: %s", elapsed, json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
