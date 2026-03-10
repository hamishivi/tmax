"""Main conversion pipeline orchestrator.

Usage::

    python -m preprocessing.pipeline [--sources ...] [--num-workers N]
                                     [--sample N] [--output-dir DIR]

Reads ``config/sources.yaml``, downloads raw data, converts every trace
through :func:`convert.convert_trace`, applies quality filters, and writes
per-source Parquet files plus an aggregate ``conversion_report.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from fnmatch import fnmatch
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Sequence, Value, load_dataset
from datasets.table import InMemoryTable
from huggingface_hub import HfApi, hf_hub_download

from preprocessing.convert import convert_trace
from preprocessing.filters import (
    FilterVerdict,
    apply_mandatory_filters,
    apply_optional_filters,
    apply_warning_flags,
)
from preprocessing.report import print_report

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "terminus2_sweagent"


# ======================================================================
# Source registry helpers
# ======================================================================

def load_source_registry(path: Path | None = None) -> list[dict]:
    """Load the YAML source registry and return the list of source entries."""
    if path is None:
        path = _CONFIG_DIR / "sources.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["sources"]


def _iter_source_subsets(
    registry: list[dict],
    selected_labels: list[str] | None = None,
) -> list[dict]:
    """Flatten registry into a list of ``(repo_id, type, subset_info)`` dicts,
    optionally filtered to *selected_labels*."""
    items: list[dict] = []
    for source in registry:
        repo_id = source["name"]
        load_type = source["type"]
        conv_col = source.get("conversations_column", "conversations")
        for sub in source.get("subsets", []):
            label = sub["source_label"]
            if selected_labels and label not in selected_labels:
                continue
            items.append({
                "repo_id": repo_id,
                "type": load_type,
                "subset": sub.get("subset"),
                "pattern": sub.get("pattern"),
                "source_label": label,
                "conversations_column": conv_col,
            })
    return items


# ======================================================================
# Raw data loading (moved from sft/data.py)
# ======================================================================

def _resolve_parquet_paths(repo_id: str, pattern: str) -> list[str]:
    all_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    paths = sorted(f for f in all_files if fnmatch(f, pattern))
    if not paths:
        raise FileNotFoundError(
            f"No parquet files matching '{pattern}' in {repo_id}"
        )
    return paths


def _ensure_downloaded(
    repo_id: str,
    paths: list[str],
    cache_dir: str | None = None,
) -> list[str]:
    return [
        hf_hub_download(repo_id, p, repo_type="dataset", cache_dir=cache_dir)
        for p in paths
    ]


def _read_parquet_files(
    local_paths: list[str],
    columns: list[str] | None = None,
) -> pa.Table:
    """Read parquet files with batched iteration to avoid the pyarrow nested-
    array size bug on large row groups.

    Uses ``pa.concat_tables`` with schema promotion so files that differ by a
    column (e.g. one subset has ``source`` and another doesn't) are merged
    gracefully instead of raising ``ArrowInvalid``.
    """
    tables: list[pa.Table] = []
    for path in local_paths:
        pf = pq.ParquetFile(path)
        batches = list(pf.iter_batches(batch_size=1024, columns=columns))
        if batches:
            tables.append(pa.Table.from_batches(batches))
    if not tables:
        raise ValueError(f"No data found in parquet files: {local_paths}")
    return pa.concat_tables(tables, promote_options="default")


def load_raw_dataset(item: dict, cache_dir: str | None = None) -> Dataset:
    """Load a single source-subset as an HF Dataset."""
    if item["type"] == "huggingface":
        ds = load_dataset(item["repo_id"], split="train", cache_dir=cache_dir)
        return ds

    # huggingface_parquet: custom loading for repos with large row groups
    hub_paths = _resolve_parquet_paths(item["repo_id"], item["pattern"])
    local_paths = _ensure_downloaded(item["repo_id"], hub_paths, cache_dir)
    table = _read_parquet_files(local_paths)
    return Dataset(InMemoryTable(table))


# ======================================================================
# Per-source processing
# ======================================================================

def _convert_fn(row: dict, source_label: str, conversations_column: str) -> dict:
    """Wrapper suitable for ``Dataset.map``."""
    return convert_trace(
        row,
        source_label=source_label,
        conversations_column=conversations_column,
    )


def _filter_and_flag(row: dict, max_turns: int) -> dict:
    """Apply filters and stamp verdicts onto the row."""
    mandatory = apply_mandatory_filters(row)
    if not mandatory.keep:
        return {
            **row,
            "_keep": False,
            "_drop_reason": mandatory.drop_reason,
            "_warning_flags": [],
        }

    optional = apply_optional_filters(row, max_turns=max_turns)
    if not optional.keep:
        return {
            **row,
            "_keep": False,
            "_drop_reason": optional.drop_reason,
            "_warning_flags": [],
        }

    flags = apply_warning_flags(row)
    return {**row, "_keep": True, "_drop_reason": None, "_warning_flags": flags}


def process_source(
    item: dict,
    *,
    output_dir: Path,
    num_workers: int,
    sample: int | None,
    sample_frac: float | None,
    max_turns: int,
    cache_dir: str | None,
    num_examples: int = 3,
) -> tuple[dict, list[dict]]:
    """Download, convert, filter, and save one source-subset.

    Returns a (statistics_dict, example_traces) tuple.
    """
    label = item["source_label"]
    safe_name = label.replace("/", "__")
    logger.info("Processing %s ...", label)
    t0 = time.time()

    # 1. Load raw data
    ds = load_raw_dataset(item, cache_dir=cache_dir)
    full_count = len(ds)

    if sample is not None and sample < full_count:
        ds = ds.shuffle(seed=42).select(range(sample))
    elif sample_frac is not None and 0 < sample_frac < 1:
        n = max(1, int(full_count * sample_frac))
        ds = ds.shuffle(seed=42).select(range(n))

    input_count = len(ds)

    # 2. Convert
    map_fn = partial(
        _convert_fn,
        source_label=label,
        conversations_column=item["conversations_column"],
    )
    ds = ds.map(map_fn, num_proc=num_workers, desc=f"Converting {label}")

    # 3. Filter
    filter_fn = partial(_filter_and_flag, max_turns=max_turns)
    filter_features = ds.features.copy()
    filter_features["_keep"] = Value("bool")
    filter_features["_drop_reason"] = Value("string")
    filter_features["_warning_flags"] = Sequence(Value("string"))
    ds = ds.map(
        filter_fn, num_proc=num_workers, features=filter_features,
        desc=f"Filtering {label}",
    )

    kept = ds.filter(lambda r: r["_keep"], num_proc=num_workers)
    dropped = ds.filter(lambda r: not r["_keep"], num_proc=num_workers)

    # 4. Collect statistics
    drop_reasons: dict[str, int] = {}
    for reason in dropped["_drop_reason"]:
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    strategy_counts = {1: 0, 2: 0, 3: 0, 0: 0}
    turn_counts: list[int] = []
    for meta in kept["metadata"]:
        for k, v in (meta.get("json_strategy_counts") or {}).items():
            strategy_counts[int(k)] = strategy_counts.get(int(k), 0) + v
        turn_counts.append(meta.get("num_turns", 0))

    warning_counts: dict[str, int] = {}
    for flags in kept["_warning_flags"]:
        for f in flags:
            warning_counts[f] = warning_counts.get(f, 0) + 1

    # Turn-count distribution
    turn_stats = {}
    if turn_counts:
        turn_counts_sorted = sorted(turn_counts)
        n = len(turn_counts_sorted)
        turn_stats = {
            "min": turn_counts_sorted[0],
            "max": turn_counts_sorted[-1],
            "mean": round(sum(turn_counts_sorted) / n, 1),
            "median": turn_counts_sorted[n // 2],
            "p95": turn_counts_sorted[int(n * 0.95)],
        }

    # 5. Collect qualitative examples (sample from kept traces)
    rng = random.Random(42)

    examples: list[dict] = []
    if len(kept) > 0 and num_examples > 0:
        indices = rng.sample(range(len(kept)), min(num_examples, len(kept)))
        for idx in indices:
            row = kept[idx]
            examples.append({
                "source": row.get("source", label),
                "messages": row.get("messages", []),
                "metadata": row.get("metadata", {}),
            })

    # 5b. Sample dropped traces per drop reason for diagnosis
    dropped_examples: list[dict] = []
    if len(dropped) > 0:
        reason_indices: dict[str, list[int]] = {}
        for idx, reason in enumerate(dropped["_drop_reason"]):
            reason_indices.setdefault(reason, []).append(idx)
        conv_col = item["conversations_column"]
        for reason, indices in reason_indices.items():
            sampled = rng.sample(indices, min(2, len(indices)))
            for idx in sampled:
                row = dropped[idx]
                raw_convos = row.get(conv_col, [])
                dropped_examples.append({
                    "source": row.get("source", label),
                    "drop_reason": reason,
                    "trial_name": row.get("metadata", {}).get("trial_name", "?"),
                    "messages": row.get("messages", []),
                    "warnings": row.get("warnings", []),
                    "num_raw_messages": len(raw_convos),
                    "raw_first_user": _extract_first_user(raw_convos),
                    "raw_last_assistant": _extract_last_assistant(raw_convos),
                })

    # 6. Save kept traces (only training-relevant columns)
    out_path = output_dir / f"{safe_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = {"messages", "source", "metadata"}
    drop_cols = [c for c in kept.column_names if c not in keep_cols]
    if drop_cols:
        kept = kept.remove_columns(drop_cols)
    kept.to_parquet(str(out_path))

    # 7. Save dropped traces for inspection
    if len(dropped) > 0:
        dropped_path = output_dir / f"{safe_name}_dropped.jsonl"
        _save_jsonl(dropped, dropped_path)

    elapsed = time.time() - t0
    stats = {
        "source_label": label,
        "input_traces": input_count,
        "output_traces": len(kept),
        "dropped": len(dropped),
        "drop_reasons": drop_reasons,
        "json_strategy_distribution": strategy_counts,
        "warning_counts": warning_counts,
        "turn_stats": turn_stats,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "  %s: %d -> %d traces (dropped %d) in %.1fs",
        label, input_count, len(kept), len(dropped), elapsed,
    )
    return stats, examples, dropped_examples


# ======================================================================
# Top-level pipeline
# ======================================================================

def run_pipeline(
    *,
    sources: list[str] | None = None,
    output_dir: Path | str | None = None,
    num_workers: int | None = None,
    sample: int | None = None,
    sample_frac: float | None = None,
    max_turns: int = 999,
    cache_dir: str | None = None,
    sources_yaml: Path | str | None = None,
    num_examples: int = 3,
) -> dict:
    """Run the full conversion pipeline.

    Parameters
    ----------
    sources : list[str] | None
        Source labels to process (``None`` = all registered).
    output_dir : path
        Where to write Parquet and report files.
    num_workers : int
        ``num_proc`` for ``datasets.map``.  Defaults to CPU count.
    sample : int | None
        If set, only convert this many traces per source (for validation).
    sample_frac : float | None
        If set, sample this fraction of each source (e.g. 0.01 for 1%).
    max_turns : int
        Optional filter: drop traces exceeding this many turns.
    cache_dir : str | None
        Override HuggingFace cache directory.
    sources_yaml : path | None
        Path to the YAML registry (default: ``config/sources.yaml``).
    num_examples : int
        Number of qualitative examples to sample per source for the report.

    Returns
    -------
    dict : the full conversion report.
    """
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, os.cpu_count() or 1)

    registry = load_source_registry(
        Path(sources_yaml) if sources_yaml else None
    )
    items = _iter_source_subsets(registry, selected_labels=sources)

    if not items:
        raise ValueError(
            f"No source subsets matched.  Requested: {sources}. "
            f"Available: {[s['source_label'] for src in registry for s in src.get('subsets', [])]}"
        )

    sample_desc = f"sample={sample}" if sample else (
        f"sample_frac={sample_frac}" if sample_frac else "full"
    )
    logger.info(
        "Pipeline starting: %d source(s), %d workers, %s",
        len(items), num_workers, sample_desc,
    )

    all_stats: list[dict] = []
    all_examples: list[dict] = []
    all_dropped_examples: list[dict] = []
    for item in items:
        stats, examples, dropped_examples = process_source(
            item,
            output_dir=output_dir,
            num_workers=num_workers,
            sample=sample,
            sample_frac=sample_frac,
            max_turns=max_turns,
            cache_dir=cache_dir,
            num_examples=num_examples,
        )
        all_stats.append(stats)
        all_examples.extend(examples)
        all_dropped_examples.extend(dropped_examples)

    # Aggregate report
    report = _build_report(all_stats)
    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", report_path)

    # Print rich terminal summary
    print_report(report, examples=all_examples, dropped_examples=all_dropped_examples)

    return report


def _build_report(all_stats: list[dict]) -> dict:
    total_in = sum(s["input_traces"] for s in all_stats)
    total_out = sum(s["output_traces"] for s in all_stats)
    total_dropped = sum(s["dropped"] for s in all_stats)

    agg_drop: dict[str, int] = {}
    agg_warn: dict[str, int] = {}
    agg_strat = {1: 0, 2: 0, 3: 0, 0: 0}
    for s in all_stats:
        for k, v in s["drop_reasons"].items():
            agg_drop[k] = agg_drop.get(k, 0) + v
        for k, v in s["warning_counts"].items():
            agg_warn[k] = agg_warn.get(k, 0) + v
        for k, v in s["json_strategy_distribution"].items():
            agg_strat[int(k)] = agg_strat.get(int(k), 0) + v

    return {
        "per_source": {s["source_label"]: s for s in all_stats},
        "aggregate": {
            "total_input_traces": total_in,
            "total_output_traces": total_out,
            "total_dropped": total_dropped,
            "drop_reasons": agg_drop,
            "warning_counts": agg_warn,
            "json_strategy_distribution": agg_strat,
            "per_source_turn_stats": {
                s["source_label"]: s.get("turn_stats", {})
                for s in all_stats if s.get("turn_stats")
            },
        },
    }


# ======================================================================
# Utilities
# ======================================================================

def _extract_first_user(raw_convos: list[dict], max_len: int = 300) -> str:
    """Return the first user message content, truncated."""
    for msg in raw_convos:
        if msg.get("role") == "user":
            text = msg.get("content", "")
            if len(text) > max_len:
                return text[:max_len] + "..."
            return text
    return ""


def _extract_last_assistant(raw_convos: list[dict], max_len: int = 300) -> str:
    """Return the last assistant message content, truncated."""
    for msg in reversed(raw_convos):
        if msg.get("role") == "assistant":
            text = msg.get("content", "")
            if len(text) > max_len:
                return text[:max_len] + "..."
            return text
    return ""


def _save_jsonl(ds: Dataset, path: Path) -> None:
    keep = {"messages", "source", "metadata", "warnings", "_drop_reason",
            "conversations"}
    cols = [c for c in ds.column_names if c in keep]
    subset = ds.select_columns(cols)
    subset.to_json(str(path), lines=True)


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Terminus-2 → SWE-agent conversion pipeline",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Source labels to process (default: all registered in sources.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for dataset.map (default: CPU count).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only convert this many traces per source (for validation runs).",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Sample this fraction of each source (e.g. 0.01 for 1%%).",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=999,
        help="Drop traces exceeding this many turns (default: 999, effectively no limit).",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of qualitative example traces to show per source (default: 3).",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override HuggingFace cache directory.",
    )
    p.add_argument(
        "--sources-yaml",
        type=str,
        default=None,
        help="Path to the YAML source registry.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    run_pipeline(
        sources=args.sources,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        sample=args.sample,
        sample_frac=args.sample_frac,
        max_turns=args.max_turns,
        cache_dir=args.cache_dir,
        sources_yaml=args.sources_yaml,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
