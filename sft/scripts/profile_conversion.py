#!/usr/bin/env python3
"""Profile convert_trace to find the real bottleneck.

Usage (from sft/ directory):
    python scripts/profile_conversion.py
    python scripts/profile_conversion.py --rows 20 --source dataset_adapters
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the sft package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.convert import convert_trace
from preprocessing.json_extraction import extract_json_from_content
from preprocessing.pipeline import load_raw_dataset, load_source_registry, _iter_source_subsets


def profile_rows(ds, label: str, conv_col: str, n: int) -> None:
    """Run convert_trace on n rows with fine-grained timing."""

    print(f"\n{'='*70}")
    print(f"Profiling {n} rows from {label}")
    print(f"{'='*70}\n")

    total_convert = 0.0
    total_json_extract = 0.0
    total_json_loads_s1 = 0.0
    total_raw_decode = 0.0
    total_brace_match = 0.0
    total_build = 0.0
    total_messages = 0
    total_content_bytes = 0

    import json
    _orig_extract = extract_json_from_content.__wrapped__ if hasattr(extract_json_from_content, '__wrapped__') else None

    for idx in range(min(n, len(ds))):
        row = ds[idx]
        messages = row.get(conv_col, [])
        n_msgs = len(messages)

        # Measure total content size
        content_bytes = sum(len(m.get("content", "")) for m in messages)
        total_content_bytes += content_bytes

        # Count assistant messages
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        n_assistant = len(assistant_msgs)
        total_messages += n_assistant

        # --- Profile extract_json_from_content on each assistant msg ---
        json_time = 0.0
        for amsg in assistant_msgs:
            t0 = time.perf_counter()
            extract_json_from_content(amsg["content"])
            json_time += time.perf_counter() - t0

        total_json_extract += json_time

        # --- Profile full convert_trace ---
        t0 = time.perf_counter()
        result = convert_trace(row, source_label=label, conversations_column=conv_col)
        convert_time = time.perf_counter() - t0
        total_convert += convert_time

        build_time = convert_time - json_time
        total_build += build_time

        print(
            f"  Row {idx:3d}: {convert_time:6.2f}s total | "
            f"json_extract={json_time:6.2f}s | build={build_time:6.2f}s | "
            f"{n_msgs:3d} msgs ({n_assistant} asst) | "
            f"{content_bytes/1024:.0f} KB content"
        )

    print(f"\n{'─'*70}")
    print(f"TOTALS ({min(n, len(ds))} rows):")
    print(f"  convert_trace total:      {total_convert:8.2f}s")
    print(f"  json extraction total:    {total_json_extract:8.2f}s  ({total_json_extract/total_convert*100:.1f}%)")
    print(f"  build/other total:        {total_build:8.2f}s  ({total_build/total_convert*100:.1f}%)")
    print(f"  total assistant messages:  {total_messages}")
    print(f"  total content:             {total_content_bytes/1024/1024:.1f} MB")
    avg_per_row = total_convert / min(n, len(ds))
    print(f"  avg per row:               {avg_per_row:.2f}s")
    print(f"  est. throughput (1 core):  {1/avg_per_row:.2f} ex/s")
    print(f"  est. throughput (28 core): {28/avg_per_row:.2f} ex/s")

    # --- Now measure datasets.map overhead ---
    print(f"\n{'─'*70}")
    print("Measuring datasets.map overhead (single-process)...")

    from functools import partial
    from datasets import Dataset

    sample = ds.select(range(min(n, len(ds))))

    def _convert_fn(row, source_label, conversations_column):
        return convert_trace(row, source_label=source_label, conversations_column=conversations_column)

    map_fn = partial(_convert_fn, source_label=label, conversations_column=conv_col)

    t0 = time.perf_counter()
    sample.map(map_fn, num_proc=1)
    map_single = time.perf_counter() - t0

    print(f"  ds.map(num_proc=1):  {map_single:.2f}s  ({map_single/min(n,len(ds)):.2f}s/row)")
    print(f"  Pure convert_trace:  {total_convert:.2f}s  ({total_convert/min(n,len(ds)):.2f}s/row)")
    overhead = map_single - total_convert
    print(f"  Arrow overhead:      {overhead:.2f}s  ({overhead/map_single*100:.1f}% of map time)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--source", type=str, default="dataset_adapters",
                        help="Substring to match in source label")
    args = parser.parse_args()

    registry = load_source_registry()
    items = _iter_source_subsets(registry)

    matched = [it for it in items if args.source in it["source_label"]]
    if not matched:
        print(f"No source matching '{args.source}'. Available:")
        for it in items:
            print(f"  {it['source_label']}")
        return

    item = matched[0]
    print(f"Loading {item['source_label']}...")
    ds = load_raw_dataset(item)
    print(f"  {len(ds)} rows total")

    # Take first N rows (no sampling, just grab the first ones for profiling)
    profile_rows(ds, item["source_label"], item["conversations_column"], args.rows)


if __name__ == "__main__":
    main()
