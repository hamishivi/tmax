"""Decontamination eval — n-gram overlap between datasets and benchmarks.

Usage::

    python -m rl_data.decontamination.cli \\
        --dataset   skill_tax:rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k \\
        --dataset   endless_terminals:rl_data/output/tasks_endless_terminals \\
        --benchmark tblite:rl_data/output/_decon_benchmarks/openthoughts-tblite \\
        --benchmark tb2:rl_data/output/_decon_benchmarks/terminal-bench \\
        --n 13,8 --stride 1 \\
        --out-dir rl_data/output/decontamination_<tag>

For each (benchmark, dataset, n) triple we build the set of all word
n-grams from the benchmark task descriptions, slide n-grams over each
dataset task description at the chosen stride, and report the fraction
of dataset documents containing at least one matching n-gram.

``--n`` accepts either a single int or a comma-separated list (e.g.
``--n 13,8,5``); the output table emits one row per (benchmark, n).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from rl_data.decontamination.loaders import (
    DescPair,
    load_benchmark_descriptions,
    load_dataset_descriptions,
)
from rl_data.decontamination.ngram import build_index, scan_doc

logger = logging.getLogger(__name__)


# Pretty names — mirrors rl_data/comparison/cli.py so columns line up
# with the existing comparison summary tables.
_DISPLAY_NAMES = {
    "skill_tax": "Skill-Tax (ours)",
    "endless_terminals": "Endless-Terminals",
    "openthoughts_tb": "OpenThoughts-TB",
    "openthoughts_agent_rl": "OpenThoughts-Agent-v1-RL",
    "termigen": "TermiGen",
    "terminaltraj": "TerminalTraj",
    "r2e_gym": "R2E-Gym",
    "tblite": "TBLite (openthoughts-tblite@2.0)",
    "tb2": "TB2 (terminal-bench@2.0)",
}


@dataclasses.dataclass
class Spec:
    name: str
    display: str
    path: Path


def _parse_spec(raw: str) -> Spec:
    if ":" in raw or "=" in raw:
        sep = ":" if ":" in raw else "="
        name, path = raw.split(sep, 1)
    else:
        path = raw
        name = Path(raw).name
    display = _DISPLAY_NAMES.get(name, name.replace("_", " ").title())
    return Spec(name=name, display=display, path=Path(path).resolve())


def _parse_n_list(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        v = int(chunk)
        if v <= 0:
            raise ValueError(f"--n values must be positive; got {v}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("--n must specify at least one n-gram size")
    # Sort descending so the more conservative (less permissive) score
    # appears first in the table.
    out.sort(reverse=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", action="append", default=[], required=True,
        help="<name>:<tasks_dir> for a midtraining dataset (repeatable)",
    )
    ap.add_argument(
        "--benchmark", action="append", default=[], required=True,
        help="<name>:<dir> for an evaluation benchmark (repeatable)",
    )
    ap.add_argument(
        "--n", type=str, default="13",
        help="Word n-gram size(s). Single int (e.g. '13') or comma-separated "
             "list (e.g. '13,8') to emit one row per (benchmark, n).",
    )
    ap.add_argument("--stride", type=int, default=1, help="n-gram sampling stride")
    ap.add_argument(
        "--max-tasks", type=int, default=0,
        help="Cap descriptions per dataset (0 = all). Useful for smoke tests.",
    )
    ap.add_argument(
        "--max-bench-tasks", type=int, default=0,
        help="Cap descriptions per benchmark (0 = all). Useful for smoke tests.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def _scan(
    docs: List[DescPair], index, n: int, stride: int,
) -> Tuple[int, int, int, int]:
    """Return (n_docs, n_contaminated, total_hits, total_ngrams)."""
    n_docs = len(docs)
    n_contam = 0
    total_hits = 0
    total_ng = 0
    for _, text in docs:
        hits, total = scan_doc(text, index, n, stride)
        total_hits += hits
        total_ng += total
        if hits > 0:
            n_contam += 1
    return n_docs, n_contam, total_hits, total_ng


# A single result cell, keyed by (benchmark_name, n, dataset_name).
CellKey = Tuple[str, int, str]


def _write_table(
    md_path: Path,
    csv_path: Path,
    dataset_specs: List[Spec],
    bench_specs: List[Spec],
    n_values: List[int],
    cells: Dict[CellKey, Dict[str, int]],
    stride: int,
) -> None:
    ds_disp = [s.display for s in dataset_specs]
    n_label = ", ".join(f"n={n}" for n in n_values)

    L: List[str] = [
        f"# Decontamination — n-gram overlap on task descriptions "
        f"({n_label}, stride={stride})",
        "",
        "Cell = % of dataset task descriptions whose n-gram window "
        "overlaps \u22651 n-gram from the benchmark. Larger n is "
        "stricter (fewer false positives); smaller n catches more "
        "near-paraphrase overlap.",
        "",
        "| Benchmark | n-gram | " + " | ".join(ds_disp) + " |",
        "|---|---:|" + "---:|" * len(ds_disp),
    ]
    for b in bench_specs:
        for n in n_values:
            row = [b.display, str(n)]
            for d in dataset_specs:
                c = cells.get((b.name, n, d.name), {})
                n_docs = c.get("n_docs", 0)
                n_contam = c.get("n_contaminated", 0)
                rate = (n_contam / n_docs) if n_docs else 0.0
                row.append(f"{rate * 100:.1f}%")
            L.append("| " + " | ".join(row) + " |")
    L.append("")

    md_path.write_text("\n".join(L) + "\n")

    fieldnames = [
        "benchmark", "n", "dataset", "n_docs", "n_contaminated_docs",
        "contamination_rate", "total_ngram_hits", "total_ngrams_scanned",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in bench_specs:
            for n in n_values:
                for d in dataset_specs:
                    c = cells.get((b.name, n, d.name), {})
                    n_docs = c.get("n_docs", 0)
                    n_contam = c.get("n_contaminated", 0)
                    rate = (n_contam / n_docs) if n_docs else 0.0
                    w.writerow({
                        "benchmark": b.name,
                        "n": n,
                        "dataset": d.name,
                        "n_docs": n_docs,
                        "n_contaminated_docs": n_contam,
                        "contamination_rate": f"{rate:.6f}",
                        "total_ngram_hits": c.get("total_hits", 0),
                        "total_ngrams_scanned": c.get("total_ngrams", 0),
                    })


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    n_values = _parse_n_list(args.n)
    dataset_specs = [_parse_spec(r) for r in args.dataset]
    bench_specs = [_parse_spec(r) for r in args.benchmark]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + cap.
    dataset_docs: Dict[str, List[DescPair]] = {}
    for s in dataset_specs:
        docs = load_dataset_descriptions(s.path)
        if args.max_tasks and args.max_tasks > 0:
            docs = docs[: args.max_tasks]
        logger.info("dataset  %s: %d descriptions from %s", s.name, len(docs), s.path)
        dataset_docs[s.name] = docs

    bench_docs: Dict[str, List[DescPair]] = {}
    # bench_indices[bench_name][n] -> set of n-gram tuples
    bench_indices: Dict[str, Dict[int, set]] = {}
    for s in bench_specs:
        docs = load_benchmark_descriptions(s.path)
        if args.max_bench_tasks and args.max_bench_tasks > 0:
            docs = docs[: args.max_bench_tasks]
        logger.info("benchmark %s: %d instructions from %s", s.name, len(docs), s.path)
        bench_docs[s.name] = docs
        bench_indices[s.name] = {}
        for n in n_values:
            idx = build_index((t for _, t in docs), n, args.stride)
            bench_indices[s.name][n] = idx
            logger.info("  -> n=%d: %d unique n-grams", n, len(idx))

    # Score.
    cells: Dict[CellKey, Dict[str, int]] = {}
    for b in bench_specs:
        for n in n_values:
            idx = bench_indices[b.name][n]
            for d in dataset_specs:
                n_docs, n_contam, total_hits, total_ng = _scan(
                    dataset_docs[d.name], idx, n, args.stride,
                )
                cells[(b.name, n, d.name)] = {
                    "n_docs": n_docs,
                    "n_contaminated": n_contam,
                    "total_hits": total_hits,
                    "total_ngrams": total_ng,
                }
                rate = (n_contam / n_docs) if n_docs else 0.0
                logger.info(
                    "  n=%d  %s vs %s: %d/%d contaminated (%.2f%%), "
                    "%d hits / %d n-grams",
                    n, b.name, d.name, n_contam, n_docs, rate * 100,
                    total_hits, total_ng,
                )

    _write_table(
        out_dir / "decontamination_table.md",
        out_dir / "decontamination_data.csv",
        dataset_specs, bench_specs, n_values, cells,
        stride=args.stride,
    )

    report = {
        "params": {"n_values": n_values, "stride": args.stride},
        "datasets": [
            {
                "name": s.name, "display_name": s.display,
                "path": str(s.path), "n_docs": len(dataset_docs[s.name]),
            }
            for s in dataset_specs
        ],
        "benchmarks": [
            {
                "name": s.name, "display_name": s.display,
                "path": str(s.path), "n_docs": len(bench_docs[s.name]),
                "n_unique_ngrams": {
                    str(n): len(bench_indices[s.name][n]) for n in n_values
                },
            }
            for s in bench_specs
        ],
        "results": [
            {
                "benchmark": b, "n": n, "dataset": d,
                **vals,
                "contamination_rate": (
                    vals["n_contaminated"] / vals["n_docs"]
                    if vals["n_docs"] else 0.0
                ),
            }
            for (b, n, d), vals in cells.items()
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    logger.info("Done. See %s/decontamination_table.md", out_dir)


if __name__ == "__main__":
    main()
