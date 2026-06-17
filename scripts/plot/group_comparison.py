#!/usr/bin/env python3
"""Plot G=32 vs G=8 validation reward from W&B.

Fetches ``val/avg_group_performance_post_filter`` for the configured W&B runs
and renders a tight one-column figure matching ``~/Downloads/group_comparison.pdf``.

Run:

    uv run --with wandb python scripts/plot/group_comparison.py

Defaults to ``ai2-llm/open_instruct_internal``.  ``WANDB_ENTITY`` and
``WANDB_PROJECT`` override those defaults when set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dppo_vs_grpo import (
    RunSeries,
    centered_moving_average,
    configure_matplotlib,
    require_wandb,
    resolve_run,
    save_figure,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "group_comparison"
DEFAULT_ENTITY = "ai2-llm"
DEFAULT_PROJECT = "open_instruct_internal"
DEFAULT_METRIC = "val/avg_group_performance_post_filter"
DEFAULT_MAX_STEP = 350
DEFAULT_SMOOTH_WINDOW = 5

DEFAULT_RUNS = [
    "swerl_qwen35_9b_fp32lm_dppo_g32__42__1779819805",
    "swerl_qwen35_9b_fp32lm_dppo__42__1779864484",
]

DISPLAY_LABELS = {
    "swerl_qwen35_9b_fp32lm_dppo_g32__42__1779819805": "G=32",
    "swerl_qwen35_9b_fp32lm_dppo__42__1779864484": "G=8",
}

DISPLAY_ORDER = ["G=32", "G=8"]

DISPLAY_COLORS = {
    "G=32": "#2ca02c",
    "G=8": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot val/avg_group_performance_post_filter for G=32 vs G=8 W&B runs."
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="W&B run display name, id, or entity/project/run_id path. Repeat to override defaults.",
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--max-step", type=int, default=DEFAULT_MAX_STEP)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path. If no suffix is given, writes both .png and .pdf.",
    )
    return parser.parse_args()


def default_label(run_ref: str) -> str:
    return DISPLAY_LABELS.get(run_ref, run_ref)


def fetch_series(api, project_path: str, run_ref: str, metric: str, max_step: int) -> RunSeries:
    run = resolve_run(api, project_path, run_ref)
    points_by_step: dict[float, float] = {}

    for index, row in enumerate(
        run.scan_history(keys=[metric], min_step=0, max_step=max_step, page_size=1000)
    ):
        raw_value = row.get(metric)
        if raw_value is None:
            continue
        raw_step = row.get("_step", index)
        try:
            step = float(raw_step)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(step) and np.isfinite(value) and step <= max_step:
            points_by_step[step] = value

    if not points_by_step:
        raise SystemExit(f"{run_ref!r} has no finite {metric!r} values through step {max_step}.")

    ordered = sorted(points_by_step.items())
    steps = np.array([step for step, _ in ordered], dtype=float)
    values = np.array([value for _, value in ordered], dtype=float)
    return RunSeries(ref=run_ref, label=default_label(run_ref), steps=steps, values=values)


def sorted_series(series: Sequence[RunSeries]) -> list[RunSeries]:
    order = {label: i for i, label in enumerate(DISPLAY_ORDER)}
    return sorted(series, key=lambda run_series: order.get(run_series.label, len(order)))


def plot_series(
    series: Sequence[RunSeries],
    max_step: int,
    smooth_window: int,
    out: Path,
) -> list[Path]:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.28, 2.20))

    for run_series in sorted_series(series):
        color = DISPLAY_COLORS.get(run_series.label, "#333333")
        values = centered_moving_average(run_series.values, smooth_window)
        ax.plot(
            run_series.steps,
            values,
            color=color,
            linewidth=1.05,
            label=run_series.label,
            solid_capstyle="round",
        )

    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 0.95)
    ax.set_xticks([0, 100, 200, 300, max_step])
    ax.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Avg Train Reward")
    ax.grid(True, color="#D3D3D3", linewidth=0.45)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.01), handlelength=2.0)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.99)
    return save_figure(fig, out)


def fetch_all_series(
    api,
    project_path: str,
    runs: Iterable[str],
    metric: str,
    max_step: int,
) -> list[RunSeries]:
    series = []
    for run_ref in runs:
        print(f"Fetching {run_ref}...", file=sys.stderr)
        series.append(fetch_series(api, project_path, run_ref, metric, max_step))
    return series


def main() -> None:
    args = parse_args()
    runs = args.runs or DEFAULT_RUNS

    if not args.entity or not args.project:
        raise SystemExit("--entity and --project are required.")
    if args.max_step < 0:
        raise SystemExit("--max-step must be non-negative.")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1.")

    wandb = require_wandb()
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    series = fetch_all_series(api, project_path, runs, args.metric, args.max_step)
    paths = plot_series(series, args.max_step, args.smooth_window, args.out)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
