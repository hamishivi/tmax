#!/usr/bin/env python3
"""Plot DPPO vs GRPO validation reward from W&B.

Fetches ``val/avg_group_performance_post_filter`` for the configured W&B runs
and renders a tight one-column figure matching ``~/Downloads/dppo_vs_grpo.pdf``.

Run:

    uv run --with wandb python scripts/plot/dppo_vs_grpo.py

Defaults to ``ai2-llm/open_instruct_internal``.  ``WANDB_ENTITY`` and
``WANDB_PROJECT`` override those defaults when set.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "dppo_vs_grpo"
DEFAULT_ENTITY = "ai2-llm"
DEFAULT_PROJECT = "open_instruct_internal"
DEFAULT_METRIC = "val/avg_group_performance_post_filter"
DEFAULT_MAX_STEP = 350
DEFAULT_SMOOTH_WINDOW = 5

DEFAULT_RUNS = [
    "swerl_qwen35_9b_fp32lm_dppo__42__1779864484",
    "swerl_qwen35_9b_fp32lm__42__1779226278",
]

DISPLAY_LABELS = {
    "swerl_qwen35_9b_fp32lm_dppo__42__1779864484": "DPPO",
    "swerl_qwen35_9b_fp32lm__42__1779226278": "GRPO",
}

DISPLAY_ORDER = ["DPPO", "GRPO"]

DISPLAY_COLORS = {
    "DPPO": "#1f77b4",
    "GRPO": "#d62728",
}


@dataclass(frozen=True)
class RunSeries:
    ref: str
    label: str
    steps: np.ndarray
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot val/avg_group_performance_post_filter for DPPO vs GRPO W&B runs."
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


def require_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in the active environment. "
            "Use `uv run --with wandb ...` for this script, or install wandb."
        ) from exc
    return wandb


def default_label(run_ref: str) -> str:
    if run_ref in DISPLAY_LABELS:
        return DISPLAY_LABELS[run_ref]
    return run_ref


def candidate_matches(run, run_ref: str) -> bool:
    path_parts = getattr(run, "path", None) or []
    names = {
        getattr(run, "id", None),
        getattr(run, "name", None),
        getattr(run, "display_name", None),
        path_parts[-1] if path_parts else None,
    }
    return run_ref in names


def resolve_run(api, project_path: str, run_ref: str):
    if run_ref.count("/") == 2:
        return api.run(run_ref)

    try:
        return api.run(f"{project_path}/{run_ref}")
    except Exception:
        pass

    for filters in ({"display_name": run_ref}, {"displayName": run_ref}, {"name": run_ref}):
        try:
            matches = [run for run in api.runs(project_path, filters=filters)]
        except Exception:
            continue
        matches = [run for run in matches if candidate_matches(run, run_ref)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            urls = "\n  ".join(getattr(run, "url", repr(run)) for run in matches)
            raise SystemExit(f"Multiple W&B runs match {run_ref!r}:\n  {urls}")

    matches = [run for run in api.runs(project_path) if candidate_matches(run, run_ref)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        urls = "\n  ".join(getattr(run, "url", repr(run)) for run in matches)
        raise SystemExit(f"Multiple W&B runs match {run_ref!r}:\n  {urls}")
    raise SystemExit(f"No W&B run matched {run_ref!r} in {project_path!r}.")


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


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    half = window // 2
    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed[i] = float(np.mean(values[start:end]))
    return smoothed


def sorted_series(series: Sequence[RunSeries]) -> list[RunSeries]:
    order = {label: i for i, label in enumerate(DISPLAY_ORDER)}
    return sorted(series, key=lambda run_series: order.get(run_series.label, len(order)))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#808080",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix:
        paths = [out]
    else:
        paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
    return paths


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
