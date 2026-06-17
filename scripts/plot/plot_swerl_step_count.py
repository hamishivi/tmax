#!/usr/bin/env python3
"""Plot W&B SWERL sandbox step-count traces.

Fetches the configured W&B runs, plots the raw
``env/swerl_vanillux_sandbox/step_count`` traces faintly, and overlays a
15-point centered moving average up to W&B step 300.  The default output is
sized for a one-column paper figure; use ``--layout paper`` for the wider
reference layout.

Run:

    uv run --with wandb python scripts/plot/plot_swerl_step_count.py

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
DEFAULT_OUT = HERE / "output" / "swerl_vanillux_step_count"
DEFAULT_ENTITY = "ai2-llm"
DEFAULT_PROJECT = "open_instruct_internal"
DEFAULT_METRIC = "env/swerl_vanillux_sandbox/step_count"
DEFAULT_MAX_STEP = 300
DEFAULT_SMOOTH_WINDOW = 15

DEFAULT_RUNS = [
    "swerl_qwen35_9b_fp32lm_dppo_swesmith__42__1781026611",
    "swerl_qwen35_9b_fp32lm_dppo_cli__42__1780590177",
    "swerl_qwen35_9b_fp32lm_dppo_endless__42__1780506505",
    "swerl_qwen35_9b_fp32lm_dppo_termigen__42__1780543002",
    "swerl_qwen35_9b_fp32lm_dppo_termigen__42__1780111694",
    "swerl_qwen35_9b_fp32lm_dppo_terminaltraj__42__1780103394",
    "swerl_qwen35_9b_fp32lm_dppo_g32__42__1779819805",
]

DISPLAY_LABELS = {
    "swerl_qwen35_9b_fp32lm_dppo_swesmith__42__1781026611": "SWE-Smith",
    "swerl_qwen35_9b_fp32lm_dppo_cli__42__1780590177": "CLI-Gym",
    "swerl_qwen35_9b_fp32lm_dppo_endless__42__1780506505": "Endless Terminals",
    "swerl_qwen35_9b_fp32lm_dppo_termigen__42__1780543002": "TermiGen",
    "swerl_qwen35_9b_fp32lm_dppo_termigen__42__1780111694": "TermiGen",
    "swerl_qwen35_9b_fp32lm_dppo_terminaltraj__42__1780103394": "TerminalTraj",
    "swerl_qwen35_9b_fp32lm_dppo_g32__42__1779819805": "Ours",
}

DISPLAY_ORDER = [
    "Ours",
    "TerminalTraj",
    "TermiGen",
    "SWE-Smith",
    "CLI-Gym",
    "Endless Terminals",
]

DISPLAY_COLORS = {
    "Ours": "#d62728",
    "TerminalTraj": "#2ca02c",
    "TermiGen": "#ff7f0e",
    "SWE-Smith": "#9467bd",
    "CLI-Gym": "#8c564b",
    "Endless Terminals": "#1f77b4",
}


@dataclass(frozen=True)
class RunSeries:
    ref: str
    label: str
    steps: np.ndarray
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot env/swerl_vanillux_sandbox/step_count for selected W&B runs."
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
        "--layout",
        choices=("one-column", "paper"),
        default="one-column",
        help="Figure layout. one-column is the compact paper-column default.",
    )
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
    prefix = run_ref.split("__", 1)[0]
    marker = "_dppo_"
    if marker in prefix:
        return prefix.split(marker, 1)[1].replace("_", " ")
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

    filter_candidates = (
        {"display_name": run_ref},
        {"displayName": run_ref},
        {"name": run_ref},
    )
    for filters in filter_candidates:
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


def grouped_series(series: Sequence[RunSeries]) -> list[tuple[str, list[RunSeries]]]:
    by_label: dict[str, list[RunSeries]] = {}
    for run_series in series:
        by_label.setdefault(run_series.label, []).append(run_series)

    groups: list[tuple[str, list[RunSeries]]] = []
    seen = set()
    for label in DISPLAY_ORDER:
        runs = by_label.get(label)
        if runs:
            groups.append((label, runs))
            seen.add(label)
    for label, runs in by_label.items():
        if label not in seen:
            groups.append((label, runs))
    return groups


def group_smoothed_curve(
    group: Sequence[RunSeries],
    window: int,
    max_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(group) == 1:
        run_series = group[0]
        return run_series.steps, centered_moving_average(run_series.values, window)

    steps = np.arange(0, max_step + 1, dtype=float)
    curves = []
    for run_series in group:
        smoothed = centered_moving_average(run_series.values, window)
        curve = np.interp(steps, run_series.steps, smoothed)
        curve[steps < run_series.steps.min()] = np.nan
        curve[steps > run_series.steps.max()] = np.nan
        curves.append(curve)

    stacked = np.vstack(curves)
    valid = np.any(np.isfinite(stacked), axis=0)
    averaged = np.full(stacked.shape[1], np.nan, dtype=float)
    averaged[valid] = np.nanmean(stacked[:, valid], axis=0)
    return steps[valid], averaged[valid]


@dataclass(frozen=True)
class Layout:
    figsize: tuple[float, float]
    font_size: float
    label_size: float
    tick_size: float
    legend_size: float
    raw_linewidth: float
    smooth_linewidth: float
    grid_linewidth: float
    legend_ncols: int
    legend_anchor: tuple[float, float]
    legend_handlelength: float
    legend_columnspacing: float
    legend_labelspacing: float
    legend_handletextpad: float
    subplot_adjust: dict[str, float]


LAYOUTS = {
    "one-column": Layout(
        figsize=(3.35, 2.55),
        font_size=7.0,
        label_size=8.0,
        tick_size=7.0,
        legend_size=6.7,
        raw_linewidth=0.65,
        smooth_linewidth=1.45,
        grid_linewidth=0.55,
        legend_ncols=2,
        legend_anchor=(0.5, 1.31),
        legend_handlelength=2.2,
        legend_columnspacing=0.85,
        legend_labelspacing=0.25,
        legend_handletextpad=0.35,
        subplot_adjust={"left": 0.15, "right": 0.94, "bottom": 0.21, "top": 0.77},
    ),
    "paper": Layout(
        figsize=(6.493, 3.226),
        font_size=11.0,
        label_size=14.0,
        tick_size=11.0,
        legend_size=11.0,
        raw_linewidth=1.0,
        smooth_linewidth=2.4,
        grid_linewidth=0.8,
        legend_ncols=3,
        legend_anchor=(0.5, 1.17),
        legend_handlelength=2.6,
        legend_columnspacing=1.8,
        legend_labelspacing=0.5,
        legend_handletextpad=0.8,
        subplot_adjust={"left": 0.083, "right": 0.97, "bottom": 0.17, "top": 0.88},
    ),
}


def configure_matplotlib(layout: Layout) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": layout.font_size,
            "axes.labelsize": layout.label_size,
            "xtick.labelsize": layout.tick_size,
            "ytick.labelsize": layout.tick_size,
            "legend.fontsize": layout.legend_size,
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
    metric: str,
    max_step: int,
    smooth_window: int,
    out: Path,
    layout_name: str,
) -> list[Path]:
    layout = LAYOUTS[layout_name]
    configure_matplotlib(layout)
    fig, ax = plt.subplots(figsize=layout.figsize)

    for label, group in grouped_series(series):
        color = DISPLAY_COLORS.get(label, "#333333")
        for run_series in group:
            ax.plot(
                run_series.steps,
                run_series.values,
                color=color,
                linewidth=layout.raw_linewidth,
                alpha=0.18,
                solid_capstyle="round",
            )
        smooth_steps, smoothed = group_smoothed_curve(group, smooth_window, max_step)
        ax.plot(
            smooth_steps,
            smoothed,
            color=color,
            linewidth=layout.smooth_linewidth,
            alpha=0.98,
            label=label,
            solid_capstyle="round",
        )

    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0, max_step + 1, 50))
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Average num. steps")
    ax.grid(True, color="#D3D3D3", linewidth=layout.grid_linewidth)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=layout.legend_anchor,
        ncols=layout.legend_ncols,
        handlelength=layout.legend_handlelength,
        columnspacing=layout.legend_columnspacing,
        labelspacing=layout.legend_labelspacing,
        handletextpad=layout.legend_handletextpad,
    )
    fig.subplots_adjust(**layout.subplot_adjust)
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
        raise SystemExit(
            "--entity and --project are required unless WANDB_ENTITY and "
            "WANDB_PROJECT are set."
        )
    if args.max_step < 0:
        raise SystemExit("--max-step must be non-negative.")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1.")

    wandb = require_wandb()
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    series = fetch_all_series(api, project_path, runs, args.metric, args.max_step)
    paths = plot_series(
        series,
        args.metric,
        args.max_step,
        args.smooth_window,
        args.out,
        args.layout,
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
