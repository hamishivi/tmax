"""Analysis modules for the dataset-comparison suite.

Each ``module_*`` function takes a :class:`~rl_data.comparison.core.RunContext`
(holding multiple :class:`~rl_data.comparison.core.DatasetSpec`), produces
its figures + CSV sidecars, and returns a structured summary dict consumed by
:mod:`rl_data.comparison.cli` to populate the overall report.

Modules:

* :func:`module_difficulty` — pass@1, pass@4, pass@8, turns, tokens, cost.
* :func:`module_command_mix` — what kinds of actions the agent takes.
* :func:`module_composition` — projection of each dataset onto our taxonomy.
* :func:`module_diversity` — shared-axis TF-IDF clustering.
* :func:`module_realism` — pkg installs, services, artifacts.
* :func:`module_verifier` — assertion count + types.

All plots are emitted through :func:`core.save_fig_with_data` so a ``.csv``
with the underlying numbers always ships next to each ``.png``.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from rl_data.comparison.command_taxonomy import CATEGORIES, classify_one
from rl_data.comparison.core import (
    COMMAND_COMPLEXITY_ORDER,
    DOMAINS_ORDER,
    SKILL_TYPES_ORDER,
    DatasetSpec,
    RunContext,
    TASK_COMPLEXITY_ORDER,
    balance_score,
    chi_squared,
    effective_command_complexity,
    effective_domain,
    effective_skill_type,
    effective_task_complexity,
    fmt_p,
    grouped_bar,
    histogram_overlay,
    iter_bash_commands,
    mann_whitney,
    nanmean,
    nanmedian,
    save_fig_with_data,
    summary_basename,
    write_csv,
)
from rl_data.comparison.styles import (
    PassAtKOverlayStyle,
    StackedCompositionStyle,
    default_pass_at_k_overlay_style,
    default_stacked_style,
    is_dark_color,
    palette_colors,
    pretty_label,
)
from rl_data.analyze import _PRICING, _estimate_cost

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# module_difficulty
# ---------------------------------------------------------------------------


def _pass_at_k(record: Dict[str, Any], k: int) -> Optional[float]:
    """Look up pass@K for a record, falling back gracefully when K is absent.

    ``analyze._load_summary`` materialises only ``pass@1`` as a top-level
    record key; every other K lives in the dense ``pass_at_k_full`` map. The
    historical comparison code did ``r.get("pass@8")`` which silently
    returned ``None`` for every record — this helper is the harness-aware
    replacement.
    """
    if k == 1:
        v = record.get("pass@1")
        if v is not None:
            return v
    pak = record.get("pass_at_k_full") or {}
    val = pak.get(k)
    return float(val) if val is not None else None


def module_difficulty(ctx: RunContext) -> Dict[str, Any]:
    """pass@1 / pass@4 / pass@8 / turns / tokens headline + pass@k overlay + turn CDF."""
    summary_per_spec: Dict[str, Dict[str, Any]] = {}
    solved_per_spec: Dict[str, List[Dict[str, Any]]] = {}

    for spec in ctx.specs:
        solved = [r for r in ctx.records_of(spec) if r.get("has_solutions")]
        solved_per_spec[spec.name] = solved
        summary_per_spec[spec.name] = {
            "n_tasks": len(ctx.records_of(spec)),
            "n_with_solutions": len(solved),
            "mean_pass_at_1": nanmean([_pass_at_k(r, 1) for r in solved]),
            "mean_pass_at_4": nanmean([_pass_at_k(r, 4) for r in solved]),
            "mean_pass_at_8": nanmean([_pass_at_k(r, 8) for r in solved]),
            "mean_turns": nanmean([r.get("avg_turns") for r in solved]),
            "median_turns": nanmedian([r.get("avg_turns") for r in solved]),
            "mean_tokens_per_run": nanmean([r.get("avg_tokens_est") for r in solved]),
            "mean_initial_input_tokens": nanmean(
                [r.get("avg_initial_input_tokens") for r in solved]
            ),
            "mean_peak_input_tokens": nanmean(
                [r.get("avg_peak_input_tokens") for r in solved]
            ),
            "mean_final_output_tokens": nanmean(
                [r.get("avg_final_output_tokens") for r in solved]
            ),
            "total_input_tokens": sum(r.get("total_input_tokens", 0) for r in solved),
            "total_output_tokens": sum(r.get("total_output_tokens", 0) for r in solved),
        }

    # ---- MAIN: headline grouped bar -------------------------------------
    metrics = [
        ("pass@1", "mean_pass_at_1"),
        ("pass@4", "mean_pass_at_4"),
        ("pass@8", "mean_pass_at_8"),
        ("avg turns", "mean_turns"),
        ("avg tok/run (k)", "mean_tokens_per_run"),
    ]
    categories = [m[0] for m in metrics]
    series: List[List[float]] = []
    data_rows: List[Dict[str, Any]] = []
    for spec in ctx.specs:
        s = summary_per_spec[spec.name]
        vals = [
            s["mean_pass_at_1"],
            s["mean_pass_at_4"],
            s["mean_pass_at_8"],
            s["mean_turns"],
            s["mean_tokens_per_run"] / 1000.0,
        ]
        series.append(vals)
        for (mlabel, _), v in zip(metrics, vals):
            data_rows.append({"dataset": spec.name, "metric": mlabel, "value": v})

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped_bar(
        ax, categories, series, ctx.specs,
        ylabel="value", title=f"Difficulty headline — {ctx.model_slug.replace('_', '/')}",
        annotate=True, value_fmt="{:.2f}",
    )
    save_fig_with_data(
        fig, data_rows, ctx.main_dir / "fig1_difficulty_headline",
        fieldnames=["dataset", "metric", "value"],
    )

    # ---- APPENDIX: pass@k overlay ---------------------------------------
    # The on-disk filename is harness-dependent: bash runs write
    # ``<MODEL>_summary.json``, vanillux writes ``<MODEL>_vanillux_summary.json``.
    # Resolve once via ctx.summary_basename so this stays in sync with the
    # naming convention defined in rl_data.generate_solutions.
    _summary_name = ctx.summary_basename

    def _pass_curve(recs: List[Dict[str, Any]]) -> Optional[Dict[int, float]]:
        agg: Dict[int, List[float]] = defaultdict(list)
        for r in recs:
            p = Path(r["dir"]) / "solutions" / _summary_name
            if not p.exists():
                continue
            try:
                sol = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for k, v in sol.get("pass_at_k", {}).items():
                agg[int(k)].append(float(v))
        if not agg:
            return None
        return {k: sum(v) / len(v) for k, v in agg.items()}

    curves: Dict[str, Dict[int, float]] = {}
    ks_union: set[int] = set()
    for spec in ctx.specs:
        c = _pass_curve(solved_per_spec[spec.name])
        if c:
            curves[spec.name] = c
            ks_union |= set(c.keys())

    if curves:
        ks = sorted(ks_union)
        # Pull dataset colours from the same palette family as the stacked
        # composition figure so the pass@k overlay reads as part of the same
        # figure suite in the writeup. We allocate one palette slot per
        # *plotted* spec (not every spec — some may have no curve), to keep
        # neighbouring lines on visually distinct hues even when a dataset
        # drops out (e.g. when R2E-Gym is excluded).
        style = default_pass_at_k_overlay_style()
        plotted_specs = [s for s in ctx.specs if curves.get(s.name)]
        line_colors = palette_colors(style.palette_name, max(len(plotted_specs), 1))

        rc_overrides = {
            "text.usetex": style.use_tex,
            "font.family": style.font_family,
            "font.serif": list(style.font_serif),
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.axes_label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.legend_size,
            "axes.spines.top": style.show_top_right_spines,
            "axes.spines.right": style.show_top_right_spines,
        }

        with plt.rc_context(rc_overrides):
            fig, ax = plt.subplots(figsize=style.figsize)
            rows: List[Dict[str, Any]] = []
            for color, spec in zip(line_colors, plotted_specs):
                c = curves[spec.name]
                ys = [c.get(k, float("nan")) for k in ks]
                ax.plot(
                    ks, ys, "o-",
                    color=color,
                    label=spec.display_name,
                    linewidth=style.line_width,
                    markersize=style.marker_size,
                    markerfacecolor=color,
                    markeredgecolor=style.marker_edge_color,
                    markeredgewidth=style.marker_edge_width,
                )
                for k, y in zip(ks, ys):
                    rows.append({"dataset": spec.name, "k": k, "mean_pass_at_k": y})

            ax.set_xticks(ks)
            ax.set_xlabel(style.xlabel)
            ax.set_ylabel(style.ylabel)
            ax.set_ylim(*style.ylim)
            if style.grid:
                ax.grid(True, alpha=style.grid_alpha, linestyle=style.grid_linestyle)
            ax.tick_params(axis="both", which="both", length=0)
            for spine_name in ("left", "bottom"):
                ax.spines[spine_name].set_color(style.spine_color)
                ax.spines[spine_name].set_linewidth(style.spine_linewidth)
            ax.set_title(
                style.title, pad=style.title_pad,
                fontweight=style.title_weight, loc=style.title_loc,
            )
            ax.legend(
                loc=style.legend_loc, frameon=style.legend_frameon,
                ncol=style.legend_ncol, title=style.legend_title,
            )
            fig.tight_layout()
            save_fig_with_data(
                fig, rows, ctx.appendix_dir / "difficulty_pass_at_k_overlay",
                fieldnames=["dataset", "k", "mean_pass_at_k"],
                dpi=style.dpi, also_pdf=style.save_pdf,
            )

    # ---- APPENDIX: turn CDF ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    plotted = False
    for spec in ctx.specs:
        xs = sorted(r["avg_turns"] for r in solved_per_spec[spec.name]
                    if r.get("avg_turns") is not None)
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, color=spec.color, label=spec.display_name, linewidth=2)
        plotted = True
        for x, y in zip(xs, ys):
            rows.append({"dataset": spec.name, "avg_turns": x, "cdf": y})
    if plotted:
        ax.set_xlabel("Mean turns to completion (per task)")
        ax.set_ylabel("CDF over tasks")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Distribution of turns-per-task", fontweight="bold")
        save_fig_with_data(
            fig, rows, ctx.appendix_dir / "difficulty_turn_cdf",
            fieldnames=["dataset", "avg_turns", "cdf"],
        )

    # ---- Stats vs reference --------------------------------------------
    # Note: pass@K for K>1 is *not* materialised as a top-level record key by
    # analyze._load_summary — only ``pass@1`` is. The pre-0515 code wrote
    # ``r["pass@8"]`` directly and silently always got None back, so the old
    # report's pass@8 Mann-Whitney p-values were a bug, not data. Use the
    # _pass_at_k helper which falls back to pass_at_k_full[k] for K>1.
    ref = ctx.reference
    ref_solved = solved_per_spec[ref.name]
    p_values: Dict[str, Dict[str, Optional[float]]] = {}

    def _pass_vals(recs: List[Dict[str, Any]], k: int) -> List[float]:
        return [_pass_at_k(r, k) for r in recs if _pass_at_k(r, k) is not None]

    for spec in ctx.baselines:
        base_solved = solved_per_spec[spec.name]
        p_values[spec.name] = {
            "pass@1": mann_whitney(_pass_vals(ref_solved, 1), _pass_vals(base_solved, 1)),
            "pass@4": mann_whitney(_pass_vals(ref_solved, 4), _pass_vals(base_solved, 4)),
            "pass@8": mann_whitney(_pass_vals(ref_solved, 8), _pass_vals(base_solved, 8)),
            "avg_turns": mann_whitney(
                [r["avg_turns"] for r in ref_solved if r.get("avg_turns") is not None],
                [r["avg_turns"] for r in base_solved if r.get("avg_turns") is not None],
            ),
        }

    # ---- Cost estimate --------------------------------------------------
    cost_per_task: Dict[str, Optional[float]] = {}
    if ctx.model_slug in _PRICING:
        for spec in ctx.specs:
            solved = solved_per_spec[spec.name]
            ti = sum(r.get("total_input_tokens", 0) for r in solved)
            to = sum(r.get("total_output_tokens", 0) for r in solved)
            cost_per_task[spec.name] = (
                _estimate_cost(ti, to, ctx.model_slug) / max(1, len(solved))
            )
    else:
        for spec in ctx.specs:
            cost_per_task[spec.name] = None

    return {
        "per_dataset": summary_per_spec,
        "p_values_vs_reference": p_values,
        "cost_per_task_usd": cost_per_task,
    }


# ---------------------------------------------------------------------------
# module_command_mix
# ---------------------------------------------------------------------------


def _load_trace_features(
    task_dir: Path,
    model_slug: str,
    harness: str = "bash",
) -> Optional[Dict[str, Any]]:
    """Distil command-mix features from a single task's per-model summary.

    ``harness`` selects between ``<MODEL>_summary.json`` (bash) and
    ``<MODEL>_vanillux_summary.json`` (vanillux); see
    :func:`rl_data.comparison.core.summary_basename`.
    """
    summary = task_dir / "solutions" / summary_basename(model_slug, harness)
    commands = list(iter_bash_commands(summary))
    if not commands:
        return None

    turn_tag_counts: Counter = Counter()
    tag_any: set = set()
    for cmd in commands:
        for t in classify_one(cmd):
            turn_tag_counts[t] += 1
            tag_any.add(t)
    return {
        "total_turns": len(commands),
        "turn_tag_counts": dict(turn_tag_counts),
        "tag_any": sorted(tag_any),
        "distinct_categories": len(tag_any - {"other"}),
    }


def module_command_mix(ctx: RunContext) -> Dict[str, Any]:
    """Coverage + distinct-categories histogram + per-turn distribution + cooccurrence."""
    feats_per_spec: Dict[str, List[Dict[str, Any]]] = {}
    for spec in ctx.specs:
        fs = []
        for r in ctx.records_of(spec):
            f = _load_trace_features(Path(r["dir"]), ctx.model_slug, ctx.harness)
            if f:
                fs.append(f)
        feats_per_spec[spec.name] = fs

    # Warn if any spec has zero traces (common during bootstrapping).
    for spec in ctx.specs:
        if not feats_per_spec[spec.name]:
            logger.warning("command_mix: no traces for %s — charts will still render with zeros",
                           spec.name)

    # ---- Coverage (MAIN) ------------------------------------------------
    def coverage(fs: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(fs)
        counts: Counter = Counter()
        for f in fs:
            for t in f["tag_any"]:
                counts[t] += 1
        return {c: (counts.get(c, 0) / n) if n else 0.0 for c in CATEGORIES}

    cov_per_spec = {s.name: coverage(feats_per_spec[s.name]) for s in ctx.specs}

    series = [[cov_per_spec[s.name][c] * 100 for c in CATEGORIES] for s in ctx.specs]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(
        ax, CATEGORIES, series, ctx.specs,
        ylabel="% of tasks using category",
        title=f"Command-category coverage per task — {ctx.model_slug.replace('_', '/')}",
        ylim=(0, 100),
    )
    rows = []
    for s in ctx.specs:
        for cat in CATEGORIES:
            rows.append({
                "dataset": s.name,
                "category": cat,
                "pct_tasks": cov_per_spec[s.name][cat] * 100,
            })
    save_fig_with_data(
        fig, rows, ctx.main_dir / "fig2_command_mix_coverage",
        fieldnames=["dataset", "category", "pct_tasks"],
    )

    # ---- Distinct categories histogram (APPENDIX) -----------------------
    per_task_distinct = {s.name: [f["distinct_categories"] for f in feats_per_spec[s.name]]
                          for s in ctx.specs}
    max_dc = max((max(v, default=0) for v in per_task_distinct.values()), default=0)
    if max_dc > 0:
        bins = np.arange(0, max_dc + 2) - 0.5
        fig, ax = plt.subplots(figsize=(7, 4.5))
        histogram_overlay(
            ax,
            [per_task_distinct[s.name] for s in ctx.specs],
            ctx.specs,
            bins=bins,
            xlabel="distinct command categories per task",
            ylabel="fraction of tasks", density=True,
            title="Recipe breadth: categories touched per task",
        )
        rows = []
        for s in ctx.specs:
            for v in per_task_distinct[s.name]:
                rows.append({"dataset": s.name, "distinct_categories": v})
        save_fig_with_data(
            fig, rows, ctx.appendix_dir / "command_mix_distinct_categories_hist",
            fieldnames=["dataset", "distinct_categories"],
        )

    # ---- Per-turn distribution (APPENDIX) -------------------------------
    def turn_dist(fs: List[Dict[str, Any]]) -> Dict[str, float]:
        agg: Counter = Counter()
        for f in fs:
            for t, c in f["turn_tag_counts"].items():
                agg[t] += c
        total = sum(agg.values())
        return {c: (agg.get(c, 0) / total) if total else 0.0 for c in CATEGORIES}

    td_per_spec = {s.name: turn_dist(feats_per_spec[s.name]) for s in ctx.specs}
    series = [[td_per_spec[s.name][c] * 100 for c in CATEGORIES] for s in ctx.specs]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(
        ax, CATEGORIES, series, ctx.specs,
        ylabel="% of all agent turns",
        title="Per-turn command-category distribution",
    )
    rows = []
    for s in ctx.specs:
        for cat in CATEGORIES:
            rows.append({
                "dataset": s.name,
                "category": cat,
                "pct_turns": td_per_spec[s.name][cat] * 100,
            })
    save_fig_with_data(
        fig, rows, ctx.appendix_dir / "command_mix_turn_distribution",
        fieldnames=["dataset", "category", "pct_turns"],
    )

    # ---- Cooccurrence heatmaps (APPENDIX) ------------------------------
    def cooc(fs: List[Dict[str, Any]]) -> np.ndarray:
        n = len(CATEGORIES)
        m = np.zeros((n, n), dtype=float)
        idx = {c: i for i, c in enumerate(CATEGORIES)}
        for f in fs:
            ts = [t for t in f["tag_any"] if t in idx]
            for a in ts:
                for b in ts:
                    m[idx[a]][idx[b]] += 1
        if len(fs):
            m /= len(fs)
        return m

    for spec in ctx.specs:
        m = cooc(feats_per_spec[spec.name])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CATEGORIES, fontsize=8)
        ax.set_title(f"Category co-occurrence — {spec.display_name}", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of tasks")
        rows = [
            {"row_category": CATEGORIES[i], "col_category": CATEGORIES[j], "frac": float(m[i][j])}
            for i in range(len(CATEGORIES))
            for j in range(len(CATEGORIES))
        ]
        save_fig_with_data(
            fig, rows, ctx.appendix_dir / f"command_mix_cooccurrence_{spec.name}",
            fieldnames=["row_category", "col_category", "frac"],
        )

    # ---- Stats ---------------------------------------------------------
    ref = ctx.reference
    p_distinct: Dict[str, Optional[float]] = {}
    ref_vals = per_task_distinct[ref.name]
    for spec in ctx.baselines:
        p_distinct[spec.name] = mann_whitney(ref_vals, per_task_distinct[spec.name])

    return {
        "n_traces": {s.name: len(feats_per_spec[s.name]) for s in ctx.specs},
        "coverage": cov_per_spec,
        "turn_distribution": td_per_spec,
        "distinct_categories": {
            s.name: {
                "mean": nanmean(per_task_distinct[s.name]),
                "median": nanmedian(per_task_distinct[s.name]),
            }
            for s in ctx.specs
        },
        "p_values_vs_reference_distinct_categories": p_distinct,
    }


# ---------------------------------------------------------------------------
# module_composition
# ---------------------------------------------------------------------------


def _bucket_records(
    recs: List[Dict[str, Any]],
    getter,
    order: Sequence[str],
) -> Dict[str, int]:
    counts: Counter = Counter()
    for r in recs:
        tj = r.get("_task_json") or {}
        v = getter(tj)
        if v is None:
            continue
        counts[v] += 1
    # Make sure every bucket appears even if count is 0.
    return {k: int(counts.get(k, 0)) for k in order}


def _collect_composition(
    ctx: RunContext, getter, order: Sequence[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """Return (counts_per_dataset, pct_per_dataset) over the given bucket order."""
    bucket_counts: Dict[str, Dict[str, int]] = {}
    bucket_pct: Dict[str, Dict[str, float]] = {}
    for spec in ctx.specs:
        recs = ctx.records_of(spec)
        counts = _bucket_records(recs, getter, order)
        total = sum(counts.values())
        bucket_counts[spec.name] = counts
        bucket_pct[spec.name] = {
            k: (counts[k] / total * 100) if total else 0.0 for k in order
        }
    return bucket_counts, bucket_pct


def _composition_csv_rows(
    specs: Sequence[DatasetSpec],
    order: Sequence[str],
    bucket_counts: Dict[str, Dict[str, int]],
    bucket_pct: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows = []
    for s in specs:
        for k in order:
            rows.append({
                "dataset": s.name,
                "bucket": k,
                "n_tasks": bucket_counts[s.name][k],
                "pct_tasks": bucket_pct[s.name][k],
            })
    return rows


def _render_ridgeline(
    ctx: RunContext,
    order: Sequence[str],
    bucket_counts: Dict[str, Dict[str, int]],
    bucket_pct: Dict[str, Dict[str, float]],
    *,
    path_base: Path,
    title: str,
    xlabel: str,
    figsize_height_per_row: float = 1.4,
) -> None:
    """Ridgeline-style chart for discrete categorical distributions.

    One row per dataset, step-filled area showing % of that dataset's tasks in
    each bucket.  All rows share the x-axis (bucket order) so shapes are
    visually comparable; each row has its own y-axis baseline like a ridgeline.
    """
    specs = ctx.specs
    n = len(specs)
    fig, axes = plt.subplots(
        n, 1, sharex=True,
        figsize=(max(9, 0.55 * len(order)), figsize_height_per_row * n),
    )
    if n == 1:
        axes = [axes]

    x = np.arange(len(order))
    y_max = max(
        (max(bucket_pct[s.name].get(k, 0.0) for k in order) for s in specs),
        default=0.0,
    )
    y_top = max(1e-6, y_max * 1.15)

    for ax, spec in zip(axes, specs):
        y = np.array([bucket_pct[spec.name].get(k, 0.0) for k in order])
        ax.fill_between(x, 0, y, step="mid", color=spec.color, alpha=0.55)
        ax.step(x, y, where="mid", color=spec.color, linewidth=2.0)
        ax.scatter(x, y, color=spec.color, s=18, zorder=3)
        ax.set_ylim(0, y_top)
        ax.set_ylabel(spec.display_name, rotation=0, ha="right", va="center",
                      fontsize=10, labelpad=12)
        # Keep only the bottom spine for that ridgeline look.
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_yticks([y_top])
        ax.set_yticklabels([f"{y_top:.0f}%"], fontsize=8, color="#888")
        ax.grid(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [pretty_label(b) for b in order],
        rotation=30, ha="right", fontsize=9,
    )
    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(title, fontweight="bold", loc="left")

    fig.tight_layout(h_pad=0.3)
    rows = _composition_csv_rows(specs, order, bucket_counts, bucket_pct)
    save_fig_with_data(
        fig, rows, path_base,
        fieldnames=["dataset", "bucket", "n_tasks", "pct_tasks"],
    )


def _render_radar(
    ctx: RunContext,
    order: Sequence[str],
    bucket_pct: Dict[str, Dict[str, float]],
    *,
    path_base: Path,
    title: str,
) -> None:
    """Polar radar chart — one polygon per dataset.

    Works well up to ~15 axes; for larger taxonomies prefer :func:`_render_ridgeline`.
    """
    specs = ctx.specs
    n_axes = len(order)
    if n_axes < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    y_max = max(
        (max(bucket_pct[s.name].get(k, 0.0) for k in order) for s in specs),
        default=0.0,
    )
    ax.set_ylim(0, max(1e-6, y_max * 1.12))

    for spec in specs:
        y = [bucket_pct[spec.name].get(k, 0.0) for k in order]
        y += y[:1]
        ax.plot(angles, y, color=spec.color, linewidth=2.0, label=spec.display_name)
        ax.fill(angles, y, color=spec.color, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([pretty_label(b) for b in order], fontsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#888")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    rows = []
    for s in specs:
        for k in order:
            rows.append({"dataset": s.name, "bucket": k,
                         "pct_tasks": bucket_pct[s.name].get(k, 0.0)})
    save_fig_with_data(
        fig, rows, path_base,
        fieldnames=["dataset", "bucket", "pct_tasks"],
    )


def _render_stacked_composition(
    ctx: RunContext,
    order: Sequence[str],
    bucket_counts: Dict[str, Dict[str, int]],
    bucket_pct: Dict[str, Dict[str, float]],
    *,
    path_base: Path,
    style: Optional[StackedCompositionStyle] = None,
) -> None:
    """Vertical stacked-bar composition figure.

    One bar per dataset; each bar is partitioned into colored segments, one
    per bucket in ``order``.  Inspired by the Tülu 2 / RDS dataset-mix
    figures: the goal is to make it visually obvious at a glance which
    datasets are concentrated on a few buckets vs. spread evenly.

    Modes:

    * ``style.normalize=True`` (default) — every bar reaches 100 %.  Best
      when datasets have very different sizes; emphasises *shape*.
    * ``style.normalize=False`` — bars retain absolute task counts.  Useful
      when the size differential is itself part of the story.

    All typographic / color choices are owned by ``style``
    (:class:`~rl_data.comparison.styles.StackedCompositionStyle`) so this
    figure can be polished independently of the others.
    """
    if style is None:
        style = default_stacked_style()

    specs = ctx.specs
    n_buckets = len(order)
    if n_buckets == 0 or not specs:
        return
    colors = palette_colors(style.palette_name, n_buckets)

    if style.normalize:
        values: Dict[str, Dict[str, float]] = bucket_pct
        ymax = 100.0 * 1.02
    else:
        values = {
            s.name: {k: float(bucket_counts[s.name][k]) for k in order}
            for s in specs
        }
        ymax = max(
            (sum(values[s.name][k] for k in order) for s in specs),
            default=0.0,
        ) * 1.08

    rc_overrides = {
        "text.usetex": style.use_tex,
        "font.family": style.font_family,
        "font.serif": list(style.font_serif),
        "font.size": style.font_size,
        "axes.titlesize": style.title_size,
        "axes.labelsize": style.axes_label_size,
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.legend_size,
        "axes.spines.top": style.show_top_right_spines,
        "axes.spines.right": style.show_top_right_spines,
    }

    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=style.figsize)

        x = np.arange(len(specs)) * (style.bar_width + style.bar_gap)
        bottoms = np.zeros(len(specs), dtype=float)

        for j, bucket in enumerate(order):
            seg = np.array([values[s.name][bucket] for s in specs], dtype=float)
            color = colors[j]
            # ``bucket`` is the canonical key (e.g. ``software_engineering``);
            # we keep it as the matplotlib label internally so legend handle
            # de-duplication still works, but render the human-readable form
            # in the actual legend draw below via ``pretty_label``.
            bars = ax.bar(
                x, seg, width=style.bar_width, bottom=bottoms,
                color=color, edgecolor=style.bar_edge_color,
                linewidth=style.bar_edge_linewidth, label=pretty_label(bucket),
            )
            if style.annotate_segments:
                text_color = (
                    style.annotate_color_dark_bg
                    if is_dark_color(color) else style.annotate_color_light_bg
                )
                for bar, v in zip(bars, seg):
                    if v < style.annotate_min_pct:
                        continue
                    cx = bar.get_x() + bar.get_width() / 2
                    cy = bar.get_y() + bar.get_height() / 2
                    label = f"{v:.1f}%" if style.normalize else f"{int(round(v))}"
                    ax.text(
                        cx, cy, label, ha="center", va="center",
                        fontsize=style.annotation_size,
                        fontweight=style.annotation_weight,
                        color=text_color,
                    )
            bottoms += seg

        if style.annotate_total_above_bars and not style.normalize:
            for xi, total in zip(x, bottoms):
                ax.text(
                    xi, total + ymax * 0.01, f"{int(round(total))}",
                    ha="center", va="bottom",
                    fontsize=style.annotation_size,
                    color=style.annotate_color_light_bg,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.display_name for s in specs],
            rotation=style.xticklabel_rotation,
            ha=style.xticklabel_ha,
            rotation_mode="anchor",
        )
        ax.tick_params(axis="both", which="both", length=0)
        for spine_name in ("left", "bottom"):
            ax.spines[spine_name].set_color(style.spine_color)
            ax.spines[spine_name].set_linewidth(style.spine_linewidth)

        ax.set_ylim(0, ymax)
        ax.set_ylabel(style.ylabel)
        if style.xlabel:
            ax.set_xlabel(style.xlabel)
        ax.set_title(
            style.title, pad=style.title_pad,
            fontweight=style.title_weight, loc=style.title_loc,
        )

        handles, labels = ax.get_legend_handles_labels()
        if style.legend_reverse:
            handles = handles[::-1]
            labels = labels[::-1]
        ax.legend(
            handles, labels,
            loc=style.legend_loc, bbox_to_anchor=style.legend_bbox,
            frameon=style.legend_frameon, ncol=style.legend_ncol,
            title=style.legend_title,
        )

        fig.tight_layout()
        rows = _composition_csv_rows(specs, order, bucket_counts, bucket_pct)
        save_fig_with_data(
            fig, rows, path_base,
            fieldnames=["dataset", "bucket", "n_tasks", "pct_tasks"],
            dpi=style.dpi, also_pdf=style.save_pdf,
        )


def _render_grouped_bar(
    ctx: RunContext,
    order: Sequence[str],
    bucket_counts: Dict[str, Dict[str, int]],
    bucket_pct: Dict[str, Dict[str, float]],
    *,
    path_base: Path,
    title: str,
) -> None:
    """Classic grouped bar — kept as a secondary/appendix view."""
    series = [[bucket_pct[s.name][k] for k in order] for s in ctx.specs]
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(order)), 5))
    grouped_bar(
        ax, list(order), series, ctx.specs,
        ylabel="% of tasks", title=title,
        ylim=(0, max(1e-6, max(max(row) for row in series) * 1.15)),
    )
    rows = _composition_csv_rows(ctx.specs, order, bucket_counts, bucket_pct)
    save_fig_with_data(
        fig, rows, path_base,
        fieldnames=["dataset", "bucket", "n_tasks", "pct_tasks"],
    )


def _compose_axis(
    ctx: RunContext,
    field: str,
    getter,
    order: Sequence[str],
    *,
    main_base: Optional[Path] = None,
    radar_base: Optional[Path] = None,
    stacked_base: Optional[Path] = None,
    stacked_style: Optional[StackedCompositionStyle] = None,
    appendix_base: Optional[Path] = None,
    ridgeline_title: str,
    radar_title: Optional[str] = None,
    stacked_title: Optional[str] = None,
    bar_title: Optional[str] = None,
    xlabel: str,
) -> Dict[str, Any]:
    """Compute counts + render any subset of {ridgeline, radar, stacked, bar}.

    Always returns a summary dict with counts / pct / balance / chi2 — the
    caller decides which figures to render.
    """
    bucket_counts, bucket_pct = _collect_composition(ctx, getter, order)

    if main_base is not None:
        _render_ridgeline(
            ctx, order, bucket_counts, bucket_pct,
            path_base=main_base, title=ridgeline_title, xlabel=xlabel,
        )
    if radar_base is not None:
        _render_radar(
            ctx, order, bucket_pct,
            path_base=radar_base, title=radar_title or ridgeline_title,
        )
    if stacked_base is not None:
        style = stacked_style or default_stacked_style()
        if stacked_title is not None:
            style.title = stacked_title
        if style.xlabel is None and xlabel:
            style.xlabel = None  # leave x-axis unlabeled — datasets are obvious
        _render_stacked_composition(
            ctx, order, bucket_counts, bucket_pct,
            path_base=stacked_base, style=style,
        )
    if appendix_base is not None:
        _render_grouped_bar(
            ctx, order, bucket_counts, bucket_pct,
            path_base=appendix_base, title=bar_title or ridgeline_title,
        )

    # Balance scalar per dataset on this axis.
    balance: Dict[str, float] = {
        s.name: balance_score([bucket_counts[s.name][k] for k in order])
        for s in ctx.specs
    }

    # Chi-squared baseline-vs-reference.
    ref = ctx.reference
    p_vs_ref: Dict[str, Optional[float]] = {}
    for spec in ctx.baselines:
        p_vs_ref[spec.name] = chi_squared(
            [bucket_counts[ref.name][k] for k in order],
            [bucket_counts[spec.name][k] for k in order],
        )

    return {
        "field": field,
        "buckets": list(order),
        "counts": bucket_counts,
        "pct": bucket_pct,
        "balance": balance,
        "chi2_p_vs_reference": p_vs_ref,
    }


def module_composition(ctx: RunContext) -> Dict[str, Any]:
    """Composition comparison using OUR taxonomy as the canonical axis.

    External datasets rely on classifier output (``classified_*`` fields on each
    ``task.json``); tasks that were not classified contribute nothing.

    Main-body figures:
        * ``main/fig3_composition_domain_ridgeline.png``  — one row per dataset
        * ``main/fig4_composition_domain_radar.png``      — polar view (9 axes)
        * ``main/fig5_composition_skill_type_ridgeline.png`` — sorted-by-ours

    Appendix:
        * ``composition_domain_bar.png``             — classic grouped bar
        * ``composition_task_complexity.png``        — 3-bucket grouped bar
        * ``composition_command_complexity.png``    — 3-bucket grouped bar

    Each axis also contributes a single-scalar ``balance`` per dataset
    (:func:`~rl_data.comparison.core.balance_score`) used in the headline
    summary table.
    """
    out: Dict[str, Any] = {}

    # ── Domain: ridgeline + radar + stacked (main) + classic bar (appendix) ──
    #
    # The stacked figure (fig6) is the "headline" composition view for the
    # paper.  Style is fully configurable via StackedCompositionStyle in
    # rl_data/comparison/styles.py — each plot can be polished independently.
    # Palette can also be overridden at run time via STACKED_PALETTE env var.
    import os
    palette_name = os.environ.get("STACKED_PALETTE", "anthropic_book")
    out["domain"] = _compose_axis(
        ctx, "domain",
        getter=lambda tj: effective_domain(tj),
        order=DOMAINS_ORDER,
        main_base=ctx.main_dir / "fig3_composition_domain_ridgeline",
        radar_base=ctx.main_dir / "fig4_composition_domain_radar",
        stacked_base=ctx.main_dir / "fig6_composition_domain_stacked",
        stacked_style=default_stacked_style(
            title="Domain composition",
            palette_name=palette_name,
        ),
        appendix_base=ctx.appendix_dir / "composition_domain_bar",
        ridgeline_title="Domain composition (ridgeline)",
        radar_title="Domain composition (radar)",
        stacked_title="Domain composition",
        bar_title="Domain composition (grouped bar)",
        xlabel="domain",
    )

    # ── Skill-type: ridgeline (main), sorted by our frequency so visual shape
    #    highlights imbalance in baselines; 29 buckets is too much for a radar.
    if any(
        effective_skill_type(r.get("_task_json") or {})
        for spec in ctx.specs
        for r in ctx.records_of(spec)
    ):
        ref = ctx.reference
        ref_counts = _bucket_records(
            ctx.records_of(ref), lambda tj: effective_skill_type(tj), SKILL_TYPES_ORDER,
        )
        active_skills = [
            s for s in sorted(SKILL_TYPES_ORDER, key=lambda k: -ref_counts.get(k, 0))
            if any(
                any(effective_skill_type(r.get("_task_json") or {}) == s
                    for r in ctx.records_of(spec))
                for spec in ctx.specs
            )
        ]
        if active_skills:
            out["skill_type"] = _compose_axis(
                ctx, "skill_type",
                getter=lambda tj: effective_skill_type(tj),
                order=active_skills,
                main_base=ctx.main_dir / "fig5_composition_skill_type_ridgeline",
                appendix_base=ctx.appendix_dir / "composition_skill_type_bar",
                ridgeline_title="Skill-type composition (sorted by ours)",
                bar_title="Skill-type composition (grouped bar)",
                xlabel="skill_type (sorted by reference frequency)",
            )

    # ── Complexity axes: only 3 buckets each → grouped bar in appendix is plenty
    out["task_complexity"] = _compose_axis(
        ctx, "task_complexity",
        getter=lambda tj: effective_task_complexity(tj),
        order=TASK_COMPLEXITY_ORDER,
        appendix_base=ctx.appendix_dir / "composition_task_complexity",
        ridgeline_title="Task complexity composition",
        bar_title="Task complexity composition",
        xlabel="task_complexity",
    )
    out["command_complexity"] = _compose_axis(
        ctx, "command_complexity",
        getter=lambda tj: effective_command_complexity(tj),
        order=COMMAND_COMPLEXITY_ORDER,
        appendix_base=ctx.appendix_dir / "composition_command_complexity",
        ridgeline_title="Command complexity composition",
        bar_title="Command complexity composition",
        xlabel="command_complexity",
    )

    # Coverage health (how many external tasks actually got classified).
    coverage: Dict[str, Dict[str, int]] = {}
    for spec in ctx.specs:
        recs = ctx.records_of(spec)
        total = len(recs)
        classified = sum(
            1 for r in recs
            if effective_domain(r.get("_task_json") or {}) is not None
        )
        coverage[spec.name] = {
            "total": total,
            "with_domain_label": classified,
            "frac": (classified / total) if total else 0.0,
        }
    out["label_coverage"] = coverage
    return out


# ---------------------------------------------------------------------------
# module_diversity
# ---------------------------------------------------------------------------


def module_diversity(ctx: RunContext) -> Dict[str, Any]:
    """Shared TF-IDF cluster analysis on the union corpus + cardinality snapshot."""
    descs_per_spec: Dict[str, List[str]] = {}
    idx_per_spec: Dict[str, List[int]] = {}
    for spec in ctx.specs:
        recs = ctx.records_of(spec)
        descs = []
        for r in recs:
            tj = r.get("_task_json") or {}
            d = (tj.get("description") or "").strip()
            descs.append(d if d else "")
        descs_per_spec[spec.name] = descs

    corpus = []
    offset = 0
    for spec in ctx.specs:
        idx_per_spec[spec.name] = list(range(offset, offset + len(descs_per_spec[spec.name])))
        corpus.extend(descs_per_spec[spec.name])
        offset += len(descs_per_spec[spec.name])

    if not any(corpus) or sum(1 for d in corpus if d.strip()) < 20:
        logger.warning("diversity: corpus too small; skipping shared-axis analysis")
        return {"shared": None}

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("diversity: scikit-learn not installed; skipping")
        return {"shared": None}

    # Sparse TF-IDF over the union corpus.
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          stop_words="english", min_df=3, max_df=0.8)
    try:
        X = vec.fit_transform(corpus)
    except ValueError:
        logger.warning("diversity: TF-IDF failed (empty vocab?); skipping")
        return {"shared": None}

    # ---- Lexical diversity (sampled pairwise cosine distance) -----------
    def sampled_pair_distance(indices: List[int]) -> float:
        if len(indices) < 2:
            return 0.0
        M = X[indices]
        rng = random.Random(ctx.seed)
        ia = [rng.choice(indices) for _ in range(ctx.sample_pairs)]
        ib = [rng.choice(indices) for _ in range(ctx.sample_pairs)]
        ia_local = [indices.index(i) for i in ia]
        ib_local = [indices.index(i) for i in ib]
        # Use sparse matmul for speed.
        sims = np.asarray((M[ia_local].multiply(M[ib_local])).sum(axis=1)).ravel()
        return float(1.0 - sims.mean())

    # For efficiency on large datasets: sample 4000 rows per dataset to compute.
    lex_div: Dict[str, float] = {}
    for spec in ctx.specs:
        idxs = idx_per_spec[spec.name]
        if len(idxs) > 4000:
            rng = random.Random(ctx.seed)
            idxs = rng.sample(idxs, 4000)
        lex_div[spec.name] = sampled_pair_distance(idxs)

    # ---- Shared KMeans clusters ----------------------------------------
    k = min(50, max(5, X.shape[0] // 50))
    km = MiniBatchKMeans(n_clusters=k, random_state=ctx.seed, batch_size=2048, n_init="auto")
    labels = km.fit_predict(X)

    clust_per_spec: Dict[str, Counter] = {
        spec.name: Counter(int(labels[i]) for i in idx_per_spec[spec.name])
        for spec in ctx.specs
    }

    def eff_clusters(counts: Counter, total: int) -> float:
        if total <= 0:
            return 0.0
        probs = [c / total for c in counts.values() if c > 0]
        h = -sum(p * math.log(p) for p in probs)
        return math.exp(h)

    shared = {
        "n_clusters": k,
        "lexical_diversity": lex_div,
        "eff_clusters": {
            s.name: eff_clusters(clust_per_spec[s.name], len(idx_per_spec[s.name]))
            for s in ctx.specs
        },
        "covered_clusters": {
            s.name: sum(1 for cid in range(k) if clust_per_spec[s.name][cid] > 0)
            for s in ctx.specs
        },
    }
    # Unique-cluster counts vs reference.
    ref = ctx.reference
    shared["unique_vs_reference"] = {}
    for spec in ctx.baselines:
        only_ref = sum(
            1 for cid in range(k)
            if clust_per_spec[ref.name][cid] > 0 and clust_per_spec[spec.name][cid] == 0
        )
        only_base = sum(
            1 for cid in range(k)
            if clust_per_spec[spec.name][cid] > 0 and clust_per_spec[ref.name][cid] == 0
        )
        shared["unique_vs_reference"][spec.name] = {
            "reference_only_clusters": only_ref,
            "baseline_only_clusters": only_base,
            "shared_clusters": sum(
                1 for cid in range(k)
                if clust_per_spec[spec.name][cid] > 0 and clust_per_spec[ref.name][cid] > 0
            ),
        }

    # ---- Stacked cluster bar (APPENDIX) ---------------------------------
    cluster_ids = sorted(range(k), key=lambda i: -sum(
        clust_per_spec[s.name][i] for s in ctx.specs
    ))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(k)
    rows = []
    for spec in ctx.specs:
        vals = np.array([clust_per_spec[spec.name][i] for i in cluster_ids])
        ax.bar(range(k), vals, bottom=bottom, color=spec.color, label=spec.display_name)
        for pos, cid in enumerate(cluster_ids):
            rows.append({
                "cluster_rank": pos, "cluster_id": cid,
                "dataset": spec.name, "n_tasks": int(clust_per_spec[spec.name][cid]),
            })
        bottom += vals
    ax.set_xlabel("cluster id (sorted by total size)")
    ax.set_ylabel("# tasks")
    ax.set_title(
        f"Shared TF-IDF clusters (k={k})", fontweight="bold",
    )
    ax.legend()
    save_fig_with_data(
        fig, rows, ctx.appendix_dir / "diversity_shared_clusters",
        fieldnames=["cluster_rank", "cluster_id", "dataset", "n_tasks"],
    )

    # ---- Top keywords per cluster (appendix CSV) ------------------------
    try:
        vocab = vec.get_feature_names_out()
        centers = km.cluster_centers_
        rows = []
        for cid in range(k):
            top_idx = np.argsort(-centers[cid])[:6]
            row = {"cluster_id": cid, "top_keywords": ", ".join(vocab[i] for i in top_idx)}
            for spec in ctx.specs:
                row[f"n_{spec.name}"] = int(clust_per_spec[spec.name][cid])
            rows.append(row)
        write_csv(
            ctx.appendix_dir / "diversity_clusters.csv",
            rows,
            fieldnames=["cluster_id", "top_keywords"] + [f"n_{s.name}" for s in ctx.specs],
        )
    except Exception as e:
        logger.warning("diversity: cluster keyword dump failed: %s", e)

    return {"shared": shared}


# ---------------------------------------------------------------------------
# module_realism
# ---------------------------------------------------------------------------


_APT_RE = re.compile(r"apt(?:-get)?\s+(?:-\S+\s+)*install\s+([^\n;&|]+)")
_PIP_RE = re.compile(r"pip3?\s+install\s+([^\n;&|]+)")
_SERVICE_RE = re.compile(
    r"\b(systemctl|service)\s+(start|restart|enable)\b|"
    r"\b(nginx|redis-server|postgres(?:ql)?|mysqld|mongod)\b"
)
_ABS_PATH_RE = re.compile(r"[\"'](/[^\"'\s]+)[\"']")


def _realism_of(task_dir: Path, description: str) -> Dict[str, Any]:
    cdef_path = task_dir / "container.def"
    cdef = cdef_path.read_text() if cdef_path.exists() else ""

    apt_pkgs: set = set()
    for m in _APT_RE.findall(cdef):
        for tok in m.replace("\\\n", " ").split():
            if tok and not tok.startswith("-") and not tok.startswith(">"):
                apt_pkgs.add(tok)
    pip_pkgs: set = set()
    for m in _PIP_RE.findall(cdef):
        for tok in m.replace("\\\n", " ").split():
            if tok and not tok.startswith("-") and tok not in ("&&", "||"):
                pip_pkgs.add(tok.split("==")[0].split(">")[0].split("<")[0])
    n_services = len(_SERVICE_RE.findall(cdef))

    artifacts: set = set()
    ttest = task_dir / "test_final_state.py"
    if ttest.exists():
        try:
            text = ttest.read_text()
            for m in _ABS_PATH_RE.findall(text):
                if m.startswith(("/home/user", "/tmp", "/var")):
                    artifacts.add(m)
        except OSError:
            pass

    return {
        "desc_chars": len(description),
        "desc_words": len(description.split()),
        "n_apt_pkgs": len(apt_pkgs),
        "n_pip_pkgs": len(pip_pkgs),
        "n_services": n_services,
        "n_artifacts_checked": len(artifacts),
    }


def module_realism(ctx: RunContext) -> Dict[str, Any]:
    feats_per_spec: Dict[str, List[Dict[str, Any]]] = {}
    for spec in ctx.specs:
        fs = []
        for r in ctx.records_of(spec):
            tj = r.get("_task_json") or {}
            fs.append(_realism_of(Path(r["dir"]), tj.get("description", "")))
        feats_per_spec[spec.name] = fs

    metrics = [
        ("desc_words", "Description length (words)"),
        ("n_apt_pkgs", "# apt packages installed"),
        ("n_pip_pkgs", "# pip packages installed"),
        ("n_services", "# services started"),
        ("n_artifacts_checked", "# artifacts checked by verifier"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    rows = []
    for ax, (key, title) in zip(axes, metrics):
        series = [[f[key] for f in feats_per_spec[s.name]] for s in ctx.specs]
        max_v = max((max(s, default=0) for s in series), default=0)
        bins = np.linspace(0, max_v + 1, 20) if max_v > 0 else np.arange(3)
        histogram_overlay(
            ax, series, ctx.specs, bins=bins,
            xlabel="", ylabel="fraction", title=title,
        )
        for spec, vals in zip(ctx.specs, series):
            for v in vals:
                rows.append({"dataset": spec.name, "metric": key, "value": v})
    save_fig_with_data(
        fig, rows, ctx.appendix_dir / "realism_histograms",
        fieldnames=["dataset", "metric", "value"],
    )

    ref = ctx.reference
    p_vs_ref: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in ctx.baselines:
        p_vs_ref[spec.name] = {
            key: mann_whitney(
                [f[key] for f in feats_per_spec[ref.name]],
                [f[key] for f in feats_per_spec[spec.name]],
            )
            for key, _ in metrics
        }

    per_dataset_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in ctx.specs:
        fs = feats_per_spec[spec.name]
        per_dataset_summary[spec.name] = {
            key: {
                "mean": nanmean([f[key] for f in fs]),
                "median": nanmedian([f[key] for f in fs]),
            }
            for key, _ in metrics
        }

    return {
        "per_dataset": per_dataset_summary,
        "p_values_vs_reference": p_vs_ref,
    }


# ---------------------------------------------------------------------------
# module_verifier
# ---------------------------------------------------------------------------


ASSERT_TYPES = [
    "equality", "ordering", "membership", "identity", "type_check",
    "file_exists", "regex_match", "numeric_tolerance",
    "subprocess_check", "call_based", "negation", "bool_op", "other",
]


def _classify_assert(expr: ast.AST) -> str:
    if isinstance(expr, ast.Compare):
        ops = expr.ops
        if any(isinstance(o, (ast.Eq, ast.NotEq)) for o in ops):
            return "equality"
        if any(isinstance(o, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in ops):
            return "ordering"
        if any(isinstance(o, (ast.In, ast.NotIn)) for o in ops):
            return "membership"
        if any(isinstance(o, (ast.Is, ast.IsNot)) for o in ops):
            return "identity"
    if isinstance(expr, ast.Call):
        name = _call_name(expr.func)
        if name in {"isinstance", "issubclass"}:
            return "type_check"
        if name in {"os.path.isfile", "os.path.isdir", "os.path.exists", "Path.exists"}:
            return "file_exists"
        if name.endswith((".search", ".match", ".fullmatch")):
            return "regex_match"
        if name in {"math.isclose", "numpy.isclose", "np.isclose", "approx"}:
            return "numeric_tolerance"
        if "subprocess" in name or name.endswith(("check_call", "check_output", ".run")):
            return "subprocess_check"
        return "call_based"
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return "negation"
    if isinstance(expr, ast.BoolOp):
        return "bool_op"
    return "other"


def _call_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _verifier_features(task_dir: Path) -> Dict[str, Any]:
    p = task_dir / "test_final_state.py"
    default = {
        "loc": 0, "n_test_functions": 0, "n_asserts": 0,
        "assertion_types": Counter(), "uses_subprocess": False,
    }
    if not p.exists():
        return default
    try:
        src = p.read_text()
        # Suppress `SyntaxWarning: invalid escape sequence` emitted on 3.12+
        # for third-party test files containing regex literals in non-raw
        # strings (e.g. "\\s", "\\d"). These are cosmetic and orthogonal to our
        # AST traversal. Passing `filename=str(p)` so any real SyntaxErrors
        # still point at the offending file.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src, filename=str(p))
    except (OSError, SyntaxError):
        return default

    n_test_fns = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    )
    assertion_types: Counter = Counter()
    n_asserts = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Assert):
            n_asserts += 1
            assertion_types[_classify_assert(n.test)] += 1

    return {
        "loc": src.count("\n") + 1,
        "n_test_functions": n_test_fns,
        "n_asserts": n_asserts,
        "assertion_types": assertion_types,
        "uses_subprocess": "subprocess" in src,
    }


def module_verifier(ctx: RunContext) -> Dict[str, Any]:
    feats_per_spec: Dict[str, List[Dict[str, Any]]] = {}
    for spec in ctx.specs:
        feats_per_spec[spec.name] = [
            _verifier_features(Path(r["dir"])) for r in ctx.records_of(spec)
        ]

    # Assertion-type distribution (APPENDIX).
    def dist(fs: List[Dict[str, Any]]) -> Dict[str, float]:
        agg: Counter = Counter()
        for f in fs:
            for t, c in f["assertion_types"].items():
                agg[t] += c
        total = sum(agg.values())
        return {t: (agg.get(t, 0) / total) if total else 0.0 for t in ASSERT_TYPES}

    dist_per_spec = {s.name: dist(feats_per_spec[s.name]) for s in ctx.specs}
    series = [[dist_per_spec[s.name][t] * 100 for t in ASSERT_TYPES] for s in ctx.specs]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(
        ax, ASSERT_TYPES, series, ctx.specs,
        ylabel="% of all asserts",
        title="Verifier assertion-type distribution",
    )
    rows = []
    for s in ctx.specs:
        for t in ASSERT_TYPES:
            rows.append({"dataset": s.name, "assertion_type": t,
                         "pct": dist_per_spec[s.name][t] * 100})
    save_fig_with_data(
        fig, rows, ctx.appendix_dir / "verifier_assertion_types",
        fieldnames=["dataset", "assertion_type", "pct"],
    )

    # LOC + #asserts (APPENDIX).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rows = []
    for ax, (key, title) in zip(axes, [("loc", "Verifier LOC"), ("n_asserts", "# assert statements")]):
        series = [[f[key] for f in feats_per_spec[s.name]] for s in ctx.specs]
        max_v = max((max(s, default=0) for s in series), default=0)
        bins = np.linspace(0, max_v + 1, 30) if max_v > 0 else np.arange(3)
        histogram_overlay(
            ax, series, ctx.specs, bins=bins,
            xlabel="", ylabel="fraction", title=title,
        )
        for s, vals in zip(ctx.specs, series):
            for v in vals:
                rows.append({"dataset": s.name, "metric": key, "value": v})
    save_fig_with_data(
        fig, rows, ctx.appendix_dir / "verifier_loc_asserts",
        fieldnames=["dataset", "metric", "value"],
    )

    # Per-dataset summaries + stats.
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for spec in ctx.specs:
        fs = feats_per_spec[spec.name]
        per_dataset[spec.name] = {
            "mean_loc": nanmean([f["loc"] for f in fs]),
            "mean_asserts": nanmean([f["n_asserts"] for f in fs]),
            "pct_uses_subprocess": (
                sum(1 for f in fs if f["uses_subprocess"]) / max(1, len(fs))
            ),
        }

    ref = ctx.reference
    p_vs_ref: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in ctx.baselines:
        p_vs_ref[spec.name] = {
            "loc": mann_whitney(
                [f["loc"] for f in feats_per_spec[ref.name]],
                [f["loc"] for f in feats_per_spec[spec.name]],
            ),
            "n_asserts": mann_whitney(
                [f["n_asserts"] for f in feats_per_spec[ref.name]],
                [f["n_asserts"] for f in feats_per_spec[spec.name]],
            ),
        }

    return {
        "per_dataset": per_dataset,
        "p_values_vs_reference": p_vs_ref,
        "assertion_type_distribution": dist_per_spec,
    }


__all__ = [
    "module_difficulty",
    "module_command_mix",
    "module_composition",
    "module_diversity",
    "module_realism",
    "module_verifier",
]
