"""Teaser figure: Terminal-Bench 2.0 performance vs (rough) model size.

The figure is the paper's opener: a scatter where the *y*-axis is the TB2.0
score and the *x*-axis is model size — but the size axis is deliberately **not**
a plain log scale.  Models bunch up at 2-32B and again in the 100B-1T frontier,
so a literal log axis wastes space and crushes the small-model region.  Instead
we use a *piecewise-linear-in-log* transform with hand-placed knots (see
``X_KNOTS``) that gives each "tier" roughly equal horizontal real-estate, parks
the undisclosed-size closed models at a dedicated ``?`` column on the far right,
and draws a visual **axis break** between the ``1T`` tier and ``?``.

Each point is drawn with its brand **logo as the marker**, centred on the exact
(x, y), with a vertical **error bar** (± std) through it and the model **name**
as a nearby annotation.  Models without a logo fall back to a small coloured
dot.  Logos are normalised to ~the same on-figure size (per-model fine-tune via
``logo_scale``).  A shaded overlay marks the "very large / undisclosed size"
region on the right, and an **axis break** separates the ``1T`` tier from the
``?`` column of closed models whose size is undisclosed.

Run::

    uv run python scripts/plot/teaser_terminal_bench.py

The default CSV and display configuration match the TMax v1.1 teaser. Scores
are percentages; missing standard deviations are unknown, not zero.

Styling mirrors ``rl_data/comparison/styles.py`` (DejaVu Serif, no top/right
spines, light-grey axes, Anthropic-warm palette) so the teaser reads as part of
the same figure family as the rest of the paper.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
DEFAULT_CSV = HERE / "terminal_bench_results.csv"
DEFAULT_OUT = HERE / "output" / "teaser_terminal_bench"


def _load_styles_module():
    """Load ``rl_data/comparison/styles.py`` *directly by file path*.

    We avoid ``from rl_data.comparison.styles import ...`` because importing it
    as a package runs ``rl_data/__init__.py``, which imports ``litellm`` (≈4s
    cold).  ``styles.py`` only uses the stdlib, so loading the file in isolation
    keeps it the single source of truth for the palette while staying fast.
    """
    styles_path = HERE.parent.parent / "rl_data" / "comparison" / "styles.py"
    spec = importlib.util.spec_from_file_location("_tmax_paper_styles", styles_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so dataclasses in styles.py resolve
    spec.loader.exec_module(module)
    return module


# Reuse the paper's colour helper so this figure matches the others.
darken_hex = _load_styles_module().darken_hex


# ===========================================================================
# 1. X-AXIS TRANSFORM  (the "untraditional" part)
# ===========================================================================
#
# Map a model's size (billions of params) to an x position by piecewise-linear
# interpolation **in log10(size)** through these knots.  Each consecutive knot
# pair gets one unit of horizontal space, so the 2->4->8->32B and 32B->...->1T
# jumps all *look* comparably wide despite spanning very different ranges.
#
#                       size(B)  ->  x-position
# NB: the 8B->32B segment is deliberately given *two* units of width (1 -> 3)
# rather than one, because that's where most models live and the labels need
# breathing room.  Every other segment gets one unit.
X_KNOTS: List[Tuple[float, float]] = [
    (2.0, -1.0),
    (4.0, 0.0),
    (8.0, 1.0),
    (32.0, 3.0),     # <- stretched: 8B->32B spans 2 units for extra room
    (128.0, 4.0),
    (1024.0, 5.0),
]
# Closed models with no public size (Claude, GPT-5, ...) are parked here, to the
# right of an axis break.
UNKNOWN_X: float = 6.0
BREAK_X: float = 5.5          # where the // axis break is drawn

# Where to draw x ticks and what to call them.
X_TICKS: List[Tuple[float, str]] = [
    (-1.0, "2B"),
    (0.0, "4B"),
    (1.0, "8B"),
    (2.0, "16B"),    # midpoint of the stretched 8B->32B segment (= exactly 16B)
    (3.0, "32B"),
    (4.0, "128B"),
    (5.0, "1T"),
    (UNKNOWN_X, "?"),
]
X_LIM: Tuple[float, float] = (-1.5, 6.5)


def size_to_x(size: Optional[float]) -> float:
    """Map a size in billions of params to an x position (see ``X_KNOTS``).

    ``None`` / NaN (undisclosed) maps to :data:`UNKNOWN_X`.  Sizes outside the
    knot range are linearly extrapolated (in log space) from the nearest segment.
    """
    if size is None or (isinstance(size, float) and np.isnan(size)):
        return UNKNOWN_X
    xs = np.log10([k[0] for k in X_KNOTS])
    ys = [k[1] for k in X_KNOTS]
    lx = float(np.log10(size))
    if lx <= xs[0]:
        return ys[0] + (lx - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if lx >= xs[-1]:
        return ys[-1] + (lx - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(lx, xs, ys))


# ===========================================================================
# 2. CATEGORY COLOURS  (drives both dots and the legend)
# ===========================================================================

GROUP_COLORS: Dict[str, str] = {
    "ours":         "#C15F3C",  # terracotta — the hero colour
    "our_baseline": "#4E6E94",  # denim
    "past_work":    "#9C6E3F",  # caramel
    "open_weights": "#3F8B85",  # teal
    "closed":       "#8E5F94",  # plum
}
GROUP_LABELS: Dict[str, str] = {
    "ours":         "TMax (ours)",
    "our_baseline": "Our base models",
    "past_work":    "Prior terminal agents",
    "open_weights": "Open-weight (best)",
    "closed":       "Closed (best)",
}
GROUP_ORDER: List[str] = ["ours", "our_baseline", "past_work", "open_weights", "closed"]


# ===========================================================================
# 2b. OPEN-RECIPE HIGHLIGHT  (ring around models whose full recipe is public)
# ===========================================================================
#
# A subset of models publish their *full* training recipe.  We circle those
# with a coloured ring whose colour encodes whether the recipe is SFT-only or
# RL-based, so the reader can see at a glance which open recipes our RL approach
# is being compared against.
#
#   recipe key in RECIPE -> "sft" | "rl"
#
RECIPE: Dict[str, str] = {
    # SFT-only open recipes
    "Nemotron-Terminal-8B":  "sft",
    "Nemotron-Terminal-14B": "sft",
    "Nemotron-Terminal-32B": "sft",
    "TerminalTraj-7B":       "sft",
    "TerminalTraj-14B":      "sft",
    "TerminalTraj-32B":      "sft",
    "TermiGen-Qwen2.5-32B":  "sft",
    "LiberCoder-32B":        "sft",
    "LiberCoder-235B-A22B":  "sft",
    "OpenThinker-Agent-32B": "sft",
    "CalibForge-30B":        "sft",
    "CalibForge-35B":        "sft",
    "Lego-32B":             "sft",
    # RL open recipes
    "Endless-Terminals-8B":  "rl",
    "OpenThinker-Agent-V1":  "rl",
    "TMax-27B":              "rl",
    "TMax-9B":               "rl",
    "TMax-4B":               "rl",
    "TMax-2B":               "rl",
    "OpenThinker-Agent-RL-8B": "rl",
    "SETA-V4-Flash":          "rl",
    "RST-122B":              "rl",
}
RECIPE_COLORS: Dict[str, str] = {
    "sft": "#4E6E94",   # denim
    "rl":  "#C15F3C",   # terracotta (matches the hero / our RL approach)
}
RECIPE_LABELS: Dict[str, str] = {
    "sft": "Open recipe (SFT)",
    "rl":  "Open recipe (RL)",
}
RECIPE_ORDER: List[str] = ["sft", "rl"]


# ===========================================================================
# 3. PER-MODEL DISPLAY CONFIG  (hand-tune everything here)
# ===========================================================================
#
# Keyed by the exact ``model_name`` in the CSV.  Every field is optional:
#
#   logo        : filename in scripts/plot/media (PNG). None -> coloured dot.
#                 When set, the logo *is the marker*, centred on the exact point.
#   logo_scale  : per-model multiplier on the (otherwise uniform) logo size.
#   label       : annotation text (defaults to the model name; "" hides it).
#   dx, dy      : name-label offset from the point, in points (default: below).
#   ha, va      : name-label alignment.
#   x_offset    : nudge the point horizontally, in x *data* units.  Kept at 0 for
#                 every sized model so it sits at its TRUE size on the axis; only
#                 used to fan out the undisclosed-size closed models (all share
#                 the single "?" column, so some spread there is unavoidable).
#   leader      : True -> draw a thin dashed line from the point to the label
#                 (for crowded / coincident markers where the label is offset far).
#   show        : set False to omit the model entirely.
#
# Anything not listed falls back to DEFAULTS below.

OURS_LOGO = "TMax-Logo-2-transparent.png"
QWEN_LOGO = "Qwen_logo.png"
OPENAI_LOGO = "openai.png"
NVIDIA_LOGO = "nvidia-logo.png"
DEEPSEEK_LOGO = "deepseek-ai-icon-seeklogo.png"

MODEL_STYLE: Dict[str, dict] = {
    # ---- ours (TMax) — the hero -------------------------------------------
    "TMax-27B": dict(logo=OURS_LOGO, logo_scale=1.25, label="TMax-27B",
                     dx=33, dy=2, ha="left", va="center"),
    "TMax-9B": dict(logo=OURS_LOGO, logo_scale=1.25, label="TMax-9B",
                    dx=32, dy=2, ha="left", va="center"),
    "TMax-4B": dict(logo=OURS_LOGO, logo_scale=1.1, label="TMax-4B",
                    dx=29, dy=2, ha="left", va="center"),
    "TMax-2B": dict(logo=OURS_LOGO, logo_scale=0.92, label="TMax-2B",
                    x_offset=0.0, dx=26, dy=3, ha="left", va="center"),
    # ---- our base models (Qwen) -------------------------------------------
    "Qwen3.5-2B": dict(logo=QWEN_LOGO, logo_scale=0.68, label="Qwen3.5-2B",
                       x_offset=0.0, dx=17, dy=2, ha="left", va="center"),
    "Qwen3.5-4B": dict(logo=QWEN_LOGO, logo_scale=0.85, label="Qwen3.5-4B",
                       dx=18, dy=1, ha="left", va="center"),
    "Qwen3-8B": dict(logo=QWEN_LOGO, logo_scale=0.90, label="Qwen3-8B",
                     dx=16, dy=6, ha="left", va="center"),
    "Qwen3.5-9B": dict(logo=QWEN_LOGO, label="Qwen3.5-9B",
                       dx=-13, dy=0, ha="right", va="center"),
    "Qwen3.6-27B": dict(logo=QWEN_LOGO, logo_scale=0.82, label="Qwen3.6-27B",
                        x_offset=0.0, dx=30, dy=-20, ha="left", va="top", leader=True),
    # ---- prior terminal agents --------------------------------------------
    "Nemotron-Terminal-8B": dict(logo=NVIDIA_LOGO, logo_scale=0.63, label="Nemotron 8B",
                                 dx=20, dy=-2, ha="left", va="center", leader=True),
    "Nemotron-Terminal-14B": dict(logo=NVIDIA_LOGO, logo_scale=0.66, label="Nemotron 14B",
                                  dx=17, dy=8, ha="left", va="bottom", leader=True),
    "Nemotron-Terminal-32B": dict(logo=NVIDIA_LOGO, logo_scale=0.70, label="Nemotron 32B",
                                  dx=-35, dy=15, ha="right", va="bottom", leader=True),
    "TermiGen-Qwen2.5-32B": dict(label="TermiGen-32B", leader=True,
                                 dx=5, dy=-42, ha="left", va="top"),
    "Endless-Terminals-8B": dict(label="Endless-8B",
                                 dx=14, dy=0, ha="left", va="center"),
    "OpenThinker-Agent-V1": dict(label="OpenThinker-Agent-V1",
                                 dx=14, dy=0, ha="left", va="center"),
    "TerminalTraj-7B": dict(label="TerminalTraj-7B",
                            dx=0, dy=-7, ha="center", va="top"),
    "TerminalTraj-14B": dict(label=""),
    "TerminalTraj-32B": dict(label="TerminalTraj 32B", leader=True,
                             dx=19, dy=0, ha="left", va="center"),
    "LiberCoder-32B": dict(label="LiberCoder-32B", leader=True,
                           dx=-5, dy=-20, ha="right", va="top"),
    "LiberCoder-235B-A22B": dict(label="LiberCoder 235B", leader=True,
                                 dx=-19, dy=-14, ha="right", va="top"),
    "OpenThinker-Agent-32B": dict(label="OpenThinker 32B", leader=True,
                                  dx=19, dy=2, ha="left", va="center"),
    "OpenThinker-Agent-RL-8B": dict(label="OT-Agent RL 8B", leader=True,
                                    dx=13, dy=13, ha="left", va="bottom"),
    "SETA-V4-Flash": dict(label="SETA V4 Flash", leader=True,
                          dx=-12, dy=-7, ha="right", va="top"),
    "CalibForge-30B": dict(label="CalibForge 30B", leader=True,
                           dx=0, dy=18, ha="center", va="bottom"),
    "CalibForge-35B": dict(label="CalibForge 35B", leader=True,
                           dx=15, dy=-7, ha="left", va="center"),
    "Lego-32B": dict(label="Lego 32B", leader=True,
                     dx=-33, dy=0, ha="right", va="center"),
    "RST-122B": dict(label="RST 122B", leader=True,
                     dx=20, dy=-12, ha="left", va="center"),
    # ---- open-weight frontier ---------------------------------------------
    "GPT-OSS-120B": dict(logo=OPENAI_LOGO, logo_scale=0.75, label="GPT-OSS-120B",
                         dx=24, dy=-5, ha="left", va="center", leader=True),
    "Kimi-K2.5": dict(logo="Kimi-logo-2025.png", logo_scale=0.95, label="Kimi-K2.5",
                      dx=18, dy=0, ha="left", va="center"),
    "MiniMax-M2.7": dict(logo="minimax-logo.png", logo_scale=1.0, label="",
                         dx=-12, dy=-16, ha="center", va="top"),
    "DeepSeek-V3.2": dict(logo=DEEPSEEK_LOGO, logo_scale=1.0, label="DeepSeek-V3.2",
                          dx=0, dy=-20, ha="center", va="top"),
    "GLM-5": dict(logo="GLM-Zai-Logo.png", logo_scale=0.82, label="GLM-5",
                  dx=13, dy=-14, ha="left", va="center"),
    # ---- closed (undisclosed size; live in the "?" column) ----------------
    "Claude-Sonnet-4.6": dict(logo="Claude_AI_logo.png", logo_scale=0.95,
                              label="Sonnet 4.6",
                              dx=0, dy=-12, ha="center", va="center"),
    "Claude-Haiku-4.5": dict(logo="Claude_AI_logo.png", logo_scale=0.78,
                             label="Haiku 4.5", x_offset=-0.22,
                             dx=0, dy=-17, ha="center", va="top"),
    "GPT-5-mini": dict(logo=OPENAI_LOGO, logo_scale=0.95, label="GPT-5-mini",
                       x_offset=0.22, dx=-5, dy=15, ha="right", va="bottom"),
    "GPT-5-nano": dict(logo=OPENAI_LOGO, logo_scale=0.75, label="GPT-5-nano",
                       x_offset=0.0, dx=0, dy=-14, ha="center", va="top"),
}

DEFAULTS = dict(
    logo=None,
    logo_scale=1.0,
    label=None,
    dx=0.0,
    dy=-15.0,
    ha="center",
    va="top",
    x_offset=0.0,
    leader=False,
    show=True,
)


def style_for(model: str) -> dict:
    """Return the merged display config for ``model`` (defaults + overrides)."""
    cfg = dict(DEFAULTS)
    cfg.update(MODEL_STYLE.get(model, {}))
    if cfg["label"] is None:
        cfg["label"] = model
    return cfg


# ===========================================================================
# 4. FIGURE STYLE
# ===========================================================================


@dataclass
class TeaserStyle:
    """All knobs for the teaser figure (mirrors the paper's style family)."""

    # Typography
    font_family: str = "serif"
    font_serif: Sequence[str] = field(default_factory=lambda: ["DejaVu Serif"])
    font_size: float = 18.0
    axes_label_size: float = 20.0
    tick_size: float = 17.0
    annotation_size: float = 13.0
    annotation_size_ours: float = 15.0
    legend_size: float = 14.0

    # Layout / spines
    figsize: Tuple[float, float] = (14.0, 8.5)
    spine_color: str = "#808080"
    spine_linewidth: float = 1.25

    # Grid
    grid: bool = True
    grid_alpha: float = 0.16
    grid_linestyle: str = "--"

    # Title (dropped by default per author feedback) / axis labels
    title: str = ""
    title_pad: float = 16.0
    title_weight: str = "bold"
    xlabel: str = "Model size (billions of parameters)"
    ylabel: str = "Terminal-Bench 2.0 (%)"
    ylim: Tuple[float, float] = (0.0, 60.0)

    # Logos — normalised so each occupies ~the same bounding box (logo == marker).
    logo_target_in: float = 0.40   # target *max* dimension of every logo, inches

    # Dots — fallback marker for models without a logo
    dot_size: float = 70.0
    dot_color: str = "#8A7B6B"      # muted warm grey for no-logo baselines
    dot_edge_color: str = "white"
    dot_edge_width: float = 1.1

    # Emphasis for "ours" (halo off — the bold caption alone carries it)
    ours_halo: bool = False
    ours_halo_size: float = 620.0   # scatter `s` for the soft halo
    ours_halo_alpha: float = 0.20
    annotation_color: str = "#4A433B"   # label text colour (non-ours)

    # Error bars (vertical, ± std from the CSV)
    show_errorbars: bool = False
    errorbar_alpha: float = 0.5
    errorbar_lw: float = 1.3
    errorbar_capsize: float = 3.0
    errorbar_color: str = "#7A7065"     # neutral so it doesn't fight the logos

    # Open-recipe highlight rings (colour encodes SFT vs RL)
    recipe_rings: bool = True
    ring_recipes: Tuple[str, ...] = ("sft", "rl")   # which recipe kinds get a ring
    ring_lw: float = 2.4
    ring_alpha: float = 0.6
    ring_logo_mult: float = 1.42   # ring diameter = logo target * logo_scale * this
    ring_dot_in: float = 0.165     # ring diameter (inches) for no-logo dot markers

    # Pareto frontier: the upper-left envelope across *all* size-disclosed models
    # (smaller is better, higher is better).  A soft band hugs the *upper* side of
    # the line so the frontier reads as a clean ridge rather than a heavy block.
    pareto: bool = True
    pareto_color: str = "#C15F3C"       # fill tint
    pareto_fill_alpha: float = 0.10     # whole region above the line is shaded
    pareto_line_color: str = "#E0A78F"  # lighter, so it doesn't fight the labels
    pareto_line_alpha: float = 0.85
    pareto_lw: float = 2.0
    pareto_line_style: Tuple = (0, (7, 4))  # dashed
    pareto_label: str = ""              # floating on-plot label (legend note instead)
    pareto_label_xy: Tuple[float, float] = (2.15, 33.5)
    pareto_legend_label: str = "Pareto frontier"

    # "Very large / undisclosed size" overlay band (right side)
    frontier_band: bool = True
    frontier_x: float = 3.5             # left edge of the shaded band
    frontier_color: str = "#C9BBA9"
    frontier_alpha: float = 0.16
    frontier_label: str = "Very large  /  undisclosed size"
    frontier_label_y: float = 25.0

    # Dashed leader lines (point -> label) for crowded coincident markers
    leader_color: str = "#888888"
    leader_lw: float = 0.8
    leader_alpha: float = 0.5
    leader_linestyle: str = "--"

    # Legend (off by default — markers are mostly logos, not coloured dots)
    legend: bool = False
    legend_loc: str = "upper left"
    legend_bbox: Tuple[float, float] = (0.01, 0.99)

    # Output
    dpi: int = 220
    save_pdf: bool = True


def _apply_rc(style: TeaserStyle) -> None:
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.serif": list(style.font_serif),
            "font.size": style.font_size,
            "axes.labelsize": style.axes_label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "svg.fonttype": "none",
        }
    )


# ===========================================================================
# 5. DATA LOADING
# ===========================================================================


@dataclass
class Row:
    model: str
    group: str
    size_b: Optional[float]
    tb2: float
    std: Optional[float]
    x: float


def load_rows(csv_path: Path) -> Tuple[List[Row], List[str]]:
    """Load plottable rows; return ``(rows, skipped)`` (skipped = no TB2 yet)."""
    rows: List[Row] = []
    skipped: List[str] = []
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            model = (r.get("model_name") or "").strip()
            if not model:
                continue
            tb2_raw = (r.get("tb2") or "").strip()
            if not tb2_raw:                      # TODO / unreleased -> not ready
                skipped.append(model)
                continue
            size_raw = (r.get("size_b") or "").strip()
            std_raw = (r.get("tb2_std") or "").strip()
            size = float(size_raw) if size_raw else None
            rows.append(
                Row(
                    model=model,
                    group=(r.get("group") or "past_work").strip(),
                    size_b=size,
                    tb2=float(tb2_raw),
                    std=float(std_raw) if std_raw else None,
                    x=size_to_x(size),
                )
            )
    return rows, skipped


# ===========================================================================
# 6. LOGO HANDLING
# ===========================================================================

_LOGO_CACHE: Dict[str, np.ndarray] = {}


def _load_logo(name: str) -> np.ndarray:
    if name not in _LOGO_CACHE:
        _LOGO_CACHE[name] = np.asarray(Image.open(MEDIA / name).convert("RGBA"))
    return _LOGO_CACHE[name]


def _logo_zoom(img: np.ndarray, style: TeaserStyle, scale: float) -> float:
    """Zoom so the logo's *max* dimension equals ``logo_target_in`` inches.

    OffsetImage renders ``zoom * native_pixels`` points, so a constant target in
    inches is ``target_in * 72 / max(h, w)`` — uniform on-figure size regardless
    of source resolution; ``scale`` is the per-model fine-tune multiplier.
    """
    h, w = img.shape[:2]
    return (style.logo_target_in * 72.0 / max(h, w)) * scale


# ===========================================================================
# 7. RENDER
# ===========================================================================


def _draw_x_break(ax, style: TeaserStyle) -> None:
    """Draw a // break on the bottom spine between the 1T tier and the ? column."""
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    # White-out a short gap on the bottom spine.
    ax.plot([BREAK_X - 0.12, BREAK_X + 0.12], [0, 0], transform=trans,
            color="white", lw=style.spine_linewidth + 2.0, clip_on=False, zorder=11)
    # Two parallel diagonal slashes.
    for s in (-0.06, 0.06):
        ax.plot([BREAK_X + s - 0.07, BREAK_X + s + 0.07], [-0.022, 0.022],
                transform=trans, color=style.spine_color,
                lw=style.spine_linewidth, clip_on=False, zorder=12)


def _pareto_front(rows: List[Row]) -> List[Row]:
    """Return the Pareto-optimal rows (minimise size, maximise score).

    A model is on the frontier if no other model is both *no larger* and
    *higher-scoring*.  Closed models have no disclosed size but live in the
    far-right "very large / undisclosed" column, so we treat their size as the
    largest tier (+inf) — the best of them therefore caps the frontier.
    Computed as the running-max score while sweeping from smallest to largest.
    """
    # smallest size first; undisclosed size sorts last; ties broken by score
    ordered = sorted(rows, key=lambda r: (r.size_b if r.size_b is not None else np.inf,
                                          -r.tb2))
    front: List[Row] = []
    best = -np.inf
    for r in ordered:
        if r.tb2 > best:
            front.append(r)
            best = r.tb2
    return front


def _draw_pareto(ax, rows: List[Row], style: TeaserStyle) -> List[str]:
    """Draw the Pareto frontier line + a soft band on its upper side.

    Returns the set of model names on the frontier so the render loop can give
    them a subtle frontier ring.
    """
    front = _pareto_front(rows)
    if len(front) < 2:
        return []

    fx = [r.x for r in front]
    fy = [r.tb2 for r in front]

    # Shade the WHOLE region above the frontier.  The boundary starts at the
    # bottom-left origin, rises through the frontier points, then runs flat to
    # the right edge so the fill spans the full width up to the top.
    fill_x = [X_LIM[0]] + fx + [X_LIM[1]]
    fill_lo = [0.0] + fy + [fy[-1]]
    ax.fill_between(fill_x, fill_lo, style.ylim[1], color=style.pareto_color,
                    alpha=style.pareto_fill_alpha, linewidth=0, zorder=1)
    # The line spans the full plot: from the bottom-left origin, through the
    # frontier points, then flat to the right edge of the figure.
    line_x = [X_LIM[0]] + fx + [X_LIM[1]]
    line_y = [0.0] + fy + [fy[-1]]
    ax.plot(line_x, line_y, color=style.pareto_line_color, lw=style.pareto_lw,
            alpha=style.pareto_line_alpha, linestyle=style.pareto_line_style,
            solid_capstyle="round", dash_capstyle="round", zorder=1.6)
    if style.pareto_label:
        ax.text(*style.pareto_label_xy, style.pareto_label, ha="center", va="center",
                fontsize=style.annotation_size, color=darken_hex(style.pareto_color, 0.9),
                style="italic", fontweight="bold", alpha=0.95, zorder=1.7)
    return [r.model for r in front]


def render(rows: List[Row], style: TeaserStyle, out_base: Path) -> None:
    _apply_rc(style)
    fig, ax = plt.subplots(figsize=style.figsize)

    # "very large / undisclosed size" overlay band (behind everything)
    if style.frontier_band:
        ax.axvspan(style.frontier_x, X_LIM[1], color=style.frontier_color,
                   alpha=style.frontier_alpha, zorder=0, linewidth=0)
        ax.text((style.frontier_x + X_LIM[1]) / 2.0, style.frontier_label_y,
                style.frontier_label, ha="center", va="top",
                fontsize=style.annotation_size, color="#9A8B79",
                style="italic", zorder=1)

    if style.grid:
        ax.grid(True, axis="both", alpha=style.grid_alpha,
                linestyle=style.grid_linestyle, color=style.spine_color, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.spine_color)
        ax.spines[side].set_linewidth(style.spine_linewidth)

    if style.pareto:
        _draw_pareto(ax, rows, style)  # noqa: returns frontier names (unused)

    for row in rows:
        cfg = style_for(row.model)
        if not cfg["show"]:
            continue
        x, y = row.x + cfg["x_offset"], row.tb2
        is_ours = row.group == "ours"

        # error bar (neutral colour so it never fights the logo)
        if style.show_errorbars and row.std is not None:
            ax.errorbar(x, y, yerr=row.std, fmt="none", ecolor=style.errorbar_color,
                        elinewidth=style.errorbar_lw, capsize=style.errorbar_capsize,
                        capthick=style.errorbar_lw, alpha=style.errorbar_alpha, zorder=2)

        # soft halo for ours
        if is_ours and style.ours_halo:
            ax.scatter([x], [y], s=style.ours_halo_size, color=GROUP_COLORS["ours"],
                       alpha=style.ours_halo_alpha, edgecolors="none", zorder=3)

        # open-recipe ring (colour encodes SFT vs RL); drawn behind the marker
        recipe = RECIPE.get(row.model)
        if style.recipe_rings and recipe in style.ring_recipes:
            if cfg["logo"]:
                ring_in = style.logo_target_in * cfg["logo_scale"] * style.ring_logo_mult
            else:
                ring_in = style.ring_dot_in
            ring_s = (ring_in * 72.0) ** 2
            ax.scatter([x], [y], s=ring_s, facecolors="none",
                       edgecolors=RECIPE_COLORS[recipe], linewidths=style.ring_lw,
                       alpha=style.ring_alpha, zorder=4)

        # marker: logo centred on the exact point, else a coloured dot
        if cfg["logo"]:
            img = _load_logo(cfg["logo"])
            oi = OffsetImage(img, zoom=_logo_zoom(img, style, cfg["logo_scale"]),
                             interpolation="hanning")
            ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0.0,
                                zorder=8 if is_ours else 5)
            ax.add_artist(ab)
        else:
            ax.scatter([x], [y], s=style.dot_size,
                       color=GROUP_COLORS.get(row.group, style.dot_color),
                       edgecolors=style.dot_edge_color,
                       linewidths=style.dot_edge_width, zorder=5)

        # name label (optionally with a dashed leader line back to the point)
        if cfg["label"]:
            arrowprops = None
            if cfg["leader"]:
                arrowprops = dict(arrowstyle="-", color=style.leader_color,
                                  lw=style.leader_lw, alpha=style.leader_alpha,
                                  linestyle=style.leader_linestyle,
                                  shrinkA=1.0, shrinkB=4.0)
            ax.annotate(
                cfg["label"], (x, y),
                xytext=(cfg["dx"], cfg["dy"]), textcoords="offset points",
                ha=cfg["ha"], va=cfg["va"],
                fontsize=style.annotation_size_ours if is_ours else style.annotation_size,
                fontweight="bold" if is_ours else "normal",
                color=darken_hex(GROUP_COLORS["ours"], 0.85) if is_ours else style.annotation_color,
                arrowprops=arrowprops,
                zorder=7,
            )

    # legend (category -> colour)
    if style.legend:
        present = [g for g in GROUP_ORDER if any(r.group == g for r in rows)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=GROUP_COLORS[g], markeredgecolor="white",
                   label=GROUP_LABELS[g])
            for g in present
        ]
        ax.legend(handles=handles, loc=style.legend_loc, bbox_to_anchor=style.legend_bbox,
                  frameon=False, fontsize=style.legend_size, handletextpad=0.4,
                  labelspacing=0.6, borderaxespad=0.0)

    # recipe-ring legend (open recipe: SFT vs RL) + Pareto-frontier note
    if style.recipe_rings or style.pareto:
        handles = []
        if style.recipe_rings:
            present = [k for k in RECIPE_ORDER if k in style.ring_recipes
                       and any(RECIPE.get(r.model) == k for r in rows)]
            handles += [
                Line2D([0], [0], marker="o", linestyle="none", markersize=12,
                       markerfacecolor="none", markeredgecolor=RECIPE_COLORS[k],
                       markeredgewidth=style.ring_lw, label=RECIPE_LABELS[k])
                for k in present
            ]
        if style.pareto:
            handles.append(
                Line2D([0], [0], color=style.pareto_line_color, lw=style.pareto_lw,
                       linestyle=style.pareto_line_style, dash_capstyle="round",
                       label=style.pareto_legend_label)
            )
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  frameon=False, fontsize=style.legend_size, handletextpad=0.6,
                  labelspacing=0.7, borderaxespad=0.0)

    # axes cosmetics
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*style.ylim)
    ax.set_xticks([t[0] for t in X_TICKS])
    ax.set_xticklabels([t[1] for t in X_TICKS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_xlabel(style.xlabel, labelpad=10)
    ax.set_ylabel(style.ylabel, labelpad=10)
    if style.title:
        ax.set_title(style.title, pad=style.title_pad, fontweight=style.title_weight)
    ax.tick_params(length=0)

    _draw_x_break(ax, style)

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=style.dpi, bbox_inches="tight")
    if style.save_pdf:
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.png%s", out_base, " (+ .pdf)" if style.save_pdf else "")


# ===========================================================================
# 8. CLI
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output path base (no extension).")
    ap.add_argument("--title", default=None, help="Set a figure title (default: none).")
    ap.add_argument("--no-errorbars", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows, skipped = load_rows(args.csv.resolve())
    if skipped:
        logger.info("Skipped %d model(s) without a TB2 score yet: %s",
                    len(skipped), ", ".join(skipped))
    logger.info("Plotting %d model(s).", len(rows))

    style = TeaserStyle()
    if args.title:
        style.title = args.title
    if args.no_errorbars:
        style.show_errorbars = False
    render(rows, style, args.out.resolve())


if __name__ == "__main__":
    main()
