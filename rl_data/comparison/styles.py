"""Style + palette config for publication-quality figures.

This module is intentionally the *only* place where colors and typographic
choices for figures should live.  Each plot type owns its own ``Style``
dataclass so it can be fine-tuned independently of others — useful when
polishing for a final paper draft, where one figure may need very different
treatment from another.

Currently provides:

* :data:`PALETTES` — registry of Anthropic / Claude-inspired palettes.
* :func:`palette_colors` — pull ``n`` colors from a palette, cycling if needed.
* :class:`StackedCompositionStyle` — knobs for the stacked composition plot
  (``fig6_composition_domain_stacked.png``).

Adding a new plot's style:

    @dataclass
    class MyOtherPlotStyle:
        font_size: float = 14.0
        ...

…and pass an instance into the corresponding ``_render_*`` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Color palettes — Anthropic / Claude inspired
# ---------------------------------------------------------------------------
#
# Each palette is curated for stacked / categorical figures: adjacent colors
# differ enough in hue *and* lightness so neighboring segments in a stacked bar
# don't visually merge.  Where possible we keep an Anthropic-feeling warmth
# (terracotta / sand / espresso) without getting muddy.

PALETTES: Dict[str, List[str]] = {
    # 1. Warm terracotta + earth tones — the most "Anthropic book cover" feel.
    #    Best on cream/white backgrounds.  Reads well in print and B&W.
    "anthropic_warm": [
        "#C46446",  # terracotta
        "#D9A574",  # sand
        "#8B6F47",  # chestnut
        "#E5C8A0",  # cream
        "#6E5C4A",  # deep brown
        "#B89B7A",  # warm gray
        "#A66E3C",  # umber
        "#735741",  # dark taupe
        "#DCBE9A",  # ivory
        "#956E3F",  # dark caramel
        "#C8A988",  # beige
        "#54453A",  # espresso
    ],
    # 2. Modern Claude — Claude-orange anchor + warm neutrals.
    #    A bit more saturated and "product-design" feeling.
    "claude_modern": [
        "#D97757",  # Claude orange
        "#6B5B47",  # warm gray
        "#C9A66B",  # mustard
        "#8B6F47",  # chestnut
        "#D4A276",  # peach
        "#936A4F",  # burnt sienna
        "#C29B6E",  # camel
        "#B07D5C",  # toffee
        "#856648",  # espresso
        "#A88862",  # taupe
        "#5C4D3D",  # dark mocha
        "#E0BE96",  # parchment
    ],
    # 3. Mixed warm + cool earth — designed to give a *full* hue rotation
    #    (red → yellow → green → teal → blue → purple → pink → brown) so
    #    9-12 categories stay visually distinct while still feeling like
    #    the rest of the Anthropic palette family.
    "anthropic_book": [
        "#C15F3C",  # 1. terracotta (red)
        "#D9A036",  # 2. mustard (yellow)
        "#7D9348",  # 3. olive (yellow-green)
        "#3F8B85",  # 4. teal (green-blue)
        "#4E6E94",  # 5. denim (blue)
        "#8E5F94",  # 6. plum (purple)
        "#C97D7E",  # 7. dusty rose (pink)
        "#9C6E3F",  # 8. caramel (warm brown)
        "#5C4D3D",  # 9. dark mocha (deep neutral)
        "#B89B7A",  # 10. warm gray (extra for cycling)
        "#6B8E7F",  # 11. sage (extra)
        "#D2A26B",  # 12. honey (extra)
    ],
    # 3b. Like anthropic_book but a touch more saturated / studio-looking.
    #     Use when projector / web display tends to wash out the muted version.
    "anthropic_studio": [
        "#D34E2C",  # vermilion
        "#E5B12A",  # marigold
        "#7AA12F",  # leaf
        "#2E9E94",  # cyan-teal
        "#3E70B0",  # cobalt
        "#9D52A8",  # orchid
        "#E27482",  # coral
        "#A56B30",  # bronze
        "#3F3026",  # bistre
        "#C0A37A",  # tan
        "#5DA084",  # jade
        "#E8B96E",  # apricot
    ],
    # 4. Quiet monochrome — sequential warm browns.  Reads as a single colour
    #    family; good when the message is "balance" rather than "diversity".
    "anthropic_mono": [
        "#3A2E2A",
        "#5C4A3F",
        "#806655",
        "#9F8470",
        "#BDA088",
        "#D7BFA3",
        "#E8D7BD",
        "#F4E8D4",
        "#A99580",
        "#7C6450",
        "#574437",
        "#382C24",
    ],
}


def list_palettes() -> List[str]:
    """Return the registered palette names (sorted)."""
    return sorted(PALETTES.keys())


def palette_colors(name: str, n: int) -> List[str]:
    """Return ``n`` colors from palette ``name``, cycling if ``n`` exceeds size.

    ``name`` is looked up case-insensitively against :data:`PALETTES`; an
    unknown name falls back to ``anthropic_warm`` so plots never crash on a
    typo.
    """
    base = PALETTES.get(name) or PALETTES.get(name.lower()) or PALETTES["anthropic_warm"]
    if n <= len(base):
        return list(base[:n])
    return [base[i % len(base)] for i in range(n)]


# ---------------------------------------------------------------------------
# Helpers — small color utilities reused by individual styles
# ---------------------------------------------------------------------------


def _hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    h = hx.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def darken_hex(hx: str, factor: float = 0.82) -> str:
    """Return a darker variant of ``hx`` for edge accents / dark text."""
    r, g, b = _hex_to_rgb(hx)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def is_dark_color(hx: str) -> bool:
    """Heuristic for whether text on top of ``hx`` should be light.

    Uses Rec.709 relative luminance.  Threshold of ~0.55 chosen empirically:
    sand (#D9A574) and beige (#E5C8A0) both come out as "light" so we put
    dark text on them, which reads better in print.
    """
    r, g, b = _hex_to_rgb(hx)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return luminance < 0.55


# ---------------------------------------------------------------------------
# StackedCompositionStyle — the only style currently formalised
# ---------------------------------------------------------------------------


@dataclass
class StackedCompositionStyle:
    """All knobs for the stacked-composition figure.

    The default values reproduce the typographic intent of the reference
    snippet (DejaVu Serif, no top/right spines, light gray axis lines, bold
    centered title) and pair it with the warm palette.  Override any field
    via ``StackedCompositionStyle(palette_name=...)`` or ``replace(...)``.
    """

    # ── Typography ─────────────────────────────────────────────────────
    font_family: str = "serif"
    font_serif: Sequence[str] = field(default_factory=lambda: ["DejaVu Serif"])
    font_size: float = 14.0
    title_size: float = 20.0
    axes_label_size: float = 15.0
    tick_size: float = 13.0
    legend_size: float = 12.0
    annotation_size: float = 10.5
    annotation_weight: str = "normal"
    use_tex: bool = False

    # ── Layout ─────────────────────────────────────────────────────────
    figsize: Tuple[float, float] = (11.0, 6.5)
    bar_width: float = 0.55
    bar_gap: float = 0.45
    show_top_right_spines: bool = False
    spine_color: str = "#808080"
    spine_linewidth: float = 1.25

    # ── Bars ───────────────────────────────────────────────────────────
    palette_name: str = "anthropic_warm"
    bar_edge_color: str = "white"
    bar_edge_linewidth: float = 1.0

    # ── Annotations on segments ───────────────────────────────────────
    annotate_segments: bool = True
    annotate_min_pct: float = 4.0  # only label segments >= 4 % of the bar
    annotate_color_dark_bg: str = "#FFFFFF"
    annotate_color_light_bg: str = "#1A1A1A"
    annotate_total_above_bars: bool = False  # totals above each bar (only useful when normalize=False)

    # ── Title / axes ───────────────────────────────────────────────────
    title: str = "Domain composition"
    title_pad: float = 16.0
    title_weight: str = "bold"
    title_loc: str = "center"  # 'left' / 'center' / 'right'
    ylabel: str = "% of tasks"
    xlabel: Optional[str] = None
    xticklabel_rotation: float = 25.0
    xticklabel_ha: str = "right"

    # ── Mode ───────────────────────────────────────────────────────────
    normalize: bool = True  # bars sum to 100 % rather than absolute counts

    # ── Legend ─────────────────────────────────────────────────────────
    legend_loc: str = "center left"
    legend_bbox: Tuple[float, float] = (1.02, 0.5)
    legend_frameon: bool = False
    legend_ncol: int = 1
    legend_title: Optional[str] = None
    legend_reverse: bool = True  # match top-down stack order in legend

    # ── Output ─────────────────────────────────────────────────────────
    dpi: int = 200
    save_pdf: bool = True  # also emit <path>.pdf for Overleaf / vector inclusion


def default_stacked_style(**overrides) -> StackedCompositionStyle:
    """Return a :class:`StackedCompositionStyle`, applying any field overrides."""
    style = StackedCompositionStyle()
    for k, v in overrides.items():
        if not hasattr(style, k):
            raise AttributeError(
                f"StackedCompositionStyle has no field {k!r}; "
                f"check rl_data/comparison/styles.py for the canonical list."
            )
        setattr(style, k, v)
    return style


# ---------------------------------------------------------------------------
# PassAtKOverlayStyle — pass@k curves matched to the stacked-composition look
# ---------------------------------------------------------------------------


@dataclass
class PassAtKOverlayStyle:
    """Typography + palette knobs for the pass@k overlay plot.

    Mirrors :class:`StackedCompositionStyle`'s typographic intent (DejaVu
    Serif, no top/right spines, bold centered title, light gray spines) but
    has its own layout knobs (line/marker sizing, legend placement) suited
    to a multi-line overlay rather than a stacked bar.  Dataset colours
    are pulled from the same palette family as the stacked figure so the
    two read as a coordinated pair in a paper.
    """

    font_family: str = "serif"
    font_serif: Sequence[str] = field(default_factory=lambda: ["DejaVu Serif"])
    font_size: float = 14.0
    title_size: float = 20.0
    axes_label_size: float = 15.0
    tick_size: float = 13.0
    legend_size: float = 12.0
    use_tex: bool = False

    figsize: Tuple[float, float] = (8.0, 5.5)
    show_top_right_spines: bool = False
    spine_color: str = "#808080"
    spine_linewidth: float = 1.25
    grid: bool = True
    grid_alpha: float = 0.25
    grid_linestyle: str = "--"

    palette_name: str = "anthropic_book"
    line_width: float = 2.4
    marker_size: float = 7.0
    marker_edge_color: str = "white"
    marker_edge_width: float = 1.0

    title: str = "Pass@k curves"
    title_pad: float = 16.0
    title_weight: str = "bold"
    title_loc: str = "center"
    xlabel: str = "k"
    ylabel: str = "Mean pass@k"
    ylim: Tuple[float, float] = (0.0, 1.05)

    legend_loc: str = "lower right"
    legend_frameon: bool = False
    legend_ncol: int = 1
    legend_title: Optional[str] = None

    dpi: int = 200
    save_pdf: bool = True


def default_pass_at_k_overlay_style(**overrides) -> PassAtKOverlayStyle:
    """Return a :class:`PassAtKOverlayStyle`, applying any field overrides."""
    style = PassAtKOverlayStyle()
    for k, v in overrides.items():
        if not hasattr(style, k):
            raise AttributeError(
                f"PassAtKOverlayStyle has no field {k!r}; "
                f"check rl_data/comparison/styles.py for the canonical list."
            )
        setattr(style, k, v)
    return style


# ---------------------------------------------------------------------------
# Label prettifier — shared across figures
# ---------------------------------------------------------------------------


def pretty_label(s: str) -> str:
    """Turn a canonical bucket key into a human-readable legend label.

    Rules (matches the writeup convention):

    * Underscores become spaces (``software_engineering`` -> ``software
      engineering``).
    * Ensure the first character is upper-case (``software engineering`` ->
      ``Software engineering``).
    * Existing internal capitalisation is preserved, so labels that are
      already Title-Cased in the source vocab (``Data Processing``,
      ``Multi-Language``) are left alone.  This is important because the
      skill-type vocab is already Title Case while domain keys are not.

    Non-string inputs are returned unchanged via ``str()`` coercion so the
    helper is safe to call on any axis label.
    """
    out = str(s).replace("_", " ")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


__all__ = [
    "PALETTES",
    "list_palettes",
    "palette_colors",
    "darken_hex",
    "is_dark_color",
    "StackedCompositionStyle",
    "default_stacked_style",
    "PassAtKOverlayStyle",
    "default_pass_at_k_overlay_style",
    "pretty_label",
]
