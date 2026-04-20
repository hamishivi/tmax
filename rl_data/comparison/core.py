"""Shared infrastructure for the dataset-comparison suite.

This file absorbs what would otherwise be split across ``base.py``,
``io.py``, and ``plotting.py`` modules.  It covers:

* :class:`DatasetSpec` — metadata for one dataset (our reference or a baseline).
* :func:`load_records` — hydrate per-task records + solution summaries.
* Canonical field accessors (:func:`effective_domain`, etc.) that fall back to
  LLM-classified labels for external datasets.
* :func:`save_fig_with_data` — write every figure as ``<name>.png`` + the
  underlying numbers to ``<name>.csv`` so you can replot in your own style.
* A small pool of plotting primitives (grouped bar over N datasets, histogram
  overlay, etc.) used from :mod:`rl_data.comparison.modules`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the existing per-task loader from analyze.py to keep summary parsing
# in one place.
from rl_data.analyze import load_tasks  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset spec
# ---------------------------------------------------------------------------

# Default distinct colors for up to 6 datasets.  The first is always "ours".
DEFAULT_COLORS = ["#1f6feb", "#d08770", "#6aa84f", "#af5fb0", "#c18e00", "#666666"]


@dataclass
class DatasetSpec:
    """Metadata describing one dataset passed to the comparison CLI.

    Attributes
    ----------
    name
        Stable machine-readable identifier (``"skill_tax"``, ``"endless_terminals"``).
    display_name
        Human-readable label used in chart titles and tables.
    tasks_dir
        Directory containing ``task_*`` subdirs in our canonical layout.
    color
        Matplotlib color string for plots.
    is_reference
        ``True`` if this dataset is the one we are arguing for (ours); statistics
        and composition deltas are always baseline-vs-reference.
    """

    name: str
    display_name: str
    tasks_dir: Path
    color: str = DEFAULT_COLORS[0]
    is_reference: bool = False


# ---------------------------------------------------------------------------
# Field accessors — handle the metadata asymmetry between datasets
# ---------------------------------------------------------------------------


def _first_non_empty(*values: Any) -> Any:
    for v in values:
        if v in (None, "", "unknown"):
            continue
        if isinstance(v, list) and not v:
            continue
        return v
    return None


def safe_task_json(task_dir: Path) -> Dict[str, Any]:
    """Read ``<task_dir>/task.json`` defensively."""
    try:
        return json.loads((task_dir / "task.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def attach_task_json(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return (and cache) the task.json dict for a record loaded by analyze."""
    cached = record.get("_task_json")
    if isinstance(cached, dict):
        return cached
    tj = safe_task_json(Path(record["dir"]))
    record["_task_json"] = tj
    return tj


def effective_domain(tj: Dict[str, Any]) -> Optional[str]:
    """Return domain label, falling back to classifier output if native is absent."""
    return _first_non_empty(tj.get("domain"), tj.get("classified_domain"))


def effective_skill_type(tj: Dict[str, Any]) -> Optional[str]:
    """Return skill_type label, falling back to classifier output if absent."""
    return _first_non_empty(tj.get("skill_type"), tj.get("classified_skill_type"))


def effective_task_complexity(tj: Dict[str, Any]) -> Optional[str]:
    """Return task_complexity (short/moderate/complex)."""
    raw = _first_non_empty(
        _shorten_task_complexity(tj.get("task_complexity")),
        _shorten_task_complexity(tj.get("classified_task_complexity")),
    )
    return raw


def effective_command_complexity(tj: Dict[str, Any]) -> Optional[str]:
    """Return command_complexity (bash-only / bash+code / bash+code+services)."""
    return _first_non_empty(
        _shorten_cmd_complexity(tj.get("command_complexity")),
        _shorten_cmd_complexity(tj.get("classified_command_complexity")),
    )


_CMD_COMPLEXITY_MAP = {
    "bash-only": "bash-only",
    "bash and code": "bash+code",
    "bash+code": "bash+code",
    "bash, code, and system services": "bash+code+services",
    "bash+code+services": "bash+code+services",
}


def _shorten_task_complexity(raw: Optional[str]) -> Optional[str]:
    if not raw or raw == "unknown":
        return None
    import re
    m = re.match(r"(short|moderate|complex)\b", raw, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _shorten_cmd_complexity(raw: Optional[str]) -> Optional[str]:
    if not raw or raw == "unknown":
        return None
    prefix = raw.split("(")[0].strip()
    return _CMD_COMPLEXITY_MAP.get(prefix, prefix if prefix in _CMD_COMPLEXITY_MAP.values() else None)


# Canonical ordered vocab — exposed for composition module + CLI summary.
DOMAINS_ORDER = [
    "data_science", "scientific_computing", "data_querying",
    "debugging", "software_engineering", "file_operations",
    "data_processing", "security", "system_administration",
]

# Skill-type vocabulary observed in our 10k dataset.  Keeping the full list
# (29 values) so composition analysis is faithful to the source taxonomy; the
# classifier prompt will present them explicitly.  Order below roughly tracks
# frequency in our dataset — callers should not rely on it.
SKILL_TYPES_ORDER = [
    "Systems", "Algorithmic", "Mathematical", "Data Processing", "Testing",
    "Web Security", "Data Comprehension", "Query Construction",
    "Result Processing", "Graph Processing", "File I/O", "Data Parsing",
    "Archives", "Statistical", "Navigation", "Transformation", "Debugging",
    "Multi-Language", "Manipulation", "String/Text", "Data I/O", "Filesystem",
    "Shell Scripting", "Process/Service", "Deployment", "Forensics",
    "Configuration", "Time Series", "Network",
]

TASK_COMPLEXITY_ORDER = ["short", "moderate", "complex"]
COMMAND_COMPLEXITY_ORDER = ["bash-only", "bash+code", "bash+code+services"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(spec: DatasetSpec, model_slug: str) -> List[Dict[str, Any]]:
    """Hydrate per-task records for a single dataset.

    Wraps :func:`rl_data.analyze.load_tasks` so we can inject the cached
    ``task.json`` dict and tag the ``dataset`` origin.
    """
    records = load_tasks(spec.tasks_dir, model_slug=model_slug)
    for r in records:
        attach_task_json(r)
        r["dataset"] = spec.name
    return records


# ---------------------------------------------------------------------------
# PNG + CSV side-by-side
# ---------------------------------------------------------------------------


def save_fig_with_data(
    fig,
    data_rows: List[Dict[str, Any]],
    path_base: Path,
    *,
    fieldnames: Sequence[str],
    dpi: int = 150,
) -> None:
    """Save a matplotlib figure as ``<path_base>.png`` and the underlying numbers
    to ``<path_base>.csv`` so the figure can be reconstructed in any style.

    Notes
    -----
    * ``path_base`` should not include an extension.
    * ``data_rows`` must be a list of dicts with keys covering ``fieldnames``.
    """
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    with path_base.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in data_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a plain CSV (no figure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Plot primitives — parametrized over N datasets
# ---------------------------------------------------------------------------


def grouped_bar(
    ax,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    specs: Sequence[DatasetSpec],
    *,
    ylabel: str,
    title: str,
    ylim: Optional[tuple] = None,
    annotate: bool = False,
    value_fmt: str = "{:.2f}",
) -> None:
    """Render a grouped bar chart with one group per category, one bar per spec."""
    assert len(series) == len(specs), "series and specs must align"
    n_specs = len(specs)
    x = np.arange(len(categories))
    group_width = 0.8
    bar_w = group_width / max(n_specs, 1)

    for i, (vals, spec) in enumerate(zip(series, specs)):
        offset = -group_width / 2 + bar_w / 2 + i * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=spec.display_name, color=spec.color)
        if annotate:
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    value_fmt.format(v),
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)


def histogram_overlay(
    ax,
    series: Sequence[Sequence[float]],
    specs: Sequence[DatasetSpec],
    *,
    bins,
    xlabel: str,
    ylabel: str,
    title: str,
    density: bool = True,
) -> None:
    for vals, spec in zip(series, specs):
        if not len(vals):
            continue
        ax.hist(vals, bins=bins, alpha=0.55, label=spec.display_name,
                color=spec.color, density=density)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def nanmean(xs: Iterable[Optional[float]]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(statistics.fmean(xs)) if xs else 0.0


def nanmedian(xs: Iterable[Optional[float]]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(statistics.median(xs)) if xs else 0.0


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Mann–Whitney U p-value, gated on scipy being installed."""
    a = [x for x in a if x is not None and not (isinstance(x, float) and math.isnan(x))]
    b = [x for x in b if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not a or not b:
        return None
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
    except ImportError:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return None


def balance_score(counts: Sequence[float]) -> float:
    """Normalised Shannon entropy: ``exp(H) / N`` over a count vector.

    Returns a value in ``[0, 1]``:

    * ``1.0`` — perfectly uniform distribution across all buckets.
    * ``0.0`` — everything concentrated in a single bucket.

    Used as the single-scalar summary of how balanced each dataset's
    composition is along a given axis (domain, skill_type, etc).
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    n = len(counts)
    if n <= 1:
        return 1.0
    probs = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in probs)
    return float(math.exp(h) / n)


def chi_squared(counts_ours: Sequence[int], counts_theirs: Sequence[int]) -> Optional[float]:
    """Chi-squared p-value for two count vectors over the same buckets."""
    if len(counts_ours) != len(counts_theirs):
        return None
    try:
        from scipy.stats import chi2_contingency  # type: ignore
    except ImportError:
        return None
    mat = np.array([counts_ours, counts_theirs], dtype=float)
    # Drop buckets where both are zero (undefined contribution).
    keep = (mat.sum(axis=0) > 0)
    if keep.sum() < 2:
        return None
    try:
        chi2, p, dof, _exp = chi2_contingency(mat[:, keep])
        return float(p)
    except Exception:
        return None


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3g}"


# ---------------------------------------------------------------------------
# Misc helpers (shared by modules / cli)
# ---------------------------------------------------------------------------


def iter_bash_commands(summary_path: Path) -> Iterable[str]:
    """Yield every bash command the agent issued in all runs of a task."""
    if not summary_path.exists():
        return
    try:
        sol = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for r in sol.get("results", []):
        for msg in r.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}) or {}
                if fn.get("name") != "bash":
                    continue
                args_raw = fn.get("arguments")
                if not args_raw:
                    continue
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                cmd = args.get("command")
                if cmd:
                    yield cmd


@dataclass
class RunContext:
    """Convenience bag passed into each analysis module."""

    specs: List[DatasetSpec]
    records_by_name: Dict[str, List[Dict[str, Any]]]
    model_slug: str
    main_dir: Path
    appendix_dir: Path
    sample_pairs: int = 2000
    seed: int = 0

    @property
    def reference(self) -> DatasetSpec:
        for s in self.specs:
            if s.is_reference:
                return s
        return self.specs[0]

    @property
    def baselines(self) -> List[DatasetSpec]:
        return [s for s in self.specs if not s.is_reference]

    def records_of(self, spec: DatasetSpec) -> List[Dict[str, Any]]:
        return self.records_by_name[spec.name]
