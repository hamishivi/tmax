"""Entry point for the dataset-comparison suite.

Usage (scales to N datasets):

    python -m rl_data.comparison.cli \\
        --reference skill_tax:rl_data/output/tasks_skill_tax_20260401_10k \\
        --baseline  endless_terminals:rl_data/output/tasks_endless_terminals \\
        --baseline  openthoughts_tb:rl_data/output/tasks_openthoughts_tb \\
        --model gemini/gemini-3-flash-preview \\
        --out-dir rl_data/output/comparison

Flags:
    --modules difficulty,command_mix,composition,diversity,realism,verifier
    --max-tasks N           Cap tasks per dataset (0 = all). For smoke tests.
    --sample-pairs N        Pairwise-distance sample size for lexical diversity.

Outputs ``<out-dir>/main/`` (paper body) and ``<out-dir>/appendix/`` (deep-dive)
with a ``.png`` + matching ``.csv`` for every figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_data.comparison.core import (
    DEFAULT_COLORS,
    DatasetSpec,
    RunContext,
    fmt_p,
    load_records,
    save_fig_with_data,
    write_csv,
    attach_task_json,
    effective_command_complexity,
    effective_domain,
    effective_task_complexity,
)
from rl_data.comparison.modules import (
    module_command_mix,
    module_composition,
    module_difficulty,
    module_diversity,
    module_realism,
    module_verifier,
)

logger = logging.getLogger(__name__)

ALL_MODULES = {
    "difficulty": module_difficulty,
    "command_mix": module_command_mix,
    "composition": module_composition,
    "diversity": module_diversity,
    "realism": module_realism,
    "verifier": module_verifier,
}


# Pretty display names per stable adapter name.
_DISPLAY_NAMES = {
    "skill_tax": "Skill-Tax (ours)",
    "endless_terminals": "Endless-Terminals",
    "openthoughts_tb": "OpenThoughts-TB",
}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_spec(raw: str, *, is_reference: bool, color: str) -> DatasetSpec:
    """Parse "name:path" or "name=path" into a DatasetSpec.

    Falls back to using the path basename as the name if no ':' is present.
    """
    if ":" in raw or "=" in raw:
        sep = ":" if ":" in raw else "="
        name, path = raw.split(sep, 1)
    else:
        path = raw
        name = Path(raw).name
    display = _DISPLAY_NAMES.get(name, name.replace("_", " ").title())
    if is_reference:
        display = f"{display} (ours)" if "(ours)" not in display else display
    return DatasetSpec(
        name=name, display_name=display, tasks_dir=Path(path).resolve(),
        color=color, is_reference=is_reference,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference", type=str, required=True,
        help="<name>:<path_to_tasks_dir> for the reference (ours)",
    )
    ap.add_argument(
        "--baseline", type=str, action="append", default=[],
        help="<name>:<path_to_tasks_dir> for a baseline (repeatable)",
    )
    ap.add_argument("--model", type=str, default="gemini/gemini-3-flash-preview")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--modules", type=str, default=",".join(ALL_MODULES.keys()),
        help="Comma-separated module names to run",
    )
    ap.add_argument("--sample-pairs", type=int, default=2000)
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Cap tasks per dataset (0 = all). Useful for smoke tests.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


def _fmt_num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == 0.0:
            return "0"
        if abs(x) < 10 ** (-digits):
            return f"{x:.2e}"
        return f"{x:.{digits}f}"
    return str(x)


def _write_summary_table(report: Dict[str, Any], specs: List[DatasetSpec],
                         path_md: Path, path_csv: Path) -> None:
    L: List[str] = [
        "# Dataset comparison — summary",
        "",
        f"- Solver model: `{report['model']}`",
        f"- Runs per task: `{report['num_runs_per_task']}`",
        "",
    ]

    # Tasks column header list (one per dataset).
    names = [s.name for s in specs]
    display = [s.display_name for s in specs]

    csv_rows: List[Dict[str, Any]] = []

    def _csv(section: str, metric: str, values: Dict[str, Any]):
        csv_rows.append({
            "section": section, "metric": metric,
            **{f"value__{n}": values.get(n) for n in names},
        })

    # ── Headline metrics ──────────────────────────────────────────────
    # Difficulty metrics + composition-balance scalars in a single table.
    if "difficulty" in report:
        d = report["difficulty"]
        per = d["per_dataset"]
        L += [
            "## Headline metrics",
            "",
            "| Metric | " + " | ".join(display) + " | " +
            " | ".join(f"p vs {specs[0].display_name} ({n})" for n in names[1:]) + " |",
            "|---|" + "---:|" * (len(display) + len(names) - 1),
        ]
        p_vs = d.get("p_values_vs_reference", {})
        for metric_key, label in [
            ("mean_pass_at_1", "Mean pass@1"),
            ("mean_pass_at_8", "Mean pass@8"),
            ("mean_turns", "Mean turns"),
            ("median_turns", "Median turns"),
            ("mean_tokens_per_run", "Mean tokens/run"),
        ]:
            values = {n: per[n].get(metric_key) for n in names}
            _csv("headline", metric_key, values)
            row = (
                f"| {label} | "
                + " | ".join(
                    _fmt_num(values[n], 0 if "tokens" in metric_key else 2)
                    for n in names
                )
                + " | "
                + " | ".join(
                    fmt_p((p_vs.get(n) or {}).get(
                        {"mean_pass_at_1": "pass@1",
                         "mean_pass_at_8": "pass@8",
                         "mean_turns": "avg_turns"}.get(metric_key, "")
                    ))
                    for n in names[1:]
                )
                + " |"
            )
            L.append(row)

        # Balance scalars from the composition module (if it ran).
        comp = report.get("composition") or {}
        for field_key, label in [
            ("domain", "Domain balance (1.0 = uniform)"),
            ("skill_type", "Skill-type balance (1.0 = uniform)"),
        ]:
            axis = comp.get(field_key) or {}
            balance = axis.get("balance") or {}
            if not balance:
                continue
            values = {n: balance.get(n) for n in names}
            _csv("headline", f"balance_{field_key}", values)
            L.append(
                "| " + label + " | "
                + " | ".join(_fmt_num(values[n], 3) for n in names)
                + " | " + " | ".join("-" for _ in names[1:]) + " |"
            )
        # cost
        if d.get("cost_per_task_usd") and any(v is not None for v in d["cost_per_task_usd"].values()):
            values = {n: d["cost_per_task_usd"].get(n) for n in names}
            _csv("difficulty", "cost_per_task_usd", values)
            L.append(
                "| Est. USD / task | "
                + " | ".join(f"${_fmt_num(values[n], 4)}" for n in names)
                + " | " + " | ".join("-" for _ in names[1:]) + " |"
            )
        L.append("")

    # ── Command mix ───────────────────────────────────────────────────
    if "command_mix" in report:
        c = report["command_mix"]
        dc = c.get("distinct_categories", {})
        p_dc = c.get("p_values_vs_reference_distinct_categories", {})
        L += [
            "## Command mix (recipe breadth)",
            "",
            "| Metric | " + " | ".join(display) + " | " +
            " | ".join(f"p vs ref ({n})" for n in names[1:]) + " |",
            "|---|" + "---:|" * (len(display) + len(names) - 1),
        ]
        for key in ("mean", "median"):
            values = {n: dc[n].get(key) for n in names}
            _csv("command_mix", f"distinct_categories_{key}", values)
            L.append(
                f"| {'Mean' if key == 'mean' else 'Median'} distinct categories/task | "
                + " | ".join(_fmt_num(values[n]) for n in names)
                + " | "
                + " | ".join(fmt_p(p_dc.get(n)) if key == "mean" else "-" for n in names[1:])
                + " |"
            )
        L += ["", "### Coverage — % of tasks using each category", "",
              "| Category | " + " | ".join(display) + " |",
              "|---|" + "---:|" * len(display)]
        cov = c.get("coverage", {})
        # Include only categories with >=1% in at least one dataset.
        from rl_data.comparison.command_taxonomy import CATEGORIES
        for cat in CATEGORIES:
            vals = [cov.get(n, {}).get(cat, 0) * 100 for n in names]
            if max(vals) < 0.5:
                continue
            _csv("command_mix_coverage", cat, dict(zip(names, vals)))
            L.append("| `" + cat + "` | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")
        L.append("")

    # ── Composition ───────────────────────────────────────────────────
    if "composition" in report:
        L += ["## Composition (projected onto our taxonomy)", ""]
        for field_key, heading in [
            ("domain", "Domain"),
            ("skill_type", "Skill type"),
            ("task_complexity", "Task complexity"),
            ("command_complexity", "Command complexity"),
        ]:
            axis = report["composition"].get(field_key) or {}
            buckets = axis.get("buckets", [])
            pct = axis.get("pct", {})
            p_ref = axis.get("chi2_p_vs_reference", {})
            if not buckets:
                continue
            L += [
                f"### {heading}",
                "",
                "| Bucket | " + " | ".join(display) + " |",
                "|---|" + "---:|" * len(display),
            ]
            for b in buckets:
                vals = {n: (pct.get(n, {}) or {}).get(b, 0.0) for n in names}
                _csv(f"composition_{field_key}", b, vals)
                L.append("| " + b + " | " +
                         " | ".join(f"{vals[n]:.1f}%" for n in names) + " |")
            if p_ref:
                L.append("\nChi-squared p-values vs "
                         f"{specs[0].display_name}: " +
                         ", ".join(f"{n}={fmt_p(p_ref.get(n))}" for n in names[1:]))
            L.append("")
        cov = report["composition"].get("label_coverage", {})
        if cov:
            L += ["### Label coverage (how many tasks got a composition label)", ""]
            for n in names:
                info = cov.get(n, {})
                L.append(f"- **{n}**: {info.get('with_domain_label', 0)}/{info.get('total', 0)} "
                         f"({info.get('frac', 0) * 100:.1f}%)")
            L.append("")

    # ── Diversity ─────────────────────────────────────────────────────
    if "diversity" in report:
        shared = (report["diversity"] or {}).get("shared") or {}
        if shared:
            L += [
                "## Diversity (shared TF-IDF clustering)",
                "",
                f"- k = {shared.get('n_clusters')} clusters over the union of all descriptions",
                "",
                "| Metric | " + " | ".join(display) + " |",
                "|---|" + "---:|" * len(display),
            ]
            lex = shared.get("lexical_diversity", {})
            eff = shared.get("eff_clusters", {})
            cov = shared.get("covered_clusters", {})
            for label, src, key in [
                ("Lexical diversity (mean pairwise TF-IDF distance)", lex, None),
                ("Effective clusters (exp H)", eff, None),
                ("Clusters with ≥1 task", cov, None),
            ]:
                values = {n: src.get(n) for n in names}
                _csv("diversity", label, values)
                if label.startswith("Effective"):
                    fmtfn = lambda v: _fmt_num(v, 1)
                elif label.startswith("Clusters"):
                    fmtfn = lambda v: _fmt_num(v)
                else:
                    fmtfn = lambda v: _fmt_num(v, 3)
                L.append(f"| {label} | " +
                         " | ".join(fmtfn(values[n]) for n in names) + " |")
            uv = shared.get("unique_vs_reference", {})
            if uv:
                L += ["", "### Unique clusters (vs reference)", ""]
                for n in names[1:]:
                    info = uv.get(n, {})
                    L.append(
                        f"- `{n}`: reference-only = {info.get('reference_only_clusters', '-')}, "
                        f"`{n}`-only = {info.get('baseline_only_clusters', '-')}, "
                        f"shared = {info.get('shared_clusters', '-')}"
                    )
                L.append("")

    # ── Realism ───────────────────────────────────────────────────────
    if "realism" in report:
        per = report["realism"].get("per_dataset", {})
        p_vs = report["realism"].get("p_values_vs_reference", {})
        L += [
            "## Realism (container + verifier)",
            "",
            "| Metric | " + " | ".join(display) + " | " +
            " | ".join(f"p vs ref ({n})" for n in names[1:]) + " |",
            "|---|" + "---:|" * (len(display) + len(names) - 1),
        ]
        for key, label in [
            ("desc_words", "Instruction length (words)"),
            ("n_apt_pkgs", "apt packages installed"),
            ("n_pip_pkgs", "pip packages installed"),
            ("n_services", "Services started"),
            ("n_artifacts_checked", "Artifacts checked by verifier"),
        ]:
            values = {n: (per.get(n, {}) or {}).get(key, {}).get("mean") for n in names}
            _csv("realism", key, values)
            ps = [fmt_p((p_vs.get(n) or {}).get(key)) for n in names[1:]]
            L.append("| " + label + " | " +
                     " | ".join(_fmt_num(values[n]) for n in names) +
                     " | " + " | ".join(ps) + " |")
        L.append("")

    # ── Verifier ──────────────────────────────────────────────────────
    if "verifier" in report:
        per = report["verifier"].get("per_dataset", {})
        p_vs = report["verifier"].get("p_values_vs_reference", {})
        L += [
            "## Verifier rigor",
            "",
            "| Metric | " + " | ".join(display) + " | " +
            " | ".join(f"p vs ref ({n})" for n in names[1:]) + " |",
            "|---|" + "---:|" * (len(display) + len(names) - 1),
        ]
        for key, label in [
            ("mean_loc", "Mean LOC"),
            ("mean_asserts", "Mean # asserts"),
        ]:
            stat_key = "loc" if key == "mean_loc" else "n_asserts"
            values = {n: (per.get(n, {}) or {}).get(key) for n in names}
            _csv("verifier", key, values)
            ps = [fmt_p((p_vs.get(n) or {}).get(stat_key)) for n in names[1:]]
            L.append("| " + label + " | " +
                     " | ".join(_fmt_num(values[n]) for n in names) +
                     " | " + " | ".join(ps) + " |")
        L.append(
            "| % verifier uses subprocess | "
            + " | ".join(f"{(per.get(n, {}) or {}).get('pct_uses_subprocess', 0) * 100:.1f}%"
                         for n in names)
            + " | " + " | ".join("-" for _ in names[1:]) + " |"
        )
        L.append("")

    path_md.write_text("\n".join(L) + "\n")

    # CSV mirror.
    fieldnames = ["section", "metric"] + [f"value__{n}" for n in names]
    write_csv(path_csv, csv_rows, fieldnames=fieldnames)


def _write_paper_snippets(report: Dict[str, Any], specs: List[DatasetSpec],
                          path: Path) -> None:
    """Drop-in sentences for the paper body."""
    L = ["# Paper-ready snippets", ""]
    ref = specs[0]
    baselines = specs[1:]

    d = report.get("difficulty", {}).get("per_dataset", {})
    p_vs = report.get("difficulty", {}).get("p_values_vs_reference", {})

    for base in baselines:
        L.append(
            f"- Using the same agent ({report['model']}) with {report['num_runs_per_task']} "
            f"attempts per task, it achieves pass@1 of "
            f"{(d.get(ref.name) or {}).get('mean_pass_at_1', 0):.2f} on our "
            f"{(d.get(ref.name) or {}).get('n_tasks', 0)} tasks vs. "
            f"{(d.get(base.name) or {}).get('mean_pass_at_1', 0):.2f} on the "
            f"{(d.get(base.name) or {}).get('n_tasks', 0)} {base.display_name} tasks "
            f"(Mann–Whitney U, p = {fmt_p((p_vs.get(base.name) or {}).get('pass@1'))})."
        )

    cov_ours = (report.get("command_mix", {}).get("coverage") or {}).get(ref.name) or {}
    for base in baselines:
        cov_base = (report.get("command_mix", {}).get("coverage") or {}).get(base.name) or {}
        richer = [cat for cat, v in cov_ours.items() if v - cov_base.get(cat, 0) >= 0.10]
        if richer:
            L.append(
                f"- Our tasks use the following action categories in "
                f"substantially more tasks than {base.display_name} "
                f"(≥+10 pp coverage): " + ", ".join(f"`{c}`" for c in richer) + "."
            )

    comp = report.get("composition", {})
    if comp:
        # Balance scalars — the money quote for the composition story.
        for field_key, human in [("domain", "domain"),
                                  ("skill_type", "skill-type")]:
            axis = comp.get(field_key) or {}
            balance = axis.get("balance") or {}
            if not balance:
                continue
            ref_b = balance.get(ref.name)
            if ref_b is None:
                continue
            for base in baselines:
                base_b = balance.get(base.name)
                if base_b is None or base_b <= 0:
                    # Baseline has no classifier-tagged tasks yet — skip so we
                    # don't emit a meaningless "inf more uniform" claim.
                    continue
                ratio = ref_b / base_b
                L.append(
                    f"- Our {human} composition is substantially more balanced than "
                    f"{base.display_name}: normalised-entropy score "
                    f"{ref_b:.2f} (ours) vs. {base_b:.2f} ({base.display_name}), "
                    f"i.e. {ratio:.1f}× more uniform "
                    f"(1.0 = perfectly uniform over all buckets)."
                )
        # Chi-squared supporting statement for domain specifically.
        axis = comp.get("domain") or {}
        for base in baselines:
            p = (axis.get("chi2_p_vs_reference") or {}).get(base.name)
            if p is None:
                continue
            L.append(
                f"- Domain-composition difference between ours and "
                f"{base.display_name} is statistically significant "
                f"(chi-squared p = {fmt_p(p)}) — see "
                f"main/fig3_composition_domain_ridgeline.png."
            )

    path.write_text("\n".join(L) + "\n")


def _dump_per_task_metrics(specs: List[DatasetSpec],
                           records_by_name: Dict[str, List[Dict[str, Any]]],
                           model_slug: str,
                           path: Path) -> None:
    from rl_data.comparison.modules import _load_trace_features
    rows = []
    for spec in specs:
        for r in records_by_name[spec.name]:
            tj = r.get("_task_json") or {}
            feats = _load_trace_features(Path(r["dir"]), model_slug) or {}
            rows.append({
                "dataset": spec.name,
                "task_name": r.get("name"),
                "domain_native": tj.get("domain", ""),
                "domain_classified": tj.get("classified_domain", ""),
                "skill_type_native": tj.get("skill_type", ""),
                "skill_type_classified": tj.get("classified_skill_type", ""),
                "task_complexity_native": tj.get("task_complexity", ""),
                "task_complexity_classified": tj.get("classified_task_complexity", ""),
                "command_complexity_native": tj.get("command_complexity", ""),
                "command_complexity_classified": tj.get("classified_command_complexity", ""),
                "num_runs": r.get("num_runs", 0),
                "num_success": r.get("num_success", 0),
                "pass@1": r.get("pass@1"),
                "pass@8": r.get("pass@8"),
                "avg_turns": r.get("avg_turns"),
                "total_input_tokens": r.get("total_input_tokens"),
                "total_output_tokens": r.get("total_output_tokens"),
                "distinct_command_categories": feats.get("distinct_categories"),
            })
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(path, rows, fieldnames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ref_spec = _parse_spec(args.reference, is_reference=True, color=DEFAULT_COLORS[0])
    baseline_specs = [
        _parse_spec(b, is_reference=False, color=DEFAULT_COLORS[(i + 1) % len(DEFAULT_COLORS)])
        for i, b in enumerate(args.baseline)
    ]
    specs = [ref_spec] + baseline_specs

    model_slug = args.model.replace("/", "_")
    out_dir = args.out_dir.resolve()
    main_dir = out_dir / "main"
    appendix_dir = out_dir / "appendix"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)

    # Load records.
    records_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        recs = load_records(spec, model_slug)
        if args.max_tasks and args.max_tasks > 0:
            recs = recs[: args.max_tasks]
        logger.info("Loaded %d records for %s (from %s)",
                    len(recs), spec.name, spec.tasks_dir)
        records_by_name[spec.name] = recs

    ctx = RunContext(
        specs=specs,
        records_by_name=records_by_name,
        model_slug=model_slug,
        main_dir=main_dir,
        appendix_dir=appendix_dir,
        sample_pairs=args.sample_pairs,
    )

    modules_to_run = [m.strip() for m in args.modules.split(",") if m.strip()]
    num_runs_per_task = 0
    for spec in specs:
        for r in records_by_name[spec.name]:
            if r.get("num_runs", 0) > num_runs_per_task:
                num_runs_per_task = r["num_runs"]

    report: Dict[str, Any] = {
        "model": args.model,
        "num_runs_per_task": num_runs_per_task,
        "datasets": [{"name": s.name, "display_name": s.display_name,
                      "tasks_dir": str(s.tasks_dir),
                      "is_reference": s.is_reference}
                     for s in specs],
        "n_tasks_total": {s.name: len(records_by_name[s.name]) for s in specs},
    }

    for m in modules_to_run:
        if m not in ALL_MODULES:
            logger.warning("Unknown module: %s (skipped)", m)
            continue
        logger.info("[module] %s", m)
        report[m] = ALL_MODULES[m](ctx)

    _write_summary_table(
        report, specs,
        path_md=main_dir / "summary_table.md",
        path_csv=main_dir / "summary_data.csv",
    )
    _write_paper_snippets(report, specs, main_dir / "paper_snippets.md")
    (out_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, default=str)
    )
    _dump_per_task_metrics(
        specs, records_by_name, model_slug,
        appendix_dir / "per_task_metrics.csv",
    )
    logger.info("Done. Outputs under %s", out_dir)


if __name__ == "__main__":
    main()
