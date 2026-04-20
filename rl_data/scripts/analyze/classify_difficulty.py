#!/usr/bin/env python3
"""Classify tasks into difficulty tiers based on Harbor evaluation results.

Supports single-model mode (one job) and dual-model mode (two jobs) for
the tier classification scheme:

    Frontier       (10-20%): Accuracy < 40% for Max(model_a, model_b)
    Advanced Plus  (30-40%): Accuracy < 40% for Min(model_a, model_b), excl. Frontier
    Advanced       (20-30%): 40% <= Accuracy < 60% for Min(model_a, model_b), excl. Adv+
    Core           (20-30%): 60% <= Accuracy < 80% for Min(model_a, model_b), excl. Adv

In single-model mode, the one model's accuracy is used directly for all
threshold checks (Max and Min collapse to the same value).

Usage:
    # Single model
    python rl_data/scripts/analyze/classify_difficulty.py \\
        --job jobs/rldata_10k_claude \\
        --dataset rl_data/output/tasks_skill_tax_20260401_10k \\
        --output rl_data/output/difficulty_report

    # Two models (after running both evals)
    python rl_data/scripts/analyze/classify_difficulty.py \\
        --job jobs/rldata_10k_claude jobs/rldata_10k_gpt5 \\
        --dataset rl_data/output/tasks_skill_tax_20260401_10k \\
        --output rl_data/output/difficulty_report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_job(job_dir: str) -> dict[str, list[float]]:
    """Return {task_name: [reward_per_attempt]} for all completed trials."""
    tasks: dict[str, list[float]] = {}
    for name in os.listdir(job_dir):
        rpath = os.path.join(job_dir, name, "result.json")
        if not os.path.isfile(rpath):
            continue
        with open(rpath) as f:
            r = json.load(f)
        task = r["task_name"]
        vr = r.get("verifier_result")
        err = r.get("exception_info")
        if err:
            tasks.setdefault(task, []).append(0.0)
            continue
        if vr and "rewards" in vr:
            reward = vr["rewards"].get("reward", 0.0)
        else:
            reward = 0.0
        tasks.setdefault(task, []).append(reward)
    return tasks


def task_accuracy(rewards: list[float]) -> float:
    """Compute accuracy (fraction of successful attempts) for a single task."""
    if not rewards:
        return 0.0
    return sum(1.0 for r in rewards if r > 0) / len(rewards)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_task_metadata(dataset_dir: Path) -> dict[str, dict]:
    """Load task.json metadata for all tasks in the original dataset."""
    meta = {}
    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        task_json = task_dir / "task.json"
        if task_json.exists():
            with open(task_json) as f:
                data = json.load(f)
            meta[data.get("name", task_dir.name)] = data
    return meta


def classify_tasks(
    accuracies_a: dict[str, float],
    accuracies_b: dict[str, float] | None,
) -> dict[str, str]:
    """Classify each task into a difficulty tier.

    Returns {task_name: tier} where tier is one of:
        "frontier", "advanced_plus", "advanced", "core", "easy"
    """
    all_tasks = set(accuracies_a.keys())
    if accuracies_b:
        all_tasks |= set(accuracies_b.keys())

    tiers: dict[str, str] = {}
    for task in all_tasks:
        acc_a = accuracies_a.get(task, 0.0)
        acc_b = accuracies_b.get(task, 0.0) if accuracies_b else acc_a

        max_acc = max(acc_a, acc_b)
        min_acc = min(acc_a, acc_b)

        if max_acc < 0.40:
            tiers[task] = "frontier"
        elif min_acc < 0.40:
            tiers[task] = "advanced_plus"
        elif min_acc < 0.60:
            tiers[task] = "advanced"
        elif min_acc < 0.80:
            tiers[task] = "core"
        else:
            tiers[task] = "easy"

    return tiers


def build_report(
    tiers: dict[str, str],
    accuracies_a: dict[str, float],
    accuracies_b: dict[str, float] | None,
    task_meta: dict[str, dict],
    job_labels: list[str],
) -> dict:
    """Build a comprehensive report dict."""
    total = len(tiers)
    tier_names = ["frontier", "advanced_plus", "advanced", "core", "easy"]

    tier_counts = Counter(tiers.values())
    tier_summary = {}
    for t in tier_names:
        count = tier_counts.get(t, 0)
        tier_summary[t] = {
            "count": count,
            "pct": round(100.0 * count / total, 1) if total else 0.0,
        }

    overall_acc_a = (
        sum(accuracies_a.values()) / len(accuracies_a) if accuracies_a else 0.0
    )
    overall = {"total_tasks": total, "overall_accuracy": {job_labels[0]: round(overall_acc_a, 4)}}
    if accuracies_b:
        overall_acc_b = sum(accuracies_b.values()) / len(accuracies_b) if accuracies_b else 0.0
        overall["overall_accuracy"][job_labels[1]] = round(overall_acc_b, 4)

    # Per-domain breakdown
    domain_tiers: dict[str, Counter] = defaultdict(Counter)
    skill_tiers: dict[str, Counter] = defaultdict(Counter)
    for task, tier in tiers.items():
        meta = task_meta.get(task, {})
        domain = meta.get("domain", "unknown")
        skill = meta.get("skill_type", "unknown")
        domain_tiers[domain][tier] += 1
        skill_tiers[skill][tier] += 1

    domain_breakdown = {}
    for domain in sorted(domain_tiers):
        counts = domain_tiers[domain]
        domain_total = sum(counts.values())
        domain_breakdown[domain] = {
            "total": domain_total,
            **{t: counts.get(t, 0) for t in tier_names},
        }

    skill_breakdown = {}
    for skill in sorted(skill_tiers):
        counts = skill_tiers[skill]
        skill_total = sum(counts.values())
        skill_breakdown[skill] = {
            "total": skill_total,
            **{t: counts.get(t, 0) for t in tier_names},
        }

    task_list = {}
    for t in tier_names:
        task_list[t] = sorted(task for task, tier in tiers.items() if tier == t)

    return {
        "overall": overall,
        "tier_summary": tier_summary,
        "domain_breakdown": domain_breakdown,
        "skill_breakdown": skill_breakdown,
        "task_list": task_list,
    }


def write_markdown_report(report: dict, out_path: Path, job_labels: list[str]) -> None:
    """Write a human-readable markdown report."""
    lines = ["# Difficulty Classification Report", ""]

    overall = report["overall"]
    lines.append(f"**Total tasks:** {overall['total_tasks']}")
    for label, acc in overall["overall_accuracy"].items():
        lines.append(f"**Overall accuracy ({label}):** {acc:.1%}")
    lines.append("")

    lines.append("## Tier Distribution")
    lines.append("")
    lines.append("| Tier | Count | % |")
    lines.append("|------|------:|--:|")
    for tier, info in report["tier_summary"].items():
        lines.append(f"| {tier} | {info['count']} | {info['pct']}% |")
    lines.append("")

    bench_total = sum(
        info["count"]
        for t, info in report["tier_summary"].items()
        if t != "easy"
    )
    lines.append(
        f"**Benchmark-worthy tasks (excluding easy):** {bench_total} "
        f"({100.0 * bench_total / overall['total_tasks']:.1f}%)"
    )
    lines.append("")

    lines.append("## Per-Domain Breakdown")
    lines.append("")
    tier_names = ["frontier", "advanced_plus", "advanced", "core", "easy"]
    header = "| Domain | Total | " + " | ".join(tier_names) + " |"
    sep = "|--------|------:|" + "|".join(["-----:" for _ in tier_names]) + "|"
    lines.append(header)
    lines.append(sep)
    for domain, info in report["domain_breakdown"].items():
        cols = [str(info.get(t, 0)) for t in tier_names]
        lines.append(f"| {domain} | {info['total']} | " + " | ".join(cols) + " |")
    lines.append("")

    lines.append("## Per-Skill-Type Breakdown")
    lines.append("")
    header = "| Skill Type | Total | " + " | ".join(tier_names) + " |"
    sep = "|------------|------:|" + "|".join(["-----:" for _ in tier_names]) + "|"
    lines.append(header)
    lines.append(sep)
    for skill, info in report["skill_breakdown"].items():
        cols = [str(info.get(t, 0)) for t in tier_names]
        lines.append(f"| {skill} | {info['total']} | " + " | ".join(cols) + " |")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify tasks into difficulty tiers from Harbor job results"
    )
    parser.add_argument(
        "--job",
        nargs="+",
        required=True,
        help="Path(s) to Harbor job directory(ies). 1 for single-model, 2 for dual-model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to original task dataset (for metadata lookup)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_data/output/difficulty_report"),
        help="Output directory for report files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each job (default: directory basenames)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(args.job) > 2:
        parser.error("At most 2 job directories supported (dual-model mode)")

    job_labels = args.labels or [Path(j).name for j in args.job]
    if len(job_labels) != len(args.job):
        parser.error("Number of --labels must match number of --job paths")

    logger.info("Loading job results from %s", args.job[0])
    rewards_a = load_job(args.job[0])
    accuracies_a = {t: task_accuracy(rs) for t, rs in rewards_a.items()}
    logger.info("  Loaded %d tasks from %s", len(accuracies_a), job_labels[0])

    accuracies_b = None
    if len(args.job) > 1:
        logger.info("Loading job results from %s", args.job[1])
        rewards_b = load_job(args.job[1])
        accuracies_b = {t: task_accuracy(rs) for t, rs in rewards_b.items()}
        logger.info("  Loaded %d tasks from %s", len(accuracies_b), job_labels[1])

    logger.info("Loading task metadata from %s", args.dataset)
    task_meta = load_task_metadata(args.dataset)
    logger.info("  Loaded metadata for %d tasks", len(task_meta))

    tiers = classify_tasks(accuracies_a, accuracies_b)

    report = build_report(tiers, accuracies_a, accuracies_b, task_meta, job_labels)

    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / "difficulty_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote JSON report to %s", json_path)

    md_path = args.output / "difficulty_report.md"
    write_markdown_report(report, md_path, job_labels)
    logger.info("Wrote markdown report to %s", md_path)

    # Print summary to stdout
    print(f"\n{'='*60}")
    print("DIFFICULTY CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {report['overall']['total_tasks']}")
    for label, acc in report["overall"]["overall_accuracy"].items():
        print(f"Overall accuracy ({label}): {acc:.1%}")
    print()
    for tier, info in report["tier_summary"].items():
        bar = "#" * int(info["pct"] / 2)
        print(f"  {tier:<15s} {info['count']:>5d}  ({info['pct']:>5.1f}%)  {bar}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
