"""Analyze generated tasks and solutions — summary tables and plots."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tasks(tasks_dir: Path) -> List[Dict[str, Any]]:
    """Scan tasks_dir for task directories and load metadata + solution summaries."""
    records = []
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.name.startswith("task_"):
            continue
        task_json = task_path / "task.json"
        if not task_json.exists():
            continue

        with open(task_json) as f:
            task_data = json.load(f)

        record: Dict[str, Any] = {
            "name": task_data.get("name", task_path.name),
            "domain": task_data.get("domain", task_data.get("category", "unknown")),
            "skill_type": task_data.get("skill_type", "unknown"),
            "primitive_skills": task_data.get("primitive_skills", []),
            "task_complexity": task_data.get(
                "task_complexity", task_data.get("complexity", "unknown")
            ),
            "command_complexity": task_data.get("command_complexity", "unknown"),
            "scenario": task_data.get("scenario", "unknown"),
            "dir": str(task_path),
        }

        solutions_dir = task_path / "solutions"
        summary_files = (
            list(solutions_dir.glob("*_summary.json"))
            if solutions_dir.exists()
            else []
        )
        summary_files = [f for f in summary_files if f.name != "summary.json"]

        if summary_files:
            with open(summary_files[0]) as f:
                sol = json.load(f)
            record["num_runs"] = sol.get("num_runs", 0)
            record["num_success"] = sol.get("num_success", 0)
            pass_at_k = sol.get("pass_at_k", {})
            record["pass@1"] = pass_at_k.get("1", pass_at_k.get(1, None))
            record["pass@8"] = pass_at_k.get("8", pass_at_k.get(8, None))

            turns_per_run = []
            for r in sol.get("results", []):
                n_turns = sum(
                    1 for m in r.get("messages", []) if m.get("role") == "tool"
                )
                turns_per_run.append(n_turns)
            record["avg_turns"] = (
                sum(turns_per_run) / len(turns_per_run) if turns_per_run else 0
            )
            record["has_solutions"] = True
        else:
            record["num_runs"] = 0
            record["num_success"] = 0
            record["pass@1"] = None
            record["pass@8"] = None
            record["avg_turns"] = 0
            record["has_solutions"] = False

        records.append(record)
    return records


def print_summary_table(records: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"{'Task':<30} {'Domain':<24} {'Skill Type':<20} "
        f"{'Task Complexity':<32} {'Runs':>5} {'Pass':>5} "
        f"{'p@1':>6} {'p@8':>6} {'Turns':>6}"
    )
    print("\n" + "=" * len(header))
    print("TASK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in records:
        p1 = f"{r['pass@1']:.2f}" if r["pass@1"] is not None else "N/A"
        p8 = f"{r['pass@8']:.2f}" if r["pass@8"] is not None else "N/A"
        print(
            f"{r['name']:<30} {r['domain']:<24} {r['skill_type']:<20} "
            f"{r['task_complexity']:<32} "
            f"{r['num_runs']:>5} {r['num_success']:>5} "
            f"{p1:>6} {p8:>6} {r['avg_turns']:>6.1f}"
        )

    solved = [r for r in records if r["has_solutions"]]
    if solved:
        avg_p1 = (
            sum(r["pass@1"] for r in solved if r["pass@1"] is not None) / len(solved)
        )
        avg_p8 = (
            sum(r["pass@8"] for r in solved if r["pass@8"] is not None) / len(solved)
        )
        avg_turns = sum(r["avg_turns"] for r in solved) / len(solved)
        print("-" * len(header))
        pad = 30 + 24 + 20 + 32 + 3
        print(
            f"{'AVERAGE':<{pad}} {'':>5} {'':>5} "
            f"{avg_p1:>6.2f} {avg_p8:>6.2f} {avg_turns:>6.1f}"
        )

    print(f"\nTotal tasks: {len(records)}, With solutions: {len(solved)}")
    print()


def plot_distributions(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate pie charts for all metadata axes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = [
        ("domain", "Domain Distribution", "dist_domain.png"),
        ("skill_type", "Skill Type Distribution", "dist_skill_type.png"),
        ("task_complexity", "Task Complexity Distribution", "dist_task_complexity.png"),
        (
            "command_complexity",
            "Command Complexity Distribution",
            "dist_command_complexity.png",
        ),
        ("scenario", "Scenario Distribution", "dist_scenario.png"),
    ]

    for field, title, fname in axes:
        counts = Counter(r[field] for r in records)
        labels = list(counts.keys())
        sizes = list(counts.values())

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, _texts, _autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.85,
            textprops={"fontsize": 9},
        )
        ax.legend(
            wedges,
            [f"{lb} ({ct})" for lb, ct in zip(labels, sizes)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


def _bar_chart(
    records: List[Dict[str, Any]],
    field: str,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    color: str = "steelblue",
) -> None:
    """Helper: grouped bar chart of *metric* averaged by *field*."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if val is not None:
            buckets[r[field]].append(val)
    if not buckets:
        return

    keys = sorted(buckets.keys())
    means = [sum(buckets[k]) / len(buckets[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.8), 5))
    bars = ax.bar(range(len(keys)), means, color=color)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if metric.startswith("pass"):
        ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, means):
        fmt = f"{val:.2f}" if metric.startswith("pass") else f"{val:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            fmt,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / fname}")


def plot_quality(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate quality analysis plots (bar charts + pass@k curve)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = [r for r in records if r["has_solutions"] and r["pass@1"] is not None]
    if not solved:
        print("  No solution data available for quality plots.")
        return

    _bar_chart(
        solved, "domain", "pass@1", "Mean pass@1",
        "Pass@1 by Domain", "quality_pass1_by_domain.png",
        out_dir, color="steelblue",
    )
    _bar_chart(
        solved, "task_complexity", "pass@1", "Mean pass@1",
        "Pass@1 by Task Complexity", "quality_pass1_by_task_complexity.png",
        out_dir, color="darkorange",
    )
    _bar_chart(
        solved, "command_complexity", "pass@1", "Mean pass@1",
        "Pass@1 by Command Complexity", "quality_pass1_by_command_complexity.png",
        out_dir, color="mediumpurple",
    )
    _bar_chart(
        solved, "task_complexity", "avg_turns", "Avg Turns",
        "Average Turns by Task Complexity", "quality_turns_by_task_complexity.png",
        out_dir, color="seagreen",
    )
    _bar_chart(
        solved, "domain", "avg_turns", "Avg Turns",
        "Average Turns by Domain", "quality_turns_by_domain.png",
        out_dir, color="teal",
    )

    # --- Pass@k curve (averaged across tasks) ---
    all_pass_at_k: Dict[int, List[float]] = defaultdict(list)
    for r in solved:
        task_dir = Path(r["dir"])
        summary_files = list((task_dir / "solutions").glob("*_summary.json"))
        summary_files = [f for f in summary_files if f.name != "summary.json"]
        if not summary_files:
            continue
        with open(summary_files[0]) as f:
            sol = json.load(f)
        for k_str, v in sol.get("pass_at_k", {}).items():
            all_pass_at_k[int(k_str)].append(v)

    if all_pass_at_k:
        ks = sorted(all_pass_at_k.keys())
        means = [sum(all_pass_at_k[k]) / len(all_pass_at_k[k]) for k in ks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, means, "o-", color="crimson", linewidth=2, markersize=5)
        ax.set_xlabel("k")
        ax.set_ylabel("Mean pass@k")
        ax.set_title("Pass@k Curve (averaged across tasks)", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_pass_at_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / 'quality_pass_at_k.png'}")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze generated RL tasks and solutions."
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing task_* subdirectories",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: <tasks-dir>/analysis)",
    )
    args = ap.parse_args()

    tasks_dir = args.tasks_dir
    plots_dir = args.plots_dir or (tasks_dir / "analysis")

    print(f"Scanning {tasks_dir}...")
    records = load_tasks(tasks_dir)

    if not records:
        print("No tasks found.")
        return

    print_summary_table(records)

    print("Generating distribution plots...")
    plot_distributions(records, plots_dir)

    print("Generating quality plots...")
    plot_quality(records, plots_dir)

    print(f"\nDone. All plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
