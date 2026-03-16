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
            "category": task_data.get("category", "unknown"),
            "complexity": task_data.get("complexity", "unknown"),
            "scenario": task_data.get("scenario", "unknown"),
            "dir": str(task_path),
        }

        solutions_dir = task_path / "solutions"
        summary_files = list(solutions_dir.glob("*_summary.json")) if solutions_dir.exists() else []
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
                n_turns = sum(1 for m in r.get("messages", []) if m.get("role") == "tool")
                turns_per_run.append(n_turns)
            record["avg_turns"] = sum(turns_per_run) / len(turns_per_run) if turns_per_run else 0
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
    header = f"{'Task':<30} {'Category':<35} {'Complexity':<30} {'Runs':>5} {'Pass':>5} {'p@1':>6} {'p@8':>6} {'Turns':>6}"
    print("\n" + "=" * len(header))
    print("TASK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in records:
        p1 = f"{r['pass@1']:.2f}" if r["pass@1"] is not None else "N/A"
        p8 = f"{r['pass@8']:.2f}" if r["pass@8"] is not None else "N/A"
        print(
            f"{r['name']:<30} {r['category']:<35} {r['complexity']:<30} "
            f"{r['num_runs']:>5} {r['num_success']:>5} {p1:>6} {p8:>6} {r['avg_turns']:>6.1f}"
        )

    solved = [r for r in records if r["has_solutions"]]
    if solved:
        avg_p1 = sum(r["pass@1"] for r in solved if r["pass@1"] is not None) / len(solved)
        avg_p8 = sum(r["pass@8"] for r in solved if r["pass@8"] is not None) / len(solved)
        avg_turns = sum(r["avg_turns"] for r in solved) / len(solved)
        print("-" * len(header))
        print(f"{'AVERAGE':<30} {'':<35} {'':<30} {'':>5} {'':>5} {avg_p1:>6.2f} {avg_p8:>6.2f} {avg_turns:>6.1f}")

    print(f"\nTotal tasks: {len(records)}, With solutions: {len(solved)}")
    print()


def plot_distributions(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate pie charts for category, complexity, and scenario distributions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for field, title, fname in [
        ("category", "Task Category Distribution", "dist_category.png"),
        ("complexity", "Task Complexity Distribution", "dist_complexity.png"),
        ("scenario", "Scenario Context Distribution", "dist_scenario.png"),
    ]:
        counts = Counter(r[field] for r in records)
        labels = list(counts.keys())
        sizes = list(counts.values())

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct="%1.0f%%", startangle=90,
            pctdistance=0.85, textprops={"fontsize": 9},
        )
        ax.legend(
            wedges, [f"{l} ({c})" for l, c in zip(labels, sizes)],
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
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

    # --- Pass@1 by category ---
    cat_scores = defaultdict(list)
    for r in solved:
        cat_scores[r["category"]].append(r["pass@1"])
    cats = sorted(cat_scores.keys())
    cat_means = [sum(cat_scores[c]) / len(cat_scores[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(max(10, len(cats) * 0.6), 5))
    bars = ax.bar(range(len(cats)), cat_means, color="steelblue")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean pass@1")
    ax.set_title("Pass@1 by Category", fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, cat_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_pass1_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'quality_pass1_by_category.png'}")

    # --- Pass@1 by complexity ---
    comp_scores = defaultdict(list)
    for r in solved:
        comp_scores[r["complexity"]].append(r["pass@1"])
    comps = sorted(comp_scores.keys())
    comp_means = [sum(comp_scores[c]) / len(comp_scores[c]) for c in comps]

    fig, ax = plt.subplots(figsize=(max(8, len(comps) * 1.2), 5))
    bars = ax.bar(range(len(comps)), comp_means, color="darkorange")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean pass@1")
    ax.set_title("Pass@1 by Complexity", fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, comp_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_pass1_by_complexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'quality_pass1_by_complexity.png'}")

    # --- Avg turns by complexity ---
    comp_turns = defaultdict(list)
    for r in solved:
        comp_turns[r["complexity"]].append(r["avg_turns"])
    turn_means = [sum(comp_turns[c]) / len(comp_turns[c]) for c in comps]

    fig, ax = plt.subplots(figsize=(max(8, len(comps) * 1.2), 5))
    bars = ax.bar(range(len(comps)), turn_means, color="seagreen")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Turns")
    ax.set_title("Average Turns to Solve by Complexity", fontweight="bold")
    for bar, val in zip(bars, turn_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_turns_by_complexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'quality_turns_by_complexity.png'}")

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
    ap = argparse.ArgumentParser(description="Analyze generated RL tasks and solutions.")
    ap.add_argument("--tasks-dir", type=Path, required=True, help="Directory containing task_* subdirectories")
    ap.add_argument("--plots-dir", type=Path, default=None, help="Where to save plots (default: <tasks-dir>/analysis)")
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
