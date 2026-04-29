#!/usr/bin/env python3
"""Compute average reward with std dev and pass@k from a harbor job directory.

Usage:
    python scripts/compute_stats.py jobs/my_job_name
    python scripts/compute_stats.py jobs/my_job_name --per-task
"""

import argparse
import json
import math
import os
import sys


def load_job(job_dir: str) -> dict[str, list[float]]:
    """Return {task_name: [reward_per_attempt]} for all completed trials."""
    tasks: dict[str, list[float]] = {}
    for name in os.listdir(job_dir):
        rpath = os.path.join(job_dir, name, "result.json")
        if not os.path.isfile(rpath):
            continue
        r = json.load(open(rpath))
        task = r["task_name"]
        vr = r.get("verifier_result")
        if vr and "rewards" in vr:
            reward = vr["rewards"].get("reward", 0.0)
        else:
            reward = 0.0
        tasks.setdefault(task, []).append(reward)
    return tasks


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k. n=total, c=correct, k=k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser(description="Compute reward stats for a harbor job.")
    parser.add_argument("job_dir", help="Path to job directory")
    parser.add_argument("--per-task", action="store_true", help="Show per-task breakdown")
    args = parser.parse_args()

    tasks = load_job(args.job_dir)
    if not tasks:
        print(f"No results found in {args.job_dir}")
        sys.exit(1)

    n_tasks = len(tasks)
    attempts_per_task = set(len(v) for v in tasks.values())

    # Per-task mean reward
    task_means = {t: sum(rs) / len(rs) for t, rs in tasks.items()}

    # Treat each attempt as an independent run.
    # For each run i, compute mean reward across tasks that have an i-th attempt.
    max_attempts = max(len(v) for v in tasks.values())
    run_scores = []
    for i in range(max_attempts):
        scores = [rs[i] for rs in tasks.values() if i < len(rs)]
        run_scores.append(sum(scores) / len(scores))

    n_runs = len(run_scores)
    overall_mean = sum(run_scores) / n_runs
    overall_std = (sum((s - overall_mean) ** 2 for s in run_scores) / n_runs) ** 0.5
    overall_sem = overall_std / n_runs**0.5

    # Pass@k stats
    ks = sorted({1, min(attempts_per_task)} | {max(attempts_per_task)})
    pass_at_k_values = {}
    for k in ks:
        scores = []
        for t, rs in tasks.items():
            n = len(rs)
            c = sum(1 for r in rs if r > 0)
            scores.append(pass_at_k(n, c, k))
        pass_at_k_values[k] = sum(scores) / len(scores)

    print(f"Job: {args.job_dir}")
    print(f"Tasks: {n_tasks}, Attempts/task: {attempts_per_task}")
    print()
    print(f"Mean reward:  {overall_mean:.4f} +/- {overall_std:.4f} (std)")
    print(f"              {overall_mean:.4f} +/- {overall_sem:.4f} (sem)")
    print()
    for k, v in pass_at_k_values.items():
        print(f"pass@{k}:       {v:.4f}")

    if args.per_task:
        print()
        print("Per-task breakdown:")
        print(f"{'Task':<50} {'Mean':>6} {'Scores'}")
        print("-" * 80)
        for t in sorted(tasks):
            rs = tasks[t]
            mean = task_means[t]
            scores_str = " ".join(f"{r:.0f}" for r in rs)
            print(f"{t:<50} {mean:>6.2f} [{scores_str}]")


if __name__ == "__main__":
    main()
