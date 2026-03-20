"""Cost estimator for RL data generation pipeline.

Estimates API costs for task generation and solution sampling.

Usage:
    uv run python -m rl_data.estimate_cost
    uv run python -m rl_data.estimate_cost --num-tasks 1000 --num-solutions 16
    uv run python -m rl_data.estimate_cost --model gemini/gemini-2.5-flash
"""
from __future__ import annotations

import argparse

PRICING = {
    "gemini/gemini-3.1-pro-preview": ("Gemini 3.1 Pro Preview", 1.25, 10.00),
    "gemini/gemini-2.5-pro":         ("Gemini 2.5 Pro",         1.25, 10.00),
    "gemini/gemini-2.5-flash":       ("Gemini 2.5 Flash",       0.30,  2.50),
    "gemini/gemini-2.0-flash":       ("Gemini 2.0 Flash",       0.10,  0.40),
}


def estimate(
    num_tasks: int = 100,
    num_solutions: int = 8,
    max_actions: int = 16,
    model: str = "gemini/gemini-3.1-pro-preview",
    survival_rate: float = 0.3,
) -> None:
    label, price_in, price_out = PRICING.get(model, (model, 1.25, 10.00))

    def cost(inp: int, outp: int) -> float:
        return (inp * price_in + outp * price_out) / 1e6

    W = 64
    sep = "=" * W
    thin = "-" * W

    print(sep)
    print("  RL DATA GENERATION COST ESTIMATE")
    print(sep)
    print()
    print(f"  Model:             {label}")
    print(f"  Input price:       ${price_in:.2f} / 1M tokens")
    print(f"  Output price:      ${price_out:.2f} / 1M tokens")
    print()
    print(f"  Tasks requested:   {num_tasks:,}")
    print(f"  Survival rate:     {survival_rate:.0%} (tasks surviving pipeline)")
    surviving = int(num_tasks * survival_rate)
    print(f"  Surviving tasks:   {surviving:,}")
    print(f"  Solutions/task:    {num_solutions}")
    print(f"  Max actions/sol:   {max_actions}")

    # ── STAGE 1: Task Generation ──
    print()
    print(sep)
    print("  STAGE 1: TASK GENERATION")
    print(sep)

    # (step_name, avg_input_tokens, avg_output_tokens, pass_rate)
    task_steps = [
        ("1. Task template gen",      1500, 2000, 0.90),
        ("2. Initial state test gen",  3000, 1500, 0.85),
        ("3. Final state test gen",    4500, 2000, 0.85),
        ("4. Apptainer def gen",       5000, 1500, 0.70),
    ]

    step_cand = num_tasks
    total_ti, total_to, total_tc = 0, 0, 0

    hdr = f"  {'Step':<30} {'Calls':>7} {'In tok':>10} {'Out tok':>10} {'Cost':>10}"
    print(f"\n{hdr}")
    print(f"  {thin}")

    for name, itok, otok, pr in task_steps:
        inp = step_cand * itok
        outp = step_cand * otok
        c = cost(inp, outp)
        total_ti += inp
        total_to += outp
        total_tc += step_cand
        print(f"  {name:<30} {step_cand:>7,} {inp:>10,} {outp:>10,} {'${:,.2f}'.format(c):>10}")
        step_cand = int(step_cand * pr)

    tg_cost = cost(total_ti, total_to)

    print(f"  {thin}")
    print(f"  {'TASK GEN TOTAL':<30} {total_tc:>7,} {total_ti:>10,} {total_to:>10,} {'${:,.2f}'.format(tg_cost):>10}")
    print(f"\n  Estimated surviving tasks: {surviving:,}")

    # ── STAGE 2: Solution Generation ──
    print()
    print(sep)
    print("  STAGE 2: SOLUTION GENERATION")
    print(sep)

    avg_turns = 8
    avg_in_per_turn = 6000
    avg_out_per_turn = 1000

    calls_per_task = num_solutions * avg_turns
    total_sc = surviving * calls_per_task
    total_si = total_sc * avg_in_per_turn
    total_so = total_sc * avg_out_per_turn
    sg_cost = cost(total_si, total_so)

    print(f"\n  Tasks with solutions:     {surviving:,}")
    print(f"  Solutions per task:       {num_solutions}")
    print(f"  Avg turns per solution:   {avg_turns}")
    print(f"  Calls per task:           {calls_per_task:,}")
    print(f"  Total LLM calls:          {total_sc:,}")
    print(f"  Avg input tokens/call:    {avg_in_per_turn:,}")
    print(f"  Avg output tokens/call:   {avg_out_per_turn:,}")
    print(f"\n  Total input tokens:       {total_si:,}")
    print(f"  Total output tokens:      {total_so:,}")
    print(f"  SOLUTION GEN COST:        ${sg_cost:,.2f}")

    # ── SUMMARY ──
    total_cost = tg_cost + sg_cost
    total_calls = total_tc + total_sc
    total_traj = surviving * num_solutions

    print()
    print(sep)
    print("  TOTAL SUMMARY")
    print(sep)
    print()
    print(f"  Requested tasks:           {num_tasks:,}")
    print(f"  Surviving tasks:           {surviving:,}")
    print(f"  Solutions per task:        {num_solutions}")
    print(f"  Total trajectories:        {total_traj:,}")
    print()
    print(f"  {thin}")
    print(f"  {'Component':<30} {'LLM Calls':>12} {'Cost':>12}")
    print(f"  {thin}")
    print(f"  {'Task generation':<30} {total_tc:>12,} {'${:,.2f}'.format(tg_cost):>12}")
    print(f"  {'Solution generation':<30} {total_sc:>12,} {'${:,.2f}'.format(sg_cost):>12}")
    print(f"  {thin}")
    print(f"  {'TOTAL':<30} {total_calls:>12,} {'${:,.2f}'.format(total_cost):>12}")
    print(f"  {thin}")
    print()
    print(f"  Total input tokens:        {total_ti + total_si:,}")
    print(f"  Total output tokens:       {total_to + total_so:,}")
    print()
    if surviving > 0:
        print(f"  Cost per surviving task:   ${total_cost / surviving:,.4f}")
    if total_traj > 0:
        print(f"  Cost per trajectory:       ${total_cost / total_traj:,.4f}")

    # ── Scaling table ──
    print()
    print(f"  {'Quick Scaling Reference':^{W}}")
    print(f"  {thin}")
    print(f"  {'Tasks':>8} {'Surviving':>10} {'Trajectories':>13} {'Est. Cost':>12}")
    print(f"  {thin}")
    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        s = int(n * survival_rate)
        traj = s * num_solutions
        c = total_cost * (n / num_tasks) if num_tasks > 0 else 0
        print(f"  {n:>8,} {s:>10,} {traj:>13,} {'${:,.2f}'.format(c):>12}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Estimate RL data generation costs.")
    ap.add_argument("--num-tasks", type=int, default=100)
    ap.add_argument("--num-solutions", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=16)
    ap.add_argument("--model", type=str, default="gemini/gemini-3.1-pro-preview")
    ap.add_argument("--survival-rate", type=float, default=0.3)
    args = ap.parse_args()

    estimate(
        num_tasks=args.num_tasks,
        num_solutions=args.num_solutions,
        max_actions=args.max_actions,
        model=args.model,
        survival_rate=args.survival_rate,
    )


if __name__ == "__main__":
    main()
