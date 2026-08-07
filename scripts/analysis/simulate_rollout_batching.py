#!/usr/bin/env python3
"""Simulate completion-ordered TMAX rollout collection.

This is a deliberately small model of the queueing-relevant behavior in
``open_instruct/data_loader.py``:

* each prompt produces a fixed-size group of rollouts;
* ``async_steps * num_unique_prompts_rollout`` prompt groups are kept in flight;
* prompt groups are consumed in completion order and immediately replenished;
* zero-variance reward groups are rejected by active sampling; and
* results may be rejected when their model-step age exceeds ``async_steps``.

It is not a performance simulator for Ray, vLLM, Docker, or Podman.  Instead it
answers a narrower question: which statistical mechanisms can make an 8x32
frozen-policy run appear both longer and more rewarding than a 32x32 run?

Example:

    uv run python scripts/analysis/simulate_rollout_batching.py --trials 200
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Workload:
    """Potential frozen-policy outcomes, indexed by dispatch ordinal."""

    rewards: np.ndarray
    lengths: np.ndarray
    failure_uniforms: np.ndarray
    environment_delay: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    reward: float
    length: float
    dispatched: int
    completed: int
    filtered: int
    stale_dropped: int
    pending: int

    @property
    def stale_fraction(self) -> float:
        return self.stale_dropped / self.completed if self.completed else 0.0


@dataclass(frozen=True)
class TrialDelta:
    reward: float
    length: float
    stale_8: float
    stale_32: float
    pending_8: float
    pending_32: float


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def make_workload(
    rng: np.random.Generator,
    num_prompts: int,
    group_size: int,
    drift_strength: float = 0.0,
    drift_horizon: int = 4_000,
) -> Workload:
    """Create heterogeneous tasks whose successful trajectories tend to be long.

    ``drift_strength`` introduces an explicit dispatch-order curriculum.  It is
    zero in the stationary experiments.  A positive value makes later prompts
    both harder and shorter, illustrating the danger of comparing equal
    optimizer steps when the two runs consume fourfold-different prompt counts.
    """

    base_pass_rate = np.clip(rng.beta(1.6, 3.8, size=num_prompts), 0.01, 0.99)
    progress = np.minimum(np.arange(num_prompts) / max(drift_horizon, 1), 1.0)
    pass_rate = 1.0 / (
        1.0 + np.exp(-(_logit(base_pass_rate) - drift_strength * progress))
    )

    # Across TMAX-like tasks, high-reward groups are allowed to be longer.  A
    # successful rollout also receives an additional within-task length bump.
    base_length = (5_000.0 + 14_000.0 * pass_rate) * (
        1.0 - 0.25 * drift_strength * progress
    )
    length_noise = rng.lognormal(mean=0.0, sigma=0.28, size=(num_prompts, group_size))
    rewards = rng.random((num_prompts, group_size)) < pass_rate[:, None]
    lengths = base_length[:, None] * length_noise + rewards * 2_500.0
    lengths = np.clip(lengths, 200.0, 65_536.0)

    return Workload(
        rewards=rewards,
        lengths=lengths,
        failure_uniforms=rng.random((num_prompts, group_size)),
        environment_delay=rng.lognormal(mean=0.0, sigma=0.5, size=num_prompts),
    )


def simulate(
    workload: Workload,
    *,
    prompt_batch_size: int,
    group_size: int,
    accepted_prompt_groups: int,
    async_steps: int = 4,
    active_sampling: bool = True,
    stale_dropping: bool = False,
    rollout_failure_probability: float = 0.0,
    failure_length: float = 300.0,
) -> SimulationResult:
    """Run a completion-ordered discrete-event simulation."""

    if accepted_prompt_groups % prompt_batch_size:
        raise ValueError(
            "accepted_prompt_groups must be divisible by prompt_batch_size"
        )
    if workload.rewards.shape[1] != group_size:
        raise ValueError("workload group size does not match group_size")

    target_steps = accepted_prompt_groups // prompt_batch_size
    in_flight_prompt_groups = async_steps * prompt_batch_size
    events: list[tuple[float, int, int, int]] = []
    next_prompt = 0
    dispatch_ordinal = 0
    training_step = 0
    completed = 0
    filtered = 0
    stale_dropped = 0
    accepted_in_step = 0
    accepted_rewards: list[float] = []
    accepted_lengths: list[float] = []

    def dispatch(now: float) -> None:
        nonlocal next_prompt, dispatch_ordinal
        if next_prompt >= workload.rewards.shape[0]:
            raise RuntimeError(
                "workload exhausted; increase the pre-generated prompt count"
            )

        prompt_idx = next_prompt
        failed = workload.failure_uniforms[prompt_idx] < rollout_failure_probability
        effective_lengths = np.where(
            failed, failure_length, workload.lengths[prompt_idx]
        )

        # A prompt group is emitted only after its slowest rollout completes.
        duration = float(
            effective_lengths.max() / 2_000.0 + workload.environment_delay[prompt_idx]
        )
        heapq.heappush(
            events, (now + duration, dispatch_ordinal, prompt_idx, training_step)
        )
        next_prompt += 1
        dispatch_ordinal += 1

    for _ in range(in_flight_prompt_groups):
        dispatch(0.0)

    while training_step < target_steps:
        completion_time, _, prompt_idx, model_step = heapq.heappop(events)
        completed += 1

        # The production code replenishes after every completion, including
        # completions subsequently rejected as stale or zero-variance.
        dispatch(completion_time)

        if stale_dropping and training_step - model_step > async_steps:
            stale_dropped += 1
            continue

        failed = workload.failure_uniforms[prompt_idx] < rollout_failure_probability
        rewards = np.where(failed, False, workload.rewards[prompt_idx])
        lengths = np.where(failed, failure_length, workload.lengths[prompt_idx])

        if active_sampling and bool(np.all(rewards == rewards[0])):
            filtered += 1
            continue

        accepted_rewards.append(float(rewards.mean()))
        accepted_lengths.append(float(lengths.mean()))
        accepted_in_step += 1

        if accepted_in_step == prompt_batch_size:
            training_step += 1
            accepted_in_step = 0

    return SimulationResult(
        reward=float(np.mean(accepted_rewards)),
        length=float(np.mean(accepted_lengths)),
        dispatched=next_prompt,
        completed=completed,
        filtered=filtered,
        stale_dropped=stale_dropped,
        pending=len(events),
    )


def _mean_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(array, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def _format_interval(values: list[float], scale: float = 1.0, digits: int = 4) -> str:
    mean, low, high = _mean_interval(values)
    return f"{mean * scale:+.{digits}f} [{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"


def run_scenario(
    *,
    trials: int,
    seed: int,
    group_size: int,
    accepted_groups_8: int,
    accepted_groups_32: int,
    stale_dropping: bool,
    failure_probability_8: float,
    failure_probability_32: float,
    drift_strength: float,
) -> list[TrialDelta]:
    deltas: list[TrialDelta] = []
    max_accepted = max(accepted_groups_8, accepted_groups_32)
    num_prompts = max(20_000, max_accepted * 4)

    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        workload = make_workload(
            rng,
            num_prompts=num_prompts,
            group_size=group_size,
            drift_strength=drift_strength,
            drift_horizon=max_accepted,
        )
        result_8 = simulate(
            workload,
            prompt_batch_size=8,
            group_size=group_size,
            accepted_prompt_groups=accepted_groups_8,
            stale_dropping=stale_dropping,
            rollout_failure_probability=failure_probability_8,
        )
        result_32 = simulate(
            workload,
            prompt_batch_size=32,
            group_size=group_size,
            accepted_prompt_groups=accepted_groups_32,
            stale_dropping=stale_dropping,
            rollout_failure_probability=failure_probability_32,
        )
        deltas.append(
            TrialDelta(
                reward=result_8.reward - result_32.reward,
                length=result_8.length - result_32.length,
                stale_8=result_8.stale_fraction,
                stale_32=result_32.stale_fraction,
                pending_8=float(result_8.pending),
                pending_32=float(result_32.pending),
            )
        )
    return deltas


def print_scenario(name: str, deltas: list[TrialDelta]) -> None:
    print(
        f"{name:<35} {_format_interval([d.reward for d in deltas]):>29} "
        f"{_format_interval([d.length for d in deltas], digits=0):>29} "
        f"{np.mean([d.stale_8 for d in deltas]):>9.3%} "
        f"{np.mean([d.stale_32 for d in deltas]):>10.3%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--accepted-groups",
        type=int,
        default=4_000,
        help="Accepted prompt groups per run for sample-aligned scenarios.",
    )
    parser.add_argument(
        "--equal-step-count",
        type=int,
        default=50,
        help="Optimizer steps per run for the deliberately misaligned scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if args.accepted_groups % 32:
        raise SystemExit("--accepted-groups must be divisible by 32")

    print(
        "Reported deltas are 8x32 minus 32x32; intervals are empirical 95% intervals."
    )
    print(
        f"{'scenario':<35} {'reward delta':>29} {'length delta (tokens)':>29} "
        f"{'stale 8':>9} {'stale 32':>10}"
    )
    print("-" * 118)

    equal_samples = dict(
        trials=args.trials,
        seed=args.seed,
        group_size=args.group_size,
        accepted_groups_8=args.accepted_groups,
        accepted_groups_32=args.accepted_groups,
        drift_strength=0.0,
    )
    print_scenario(
        "completion order only",
        run_scenario(
            **equal_samples,
            stale_dropping=False,
            failure_probability_8=0.0,
            failure_probability_32=0.0,
        ),
    )
    print_scenario(
        "+ step-age stale dropping",
        run_scenario(
            **equal_samples,
            stale_dropping=True,
            failure_probability_8=0.0,
            failure_probability_32=0.0,
        ),
    )
    print_scenario(
        "+ 2% partial failures at 32x32",
        run_scenario(
            **equal_samples,
            stale_dropping=False,
            failure_probability_8=0.0,
            failure_probability_32=0.02,
        ),
    )
    print_scenario(
        "+ 5% partial failures at 32x32",
        run_scenario(
            **equal_samples,
            stale_dropping=False,
            failure_probability_8=0.0,
            failure_probability_32=0.05,
        ),
    )
    print_scenario(
        "+ 10% partial failures at 32x32",
        run_scenario(
            **equal_samples,
            stale_dropping=False,
            failure_probability_8=0.0,
            failure_probability_32=0.10,
        ),
    )

    # This scenario intentionally compares equal optimizer steps, exactly as a
    # default W&B x-axis does.  The 32x32 run therefore consumes 4x the prompts.
    equal_steps_8 = args.equal_step_count * 8
    equal_steps_32 = args.equal_step_count * 32
    equal_steps = dict(
        trials=args.trials,
        seed=args.seed,
        group_size=args.group_size,
        accepted_groups_8=equal_steps_8,
        accepted_groups_32=equal_steps_32,
        stale_dropping=False,
        failure_probability_8=0.0,
        failure_probability_32=0.0,
    )
    print_scenario(
        "equal steps, stationary data",
        run_scenario(
            **equal_steps,
            drift_strength=0.0,
        ),
    )
    print_scenario(
        "equal steps + dataset-order drift",
        run_scenario(
            **equal_steps,
            drift_strength=0.6,
        ),
    )


if __name__ == "__main__":
    main()
