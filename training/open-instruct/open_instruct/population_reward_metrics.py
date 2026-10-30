"""Reward accounting over sampled completions, before zero-std selection.

These are descriptive metrics, not the reward of the selected training batch.
Both include constant-reward groups. Only the configured overlong filter affects
the post-filter population; non-submission filtering and advantage/loss masks do
not redefine it. Consequently, pre-filter and post-filter are equal whenever
``mask_truncated_completions`` is false, as on the runs active when this metric
was introduced. The population contains this step's fresh results after stale
results are rejected; each fresh resample is another observation. Accumulate raw
sums/counts so unequal retained group sizes and constant intermediate rewards are
handled without reconstructing group means.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field


def is_overlong(finish_reason: str | None, length: int, response_length: int) -> bool:
    """The training filter's existing truncation rule, including tool-token budget exhaustion."""
    return finish_reason != "stop" or length >= response_length


@dataclass
class PopulationRewardMetrics:
    max_possible_score: float = 1.0
    mask_truncated_completions: bool = False
    response_length: int | None = None
    pre_reward_sum: float = field(default=0.0, init=False)
    pre_count: int = field(default=0, init=False)
    post_reward_sum: float = field(default=0.0, init=False)
    post_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_possible_score <= 0:
            raise ValueError("max_possible_score must be positive")
        if self.mask_truncated_completions and (self.response_length is None or self.response_length <= 0):
            raise ValueError("overlong filtering requires a positive response_length")

    def add_group(
        self, rewards: Sequence[float], finish_reasons: Sequence[str | None], response_lengths: Sequence[int]
    ) -> None:
        """Record a fresh prompt group BEFORE it can be discarded for zero reward variance."""
        if not len(rewards) == len(finish_reasons) == len(response_lengths):
            raise ValueError("reward, finish-reason and response-length counts must match")
        for reward, reason, length in zip(rewards, finish_reasons, response_lengths):
            self.pre_reward_sum += float(reward)
            self.pre_count += 1
            if self.mask_truncated_completions:
                assert self.response_length is not None
                if is_overlong(reason, length, self.response_length):
                    continue
            self.post_reward_sum += float(reward)
            self.post_count += 1

    def as_metrics(self) -> dict[str, float | int]:
        """Keep legacy mean keys (normalized to max score), plus raw sums and exact denominators.

        A population with no eligible samples has no mean: omit that key instead
        of reporting a fabricated zero. Counts distinguish absence from failure.
        """
        metrics: dict[str, float | int] = {
            "val/reward_sum_pre_filter": self.pre_reward_sum,
            "val/reward_count_pre_filter": self.pre_count,
            "val/reward_sum_post_filter": self.post_reward_sum,
            "val/reward_count_post_filter": self.post_count,
        }
        if self.pre_count:
            metrics["val/avg_group_performance_pre_filter"] = (
                self.pre_reward_sum / self.pre_count / self.max_possible_score
            )
        if self.post_count:
            metrics["val/avg_group_performance_post_filter"] = (
                self.post_reward_sum / self.post_count / self.max_possible_score
            )
        return metrics
