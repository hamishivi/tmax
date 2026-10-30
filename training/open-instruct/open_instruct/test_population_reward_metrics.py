import random
import unittest

from open_instruct import population_reward_metrics


class TestPopulationRewardMetrics(unittest.TestCase):
    def make_metrics(self, enabled=True, max_score=1.0):
        return population_reward_metrics.PopulationRewardMetrics(
            max_possible_score=max_score, mask_truncated_completions=enabled, response_length=10
        )

    def test_overlong_all_zero_group_is_removed_only_from_post_filter(self):
        metrics = self.make_metrics()
        metrics.add_group([0, 0], ["stop", "stop"], [10, 12])
        metrics.add_group([0, 1], ["stop", "stop"], [4, 5])
        values = metrics.as_metrics()
        self.assertEqual(values["val/avg_group_performance_pre_filter"], 0.25)
        self.assertEqual(values["val/avg_group_performance_post_filter"], 0.5)
        self.assertEqual(values["val/reward_count_pre_filter"], 4)
        self.assertEqual(values["val/reward_count_post_filter"], 2)

    def test_overlong_all_one_group_is_removed_only_from_post_filter(self):
        metrics = self.make_metrics()
        metrics.add_group([1, 1], ["length", "length"], [4, 5])
        metrics.add_group([0, 1], ["stop", "stop"], [4, 5])
        values = metrics.as_metrics()
        self.assertEqual(values["val/avg_group_performance_pre_filter"], 0.75)
        self.assertEqual(values["val/avg_group_performance_post_filter"], 0.5)

    def test_partial_group_removal_uses_completion_count_not_original_group_weight(self):
        metrics = self.make_metrics()
        metrics.add_group([0, 0], ["stop", "stop"], [3, 3])
        metrics.add_group([0, 1], ["length", "stop"], [3, 3])
        values = metrics.as_metrics()
        self.assertEqual(values["val/avg_group_performance_pre_filter"], 0.25)
        self.assertAlmostEqual(values["val/avg_group_performance_post_filter"], 1 / 3)
        self.assertEqual(values["val/reward_count_post_filter"], 3)

    def test_constant_intermediate_rewards_are_not_lost(self):
        metrics = self.make_metrics()
        metrics.add_group([0.25, 0.25], ["stop", "length"], [3, 3])
        metrics.add_group([0, 1], ["stop", "stop"], [3, 3])
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 0.375)
        self.assertAlmostEqual(metrics.as_metrics()["val/avg_group_performance_post_filter"], 1.25 / 3)

    def test_disabled_overlong_filter_keeps_all_samples(self):
        metrics = self.make_metrics(enabled=False)
        metrics.add_group([0, 1], ["length", "stop"], [100, 100])
        self.assertEqual(metrics.pre_count, metrics.post_count)
        self.assertEqual(metrics.pre_reward_sum, metrics.post_reward_sum)

    def test_shared_predicate_matches_existing_training_rule(self):
        for reason in ["stop", "length", "abort", None]:
            for length in [0, 9, 10, 11]:
                with self.subTest(reason=reason, length=length):
                    self.assertEqual(
                        population_reward_metrics.is_overlong(reason, length, 10), reason != "stop" or length >= 10
                    )

    def test_means_are_normalized_but_sums_are_raw(self):
        metrics = self.make_metrics(max_score=5)
        metrics.add_group([0, 5], ["stop", "stop"], [3, 3])
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 0.5)
        self.assertEqual(metrics.as_metrics()["val/reward_sum_pre_filter"], 5)

    def test_empty_post_population_has_no_mean_not_fake_zero(self):
        metrics = self.make_metrics()
        metrics.add_group([1, 1], ["length", "length"], [3, 3])
        values = metrics.as_metrics()
        self.assertEqual(values["val/avg_group_performance_pre_filter"], 1)
        self.assertEqual(values["val/reward_count_post_filter"], 0)
        self.assertNotIn("val/avg_group_performance_post_filter", values)

    def test_empty_accumulator_has_counts_without_means(self):
        values = self.make_metrics().as_metrics()
        self.assertEqual(values["val/reward_count_pre_filter"], 0)
        self.assertNotIn("val/avg_group_performance_pre_filter", values)

    def test_invalid_config_and_mismatched_record_are_rejected(self):
        with self.assertRaises(ValueError):
            population_reward_metrics.PopulationRewardMetrics(max_possible_score=0)
        with self.assertRaises(ValueError):
            population_reward_metrics.PopulationRewardMetrics(mask_truncated_completions=True)
        metrics = self.make_metrics()
        with self.assertRaises(ValueError):
            metrics.add_group([0, 1], ["stop"], [3, 3])
        self.assertEqual(metrics.pre_count, 0)

    def test_random_groups_match_direct_population_calculation(self):
        rng = random.Random(20260904)
        for enabled in [False, True]:
            metrics = self.make_metrics(enabled=enabled)
            population = []
            for _ in range(100):
                rewards = [rng.choice([0.0, 0.25, 1.0]) for _ in range(4)]
                reasons = [rng.choice(["stop", "length"]) for _ in rewards]
                lengths = [rng.randrange(1, 13) for _ in rewards]
                metrics.add_group(rewards, reasons, lengths)
                population.extend(zip(rewards, reasons, lengths))
            before = [reward for reward, _, _ in population]
            after = [
                reward for reward, reason, length in population if not enabled or (reason == "stop" and length < 10)
            ]
            self.assertEqual(metrics.pre_count, len(before))
            self.assertEqual(metrics.post_count, len(after))
            self.assertAlmostEqual(
                metrics.as_metrics()["val/avg_group_performance_pre_filter"], sum(before) / len(before)
            )
            self.assertAlmostEqual(
                metrics.as_metrics()["val/avg_group_performance_post_filter"], sum(after) / len(after)
            )


if __name__ == "__main__":
    unittest.main()
