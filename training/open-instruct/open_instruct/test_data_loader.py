import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock, patch

import numpy as np
import parameterized
import torch
from datasets import Dataset

if importlib.util.find_spec("vllm") is None:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    sys.modules["vllm"] = vllm_stub

from open_instruct import data_loader, data_types, population_reward_metrics
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


def _make_dpo_dataset(num_samples: int, max_seq_length: int) -> Dataset:
    rng = torch.Generator().manual_seed(42)
    data = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "index": list(range(num_samples)),
    }
    for _ in range(num_samples):
        chosen_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        rejected_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        data["chosen_input_ids"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["chosen_labels"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["rejected_input_ids"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
        data["rejected_labels"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
    ds = Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


class TestWorldAwarePacking(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("olmo3_7b_dp2", 16384, 8, 2, True, 200),
            ("olmo3_7b_dp4", 16384, 16, 4, True, 200),
            ("olmo3_32b_dp4", 8192, 8, 4, True, 200),
            ("olmo3_32b_dp8", 8192, 16, 8, True, 200),
            ("debug_multi_node", 16384, 32, 2, True, 200),
            ("olmo3_7b_dp2_no_drop", 16384, 8, 2, False, 200),
            ("olmo3_32b_dp4_no_drop", 8192, 8, 4, False, 200),
        ]
    )
    def test_packing_equal_batches_across_ranks(
        self, _name, max_seq_length, global_batch_size, dp_world_size, drop_last, num_samples
    ):
        dataset = _make_dpo_dataset(num_samples, max_seq_length)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loaders = [
                data_loader.HFDataLoader(
                    dataset=dataset,
                    batch_size=global_batch_size,
                    seed=42,
                    dp_rank=rank,
                    dp_world_size=dp_world_size,
                    work_dir=work_dir,
                    collator=collator,
                    drop_last=drop_last,
                )
                for rank in range(dp_world_size)
            ]

            batch_counts = [loader.total_batches for loader in loaders]
            self.assertTrue(
                all(c == batch_counts[0] for c in batch_counts), f"Batch counts differ across ranks: {batch_counts}"
            )

            all_indices = set()
            for loader in loaders:
                for batch in loader:
                    if "index" in batch:
                        all_indices.update(batch["index"].tolist())

            if not drop_last:
                expected_indices = set(range(num_samples))
                self.assertEqual(all_indices, expected_indices, f"Missing indices: {expected_indices - all_indices}")


class TestComputeGroupAdvantages(unittest.TestCase):
    def test_maxrl_normalizes_by_per_prompt_mean_reward(self):
        scores = np.array([1.0, 0.0, 1.0, 0.0])

        advantages = data_loader.compute_group_advantages(
            scores=scores, num_samples_per_prompt=4, advantage_normalization_type="maxrl"
        )

        np.testing.assert_allclose(advantages, np.array([1.0, -1.0, 1.0, -1.0]))

    def test_maxrl_zeroes_groups_with_no_successes(self):
        scores = np.array([0.0, 0.0, 0.0, 0.0])

        advantages = data_loader.compute_group_advantages(
            scores=scores, num_samples_per_prompt=4, advantage_normalization_type="maxrl"
        )

        np.testing.assert_allclose(advantages, np.zeros_like(scores))

    def test_maxrl_zeroes_groups_with_all_successes(self):
        scores = np.array([1.0, 1.0, 1.0, 1.0])

        advantages = data_loader.compute_group_advantages(
            scores=scores, num_samples_per_prompt=4, advantage_normalization_type="maxrl"
        )

        np.testing.assert_allclose(advantages, np.zeros_like(scores))


class TestPopulationRewardAccumulation(unittest.TestCase):
    """Exercise the actual accumulation path without Ray actors, GPUs or model downloads."""

    def make_result(self, index, rewards, lengths, model_step=10):
        count = len(rewards)
        return data_types.GenerationResult(
            responses=[[7] * length for length in lengths],
            finish_reasons=["stop"] * count,
            masks=[[1] * length for length in lengths],
            logprobs=[[0.0] * length for length in lengths],
            request_info=data_types.RequestInfo(
                num_calls=[0] * count,
                timeouts=[0] * count,
                tool_errors=[""] * count,
                tool_outputs=[""] * count,
                tool_runtimes=[0.0] * count,
                tool_calleds=[False] * count,
                rollout_states=[{} for _ in range(count)],
            ),
            index=index,
            prompt_id=str(index),
            reward_scores=rewards,
            reward_metrics={},
            start_time=1.0,
            token_statistics=data_types.TokenStatistics(2, sum(lengths), 0.1),
            model_step=model_step,
        )

    def accumulate(self, results, metrics, num_prompts=2, **kwargs):
        queue = Queue()
        for result in results:
            queue.put(result)
        dataset = [
            {
                data_loader.INPUT_IDS_PROMPT_KEY: [1, 2],
                data_loader.GROUND_TRUTHS_KEY: "test",
                data_loader.VERIFIER_SOURCE_KEY: "test",
                data_loader.RAW_PROMPT_KEY: "test",
            }
            for _ in results
        ]
        tokenizer = Mock()
        tokenizer.batch_decode.side_effect = lambda responses, **_: ["response"] * len(responses)
        return data_loader.accumulate_inference_batches(
            queue,
            Mock(n=2),
            num_prompts=num_prompts,
            model_dims=Mock(),
            tokenizer=tokenizer,
            dataset=dataset,
            base_env_config=data_types.EnvConfig(),
            filter_zero_std_samples=True,
            population_metrics=metrics,
            timeout=0.1,
            **kwargs,
        )

    def test_counts_overlong_zero_std_group_without_adding_it_to_training(self):
        metrics = population_reward_metrics.PopulationRewardMetrics(
            mask_truncated_completions=True, response_length=10
        )
        result, batch, _, stats = self.accumulate(
            [self.make_result(0, [0, 0], [10, 10]), self.make_result(1, [0, 1], [3, 3])], metrics
        )
        self.assertEqual(batch.scores, [0, 1])
        self.assertEqual(batch.indices, [1, 1])
        self.assertEqual(len(result.responses), 2)
        self.assertEqual(stats.filtered_prompts_zero, 1)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 0.25)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_post_filter"], 0.5)

    def test_active_sampling_counts_rejected_and_replacement_groups(self):
        metrics = population_reward_metrics.PopulationRewardMetrics(
            mask_truncated_completions=True, response_length=10
        )
        _, batch, _, _ = self.accumulate(
            [self.make_result(0, [1, 1], [10, 10]), self.make_result(1, [0, 1], [3, 3])],
            metrics,
            num_prompts=1,
            active_sampling=True,
        )
        self.assertEqual(batch.scores, [0, 1])
        self.assertEqual(metrics.pre_count, 4)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 0.75)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_post_filter"], 0.5)

    def test_all_zero_std_groups_still_have_population_metrics(self):
        metrics = population_reward_metrics.PopulationRewardMetrics(
            mask_truncated_completions=True, response_length=10
        )
        output = self.accumulate([self.make_result(0, [1, 1], [3, 3])], metrics, num_prompts=1)
        self.assertEqual(output, (None, None, None, None))
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 1)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_post_filter"], 1)

    def test_stale_results_do_not_enter_current_population(self):
        metrics = population_reward_metrics.PopulationRewardMetrics()
        self.accumulate(
            [self.make_result(0, [1, 1], [3, 3], model_step=0), self.make_result(1, [0, 1], [3, 3])],
            metrics,
            num_prompts=1,
            training_step=10,
            max_result_age_steps=1,
        )
        self.assertEqual(metrics.pre_count, 2)
        self.assertEqual(metrics.as_metrics()["val/avg_group_performance_pre_filter"], 0.5)

    def test_optional_metrics_do_not_change_training_batch(self):
        groups = [self.make_result(0, [0, 0], [10, 10]), self.make_result(1, [0, 1], [3, 3])]
        baseline_result, baseline_batch, baseline_rewards, baseline_stats = self.accumulate(groups, None)
        measured_result, measured_batch, measured_rewards, measured_stats = self.accumulate(
            groups,
            population_reward_metrics.PopulationRewardMetrics(mask_truncated_completions=True, response_length=10),
        )
        self.assertEqual(baseline_result, measured_result)
        self.assertEqual(baseline_batch, measured_batch)
        self.assertEqual(baseline_rewards, measured_rewards)
        self.assertEqual(baseline_stats, measured_stats)

    def test_preparation_publishes_metrics_even_when_training_batch_is_empty(self):
        actor = Mock(
            config=Mock(
                save_traces=False,
                save_filtered_rollouts=False,
                async_steps=1,
                max_possible_score=1.0,
                mask_truncated_completions=True,
                response_length=10,
                rollouts_save_path="/unused",
            ),
            training_step=0,
            num_training_steps=1,
            global_batch_size=1,
            dp_world_size=1,
            _last_consumed_step=-1,
            lock=threading.Lock(),
            prepared_data={},
            metrics={},
        )

        def all_groups_filtered(*args, **kwargs):
            kwargs["population_metrics"].add_group([1, 1], ["stop", "stop"], [3, 3])
            return None, None, None, None

        actor_class = data_loader.DataPreparationActor.__ray_metadata__.modified_class
        with patch.object(data_loader, "accumulate_inference_batches", side_effect=all_groups_filtered):
            actor_class._data_preparation_loop(actor)
        self.assertEqual(actor.metrics[0]["val/avg_group_performance_pre_filter"], 1)
        self.assertEqual(actor.metrics[0]["val/avg_group_performance_post_filter"], 1)
        self.assertEqual(actor.prepared_data[0][0].query_responses, [])


if __name__ == "__main__":
    unittest.main()
