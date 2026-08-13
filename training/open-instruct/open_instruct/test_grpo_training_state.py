import unittest
from typing import Any, cast

import torch

from open_instruct import grpo_utils


class _FakeDeepSpeedEngine(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.empty_partition_cache_calls = 0

    def empty_partition_cache(self) -> None:
        self.empty_partition_cache_calls += 1


class _OptimizerWrapper:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer


class TestReferencePolicyUpdate(unittest.TestCase):
    def test_dense_polyak_update(self):
        policy = torch.nn.Linear(2, 1, bias=False)
        ref_policy = torch.nn.Linear(2, 1, bias=False)
        policy.weight.data.fill_(3.0)
        ref_policy.weight.data.fill_(1.0)

        grpo_utils.update_reference_policy(policy, ref_policy, alpha=0.25)

        torch.testing.assert_close(ref_policy.weight, torch.full_like(ref_policy.weight, 1.5))

    def test_zero3_updates_local_shards_without_gathering(self):
        policy = torch.nn.Linear(2, 1, bias=False)
        ref_policy = torch.nn.Linear(2, 1, bias=False)
        policy_weight = cast(Any, policy.weight)
        ref_policy_weight = cast(Any, ref_policy.weight)
        policy_weight.ds_shape = policy.weight.shape
        ref_policy_weight.ds_shape = ref_policy.weight.shape
        policy_weight.ds_tensor = torch.full((1,), 5.0)
        ref_policy_weight.ds_tensor = torch.full((1,), 1.0)
        policy_engine = _FakeDeepSpeedEngine(policy)
        ref_engine = _FakeDeepSpeedEngine(ref_policy)

        grpo_utils.update_reference_policy(policy_engine, ref_engine, alpha=0.5, deepspeed_stage_3=True)

        torch.testing.assert_close(ref_policy_weight.ds_tensor, torch.full((1,), 3.0))
        self.assertEqual(policy_engine.empty_partition_cache_calls, 1)
        self.assertEqual(ref_engine.empty_partition_cache_calls, 1)

    def test_zero3_rejects_mismatched_shards(self):
        policy = torch.nn.Linear(2, 1, bias=False)
        ref_policy = torch.nn.Linear(2, 1, bias=False)
        policy_weight = cast(Any, policy.weight)
        ref_policy_weight = cast(Any, ref_policy.weight)
        policy_weight.ds_shape = policy.weight.shape
        ref_policy_weight.ds_shape = ref_policy.weight.shape
        policy_weight.ds_tensor = torch.ones(1)
        ref_policy_weight.ds_tensor = torch.ones(2)

        with self.assertRaisesRegex(ValueError, "shard shapes do not match"):
            grpo_utils.update_reference_policy(
                _FakeDeepSpeedEngine(policy), _FakeDeepSpeedEngine(ref_policy), alpha=1.0, deepspeed_stage_3=True
            )


class TestOptimizerReset(unittest.TestCase):
    def test_clears_wrapped_optimizer_state_and_preserves_lr(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.123)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertTrue(optimizer.state)

        grpo_utils.reset_optimizer_state(_OptimizerWrapper(optimizer))

        self.assertFalse(optimizer.state)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.123)

    def test_config_accepts_positive_frequency(self):
        config = grpo_utils.GRPOExperimentConfig(optimizer_reset_freq=10)

        self.assertEqual(config.optimizer_reset_freq, 10)

    def test_config_rejects_non_positive_frequencies(self):
        with self.assertRaisesRegex(ValueError, "optimizer_reset_freq"):
            grpo_utils.GRPOExperimentConfig(optimizer_reset_freq=0)
        with self.assertRaisesRegex(ValueError, "ref_policy_update_freq"):
            grpo_utils.GRPOExperimentConfig(ref_policy_update_freq=0)


if __name__ == "__main__":
    unittest.main()
