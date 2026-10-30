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


if __name__ == "__main__":
    unittest.main()
