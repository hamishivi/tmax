import unittest

from open_instruct.grpo_fast import WeightSyncTrigger


class TestWeightSyncTrigger(unittest.TestCase):
    def test_completion_is_monotonic(self):
        trigger = WeightSyncTrigger()

        self.assertFalse(trigger.wait_until_complete(0, timeout=0.0))
        trigger.mark_complete(2)

        self.assertTrue(trigger.wait_until_complete(0, timeout=0.0))
        self.assertTrue(trigger.wait_until_complete(2, timeout=0.0))
        self.assertFalse(trigger.wait_until_complete(3, timeout=0.0))
        with self.assertRaisesRegex(RuntimeError, "out of order"):
            trigger.mark_complete(1)


if __name__ == "__main__":
    unittest.main()
