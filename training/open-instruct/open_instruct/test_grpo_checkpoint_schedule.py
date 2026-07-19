import unittest

from open_instruct import grpo_utils
from open_instruct.utils import ArgumentParserPlus

should_capture_policy_checkpoint = grpo_utils.should_capture_policy_checkpoint


class TestPolicyCheckpointCaptureWindow(unittest.TestCase):
    def test_captures_steps_after_regular_checkpoint(self):
        captured = [
            step
            for step in range(95, 206)
            if should_capture_policy_checkpoint(step, save_freq=100, capture_checkpoint_window=4)
        ]

        self.assertEqual(captured, [100, 101, 102, 103, 104, 200, 201, 202, 203, 204])

    def test_zero_window_preserves_regular_schedule(self):
        captured = [
            step
            for step in range(1, 206)
            if should_capture_policy_checkpoint(step, save_freq=100, capture_checkpoint_window=0)
        ]

        self.assertEqual(captured, [100, 200])

    def test_disabled_saving_disables_capture_window(self):
        self.assertFalse(should_capture_policy_checkpoint(101, save_freq=0, capture_checkpoint_window=4))

    def test_step_zero_checkpoint_gets_its_own_window(self):
        captured = [
            step
            for step in range(1, 7)
            if should_capture_policy_checkpoint(
                step, save_freq=100, capture_checkpoint_window=4, eval_on_step_0=True
            )
        ]

        self.assertEqual(captured, [1, 2, 3, 4, 5])

    def test_config_rejects_negative_window(self):
        with self.assertRaisesRegex(ValueError, "capture_checkpoint_window"):
            grpo_utils.GRPOExperimentConfig(capture_checkpoint_window=-1)

    def test_hyphenated_cli_flag(self):
        parser = ArgumentParserPlus((grpo_utils.GRPOExperimentConfig,))

        (config,) = parser.parse_args_into_dataclasses(["--capture-checkpoint-window", "4"])

        self.assertEqual(config.capture_checkpoint_window, 4)


if __name__ == "__main__":
    unittest.main()
