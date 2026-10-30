"""CPU-only checks of the production reward methods (no GPU imports)."""

import ast
import math
import os
import shlex
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load_methods():
    path = Path(__file__).parents[1] / "open_instruct/environments/swerl_vanillux_sandbox.py"
    tree = ast.parse(path.read_text())
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_parse_reward",
            "_run_tests",
            "_grade",
            "_effective_test_timeout",
            "_verifier_budget_was_clamped",
            "_tests_destination",
        }
    ]
    namespace = {
        "math": math,
        "os": os,
        "logger": Mock(),
        "shlex": shlex,
        "uuid": uuid,
        "StepResult": lambda **kwargs: SimpleNamespace(**{"metadata": {}, **kwargs}),
        "truncate_observation": lambda text: text,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return type(
        "Harness",
        (),
        {
            name: namespace[name]
            for name in {
                "_parse_reward",
                "_run_tests",
                "_grade",
                "_effective_test_timeout",
                "_verifier_budget_was_clamped",
                "_tests_destination",
            }
        },
    )


class RewardValidityTest(unittest.TestCase):
    def setUp(self):
        self.env = load_methods()()
        self.env._has_task_setup = False
        self.env._backend = Mock()
        self.env._tests_dir = "tests"
        self.env._task_id = "task"
        self.env._test_timeout = 600
        self.env._task_test_timeout = None
        self.env._max_verifier_timeout_s = 0
        self.env._backend_kwargs = {}
        self.env._upload_directory = Mock()

    def test_numeric_domain(self):
        for text, expected in [
            ("0", 0),
            ("1", 1),
            ("0.43", 0.43),
            ("nan", None),
            ("inf", None),
            ("-0.01", None),
            ("1.01", None),
            ("", None),
            ("bad", None),
        ]:
            with self.subTest(text=text):
                self.env._backend.run_command.return_value = SimpleNamespace(exit_code=0, stdout=text)
                self.assertEqual(self.env._parse_reward(), expected)

    def run_verifier(self, exit_code, marker, reward):
        self.env._backend.run_command.side_effect = [
            SimpleNamespace(exit_code=0),
            SimpleNamespace(stdout="EXISTS"),
            SimpleNamespace(exit_code=exit_code, stdout="logs", stderr="", timed_out=None),
            SimpleNamespace(exit_code=0, stdout=marker),
            SimpleNamespace(exit_code=0, stdout=reward),
        ]
        return self.env._run_tests()

    def test_missing_reward_is_flagged(self):
        result = self.run_verifier(1, "1", "")
        self.assertTrue(result.metadata["invalid_reward"])

    def test_genuine_timeout_is_valid_zero(self):
        result = self.run_verifier(124, "", "")
        self.assertTrue(result.metadata["timeout"])
        self.assertNotIn("invalid_reward", result.metadata)
        self.assertEqual(result.reward, 0)

    def test_ordinary_exit_124_is_not_timeout(self):
        result = self.run_verifier(124, "124", "")
        self.assertTrue(result.metadata["invalid_reward"])

    def test_timeout_preserves_valid_verifier_reward(self):
        for reward in ("0", ".625", "1"):
            with self.subTest(reward=reward):
                result = self.run_verifier(124, "", reward)
                self.assertTrue(result.metadata["timeout"])
                self.assertEqual(result.reward, float(reward))

    def test_fractional_reward_survives_nonzero_exit(self):
        self.assertEqual(self.run_verifier(1, "1", ".625").reward, 0.625)
