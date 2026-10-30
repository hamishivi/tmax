import ast
import threading
import unittest
from pathlib import Path

# Load the production synchronization primitive without GPU inference imports.
_source = Path(__file__).with_name("grpo_fast.py")
_definition = next(
    node
    for node in ast.parse(_source.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "WeightSyncTrigger"
)
_namespace = {"threading": threading}
exec(compile(ast.Module(body=[_definition], type_ignores=[]), str(_source), "exec"), _namespace)
WeightSyncTrigger = _namespace["WeightSyncTrigger"]


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
