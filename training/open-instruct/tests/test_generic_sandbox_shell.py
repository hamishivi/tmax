"""Exercise the real shell wrapper without importing the training/GPU stack."""

import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class GenericSandboxShellTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).parents[1] / "open_instruct/environments/generic_sandbox.py"
        module = ast.parse(source.read_text())
        wrapper = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_BASH_WRAPPER" for target in node.targets)
        )
        wrapper = wrapper.replace("/tmp/.sandbox_", str(self.root / ".sandbox_"))
        self.script = self.root / "wrapper.sh"
        self.script.write_text(wrapper)
        (self.root / ".sandbox_env").write_text("")
        (self.root / ".sandbox_cwd").write_text(str(self.root))

    def run_shell(self, command, key):
        env = os.environ.copy()
        env["FAKEROOTKEY"] = key
        return subprocess.run(
            ["bash", str(self.script), command], env=env, capture_output=True, text=True, check=False
        )

    def test_runtime_key_is_not_saved_and_next_call_uses_fresh_key(self):
        first = self.run_shell(
            'export USER_VALUE="hello world"; mkdir child; cd child; printf "%s" "$FAKEROOTKEY"', "first"
        )
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, "first")
        self.assertNotIn("FAKEROOTKEY", (self.root / ".sandbox_env").read_text())
        second = self.run_shell('printf "%s|%s|%s" "$FAKEROOTKEY" "$USER_VALUE" "$PWD"', "second")
        self.assertEqual(second.returncode, 0)
        self.assertEqual(second.stdout, f"second|hello world|{self.root}/child")

    def test_command_failure_status_is_preserved(self):
        result = self.run_shell("false", "runtime")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("FAKEROOTKEY", (self.root / ".sandbox_env").read_text())


if __name__ == "__main__":
    unittest.main()
