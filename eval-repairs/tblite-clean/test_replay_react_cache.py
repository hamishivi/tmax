import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replay_react_cache import replay


class ReplayTest(unittest.TestCase):
    def test_preserves_inputs_and_failed_install_output(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            for name in ("package.json", "package-lock.json", "npm-shrinkwrap.json"):
                (trial / name).write_text("{}\n")
            completed = subprocess.CompletedProcess([], 1, "npm output", "ENOTCACHED")
            with patch("replay_react_cache.subprocess.run", return_value=completed) as run:
                result = replay("sha256:image", trial, trial)
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["stderr"], "ENOTCACHED")
            self.assertEqual(len(result["input_sha256"]), 3)
            self.assertEqual(result["input_sha256"]["package.json"], hashlib.sha256(b"{}\n").hexdigest())
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("--ignore-scripts", command[-1])
            self.assertIn("npm-shrinkwrap.json", command[-1])
            self.assertEqual((trial / "package.json").read_text(), "{}\n")

    def test_missing_manifest_fails_before_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch("replay_react_cache.subprocess.run") as run:
            with self.assertRaises(ValueError):
                replay("image", Path(directory), Path(directory))
            run.assert_not_called()
