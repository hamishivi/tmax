"""Run with TBLITE_SOURCE set to the pinned, complete upstream task corpus."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PreparationTest(unittest.TestCase):
    def test_grader_controlled_cache_text_keeps_failure_status(self):
        guard = Path(__file__).resolve().parent / "offline_guard.sh"
        for message in (
            "Cannot access central in offline mode",
            "has not been downloaded from it before",
            "No cached version of example available for offline mode",
            "No cached resource available for offline mode",
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; run_offline_verifier bash -c \'echo "$1"; exit 7\' _ "$2"',
                        "_",
                        str(guard),
                        message,
                    ],
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 7)
                self.assertNotIn(b"SETUP-FAILED", result.stderr)

    def test_output_file_failure_aborts_before_reward(self):
        guard = Path(__file__).resolve().parent / "offline_guard.sh"
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; mktemp() { return 1; }; run_offline_verifier true; echo REWARD_ZERO',
                "_",
                str(guard),
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 90)
        self.assertNotIn(b"REWARD_ZERO", result.stdout)
        self.assertIn(b"SETUP-FAILED", result.stderr)

    def test_offline_guard_preserves_model_failures(self):
        guard = Path(__file__).resolve().parent / "offline_guard.sh"
        for message, status, expected in (
            ("No cached version of example:lib:1 available for offline mode.", 1, 1),
            ("Cannot access central in offline mode", 1, 1),
            ("Compilation failure", 1, 1),
            ("AssertionError: incorrect result", 1, 1),
            ("All tests passed", 0, 0),
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; run_offline_verifier bash -c \'echo "$1"; exit "$2"\' _ "$2" "$3"',
                        "_",
                        str(guard),
                        message,
                        str(status),
                    ],
                    capture_output=True,
                )
                self.assertEqual(result.returncode, expected)
                self.assertEqual(b"SETUP-FAILED" in result.stderr, expected == 90)

    def test_maven_rejects_invocation_and_heredoc_drift(self):
        scripts = Path(__file__).resolve().parent
        invocation = 'pytest "$TEST_DIR/test_outputs.py" -rA -v'
        cases = [
            ("prepare_maven.py", "tests/test.sh", "apt-get update && apt-get install -y maven\nuv venv\n"),
            (
                "prepare_maven.py",
                "tests/test.sh",
                f"apt-get update && apt-get install -y maven\nuv venv\n{invocation}\n{invocation}",
            ),
            ("extract_dependency_pom.py", "solution/solve.sh", "echo missing"),
            ("extract_dependency_pom.py", "solution/solve.sh", "cat > pom.xml << 'EOF'\n<project>"),
            (
                "extract_dependency_pom.py",
                "solution/solve.sh",
                "cat > pom.xml << 'EOF'\nEOF\ncat > pom.xml << 'EOF'\nEOF",
            ),
        ]
        for script, relative, contents in cases:
            with self.subTest(script=script, contents=contents), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = root / "tasks/maven-slf4j-conflict"
                path = task / relative
                path.parent.mkdir(parents=True)
                path.write_text(contents)
                shutil.copyfile(scripts / script, root / script)
                result = subprocess.run(
                    [sys.executable, str(root / script)],
                    env={**os.environ, "TBLITE_SOURCE": str(root / "tasks")},
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"ValueError", result.stderr)

    def test_mlflow_rejects_marker_and_invocation_drift(self):
        script = Path(__file__).resolve().parent / "prepare.py"
        start = "# Install curl"
        end = "# Check if we're in a valid working directory"
        invocation = "uv run pytest /tests/test_outputs.py -rA"
        cases = [
            f"{end}\n{invocation}",
            f"{start}\n{start}\n{end}\n{invocation}",
            f"{end}\n{start}\n{invocation}",
            f"{start}\n{end}\n{end}\n{invocation}",
            f"{start}\n{end}\n",
            f"{start}\n{end}\n{invocation}\n{invocation}",
        ]
        for contents in cases:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "upstream"
                for name in ("maven-slf4j-conflict", "okhttp-trailers-crash", "breast-cancer-mlflow"):
                    tests = source / name / "tests"
                    tests.mkdir(parents=True)
                    (tests / "test.sh").write_text(contents)
                shutil.copyfile(script, root / "prepare.py")
                shutil.copyfile(script.parent / "offline_guard.sh", root / "offline_guard.sh")
                result = subprocess.run(
                    [sys.executable, str(root / "prepare.py")],
                    env={**os.environ, "TBLITE_SOURCE": str(source)},
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"ValueError", result.stderr)

    @unittest.skipUnless(os.environ.get("TBLITE_SOURCE"), "Requires the pinned upstream task corpus")
    def test_preserves_tasks_and_records_final_hashes(self):
        source = Path(os.environ["TBLITE_SOURCE"])
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(scripts / "offline_guard.sh", root / "offline_guard.sh")
            for name in ("prepare.py", "prepare_maven.py", "extract_dependency_pom.py"):
                shutil.copyfile(scripts / name, root / name)
                subprocess.run([sys.executable, str(root / name)], check=True)
            for name in ("maven-slf4j-conflict", "breast-cancer-mlflow", "okhttp-trailers-crash"):
                task = root / "tasks" / name
                hashes = json.loads((root / f"{name}-files.json").read_text())
                actual = {
                    str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in task.rglob("*")
                    if p.is_file()
                }
                self.assertEqual(hashes, actual)
                allowed = {"tests/test.sh"}
                if name == "maven-slf4j-conflict":
                    allowed.add("solution/solve.sh")
                for path in (source / name).rglob("*"):
                    relative = path.relative_to(source / name)
                    if path.is_file() and str(relative) not in allowed:
                        self.assertEqual(path.read_bytes(), (task / relative).read_bytes(), str(relative))
                subprocess.run(["bash", "-n", str(task / "tests/test.sh")], check=True)
                subprocess.run(["bash", "-n", str(task / "solution/solve.sh")], check=True)
            original = (source / "maven-slf4j-conflict/solution/solve.sh").read_text()
            prepared = (root / "tasks/maven-slf4j-conflict/solution/solve.sh").read_text()
            self.assertEqual(
                prepared,
                original.replace(
                    "apt-get update && apt-get install -y maven", "export MAVEN_ARGS=--offline\ncommand -v mvn"
                ),
            )
            self.assertIn("<project", (root / "dependency-pom.xml").read_text())
            repeat = subprocess.run([sys.executable, str(root / "prepare.py")], capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn(b"FileExistsError", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
