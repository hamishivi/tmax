"""Cache additions must never install a repaired app into the runtime image."""

import tempfile
import unittest
from pathlib import Path

from prepare_react import prepare


class ReactCacheTest(unittest.TestCase):
    def test_extra_dependencies_only_in_cache_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            for name in ("environment/app", "solution", "tests"):
                (source / name).mkdir(parents=True)
            (source / "environment/Dockerfile").write_text("FROM node:20\nCOPY app /app\n")
            (source / "solution/solve.sh").write_text(
                'cat > package.json << \'EOF\'\n{"dependencies": {}, "devDependencies": {}}\nEOF\n'
            )
            (source / "tests/test.sh").write_text(
                "# Install system dependencies\nold bootstrap\n"
                "# Check if we're in a valid working directory\noriginal grader\n"
            )
            preserved = ("environment/app/package.json", "tests/test_outputs.py", "task.toml", "instruction.md")
            for path in preserved:
                (source / path).write_text("original")
            output = Path(directory) / "output"
            prepare(source, output, ["@types/react@18.3.10", "identity-obj-proxy@3.0.0"])
            dockerfile = (output / "environment/Dockerfile").read_text()
            stage, runtime = dockerfile.rsplit("FROM original\n", 1)
            self.assertIn("--ignore-scripts", stage)
            self.assertIn("@types/react@18.3.10", stage)
            self.assertNotIn("cache-extra", runtime)
            self.assertIn("COPY --from=dependency_cache /root/.npm /opt/npm-cache", runtime)
            self.assertNotIn("node_modules", runtime)
            for path in preserved:
                self.assertEqual((source / path).read_bytes(), (output / path).read_bytes())

    def test_reject_unpinned_or_nonregistry_sources(self):
        for package in ("react", "react@latest", "react@^18.0.0", "file:/app", "https://example.com/pkg.tgz"):
            with self.subTest(package=package), self.assertRaises(ValueError):
                prepare(Path("unused"), Path("unused"), [package])
