import tempfile
import unittest
from pathlib import Path

from prepare_etl import prepare


class ETLTest(unittest.TestCase):
    def test_repairs_image_source_without_verifier_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "environment").mkdir(parents=True)
            (source / "tests").mkdir()
            (source / "environment/start_postgres.sh").write_text("pg_isready -q\npsql -c create1\npsql -c create2\n")
            (source / "tests/test.sh").write_text(
                "#!/bin/bash\n\n# Start the application in the background\n/workspace/start.sh &\nAPP_PID=$!\n\npython -m pytest tests/test_outputs.py\n"
            )
            (source / "task.toml").write_text("original resources")
            (source / "tests/test_outputs.py").write_text("original assertions")
            output = Path(directory) / "candidate"
            prepare(source, output)
            for path in ("task.toml", "tests/test_outputs.py"):
                self.assertEqual((source / path).read_bytes(), (output / path).read_bytes())
            startup = (output / "environment/start_postgres.sh").read_text()
            verifier = (output / "tests/test.sh").read_text()
            self.assertIn("flock -x 9", startup)
            self.assertIn('"$@"', startup)
            self.assertIn("/workspace/start_postgres.sh bash -c", verifier)
            self.assertNotIn("APP_PID", verifier)
            self.assertNotIn("bootstrap_ready", startup)
            self.assertIn("python -m pytest tests/test_outputs.py", verifier)
            with self.assertRaises(FileExistsError):
                prepare(source, output)


if __name__ == "__main__":
    unittest.main()
