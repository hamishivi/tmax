import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("package", Path(__file__).with_name("package.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageTest(unittest.TestCase):
    def test_release_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            task = source / "example"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("cpus = 1\n")
            (task / "instruction.md").write_text("Solve the task.\n")
            entry = {"status": "unchanged", "source_files": package.hashes(task)}
            plan = {"source_revision": "pinned", "tasks": {"example": entry}}
            out = root / "release"
            manifest = package.build(source, root / "repairs", plan, out)
            self.assertEqual(manifest["included_count"], 1)
            self.assertEqual(package.hashes(task), package.hashes(out / "tasks/example"))
            with self.assertRaises(FileExistsError):
                package.build(source, root / "repairs", plan, out)
            entry["status"] = "pending"
            with self.assertRaisesRegex(ValueError, "Unvalidated"):
                package.build(source, root / "repairs", plan, root / "pending")
            self.assertFalse((root / "pending").exists())
            entry["status"] = "unchanged"
            (task / "instruction.md").write_text("Changed!")
            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                package.build(source, root / "repairs", plan, root / "drift")

    def test_protected_change_rejected_even_if_listed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            task = source / "example"
            task.mkdir(parents=True)
            (task / "task.toml").write_text("cpus = 1")
            original = package.hashes(task)
            repaired = root / "repairs/example"
            repaired.mkdir(parents=True)
            (repaired / "task.toml").write_text("cpus = 100")
            final = package.hashes(repaired)
            entry = {
                "status": "validated-repair",
                "source_files": original,
                "evidence": "oracle receipt",
                "image_sha256": "pinned-image",
                "changes": {"task.toml": {"before": original["task.toml"], "after": final["task.toml"]}},
            }
            with self.assertRaisesRegex(ValueError, "resources or configuration"):
                package.build(
                    source, root / "repairs", {"source_revision": "pinned", "tasks": {"example": entry}}, root / "out"
                )

    def test_only_offline_verifier_env_additions_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            config = "[environment]\ncpus = 2\n[verifier]\ntimeout_sec = 180\n"
            (source / "task.toml").write_text(config)
            offline = '\n[verifier.env]\nUV_OFFLINE = "1"\n'
            (target / "task.toml").write_text(config + offline)
            package.check_task_config(source, target)
            for invalid in (
                (config + offline).replace("cpus = 2", "cpus = 4"),
                (config + offline).replace("180", "900"),
                config + '\n[verifier.env]\nPYTHONPATH = "/solution"\n',
                config + '\n[verifier.env]\nUV_OFFLINE = "0"\n',
            ):
                with self.subTest(config=invalid):
                    (target / "task.toml").write_text(invalid)
                    with self.assertRaises(ValueError):
                        package.check_task_config(source, target)
            (source / "task.toml").write_text(config + offline)
            (target / "task.toml").write_text(config)
            with self.assertRaises(ValueError):
                package.check_task_config(source, target)


if __name__ == "__main__":
    unittest.main()
