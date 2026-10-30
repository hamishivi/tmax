import unittest

from integrate_images import compose


class ComposeTest(unittest.TestCase):
    def test_context_is_self_contained_and_preserves_runtime_stage(self):
        original = "FROM example:1\nCOPY src /app\n"
        repair = "FROM external:1 AS tools\nRUN install-tools\nFROM tools AS cache\nRUN warm\nFROM tools\nCOPY --from=cache /cache /cache\n"
        result = compose(original, repair, "external:1")
        self.assertTrue(result.startswith("FROM example:1 AS tblite_upstream\nCOPY src /app\n"))
        self.assertIn("FROM tblite_upstream AS tools", result)
        self.assertNotIn("external:1", result)
        self.assertTrue(result.endswith("FROM tools\nCOPY --from=cache /cache /cache\n"))

    def test_unknown_or_repeated_builds_fail(self):
        for original in ("FROM x AS base", "FROM x\nFROM y", "RUN false"):
            with self.assertRaises(ValueError):
                compose(original, "FROM external:1", "external:1")
        with self.assertRaises(ValueError):
            compose("FROM x", "FROM unexpected:1", "external:1")


if __name__ == "__main__":
    unittest.main()
