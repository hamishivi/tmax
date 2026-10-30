import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from capture_react_manifests import capture


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_bytes_and_optional_missing_files(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "trial"
            data = b'{ "dependencies": { "react": "18.3.1" } }\n'

            async def download(source, target):
                target.write_bytes(data)

            environment = SimpleNamespace(
                exec=AsyncMock(side_effect=[SimpleNamespace(exit_code=n) for n in (0, 1, 1)]),
                download_file=AsyncMock(side_effect=download),
            )
            inventory = await capture(environment, output)
            self.assertEqual((output / "package.json").read_bytes(), data)
            self.assertIsNone(inventory["package-lock.json"])
            self.assertEqual(len(inventory["package.json"]), 64)
            self.assertEqual(environment.download_file.call_count, 1)

    async def test_transport_error_is_not_reported_as_missing_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            environment = SimpleNamespace(exec=AsyncMock(side_effect=ConnectionError("lost")))
            with self.assertRaises(ConnectionError):
                await capture(environment, Path(root) / "trial")

    async def test_existing_capture_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            await capture(None, root)
