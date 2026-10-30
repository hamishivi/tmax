"""Create explicitly labelled task variants; never modify upstream task inputs."""

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ["TBLITE_SOURCE"])

for name in ("maven-slf4j-conflict", "okhttp-trailers-crash", "breast-cancer-mlflow"):
    source = UPSTREAM / name
    target = ROOT / "tasks" / name
    shutil.copytree(source, target)
    original = (source / "tests/test.sh").read_text()
    if name in ("maven-slf4j-conflict", "okhttp-trailers-crash"):
        shutil.copyfile(ROOT / "offline_guard.sh", target / "tests/offline_guard.sh")
    if name == "okhttp-trailers-crash":
        invocation = "./gradlew :okhttp:jvmTest --tests"
        if original.count(invocation) != 1:
            raise ValueError("Expected exactly one OkHttp Gradle invocation")
        modified = original.replace(
            invocation, "source /tests/offline_guard.sh\nrun_offline_verifier " + invocation, 1
        )
        (target / "tests/test.sh").write_text(modified)
    if name == "breast-cancer-mlflow":
        for marker in ("# Install curl", "# Check if we're in a valid working directory"):
            if original.count(marker) != 1:
                raise ValueError(f"Expected exactly one bootstrap marker: {marker}")
        start = original.index("# Install curl")
        end = original.index("# Check if we're in a valid working directory", start)
        modified = (
            original[:start]
            + "set -e\nexport UV_CACHE_DIR=/opt/uv-cache UV_OFFLINE=1 UV_LINK_MODE=copy\n\n"
            + original[end:]
        )
        invocation = "uv run pytest /tests/test_outputs.py -rA"
        if modified.count(invocation) != 1:
            raise ValueError("Expected exactly one MLflow pytest invocation")
        modified = modified.replace(invocation, "set +e\n" + invocation, 1)
        (target / "tests/test.sh").write_text(modified)
    assert (target / "tests/test.sh").is_file()
    # Verifier assertions and solution are byte-identical; only bootstrap may differ.
    for path in source.rglob("*"):
        if path.is_file() and path.relative_to(source).as_posix() != "tests/test.sh":
            assert path.read_bytes() == (target / path.relative_to(source)).read_bytes()
    receipt = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    (ROOT / f"{name}-files.json").write_text(json.dumps(receipt, indent=2))
