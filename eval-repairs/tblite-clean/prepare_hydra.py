"""Create a self-contained offline Hydra Harbor task candidate."""

import argparse
import shutil
from pathlib import Path


def prepare(source, output):
    original = (source / "tests/test.sh").read_text()
    start = "# Install curl\n"
    end = "# Check if we're in a valid working directory\n"
    command = "uv pip install git+https://github.com/facebookresearch/hydra.git@v1.3.0#subdirectory=plugins/hydra_submitit_launcher"
    if any(original.count(marker) != 1 for marker in (start, end, command)):
        raise ValueError("Unexpected upstream Hydra verifier")
    first, last = original.index(start), original.index(end)
    if first >= last:
        raise ValueError("Unexpected bootstrap order")
    modified = original[:first] + "set -e\nexport UV_OFFLINE=1 UV_FIND_LINKS=/opt/wheelhouse\n\n" + original[last:]
    modified = modified.replace(
        command, "uv pip install /opt/wheelhouse/hydra_submitit_launcher-1.3.0.dev0-py3-none-any.whl"
    )
    grader = 'uv run pytest "/tests"/test_outputs.py -rA'
    if modified.count(grader) != 1:
        raise ValueError("Unexpected upstream Hydra grader")
    modified = modified.replace(grader, "set +e\n" + grader)
    dockerfile = (source / "environment/Dockerfile").read_text()
    appendix = Path(__file__).with_name("Hydra.Dockerfile.append").read_text()
    shutil.copytree(source, output)
    (output / "tests/test.sh").write_text(modified)
    (output / "environment/Dockerfile").write_text(dockerfile + appendix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
