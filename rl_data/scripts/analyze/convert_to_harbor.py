#!/usr/bin/env python3
"""Convert generated RL task directories into Harbor-compatible local dataset format.

Reads tasks from a source directory (Apptainer-based format) and writes
a Harbor dataset to an output directory with the structure Harbor expects:

    task_NNNNNN_XXXXXXXX/
        instruction.md
        task.toml
        environment/
            Dockerfile
        tests/
            test.sh
            test_final_state.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

ORG_NAME = "tmax"


def parse_container_def(def_path: Path) -> tuple[str, str]:
    """Extract base image and %post body from an Apptainer container.def."""
    text = def_path.read_text()

    match = re.search(r"^From:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    base_image = match.group(1).strip() if match else "ubuntu:22.04"

    match = re.search(r"%post\s*\n(.*)", text, re.DOTALL)
    post_body = match.group(1).rstrip() if match else ""

    return base_image, post_body


def generate_dockerfile(base_image: str, post_body: str) -> str:
    """Convert Apptainer %post commands into a Dockerfile.

    The %post section can contain complex multi-line heredocs, inline
    Python scripts, gcc invocations, etc. To preserve exact semantics
    we write the whole block as a single shell script executed via
    RUN bash -c, using a heredoc.
    """
    lines = [
        f"FROM {base_image}",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "",
    ]

    if post_body.strip():
        # Remove the leading `export DEBIAN_FRONTEND=noninteractive` if present
        # since we already set it as an ENV directive.
        cleaned = re.sub(
            r"^\s*export\s+DEBIAN_FRONTEND=noninteractive\s*\n?",
            "",
            post_body,
            count=1,
        )
        # Write the post body as a script copied into the image and executed.
        # This preserves heredocs, multi-line commands, etc.
        lines.append("COPY post_install.sh /tmp/post_install.sh")
        lines.append("RUN bash /tmp/post_install.sh && rm /tmp/post_install.sh")

    lines.append("")
    return "\n".join(lines)


def generate_task_toml(task_meta: dict) -> str:
    """Generate a task.toml from task.json metadata."""
    name = task_meta.get("name", "unknown")
    scenario = task_meta.get("scenario", "")
    domain = task_meta.get("domain", "")
    skill_type = task_meta.get("skill_type", "")
    primitive_skills = task_meta.get("primitive_skills", [])
    task_complexity = task_meta.get("task_complexity", "")
    command_complexity = task_meta.get("command_complexity", "")
    language = task_meta.get("language", "")

    skills_toml = ", ".join(f'"{s}"' for s in primitive_skills)

    return textwrap.dedent(f"""\
        schema_version = "1.1"

        [task]
        name = "{ORG_NAME}/{name}"
        description = "{_escape_toml(scenario)}"

        [metadata]
        domain = "{_escape_toml(domain)}"
        skill_type = "{_escape_toml(skill_type)}"
        primitive_skills = [{skills_toml}]
        task_complexity = "{_escape_toml(task_complexity)}"
        command_complexity = "{_escape_toml(command_complexity)}"
        scenario = "{_escape_toml(scenario)}"
        language = "{_escape_toml(language)}"

        [agent]
        timeout_sec = 600.0

        [verifier]
        timeout_sec = 120.0

        [environment]
        cpus = 1
        memory_mb = 2048
        allow_internet = true
    """)


def _escape_toml(s: str) -> str:
    """Escape special characters for TOML double-quoted strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


TEST_SH = textwrap.dedent("""\
    #!/bin/bash
    set -e

    mkdir -p /logs/verifier

    cd /tests
    python3 -m pytest test_final_state.py -v 2>&1 | tee /logs/verifier/test-stdout.txt
    TEST_EXIT=${PIPESTATUS[0]}

    if [ $TEST_EXIT -eq 0 ]; then
        echo 1 > /logs/verifier/reward.txt
    else
        echo 0 > /logs/verifier/reward.txt
    fi

    exit 0
""")


def convert_task(src_dir: Path, dst_dir: Path) -> str | None:
    """Convert a single task directory. Returns task name on success, None on skip."""
    task_json_path = src_dir / "task.json"
    container_def_path = src_dir / "container.def"
    test_final_path = src_dir / "test_final_state.py"

    if not task_json_path.exists():
        return None
    if not container_def_path.exists():
        return None
    if not test_final_path.exists():
        return None

    task_meta = json.loads(task_json_path.read_text())
    task_name = task_meta.get("name", src_dir.name)
    description = task_meta.get("description", "")

    if not description.strip():
        return None

    out_task = dst_dir / src_dir.name
    out_task.mkdir(parents=True, exist_ok=True)

    # 1. instruction.md
    (out_task / "instruction.md").write_text(description)

    # 2. task.toml
    (out_task / "task.toml").write_text(generate_task_toml(task_meta))

    # 3. environment/Dockerfile + post_install.sh
    env_dir = out_task / "environment"
    env_dir.mkdir(exist_ok=True)

    base_image, post_body = parse_container_def(container_def_path)
    (env_dir / "Dockerfile").write_text(generate_dockerfile(base_image, post_body))

    cleaned_post = re.sub(
        r"^\s*export\s+DEBIAN_FRONTEND=noninteractive\s*\n?",
        "",
        post_body,
        count=1,
    )
    if cleaned_post.strip():
        (env_dir / "post_install.sh").write_text(cleaned_post)

    # 4. tests/test.sh + test_final_state.py
    tests_dir = out_task / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text(TEST_SH)
    shutil.copy2(test_final_path, tests_dir / "test_final_state.py")

    return task_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RL task directories to Harbor dataset format"
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing task_* subdirectories",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory for Harbor dataset (default: <src>_harbor)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src = args.src.resolve()
    dst = (args.dst or Path(str(src) + "_harbor")).resolve()

    if not src.is_dir():
        parser.error(f"Source directory does not exist: {src}")

    task_dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("task_"))
    logger.info("Found %d task directories in %s", len(task_dirs), src)

    dst.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_task, td, dst): td for td in task_dirs}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                converted += 1
            else:
                skipped += 1
                logger.debug("Skipped %s", futures[future].name)

            total = converted + skipped
            if total % 1000 == 0:
                logger.info("Progress: %d/%d (converted=%d, skipped=%d)", total, len(task_dirs), converted, skipped)

    logger.info(
        "Done. Converted %d tasks, skipped %d. Output: %s",
        converted,
        skipped,
        dst,
    )


if __name__ == "__main__":
    main()
