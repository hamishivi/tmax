#!/usr/bin/env python3
"""Convert generated RL task directories into Harbor-compatible local dataset format.

Reads tasks from a source directory (Apptainer-based format) and writes
a Harbor dataset to an output directory with the structure Harbor expects:

    task_NNNNNN_XXXXXXXX/
        instruction.md
        task.toml
        environment/
            Dockerfile
            post_install.sh          # task-specific container.def %post
            base_install.sh          # (intricate tasks only) base_intricate layer
            _fixtures/...            # (tasks with a %files section) staged fixture files
        tests/
            test.sh
            test_final_state.py

Two task flavours are handled automatically (detected per-task, no flags needed):

* **legacy** tasks (e.g. tasks_skill_tax_20260401_10k): their ``container.def``
  is fully self-contained -- the ``%post`` installs every dependency on top of
  ``ubuntu:22.04``. These convert exactly as before.

* **intricate** v2 tasks (e.g. tasks_skill_tax_v2_20260506_5k): ``task.json`` has
  ``base_image == "intricate"``. At solve-time these ran on the prebuilt
  ``base_intricate.sif`` (numpy/scipy/torch-cpu/Pillow/ffmpeg/tesseract/...), so
  their ``container.def`` ``%post`` does NOT reinstall the heavy stack and their
  tests import it directly. To make a self-contained Harbor image we:
    1. inline the ``base_intricate.def`` ``%post`` as ``base_install.sh`` (or, with
       ``--intricate-base-image TAG``, build ``FROM TAG`` instead), then
    2. COPY the ``%files`` fixtures into the image (before the task ``%post``,
       matching Apptainer's %files-before-%post ordering), then
    3. RUN the task's own ``%post``.

Fixture file permissions (e.g. executable oracle binaries) are preserved; Docker
``COPY`` carries the build-context file mode into the image.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

ORG_NAME = "tmax"

# base_intricate.def lives at <repo>/rl_data/containers/base_intricate.def.
# This file is <repo>/rl_data/scripts/analyze/convert_to_harbor.py.
DEFAULT_BASE_INTRICATE_DEF = (
    Path(__file__).resolve().parents[2] / "containers" / "base_intricate.def"
)


# ---------------------------------------------------------------------------
# Apptainer .def parsing
# ---------------------------------------------------------------------------

def _extract_section(text: str, section: str) -> str:
    """Return the body of an Apptainer ``%<section>`` block (up to the next
    ``%section`` header or EOF). Empty string if the section is absent."""
    pattern = re.compile(
        rf"^%{section}\b[^\n]*\n(.*?)(?=^%\w+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).rstrip() if match else ""


def parse_container_def(def_path: Path) -> tuple[str, str]:
    """Extract base image and %post body from an Apptainer container.def."""
    text = def_path.read_text()

    match = re.search(r"^From:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    base_image = match.group(1).strip() if match else "ubuntu:22.04"

    return base_image, _extract_section(text, "post")


def parse_files_section(def_path: Path) -> list[tuple[str, str]]:
    """Parse an Apptainer ``%files`` block into ``(src, dest)`` pairs.

    Each non-empty line is ``<host_src> <container_dest>``. A line with a single
    token copies to the same path. Returns ``[]`` when there is no ``%files``
    section.
    """
    body = _extract_section(def_path.read_text(), "files")
    pairs: list[tuple[str, str]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            pairs.append((parts[0], parts[0]))
        else:
            pairs.append((parts[0], parts[1]))
    return pairs


def load_intricate_base(def_path: Path) -> tuple[str, list[str]]:
    """Read base_intricate.def → (``%post`` body, list of ``KEY=VALUE`` env vars).

    The env vars come from the ``%environment`` block's ``export`` lines and are
    emitted as Docker ``ENV`` directives.
    """
    text = def_path.read_text()
    post_body = _extract_section(text, "post")

    env_body = _extract_section(text, "environment")
    env_vars: list[str] = []
    for raw in env_body.splitlines():
        m = re.match(r"\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", raw)
        if m:
            env_vars.append(f"{m.group(1)}={m.group(2).strip()}")
    return post_body, env_vars


# ---------------------------------------------------------------------------
# Dockerfile generation
# ---------------------------------------------------------------------------

def _strip_leading_debian_frontend(post_body: str) -> str:
    return re.sub(
        r"^\s*export\s+DEBIAN_FRONTEND=noninteractive\s*\n?",
        "",
        post_body,
        count=1,
    )


def generate_dockerfile(
    base_image: str,
    *,
    intricate: bool,
    intricate_base_image_tag: str | None,
    extra_env: list[str],
    fixture_copies: list[tuple[str, str]],
    has_base_install: bool,
    has_post_install: bool,
) -> str:
    """Assemble the Harbor environment/Dockerfile.

    Ordering mirrors the Apptainer build: base layer → %files → task %post.
    """
    from_image = (
        intricate_base_image_tag
        if (intricate and intricate_base_image_tag)
        else base_image
    )

    lines = [f"FROM {from_image}", "", "ENV DEBIAN_FRONTEND=noninteractive"]
    for kv in extra_env:
        lines.append(f"ENV {kv}")
    lines.append("")

    # 1. Intricate base layer (only when we are NOT using a prebuilt base image).
    if intricate and not intricate_base_image_tag and has_base_install:
        lines.append("COPY base_install.sh /tmp/base_install.sh")
        lines.append("RUN bash /tmp/base_install.sh && rm /tmp/base_install.sh")
        lines.append("")

    # 2. Fixtures from the %files section, staged under _fixtures/ in the build
    #    context. Copied before the task %post so the post step can reference them.
    #    NOTE: use the shell (whitespace-delimited) COPY form, NOT the JSON/exec
    #    array form. Daytona's Dockerfile build-context parser does not handle the
    #    array form and mis-reads the source path (trailing comma), which breaks
    #    the build. Our fixture paths contain no spaces, so shell form is safe.
    if fixture_copies:
        for ctx_path, dest in fixture_copies:
            lines.append(f"COPY {ctx_path} {dest}")
        lines.append("")

    # 3. Task-specific %post.
    if has_post_install:
        lines.append("COPY post_install.sh /tmp/post_install.sh")
        lines.append("RUN bash /tmp/post_install.sh && rm /tmp/post_install.sh")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# task.toml generation
# ---------------------------------------------------------------------------

def _escape_toml(s: str) -> str:
    """Escape special characters for TOML double-quoted strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def generate_task_toml(task_meta: dict, *, source: str) -> str:
    """Generate a task.toml from task.json metadata."""
    name = task_meta.get("name", "unknown")
    scenario = task_meta.get("scenario", "")
    domain = task_meta.get("domain", "")
    skill_type = task_meta.get("skill_type", "")
    primitive_skills = task_meta.get("primitive_skills", [])
    task_complexity = task_meta.get("task_complexity", "")
    command_complexity = task_meta.get("command_complexity", "")
    language = task_meta.get("language", "")
    base_image = task_meta.get("base_image", "")

    skills_toml = ", ".join(f'"{_escape_toml(s)}"' for s in primitive_skills)

    return textwrap.dedent(f"""\
        schema_version = "1.1"

        [task]
        name = "{ORG_NAME}/{name}"
        description = "{_escape_toml(scenario)}"

        [metadata]
        source = "{_escape_toml(source)}"
        base_image = "{_escape_toml(base_image)}"
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


TEST_SH = textwrap.dedent("""\
    #!/bin/bash
    set -e

    mkdir -p /logs/verifier

    # Some verifiers import `requests` (allow-listed for multi_protocol
    # verifier kinds). It ships in base_intricate.sif at solve time but is not
    # guaranteed in every published prebuilt image, so ensure it is importable
    # before grading. No-op when already present; falls through harmlessly
    # (the verifier then fails as before) if the install cannot run.
    if ! python3 -c "import requests" >/dev/null 2>&1; then
        pip3 install --quiet --no-input requests >/dev/null 2>&1 || true
    fi

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


# ---------------------------------------------------------------------------
# Fixture staging
# ---------------------------------------------------------------------------

def _resolve_fixture_src(src_token: str, src_dir: Path) -> Path | None:
    """Resolve a ``%files`` source path to an existing file/dir.

    Prefer the task's own (possibly symlinked) ``fixtures/`` copy so the
    conversion is robust to the original corpus being moved; fall back to the
    absolute path recorded in the def.
    """
    if "/fixtures/" in src_token:
        rel = src_token.split("/fixtures/", 1)[1]
        local = src_dir / "fixtures" / rel
        if local.exists():
            return local
    p = Path(src_token)
    return p if p.exists() else None


def _stage_fixtures(
    file_pairs: list[tuple[str, str]],
    src_dir: Path,
    env_dir: Path,
) -> tuple[list[tuple[str, str]], int]:
    """Copy fixtures into ``env_dir/_fixtures`` and return Dockerfile COPY pairs.

    Returns ``(copies, missing)`` where ``copies`` is a list of
    ``(context_relative_src, container_dest)`` and ``missing`` counts
    unresolvable sources.
    """
    copies: list[tuple[str, str]] = []
    missing = 0
    stage_root = env_dir / "_fixtures"

    for src_token, dest in file_pairs:
        resolved = _resolve_fixture_src(src_token, src_dir)
        if resolved is None:
            missing += 1
            logger.warning("Missing fixture source %s (task %s)", src_token, src_dir.name)
            continue

        rel = dest.lstrip("/")
        staged = stage_root / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        if resolved.is_dir():
            if staged.exists():
                shutil.rmtree(staged)
            shutil.copytree(resolved, staged)
        else:
            shutil.copy2(resolved, staged)  # copy2 preserves mode bits

        copies.append((f"_fixtures/{rel}", dest))

    return copies, missing


# ---------------------------------------------------------------------------
# Per-task conversion
# ---------------------------------------------------------------------------

def convert_task(
    src_dir: Path,
    dst_dir: Path,
    *,
    intricate_base_post: str,
    intricate_env: list[str],
    intricate_base_image_tag: str | None,
) -> str | None:
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

    intricate = task_meta.get("base_image") == "intricate"
    source = "v2" if intricate else "legacy"

    out_task = dst_dir / src_dir.name
    out_task.mkdir(parents=True, exist_ok=True)

    # 1. instruction.md
    (out_task / "instruction.md").write_text(description)

    # 2. task.toml
    (out_task / "task.toml").write_text(generate_task_toml(task_meta, source=source))

    # 3. environment/
    env_dir = out_task / "environment"
    env_dir.mkdir(exist_ok=True)

    base_image, post_body = parse_container_def(container_def_path)
    cleaned_post = _strip_leading_debian_frontend(post_body)
    has_post = bool(cleaned_post.strip())
    if has_post:
        (env_dir / "post_install.sh").write_text(cleaned_post)

    has_base_install = False
    extra_env: list[str] = []
    if intricate and not intricate_base_image_tag:
        base_post = _strip_leading_debian_frontend(intricate_base_post)
        if base_post.strip():
            (env_dir / "base_install.sh").write_text(base_post)
            has_base_install = True
        extra_env = intricate_env

    fixture_copies: list[tuple[str, str]] = []
    file_pairs = parse_files_section(container_def_path)
    if file_pairs:
        fixture_copies, _missing = _stage_fixtures(file_pairs, src_dir, env_dir)

    (env_dir / "Dockerfile").write_text(
        generate_dockerfile(
            base_image,
            intricate=intricate,
            intricate_base_image_tag=intricate_base_image_tag,
            extra_env=extra_env,
            fixture_copies=fixture_copies,
            has_base_install=has_base_install,
            has_post_install=has_post,
        )
    )

    # 4. tests/
    tests_dir = out_task / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text(TEST_SH)
    shutil.copy2(test_final_path, tests_dir / "test_final_state.py")

    return task_name


def _convert_task_safe(src_dir, dst_dir, kwargs):
    """ProcessPool entrypoint: never let one bad task kill the pool."""
    try:
        return convert_task(src_dir, dst_dir, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to convert %s: %s", src_dir.name, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RL task directories to Harbor dataset format"
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing task_* subdirectories (symlinks ok)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory for Harbor dataset (default: <src>_harbor)",
    )
    parser.add_argument(
        "--base-intricate-def",
        type=Path,
        default=DEFAULT_BASE_INTRICATE_DEF,
        help="Path to base_intricate.def (its %%post is inlined for intricate tasks)",
    )
    parser.add_argument(
        "--intricate-base-image",
        type=str,
        default=None,
        help=(
            "If set, intricate tasks build FROM this image tag instead of inlining "
            "the base_intricate layer (recommended for publishing: build/push the "
            "base once, e.g. tmax/base-intricate:1.0)."
        ),
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

    intricate_base_post = ""
    intricate_env: list[str] = []
    if args.intricate_base_image:
        logger.info("Intricate tasks will build FROM %s", args.intricate_base_image)
    elif args.base_intricate_def.exists():
        intricate_base_post, intricate_env = load_intricate_base(args.base_intricate_def)
        logger.info(
            "Loaded intricate base layer from %s (%d env vars)",
            args.base_intricate_def,
            len(intricate_env),
        )
    else:
        logger.warning(
            "base_intricate.def not found at %s and no --intricate-base-image given; "
            "intricate tasks may produce images missing heavy deps.",
            args.base_intricate_def,
        )

    task_dirs = sorted(
        p for p in src.iterdir() if p.is_dir() and p.name.startswith("task_")
    )
    logger.info("Found %d task directories in %s", len(task_dirs), src)

    dst.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "intricate_base_post": intricate_base_post,
        "intricate_env": intricate_env,
        "intricate_base_image_tag": args.intricate_base_image,
    }

    converted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_convert_task_safe, td, dst, kwargs): td for td in task_dirs
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                converted += 1
            else:
                skipped += 1
                logger.debug("Skipped %s", futures[future].name)

            total = converted + skipped
            if total % 1000 == 0:
                logger.info(
                    "Progress: %d/%d (converted=%d, skipped=%d)",
                    total,
                    len(task_dirs),
                    converted,
                    skipped,
                )

    logger.info(
        "Done. Converted %d tasks, skipped %d. Output: %s",
        converted,
        skipped,
        dst,
    )


if __name__ == "__main__":
    main()
