"""Generate lightweight per-task Apptainer defs on top of pre-built domain base images."""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rl_data import chat_completion_batch, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base image registry: domain -> .sif path
# ---------------------------------------------------------------------------

CONTAINERS_DIR = Path(__file__).resolve().parent.parent / "containers"

BASE_IMAGES: dict[str, Path] = {
    "security":              CONTAINERS_DIR / "base_security.sif",
    "software_engineering":  CONTAINERS_DIR / "base_software_engineering.sif",
    "file_operations":       CONTAINERS_DIR / "base_file_operations.sif",
    "data_querying":         CONTAINERS_DIR / "base_data_querying.sif",
    "data_science":          CONTAINERS_DIR / "base_data_science.sif",
    "debugging":             CONTAINERS_DIR / "base_debugging.sif",
    "scientific_computing":  CONTAINERS_DIR / "base_scientific_computing.sif",
    "data_processing":       CONTAINERS_DIR / "base_data_processing.sif",
    "system_administration": CONTAINERS_DIR / "base_system_administration.sif",
}

DEFAULT_BASE = CONTAINERS_DIR / "base_software_engineering.sif"

DOMAIN_LIST = list(BASE_IMAGES.keys())


def _resolve_base(domain: str, base_sifs_dir: Optional[Path] = None) -> Path:
    """Return the path to the base SIF for a domain.

    If *base_sifs_dir* is given, look there; otherwise use ``CONTAINERS_DIR``.
    """
    root = Path(base_sifs_dir) if base_sifs_dir else CONTAINERS_DIR
    path = root / f"base_{domain}.sif"
    if path.exists():
        return path
    fallback = root / "base_software_engineering.sif"
    return fallback if fallback.exists() else DEFAULT_BASE


# ---------------------------------------------------------------------------
# Def → delta parser: extract task-specific setup from a full .def
# ---------------------------------------------------------------------------

_PREAMBLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*Bootstrap\s*:", re.IGNORECASE),
    re.compile(r"^\s*From\s*:", re.IGNORECASE),
    re.compile(r"^\s*%post\b"),
    re.compile(r"^\s*export\s+DEBIAN_FRONTEND\s*="),
    re.compile(r"^\s*apt-get\s+update"),
    re.compile(r"^\s*pip3?\s+install\s+pytest\s*$"),
    re.compile(r"^\s*useradd\b"),
    re.compile(r"^\s*chmod\s+.*(/home/user|777|755)"),
    re.compile(r"^\s*%environment\b"),
    re.compile(r"^\s*export\s+(LC_ALL|LANG)\s*="),
]

_APT_GET_RE = re.compile(r"^\s*apt-get\s+(install|update)")
_PIP_INSTALL_RE = re.compile(r"^\s*pip3?\s+install\s+(.+)")


def _is_preamble_line(line: str) -> bool:
    return any(p.search(line) for p in _PREAMBLE_PATTERNS)


def parse_def_to_delta(def_text: str, domain_hint: Optional[str] = None) -> tuple[str, str]:
    """Split a full ``.def`` into ``(base_domain, setup_script_body)``.

    *base_domain* is a key of ``BASE_IMAGES`` (e.g. ``"debugging"``).
    *setup_script_body* is a shell script containing only the task-specific
    ``%post`` commands (file creation, extra pip installs, data generation, etc.)
    with the standard preamble stripped.

    Falls back to *domain_hint* (from ``task.json``) or ``"software_engineering"``.
    """
    lines = def_text.split("\n")

    delta_lines: list[str] = []
    in_post = False
    in_heredoc = False
    heredoc_marker = ""

    for line in lines:
        if not in_post:
            if re.match(r"^\s*%post\b", line):
                in_post = True
            continue

        if re.match(r"^\s*%\w+", line) and not in_heredoc:
            break

        if in_heredoc:
            delta_lines.append(line)
            if line.strip() == heredoc_marker:
                in_heredoc = False
            continue

        heredoc_match = re.search(r"<<\s*['\"]?(\w+)['\"]?", line)
        if heredoc_match and not _is_preamble_line(line):
            in_heredoc = True
            heredoc_marker = heredoc_match.group(1)
            delta_lines.append(line)
            continue

        if _is_preamble_line(line) or _APT_GET_RE.match(line) or _PIP_INSTALL_RE.match(line):
            continue

        stripped = line.rstrip()
        if stripped:
            delta_lines.append(stripped)

    while delta_lines and not delta_lines[0].strip():
        delta_lines.pop(0)
    while delta_lines and not delta_lines[-1].strip():
        delta_lines.pop()

    setup_body = "\n".join(delta_lines)
    base_domain = domain_hint if domain_hint and domain_hint in BASE_IMAGES else "software_engineering"

    # Rewrite /tmp/ references to /home/user/.setup_tmp/ so temp files land on the
    # writable bind mount instead of fuse-overlayfs (which can EINVAL on file creation).
    setup_body = setup_body.replace("/tmp/", "/home/user/.setup_tmp/")

    header = "#!/bin/bash\nset -euo pipefail\nmkdir -p /home/user/.setup_tmp\n"
    footer = "\nrm -rf /home/user/.setup_tmp\n"
    return base_domain, header + setup_body + footer


def save_setup_artifacts(task_dir: Path, def_text: str, domain: str) -> Path:
    """Parse a ``.def`` and write ``setup.sh`` into *task_dir*.

    Returns the path to the written ``setup.sh``.
    """
    _base_domain, setup_body = parse_def_to_delta(def_text, domain_hint=domain)
    setup_path = task_dir / "setup.sh"
    setup_path.write_text(setup_body, encoding="utf-8")
    return setup_path


# ---------------------------------------------------------------------------
# Def sanitisation & transient-error detection
# ---------------------------------------------------------------------------

def _sanitize_def(def_text: str) -> str:
    """Strip ``set -e`` variants that cause spurious failures on benign errors."""
    return re.sub(
        r"^\s*set\s+-[euxo]+(?:\s+pipefail)?\s*$", "", def_text, flags=re.MULTILINE
    )


_TRANSIENT_PATTERNS = (
    "conveyor failed to get",
    "unexpected end of json input",
    "connection reset by peer",
    "tls handshake timeout",
    "could not resolve host",
    "temporary failure in name resolution",
    "index files failed to download",
    "failed to fetch",
    "hash sum mismatch",
)


def _is_transient_error(output: str) -> bool:
    low = output.lower()
    return any(p in low for p in _TRANSIENT_PATTERNS)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MSG = """\
You are an expert in Apptainer/Singularity container setup.

You will be given a task description, ground truth, and initial-state tests.
Your job is to write an Apptainer .def file that sets up the initial state
of the container so that an agent can be tested on the task.

IMPORTANT RULES:
- Always start the def file with exactly:
  Bootstrap: docker
  From: ubuntu:22.04
- In the %post section:
  1. Start with: export DEBIAN_FRONTEND=noninteractive
  2. Run: apt-get update && apt-get install -y python3 python3-pip
  3. Run: pip3 install pytest
  4. Install ONLY the additional system or Python packages the task needs.
     Keep package installs minimal. Prefer pip over apt when possible.
  5. Create files, directories, and data needed for the task.
  6. Create the user: useradd -m -s /bin/bash user || true
  7. End with: chmod -R 777 /home/user
- Do NOT include %test sections.
- Do NOT create output files that the agent should produce.
- The home path is /home/user.
- Do NOT override HOME in %environment.
- Do NOT use Apptainer build variables (no {{ }}).
- Do NOT use exotic package names. If you need awk, install gawk.
  The command 'tr' is part of coreutils, not a separate package."""

BASE_USER_TEMPLATE = """\
Write an Apptainer .def file for this task.

The task domain is: {domain}

Task description given to the agent:
{task_description}

Ground truth (for setting up initial state):
{truth}

Tests that will verify the initial container state:
{test_py}

Previous failures (may be empty):
{failures}

Respond with ONLY the Apptainer .def file. It must start with:
Bootstrap: docker
From: ubuntu:22.04

Keep the %post section focused: install only what's needed, create the
required files/directories/data, and ensure /home/user is writable."""


def build_and_test(
    def_template: str, test_py: str, *, build_retries: int = 2
) -> tuple[bool, str]:
    """Build an Apptainer image from a def and run initial-state tests.

    Transient network / OCI-pull errors are retried up to *build_retries* times.
    """
    import os

    tmp_base = os.environ.get("APPTAINER_TMPDIR", None)
    with tempfile.TemporaryDirectory(dir=tmp_base) as td:
        td_path = Path(td)

        def_path = td_path / "container.def"
        def_path.write_text(def_template)

        test_file = td_path / "test_initial_state.py"
        test_file.write_text(test_py)

        sif_path = td_path / "img.sif"
        last_err = ""
        for attempt in range(1 + build_retries):
            if sif_path.exists():
                sif_path.unlink()
            build_proc = subprocess.run(
                ["apptainer", "build", str(sif_path), str(def_path)],
                capture_output=True, text=True, timeout=300,
            )
            if build_proc.returncode == 0:
                break
            last_err = (build_proc.stderr or build_proc.stdout or "")[-500:]
            if attempt < build_retries and _is_transient_error(last_err):
                import time as _time
                delay = 2 ** (attempt + 1)
                logger.info("Transient build error (attempt %d), retrying in %ds…", attempt + 1, delay)
                _time.sleep(delay)
                continue
            break

        if build_proc.returncode:
            print(f"Apptainer build failed (rc={build_proc.returncode}): {last_err}")
            return False, f"Apptainer build failed: {last_err}"

        try:
            proc = subprocess.run(
                [
                    "apptainer", "exec",
                    "--fakeroot", "--userns", "--writable-tmpfs", "--cleanenv",
                    str(sif_path),
                    "pytest", "-q", str(test_file.name),
                ],
                cwd=td,
                capture_output=True, text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "Initial-state test timed out after 120s"
        finally:
            if sif_path.exists():
                sif_path.unlink()

        return proc.returncode == 0, proc.stdout + proc.stderr


def parse_def_template(def_template: str) -> str:
    """Extract and clean a .def file from LLM output."""
    cleaned = def_template.replace("\r\n", "\n").strip()

    fence_re = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(?P<code>[\s\S]*?)```", re.MULTILINE)
    match = fence_re.search(cleaned)
    if match:
        cleaned = match.group("code").strip()

    cleaned = textwrap.dedent(cleaned).strip()
    return cleaned


def iterate_def_template_batch(
    items: List[Tuple[str, str, str]],
    *,
    domains: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
    max_retries: int = 0,
    max_build_workers: int = 4,
    skip_indices: Optional[set] = None,
    on_round_complete: Optional[object] = None,
) -> List[Optional[str]]:
    """Batched def generation with retry loop and error feedback.

    Failed items are re-prompted with the build/test error so the LLM can
    self-correct.  Defs containing ``{{ }}`` build variables are caught
    before an expensive build attempt.

    Parameters
    ----------
    skip_indices:
        Indices to skip entirely (e.g. already processed in a previous run).
    on_round_complete:
        ``callback(round_idx, newly_succeeded)`` called after each round.
        *newly_succeeded* is ``Dict[int, str]`` mapping item index → def text
        for items that passed build+test in this round.
    """
    if domains is None:
        domains = ["software_engineering"] * len(items)

    results: List[Optional[str]] = [None] * len(items)
    error_msgs: List[Optional[str]] = [None] * len(items)
    pending = [i for i in range(len(items)) if not skip_indices or i not in skip_indices]

    if skip_indices:
        print(f"  Skipping {len(skip_indices)} already-completed items, {len(pending)} to process")

    def _try_build(idx: int, resp_obj) -> Tuple[int, Optional[str], Optional[str]]:
        """Return (original_index, def_text | None, error_msg | None)."""
        try:
            if resp_obj is None:
                return idx, None, "LLM returned no response"
            def_text = _sanitize_def(parse_def_template(
                resp_obj.choices[0].message.content))
            if re.search(r"\{\{.*?\}\}", def_text):
                return idx, None, "Def contains {{ }} build variables (forbidden)."
            _, _, test_py = items[idx]
            ok, output = build_and_test(def_text, test_py)
            return (idx, def_text, None) if ok else (idx, None, output)
        except Exception as exc:
            logger.warning("Def worker failed for item %d: %s", idx, exc)
            return idx, None, str(exc)

    total_succeeded = len(skip_indices) if skip_indices else 0

    for attempt in range(1 + max_retries):
        if not pending:
            break

        round_label = f"Round {attempt}/{max_retries}"
        print(f"\n{'─'*60}")
        print(f"  {round_label}: generating {len(pending)} defs via LLM...")

        messages: list[list[dict[str, str]]] = []
        for idx in pending:
            task_description, truth, test_py = items[idx]
            prompt = BASE_USER_TEMPLATE.format(
                domain=domains[idx].replace("_", " "),
                task_description=task_description,
                truth=truth,
                test_py=test_py,
                failures=error_msgs[idx] or "None yet",
            )
            messages.append([
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ])

        responses = chat_completion_batch(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_completions=1,
            max_concurrency=max_concurrency,
        )

        workers = min(max_build_workers, len(pending))
        print(f"  {round_label}: building + testing {len(pending)} defs with {workers} workers...")

        next_pending: list[int] = []
        newly_succeeded: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(_try_build, pending[pos], resp)
                for pos, resp in enumerate(responses)
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=round_label):
                idx, def_text, err = fut.result()
                if def_text is not None:
                    results[idx] = def_text
                    newly_succeeded[idx] = def_text
                else:
                    error_msgs[idx] = err
                    next_pending.append(idx)

        total_succeeded += len(newly_succeeded)
        print(
            f"  {round_label} done: {len(newly_succeeded)} succeeded, "
            f"{len(next_pending)} failed  "
            f"(cumulative: {total_succeeded}/{len(items)})"
        )

        if on_round_complete and newly_succeeded:
            on_round_complete(attempt, newly_succeeded)

        pending = next_pending

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = ap.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"

    test_py = initial_test_path.read_text()
    def_text = def_path.read_text()

    success, output = build_and_test(def_text, test_py)
    print(success)
    print(output)
