"""Adapter for ``hamishivi/agent-task-r2e-gym`` (R2E Gym, 8,101 tasks).

R2E Gym is a SWE-bench-style benchmark: each task is a real bug from a popular
Python repo (sympy/pandas/matplotlib/numpy/pillow/orange3/moto/aiohttp/tornado/
scrapy/pyramid/datalad/coveragepy), shipped as a per-task Docker image that
bakes the repo at the buggy commit + a ``/r2e_tests`` test directory. The agent
is dropped in and asked to patch ``/testbed`` so the gold pytest suite passes.

Source layout
-------------
The dataset ships **two** assets on the HF Hub:

1. ``data/train-*.parquet``  -- 8,101 conversation-formatted rows. These include
   R2E's own system prompt + user message, plus per-row ``env_config`` /
   ``ground_truth`` / ``source`` metadata. We deliberately **ignore** the chat
   messages and use the per-task ``instruction.md`` from the tarball instead
   so the R2E system prompt does not collide with our vanillux harness's
   mini-swe-agent system prompt. (Consistent with every other adapter:
   the dataset's framing/persona belongs to the user-side task description;
   the harness owns its own system prompt.)

2. ``task-data.tar.gz``  -- 14 MB tarball with the actual per-task payload:

       <repo>__<commit_hash_short>/
         image.txt                      # "namanjain12/<repo>_final:<full_commit_hash>"
         instruction.md                 # the human-readable problem statement
         tests/
           expected_output.json         # gold pytest statuses (PASSED/FAILED/...)
           score.py                     # writes /logs/verifier/reward.txt
           test.sh                      # pytest /r2e_tests | tee | score.py

   ``test.sh`` and ``score.py`` are dataset-wide constants (identical md5
   across all 8,101 tasks); only ``expected_output.json`` varies.

Verifier contract
-----------------
``test.sh`` (1) activates ``/testbed/.venv`` if present, (2) runs ``pytest
/r2e_tests --junit-xml=/tmp/report.xml``, (3) invokes ``score.py`` which diffs
the JUnit report against ``expected_output.json`` and writes ``1.0`` (all gold
statuses match) or ``0.0`` to ``/logs/verifier/reward.txt``.

Our pytest wrapper (``test_final_state.py``) just shells out to
``bash /tests/test.sh`` and asserts ``reward == 1.0``, matching the
``test.sh``-as-shell-verifier shape we already use for OT-Agent-v1-RL.

Wrinkles vs. the other adapters
-------------------------------
1. **Per-task Docker Hub image.** Like TerminalTraj, every task has its own
   image (``namanjain12/<repo>_final:<commit>``, 8,101 distinct images,
   typically 400-800 MB compressed). We cannot prebuild a shared base SIF;
   the solve script's pre-build phase pulls them via ``apptainer build``.

2. **Agent CWD is ``/testbed``, not ``/home/user``.** The harness drops the
   agent at ``/home/user`` (writable tmpfs bound over the image's
   ``/home/user``); the instruction text itself tells the agent to
   ``cd /testbed && source .venv/bin/activate``. No adapter-side cwd
   override is required.

3. **Some base images may lack a system-level pytest.** The harness invokes
   our ``test_final_state.py`` via ``pytest pytest_final_state.py`` from the
   container's default PATH, *not* from inside ``/testbed/.venv``. The R2E
   images are Python 3.x based and typically ship pip3, but we still inject
   TerminalTraj's robust pytest-bootstrap ``%post`` (pip3 -> apt/dnf/yum/apk
   -> get-pip.py) defensively so the verifier wrapper always finds a
   pytest on PATH.

CLI:

    python -m rl_data.comparison.adapters.r2e_gym \\
        --dst rl_data/output/tasks_r2e_gym

Options mirror the other adapters (``--limit``, ``--workers``,
``--skip-download``, ``--revision``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

from rl_data.comparison.adapters import (
    Adapter,
    _PLACEHOLDER_INITIAL_STATE,
    register_adapter,
)
from rl_data.comparison.adapters.terminaltraj import _PYTEST_BOOTSTRAP_POST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HF_REPO_ID = "hamishivi/agent-task-r2e-gym"
TARBALL_NAME = "task-data.tar.gz"
# task-data.tar.gz unpacks task directories at its top level (no wrapping
# parent dir). We extract into ``cache_dir/extracted`` and treat that
# directly as the snapshot dir.
EXTRACT_DIRNAME = "extracted"


# ---------------------------------------------------------------------------
# Pytest verifier wrapper.
#
# Shells out to /tests/test.sh (dataset-supplied), then checks the reward
# file score.py writes. We surface tail-of-test.sh-output in the assertion
# message so failed rollouts have something useful to grep in the trajectory.
# ---------------------------------------------------------------------------
_TEST_FINAL_WRAPPER = r'''"""Pytest wrapper around R2E Gym's shell verifier.

R2E Gym ships ``/tests/test.sh`` which (1) runs the gold pytest suite at
``/r2e_tests`` against ``/testbed`` and writes a JUnit XML, then (2) invokes
``/tests/score.py`` which compares actual vs. expected pytest statuses and
writes ``1.0`` (all match) or ``0.0`` to ``/logs/verifier/reward.txt``.

We expose that outcome to our pytest-based harness so no changes to
generate_solutions.py are required.
"""
import subprocess
from pathlib import Path


def test_r2e_gym_verifier():
    proc = subprocess.run(
        ["bash", "/tests/test.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    reward_path = Path("/logs/verifier/reward.txt")
    if not reward_path.exists():
        tail = (proc.stdout or "")[-2000:]
        err = (proc.stderr or "").strip()[-1000:]
        raise AssertionError(
            "score.py did not produce /logs/verifier/reward.txt. "
            f"test.sh exit={proc.returncode}. "
            f"stdout tail:\n{tail}\nstderr tail:\n{err}"
        )
    raw = reward_path.read_text().strip()
    try:
        reward = float(raw)
    except ValueError:
        raise AssertionError(f"unparseable reward.txt: {raw!r}")
    assert reward >= 1.0, (
        f"R2E Gym verifier reported reward={reward} (expected 1.0). "
        f"test.sh stdout tail:\n" + (proc.stdout or "")[-2000:]
    )
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Strip the upstream "use the submit tool" trailer. Our vanillux harness has
# its own submit semantics that mini-swe-agent's prompts already explain in
# the system prompt; the dataset's wording ("use the submit tool to verify
# your fix") is unique to R2E Gym's harness and would confuse the agent into
# looking for a tool we do not expose.
_INSTRUCTION_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "When you are done, use the submit tool to verify your fix.",
        "When you are done, finalize per the harness instructions in your system prompt.",
    ),
)


def _patch_instruction(text: str) -> str:
    for old, new in _INSTRUCTION_REWRITES:
        text = text.replace(old, new)
    return text


# The upstream test.sh runs ``pytest /r2e_tests``. With recent pytest (8.x),
# ``pytest`` no longer inserts the current working directory into sys.path,
# so the bundled tests (which do ``import numpy``, ``import sympy``, etc.)
# cannot find the project package sitting at ``/testbed``. The repository's
# own ``/testbed/run_tests.sh`` script invokes pytest as
# ``.venv/bin/python -m pytest`` instead, which DOES put cwd on sys.path
# and is the contract every R2E Gym task is built against. We rewrite the
# bundled test.sh to follow the same idiom, otherwise every R2E task
# reports a collection-time ImportError and the harness records a 0.0
# reward regardless of whether the agent actually fixed the bug. (Without
# this rewrite, R2E Gym's solve rate was ~5% on gemini-3-flash-preview --
# essentially the noise floor of tasks where the test happened not to
# import a project module.)
_TEST_SH_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "pytest /r2e_tests --junit-xml=/tmp/report.xml --tb=short -q",
        "python -m pytest /r2e_tests --junit-xml=/tmp/report.xml --tb=short -q",
    ),
)


def _patch_test_sh(text: str) -> str:
    for old, new in _TEST_SH_REWRITES:
        text = text.replace(old, new)
    return text


def _build_container_def(
    *,
    base_image: str,
    task_dir: Path,
) -> str:
    """Build an Apptainer container.def for one R2E Gym task.

    Bootstraps from the per-task Docker Hub image and stages the task's
    ``test.sh``, ``score.py``, ``expected_output.json`` into ``/tests/``
    (where score.py reads them via absolute paths). Per-task payload is
    addressed by absolute path rooted at ``task_dir`` so ``apptainer build``
    works regardless of invocation CWD.

    The pytest bootstrap (shared with TerminalTraj) guarantees a global
    ``pytest`` is on PATH so the harness's
    ``pytest pytest_final_state.py`` invocation succeeds even on R2E
    images that ship pytest only inside ``/testbed/.venv``.
    """
    test_sh_abs = (task_dir / "tests" / "test.sh").resolve()
    score_py_abs = (task_dir / "tests" / "score.py").resolve()
    expected_abs = (task_dir / "tests" / "expected_output.json").resolve()

    # Pytest must be available on the system PATH (outside /testbed/.venv)
    # so the harness's verifier wrapper can be executed. The bootstrap block
    # is fail-soft (toggles set +e/-e internally), so we do not bracket it
    # with set -e of our own.
    lines: list[str] = [
        "Bootstrap: docker",
        f"From: {base_image}",
        "",
        "%post",
        _PYTEST_BOOTSTRAP_POST.rstrip(),
        "    mkdir -p /tests /logs/verifier /home/user",
        "    chmod 755 /home/user",
        "",
        "%labels",
        "    Author r2e-gym-adapter",
        f"    BaseImage {base_image}",
        '    Description "R2E Gym task layer (per-task /tests payload + pytest bootstrap)"',
        "",
        "%files",
        f"    {test_sh_abs} /tests/test.sh",
        f"    {score_py_abs} /tests/score.py",
        f"    {expected_abs} /tests/expected_output.json",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_task_name(src_name: str) -> tuple[str, str]:
    """Split ``<repo>__<commit_short>`` into (repo, commit_short).

    Falls back to (``src_name``, "") if the underscore-pair separator is
    absent (e.g. unexpected/legacy slugs)."""
    parts = src_name.split("__", 1)
    if len(parts) != 2:
        return src_name, ""
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class R2eGymAdapter(Adapter):
    """``hamishivi/agent-task-r2e-gym`` (per-commit Docker images + shell verifier)."""

    name = "r2e_gym"
    hf_repo_id = HF_REPO_ID
    default_dst = "rl_data/output/tasks_r2e_gym"

    # -- Fetch -------------------------------------------------------------
    def fetch(self, cache_dir: Path, *, revision: Optional[str] = None,
              skip_download: bool = False) -> Path:
        """Download + extract ``task-data.tar.gz`` into ``cache_dir`` and
        return the directory holding per-task dirs.

        Idempotent: ``hf_hub_download`` no-ops on existing files. We
        re-extract whenever the destination is missing or empty (no
        per-task validation here, unlike OT-Agent-v1-RL — R2E Gym only
        ships static text payload, no binary seeds, so corruption is
        unlikely and a wipe-and-redo is cheap).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        extracted = cache_dir / EXTRACT_DIRNAME

        if skip_download:
            if not extracted.is_dir() or not any(extracted.iterdir()):
                raise RuntimeError(
                    f"--skip-download requested but no extracted tasks at "
                    f"{extracted}. Run once without --skip-download first."
                )
            logger.info("Skipping download; using cached extract %s", extracted)
            return extracted

        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s/%s ...", self.hf_repo_id, TARBALL_NAME)
        tar_path = Path(hf_hub_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            filename=TARBALL_NAME,
            revision=revision,
            cache_dir=str(cache_dir / "_hf_cache"),
        ))

        extracted.mkdir(parents=True, exist_ok=True)
        # Skip extraction if the destination already has tasks AND looks
        # healthy. The tarball's top-level entries are the task dirs
        # themselves (no wrapping parent), so any non-hidden subdir of
        # ``extracted`` is a candidate task.
        already = [
            p for p in extracted.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if already:
            logger.info(
                "Found %d task dirs already in %s; skipping extraction. "
                "Remove the dir to force a re-extract.",
                len(already), extracted,
            )
            return extracted

        logger.info("Extracting %s into %s ...", tar_path, extracted)
        with tarfile.open(tar_path, "r:gz") as tf:
            try:
                tf.extractall(extracted, filter="data")
            except TypeError:
                tf.extractall(extracted)
        return extracted

    # -- Iterate source task dirs ------------------------------------------
    # Default ``list_source_tasks`` (sorted direct children that are not
    # hidden) matches the tarball's flat layout.

    # -- Convert one task --------------------------------------------------
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:
        image_txt = src / "image.txt"
        instruction_md = src / "instruction.md"
        test_sh = src / "tests" / "test.sh"
        score_py = src / "tests" / "score.py"
        expected_json = src / "tests" / "expected_output.json"

        if not (image_txt.exists() and instruction_md.exists()
                and test_sh.exists() and score_py.exists()
                and expected_json.exists()):
            logger.warning("convert_one: %s missing expected files; skipping", src.name)
            return None

        try:
            base_image = image_txt.read_text().strip()
        except OSError as exc:
            logger.warning("convert_one(%s): cannot read image.txt: %s",
                           src.name, exc)
            return None
        if not base_image:
            logger.warning("convert_one(%s): empty image.txt; skipping", src.name)
            return None

        try:
            description = _patch_instruction(instruction_md.read_text())
        except OSError as exc:
            logger.warning("convert_one(%s): cannot read instruction.md: %s",
                           src.name, exc)
            description = ""

        repo, commit = _parse_task_name(src.name)

        # Prefix with ``r2e_`` to avoid task-name collisions with any other
        # adapter (each adapter uses its own short prefix; otrl_, et_, tg_,
        # tt_, ...).
        task_name = "r2e_" + re.sub(r"\s+", "_", src.name)
        out = dst_root / task_name
        out.mkdir(parents=True, exist_ok=True)

        enriched: Dict[str, Any] = {
            "name": task_name,
            # Native taxonomy left empty; downstream classifier fills
            # classified_* fields. R2E Gym does not ship category/difficulty
            # metadata of its own (the repo name itself is the closest
            # proxy, preserved below as ``r2e_repo``).
            "domain": "unknown",
            "skill_type": "unknown",
            "primitive_skills": [],
            "task_complexity": "unknown",
            "command_complexity": "unknown",
            "scenario": "",
            "language": "any (model's choice)",
            "description": description,
            "truth": "",
            # Dataset-native metadata preserved for the appendix panels.
            "r2e_repo": repo,
            "r2e_commit": commit,
            "r2e_image": base_image,
            # Provenance.
            "source": "r2e_gym",
            "source_repo": HF_REPO_ID,
            "source_slug": src.name,
        }
        (out / "task.json").write_text(json.dumps(enriched, indent=2))

        # Materialize a stable copy of the per-task verifier payload
        # alongside the def so the absolute paths in %files do not depend
        # on the ephemeral HF cache (matches the OT-Agent-v1-RL pattern).
        # ``test.sh`` is rewritten to use ``python -m pytest`` (see
        # ``_TEST_SH_REWRITES`` for the rationale); the others are copied
        # verbatim.
        for sub in ("tests/score.py", "tests/expected_output.json"):
            s = src / sub
            d = out / sub
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(s, d)
            except OSError as exc:
                logger.warning("convert_one(%s): failed to stage %s: %s",
                               task_name, sub, exc)
                return None
        try:
            test_sh_text = _patch_test_sh((src / "tests" / "test.sh").read_text())
        except OSError as exc:
            logger.warning("convert_one(%s): failed to read test.sh: %s",
                           task_name, exc)
            return None
        test_sh_dst = out / "tests" / "test.sh"
        test_sh_dst.parent.mkdir(parents=True, exist_ok=True)
        test_sh_dst.write_text(test_sh_text)
        # Preserve the executable bit (pytest invocations go through bash
        # so it isn't strictly needed, but matches the upstream layout).
        test_sh_dst.chmod(0o755)

        (out / "container.def").write_text(_build_container_def(
            base_image=base_image,
            task_dir=out,
        ))

        (out / "test_final_state.py").write_text(_TEST_FINAL_WRAPPER)
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

        return task_name


register_adapter(R2eGymAdapter())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_r2e_gym_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(R2eGymAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse the already-extracted tarball in cache-dir "
                         "instead of re-downloading from the HF Hub.")
    ap.add_argument("--revision", type=str, default=None,
                    help="Dataset revision (commit SHA or tag) to pin.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = R2eGymAdapter()

    snapshot = adapter.fetch(
        args.cache_dir.resolve(),
        revision=args.revision,
        skip_download=args.skip_download,
    )

    converted, skipped = adapter.convert_all(
        snapshot, args.dst.resolve(),
        limit=args.limit, workers=args.workers,
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
