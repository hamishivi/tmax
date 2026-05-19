"""Dataset adapters — one per external baseline, plus the identity adapter
for our own dataset.

Each adapter is responsible for:

1. Fetching the raw data (usually via ``huggingface_hub.snapshot_download``).
2. Converting each task directory into our canonical Apptainer layout
   (``task.json`` + ``container.def`` + ``test_final_state.py`` +
   ``test_initial_state.py``) expected by
   :mod:`rl_data.generate_solutions`.
3. Preserving whatever dataset-native metadata exists in the enriched
   ``task.json`` under ``*_category`` / ``*_difficulty`` keys so it is
   available to the comparison script.

Adapters register themselves through :func:`register_adapter`; they expose a
``main()`` for CLI use:

    python -m rl_data.comparison.adapters.endless_terminals --limit 20
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level registry populated by ``register_adapter``.
ADAPTERS: Dict[str, "Adapter"] = {}


class Adapter:
    """Base class for a dataset adapter."""

    name: str                  # "endless_terminals", etc.
    hf_repo_id: Optional[str] = None
    default_dst: str           # e.g. "rl_data/output/tasks_endless_terminals"

    # -- Fetch -------------------------------------------------------------
    def fetch(self, cache_dir: Path, *, revision: Optional[str] = None) -> Path:
        """Download raw HF snapshot into ``cache_dir`` and return its path."""
        from huggingface_hub import snapshot_download

        if not self.hf_repo_id:
            raise RuntimeError(f"Adapter {self.name!r} has no hf_repo_id; override fetch()")
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s ...", self.hf_repo_id)
        return Path(snapshot_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            local_dir=str(cache_dir),
            revision=revision,
        ))

    # -- Convert one task --------------------------------------------------
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:
        """Override in subclasses."""
        raise NotImplementedError

    # -- Iterate source task dirs ------------------------------------------
    def list_source_tasks(self, snapshot_dir: Path) -> list[Path]:
        """Return the list of source task directories under ``snapshot_dir``.

        Default assumes directories are direct children of ``snapshot_dir``.
        """
        return sorted(
            p for p in snapshot_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
            and not p.name.startswith(".")
        )

    # -- Bulk convert ------------------------------------------------------
    def convert_all(
        self,
        snapshot_dir: Path,
        dst_root: Path,
        *,
        limit: int = 0,
        workers: int = 16,
    ) -> tuple[int, int]:
        """Convert every source task in parallel. Returns (converted, skipped)."""
        srcs = self.list_source_tasks(snapshot_dir)
        if limit and limit > 0:
            srcs = srcs[:limit]
        dst_root.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.convert_one, td, dst_root): td for td in srcs}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    logger.warning("convert_one failed on %s: %s", futs[fut].name, e)
                    r = None
                if r is not None:
                    converted += 1
                else:
                    skipped += 1
                total = converted + skipped
                if total % 500 == 0:
                    logger.info("Progress: %d/%d (converted=%d, skipped=%d)",
                                total, len(srcs), converted, skipped)
        return converted, skipped


def register_adapter(adapter: Adapter) -> Adapter:
    ADAPTERS[adapter.name] = adapter
    return adapter


# ---------------------------------------------------------------------------
# Shared Harbor-layout flatten helper
# ---------------------------------------------------------------------------

# Generic mapping for task.toml metadata field names.
# Both ET and OpenThoughts use this layout.
_PLACEHOLDER_INITIAL_STATE = (
    "def test_placeholder_initial_state():\n"
    "    assert True\n"
)


def _load_task_toml(toml_path: Path) -> Dict[str, Any]:
    try:
        data = tomllib.loads(toml_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    out: Dict[str, Any] = {}
    meta = data.get("metadata", {}) or {}
    for k in ("category", "difficulty", "tags", "author_name", "description"):
        if k in meta:
            out[k] = meta[k]
    return out


def _dockerfile_to_apptainer_def(
    dockerfile_text: str,
    build_context_dir: Optional[Path] = None,
) -> str:
    """Convert a minimal Dockerfile into an Apptainer container.def.

    Supports the common subset used by Harbor datasets: ``FROM``, ``RUN``,
    ``COPY``, ``ENV``, ``WORKDIR``.  Anything unrecognized is appended as
    shell commands in ``%post`` with a comment marker so builds still progress.

    Handles shell-style line continuations (``\\`` at end of line) by folding
    them into a single logical line before keyword dispatch. Without this, a
    multi-line ``RUN apt-get install -y \\ \\n  curl \\ \\n  python3`` block
    would produce one legitimate RUN line followed by garbage comment lines.

    ``build_context_dir`` should be the absolute path to the (already-
    flattened) task directory that holds the payload files referenced by
    ``COPY`` directives. When provided, ``COPY src dst`` turns into an
    Apptainer ``%files`` entry whose host side is
    ``<build_context_dir>/<src>`` -- absolute, so ``apptainer build`` can
    resolve it regardless of the caller's CWD. When ``None``, ``COPY`` lines
    degrade to a comment (old behavior) for call sites that don't ship payload.
    """
    # Fold `\<newline><indent>` into a single space so RUN blocks stay intact.
    folded = re.sub(r"\\\s*\n\s*", " ", dockerfile_text)

    base = "ubuntu:22.04"
    post_lines: list[str] = []
    files_lines: list[str] = []
    env_lines: list[str] = []

    def _resolve_copy_src(raw_src: str) -> Optional[Path]:
        """Map a Dockerfile COPY source token to an absolute host path.

        Returns None if we can't safely materialize this (e.g. multi-stage
        ``--from=builder`` which we don't emulate, or a path that escapes
        the build context).
        """
        if build_context_dir is None:
            return None
        # Strip leading ``./``. Reject absolute and parent-traversing paths.
        t = raw_src.lstrip()
        if t.startswith("./"):
            t = t[2:]
        if t.startswith("/") or ".." in Path(t).parts:
            return None
        candidate = build_context_dir / t
        return candidate if candidate.exists() else None

    for raw in folded.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.split(None, 1)
        if not upper:
            continue
        kw = upper[0].upper()
        rest = upper[1] if len(upper) > 1 else ""

        if kw == "FROM":
            base = rest.split(" as ")[0].split(" AS ")[0].strip()
        elif kw == "RUN":
            # Collapse continuation; keep multiline shell blocks intact.
            post_lines.append(f"    {rest}")
        elif kw == "COPY":
            # `COPY --from=builder ...` (multi-stage) or other flag forms are
            # not supported; emit a comment so we don't silently pretend.
            if rest.lstrip().startswith("--"):
                post_lines.append(f"    # COPY {rest}  (skipped; multi-stage/flagged COPY)")
                continue
            # Support `COPY src1 src2 ... dst` (last token is the destination).
            toks = rest.split()
            if len(toks) < 2:
                post_lines.append(f"    # COPY {rest}  (skipped; malformed)")
                continue
            dst = toks[-1]
            srcs = toks[:-1]
            resolved: list[tuple[Path, str]] = []
            failed: list[str] = []
            for s in srcs:
                host = _resolve_copy_src(s)
                if host is None:
                    failed.append(s)
                else:
                    resolved.append((host, s))
            if failed:
                post_lines.append(f"    # COPY {rest}  (skipped; sources missing: {failed})")
                continue
            # Multi-source COPY implies dst is a directory. Apptainer %files
            # honours that with a trailing slash; add one defensively.
            dst_display = dst if (len(resolved) == 1 and not dst.endswith("/")) else (
                dst if dst.endswith("/") else dst + "/"
            )
            for host, _ in resolved:
                files_lines.append(f"    {host} {dst_display}")
            # Make sure the destination directory exists (apptainer %files
            # does not mkdir intermediate paths for file destinations).
            dst_parent = dst_display.rsplit("/", 1)[0] or "/"
            post_lines.append(f"    mkdir -p {dst_parent}")
        elif kw == "ENV":
            env_lines.append(f"    export {rest.replace(' ', '=', 1)}")
        elif kw == "WORKDIR":
            post_lines.append(f"    mkdir -p {rest} && cd {rest}")
        else:
            post_lines.append(f"    # ({kw}) {rest}")

    hdr = ["Bootstrap: docker", f"From: {base}", ""]
    body: list[str] = ["%post"]
    body += env_lines
    body += post_lines
    body += ["    mkdir -p /home/user", "    chmod 755 /home/user"]
    out = "\n".join(hdr + body) + "\n"
    if files_lines:
        out += "\n%files\n" + "\n".join(files_lines) + "\n"
    return out


def flatten_harbor_task(
    src: Path,
    dst_root: Path,
    *,
    source_name: str,
    source_repo: str,
    prefix: str = "",
    extra_task_json: Optional[Callable[[Dict[str, Any], Path], Dict[str, Any]]] = None,
    task_name_override: Optional[str] = None,
    test_final_candidates: tuple[str, ...] = ("test_final_state.py",),
    copy_aux_test_files: bool = False,
) -> Optional[str]:
    """Flatten a Harbor-layout task dir (environment/+tests/+instruction.md)
    into our canonical layout under ``dst_root``.

    ``test_final_candidates`` lists filenames to try (in order) inside the
    ``tests/`` source directory; the first one that exists is copied to
    ``dst/test_final_state.py``. OpenThoughts-TB ships ``test_outputs.py``
    rather than ``test_final_state.py``, for example.

    When ``copy_aux_test_files`` is True, all other (non-main, non-shell) files
    from ``tests/`` are copied verbatim into the flattened task dir so the
    main test can ``import`` them (OT's ``test_outputs.py`` does
    ``from grader import grade``) and read sibling data files
    (``answers.json``, etc.) at grading time.

    Returns the produced task name, or ``None`` if the source is unusable
    (missing container recipe or none of the test_final_candidates present).
    """
    env = src / "environment"
    tests = src / "tests"

    container_def = env / "container.def"
    test_initial = env / "test_initial_state.py"
    test_final: Optional[Path] = None
    for cand in test_final_candidates:
        p = tests / cand
        if p.exists():
            test_final = p
            break
    if test_final is None:
        return None
    if not container_def.exists():
        # Try to derive one from a Dockerfile if present.
        dockerfile = env / "Dockerfile"
        if not dockerfile.exists():
            return None

    # Pull instruction + metadata.
    instruction_md = src / "instruction.md"
    description = ""
    if instruction_md.exists():
        try:
            description = instruction_md.read_text()
        except OSError:
            description = ""

    # ET also ships environment/task.json with {description, truth, name}; honour it
    # when present so we get the author's own truth.
    native_task_json: Dict[str, Any] = {}
    nj = env / "task.json"
    if nj.exists():
        try:
            native_task_json = json.loads(nj.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    toml_meta = _load_task_toml(src / "task.toml")

    task_name = task_name_override or (prefix + re.sub(r"\s+", "_", src.name))

    enriched: Dict[str, Any] = {
        "name": task_name,
        # Leave our native taxonomy empty; a downstream classifier fills
        # classified_* fields used by composition analysis.
        "domain": "unknown",
        "skill_type": "unknown",
        "primitive_skills": [],
        "task_complexity": "unknown",
        "command_complexity": "unknown",
        "scenario": "",
        "language": "any (model's choice)",
        "description": native_task_json.get("description") or description,
        "truth": native_task_json.get("truth", ""),
        # Dataset-native metadata preserved for the appendix panels.
        f"{source_name}_category": toml_meta.get("category", ""),
        f"{source_name}_difficulty": toml_meta.get("difficulty", ""),
        f"{source_name}_tags": toml_meta.get("tags", []),
        f"{source_name}_author_name": toml_meta.get("author_name", ""),
        # Provenance.
        "source": source_name,
        "source_repo": source_repo,
        "source_slug": src.name,
    }

    if extra_task_json is not None:
        extras = extra_task_json(enriched, src) or {}
        enriched.update(extras)

    out = dst_root / task_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "task.json").write_text(json.dumps(enriched, indent=2))

    # container.def: prefer as-shipped; derive from Dockerfile if needed.
    if container_def.exists():
        shutil.copy2(container_def, out / "container.def")
    else:
        dockerfile = env / "Dockerfile"
        # The Dockerfile's COPY sources are relative to the Dockerfile's dir
        # (the `environment/` directory), so we stage that payload next to
        # our output def and hand the derived def absolute paths rooted there.
        if env.exists():
            for entry in env.iterdir():
                if entry.name in ("Dockerfile", "__pycache__"):
                    continue
                if entry.name.startswith("."):
                    continue
                target = out / entry.name
                try:
                    if entry.is_dir():
                        shutil.copytree(entry, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(entry, target)
                except OSError as exc:
                    logger.warning("flatten_harbor_task: could not copy env payload %s -> %s: %s",
                                   entry, target, exc)
        derived = _dockerfile_to_apptainer_def(
            dockerfile.read_text(),
            build_context_dir=out,
        )
        (out / "container.def").write_text(derived)

    shutil.copy2(test_final, out / "test_final_state.py")
    if test_initial.exists():
        shutil.copy2(test_initial, out / "test_initial_state.py")
    else:
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

    # Optionally copy auxiliary files that the main test relies on (OT ships
    # e.g. grader.py and answers.json next to test_outputs.py). We skip shell
    # scripts and the main test itself (already handled above). Directories are
    # copied recursively.
    if copy_aux_test_files and tests.exists():
        for entry in tests.iterdir():
            # Don't re-copy the chosen test file (we've already renamed it
            # to test_final_state.py above).
            if entry == test_final:
                continue
            # Skip bash helpers; the harness only uses Python tests.
            if entry.suffix == ".sh":
                continue
            # Hide pyc/__pycache__ noise and dotfiles.
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            target = out / entry.name
            try:
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, target)
            except OSError as exc:
                logger.warning("flatten_harbor_task: could not copy aux file %s -> %s: %s",
                               entry, target, exc)

    return task_name


# Import adapter modules so they register themselves.
from rl_data.comparison.adapters import (  # noqa: E402, F401
    skill_tax,
    endless_terminals,
    openthoughts_tb,
    openthoughts_agent_rl,
    termigen,
    terminaltraj,
    r2e_gym,
)
