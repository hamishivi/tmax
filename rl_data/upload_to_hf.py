"""Upload RL task dataset to Hugging Face.

Uploads the raw task folder structure (task_*/*, analysis/*) AND a
consolidated .parquet file so HuggingFace Dataset Viewer can preview
the data directly on the web.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import tempfile
import zipfile

from huggingface_hub import HfApi

# Internal pipeline artifacts that should never be uploaded.
ALWAYS_IGNORE = ["logs/**", "_*.jsonl", "_*.txt"]


def _read_file_text(path: Path) -> str:
    """Read a file as UTF-8 text, returning empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return ""


def is_task_verified(task_dir: Path) -> bool:
    """Check if a task has at least one non-zero pass_at_k across all summary files."""
    solutions_dir = task_dir / "solutions"
    if not solutions_dir.is_dir():
        return False

    for summary_path in solutions_dir.glob("*_summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            pass_at_k = summary.get("pass_at_k", {})
            if any(v > 0 for v in pass_at_k.values()):
                return True
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return False


def build_parquet(
    input_dir: Path, *, allowed_tasks: set[str] | None = None
) -> Path | None:
    """Build a train.parquet from all task_* dirs for HF Dataset Viewer.

    Reads each task's task.json plus its companion files and writes a single
    Parquet file to ``<input_dir>/data/train-00000-of-00001.parquet``.
    HuggingFace auto-discovers parquet files under ``data/`` for preview.

    If *allowed_tasks* is given, only those task directory names are included.
    """
    try:
        import pandas as pd
    except ImportError:
        print("WARNING: pandas/pyarrow not available, skipping parquet generation")
        return None

    task_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_")
    )
    if allowed_tasks is not None:
        task_dirs = [d for d in task_dirs if d.name in allowed_tasks]
    if not task_dirs:
        print("No task directories found, skipping parquet")
        return None

    rows: list[dict] = []
    for td in task_dirs:
        task_json_path = td / "task.json"
        if not task_json_path.exists():
            continue

        with open(task_json_path, "r", encoding="utf-8") as f:
            task = json.load(f)

        prim = task.get("primitive_skills", [])

        rows.append(
            {
                "task_id": task.get("name", td.name),
                "domain": task.get("domain", ""),
                "skill_type": task.get("skill_type", ""),
                "primitive_skills": json.dumps(prim) if isinstance(prim, list) else str(prim),
                "task_complexity": task.get("task_complexity", ""),
                "command_complexity": task.get("command_complexity", ""),
                "scenario": task.get("scenario", ""),
                "description": task.get("description", ""),
                "truth": task.get("truth", ""),
                "test_initial_state": _read_file_text(td / "test_initial_state.py"),
                "test_final_state": _read_file_text(td / "test_final_state.py"),
                "container_def": _read_file_text(td / "container.def"),
            }
        )

    if not rows:
        print("No valid task.json files found, skipping parquet")
        return None

    df = pd.DataFrame(rows)

    data_dir = input_dir / "data"
    data_dir.mkdir(exist_ok=True)
    parquet_path = data_dir / "train-00000-of-00001.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(
        f"Generated parquet: {parquet_path} "
        f"({len(df)} rows, {parquet_path.stat().st_size / 1024:.1f} KB)"
    )
    return parquet_path


def _clear_upload_cache(input_dir: Path) -> None:
    """Remove the upload cache that ``upload_large_folder`` stores in the source tree.

    The cache lives at ``<input_dir>/.cache/huggingface/upload/`` and tracks
    which files were already committed.  It is **not** repo-specific, so a
    previous upload to repo A will cause a subsequent upload to repo B to skip
    all unchanged files.  Clearing it forces a full re-hash and re-upload.
    """
    cache_dir = input_dir / ".cache" / "huggingface" / "upload"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        print(f"Cleared stale upload cache: {cache_dir}")


def _build_compact_staging(
    input_dir: Path,
    staging_dir: Path,
    allowed_tasks: set[str] | None,
) -> None:
    """Build a staging directory containing parquet + analysis + tasks.zip."""
    task_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_")
    )
    if allowed_tasks is not None:
        task_dirs = [d for d in task_dirs if d.name in allowed_tasks]

    # Copy data/ (parquet) and analysis/ — must be real copies since
    # upload_folder does not follow symlinks.
    for name in ("data", "analysis"):
        src = input_dir / name
        if src.is_dir():
            shutil.copytree(src, staging_dir / name)

    # Zip all eligible task folders
    zip_path = staging_dir / "tasks.zip"
    print(f"Zipping {len(task_dirs)} task folders → {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for td in task_dirs:
            for fpath in sorted(td.rglob("*")):
                if fpath.is_file():
                    arcname = str(fpath.relative_to(input_dir))
                    zf.write(fpath, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created {zip_path.name} ({size_mb:.1f} MB, {len(task_dirs)} tasks)")


def upload(
    repo_id: str,
    input_dir: Path,
    *,
    private: bool = False,
    generate_parquet: bool = True,
    verified_only: bool = False,
    clean: bool = True,
    fast: bool = False,
    compact: bool = False,
) -> None:
    """Upload a task output directory to a HuggingFace dataset repo.

    Upload modes (mutually exclusive, checked in this order):

    * **compact** — zips task folders into a single ``tasks.zip``, uploads
      parquet + zip + analysis via ``upload_folder`` (fastest).
    * **fast** — uploads raw files via ``upload_folder`` in a single commit
      (fast for < ~25 K files, auto-falls back otherwise).
    * **default** — ``upload_large_folder`` with batched commits (slowest but
      resumable).
    """
    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_")
    )
    other_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("task_")
    )
    print(
        f"Found {len(task_dirs)} task folders, "
        f"{len(other_dirs)} other folders ({', '.join(d.name for d in other_dirs)})"
    )

    ignore_patterns: list[str] = list(ALWAYS_IGNORE)
    allowed_tasks: set[str] | None = None

    if verified_only:
        print("\n── Filtering to verified tasks only ──")
        skipped = [d for d in task_dirs if not is_task_verified(d)]
        n_verified = len(task_dirs) - len(skipped)
        print(
            f"Verified filter: {n_verified}/{len(task_dirs)} tasks pass "
            f"(skipped {len(skipped)} with all-zero pass@k)"
        )
        if n_verified == 0:
            print("No verified tasks found — nothing to upload.", file=sys.stderr)
            sys.exit(1)
        allowed_tasks = {d.name for d in task_dirs} - {d.name for d in skipped}
        ignore_patterns += [f"{d.name}/**" for d in skipped]

    if generate_parquet:
        print("\n── Generating parquet for HF Dataset Viewer ──")
        build_parquet(input_dir, allowed_tasks=allowed_tasks)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=private)
    visibility = "private" if private else "public"
    print(f"\nRepo ready ({visibility}): https://huggingface.co/datasets/{repo_id}")

    # ── compact mode: parquet + analysis + tasks.zip ──────────────────────
    if compact:
        with tempfile.TemporaryDirectory(prefix="hf_compact_") as tmp:
            staging_dir = Path(tmp)
            _build_compact_staging(input_dir, staging_dir, allowed_tasks)
            print(f"Uploading staging dir {staging_dir} (compact / single-commit) ...")
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(staging_dir),
                commit_message="Upload dataset (compact: parquet + tasks.zip)",
            )
        print(f"\nDone! https://huggingface.co/datasets/{repo_id}")
        return

    # ── fast mode: single-commit raw files ────────────────────────────────
    if fast:
        n_files = sum(1 for _ in input_dir.rglob("*") if _.is_file())
        if n_files > 25_000:
            print(
                f"WARNING: --fast skipped — {n_files:,} files is too many for "
                f"a single commit (HF Hub will 504 timeout).\n"
                f"Falling back to resilient multi-commit mode."
            )
            fast = False

    if fast:
        print(f"Uploading folder {input_dir} (fast / single-commit mode) ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(input_dir),
            ignore_patterns=ignore_patterns,
            commit_message="Upload dataset",
        )
        print(f"\nDone! https://huggingface.co/datasets/{repo_id}")
        return

    # ── default mode: resilient batched commits ───────────────────────────
    if clean:
        _clear_upload_cache(input_dir)
    print(f"Uploading folder {input_dir} (resilient / multi-commit mode) ...")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(input_dir),
        ignore_patterns=ignore_patterns,
    )

    print(f"\nDone! https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="HF dataset repo id (e.g. user/dataset-name)")
    p.add_argument("--input-dir", required=True, help="Path to task output directory")
    p.add_argument("--private", action="store_true", help="Make the repo private")
    p.add_argument("--no-parquet", action="store_true", help="Skip parquet generation")
    p.add_argument(
        "--verified-only",
        action="store_true",
        help="Only upload tasks with at least one non-zero pass@k",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep stale upload cache (allows resuming an interrupted upload)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use single-commit upload_folder (faster, no resume; auto-falls "
        "back to multi-commit above ~25K files to avoid HF Hub timeout)",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Zip task folders into tasks.zip and upload parquet + zip only "
        "(fastest for large datasets; users unzip after download)",
    )
    args = p.parse_args()

    upload(
        repo_id=args.repo,
        input_dir=Path(args.input_dir),
        private=args.private,
        generate_parquet=not args.no_parquet,
        verified_only=args.verified_only,
        clean=not args.no_clean,
        fast=args.fast,
        compact=args.compact,
    )


if __name__ == "__main__":
    main()
