"""Upload RL task dataset to Hugging Face.

Uploads the raw task folder structure (task_*/*, analysis/*) AND a
consolidated .parquet file so HuggingFace Dataset Viewer can preview
the data directly on the web.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


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


def upload(
    repo_id: str,
    input_dir: Path,
    *,
    private: bool = False,
    generate_parquet: bool = True,
    verified_only: bool = False,
) -> None:
    """Upload a task output directory to a HuggingFace dataset repo."""
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

    ignore_patterns: list[str] | None = None
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
        ignore_patterns = [f"{d.name}/**" for d in skipped]

    if generate_parquet:
        print("\n── Generating parquet for HF Dataset Viewer ──")
        build_parquet(input_dir, allowed_tasks=allowed_tasks)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=private)
    visibility = "private" if private else "public"
    print(f"\nRepo ready ({visibility}): https://huggingface.co/datasets/{repo_id}")

    print(f"Uploading folder {input_dir} ...")
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
    args = p.parse_args()

    upload(
        repo_id=args.repo,
        input_dir=Path(args.input_dir),
        private=args.private,
        generate_parquet=not args.no_parquet,
        verified_only=args.verified_only,
    )


if __name__ == "__main__":
    main()
