#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Upload RL task dataset to Hugging Face ───────────────────────────
#
# Uploads the raw task folder structure (task_*/*, analysis/*) directly
# to a HF dataset repo, preserving the on-disk layout.
#
# Usage:
#   bash rl_data/scripts/upload_data_to_hf.sh
#   bash rl_data/scripts/upload_data_to_hf.sh --input-dir rl_data/output/tasks_v2
#   bash rl_data/scripts/upload_data_to_hf.sh --repo osieosie/tmax-rl-v2 --private
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with huggingface_hub

REPO_ID="osieosie/tmax-tasks-skill-taxonomy-20260323-1k"
INPUT_DIR="/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_20260323_1k"
PRIVATE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)              REPO_ID="$2"; shift 2 ;;
        --input-dir)         INPUT_DIR="$2"; shift 2 ;;
        --private)           PRIVATE="true"; shift ;;
        --public)            PRIVATE="false"; shift ;;
        *)                   echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Upload RL Dataset to Hugging Face ==="
echo "  Repo:              ${REPO_ID}"
echo "  Input dir:         ${INPUT_DIR}"
echo "  Private:           ${PRIVATE}"
echo ""

uv run python - --repo "${REPO_ID}" --input-dir "${INPUT_DIR}" \
    --private "${PRIVATE}" << 'PYTHON_EOF'
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--private", default="false")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    private = args.private.lower() == "true"

    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("task_")
    )
    other_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("task_")
    )
    print(f"Found {len(task_dirs)} task folders, {len(other_dirs)} other folders ({', '.join(d.name for d in other_dirs)})")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=private, exist_ok=True)
    api.update_repo_settings(args.repo, repo_type="dataset", private=private)
    print(f"Repo ready ({'private' if private else 'public'}): https://huggingface.co/datasets/{args.repo}")

    print(f"Uploading folder {input_dir} ...")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(input_dir),
    )

    print(f"\nDone! https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
PYTHON_EOF
