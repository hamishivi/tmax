#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Upload converted dataset to Hugging Face ─────────────────────────
#
# Uploads the parquet files from a pipeline output directory to a HF
# dataset repo.  The dataset viewer ("preview") works automatically
# for parquet files once pushed.
#
# Usage:
#   bash scripts/upload_to_hf.sh                       # defaults below
#   bash scripts/upload_to_hf.sh --input-dir preprocessing/terminus2_sweagent_1pct
#   bash scripts/upload_to_hf.sh --repo osieosie/tmax-sft-preview --private
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with datasets + huggingface_hub in PATH

REPO_ID="${HF_REPO:-osieosie/tmax-sft-preview}"
INPUT_DIR="preprocessing/terminus2_sweagent_1pct"
PRIVATE="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)       REPO_ID="$2"; shift 2 ;;
        --input-dir)  INPUT_DIR="$2"; shift 2 ;;
        --private)    PRIVATE="true"; shift ;;
        --public)     PRIVATE="false"; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Upload to Hugging Face ==="
echo "  Repo:      ${REPO_ID}"
echo "  Input dir: ${INPUT_DIR}"
echo "  Private:   ${PRIVATE}"
echo ""

python - --repo "${REPO_ID}" --input-dir "${INPUT_DIR}" --private "${PRIVATE}" << 'PYTHON_EOF'
import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--private", default="true")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    repo_id = args.repo
    private = args.private.lower() == "true"

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")

    # Load each parquet as a split named after the source
    splits = {}
    for pf in parquet_files:
        split_name = pf.stem.replace("-", "_")
        ds = Dataset.from_parquet(str(pf))
        splits[split_name] = ds
        print(f"  Loaded {pf.name}: {len(ds)} rows → split '{split_name}'")

    dd = DatasetDict(splits)
    dd.push_to_hub(repo_id, private=private)
    print(f"\nPushed {len(splits)} split(s) to https://huggingface.co/datasets/{repo_id}")

    # Also upload the conversion report if present
    report_path = input_dir / "conversion_report.json"
    if report_path.exists():
        api.upload_file(
            path_or_fileobj=str(report_path),
            path_in_repo="conversion_report.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded conversion_report.json")

    print(f"\nDone. Preview at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
PYTHON_EOF
