#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Upload a trained model to Hugging Face ───────────────────────────
#
# Uploads the final model (safetensors, config, tokenizer, etc.) from
# a training output directory to a HF model repo.
#
# Usage:
#   bash scripts/upload_model_to_hf.sh
#   bash scripts/upload_model_to_hf.sh --model-dir /path/to/output --repo osieosie/my-model
#   bash scripts/upload_model_to_hf.sh --private
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with huggingface_hub in PATH

MODEL_DIR="/gpfs/scrubbed/osey/tmax/sft/output/Qwen3.5-4B_tbmax_terminus2_sweagent_full_20260310_v2_qwen3.5_42"
REPO_ID="osieosie/tmax-qwen3.5-4b-sft-20260313"
PRIVATE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --repo)       REPO_ID="$2"; shift 2 ;;
        --private)    PRIVATE="true"; shift ;;
        --public)     PRIVATE="false"; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Default repo name derived from the output directory basename
if [ -z "$REPO_ID" ]; then
    REPO_ID="osieosie/$(basename "$MODEL_DIR")"
fi

echo "=== Upload Model to Hugging Face ==="
echo "  Model dir: ${MODEL_DIR}"
echo "  Repo:      ${REPO_ID}"i
echo "  Private:   ${PRIVATE}"
echo ""

python - --model-dir "${MODEL_DIR}" --repo "${REPO_ID}" --private "${PRIVATE}" << 'PYTHON_EOF'
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


UPLOAD_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.jinja",
    "tokenizer.*",
    "*.model",
    "*.txt",
    "*.md",
]

IGNORE_PATTERNS = [
    "checkpoint-*",
    "runs/**",
    "*.log",
    "*.sh",
    "*.bin",
    "wandb/**",
    ".node_launcher_*",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--private", default="false")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    repo_id = args.repo
    private = args.private.lower() == "true"

    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        print(f"Error: no .safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"\nUploading from {model_dir} ...")
    print(f"  Include: {UPLOAD_PATTERNS}")
    print(f"  Ignore:  {IGNORE_PATTERNS}")
    print()

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=UPLOAD_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nDone! Model uploaded to: {url}")


if __name__ == "__main__":
    main()
PYTHON_EOF
