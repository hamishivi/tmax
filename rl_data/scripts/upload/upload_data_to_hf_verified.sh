#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

# ── Upload VERIFIED RL tasks to Hugging Face ─────────────────────────
#
# Same as upload_data_to_hf.sh but only includes tasks that have at
# least one non-zero pass@k in their *_summary.json — i.e. tasks where
# a solution has been tested and verified to work.
#
# Usage:
#   bash rl_data/scripts/upload/upload_data_to_hf_verified.sh
#   bash rl_data/scripts/upload/upload_data_to_hf_verified.sh --input-dir rl_data/output/tasks_v2
#   bash rl_data/scripts/upload/upload_data_to_hf_verified.sh --repo osieosie/tmax-rl-v2-verified --private
#   bash rl_data/scripts/upload/upload_data_to_hf_verified.sh --no-parquet
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with huggingface_hub, pandas, pyarrow

REPO_ID="osieosie/tmax-tasks-skill-taxonomy-20260401-10k-verified"
INPUT_DIR="/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_20260401_10k"
PRIVATE=""
# These are "opt-out" flags — empty by default (feature ON), set to the
# corresponding CLI flag string when the user passes the option.
#   NO_PARQUET=""        → parquet generation is enabled (default)
#   NO_PARQUET="--no-parquet" → parquet generation is skipped
#   NO_CLEAN=""          → stale upload cache is cleared before upload (default)
#   NO_CLEAN="--no-clean"    → cache is kept, allowing resume of interrupted uploads
#   FAST=""              → use resilient multi-commit upload (default)
#   FAST="--fast"        → use single-commit upload (faster, no resume)
#   COMPACT=""           → upload raw files (default)
#   COMPACT="--compact"  → zip task folders + upload parquet & zip (fastest)
NO_PARQUET=""
NO_CLEAN=""
FAST=""
COMPACT="--compact"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO_ID="$2"; shift 2 ;;
        --input-dir)    INPUT_DIR="$2"; shift 2 ;;
        --private)      PRIVATE="--private"; shift ;;
        --public)       PRIVATE=""; shift ;;
        --no-parquet)   NO_PARQUET="--no-parquet"; shift ;;
        --no-clean)     NO_CLEAN="--no-clean"; shift ;;
        --fast)         FAST="--fast"; shift ;;
        --compact)      COMPACT="--compact"; shift ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Upload VERIFIED RL Dataset to Hugging Face ==="
echo "  Repo:       ${REPO_ID}"
echo "  Input dir:  ${INPUT_DIR}"
echo "  Filter:     verified-only (pass@k > 0)"
echo ""

exec uv run python -m rl_data.upload_to_hf \
    --repo "${REPO_ID}" \
    --input-dir "${INPUT_DIR}" \
    --verified-only \
    ${PRIVATE} ${NO_PARQUET} ${NO_CLEAN} ${FAST} ${COMPACT}
