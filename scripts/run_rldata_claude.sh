#!/usr/bin/env bash
set -euo pipefail

# Run Terminus-2 on our generated RL dataset with Claude Sonnet 4.5 (Daytona backend)
# Resumable: re-running this script resumes the previous job if it exists.
#
# Before running this, convert the tasks to Harbor format:
#   uv run python rl_data/scripts/analyze/convert_to_harbor.py \
#       --src rl_data/output/tasks_skill_tax_20260401_10k \
#       --dst rl_data/output/tasks_skill_tax_20260401_10k_harbor
#
# Required env vars:
#   DAYTONA_API_KEY    - Daytona API key
#   ANTHROPIC_API_KEY  - Anthropic API key
#
# Optional env vars:
#   MODEL              - Model name (default: anthropic/claude-sonnet-4-20250514)
#   N_CONCURRENT       - Number of concurrent trials (default: 25)
#   JOB_NAME           - Job name for resumability (default: rldata_10k_claude)
#   DATASET_PATH       - Path to converted Harbor dataset

MODEL="${MODEL:-anthropic/claude-sonnet-4-20250514}"
N_CONCURRENT="${N_CONCURRENT:-25}"
JOB_NAME="${JOB_NAME:-rldata_10k_claude}"
DATASET_PATH="${DATASET_PATH:-rl_data/output/tasks_skill_tax_20260401_10k_harbor}"
JOB_DIR="jobs/${JOB_NAME}"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run the conversion script first:"
    echo "  uv run python rl_data/scripts/analyze/convert_to_harbor.py \\"
    echo "      --src rl_data/output/tasks_skill_tax_20260401_10k \\"
    echo "      --dst $DATASET_PATH"
    exit 1
fi

if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    uv run harbor jobs resume \
        --job-path "$JOB_DIR" \
        --filter-error-type DaytonaError
else
    uv run harbor run \
        --path "$DATASET_PATH" \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-concurrent "$N_CONCURRENT" \
        --job-name "$JOB_NAME"
fi
