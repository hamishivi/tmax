#!/usr/bin/env bash
set -euo pipefail

# Test run: Terminus-2 + Claude Sonnet 4.5 on a 10-task subset of our RL dataset.
# Uses Daytona cloud sandbox by default (Docker not available on HPC login nodes).
#
# Usage:
#   bash scripts/run_rldata_claude_test.sh
#
#   # Override environment backend (docker requires local Docker daemon)
#   ENV=docker bash scripts/run_rldata_claude_test.sh
#
# Required env vars:
#   ANTHROPIC_API_KEY  - Anthropic API key
#   DAYTONA_API_KEY    - Daytona API key
#
# Optional env vars:
#   MODEL              - Model name (default: anthropic/claude-sonnet-4-20250514)
#   ENV                - Environment backend: "docker" or "daytona" (default: daytona)
#   N_CONCURRENT       - Number of concurrent trials (default: 5 for docker, 10 for daytona)
#   N_TASKS            - Number of tasks in the test subset (default: 10)
#   JOB_NAME           - Job name (default: rldata_test_claude_<env>)

MODEL="${MODEL:-anthropic/claude-sonnet-4-20250514}"
ENV="${ENV:-daytona}"
N_TASKS="${N_TASKS:-10}"
FULL_DATASET="rl_data/output/tasks_skill_tax_20260401_10k_harbor"
TEST_DATASET="rl_data/output/tasks_skill_tax_test_${N_TASKS}_harbor"

if [ "$ENV" = "daytona" ]; then
    N_CONCURRENT="${N_CONCURRENT:-10}"
else
    N_CONCURRENT="${N_CONCURRENT:-5}"
fi

JOB_NAME="${JOB_NAME:-rldata_test_claude_${ENV}}"
JOB_DIR="jobs/${JOB_NAME}"

# ---------- Validate environment ----------
if [ "$ENV" = "docker" ]; then
    if ! command -v docker &>/dev/null || ! docker info &>/dev/null; then
        echo "ERROR: Docker is not available on this system."
        echo "This HPC cluster has Apptainer but not Docker."
        echo "Use Daytona instead (the default):"
        echo "  bash scripts/run_rldata_claude_test.sh"
        exit 1
    fi
fi

if [ "$ENV" = "daytona" ] && [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY is required for Daytona backend."
    echo "  export DAYTONA_API_KEY='your-key'"
    exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is required."
    echo "  export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# ---------- Validate source dataset ----------
if [ ! -d "$FULL_DATASET" ]; then
    echo "ERROR: Harbor dataset not found at $FULL_DATASET"
    echo "Run the conversion script first:"
    echo "  uv run python rl_data/scripts/analyze/convert_to_harbor.py \\"
    echo "      --src rl_data/output/tasks_skill_tax_20260401_10k \\"
    echo "      --dst $FULL_DATASET"
    exit 1
fi

# ---------- Create test subset (first N tasks) ----------
if [ ! -d "$TEST_DATASET" ] || [ "$(ls -d "$TEST_DATASET"/task_* 2>/dev/null | wc -l)" -ne "$N_TASKS" ]; then
    echo "Creating test subset with $N_TASKS tasks..."
    rm -rf "$TEST_DATASET"
    mkdir -p "$TEST_DATASET"

    # Symlink first N task directories for speed (no disk copy needed)
    count=0
    for task_dir in $(ls -d "$FULL_DATASET"/task_* | head -n "$N_TASKS"); do
        ln -s "$(realpath "$task_dir")" "$TEST_DATASET/$(basename "$task_dir")"
        count=$((count + 1))
    done
    echo "Created test dataset with $count tasks at $TEST_DATASET"
else
    echo "Using existing test dataset at $TEST_DATASET"
fi

# ---------- Run or resume ----------
if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    if [ "$ENV" = "daytona" ]; then
        uv run harbor jobs resume \
            --job-path "$JOB_DIR" \
            --filter-error-type DaytonaError
    else
        uv run harbor jobs resume \
            --job-path "$JOB_DIR"
    fi
else
    echo "Starting new test job: $JOB_NAME (env=$ENV, model=$MODEL, n=$N_CONCURRENT)"
    uv run harbor run \
        --path "$TEST_DATASET" \
        --agent terminus-2 \
        --model "$MODEL" \
        --env "$ENV" \
        --n-concurrent "$N_CONCURRENT" \
        --job-name "$JOB_NAME"
fi
