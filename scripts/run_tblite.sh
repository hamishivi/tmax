#!/usr/bin/env bash
set -euo pipefail

# Run TassieAgent on OpenThoughts-TBLite (Daytona backend)
# Resumable: re-running this script resumes the previous job if it exists.
#
# Required env vars:
#   DAYTONA_API_KEY    - Daytona API key
#   OPENAI_API_BASE    - vLLM server URL (e.g. http://localhost:8000/v1)
#
# Optional env vars:
#   MODEL              - Model name (default: openai/default)
#   N_CONCURRENT       - Number of concurrent trials (default: 25)
#   MAX_STEPS          - Max agent steps per trial (default: 50)
#   JOB_NAME           - Job name for resumability (default: tblite)

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
MODEL_NAME="${MODEL_NAME:-default}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8008}"
N_CONCURRENT="${N_CONCURRENT:-20}"
MAX_STEPS="${MAX_STEPS:-50}"
N_ATTEMPTS="${N_ATTEMPTS:-3}"
JOB_NAME="${JOB_NAME:-tblite}"
JOB_DIR="jobs/${JOB_NAME}"

if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    uv run harbor jobs resume \
        --job-path "$JOB_DIR" \
        --filter-error-type DaytonaError
else
    uv run harbor run \
        --dataset openthoughts-tblite@2.0 \
        --agent-import-path TassieAgent:TassieAgent \
        --model "hosted_vllm/${MODEL_NAME}" \
        --env daytona \
        --n-concurrent "$N_CONCURRENT" \
        --agent-kwarg "max_steps=$MAX_STEPS" \
        --agent-kwarg "api_base=http://${VLLM_HOST}:${VLLM_PORT}/v1" \
        --job-name "$JOB_NAME" \
        -k "$N_ATTEMPTS"
fi
