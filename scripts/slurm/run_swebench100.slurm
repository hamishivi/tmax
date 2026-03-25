#!/usr/bin/env bash
#
# Run TassieAgent on 100 deterministic SWE-Bench Verified tasks (Daytona backend)
# on Tillicum. The 100 tasks are pre-selected in scripts/swebench100_tasks.txt
# (seed=42). Resumable.
#
# Usage:
#   sbatch scripts/slurm/run_swebench100.slurm
#   sbatch --export=ALL,MODEL=openai/gpt-4o scripts/slurm/run_swebench100.slurm
#
# Secrets:
#   API keys are loaded from ~/.secrets/tmax.env (chmod 600). Example contents:
#     export DAYTONA_API_KEY=...
#     export ANTHROPIC_API_KEY=sk-ant-...
#     export OPENAI_API_KEY=sk-...
#     export OPENAI_API_BASE=http://host:8000/v1   # only for self-hosted vLLM
#
# Optional env vars:
#   MODEL              - Model name (default: anthropic/claude-sonnet-4-20250514)
#   N_CONCURRENT       - Number of concurrent trials (default: 25)
#   MAX_STEPS          - Max agent steps per trial (default: 100)
#   JOB_NAME           - Job name for resumability (default: swebench100)
#   N_ATTEMPTS         - Attempts per task (default: 5)
#   TMAX_DIR           - Project root on GPFS (default: /gpfs/scrubbed/osey/tmax)
#   SECRETS_FILE       - Path to secrets file (default: ~/.secrets/tmax.env)
#
# Resource notes:
#   Tillicum requires >=1 GPU per job. 1 GPU = 8 CPUs + 200GB RAM.
#   The GPU is unused; sandboxes run remotely via Daytona.
#
# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=swebench100
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-swebench100.out

set -euo pipefail

TMAX_DIR="${TMAX_DIR:-/gpfs/scrubbed/osey/tmax}"
cd "$TMAX_DIR"

# ── Secrets ────────────────────────────────────────────────────────────────────
SECRETS_FILE="${SECRETS_FILE:-$HOME/.secrets/tmax.env}"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
else
    echo "Warning: secrets file not found at $SECRETS_FILE"
    echo "Create it with: mkdir -p ~/.secrets && chmod 700 ~/.secrets"
    echo "Then add your API keys and chmod 600 ~/.secrets/tmax.env"
fi

TASKS_FILE="${TMAX_DIR}/scripts/swebench100_tasks.txt"

MODEL_NAME="${MODEL_NAME:-default}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8008}"
N_CONCURRENT="${N_CONCURRENT:-20}"
MAX_STEPS="${MAX_STEPS:-100}"
JOB_NAME="${JOB_NAME:-swebench100}"
N_ATTEMPTS="${N_ATTEMPTS:-3}"
JOB_DIR="jobs/${JOB_NAME}"

TASK_ARGS=()
while IFS= read -r task; do
    TASK_ARGS+=("--task-name" "$task")
done < "$TASKS_FILE"

echo "=== SWE-Bench 100 (Daytona) ==="
echo "  Model:       hosted_vllm/${MODEL_NAME}"
echo "  VLLM:        ${VLLM_HOST}:${VLLM_PORT}"
echo "  Concurrent:  $N_CONCURRENT"
echo "  Max steps:   $MAX_STEPS"
echo "  Attempts:    $N_ATTEMPTS"
echo "  Tasks:       ${#TASK_ARGS[@]} args (100 tasks)"
echo "  Job dir:     $JOB_DIR"
echo ""

if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    uv run harbor jobs resume \
        --job-path "$JOB_DIR" \
        --filter-error-type DaytonaError
else
    uv run harbor run \
        --dataset swebench-verified@1.0 \
        --agent-import-path TassieAgent:TassieAgent \
        --model "hosted_vllm/${MODEL_NAME}" \
        --env daytona \
        --n-concurrent "$N_CONCURRENT" \
        --agent-kwarg "max_steps=$MAX_STEPS" \
        --agent-kwarg "api_base=http://${VLLM_HOST}:${VLLM_PORT}/v1" \
        --job-name "$JOB_NAME" \
        -k "$N_ATTEMPTS" \
        "${TASK_ARGS[@]}"
fi
