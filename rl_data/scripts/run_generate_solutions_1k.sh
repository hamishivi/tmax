#!/bin/bash
#SBATCH --job-name=rl-gen-sol-1k
#SBATCH --output=logs/gen_sol_1k_%A_%a.out
#SBATCH --error=logs/gen_sol_1k_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Multi-node configuration — pick ONE block and uncomment it.        ║
# ║                                                                     ║
# ║  Both setups use 4 GPUs total and finish in ~42 h.                  ║
# ║  • 4-node: lower per-node load, better fault isolation              ║
# ║  • 2-node: fewer jobs to monitor, slightly more per-node headroom   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Option A: 4 nodes × 1 GPU (default) ──
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
N_NODES=4
WORKERS=1                   # 1 task at a time → 8 containers = 8 CPUs

# ── Option B: 2 nodes × 2 GPUs ──
# Uncomment below (and comment Option A lines above):
# #SBATCH --array=0-1
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=480G
# N_NODES=2
# WORKERS=2                 # 2 tasks at a time → 16 containers = 16 CPUs

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260323_1k"
MODEL="gemini/gemini-3-flash-preview"
NUM_SOLUTIONS=8
MAX_ACTIONS=16
MAX_TOKENS=65536
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=30
SHELL_INIT_TIMEOUT=180      # higher than toy — more concurrent containers system-wide
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=2              # concurrent SIF builds in pre-pass (safe at 2 on GPFS)
BUILD_RETRIES=3
BASE_SIFS_DIR="rl_data/containers"
FORCE_RERUN=0                # set 1 to regenerate even if *_summary.json exists
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# ---- Multi-node task partitioning (auto-computed) ----
# Total tasks in TASKS_DIR that we want to process across all array members.
# Set to the number of surviving tasks from the task-generation step.
# If the directory has fewer task_* dirs, each node just processes what exists.
TOTAL_TASKS=1000
TASKS_PER_NODE=$(( (TOTAL_TASKS + N_NODES - 1) / N_NODES ))  # ceiling division
START_AT=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_NODE ))
NUM_TASKS=$TASKS_PER_NODE

# Pool workers: controls ThreadPoolExecutor size for container init, command
# execution, and final tests *within each task*.  Only needs to be ≥ NUM_SOLUTIONS.
# Set to 2× for headroom.
NUM_POOL_WORKERS=16
# --------------------------------

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_node${SLURM_ARRAY_TASK_ID}_${_RUN_TS}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# Docker Hub auth — avoids anonymous rate limit (100 pulls/6h).
# ⚠️  Do NOT commit real credentials.  Replace or source from a secrets file.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ -n "${BASE_SIFS_DIR:-}" ]]; then
  EXTRA_ARGS+=(--base-sifs-dir "$BASE_SIFS_DIR")
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== Array task ${SLURM_ARRAY_TASK_ID}/${N_NODES}: tasks ${START_AT}..$(( START_AT + NUM_TASKS - 1 )) ==="
echo "=== WORKERS=${WORKERS}  NUM_SOLUTIONS=${NUM_SOLUTIONS}  containers=$(( WORKERS * NUM_SOLUTIONS )) ==="

uv run python -m rl_data.generate_solutions \
    --tasks-dir "$TASKS_DIR" \
    --model "$MODEL" \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --num-tasks "$NUM_TASKS" \
    --start-at "$START_AT" \
    --workers "$WORKERS" \
    --num-pool-workers "$NUM_POOL_WORKERS" \
    --solution-temperature "$SOLUTION_TEMPERATURE" \
    --command-timeout "$COMMAND_TIMEOUT" \
    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
    --build-workers "$BUILD_WORKERS" \
    --build-retries "$BUILD_RETRIES" \
    --verbose \
    "${EXTRA_ARGS[@]}"
