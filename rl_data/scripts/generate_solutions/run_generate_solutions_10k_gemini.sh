#!/bin/bash
#SBATCH --job-name=rl-gen-sol-1k
#SBATCH --output=logs/gen_sol_1k_%j.out
#SBATCH --error=logs/gen_sol_1k_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Single-node setup — pick ONE GPU allocation.                       ║
# ║                                                                     ║
# ║  Each GPU gives 8 CPUs + 240 GB RAM.                               ║
# ║  Concurrent containers = WORKERS × NUM_SOLUTIONS.                  ║
# ║  Rule: containers ≤ CPUs (1:1) for safety, up to 1.5× for I/O     ║
# ║  heavy workloads.                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Option A: 4 GPUs (default) ──
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G
# WORKERS=4                    # 4 tasks × 8 solutions = 32 containers = 32 CPUs
                             # ~1000/4 = 250 rounds × ~10 min = ~42h

# ── Option B: 8 GPUs (faster) ──
# #SBATCH --gres=gpu:8
# #SBATCH --cpus-per-task=64
# #SBATCH --mem=1920G
WORKERS=12                  # 8 tasks × 8 solutions = 64 containers = 64 CPUs

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260401_10k"
MODEL="gemini/gemini-3-flash-preview"
NUM_SOLUTIONS=8              # solution attempts per task (determines pass@k quality)
MAX_ACTIONS=16               # max agent turns per solution attempt
MAX_TOKENS=65536             # LLM context window for the agent
NUM_TASKS=999999             # processes all task_* dirs in TASKS_DIR
START_AT=0                   # skip first N tasks (useful for resuming a partial run)
SOLUTION_TEMPERATURE=0.7     # LLM sampling temperature for solutions
COMMAND_TIMEOUT=60           # per-command timeout (seconds) inside containers;
                             # bumped from 30 — C/Rust/Go compile commands can take 10-30s
SHELL_INIT_TIMEOUT=240       # seconds to wait for Apptainer shell to initialise;
                             # higher than default (120) due to many concurrent containers
SHELL_INIT_ATTEMPTS=3        # retries if shell init times out
BUILD_WORKERS=4              # concurrent SIF builds in pre-pass; mostly a no-op with BASE_SIFS_DIR
BUILD_RETRIES=3              # retries per SIF build with exponential backoff
BASE_SIFS_DIR="rl_data/containers"  # pre-built base SIFs; skips per-task SIF builds entirely
FORCE_RERUN=0                # 0 = skip tasks that already have a *_summary.json (default)
                             # 1 = re-run all tasks even if solutions already exist (overwrites)
LOG_COMMANDS=0               # 0 = off (default); 1 = write every bash command + raw PTY output
                             # to per-solution log files under each task dir (debug only, large files)
DISABLE_TERMINAL_LOG=0       # 0 = tee stdout/stderr to a log file in TASKS_DIR/logs/ (default)
                             # 1 = skip the terminal log file, only print to console

# Pool workers: ThreadPoolExecutor size for container init, command exec,
# and final tests *within each task*.  Needs to be ≥ NUM_SOLUTIONS.
NUM_POOL_WORKERS=16
# --------------------------------

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# Docker Hub auth — avoids anonymous rate limit.
# ⚠️  Do NOT commit real credentials.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Redirect Apptainer instance logs off $HOME to avoid home quota exhaustion
# under high concurrency (12 workers × 8 solutions = thousands of instance logs).
# Keep apptainer instance logs off GPFS; heal dangling symlinks left behind
# when /tmp was cleaned between runs (see run_generate_solutions_et.sh for
# rationale).
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

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

echo "=== Single-node: ${NUM_TASKS} tasks, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
echo "=== Concurrent containers: $(( WORKERS * NUM_SOLUTIONS )) ==="

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
