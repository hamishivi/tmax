#!/bin/bash
#SBATCH --job-name=rl-gen-solutions
#SBATCH --output=logs/gen_solutions_%j.out
#SBATCH --error=logs/gen_solutions_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260327_toy"
MODEL="gemini/gemini-3-flash-preview" #gemini-3-flash-preview, gemini-3.1-flash-lite-preview, "gemini/gemini-3.1-pro-preview"
NUM_SOLUTIONS=8
MAX_ACTIONS=16 # max turns
MAX_TOKENS=65536
NUM_TASKS=10
START_AT=0
WORKERS=10                   # parallel tasks (each runs NUM_SOLUTIONS agent loops)
NUM_POOL_WORKERS=128        # parallel LLM calls within each turn
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=30          # per-command timeout in seconds inside containers
# First shell prompt: under WORKERS×NUM_SOLUTIONS concurrent Apptainers, raise if you see "Shell init timed out"
SHELL_INIT_TIMEOUT=120
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=1             # concurrent SIF builds in pre-pass (1 = serial, safe; bump to 2-3 if I/O allows)
BUILD_RETRIES=3             # retries per SIF build with exponential backoff
BASE_SIFS_DIR="rl_data/containers"  # shared base SIFs; set empty to use per-task SIF builds
FORCE_RERUN=1               # set to 1 to re-run all tasks even if *_summary.json exists
LOG_COMMANDS=0              # 1: append bash I/O to per-task log dir (default: solutions/debug_commands)
# COMMAND_LOG_DIR=output/debug_commands   # optional; relative to each task dir if not absolute
# Full copy of stdout+stderr from this Python process (see also SBATCH --output above):
DISABLE_TERMINAL_LOG=0      # set to 1 to skip --terminal-log
# Each run gets a unique log: <model>_<timestamp>.log
_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# OCI blob cache on GPFS (persistent across jobs — avoids re-pulling from Docker Hub)
export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
# Build scratch on local NVMe (fast I/O, ephemeral — cleaned up when allocation ends)
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ -n "${COMMAND_LOG_DIR:-}" ]]; then
  EXTRA_ARGS+=(--command-log-dir "$COMMAND_LOG_DIR")
fi
if [[ -n "${BASE_SIFS_DIR:-}" ]]; then
  EXTRA_ARGS+=(--base-sifs-dir "$BASE_SIFS_DIR")
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG:-logs/gen_solutions_terminal.log}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

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
