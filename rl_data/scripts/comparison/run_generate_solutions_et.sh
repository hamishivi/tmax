#!/bin/bash
#SBATCH --job-name=rl-gen-sol-et
#SBATCH --output=logs/gen_sol_et_%j.out
#SBATCH --error=logs/gen_sol_et_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) Endless-     ║
# ║  Terminals dataset, using THE SAME model+settings as our 10k run so  ║
# ║  head-to-head comparison is apples-to-apples.                        ║
# ║                                                                       ║
# ║  Prerequisite: run ingest_endless_terminals.py to populate TASKS_DIR. ║
# ║  ET does NOT use shared base SIFs (its container.defs are self-       ║
# ║  contained), so per-task SIF builds happen lazily.                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_endless_terminals"
# Override via env.  Examples:
#   API model:   MODEL="gemini/gemini-3-flash-preview"
#   Local vLLM:  MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct" \
#                HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
#   Ollama:      MODEL="ollama_chat/qwen2.5-coder:7b" \
#                OLLAMA_API_BASE="http://localhost:11434"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
NUM_SOLUTIONS=1              # match 10k run
MAX_ACTIONS=16
MAX_TOKENS=65536
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=60
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4
BUILD_RETRIES=3
# NOTE: no BASE_SIFS_DIR — ET tasks don't share the 9 prebuilt bases.
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Optional cost-bounded subsample. Set SAMPLE_SIZE=250 (or similar) in the
# environment to randomly pick N tasks rather than processing the whole set.
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

WORKERS=12
NUM_POOL_WORKERS=16
# --------------------------------

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Local-model support via litellm env passthrough.
# These are harmless if unset (API model is used instead).
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"  # some local servers don't care but litellm checks
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  mkdir -p /tmp/apptainer_instances
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ "${SAMPLE_SIZE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== ET comparison run: WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
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
