#!/bin/bash
#SBATCH --job-name=rl-gen-sol-sft-v2-1k
#SBATCH --output=logs/gen_sol_sft_v2_1k_%j.out
#SBATCH --error=logs/gen_sol_sft_v2_1k_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  v2 SFT solutions — pointed at tasks_skill_tax_v2_sft_1k.                  ║
# ║                                                                            ║
# ║  This script is a thin re-pointing of the existing                         ║
# ║  ``run_generate_solutions_skill_tax_1k.sh``: same local Qwen3.6-27B vLLM   ║
# ║  teacher, same bash-tool-calling harness, same MAX_ACTIONS=16, same        ║
# ║  NUM_SOLUTIONS=8. The only change is TASKS_DIR. **No Vanillux harness**    ║
# ║  for the SFT corpus — see plan §6 for the rationale (SFT inherits the     ║
# ║  legacy local-vLLM SFT pipeline).                                          ║
# ║                                                                            ║
# ║  Tasks routed to base_intricate.sif at solve time (i.e. those with a       ║
# ║  non-legacy verifier_kind / fixture_kind / intricate complexity) require   ║
# ║  the prerequisite SIF to exist:                                            ║
# ║    apptainer build rl_data/containers/base_intricate.sif \                 ║
# ║                    rl_data/containers/base_intricate.def                   ║
# ║  The pre-build phase in this script auto-builds any missing base SIFs.    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_v2_sft_1k"

# Local-vLLM only (same as the legacy SFT script).
export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
MAX_ACTIONS=16
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=60
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4
BUILD_RETRIES=3
# Use pre-built base SIFs (per-domain + base_intricate.sif). The pre-build
# phase in generate_solutions.py will build any missing bases on entry.
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# v2 SFT keeps the legacy bash-tool-calling harness. Only the v2 RL script
# switches to ``--harness vanillux`` (see plan §4).
HARNESS="${HARNESS:-bash}"

SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

# ---- vLLM tensor-parallel sizing (mirrors run_generate_solutions_skill_tax_1k.sh) ----
export VLLM_TP="${VLLM_TP:-2}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"

cd "$PROJECT_ROOT"
mkdir -p logs

# Bring up the in-job vLLM server (no-op when LAUNCH_VLLM!=1).
# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

# Apptainer Docker Hub creds — required because v2 SFT def files use
# Bootstrap: docker From: ubuntu:22.04 directly. Same pattern as the legacy
# SFT script.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

_vllm_wait_ready_local

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"

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

echo "=== v2 SFT 1k run: MODEL=${MODEL}, HARNESS=${HARNESS}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
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
    --harness "$HARNESS" \
    --verbose \
    "${EXTRA_ARGS[@]}"
