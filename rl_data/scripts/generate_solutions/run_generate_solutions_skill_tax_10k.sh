#!/bin/bash
#SBATCH --job-name=rl-gen-sol-stx-10k
#SBATCH --output=logs/gen_sol_stx_10k_%j.out
#SBATCH --error=logs/gen_sol_stx_10k_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on tasks_skill_tax_20260401_10k  ║
# ║  (9468 tasks, our reference 10k dataset) under a LOCAL Qwen3.5-9B    ║
# ║  served in-job via vLLM.  This produces apples-to-apples pass@k     ║
# ║  numbers vs the existing gemini-3-flash-preview baseline that lives   ║
# ║  alongside in solutions/gemini_*_summary.json.                        ║
# ║                                                                        ║
# ║  Defaults are tuned to MATCH the comparison runs we already did for   ║
# ║  ET / OpenThoughts / TermiGen / TerminalTraj:                         ║
# ║    SAMPLE_SIZE=500 SAMPLE_SEED=0  -> same fixed 500-task subset       ║
# ║    NUM_SOLUTIONS=8                -> pass@1..pass@8                   ║
# ║    Qwen3.5-9B at TP=1 DP=8 on 8xH200 (helper auto-detects 8 GPUs)    ║
# ║                                                                        ║
# ║  Override SAMPLE_SIZE=0 to run the full 10k (~20 h on this hardware). ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260401_10k"

# This script is local-vLLM ONLY.  Default LAUNCH_VLLM=1 so users don't have
# to remember the flag, and default MODEL to the canonical hosted_vllm/...
# id that _vllm_wait_ready_local would otherwise rewrite it to.  Override
# via env to point at a different vLLM model (set both VLLM_MODEL + MODEL).
export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
MAX_ACTIONS=16
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=600          # was 180 — heavy compile / install / setup.sh
                             # commands (apt-install, large Go/Rust/C++ builds,
                             # numpy/Pillow wheel pulls) commonly exceed 180s
                             # under disk + network contention. 600s drops the
                             # timeout-recovery overhead substantially.
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4              # mostly a no-op with BASE_SIFS_DIR (no per-task SIFs to build)
BUILD_RETRIES=3
# Use the 9 pre-built domain base SIFs in rl_data/containers/ + per-task
# setup.sh deltas instead of building per-task SIFs from scratch.  Without
# this, every task's container.def would fresh-install python3+pip+gcc+gawk
# against archive.ubuntu.com, thunder-herding the build pipeline.
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# SAMPLE_SIZE=500 matches the comparison runs (ET/OT/TG/TT) for direct
# apples-to-apples vs gemini.  Override to 0 for the full 10k.
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

WORKERS="${WORKERS:-24}"     # 24 × NUM_SOLUTIONS=8 = 192 ctr (3× CPU on 8×H200).
                             # NOTE: vLLM is the throughput ceiling here; going
                             # past 192 mostly queues at the server. Drop to 12
                             # for 4-GPU allocations.
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

# ---- vLLM defaults: identical to the ET 500-task run ----
# Qwen3.5-9B at bf16 = 18 GB params; TP=1 DP=8 on 8xH200 fits easily.
# (helper picks DP = visible_gpus / TP automatically; TP defaults to 1.)
# VLLM_MODEL is already set above (right next to MODEL) so the helper banner
# and MODEL stay in sync; nothing else to set here.
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"

cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up.  No-op unless LAUNCH_VLLM=1.  Helper auto-picks the
# qwen3_coder tool-call parser + qwen3 reasoning parser + --language-model-only
# flag for Qwen3.5.
# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

# Apptainer Docker Hub creds: required because skill-tax per-task defs use
# Bootstrap: docker From: ubuntu:22.04 directly, and N concurrent anonymous
# pulls would hit the 100 / 6 h / IP limit.  Same pattern as ET.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Local-model support via litellm env passthrough.
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

echo "=== skill_tax 10k comparison run: MODEL=${MODEL}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
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
