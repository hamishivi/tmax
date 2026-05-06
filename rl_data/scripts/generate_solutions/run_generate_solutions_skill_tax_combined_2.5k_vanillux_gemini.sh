#!/bin/bash
#SBATCH --job-name=rl-vlx-gem-comb-2.5k
#SBATCH --output=logs/vlx_gem_comb_2.5k_%j.out
#SBATCH --error=logs/vlx_gem_comb_2.5k_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Solution-generation under the vanillux harness (mini-swe-agent-     ║
# ║  style bash-tool harness with vendored prompts, 64-action budget) on ║
# ║  the COMBINED 2.5k SFT corpus                                        ║
# ║  (rl_data/output/tasks_skill_tax_combined_20260506_2.5k), using      ║
# ║  Gemini API as the teacher model.                                    ║
# ║                                                                       ║
# ║  The combined corpus is a balanced mix of:                           ║
# ║    * 1031 legacy tasks  (exact_text + text_only baseline)            ║
# ║    * 1469 v2 tasks      (verifier_kind / fixture_kind / intricate    ║
# ║                          axes; intricate down-sampled to 798 to keep ║
# ║                          the bucket roughly balanced)                ║
# ║  Final task_complexity split: 22/23/23/32 across short / moderate /  ║
# ║  complex / intricate. See _combine_manifest.json for the per-source  ║
# ║  breakdown.                                                          ║
# ║                                                                       ║
# ║  Sister script of:                                                   ║
# ║    * run_generate_solutions_skill_tax_combined_2.5k.sh               ║
# ║      (same corpus, bash harness + local Qwen3.6-27B via vLLM)        ║
# ║    * run_generate_solutions_skill_tax_1k_vanillux_smoke.sh           ║
# ║      (vanillux harness, but a 25-task smoke against local Qwen on    ║
# ║      the legacy 1k corpus)                                           ║
# ║    * run_generate_solutions_skill_tax_1k_vanillux_gemini_smoke.sh    ║
# ║      (same harness + same model, but a 25-task smoke)                ║
# ║                                                                       ║
# ║  When to use this script vs. the vLLM siblings:                      ║
# ║    * Use this when the teacher you actually want is Gemini.          ║
# ║    * Use the vLLM scripts when you need a local model (offline       ║
# ║      reproducibility, no per-token cost, deterministic across runs). ║
# ║                                                                       ║
# ║  Required env vars:                                                   ║
# ║    GEMINI_API_KEY            — Google AI Studio key                  ║
# ║    APPTAINER_DOCKER_USERNAME — Docker Hub creds                      ║
# ║    APPTAINER_DOCKER_PASSWORD                                          ║
# ║                                                                       ║
# ║  Output: per-task                                                    ║
# ║    <task>/solutions/gemini_gemini-3-flash-preview_vanillux_summary.json║
# ║  (no overlap with bash-harness or local-Qwen runs on the same tasks).║
# ║                                                                       ║
# ║  SBATCH allocation — pick the line that matches your wall budget.    ║
# ║  GPUs are reserved only because they bring CPUs+RAM with them on h200║
# ║  nodes; vanillux+Gemini does NOT use GPU at all.                     ║
# ║    8 GPUs: 64 CPUs / ~960 GB RAM  → WORKERS=24 NUM_SOLUTIONS=8 = 192 ║
# ║    4 GPUs: 32 CPUs / ~480 GB RAM  → WORKERS=12 NUM_SOLUTIONS=8 = 96  ║
# ║                                                                       ║
# ║  Sharding across two nodes (~1250 tasks per node, ~2× faster overall):║
# ║                                                                       ║
# ║      # node A                                                         ║
# ║      START_AT=0    NUM_TASKS=1250 \                                   ║
# ║          bash run_generate_solutions_skill_tax_combined_2.5k_vanillux_gemini.sh
# ║      # node B                                                         ║
# ║      START_AT=1250 NUM_TASKS=1250 \                                   ║
# ║          bash run_generate_solutions_skill_tax_combined_2.5k_vanillux_gemini.sh
# ║                                                                       ║
# ║  Both nodes write into the same TASKS_DIR — each task's solutions/    ║
# ║  subdir is touched by exactly one node. The combined corpus's        ║
# ║  interleaved sort order gives the two shards comparable complexity   ║
# ║  mixes.                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Option A: 8 GPUs (default, faster) ──
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
WORKERS="${WORKERS:-24}"          # 24 tasks × 8 solutions = 192 containers (3× over 64 CPUs).
                                  # Vanillux+Gemini is heavily I/O-bound, so 3×
                                  # CPU oversubscription is fine.

# ── Option B: 4 GPUs (lighter, ~half the throughput) ──
# #SBATCH --gres=gpu:h200:4
# #SBATCH --cpus-per-task=32
# #SBATCH --mem=480G
# WORKERS="${WORKERS:-12}"          # 12 tasks × 8 solutions = 96 containers (3× over 32 CPUs)

set -euo pipefail

# ---- Parameters (edit here) ----
# Default to the combined 2.5k SFT corpus. Override to the legacy 1k /
# 10k or v2 2k corpus to A/B against an unmixed-axes corpus.
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_20260505_2.2k_combined_balanced}"

HARNESS="${HARNESS:-vanillux}"         # 'bash' or 'vanillux'

# Gemini API model. Override via env to A/B against gemini-3.1-pro-preview.
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"

# 8 trajectories per task — same as the bash-harness 1k script. Critical
# for SFT data quality (success-only filtering needs multiple attempts to
# differ meaningfully from "any attempt").
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# Vanillux's mini-swe-agent-style "Recommended Workflow" prompt benefits
# from a 64-action budget; lines up with upstream's per_instance_call_limit.
MAX_ACTIONS="${MAX_ACTIONS:-64}"

# Gemini context is 1M tokens; 65536 matches the legacy 10k Gemini script.
# (Gemini doesn't have vLLM's per-sequence KV-reservation cost, so the high
# cap here is essentially free — there's no throughput penalty to leaving it.)
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS="${NUM_TASKS:-999999}"     # cap on tasks processed (sharding hook)
START_AT="${START_AT:-0}"            # skip first N tasks (sharding hook)
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=180           # was 60 — v2 tasks need more headroom for package
                              # installs, vendored_package builds, multi_service
                              # boot, image/audio toolchain init.
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS="${BUILD_WORKERS:-8}"
BUILD_RETRIES=3
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# SAMPLE_SIZE=0 -> process all tasks (default for the full run). Override
# to a positive int to cap (mostly for debugging — for the 25-task smoke,
# use the dedicated *_vanillux_gemini_smoke.sh script).
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# NUM_POOL_WORKERS = concurrent solutions / shell ops within a single task.
# Must be >= NUM_SOLUTIONS for full parallelism within a task.
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# Gemini API key — required.
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running (Google AI Studio key)}"
export GEMINI_API_KEY

# Apptainer Docker Hub creds — required for skill-tax per-task base pulls.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Clear local-vLLM passthrough vars so litellm uses Gemini's native endpoint,
# not whatever stale HOSTED_VLLM_API_BASE is in the shell.
unset HOSTED_VLLM_API_BASE OLLAMA_API_BASE OPENAI_API_BASE OPENAI_API_KEY 2>/dev/null || true

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}_${_RUN_TS}.log"

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

echo "=== Vanillux × Gemini × combined-2.5k run: MODEL=${MODEL}, HARNESS=${HARNESS}, MAX_ACTIONS=${MAX_ACTIONS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
echo "=== Tasks dir: ${TASKS_DIR}  (SAMPLE_SIZE=${SAMPLE_SIZE}; 0 = all) ==="
echo "=== Concurrent containers: $(( WORKERS * NUM_SOLUTIONS ))  (WORKERS=${WORKERS} × NUM_SOLUTIONS=${NUM_SOLUTIONS}) ==="

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
