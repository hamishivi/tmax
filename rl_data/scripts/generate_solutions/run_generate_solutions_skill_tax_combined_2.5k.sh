#!/bin/bash
#SBATCH --job-name=rl-gen-sol-stx-comb-2.5k
#SBATCH --output=logs/gen_sol_stx_comb_2.5k_%j.out
#SBATCH --error=logs/gen_sol_stx_comb_2.5k_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the COMBINED 2.5k SFT corpus  ║
# ║  (rl_data/output/tasks_skill_tax_combined_20260506_2.5k), which is   ║
# ║  a balanced mix of:                                                   ║
# ║    * 1031 legacy tasks (tasks_skill_tax_20260505_1k_legacy) — pure    ║
# ║      exact_text + text_only, evenly split short / moderate / complex  ║
# ║    * 1469 v2 tasks      (tasks_skill_tax_v2_20260505_2k) — sft_v2 mix ║
# ║      with verifier_kind / fixture_kind / intricate axes; intricate    ║
# ║      down-sampled from 1382→798 to keep the bucket roughly balanced.  ║
# ║  Final task_complexity split: short=22% / moderate=23% / complex=23%  ║
# ║  / intricate=32% across 2500 tasks (see _combine_manifest.json).      ║
# ║                                                                        ║
# ║  These trajectories feed the SFT data we use to teach Qwen3.x-{4B,9B} ║
# ║  the agent loop. Combined corpus replaces the legacy 1k corpus in   ║
# ║  this slot.                                                          ║
# ║                                                                        ║
# ║  Harness: vanillux (default; override via HARNESS=bash). Matches the ║
# ║  vanillux+Gemini sister script so both teacher models produce       ║
# ║  trajectories under the SAME harness — apples-to-apples for SFT mix. ║
# ║  Per-task summary lands at:                                          ║
# ║     <task>/solutions/<MODEL_TAG>_vanillux_summary.json               ║
# ║  (note the _vanillux suffix). Pass `--harness vanillux` to            ║
# ║  sft/preprocessing/convert_trajectories.py at conversion time.       ║
# ║                                                                        ║
# ║  Default model: Qwen/Qwen3.6-27B (override via VLLM_MODEL=...). The  ║
# ║  family detection in comparison/_vllm_local.sh picks the qwen3_coder ║
# ║  tool-call parser / qwen3 reasoning parser / --language-model-only.  ║
# ║                                                                        ║
# ║  Defaults are tuned for: ALL 2500 tasks × 8 solutions each, on a      ║
# ║  single 8xH200 allocation, with TP=1 DP=8 for the 27B (54 GB bf16    ║
# ║  params; H200 has 141 GB, so a single GPU holds the full model with   ║
# ║  ~66 GB free for KV cache). Going from TP=2 DP=4 to TP=1 DP=8 doubles ║
# ║  the inference replica count and removes the per-layer all-reduce.    ║
# ║  Combined with --enable-prefix-caching (helps agent loops where each  ║
# ║  turn's prompt is a prefix of the next) and MAX_TOKENS=4096 (per-turn  ║
# ║  output cap; was 65536, now matches what agents actually emit), this  ║
# ║  is roughly 2-3× the throughput of the pre-2026-05-06 config.         ║
# ║                                                                        ║
# ║  Sharding across two 8×H200 nodes (~1250 tasks per node, halving the  ║
# ║  wall time again) — both nodes write to the same TASKS_DIR safely:    ║
# ║                                                                        ║
# ║      # node A                                                         ║
# ║      START_AT=0    NUM_TASKS=1250 \                                   ║
# ║          bash run_generate_solutions_skill_tax_combined_2.5k.sh       ║
# ║      # node B                                                         ║
# ║      START_AT=1250 NUM_TASKS=1250 \                                   ║
# ║          bash run_generate_solutions_skill_tax_combined_2.5k.sh       ║
# ║                                                                        ║
# ║  No write contention: each task's solutions/ subdir is written by    ║
# ║  exactly one node. The combined corpus's interleaved sort order      ║
# ║  (legacy and v2 task names alternate by hash prefix) gives the two    ║
# ║  shards comparable complexity mixes.                                  ║
# ║                                                                        ║
# ║  Notes:                                                                ║
# ║   * The TASKS_DIR is a *symlink* folder: every task_* entry resolves  ║
# ║     back to either the legacy 1k or the v2 2k source corpus, so do   ║
# ║     NOT delete those source dirs while this run is in flight. Per-   ║
# ║     task summaries land in the source corpora (via symlink), not in   ║
# ║     the combined dir.                                                 ║
# ║   * v2-axis tasks (intricate complexity, non-legacy verifier_kind /  ║
# ║     fixture_kind) route to base_intricate.sif at solve time. Build   ║
# ║     it once on a build node before launching:                        ║
# ║         apptainer build rl_data/containers/base_intricate.sif \      ║
# ║                         rl_data/containers/base_intricate.def        ║
# ║   * Anonymous Docker Hub pulls are rate-limited (100 / 6 h / IP).    ║
# ║     APPTAINER_DOCKER_* creds are required.                           ║
# ║   * The trajectory output (per-task hosted_vllm_Qwen_Qwen3.6-27B_    ║
# ║     summary.json) is the input to                                    ║
# ║     `sft/preprocessing/convert_trajectories.py` — run that next.    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260505_2.2k_combined_balanced"

# This script is local-vLLM ONLY.  Default LAUNCH_VLLM=1 so users don't have
# to remember the flag, and default MODEL to the canonical hosted_vllm/...
# id that _vllm_wait_ready_local would otherwise rewrite it to.  Override
# via env to point at a different vLLM model (set both VLLM_MODEL + MODEL).
export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

# NUM_SOLUTIONS: 8 trajectories per task.  Critical for the SFT-data use case --
# the success-only variant of the SFT data is meaningfully different from the
# all variant only when there are multiple trajectories per task to filter.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# Harness: matches the vanillux+Gemini sister script for apples-to-apples
# teacher signal across the SFT corpus. Vanillux uses the mini-swe-agent
# "Recommended Workflow" prompt + 64-action budget + head/tail observation
# truncation. Override via env (e.g. HARNESS=bash for a legacy A/B).
# IMPORTANT: vanillux's per-task summary lands at
#   <task>/solutions/<MODEL_TAG>_vanillux_summary.json
# (note the _vanillux suffix). Pass --harness vanillux to
# sft/preprocessing/convert_trajectories.py at conversion time so it picks
# the right summary file.
HARNESS="${HARNESS:-vanillux}"
MAX_ACTIONS=64
# MAX_TOKENS = max output tokens PER TURN (per LLM call), not total. Real
# agent steps emit <2K tokens, so the previous 65536 was wildly over-
# reserving KV per sequence. 4096 leaves comfortable headroom for any
# single turn while letting vLLM pack many more concurrent sequences per
# replica. The helper's auto-cap (vllm_max_len/4) only kicks in when we
# request *more* than the cap, so 4096 < 10240 is honoured as-is.
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_TASKS="${NUM_TASKS:-999999}"     # cap on tasks processed (sharding hook)
START_AT="${START_AT:-0}"            # skip first N tasks (sharding hook)
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=180          # was 60 — v2 tasks need more headroom for package
                             # installs, vendored_package builds, multi_service
                             # boot, image/audio toolchain init.
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4              # mostly a no-op with BASE_SIFS_DIR (no per-task SIFs to build)
BUILD_RETRIES=3
# CRITICAL: skill-tax's per-task `container.def`s would each fresh-install
# python3+pip+pytest+gcc+gawk against archive.ubuntu.com if built from scratch,
# which thunder-herds under WORKERS=12 concurrent builds and trips the 300s
# build timeout on ~40% of tasks.  Instead we use the 9 pre-built domain base
# SIFs in rl_data/containers/ + the per-task setup.sh deltas (already extracted
# next to each task.json).  This is the SAME mode the original gemini 10k run
# used; see rl_data/scripts/generate_solutions/run_generate_solutions_10k.sh
# for the canonical pattern.
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# SAMPLE_SIZE=0 -> process all 2500 tasks in the combined corpus.
# Override via env if you want a quick smoke test.
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# WORKERS = concurrent TASKS at once.  NUM_POOL_WORKERS = concurrent solutions
# within a single task.  Both env-overridable.
WORKERS="${WORKERS:-24}"     # 24 × NUM_SOLUTIONS=8 = 192 ctr (3× CPU on 8×H200).
                             # NOTE: vLLM is the throughput ceiling here; going
                             # past 192 mostly queues at the server. Drop to 12
                             # for 4-GPU allocations.
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

# ---- vLLM tensor-parallel sizing for the 27B ----
# Qwen3.5-27B / Qwen3.6-27B at bf16 are ~54-56 GB of params; H200 has 141 GB,
# so the model FITS ON ONE GPU with ~66 GB free for KV cache + activations.
# TP=1 DP=8 on 8×H200 therefore gives 8 inference replicas (vs 4 with TP=2)
# AND saves the all-reduce overhead at every layer, which is throughput-
# optimal for agent-loop workloads at this concurrency. DP is auto-derived
# as visible_gpus / TP by _vllm_local.sh, so we only need to set TP here.
export VLLM_TP="${VLLM_TP:-1}"

# Prefix caching: agent loops have prompts that grow turn-by-turn, where
# turn N+1's prompt is `turn N's prompt + new observation`. With caching,
# only the new tokens are prefilled (vs re-prefilling the full growing
# history every turn — quadratic in n_turns). Default ON; set
# VLLM_PREFIX_CACHE=0 to opt out if a corner case ever breaks.
export VLLM_PREFIX_CACHE="${VLLM_PREFIX_CACHE:-1}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"

cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up.  No-op unless LAUNCH_VLLM=1.  Helper auto-detects
# the Qwen3.5+ arch family (matches `qwen3.[5-9]` case-insensitive, so it
# picks up Qwen3.5 / Qwen3.6 / future Qwen3.x) and switches to the
# qwen3_coder tool-call parser + qwen3 reasoning parser + --language-model-only
# flag.  Sourced from comparison/ since that's where the canonical helper
# lives (one source of truth for all solve scripts).
# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

# Apptainer Docker Hub creds: required because skill-tax per-task defs use
# Bootstrap: docker From: ubuntu:22.04 directly, and 1k concurrent anonymous
# pulls would exceed the 100 / 6 h / IP rate limit.  Same pattern as ET.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Local-model support via litellm env passthrough.  Harmless if unset.
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Keep apptainer instance logs off GPFS; heal dangling symlinks left behind
# when /tmp was cleaned between runs.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# Block until the in-job vLLM server is ready (no-op when LAUNCH_VLLM!=1).
# Also (re)exports MODEL/HOSTED_VLLM_API_BASE/OPENAI_API_KEY for the solver,
# and auto-caps MAX_TOKENS to fit the 27B's max_model_len (default 40960 ->
# cap at ~10240 leaving 30720 for the prompt + history).
_vllm_wait_ready_local

# Derive model-tagged paths NOW (after the helper may have rewritten MODEL).
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

echo "=== skill_tax combined 2.5k SFT-data run: MODEL=${MODEL}, HARNESS=${HARNESS}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
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
