#!/bin/bash
# SBATCH directives left in place so this script can also run via sbatch on
# clusters where that's preferred. On an interactive node, just `bash` it
# directly — the SBATCH lines are bash comments and are ignored.
#SBATCH --job-name=rl-vlx-gem-smoke-25
#SBATCH --output=logs/vlx_gem_smoke_25_%j.out
#SBATCH --error=logs/vlx_gem_smoke_25_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Harness smoke — vanillux × Gemini-API on a 25-task sample                 ║
# ║                                                                            ║
# ║  Sister script of run_generate_solutions_skill_tax_1k_vanillux_smoke.sh    ║
# ║  but uses Google's Gemini API (`gemini/gemini-3-flash-preview`) via        ║
# ║  litellm instead of an in-job vLLM-hosted Qwen3.x. Useful when:            ║
# ║                                                                            ║
# ║    1. You want to compare the vanillux harness across teacher models       ║
# ║       (frontier API teacher vs. local Qwen) — same harness, same tasks,   ║
# ║       same prompts, only the LLM endpoint changes.                         ║
# ║    2. You want a GPU-light smoke iteration: GPUs aren't actually used      ║
# ║       (Gemini lives behind a network API), so you can run on whatever      ║
# ║       smallest h200 allocation gives you ~32-64 CPUs for concurrent        ║
# ║       containers. (See SBATCH alternatives at the bottom of the comment    ║
# ║       block.)                                                              ║
# ║    3. You're scoping out how the v2 corpus tasks land before paying for    ║
# ║       a full Qwen-served run.                                              ║
# ║                                                                            ║
# ║  Required env vars:                                                        ║
# ║    GEMINI_API_KEY            — Google AI Studio key, picked up by litellm  ║
# ║    APPTAINER_DOCKER_USERNAME — Docker Hub creds; per-task bases need them  ║
# ║    APPTAINER_DOCKER_PASSWORD                                                ║
# ║                                                                            ║
# ║  Approach: re-uses an EXISTING task corpus (no fresh task gen), random-    ║
# ║  samples 25 tasks (--sample-size 25 --sample-seed 0), runs k=4 solutions   ║
# ║  per task (matches the local-vLLM smoke for apples-to-apples). Default     ║
# ║  TASKS_DIR points at the v2 SFT 2k corpus.                                 ║
# ║                                                                            ║
# ║  How summaries are kept apart from local-vLLM runs:                        ║
# ║    rl_data.generate_solutions writes summaries to                          ║
# ║      <task>/solutions/<MODEL_TAG>[_<HARNESS>]_summary.json                 ║
# ║    The model tag includes the provider prefix (e.g. ``gemini_gemini-3-     ║
# ║    flash-preview``), so a local-Qwen vanillux summary and a Gemini         ║
# ║    vanillux summary on the same task land side-by-side as different files. ║
# ║                                                                            ║
# ║  SBATCH allocation — pick the line that matches your wall-clock budget.    ║
# ║  GPUs are reserved only because they bring CPUs+RAM with them on h200      ║
# ║  nodes; vanillux+Gemini does NOT use GPU at all.                           ║
# ║    8 GPUs: 64 CPUs / ~960 GB RAM  → WORKERS=24 NUM_SOLUTIONS=8 = 192 ctr   ║
# ║    4 GPUs: 32 CPUs / ~480 GB RAM  → WORKERS=12 NUM_SOLUTIONS=8 = 96 ctr    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Option A: 8 GPUs (default, faster) ──
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ── Option B: 4 GPUs (lighter, ~half the throughput) ──
# #SBATCH --gres=gpu:h200:4
# #SBATCH --cpus-per-task=32
# #SBATCH --mem=480G

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_v2_20260505_2k}"
HARNESS="${HARNESS:-vanillux}"          # 'bash' or 'vanillux'
OUT_TAG="${OUT_TAG:-gemini_${HARNESS}_smoke}"   # for log file naming only
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# Gemini API model. Override via env if you want to A/B against a different
# Gemini variant (e.g. gemini-3.1-pro-preview for the larger model).
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"

# 8-attempt pass@k for the smoke (matches the full-corpus runs). pass@1 +
# pass@8 are the two numbers we read off the final tally section.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# Apples-to-apples step budget across both harnesses for the smoke A/B.
# v2 tasks (intricate-complexity) routinely need >16 turns; 64 lines up with
# upstream mini-swe-agent's per-instance call limit and the local-vLLM smoke.
MAX_ACTIONS="${MAX_ACTIONS:-64}"

# Per-turn output cap. Gemini's context is 1M tokens so we don't have the
# tight 128K cap the local Qwen smoke had to dance around. 65536 matches the
# legacy 10k Gemini script (run_generate_solutions_10k.sh) for consistency.
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=600           # was 180 — v2 setup.sh (apt + pip + large
                              # builds) routinely exceeds 180s under disk/
                              # network contention from 8 parallel solutions
                              # per task. 600s drops timeout-recovery
                              # overhead substantially.
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
# BUILD_WORKERS only matters when missing base SIFs need to be built. With
# BASE_SIFS_DIR set and all base SIFs already present, this is a no-op.
BUILD_WORKERS="${BUILD_WORKERS:-8}"
BUILD_RETRIES=3
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Concurrency model:
#   * WORKERS           = concurrent TASKS at once.
#   * NUM_POOL_WORKERS  = concurrent solutions / shell ops *within* one task.
#                         Must be >= NUM_SOLUTIONS for full parallelism.
#   * Total concurrent containers = WORKERS * NUM_SOLUTIONS.
#
# h200 nodes give 8 CPUs + ~120 GB RAM per GPU. Defaults below target the
# 8-GPU allocation (64 CPUs); halve WORKERS for a 4-GPU allocation. The
# 1.5x CPU oversubscription is fine because the agent loop is heavily
# I/O-bound (Gemini round-trip + apptainer exec, never CPU-pinned).
WORKERS="${WORKERS:-24}"          # 8 GPUs default; use 12 for 4 GPUs
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-8}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# Gemini API key — required. Fail fast with a clear error if unset.
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running (Google AI Studio key)}"
export GEMINI_API_KEY

# Apptainer Docker Hub creds: required because skill-tax per-task defs use
# Bootstrap: docker From: ubuntu:22.04 directly, and concurrent anonymous
# pulls would exceed the 100 / 6 h / IP rate limit.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Explicitly clear the local-vLLM passthrough vars so litellm uses Gemini's
# native endpoint, not whatever stale HOSTED_VLLM_API_BASE is in the shell.
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
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}_${OUT_TAG}_${_RUN_TS}.log"

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
EXTRA_ARGS+=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== Vanillux × Gemini smoke (25): MODEL=${MODEL}, HARNESS=${HARNESS}, MAX_ACTIONS=${MAX_ACTIONS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
echo "=== Tasks dir: ${TASKS_DIR} (sampling ${SAMPLE_SIZE} with seed ${SAMPLE_SEED}) ==="
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

# Final tally — read the summaries we just produced and report aggregate
# pass@k, mirroring the local-vLLM smoke. The summary file matches the
# (model, harness) pair via _summary_basename in rl_data.generate_solutions:
#   bash      -> <MODEL_TAG>_summary.json
#   vanillux  -> <MODEL_TAG>_vanillux_summary.json
echo
echo "=== Summary across the ${SAMPLE_SIZE} sampled tasks (model=${MODEL}, harness=${HARNESS}) ==="
uv run python <<PYEOF
import json, math, glob, os, random

random.seed($SAMPLE_SEED)
all_dirs = sorted(d for d in glob.glob("$TASKS_DIR/task_*") if os.path.isdir(d))
sample = random.sample(all_dirs, min($SAMPLE_SIZE, len(all_dirs)))
sample = sorted(sample)

model_tag = "$MODEL".replace("/", "_")
harness = "$HARNESS"
suffix = "" if harness == "bash" else f"_{harness}"
summary_name = f"{model_tag}{suffix}_summary.json"

n_eval = 0
sum_pass1 = 0.0
sum_passk = 0.0
n_skipped = 0
solved_some = 0
ks_observed = set()
for d in sample:
    p = os.path.join(d, "solutions", summary_name)
    if not os.path.exists(p):
        n_skipped += 1
        continue
    s = json.load(open(p))
    n = s.get("num_runs", 0)
    c = s.get("num_success", 0)
    if n == 0:
        n_skipped += 1
        continue
    ks_observed.add(n)
    n_eval += 1
    sum_pass1 += c / n
    sum_passk += 1.0 if c >= n else (1.0 - math.comb(n - c, n) / math.comb(n, n))
    if c > 0:
        solved_some += 1

k_label = max(ks_observed) if ks_observed else 0
print(f"  summary file   : {summary_name}")
print(f"  evaluated      : {n_eval} / {len(sample)}  (skipped {n_skipped})")
if n_eval:
    print(f"  mean pass@1    : {sum_pass1 / n_eval:.3f}")
    print(f"  mean pass@{k_label}    : {sum_passk / n_eval:.3f}")
    print(f"  pass@{k_label} > 0     : {solved_some} / {n_eval}  ({solved_some / n_eval:.1%})")
PYEOF
