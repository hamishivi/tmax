#!/bin/bash
#SBATCH --job-name=rl-vlx-smoke-50
#SBATCH --output=logs/vlx_smoke_50_%j.out
#SBATCH --error=logs/vlx_smoke_50_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Smoke test: VanilluxAgent harness on 50 samples of the existing 1k corpus.║
# ║                                                                            ║
# ║  Purpose: shake out the new --harness vanillux code path (str_replace_     ║
# ║  editor / submit / ATIF dump) end-to-end on a small, well-known set of    ║
# ║  tasks BEFORE we commit to a multi-day v2 RL solution-sampling run.       ║
# ║                                                                            ║
# ║  Approach: re-uses the existing skill_tax 1k corpus (no fresh task gen),  ║
# ║  random-samples 50 tasks (--sample-size 50 --sample-seed 0), and runs    ║
# ║  k=8 solutions per task with the vanillux harness.                        ║
# ║                                                                            ║
# ║  How to compare against the legacy bash harness:                          ║
# ║    1. First run THIS script with HARNESS=bash and a different OUT_TAG     ║
# ║       (export HARNESS=bash OUT_TAG=bash) to produce baseline summaries.   ║
# ║    2. Then run with HARNESS=vanillux OUT_TAG=vanillux for the new path.   ║
# ║    3. Both runs will write per-task                                       ║
# ║         <task>/solutions/<MODEL_TAG>_summary.json                         ║
# ║       — note the same model tag, so RUN1 and RUN2 race for the same       ║
# ║       filename. To keep both, set FORCE_RERUN=1 on the second run AND    ║
# ║       set OUT_TAG to a different value so the analyzer picks them apart. ║
# ║    4. Or simpler: copy the 50 surviving summaries off-corpus before the  ║
# ║       second run.                                                         ║
# ║                                                                            ║
# ║  Recommended sequence for clean A/B:                                      ║
# ║    HARNESS=bash     bash run_generate_solutions_skill_tax_1k_vanillux_smoke.sh
# ║    cp -r rl_data/output/tasks_skill_tax_20260324_1k/<task>/solutions \    ║
# ║          /tmp/bash_baseline/<task>/                                       ║
# ║    HARNESS=vanillux FORCE_RERUN=1 bash ... vanillux_smoke.sh              ║
# ║                                                                            ║
# ║  Same teacher model and SBATCH topology as the legacy SFT script          ║
# ║  (run_generate_solutions_skill_tax_1k.sh) — Qwen3.6-27B local vLLM,       ║
# ║  TP=2 DP=4 on 8×H200. This way the smoke matches the deployment harness  ║
# ║  on apples-to-apples teacher-side, isolating the harness change.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_20260505_1k_legacy}"
HARNESS="${HARNESS:-vanillux}"          # 'bash' or 'vanillux'
OUT_TAG="${OUT_TAG:-vanillux_smoke}"    # for log file naming only
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
# vanillux's intricate prompts can use more turns; bash legacy stays at 16.
if [ "$HARNESS" = "vanillux" ]; then
    MAX_ACTIONS="${MAX_ACTIONS:-60}"
else
    MAX_ACTIONS="${MAX_ACTIONS:-16}"
fi
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=60
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4
BUILD_RETRIES=3
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

export VLLM_TP="${VLLM_TP:-2}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"

cd "$PROJECT_ROOT"
mkdir -p logs

# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

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

echo "=== Vanillux smoke (50): MODEL=${MODEL}, HARNESS=${HARNESS}, MAX_ACTIONS=${MAX_ACTIONS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
echo "=== Tasks dir: ${TASKS_DIR} (sampling ${SAMPLE_SIZE} with seed ${SAMPLE_SEED}) ==="
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

# Final tally — read the 50 summaries we just produced and report aggregate
# pass@k. This is the single most useful number from the smoke run: if
# vanillux's pass@1 on the existing 1k corpus is comparable to the legacy
# bash run, the new harness is healthy.
echo
echo "=== Summary across the ${SAMPLE_SIZE} sampled tasks ==="
uv run python <<PYEOF
import json, math, glob, os, random

random.seed($SAMPLE_SEED)
all_dirs = sorted(d for d in glob.glob("$TASKS_DIR/task_*") if os.path.isdir(d))
sample = random.sample(all_dirs, min($SAMPLE_SIZE, len(all_dirs)))
sample = sorted(sample)

model_tag = "$MODEL".replace("/", "_")
summary_name = f"{model_tag}_summary.json"

n_eval = 0
n_pass1 = 0
n_pass8 = 0
n_skipped = 0
solved_some = 0
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
    n_eval += 1
    n_pass1 += c / n
    n_pass8 += 1 - math.comb(max(0, n-c), min(8, n)) / math.comb(n, min(8, n)) if c < n else 1.0
    if c > 0:
        solved_some += 1
print(f"  evaluated      : {n_eval} / {len(sample)}  (skipped {n_skipped})")
if n_eval:
    print(f"  mean pass@1    : {n_pass1 / n_eval:.3f}")
    print(f"  mean pass@8    : {n_pass8 / n_eval:.3f}")
    print(f"  pass@8 > 0     : {solved_some} / {n_eval}  ({solved_some / n_eval:.1%})")
PYEOF
