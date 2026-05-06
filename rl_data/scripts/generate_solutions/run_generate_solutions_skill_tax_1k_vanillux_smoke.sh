#!/bin/bash
# SBATCH directives left in place so this script can also run via sbatch on
# clusters where that's preferred. On an interactive node, just `bash` it
# directly — the SBATCH lines are bash comments and are ignored.
#SBATCH --job-name=rl-vlx-smoke-50
#SBATCH --output=logs/vlx_smoke_50_%j.out
#SBATCH --error=logs/vlx_smoke_50_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Harness A/B smoke — bash vs vanillux on a 25-task sample                  ║
# ║                                                                            ║
# ║  Purpose: shake out the new --harness vanillux code path (mini-swe-agent-  ║
# ║  style bash-tool harness with vendored prompts + 64-action budget) end-to- ║
# ║  end on a small, well-known set of tasks BEFORE committing to a multi-day  ║
# ║  v2 RL solution-sampling run.                                              ║
# ║                                                                            ║
# ║  Harness shape (post-pivot to vendored mini-swe-agent prompts):            ║
# ║    bash      — terse system prompt (sft/preprocessing/config/system_prompt ║
# ║                .txt), single bash tool, 16-action default.                 ║
# ║    vanillux  — mini-swe-agent's "Recommended Workflow" prompt              ║
# ║                (rl_data/generator/vanillux_prompts.yaml; vendored from     ║
# ║                upstream), single bash tool (same), 64-action default,      ║
# ║                head/tail observation truncation. No SIF dependencies; the  ║
# ║                agent loop runs in our solver process and only bash actions ║
# ║                cross into the container.                                   ║
# ║                                                                            ║
# ║  Approach: re-uses an EXISTING task corpus (no fresh task gen), random-    ║
# ║  samples 25 tasks (--sample-size 25 --sample-seed 0), runs k=4 solutions  ║
# ║  per task (was k=8 — halved to keep iteration fast). The default          ║
# ║  TASKS_DIR points at the new v2 SFT 2k corpus — which exercises both     ║
# ║  axes:                                                                     ║
# ║    1. v2 fixture / verifier / intricate-complexity routing at solve time   ║
# ║       (since v2 tasks have base_image=intricate set in task.json).         ║
# ║    2. The harness change itself.                                           ║
# ║                                                                            ║
# ║  vLLM context: VLLM_MAX_LEN=131072 (128K) by default. The pre-v2 default   ║
# ║  of 40960 was too tight — 64-turn intricate trajectories blew past 30K    ║
# ║  input tokens by mid-run and tripped 400-Bad-Request from vLLM. Override  ║
# ║  via VLLM_MAX_LEN if you need to dial it down for a tighter memory job.   ║
# ║                                                                            ║
# ║  How summaries are kept apart:                                             ║
# ║    rl_data.generate_solutions writes summaries to                         ║
# ║      <task>/solutions/<MODEL_TAG>[_<HARNESS>]_summary.json                 ║
# ║    The harness suffix is OMITTED for HARNESS=bash (so legacy summaries    ║
# ║    in skill_tax 1k / 10k stay valid) and INCLUDED for HARNESS=vanillux.   ║
# ║    A bash run + a vanillux run on the same task therefore produce         ║
# ║    side-by-side files, no overwriting, no cp-r dance needed.              ║
# ║                                                                            ║
# ║  Recommended sequence (interactive bash on an h200 node):                  ║
# ║    HARNESS=bash     bash rl_data/scripts/generate_solutions/run_..._smoke.sh
# ║    HARNESS=vanillux bash rl_data/scripts/generate_solutions/run_..._smoke.sh
# ║                                                                            ║
# ║  No --force-rerun needed between the two passes: the harness-suffixed     ║
# ║  filenames don't collide. Tasks that already have the *current run's*     ║
# ║  summary file are skipped, so re-invoking the same HARNESS is a cheap     ║
# ║  no-op.                                                                    ║
# ║                                                                            ║
# ║  Same teacher model topology as the legacy SFT script                     ║
# ║  (run_generate_solutions_skill_tax_combined_2.5k.sh) — Qwen3.6-27B local  ║
# ║  vLLM,                                                                    ║
# ║  TP=2 DP=4 on 8×H200. This way the smoke matches the deployment harness  ║
# ║  on apples-to-apples teacher-side, isolating the harness change.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
# v2 SFT 2k by default; override to the legacy 1k or 10k corpus to A/B
# against unchanged-axes tasks.
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_v2_20260505_2k}"
HARNESS="${HARNESS:-vanillux}"          # 'bash' or 'vanillux'
OUT_TAG="${OUT_TAG:-${HARNESS}_smoke}"  # for log file naming only
SAMPLE_SIZE="${SAMPLE_SIZE:-25}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

# 8-attempt pass@k for the smoke (matches the full corpus runs). pass@1 +
# pass@8 are the two numbers we read off the final tally section.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# Apples-to-apples step budget across both harnesses for the smoke A/B.
# v2 tasks (intricate-complexity) routinely need >16 turns, and we want a
# single number that exercises both harnesses fairly. 64 is generous enough
# for vanillux's str_replace_editor + submit loop, while still bounding bash
# runs to a similar wall-clock ceiling.
MAX_ACTIONS="${MAX_ACTIONS:-64}"

# Pre-v2 default (40960) was too tight: 64-turn intricate runs commonly
# blow past 30K input tokens by the late turns, hitting the model-context
# ceiling and 400-Bad-Request from vLLM. Qwen3.6-27B's native context is
# 262144; 131072 (128K) leaves comfortable headroom for the longest agent
# trajectories without paying the throughput hit of going wider.
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-131072}"

# Per-turn output cap. Keep small so a single LLM step can't claim a huge
# slice of the context window; 8192 is plenty for one bash command + thought.
# Note: _vllm_wait_ready_local further auto-caps this to ~vllm_max_len/4.
MAX_TOKENS="${MAX_TOKENS:-8192}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=600            # was 180 — v2 setup.sh (apt + pip + large
                               # builds) routinely exceeds 180s under disk/
                               # network contention from 8 parallel solutions
                               # per task. 600s drops timeout-recovery
                               # overhead substantially.
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
# BUILD_WORKERS only matters when missing base SIFs need to be built. With
# BASE_SIFS_DIR set and all 10 base SIFs already present, this is a no-op.
# Bumped from 4 to 8 so future fresh builds on a multi-base node also
# parallelise cleanly.
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
# h200 nodes give 8 CPUs + ~240 GB RAM per GPU, so an 8×H200 allocation has
# 64 CPUs + ~1.28 TB RAM. With NUM_SOLUTIONS=8 the smoke fits
#   24 workers × 8 = 192 containers (~3× CPU oversubscription).
# That's well above the 1.5× I/O-bound rule but vLLM is the actual throughput
# ceiling for local-Qwen smokes anyway, so client-side concurrency mostly just
# fills vLLM's queue. Drop WORKERS=12 on a 4-GPU/32-CPU allocation.
WORKERS="${WORKERS:-24}"
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

# Final tally — read the summaries we just produced and report aggregate
# pass@k. This is the single most useful number from the smoke run: if
# vanillux's pass@1 is comparable to (or higher than) the legacy bash
# baseline on the same tasks, the new harness is healthy. The summary file
# matches the harness via _summary_basename in rl_data.generate_solutions:
#   bash      -> <MODEL_TAG>_summary.json
#   vanillux  -> <MODEL_TAG>_vanillux_summary.json
echo
echo "=== Summary across the ${SAMPLE_SIZE} sampled tasks (harness=${HARNESS}) ==="
uv run python <<PYEOF
import json, math, glob, os, random

random.seed($SAMPLE_SEED)
all_dirs = sorted(d for d in glob.glob("$TASKS_DIR/task_*") if os.path.isdir(d))
sample = random.sample(all_dirs, min($SAMPLE_SIZE, len(all_dirs)))
sample = sorted(sample)

model_tag = "$MODEL".replace("/", "_")
harness = "$HARNESS"
# Mirror rl_data.generate_solutions._summary_basename: bash keeps the legacy
# filename, non-bash gets a harness suffix.
suffix = "" if harness == "bash" else f"_{harness}"
summary_name = f"{model_tag}{suffix}_summary.json"

# k for pass@k is dynamic — we report pass@N where N = the actual number of
# solution attempts run for this task. The script-level NUM_SOLUTIONS is just
# the upper bound; if some env-init failure shrunk a task's effective N, we
# pick the per-task min so the math stays correct (and well-labelled).
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
    # pass@n unbiased estimator. With k = n, this collapses to "any succeeded".
    sum_passk += 1.0 if c >= n else (1.0 - math.comb(n - c, n) / math.comb(n, n))
    if c > 0:
        solved_some += 1

# Label pass@k with the modal k actually present in the run.
k_label = max(ks_observed) if ks_observed else 0
print(f"  summary file   : {summary_name}")
print(f"  evaluated      : {n_eval} / {len(sample)}  (skipped {n_skipped})")
if n_eval:
    print(f"  mean pass@1    : {sum_pass1 / n_eval:.3f}")
    print(f"  mean pass@{k_label}    : {sum_passk / n_eval:.3f}")
    print(f"  pass@{k_label} > 0     : {solved_some} / {n_eval}  ({solved_some / n_eval:.1%})")
PYEOF
