#!/bin/bash
#SBATCH --job-name=rl-gen-sol-r2e
#SBATCH --output=logs/gen_sol_r2e_%j.out
#SBATCH --error=logs/gen_sol_r2e_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) R2E Gym       ║
# ║  (hamishivi/agent-task-r2e-gym) dataset under the VANILLUX harness,   ║
# ║  using the uniform comparison config (NUM_SOLUTIONS=8, MAX_ACTIONS=64,║
# ║  COMMAND_TIMEOUT=600, SAMPLE_SIZE=250, fixed SAMPLE_SEED=0).          ║
# ║                                                                        ║
# ║  Output: per-task                                                      ║
# ║    <task>/solutions/<MODEL_TAG>_vanillux_summary.json                  ║
# ║                                                                        ║
# ║  Prerequisite: run rl_data/scripts/comparison/run_ingest_r2e_gym.sh   ║
# ║  once to extract 8,101 tasks into TASKS_DIR (downloads + extracts the ║
# ║  14 MB task-data.tar.gz).                                              ║
# ║                                                                        ║
# ║  WRINKLES SPECIFIC TO R2E GYM:                                        ║
# ║                                                                        ║
# ║  1. Every task has a UNIQUE Docker Hub image                           ║
# ║     (namanjain12/<repo>_final:<commit>) -- 8,101 distinct images,    ║
# ║     ~400-800 MB compressed each. We cannot prebuild a shared base    ║
# ║     SIF; each per-task SIF starts from its own FROM layer.           ║
# ║                                                                        ║
# ║  2. R2E images bake the repo at the buggy commit at /testbed, with   ║
# ║     a project-local virtualenv at /testbed/.venv. The instruction    ║
# ║     itself tells the agent to ``cd /testbed && source .venv/bin/     ║
# ║     activate``; the harness lands the agent in /home/user (writable  ║
# ║     tmpfs) per the standard convention, and the agent navigates.     ║
# ║                                                                        ║
# ║  3. The adapter has already injected a robust pytest-bootstrap       ║
# ║     %post (the same one TerminalTraj uses) so the SIFs ship with a   ║
# ║     pytest on global PATH -- the harness's `pytest pytest_final_    ║
# ║     state.py` invocation runs OUTSIDE /testbed/.venv and needs that. ║
# ║                                                                        ║
# ║  4. Some R2E base images may have old glibc incompatible with        ║
# ║     Apptainer's bundled fakeroot. We pre-build with                  ║
# ║     --ignore-fakeroot-command (as TerminalTraj does); the harness's  ║
# ║     own build step then short-circuits because the SIF already       ║
# ║     exists.                                                            ║
# ║                                                                        ║
# ║  5. DISK: 8,101 SIFs × ~600 MB ≈ 4.9 TB. SAMPLE_SIZE=250 (uniform    ║
# ║     comparison default) caps this to ~150 GB.                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Load tmax secrets (GEMINI_API_KEY, Docker Hub creds, etc.) ---------
# When this script is launched via `sbatch` rather than `bash` inside an
# already-`source ~/.tmax_secrets`'d interactive shell, the sbatch job
# inherits a stripped login env that does NOT have GEMINI_API_KEY set,
# which causes generate_solutions to silently fall back to anonymous /
# unauthenticated requests (= 100% LLM-call failures over the entire wall
# time). Sourcing the secrets file here is idempotent.
if [[ -f "$HOME/.tmax_secrets" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.tmax_secrets"
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set; ensure ~/.tmax_secrets exists and is sourced}"
export GEMINI_API_KEY

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_r2e_gym"
# Override via env. Examples:
#   API model:   MODEL="gemini/gemini-3-flash-preview"
#   Local vLLM:  MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct" \
#                HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
# Solution-sampling harness. Every comparison baseline is vanillux by default
# under the 0515 rebase (see run_generate_solutions_et.sh header).
HARNESS="${HARNESS:-vanillux}"
# 8 attempts/task = pass@1/4/8 in one run. Uniform across all baselines.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
# 64 = vanillux convention; the mini-swe-agent prompts need the larger budget.
MAX_ACTIONS="${MAX_ACTIONS:-64}"
# MAX_TOKENS is the per-turn generation cap. Gemini default 65536; auto-capped
# to VLLM_MAX_LEN-safety_margin when LAUNCH_VLLM=1 (see _vllm_wait_ready_local).
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
# 600s matches the vanillux reference: R2E tasks involve a real pytest suite
# over a real repo (sympy/pandas/...), which can be heavy under parallelism.
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=12
BUILD_RETRIES=3
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Cost-bounded subsample: 250 tasks (uniform across all baselines). Fixed
# seed=0 so the same 250 R2E tasks are picked on every rerun. Disk budget:
# 250 × ~600 MB SIFs ≈ 150 GB.
SAMPLE_SIZE="${SAMPLE_SIZE:-250}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# WORKERS = concurrent TASKS processed at once. NUM_POOL_WORKERS = concurrent
# solutions/env operations within a single task. Both env-overridable.
WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

# Pre-build phase: how many SIFs to build in parallel. Each build pulls a
# ~600 MB Docker image, so mostly I/O-bound. 4-8 workers is the sweet spot;
# higher values saturate the shared apptainer cache on GPFS and the link.
PREBUILD_WORKERS="${PREBUILD_WORKERS:-4}"
PREBUILD_RETRIES="${PREBUILD_RETRIES:-2}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up. No-op unless LAUNCH_VLLM=1. We kick it off pre-prebuild
# so weight loading overlaps with the (slow) per-task SIF prebuild, then block
# on readiness right before the solver call.
# shellcheck source=./_vllm_local.sh
source "$SCRIPT_DIR/_vllm_local.sh"
_vllm_start_local

# Docker Hub creds are OPTIONAL for R2E Gym because every namanjain12 image is
# PUBLIC on Docker Hub. We still pass them through if set so anonymous pull
# rate limits (100 pulls/6h/IP) don't bite mid-run.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-}"
if [[ -z "$APPTAINER_DOCKER_USERNAME" ]]; then
  echo "WARN: APPTAINER_DOCKER_USERNAME is unset; anonymous Docker Hub pulls are" >&2
  echo "      rate-limited to 100 per 6h per IP. Set it to avoid mid-run throttling." >&2
fi

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

# Some R2E base images may ship a bundled /.singularity.d/libs/fakeroot
# whose embedded glibc is incompatible with the host (mirrors the
# TerminalTraj Fedora-27 failure mode), so solve-time
# `apptainer instance start --fakeroot` and `apptainer exec --fakeroot`
# crash. Setting this env var makes the harness add
# --ignore-fakeroot-command to every fakeroot invocation, falling back to
# user-namespace fakeroot emulation that we already use during the
# prebuild phase.
export APPTAINER_IGNORE_FAKEROOT_COMMAND="${APPTAINER_IGNORE_FAKEROOT_COMMAND:-1}"

# Keep apptainer instance logs off GPFS; heal dangling symlinks left behind
# when /tmp was cleaned between runs.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# ---- R2E Gym pre-build phase --------------------------------------------
# The harness's build_sif() in rl_data/generate_solutions.py does not pass
# --ignore-fakeroot-command, which some R2E images (old glibc) may require.
# We resolve this by pre-building the per-task SIFs HERE with the flag; the
# harness's pre-build then detects the existing container.sif and skips its
# own build step.
#
# Compute the list of tasks we're about to solve, matching exactly what
# generate_solutions.py will pick (same seed + size + start/num slicing).
echo "=== R2E Gym pre-build: computing task list ==="
mapfile -t R2E_TASKS < <(
  uv run python - <<PYEOF
import random
from pathlib import Path

tasks_dir = Path("$TASKS_DIR")
sample_size = $SAMPLE_SIZE
sample_seed = $SAMPLE_SEED
start_at = $START_AT
num_tasks = $NUM_TASKS

all_task_dirs = sorted(
    str(p) for p in tasks_dir.iterdir()
    if p.is_dir() and p.name.startswith("r2e_")
)
if sample_size and sample_size > 0 and sample_size < len(all_task_dirs):
    rng = random.Random(sample_seed)
    sampled = rng.sample(all_task_dirs, sample_size)
    sampled = sorted(sampled)
else:
    sampled = all_task_dirs
window = sampled[start_at:min(start_at + num_tasks, len(sampled))]
print("\n".join(window))
PYEOF
)

R2E_N="${#R2E_TASKS[@]}"
echo "=== R2E Gym: ${R2E_N} tasks selected (sample_size=${SAMPLE_SIZE}, seed=${SAMPLE_SEED}) ==="

if [ "$R2E_N" = "0" ]; then
  echo "ERROR: no R2E tasks selected -- did you run run_ingest_r2e_gym.sh first?" >&2
  exit 1
fi

# Build one SIF. Exits 0 on success, non-zero on failure (but never with
# `set -e` propagation so one bad task doesn't abort the whole pre-build).
r2e_build_one() {
  local task_dir="$1"
  local sif="${task_dir}/container.sif"
  local def="${task_dir}/container.def"
  local tag="$(basename "$task_dir")"
  if [ -f "$sif" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    echo "  skip  ${tag} (sif already exists)"
    return 0
  fi
  if [ ! -f "$def" ]; then
    echo "  miss  ${tag} (no container.def)" >&2
    return 1
  fi
  local attempt=1
  while [ $attempt -le "$PREBUILD_RETRIES" ]; do
    if apptainer build --force --ignore-fakeroot-command \
         "$sif" "$def" >"${task_dir}/.prebuild.log" 2>&1; then
      echo "  ok    ${tag} (attempt ${attempt})"
      rm -f "${task_dir}/.prebuild.log"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep $((attempt * 2))
  done
  echo "  FAIL  ${tag} -- see ${task_dir}/.prebuild.log" >&2
  return 1
}
export -f r2e_build_one
export PREBUILD_RETRIES FORCE_RERUN

echo "=== R2E Gym pre-build: ${R2E_N} SIF(s), workers=${PREBUILD_WORKERS} ==="
R2E_BUILD_LIST="$(mktemp)"
trap 'rm -f "$R2E_BUILD_LIST"' EXIT
printf '%s\n' "${R2E_TASKS[@]}" > "$R2E_BUILD_LIST"

# xargs -P parallelises; we tolerate individual failures because one bad
# task shouldn't block the other N-1 builds. The harness will mark the
# still-no-container.sif tasks as failed in the solution phase anyway.
xargs -a "$R2E_BUILD_LIST" -I{} -P "$PREBUILD_WORKERS" \
  bash -c 'r2e_build_one "$@"' _ {} || true

_BUILT=0
for td in "${R2E_TASKS[@]}"; do
  [ -f "${td}/container.sif" ] && _BUILT=$((_BUILT + 1))
done
echo "=== R2E Gym pre-build done: ${_BUILT}/${R2E_N} SIFs ready ==="

# ---- Solution phase -----------------------------------------------------
_vllm_wait_ready_local

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
# Include $HARNESS in the log filename so vanillux reruns don't clobber legacy
# bash-harness logs from earlier comparison runs.
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}_${_RUN_TS}.log"

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

echo "=== R2E Gym comparison run: MODEL=${MODEL}, HARNESS=${HARNESS}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS}, SAMPLE_SIZE=${SAMPLE_SIZE} ==="
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
