#!/bin/bash
# Orchestrator for the dataset-decontamination eval.
#
# Pipeline:
#   1. Fetch evaluation benchmarks via `harbor download` (idempotent):
#        - terminal-bench@2.0           -> $BENCH_CACHE_DIR/terminal-bench
#        - openthoughts-tblite@2.0      -> $BENCH_CACHE_DIR/openthoughts-tblite
#   2. Run the decon CLI:
#        python -m rl_data.decontamination.cli ... --n $NGRAM_N --stride $NGRAM_STRIDE
#
# Typical use:
#   bash rl_data/scripts/decontamination/run_decontamination.sh
#
# Env toggles (defaults mirror rl_data/scripts/comparison/run_comparison.sh):
#   OURS_TASKS_DIR    Default: rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k
#   ET_TASKS_DIR      Default: rl_data/output/tasks_endless_terminals
#   OT_TASKS_DIR      Default: rl_data/output/tasks_openthoughts_agent_rl
#   TG_TASKS_DIR      Default: rl_data/output/tasks_termigen
#   TT_TASKS_DIR      Default: rl_data/output/tasks_terminaltraj
#   R2E_TASKS_DIR     Default: rl_data/output/tasks_r2e_gym
#   BENCH_CACHE_DIR   Default: rl_data/output/_decon_benchmarks
#   DECON_OUT_DIR     Default: rl_data/output/decontamination_0518
#   NGRAM_N           Default: "13,8"  (comma-separated list of n-gram sizes)
#   NGRAM_STRIDE      Default: 1
#
# Stage skips:
#   SKIP_FETCH_TB2=1, SKIP_FETCH_TBLITE=1, SKIP_RUN=1

set -euo pipefail

OURS_TASKS_DIR="${OURS_TASKS_DIR:-rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k}"
ET_TASKS_DIR="${ET_TASKS_DIR:-rl_data/output/tasks_endless_terminals}"
OT_TASKS_DIR="${OT_TASKS_DIR:-rl_data/output/tasks_openthoughts_agent_rl}"
TG_TASKS_DIR="${TG_TASKS_DIR:-rl_data/output/tasks_termigen}"
TT_TASKS_DIR="${TT_TASKS_DIR:-rl_data/output/tasks_terminaltraj}"
R2E_TASKS_DIR="${R2E_TASKS_DIR:-rl_data/output/tasks_r2e_gym}"
BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-rl_data/output/_decon_benchmarks}"
DECON_OUT_DIR="${DECON_OUT_DIR:-rl_data/output/decontamination_0518}"
NGRAM_N="${NGRAM_N:-13,8}"
NGRAM_STRIDE="${NGRAM_STRIDE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Decontamination pipeline ==="
echo "  ours           : $OURS_TASKS_DIR"
echo "  ET             : $ET_TASKS_DIR"
echo "  OT             : $OT_TASKS_DIR"
echo "  TermiGen       : $TG_TASKS_DIR"
echo "  TerminalTraj   : $TT_TASKS_DIR"
echo "  R2E Gym        : $R2E_TASKS_DIR"
echo "  bench cache    : $BENCH_CACHE_DIR"
echo "  out dir        : $DECON_OUT_DIR"
echo "  n / stride     : $NGRAM_N / $NGRAM_STRIDE"
echo

mkdir -p "$BENCH_CACHE_DIR"

_has_tasks() {
  # True if the directory exists and contains at least one instruction.md.
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -maxdepth 4 -name instruction.md -print -quit 2>/dev/null | grep -q .
}

# ── 1. Fetch benchmarks ────────────────────────────────────────────────
if [[ "${SKIP_FETCH_TB2:-0}" != "1" ]]; then
  if _has_tasks "$BENCH_CACHE_DIR/terminal-bench"; then
    echo ">>> 1a. terminal-bench@2.0 already cached, skipping"
  else
    echo ">>> 1a. Downloading terminal-bench@2.0"
    uv run harbor download terminal-bench@2.0 --export -o "$BENCH_CACHE_DIR"
  fi
fi
if [[ "${SKIP_FETCH_TBLITE:-0}" != "1" ]]; then
  if _has_tasks "$BENCH_CACHE_DIR/openthoughts-tblite"; then
    echo ">>> 1b. openthoughts-tblite@2.0 already cached, skipping"
  else
    echo ">>> 1b. Downloading openthoughts-tblite@2.0"
    uv run harbor download openthoughts-tblite@2.0 --export -o "$BENCH_CACHE_DIR"
  fi
fi

# ── 2. Run decon CLI ───────────────────────────────────────────────────
if [[ "${SKIP_RUN:-0}" != "1" ]]; then
  echo ">>> 2. Running decontamination CLI"
  DATASET_ARGS=()
  if [[ -d "$OURS_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "skill_tax:$OURS_TASKS_DIR")
  fi
  if [[ -d "$ET_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "endless_terminals:$ET_TASKS_DIR")
  fi
  if [[ -d "$OT_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "openthoughts_agent_rl:$OT_TASKS_DIR")
  fi
  if [[ -d "$TG_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "termigen:$TG_TASKS_DIR")
  fi
  if [[ -d "$TT_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "terminaltraj:$TT_TASKS_DIR")
  fi
  if [[ -d "$R2E_TASKS_DIR" ]]; then
    DATASET_ARGS+=(--dataset "r2e_gym:$R2E_TASKS_DIR")
  fi

  BENCH_ARGS=()
  if [[ -d "$BENCH_CACHE_DIR/openthoughts-tblite" ]]; then
    BENCH_ARGS+=(--benchmark "tblite:$BENCH_CACHE_DIR/openthoughts-tblite")
  fi
  if [[ -d "$BENCH_CACHE_DIR/terminal-bench" ]]; then
    BENCH_ARGS+=(--benchmark "tb2:$BENCH_CACHE_DIR/terminal-bench")
  fi

  uv run python -m rl_data.decontamination.cli \
      "${DATASET_ARGS[@]}" \
      "${BENCH_ARGS[@]}" \
      --n "$NGRAM_N" --stride "$NGRAM_STRIDE" \
      --out-dir "$DECON_OUT_DIR"
fi

echo
echo "Done. See ${DECON_OUT_DIR}/decontamination_table.md."
