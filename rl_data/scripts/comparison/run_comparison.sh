#!/bin/bash
# Orchestrator for the dataset-comparison suite.
#
# Pipeline:
#   1. Ingest baselines  (scripts/run_ingest_et.sh,
#                         scripts/run_ingest_openthoughts.sh,
#                         scripts/run_ingest_termigen.sh,
#                         scripts/run_ingest_terminaltraj.sh,
#                         scripts/run_ingest_r2e_gym.sh)
#   2. Classify external tasks into OUR taxonomy via an LLM
#      (scripts/run_classify_taxonomy.sh)
#   3. Run our solution harness on each baseline + our rebased reference
#      under a uniform config (vanillux harness, NUM_SOLUTIONS=8,
#      MAX_ACTIONS=64, COMMAND_TIMEOUT=600, SAMPLE_SIZE=250) so head-to-
#      head pass@1/4/8 is apples-to-apples across all 6 datasets. See
#      scripts/run_generate_solutions_{et,openthoughts,termigen,terminaltraj,
#      r2e_gym,skill_tax_combined_legacy10k_new5k}.sh.
#   4. Compare (python -m rl_data.comparison.cli --harness $HARNESS)
#
# Each stage can be skipped with SKIP_* env vars. Typical flow:
#
#   SKIP_SOLVE=1 bash rl_data/scripts/comparison/run_comparison.sh    # quick dry-run
#   bash rl_data/scripts/comparison/run_comparison.sh                 # full pipeline
#
# Env toggles:
#   OURS_TASKS_DIR         Default: rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k
#   ET_TASKS_DIR           Default: rl_data/output/tasks_endless_terminals
#   OT_TASKS_DIR           Default: rl_data/output/tasks_openthoughts_agent_rl
#   TG_TASKS_DIR           Default: rl_data/output/tasks_termigen
#   TT_TASKS_DIR           Default: rl_data/output/tasks_terminaltraj
#   R2E_TASKS_DIR          Default: rl_data/output/tasks_r2e_gym
#   COMPARE_OUT_DIR        Default: rl_data/output/comparison_vanillux_0515
#   MODEL                  Default: gemini/gemini-3-flash-preview
#   HARNESS                Default: vanillux  (set to 'bash' to read legacy
#                          <MODEL>_summary.json files from a pre-0515 run)
#
# Stage skips:
#   SKIP_INGEST_ET=1, SKIP_INGEST_OT=1, SKIP_INGEST_TG=1, SKIP_INGEST_TT=1, SKIP_INGEST_R2E=1
#   SKIP_CLASSIFY=1
#   SKIP_SOLVE_OURS=1, SKIP_SOLVE_ET=1, SKIP_SOLVE_OT=1, SKIP_SOLVE_TG=1, SKIP_SOLVE_TT=1, SKIP_SOLVE_R2E=1
#   SKIP_COMPARE=1

set -euo pipefail

OURS_TASKS_DIR="${OURS_TASKS_DIR:-rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k}"
ET_TASKS_DIR="${ET_TASKS_DIR:-rl_data/output/tasks_endless_terminals}"
OT_TASKS_DIR="${OT_TASKS_DIR:-rl_data/output/tasks_openthoughts_agent_rl}"
TG_TASKS_DIR="${TG_TASKS_DIR:-rl_data/output/tasks_termigen}"
TT_TASKS_DIR="${TT_TASKS_DIR:-rl_data/output/tasks_terminaltraj}"
R2E_TASKS_DIR="${R2E_TASKS_DIR:-rl_data/output/tasks_r2e_gym}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-rl_data/output/comparison_vanillux_0515}"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
HARNESS="${HARNESS:-vanillux}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Comparison pipeline ==="
echo "  ours         : $OURS_TASKS_DIR"
echo "  ET           : $ET_TASKS_DIR"
echo "  OT           : $OT_TASKS_DIR"
echo "  TermiGen     : $TG_TASKS_DIR"
echo "  TerminalTraj : $TT_TASKS_DIR"
echo "  R2E Gym      : $R2E_TASKS_DIR"
echo "  model        : $MODEL"
echo "  harness      : $HARNESS"
echo "  out-dir      : $COMPARE_OUT_DIR"
echo

# ── 1. Ingest ──────────────────────────────────────────────────────────
# if [[ "${SKIP_INGEST_ET:-0}" != "1" ]]; then
#   echo ">>> 1a. Ingesting endless-terminals"
#   ET_DST="$ET_TASKS_DIR" bash "$SCRIPT_DIR/run_ingest_et.sh"
# fi
# if [[ "${SKIP_INGEST_OT:-0}" != "1" ]]; then
#   echo ">>> 1b. Ingesting OpenThoughts-Agent-v1-RL"
#   OT_DST="$OT_TASKS_DIR" bash "$SCRIPT_DIR/run_ingest_openthoughts.sh"
# fi
# if [[ "${SKIP_INGEST_TG:-0}" != "1" ]]; then
#   echo ">>> 1c. Ingesting TermiGen (ucsb-mlsec/terminal-bench-env)"
#   TG_DST="$TG_TASKS_DIR" bash "$SCRIPT_DIR/run_ingest_termigen.sh"
# fi
# if [[ "${SKIP_INGEST_TT:-0}" != "1" ]]; then
#   echo ">>> 1d. Ingesting TerminalTraj (m-a-p/TerminalTraj-5k-instances)"
#   TT_DST="$TT_TASKS_DIR" bash "$SCRIPT_DIR/run_ingest_terminaltraj.sh"
# fi
# if [[ "${SKIP_INGEST_R2E:-0}" != "1" ]]; then
#   echo ">>> 1e. Ingesting R2E Gym (hamishivi/agent-task-r2e-gym)"
#   R2E_DST="$R2E_TASKS_DIR" bash "$SCRIPT_DIR/run_ingest_r2e_gym.sh"
# fi

# # ── 2. Classify into our taxonomy ──────────────────────────────────────
# if [[ "${SKIP_CLASSIFY:-0}" != "1" ]]; then
#   echo ">>> 2. Classifying external tasks into our taxonomy"
#   CLASSIFY_DIRS="$ET_TASKS_DIR $OT_TASKS_DIR $TG_TASKS_DIR $TT_TASKS_DIR $R2E_TASKS_DIR" \
#       CLASSIFY_MODEL="$MODEL" \
#       bash "$SCRIPT_DIR/run_classify_taxonomy.sh"
# fi

# # ── 3. Solve ───────────────────────────────────────────────────────────
# # Each of the five solve scripts already exports the uniform comparison
# # config (HARNESS=vanillux, NUM_SOLUTIONS=8, MAX_ACTIONS=64,
# # COMMAND_TIMEOUT=600, SAMPLE_SIZE=250, SAMPLE_SEED=0) as its default, so
# # uncommenting these calls is enough to reproduce the 0515 run. In
# # practice you'll want to sbatch each one onto a dedicated node rather
# # than run them serially here (each writes into its own TASKS_DIR, so
# # they can all run in parallel on different nodes without conflict).
# if [[ "${SKIP_SOLVE_OURS:-0}" != "1" ]]; then
#   echo ">>> 3a. Solving skill-tax combined (legacy10k + new5k) — OURS reference"
#   TASKS_DIR="$OURS_TASKS_DIR" \
#       bash "$SCRIPT_DIR/run_generate_solutions_skill_tax_combined_legacy10k_new5k.sh"
# fi
# if [[ "${SKIP_SOLVE_ET:-0}" != "1" ]]; then
#   echo ">>> 3b. Solving endless-terminals (prefer Slurm for the real run)"
#   bash "$SCRIPT_DIR/run_generate_solutions_et.sh"
# fi
# if [[ "${SKIP_SOLVE_OT:-0}" != "1" ]]; then
#   echo ">>> 3c. Solving OpenThoughts-Agent-v1-RL (prefer Slurm for the real run)"
#   bash "$SCRIPT_DIR/run_generate_solutions_openthoughts.sh"
# fi
# if [[ "${SKIP_SOLVE_TG:-0}" != "1" ]]; then
#   echo ">>> 3d. Solving TermiGen (prefer Slurm for the real run)"
#   bash "$SCRIPT_DIR/run_generate_solutions_termigen.sh"
# fi
# if [[ "${SKIP_SOLVE_TT:-0}" != "1" ]]; then
#   echo ">>> 3e. Solving TerminalTraj (prefer Slurm for the real run)"
#   bash "$SCRIPT_DIR/run_generate_solutions_terminaltraj.sh"
# fi
# if [[ "${SKIP_SOLVE_R2E:-0}" != "1" ]]; then
#   echo ">>> 3f. Solving R2E Gym (prefer Slurm for the real run)"
#   bash "$SCRIPT_DIR/run_generate_solutions_r2e_gym.sh"
# fi

# ── 4. Compare ─────────────────────────────────────────────────────────
if [[ "${SKIP_COMPARE:-0}" != "1" ]]; then
  echo ">>> 4. Running comparison"
  BASELINE_ARGS=()
  if [[ -d "$ET_TASKS_DIR" ]]; then
    BASELINE_ARGS+=(--baseline "endless_terminals:$ET_TASKS_DIR")
  fi
  if [[ -d "$OT_TASKS_DIR" ]]; then
    BASELINE_ARGS+=(--baseline "openthoughts_agent_rl:$OT_TASKS_DIR")
  fi
  if [[ -d "$TG_TASKS_DIR" ]]; then
    BASELINE_ARGS+=(--baseline "termigen:$TG_TASKS_DIR")
  fi
  if [[ -d "$TT_TASKS_DIR" ]]; then
    BASELINE_ARGS+=(--baseline "terminaltraj:$TT_TASKS_DIR")
  fi
  if [[ -d "$R2E_TASKS_DIR" ]]; then
    BASELINE_ARGS+=(--baseline "r2e_gym:$R2E_TASKS_DIR")
  fi

  uv run python -m rl_data.comparison.cli \
      --reference "skill_tax:$OURS_TASKS_DIR" \
      "${BASELINE_ARGS[@]}" \
      --model "$MODEL" \
      --harness "$HARNESS" \
      --out-dir "$COMPARE_OUT_DIR"
fi

echo
echo "Done. See ${COMPARE_OUT_DIR}/main/ for headline figures + summary_table.md."
