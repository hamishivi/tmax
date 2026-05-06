#!/bin/bash
set -euo pipefail

# ---- Parameters (edit here) ----
# Pick a corpus to analyze. Recent options:
#   tasks_skill_tax_20260324_1k        (legacy 1k SFT)
#   tasks_skill_tax_20260401_10k       (legacy 10k)
#   tasks_skill_tax_20260505_1k_legacy (latest legacy 1k re-gen)
#   tasks_skill_tax_v2_20260505_2k     (v2 SFT, ad9d7fe — verifier × fixture × intricate)
TASKS_DIR="/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_v2_20260505_2k_sft"

PLOTS_DIR=""   # leave empty to default to <TASKS_DIR>/analysis
MODEL=""       # e.g. "gemini/gemini-3-flash-preview"; leave empty to auto-discover all models
# Harness selector. Empty = auto-discover all (model, harness) pairs.
# Set to "bash" or "vanillux" to restrict to one solver harness.
# vanillux summaries are written by --harness vanillux in generate_solutions.py
# (commit ad9d7fe) and live alongside bash summaries as
# <MODEL_TAG>_vanillux_summary.json.
HARNESS=""
MAX_ROWS=50    # 0 to show all per-task rows
# Pass@K ceiling. Empty = auto-detect the largest k that every solved task
# has data for (= min of NUM_SOLUTIONS across solved tasks). Set to a fixed
# integer (e.g. 4 or 8) to force that k — useful for comparing two corpora
# at a common ceiling.
PASS_AT_K=""
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

ARGS=(--tasks-dir "$TASKS_DIR" --max-rows "$MAX_ROWS")
if [[ -n "$PLOTS_DIR" ]]; then
    ARGS+=(--plots-dir "$PLOTS_DIR")
fi
if [[ -n "$MODEL" ]]; then
    ARGS+=(--model "$MODEL")
fi
if [[ -n "$HARNESS" ]]; then
    ARGS+=(--harness "$HARNESS")
fi
if [[ -n "$PASS_AT_K" ]]; then
    ARGS+=(--pass-at-k "$PASS_AT_K")
fi

uv run python -m rl_data.analyze "${ARGS[@]}"
