#!/usr/bin/env bash
set -euo pipefail

# ── Cost Estimator for RL Data Generation ────────────────────────────
#
# Edit the variables below, then run:
#   bash rl_data/scripts/analyze/estimate_cost.sh
#
# Or override from the command line:
#   bash rl_data/scripts/analyze/estimate_cost.sh --num-tasks 1000

# ---- Parameters (edit here) ----
NUM_TASKS=1000
MODEL="gemini/gemini-3.1-pro-preview"
SURVIVAL_RATE=0.5

# Generate solution config
NUM_SOLUTIONS=8
MAX_ACTIONS=16
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

uv run python -m rl_data.estimate_cost \
    --num-tasks "$NUM_TASKS" \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --model "$MODEL" \
    --survival-rate "$SURVIVAL_RATE" \
    "$@"
