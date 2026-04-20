#!/bin/bash
# Run the LLM-based taxonomy classifier on external datasets so they can be
# compared against our native taxonomy in the composition module.
#
# Env:
#   CLASSIFY_DIRS    Space-separated list of tasks dirs to classify. Default:
#                    "rl_data/output/tasks_endless_terminals rl_data/output/tasks_openthoughts_tb"
#   CLASSIFY_MODEL   LLM slug. Default: gemini/gemini-3-flash-preview
#   CLASSIFY_LIMIT   Classify at most N unclassified tasks per dir (0 = all).
#   CLASSIFY_FORCE=1 Re-classify even if classified_* fields already exist.
#   CLASSIFY_CONCURRENCY  Max concurrent LLM calls. Default: 32

set -euo pipefail

CLASSIFY_DIRS=${CLASSIFY_DIRS:-"rl_data/output/tasks_endless_terminals rl_data/output/tasks_openthoughts_tb"}
CLASSIFY_MODEL="${CLASSIFY_MODEL:-gemini/gemini-3-flash-preview}"
CLASSIFY_LIMIT="${CLASSIFY_LIMIT:-0}"
CLASSIFY_CONCURRENCY="${CLASSIFY_CONCURRENCY:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

for d in $CLASSIFY_DIRS; do
  if [[ ! -d "$d" ]]; then
    echo "SKIP: $d does not exist (run the ingest script first)"
    continue
  fi
  echo ">>> Classifying $d"
  ARGS=(--tasks-dir "$d" --model "$CLASSIFY_MODEL"
        --max-concurrency "$CLASSIFY_CONCURRENCY" --limit "$CLASSIFY_LIMIT")
  if [[ "${CLASSIFY_FORCE:-0}" == "1" ]]; then
    ARGS+=(--force)
  fi
  uv run python -m rl_data.comparison.taxonomy_classifier "${ARGS[@]}"
done
