#!/bin/bash
# Ingest open-thoughts/OpenThoughts-TB-dev into our canonical Apptainer layout.

set -euo pipefail

OT_LIMIT="${OT_LIMIT:-0}"
OT_DST="${OT_DST:-rl_data/output/tasks_openthoughts_tb}"
OT_CACHE="${OT_CACHE:-rl_data/output/_otb_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$OT_DST" --cache-dir "$OT_CACHE" --limit "$OT_LIMIT")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi

uv run python -m rl_data.comparison.adapters.openthoughts_tb "${ARGS[@]}"
