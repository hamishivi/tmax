#!/bin/bash
# Ingest obiwan96/endless-terminals into our canonical Apptainer layout.
#
# Env:
#   ET_LIMIT   Convert only the first N tasks (0 = all). Default: 0
#   ET_DST     Destination dir. Default: rl_data/output/tasks_endless_terminals
#   ET_CACHE   HF snapshot cache dir. Default: rl_data/output/_et_cache
#   SKIP_DOWNLOAD=1  Reuse whatever is already in ET_CACHE

set -euo pipefail

ET_LIMIT="${ET_LIMIT:-0}"
ET_DST="${ET_DST:-rl_data/output/tasks_endless_terminals}"
ET_CACHE="${ET_CACHE:-rl_data/output/_et_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$ET_DST" --cache-dir "$ET_CACHE" --limit "$ET_LIMIT")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi

uv run python -m rl_data.comparison.adapters.endless_terminals "${ARGS[@]}"
