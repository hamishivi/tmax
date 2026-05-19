#!/bin/bash
# Ingest hamishivi/agent-task-r2e-gym (R2E Gym, 8,101 SWE-bench-style tasks
# shipped as a single 14 MB task-data.tar.gz on the HF Hub) into our
# canonical Apptainer layout.
#
# The dataset's tarball unpacks 8,101 ``<repo>__<commit_short>/`` task dirs
# containing image.txt + instruction.md + tests/{test.sh,score.py,
# expected_output.json}. The adapter materializes these into our standard
# task layout (task.json + container.def + test_final_state.py +
# test_initial_state.py) so the solve script and the comparison CLI can
# treat R2E Gym uniformly with every other baseline.
#
# Env overrides:
#   R2E_LIMIT=N         Convert only the first N tasks (0 = all, default).
#   R2E_DST=...         Destination tasks dir. Default: rl_data/output/tasks_r2e_gym
#   R2E_CACHE=...       Download cache dir.   Default: rl_data/output/_r2e_gym_cache
#   R2E_REVISION=<sha>  Pin to a specific HF dataset revision.
#   SKIP_DOWNLOAD=1     Reuse the existing extracted tarball without fetching.
#   WORKERS=16          Parallel conversion workers (default: 16).

set -euo pipefail

R2E_LIMIT="${R2E_LIMIT:-0}"
R2E_DST="${R2E_DST:-rl_data/output/tasks_r2e_gym}"
R2E_CACHE="${R2E_CACHE:-rl_data/output/_r2e_gym_cache}"
R2E_REVISION="${R2E_REVISION:-}"
WORKERS="${WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$R2E_DST" --cache-dir "$R2E_CACHE" --limit "$R2E_LIMIT" --workers "$WORKERS")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi
if [[ -n "$R2E_REVISION" ]]; then
  ARGS+=(--revision "$R2E_REVISION")
fi

uv run python -m rl_data.comparison.adapters.r2e_gym "${ARGS[@]}"
