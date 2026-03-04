#!/usr/bin/env bash
set -euo pipefail

# Run TassieAgent on SWE-Bench Verified (Daytona backend)
#
# Required env vars:
#   DAYTONA_API_KEY    - Daytona API key
#   OPENAI_API_BASE    - vLLM server URL (e.g. http://localhost:8000/v1)
#
# Optional env vars:
#   MODEL              - Model name (default: openai/default)
#   N_CONCURRENT       - Number of concurrent trials (default: 16)
#   MAX_STEPS          - Max agent steps per trial (default: 30)

MODEL="${MODEL:-openai/default}"
N_CONCURRENT="${N_CONCURRENT:-25}"
MAX_STEPS="${MAX_STEPS:-50}"

harbor run \
    --dataset swebench-verified@1.0 \
    --agent-import-path TassieAgent:TassieAgent \
    --model "$MODEL" \
    --env daytona \
    --n-concurrent "$N_CONCURRENT" \
    --agent-kwargs "{\"max_steps\": $MAX_STEPS}"
