#!/usr/bin/env bash
#
# Launch a single beaker task that:
#   1. spins up a vLLM server on the local GPUs (8 by default)
#   2. configures podman + harbor (incl. the patches discovered while bringing
#      up harbor on the podman socket — see scripts/setup_podman_harbor.sh and
#      scripts/beaker/run_eval_in_job.sh)
#   3. runs `harbor run` on the chosen dataset against the local vLLM
#   4. copies the resulting jobs/<name>/ tree to a /weka path you can fetch.
#
# Usage:
#   ./beaker_configs/launch_eval.sh <model_path> [options]
#
# Example:
#   ./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
#       --revision sft_qwen3_4b_tmax_4node \
#       --name sft-4b-tb2 \
#       --gpus 8 \
#       --dataset terminal-bench@2.0
#
# Outputs (set via --results-dir, default below) end up on weka:
#   /weka/oe-adapt-default/${USER}/tmax-eval/<job-name>/jobs/<job-name>/
#
# Uses Beaker Gantry to submit the current git HEAD. The SHA must be pushed to
# the remote; local dirty changes are not included in the remote job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- defaults ----------------------------------------------------------------
REVISION="main"
SERVED_MODEL_NAME=""
GPU_COUNT=8
TP_SIZE=""
DP_SIZE=""
VLLM_PORT=8008
VLLM_VERSION="0.19.1"
MAX_MODEL_LEN=""
DATASET="terminal-bench@2.0"
AGENT_IMPORT_PATH="VanilluxAgent:VanilluxAgent"
N_CONCURRENT=8
N_ATTEMPTS=1
JOB_NAME=""
RESULTS_DIR=""
CLUSTER="ai2/saturn"
BUDGET=""
PRIORITY="urgent"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/tmax}"
BEAKER_IMAGE="hamishivi/hamishivi-interactive"
REPO_GIT_URL=""
REPO_GIT_REF=""

usage() {
    cat <<EOF
Usage: $0 <model_path> [options]

Required:
  <model_path>           HF model path (e.g. allenai/Llama-3.1-Tulu-3-8B)
                         or a weka path the beaker image can read.

Options:
  --revision REV         HF revision/branch (default: main)
  --name NAME            served-model-name (default: basename of model_path)
  --gpus N               GPUs (default: 8)
  --tp N                 tensor-parallel-size (default: GPU_COUNT)
  --dp N                 data-parallel-size (default: 1)
  --port PORT            vllm port (default: 8008)
  --vllm-version VER     vLLM package version for uvx (default: 0.19.1)
  --max-model-len LEN    pass --max-model-len to vllm
  --dataset DS           harbor dataset (default: terminal-bench@2.0; also
                         valid: openthoughts-tblite@2.0)
  --agent IMPORT_PATH    harbor --agent-import-path (default: VanilluxAgent:VanilluxAgent)
  --n-concurrent N       harbor --n-concurrent (default: 8)
  --n-attempts N         harbor -k (default: 1)
  --job-name NAME        harbor --job-name (default: <served-name>-<dataset>)
  --results-dir DIR      where to copy the harbor jobs/ output
                         (default: /results; persisted by Gantry)
  --cluster CLUSTER      beaker cluster (default: ai2/saturn)
  --budget BUDGET        beaker budget (default: omitted; uses workspace default)
  --priority PRI         beaker priority (default: urgent)
  --workspace WS         beaker workspace (default: \$BEAKER_WORKSPACE or ai2/tmax)
  --image IMAGE          beaker image (default: $BEAKER_IMAGE)
  --repo-url URL         git URL of tmax (default: current 'origin' remote)
  --repo-ref REF         git SHA/branch of tmax (default: current HEAD SHA)
EOF
    exit 1
}

[ $# -lt 1 ] && usage
MODEL_PATH="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --revision)        REVISION="$2"; shift 2 ;;
        --name)            SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpus)            GPU_COUNT="$2"; shift 2 ;;
        --tp)              TP_SIZE="$2"; shift 2 ;;
        --dp)              DP_SIZE="$2"; shift 2 ;;
        --port)            VLLM_PORT="$2"; shift 2 ;;
        --vllm-version)    VLLM_VERSION="$2"; shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --agent)           AGENT_IMPORT_PATH="$2"; shift 2 ;;
        --n-concurrent)    N_CONCURRENT="$2"; shift 2 ;;
        --n-attempts)      N_ATTEMPTS="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
        --cluster)         CLUSTER="$2"; shift 2 ;;
        --budget)          BUDGET="$2"; shift 2 ;;
        --priority)        PRIORITY="$2"; shift 2 ;;
        --workspace)       BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)           BEAKER_IMAGE="$2"; shift 2 ;;
        --repo-url)        REPO_GIT_URL="$2"; shift 2 ;;
        --repo-ref)        REPO_GIT_REF="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *) echo "unknown option: $1"; usage ;;
    esac
done

# --- derive defaults ---------------------------------------------------------
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
TP_SIZE="${TP_SIZE:-$GPU_COUNT}"
DP_SIZE="${DP_SIZE:-1}"

DATASET_SLUG="${DATASET//[^A-Za-z0-9]/-}"
JOB_NAME="${JOB_NAME:-${SERVED_MODEL_NAME}-${DATASET_SLUG}}"
RESULTS_DIR="${RESULTS_DIR:-/results}"

if [ -z "$REPO_GIT_REF" ]; then
    REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

BEAKER_NAME="eval-${JOB_NAME}"

cat <<EOF
=== Launching tmax eval on Beaker ===
  Model:        ${MODEL_PATH}@${REVISION}
  Served name:  ${SERVED_MODEL_NAME}
  vLLM version: ${VLLM_VERSION}
  GPUs:         ${GPU_COUNT} (TP=${TP_SIZE}, DP=${DP_SIZE})
  Dataset:      ${DATASET}
  Agent:        ${AGENT_IMPORT_PATH}
  Job name:     ${JOB_NAME}
  Results dir:  ${RESULTS_DIR}
  Repo ref:     ${REPO_GIT_REF}
  Gantry:       ${BEAKER_NAME}  cluster=${CLUSTER}  workspace=${BEAKER_WORKSPACE}
EOF

# Sanity check: the SHA must be reachable on the remote.
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo
    echo "warning: $REPO_GIT_REF doesn't appear to be on any remote branch."
    echo "         the beaker job will fail to clone it. push first or pass --repo-ref."
fi

# --- launch via Gantry --------------------------------------------------------
GANTRY_CMD=(
    uvx --from beaker-gantry gantry --quiet run
    --yes
    --allow-dirty
    --workspace "$BEAKER_WORKSPACE"
    --name "$BEAKER_NAME"
    --description "Harbor eval (${DATASET}) of ${SERVED_MODEL_NAME} (${MODEL_PATH}@${REVISION}) via vLLM"
    --ref "$REPO_GIT_REF"
    --cluster "$CLUSTER"
    --gpus "$GPU_COUNT"
    --priority "$PRIORITY"
    --beaker-image "$BEAKER_IMAGE"
    --weka "oe-adapt-default:/weka/oe-adapt-default"
    --upload "$REPO_ROOT/scripts/beaker:/uploaded-beaker-scripts"
    --env-secret HF_TOKEN
    --env-secret "DOCKER_PAT=${DOCKER_PAT_SECRET:-hamishivi_DOCKER_PAT}"
    --env "MODEL_PATH=${MODEL_PATH}"
    --env "MODEL_REVISION=${REVISION}"
    --env "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    --env "VLLM_VERSION=${VLLM_VERSION}"
    --env "VLLM_PORT=${VLLM_PORT}"
    --env "TP_SIZE=${TP_SIZE}"
    --env "DP_SIZE=${DP_SIZE}"
    --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    --env "DATASET=${DATASET}"
    --env "AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH}"
    --env "N_CONCURRENT=${N_CONCURRENT}"
    --env "N_ATTEMPTS=${N_ATTEMPTS}"
    --env "JOB_NAME=${JOB_NAME}"
    --env BEAKER_ALLOW_SUBCONTAINERS=1
    --env BEAKER_SKIP_DOCKER_SOCKET=1
    --host-networking
    --propagate-failure
    --no-python
)

if [ -n "$BUDGET" ]; then
    GANTRY_CMD+=(--budget "$BUDGET")
fi

if [ "$RESULTS_DIR" = "/results" ]; then
    GANTRY_CMD+=(-- bash /uploaded-beaker-scripts/run_eval_in_job.sh)
else
    GANTRY_CMD+=(-- env "RESULTS_DIR=${RESULTS_DIR}" bash /uploaded-beaker-scripts/run_eval_in_job.sh)
fi

echo
printf 'Launching with:'
printf ' %q' "${GANTRY_CMD[@]}"
printf '\n\n'

"${GANTRY_CMD[@]}"
