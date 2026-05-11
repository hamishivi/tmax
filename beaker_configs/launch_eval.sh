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
# The script reads the current repo's git remote + HEAD SHA to tell the beaker
# task which commit of tmax to clone. The SHA must be pushed to the remote.

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
MAX_MODEL_LEN=""
DATASET="terminal-bench@2.0"
AGENT_IMPORT_PATH="VanilluxAgent:VanilluxAgent"
N_CONCURRENT=8
N_ATTEMPTS=1
JOB_NAME=""
RESULTS_DIR=""
CLUSTER="ai2/saturn"
BUDGET="ai2/oe-adapt"
PRIORITY="high"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/tmax}"
BEAKER_IMAGE="ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0"
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
  --max-model-len LEN    pass --max-model-len to vllm
  --dataset DS           harbor dataset (default: terminal-bench@2.0; also
                         valid: openthoughts-tblite@2.0)
  --agent IMPORT_PATH    harbor --agent-import-path (default: VanilluxAgent:VanilluxAgent)
  --n-concurrent N       harbor --n-concurrent (default: 8)
  --n-attempts N         harbor -k (default: 1)
  --job-name NAME        harbor --job-name (default: <served-name>-<dataset>)
  --results-dir DIR      where to copy the harbor jobs/ output
                         (default: /weka/oe-adapt-default/\$USER/tmax-eval/<job-name>)
  --cluster CLUSTER      beaker cluster (default: ai2/saturn)
  --budget BUDGET        beaker budget (default: ai2/oe-adapt)
  --priority PRI         beaker priority (default: high)
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
RESULTS_DIR="${RESULTS_DIR:-/weka/oe-adapt-default/${USER:-runner}/tmax-eval/${JOB_NAME}}"

# Pin the tmax commit so the beaker container clones exactly what we have.
if [ -z "$REPO_GIT_URL" ]; then
    REPO_GIT_URL="$(git -C "$REPO_ROOT" config --get remote.origin.url)"
fi
if [ -z "$REPO_GIT_REF" ]; then
    REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

BEAKER_NAME="eval-${JOB_NAME}"

cat <<EOF
=== Launching tmax eval on Beaker ===
  Model:        ${MODEL_PATH}@${REVISION}
  Served name:  ${SERVED_MODEL_NAME}
  GPUs:         ${GPU_COUNT} (TP=${TP_SIZE}, DP=${DP_SIZE})
  Dataset:      ${DATASET}
  Agent:        ${AGENT_IMPORT_PATH}
  Job name:     ${JOB_NAME}
  Results dir:  ${RESULTS_DIR}
  Repo:         ${REPO_GIT_URL} @ ${REPO_GIT_REF}
  Beaker:       ${BEAKER_NAME}  cluster=${CLUSTER}  workspace=${BEAKER_WORKSPACE}
EOF

# Sanity check: the SHA must be reachable on the remote.
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo
    echo "warning: $REPO_GIT_REF doesn't appear to be on any remote branch."
    echo "         the beaker job will fail to clone it. push first or pass --repo-ref."
fi

# --- generate yaml -----------------------------------------------------------
TMP_YAML=$(mktemp /tmp/eval-beaker-XXXXXXXX).yaml
trap 'rm -f "$TMP_YAML"' EXIT

cat > "$TMP_YAML" <<YAML
version: v2
budget: ${BUDGET}
description: "Harbor eval (${DATASET}) of ${SERVED_MODEL_NAME} (${MODEL_PATH}@${REVISION}) via vLLM"
tasks:
  - name: "tmax-eval"
    image:
      beaker: ${BEAKER_IMAGE}
    hostNetworking: true
    # podman needs CAP_SYS_ADMIN + mknod for /dev/net/tun.
    propagateFailure: true
    command: ["/bin/bash", "-c"]
    arguments:
      - |
        set -euo pipefail
        bash scripts/beaker/run_eval_in_job.sh
    envVars:
      - name: HF_TOKEN
        secret: HF_TOKEN
      - name: MODEL_PATH
        value: "${MODEL_PATH}"
      - name: MODEL_REVISION
        value: "${REVISION}"
      - name: SERVED_MODEL_NAME
        value: "${SERVED_MODEL_NAME}"
      - name: VLLM_PORT
        value: "${VLLM_PORT}"
      - name: TP_SIZE
        value: "${TP_SIZE}"
      - name: DP_SIZE
        value: "${DP_SIZE}"
      - name: MAX_MODEL_LEN
        value: "${MAX_MODEL_LEN}"
      - name: DATASET
        value: "${DATASET}"
      - name: AGENT_IMPORT_PATH
        value: "${AGENT_IMPORT_PATH}"
      - name: N_CONCURRENT
        value: "${N_CONCURRENT}"
      - name: N_ATTEMPTS
        value: "${N_ATTEMPTS}"
      - name: JOB_NAME
        value: "${JOB_NAME}"
      - name: RESULTS_DIR
        value: "${RESULTS_DIR}"
      - name: REPO_GIT_URL
        value: "${REPO_GIT_URL}"
      - name: REPO_GIT_REF
        value: "${REPO_GIT_REF}"
    datasets:
      - mountPath: /weka/oe-adapt-default
        source:
          weka: oe-adapt-default
    constraints:
      cluster:
        - ${CLUSTER}
    resources:
      gpuCount: ${GPU_COUNT}
    context:
      priority: ${PRIORITY}
      preemptible: true
YAML

echo
echo "Generated beaker config:"
echo "------------------------"
cat "$TMP_YAML"
echo "------------------------"
echo

beaker experiment create "$TMP_YAML" --name "$BEAKER_NAME" --workspace "$BEAKER_WORKSPACE"
