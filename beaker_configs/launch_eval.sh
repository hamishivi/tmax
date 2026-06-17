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
HARBOR_MODEL_NAME=""
GPU_COUNT=8
TP_SIZE=""
DP_SIZE=""
VLLM_PORT=8008
VLLM_VERSION="0.19.1"
VLLM_TOOL_CALL_PARSER="hermes"
VLLM_LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=""
DATASET="terminal-bench@2.0"
HARBOR_ENV="docker"
AGENT_IMPORT_PATH="Vanillux2Agent:Vanillux2Agent"
N_CONCURRENT=8
N_ATTEMPTS=1
N_TASKS=""
JOB_NAME=""
RESULTS_DIR=""
CLUSTER="ai2/saturn"
BUDGET=""
PRIORITY="urgent"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/tmax}"
BEAKER_IMAGE="${BEAKER_IMAGE:-hamishivi/tmax-eval-interactive}"
BEAKER_DOCKER_IMAGE="${BEAKER_DOCKER_IMAGE:-}"
REPO_GIT_URL=""
REPO_GIT_REF=""
BEAKER_SCRIPTS_DATASET=""
EXTRA_UV_PIP_INSTALLS=""
EXTRA_AGENT_KWARGS=""
EXTRA_AGENT_ENVS=""
HOSTED_VLLM_MODEL_INFO=""
HARBOR_OVERRIDE_CPUS=""
HARBOR_OVERRIDE_MEMORY_MB=""
HARBOR_OVERRIDE_STORAGE_MB=""
HARBOR_OVERRIDE_GPUS=""
HARBOR_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_TIMEOUT_MULTIPLIER=""
HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER=""
HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_TIMEOUT_SEC=""

usage() {
    cat <<EOF
Usage: $0 <model_path> [options]

Required:
  <model_path>           HF model path (e.g. allenai/Llama-3.1-Tulu-3-8B)
                         or a weka path the beaker image can read.

Options:
  --revision REV         HF revision/branch (default: main)
  --name NAME            served-model-name (default: basename of model_path)
  --harbor-model-name NAME
                        model name passed to harbor (default: hosted_vllm/<served-name>)
  --gpus N               GPUs (default: 8)
  --tp N                 tensor-parallel-size (default: GPU_COUNT)
  --dp N                 data-parallel-size (default: 1)
  --port PORT            vllm port (default: 8008)
  --vllm-version VER     vLLM package version for uvx (default: 0.19.1)
  --tool-call-parser P   vLLM tool call parser (default: hermes)
  --language-model-only  pass --language_model_only to vLLM
  --max-model-len LEN    pass --max-model-len to vllm
  --dataset DS           harbor dataset (default: terminal-bench@2.0; also
                         valid: openthoughts-tblite@2.0)
  --harbor-env ENV       harbor environment backend (default: docker)
  --agent AGENT          harbor agent import path or named agent (default: Vanillux2Agent:Vanillux2Agent)
  --n-concurrent N       harbor --n-concurrent (default: 8)
  --n-attempts N         harbor -k (default: 1)
  --n-tasks N            harbor --n-tasks limit
  --job-name NAME        harbor --job-name (default: <served-name>-<dataset>)
  --results-dir DIR      where to copy the harbor jobs/ output
                         (default: /results; persisted by Gantry)
  --cluster CLUSTER      beaker cluster (default: ai2/saturn)
  --budget BUDGET        beaker budget (default: omitted; uses workspace default)
  --priority PRI         beaker priority (default: urgent)
  --workspace WS         beaker workspace (default: \$BEAKER_WORKSPACE or ai2/tmax)
  --image IMAGE          beaker image name or ID (clears default --docker-image)
  --docker-image IMAGE   public Docker image (default: $BEAKER_DOCKER_IMAGE)
  --beaker-scripts-dataset DS
                         existing Beaker dataset to mount at /uploaded-beaker-scripts
                         (default: upload local scripts/beaker)
  --repo-url URL         git URL of tmax (default: current 'origin' remote)
  --repo-ref REF         git SHA/branch of tmax (default: current HEAD SHA)
  --extra-uv-pip-install SPEC
                        extra package spec(s) to uv pip install in the job
  --agent-kwarg KV       extra harbor --agent-kwarg value (can be repeated)
  --agent-env KV         extra harbor --agent-env value (can be repeated)
  --hosted-vllm-model-info JSON
                        Harbor model_info JSON for hosted_vllm agents
  --override-cpus N      harbor per-task environment CPU override
  --override-memory-mb N harbor per-task environment memory override in MB
  --override-storage-mb N
                        harbor per-task environment storage override in MB
  --override-gpus N      harbor per-task environment GPU override
  --timeout-multiplier X harbor task timeout multiplier
  --agent-timeout-multiplier X
                        harbor agent timeout multiplier
  --verifier-timeout-multiplier X
                        harbor verifier timeout multiplier
  --agent-setup-timeout-multiplier X
                        harbor agent setup timeout multiplier
  --environment-build-timeout-multiplier X
                        harbor environment build timeout multiplier
  --agent-timeout-sec SEC
                        exact harbor agent timeout override in seconds
EOF
    exit 1
}

[ $# -lt 1 ] && usage
MODEL_PATH="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --revision)        REVISION="$2"; shift 2 ;;
        --name)            SERVED_MODEL_NAME="$2"; shift 2 ;;
        --harbor-model-name) HARBOR_MODEL_NAME="$2"; shift 2 ;;
        --gpus)            GPU_COUNT="$2"; shift 2 ;;
        --tp)              TP_SIZE="$2"; shift 2 ;;
        --dp)              DP_SIZE="$2"; shift 2 ;;
        --port)            VLLM_PORT="$2"; shift 2 ;;
        --vllm-version)    VLLM_VERSION="$2"; shift 2 ;;
        --tool-call-parser) VLLM_TOOL_CALL_PARSER="$2"; shift 2 ;;
        --language-model-only|--language_model_only) VLLM_LANGUAGE_MODEL_ONLY=1; shift ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --harbor-env)      HARBOR_ENV="$2"; shift 2 ;;
        --agent)           AGENT_IMPORT_PATH="$2"; shift 2 ;;
        --n-concurrent)    N_CONCURRENT="$2"; shift 2 ;;
        --n-attempts)      N_ATTEMPTS="$2"; shift 2 ;;
        --n-tasks)         N_TASKS="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
        --cluster)         CLUSTER="$2"; shift 2 ;;
        --budget)          BUDGET="$2"; shift 2 ;;
        --priority)        PRIORITY="$2"; shift 2 ;;
        --workspace)       BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)           BEAKER_IMAGE="$2"; BEAKER_DOCKER_IMAGE=""; shift 2 ;;
        --docker-image)    BEAKER_DOCKER_IMAGE="$2"; BEAKER_IMAGE=""; shift 2 ;;
        --beaker-scripts-dataset) BEAKER_SCRIPTS_DATASET="$2"; shift 2 ;;
        --repo-url)        REPO_GIT_URL="$2"; shift 2 ;;
        --repo-ref)        REPO_GIT_REF="$2"; shift 2 ;;
        --extra-uv-pip-install) EXTRA_UV_PIP_INSTALLS="$2"; shift 2 ;;
        --agent-kwarg)     EXTRA_AGENT_KWARGS+="${EXTRA_AGENT_KWARGS:+$'\n'}$2"; shift 2 ;;
        --agent-env)       EXTRA_AGENT_ENVS+="${EXTRA_AGENT_ENVS:+$'\n'}$2"; shift 2 ;;
        --hosted-vllm-model-info) HOSTED_VLLM_MODEL_INFO="$2"; shift 2 ;;
        --override-cpus)   HARBOR_OVERRIDE_CPUS="$2"; shift 2 ;;
        --override-memory-mb) HARBOR_OVERRIDE_MEMORY_MB="$2"; shift 2 ;;
        --override-storage-mb) HARBOR_OVERRIDE_STORAGE_MB="$2"; shift 2 ;;
        --override-gpus)   HARBOR_OVERRIDE_GPUS="$2"; shift 2 ;;
        --timeout-multiplier) HARBOR_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-timeout-multiplier) HARBOR_AGENT_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --verifier-timeout-multiplier) HARBOR_VERIFIER_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-setup-timeout-multiplier) HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --environment-build-timeout-multiplier) HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-timeout-sec) HARBOR_AGENT_TIMEOUT_SEC="$2"; shift 2 ;;
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
  Harbor model: ${HARBOR_MODEL_NAME:-hosted_vllm/${SERVED_MODEL_NAME}}
  vLLM version: ${VLLM_VERSION}
  Tool parser:  ${VLLM_TOOL_CALL_PARSER}
  LM only:      ${VLLM_LANGUAGE_MODEL_ONLY}
  GPUs:         ${GPU_COUNT} (TP=${TP_SIZE}, DP=${DP_SIZE})
  Dataset:      ${DATASET}
  Harbor env:   ${HARBOR_ENV}
  Agent:        ${AGENT_IMPORT_PATH}
  Agent kwargs: ${EXTRA_AGENT_KWARGS:-<none>}
  Agent envs:   ${EXTRA_AGENT_ENVS:-<none>}
  Extra pip:    ${EXTRA_UV_PIP_INSTALLS:-<none>}
  N tasks:      ${N_TASKS:-<all>}
  Task resources: cpus=${HARBOR_OVERRIDE_CPUS:-<task default>} memory_mb=${HARBOR_OVERRIDE_MEMORY_MB:-<task default>} storage_mb=${HARBOR_OVERRIDE_STORAGE_MB:-<task default>} gpus=${HARBOR_OVERRIDE_GPUS:-<task default>}
  Timeouts:     agent_sec=${HARBOR_AGENT_TIMEOUT_SEC:-<task default>} timeout_mult=${HARBOR_TIMEOUT_MULTIPLIER:-<default>} agent_mult=${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-<default>} verifier_mult=${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER:-<default>}
  Model info:   ${HOSTED_VLLM_MODEL_INFO:-<auto>}
  Job name:     ${JOB_NAME}
  Results dir:  ${RESULTS_DIR}
  Image:        ${BEAKER_IMAGE:+beaker:${BEAKER_IMAGE}}${BEAKER_DOCKER_IMAGE:+docker:${BEAKER_DOCKER_IMAGE}}
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
    --weka "oe-adapt-default:/weka/oe-adapt-default"
    --env-secret HF_TOKEN
    --env-secret "DOCKER_PAT=${DOCKER_PAT_SECRET:-hamishivi_DOCKER_PAT}"
    --env-secret "DAYTONA_API_KEY=${DAYTONA_API_KEY_SECRET:-hamishivi_DAYTONA_API_KEY}"
    --env "MODEL_PATH=${MODEL_PATH}"
    --env "MODEL_REVISION=${REVISION}"
    --env "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    --env "HARBOR_MODEL_NAME=${HARBOR_MODEL_NAME}"
    --env "VLLM_VERSION=${VLLM_VERSION}"
    --env "VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER}"
    --env "VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY}"
    --env "VLLM_PORT=${VLLM_PORT}"
    --env "TP_SIZE=${TP_SIZE}"
    --env "DP_SIZE=${DP_SIZE}"
    --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    --env "DATASET=${DATASET}"
    --env "HARBOR_ENV=${HARBOR_ENV}"
    --env "AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH}"
    --env "EXTRA_AGENT_KWARGS=${EXTRA_AGENT_KWARGS}"
    --env "EXTRA_AGENT_ENVS=${EXTRA_AGENT_ENVS}"
    --env "EXTRA_UV_PIP_INSTALLS=${EXTRA_UV_PIP_INSTALLS}"
    --env "HOSTED_VLLM_MODEL_INFO=${HOSTED_VLLM_MODEL_INFO}"
    --env "N_CONCURRENT=${N_CONCURRENT}"
    --env "N_ATTEMPTS=${N_ATTEMPTS}"
    --env "N_TASKS=${N_TASKS}"
    --env "HARBOR_OVERRIDE_CPUS=${HARBOR_OVERRIDE_CPUS}"
    --env "HARBOR_OVERRIDE_MEMORY_MB=${HARBOR_OVERRIDE_MEMORY_MB}"
    --env "HARBOR_OVERRIDE_STORAGE_MB=${HARBOR_OVERRIDE_STORAGE_MB}"
    --env "HARBOR_OVERRIDE_GPUS=${HARBOR_OVERRIDE_GPUS}"
    --env "HARBOR_TIMEOUT_MULTIPLIER=${HARBOR_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=${HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC}"
    --env "JOB_NAME=${JOB_NAME}"
    --env BEAKER_ALLOW_SUBCONTAINERS=1
    --env BEAKER_SKIP_DOCKER_SOCKET=1
    --host-networking
    --propagate-failure
    --no-python
)

if [ -n "$BEAKER_IMAGE" ]; then
    GANTRY_CMD+=(--beaker-image "$BEAKER_IMAGE")
elif [ -n "$BEAKER_DOCKER_IMAGE" ]; then
    GANTRY_CMD+=(--docker-image "$BEAKER_DOCKER_IMAGE")
else
    echo "error: either --image or --docker-image must be set" >&2
    exit 1
fi

if [ -n "$BEAKER_SCRIPTS_DATASET" ]; then
    GANTRY_CMD+=(--dataset "${BEAKER_SCRIPTS_DATASET}:/uploaded-beaker-scripts")
else
    GANTRY_CMD+=(--upload "$REPO_ROOT/scripts/beaker:/uploaded-beaker-scripts")
fi

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
