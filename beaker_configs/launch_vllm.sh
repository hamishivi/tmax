#!/usr/bin/env bash
#
# Launch a vLLM server on Beaker for TassieAgent eval.
#
# Usage:
#   ./beaker_configs/launch_vllm.sh <model_path> [options]
#
# Examples:
#   # Serve a HuggingFace model with a specific revision
#   ./beaker_configs/launch_vllm.sh allenai/open_instruct_dev \
#       --revision sft_qwen3_4b_tmax_4node \
#       --name sft-4b-oi-big
#
#   # Serve with multiple GPUs
#   ./beaker_configs/launch_vllm.sh allenai/open_instruct_dev \
#       --revision sft_qwen3_4b_tmax_4node \
#       --name sft-4b-oi-big \
#       --gpus 2
#
#   # Custom port and max context length
#   ./beaker_configs/launch_vllm.sh Qwen/Qwen3-4B \
#       --name qwen3-4b \
#       --port 8008 \
#       --max-model-len 16384

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
REVISION="main"
SERVED_MODEL_NAME=""
GPU_COUNT=1
TP_SIZE=""
DP_SIZE=""
PORT=8008
MAX_MODEL_LEN=""
TOOL_CALL_PARSER="hermes"
CLUSTER="ai2/saturn"
BUDGET="ai2/oe-adapt"
PRIORITY="high"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/open-instruct-dev}"
EXTRA_ARGS=""

usage() {
    echo "Usage: $0 <model_path> [options]"
    echo ""
    echo "Options:"
    echo "  --revision REV       HuggingFace revision/branch (default: main)"
    echo "  --name NAME          Served model name (default: derived from model path)"
    echo "  --gpus N             Number of GPUs (default: 1)"
    echo "  --tp N               Tensor parallel size (default: auto — 1 for small models, gpus for large)"
    echo "  --dp N               Data parallel size (default: auto — gpus/tp)"
    echo "  --port PORT          Port to serve on (default: 8008)"
    echo "  --max-model-len LEN  Max context length (default: vllm auto)"
    echo "  --cluster CLUSTER    Beaker cluster (default: ai2/saturn)"
    echo "  --budget BUDGET      Beaker budget (default: ai2/oe-adapt)"
    echo "  --priority PRIORITY  Priority: high, normal, low (default: high)"
    echo "  --workspace WS       Beaker workspace (default: ai2/tmax)"
    echo "  --tool-call-parser P Tool call parser (default: hermes)"
    echo "  --extra-args ARGS    Extra arguments to pass to vllm serve"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

MODEL_PATH="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --revision) REVISION="$2"; shift 2 ;;
        --name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpus) GPU_COUNT="$2"; shift 2 ;;
        --tp) TP_SIZE="$2"; shift 2 ;;
        --dp) DP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --priority) PRIORITY="$2"; shift 2 ;;
        --workspace) BEAKER_WORKSPACE="$2"; shift 2 ;;
        --tool-call-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Derive served model name from model path if not set
if [ -z "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME="$(basename "$MODEL_PATH")"
fi

# Auto-select TP/DP split if not explicitly set
# Small models (fit on 1 GPU) benefit from DP over TP
if [ -z "$TP_SIZE" ] && [ -z "$DP_SIZE" ]; then
    TP_SIZE=1
    DP_SIZE=${GPU_COUNT}
elif [ -z "$TP_SIZE" ]; then
    TP_SIZE=$((GPU_COUNT / DP_SIZE))
elif [ -z "$DP_SIZE" ]; then
    DP_SIZE=$((GPU_COUNT / TP_SIZE))
fi

# Build the vllm command
VLLM_CMD="uvx vllm serve ${MODEL_PATH}"
VLLM_CMD+=" --revision ${REVISION}"
VLLM_CMD+=" --tokenizer-revision ${REVISION}"
VLLM_CMD+=" --served-model-name ${SERVED_MODEL_NAME}"
VLLM_CMD+=" --enable-auto-tool-choice"
VLLM_CMD+=" --tool-call-parser ${TOOL_CALL_PARSER}"
VLLM_CMD+=" --port ${PORT}"
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_CMD+=" --max-model-len ${MAX_MODEL_LEN}"
fi
VLLM_CMD+=" --gpu-memory-utilization 0.85"
VLLM_CMD+=" --tensor-parallel-size ${TP_SIZE}"
if [ "${DP_SIZE}" -gt 1 ]; then
    VLLM_CMD+=" --data-parallel-size ${DP_SIZE}"
fi
if [ -n "${EXTRA_ARGS}" ]; then
    VLLM_CMD+=" ${EXTRA_ARGS}"
fi

BEAKER_NAME="vllm-${SERVED_MODEL_NAME}"

echo "=== Launching vLLM on Beaker ==="
echo "  Model:      ${MODEL_PATH}"
echo "  Revision:   ${REVISION}"
echo "  Name:       ${SERVED_MODEL_NAME}"
echo "  GPUs:       ${GPU_COUNT} (TP=${TP_SIZE}, DP=${DP_SIZE})"
echo "  Port:       ${PORT}"
echo "  Max len:    ${MAX_MODEL_LEN:-auto}"
echo "  Cluster:    ${CLUSTER}"
echo "  Budget:     ${BUDGET}"
echo "  Priority:   ${PRIORITY}"
echo ""

# Generate config from template with substitutions
TMP_YAML=$(mktemp /tmp/vllm-beaker-XXXXXXXX).yaml
cat > "$TMP_YAML" << YAML
version: v2
budget: ${BUDGET}
description: "VLLM Server: ${SERVED_MODEL_NAME} (${MODEL_PATH}@${REVISION})"
tasks:
  - name: "vllm-job"
    image:
      beaker: ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0
    hostNetworking: true
    command: ["/bin/sh", "-c"]
    arguments:
      - "${VLLM_CMD}"
    envVars:
      - name: HF_TOKEN
        secret: HF_TOKEN
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
      priority: urgent
      preemptible: true
YAML

echo "Generated config:"
cat "$TMP_YAML"
echo ""

beaker experiment create "$TMP_YAML" --name "$BEAKER_NAME" --workspace "$BEAKER_WORKSPACE"

rm -f "$TMP_YAML"
