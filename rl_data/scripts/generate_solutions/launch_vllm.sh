#!/bin/bash
# Launch a vLLM OpenAI-compatible server for a local model so the rest of the
# pipeline (generate_solutions.py, taxonomy_classifier.py, etc.) can call it
# through litellm by setting:
#
#   export MODEL="hosted_vllm/<HF_MODEL>"
#   export HOSTED_VLLM_API_BASE="http://$HOST:$PORT/v1"
#
# Env:
#   MODEL            HF repo to serve.  Default: Qwen/Qwen2.5-Coder-7B-Instruct
#   PORT             Default: 8000
#   HOST             Bind address.  Default: 0.0.0.0
#   TP               Tensor-parallel size (== #GPUs).  Default: 1
#   MAX_LEN          Max model len.  Default: 32768
#   GPU_UTIL         GPU memory utilisation.  Default: 0.90
#   DTYPE            Default: bfloat16
#   VLLM_VERSION     vLLM package version for uvx. Default: 0.19.1
#   TOOL_CALL_PARSER Optional --tool-call-parser value (hermes for Qwen2.5).
#                    Set empty to disable.  Default: hermes
#
# For larger models, run under Slurm and request the appropriate GPUs.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

echo "=== vLLM ==="
echo "  model        : $MODEL"
echo "  bind         : $HOST:$PORT"
echo "  tp           : $TP"
echo "  max_model_len: $MAX_LEN"
echo "  dtype        : $DTYPE"
echo "  vllm version : $VLLM_VERSION"
echo "  parser       : ${TOOL_CALL_PARSER:-<none>}"
echo
echo "In your solve script, set:"
echo "  export MODEL=\"hosted_vllm/$MODEL\""
echo "  export HOSTED_VLLM_API_BASE=\"http://<this-node>:$PORT/v1\""
echo

EXTRA=()
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  EXTRA+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

exec uvx --from "vllm==${VLLM_VERSION}" vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --dtype "$DTYPE" \
    --served-model-name "$MODEL" \
    "${EXTRA[@]}"
