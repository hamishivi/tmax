#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ───────────────────────────────────────────────────────────────────
MODEL=Qwen/Qwen3.5-4B

# Base path
BASE_PATH="/gpfs/scrubbed/osey/tmax"

# Data — points to the full converted output from run_conversion.sh
DATA_DIR="${BASE_PATH}/sft/output/preprocessing/terminus2_sweagent_full_20260310_v2"
SEED=42

# Tokenization
MAX_LENGTH=65536
NUM_PROC="$(nproc)"

# Output path
DATASET_NAME="tbmax_terminus2_sweagent_full_20260310_v2_qwen3.5"
BASE_NAME="${DATASET_NAME}_${SEED}"

SHARD_ARGS=()
if [ -n "${NUM_SHARDS:-}" ] && [ -n "${SHARD_INDEX:-}" ]; then
    OUTPUT_PATH="${BASE_PATH}/sft/data/tokenized_${BASE_NAME}_shard_${SHARD_INDEX}_of_${NUM_SHARDS}"
    SHARD_ARGS=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
else
    OUTPUT_PATH="${BASE_PATH}/sft/data/tokenized_${BASE_NAME}"
fi

echo "=== Pre-tokenization: ${DATASET_NAME} ==="
echo "  Model:      ${MODEL}"
echo "  Data dir:   ${DATA_DIR}"
echo "  Output:     ${OUTPUT_PATH}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Num proc:   ${NUM_PROC}"
echo "  Seed:       ${SEED}"
echo ""

# ── Run pre-tokenization ─────────────────────────────────────────────────────
python pre_tokenize.py \
    --model_name_or_path "$MODEL" \
    --data_dir "$DATA_DIR" \
    --output_path "$OUTPUT_PATH" \
    --max_length "$MAX_LENGTH" \
    --num_proc "$NUM_PROC" \
    --seed "$SEED" \
    "${SHARD_ARGS[@]}"
