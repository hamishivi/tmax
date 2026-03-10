#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ───────────────────────────────────────────────────────────────────
MODEL=Qwen/Qwen3.5-4B

# Base path
BASE_PATH="/gpfs/scrubbed/osey/tmax"

# Data
SUBSETS="dataset_adapters skill_based_easy skill_based_medium skill_based_mixed"
SEED=42
SAMPLE_FRAC=0.05  # set empty to disable sub-sampling

# Tokenization
MAX_LENGTH=65536 # 32768 * 2
NUM_PROC=16 

# Sharding and path construction
DATASET_NAME="nemotron-terminal"
FRAC_STR=${SAMPLE_FRAC:-"full"}
BASE_NAME="${DATASET_NAME}_${FRAC_STR}_${SEED}"

SHARD_ARGS=()
if [ -n "${NUM_SHARDS:-}" ] && [ -n "${SHARD_INDEX:-}" ]; then
    OUTPUT_PATH="${BASE_PATH}/sft/data/tokenized_${BASE_NAME}_shard_${SHARD_INDEX}_of_${NUM_SHARDS}"
    SHARD_ARGS=(--num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
else
    OUTPUT_PATH="${BASE_PATH}/sft/data/tokenized_${BASE_NAME}"
fi

# ── Run pre-tokenization ─────────────────────────────────────────────────────
python pre_tokenize.py \
    --model_name_or_path "$MODEL" \
    --output_path "$OUTPUT_PATH" \
    --subsets $SUBSETS \
    --max_length "$MAX_LENGTH" \
    --num_proc "$NUM_PROC" \
    --seed "$SEED" \
    "${SHARD_ARGS[@]}" \
    ${SAMPLE_FRAC:+--sample_frac "$SAMPLE_FRAC"}

