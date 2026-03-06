#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3.5-4B"
ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh200.yaml"
NUM_GPUS=8

# Data
SUBSETS="dataset_adapters skill_based_easy skill_based_medium skill_based_mixed"
SEED=42
# SAMPLE_FRAC=0.1  # uncomment for a quick test run
# Optional: path to a pre-tokenized dataset created by pre_tokenize.py
TOKENIZED_DATASET="/gpfs/scrubbed/osey/tmax/sft/data/tokenized_nemotron-terminal_0.1_42"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_PATH="/gpfs/scrubbed/osey/tmax/sft/output"
MODEL_NAME=$(basename "$MODEL")

if [ -n "${TOKENIZED_DATASET:-}" ]; then
    DATA_NAME=""
    for path in $TOKENIZED_DATASET; do
        b=$(basename "$path" | sed 's/^tokenized_//')
        if [ -z "$DATA_NAME" ]; then
            DATA_NAME="$b"
        else
            DATA_NAME="${DATA_NAME}_${b}"
        fi
    done
else
    DATA_NAME="${SUBSETS// /-}"
    if [ -n "${SAMPLE_FRAC:-}" ]; then
        DATA_NAME="${DATA_NAME}_frac${SAMPLE_FRAC}"
    fi
    DATA_NAME="${DATA_NAME}_seed${SEED}"
fi

OUTPUT_DIR="${BASE_PATH}/${MODEL_NAME}_${DATA_NAME}"

# Training parameters. Match nemontron-terminal-8B
GLOBAL_BATCH_SIZE=128
MAX_LENGTH=65536 # 32768 * 2
NUM_EPOCHS=2
LR=2e-5

# Logging / saving (fractional = ratio of total steps; 0.05 ≈ every 0.1 epoch)
LOGGING_STEPS=0.01
SAVE_STEPS=0.05

# ── Launch ───────────────────────────────────────────────────────────────────
DATA_ARGS=(--subsets $SUBSETS)
if [ -n "${SAMPLE_FRAC:-}" ]; then
    DATA_ARGS+=(--sample_frac "$SAMPLE_FRAC")
fi
if [ -n "$TOKENIZED_DATASET" ]; then
    # We deliberately don't quote $TOKENIZED_DATASET here so multiple paths split on spaces
    DATA_ARGS=(--tokenized_dataset_path $TOKENIZED_DATASET)
fi

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    train.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    "${DATA_ARGS[@]}" \
    --num_gpus "$NUM_GPUS" \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LR" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --seed "$SEED" \
    --dataset_num_proc 1
