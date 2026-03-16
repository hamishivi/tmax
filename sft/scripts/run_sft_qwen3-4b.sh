#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-4B-Instruct-2507"
BACKEND="deepspeed" # "fsdp" or "deepspeed"

# Accelerate config files (pick one per backend)
FSDP_CONFIG="configs/accelerate_fsdp_8xh200.yaml"
DS_CONFIG="configs/accelerate_ds_z3_sp4_8xh200.yaml"
# Alternatives:
#   configs/accelerate_ds_z3_sp8_8xh200.yaml   (SP=8 + DP=1, max sequence length)

NUM_GPUS=8

# Data
SUBSETS="dataset_adapters skill_based_easy skill_based_medium skill_based_mixed"
SEED=42
# SAMPLE_FRAC=0.1  # uncomment for a quick test run
TOKENIZED_DATASET="/gpfs/scrubbed/osey/tmax/sft/data/tokenized_tbmax_terminus2_sweagent_full_20260315_qwen3_42"

# Subsampling (comment out to train on the full dataset)
# MAX_TRAIN_SAMPLES=10000

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

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
    DATA_NAME="${DATA_NAME}_n${MAX_TRAIN_SAMPLES}"
fi

OUTPUT_DIR="${BASE_PATH}/${MODEL_NAME}_${DATA_NAME}"
RUN_NAME="${MODEL_NAME}_${DATA_NAME}"

# Training parameters
GLOBAL_BATCH_SIZE=128
MAX_LENGTH=65536
NUM_EPOCHS=2
LR=2e-5

# Logging / saving
LOGGING_STEPS=1
SAVE_STEPS=0.1
WANDB_PROJECT="tmax-sft"

# ── Launch ───────────────────────────────────────────────────────────────────
DATA_ARGS=(--subsets $SUBSETS)
if [ -n "${SAMPLE_FRAC:-}" ]; then
    DATA_ARGS+=(--sample_frac "$SAMPLE_FRAC")
fi
if [ -n "$TOKENIZED_DATASET" ]; then
    DATA_ARGS=(--tokenized_dataset_path $TOKENIZED_DATASET)
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"
echo "Starting training. Logging output to: $LOG_FILE"

export TRITON_CACHE_DIR="/gpfs/scrubbed/osey/.triton_cache"

COMMON_ARGS=(
    train.py
    --model_name_or_path "$MODEL"
    --output_dir "$OUTPUT_DIR"
    "${DATA_ARGS[@]}"
    --num_gpus "$NUM_GPUS"
    --per_device_train_batch_size 1  # MUST be 1 with Ulysses SP: SP registers
                                     # attention shapes on the first forward pass,
                                     # so batch size must be constant. Effective
                                     # batch size is controlled via global_batch_size
                                     # and gradient accumulation instead.
    --max_length "$MAX_LENGTH"
    --num_train_epochs "$NUM_EPOCHS"
    --learning_rate "$LR"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$SAVE_STEPS"
    --seed "$SEED"
    --dataset_num_proc 1
    --packing
    --optim adamw_torch_fused
    --wandb_project "$WANDB_PROJECT"
    --run_name "$RUN_NAME"
)

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
    COMMON_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

if [ "$BACKEND" = "fsdp" ]; then
    echo "Using FSDP backend with config: $FSDP_CONFIG"
    accelerate launch \
        --config_file "$FSDP_CONFIG" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
elif [ "$BACKEND" = "deepspeed" ]; then
    echo "Using DeepSpeed Ulysses SP with config: $DS_CONFIG"
    accelerate launch \
        --config_file "$DS_CONFIG" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
else
    echo "Error: Unknown BACKEND '$BACKEND'"
    exit 1
fi
