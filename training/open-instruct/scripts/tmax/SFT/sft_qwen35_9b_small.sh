#!/bin/bash
# SFT On Qwen3.5 9B on *just* tmax data ('small' SFT)
# We use a version of Qwen 3.5 with an interleaved reasoning chat template

set -eo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
DATASET="${DATASET:-allenai/tmax-sft}"
MASON_ARGS=(--cluster "${BEAKER_CLUSTER:?Set BEAKER_CLUSTER}")
if [ -n "${BEAKER_WORKSPACE:-}" ]; then
    MASON_ARGS+=(--workspace "$BEAKER_WORKSPACE")
fi
if [ -n "${BEAKER_BUDGET:-}" ]; then
    MASON_ARGS+=(--budget "$BEAKER_BUDGET")
fi
DATA_ARGS=(--dataset_mixer_list "$DATASET" 1.0)
if [ -n "${PRETOKENIZED_JSONL:-}" ]; then
    # This path must be visible on every training node.
    DATA_ARGS=(--pretokenized_jsonl "$PRETOKENIZED_JSONL")
else
    DATA_ARGS+=(--dataset_mixer_list_config_names skill_tax_20260505_2.2k_combined_balanced_thinking_all --add_bos)
fi

uv run python mason.py \
    "${MASON_ARGS[@]}" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --gpus 8 \
    --no_auto_dataset_cache \
    --auto_output_dir_path "" \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_9b_small \
    --model_name_or_path hamishivi/Qwen3.5-9B \
    --tokenizer_name hamishivi/Qwen3.5-9B \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    "${DATA_ARGS[@]}" \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --output_dir "${OUTPUT_DIR:-/output}" \
    --seed 42
