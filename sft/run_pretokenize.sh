#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL=Qwen/Qwen3.5-4B

# Data
SUBSETS="dataset_adapters skill_based_easy skill_based_medium skill_based_mixed"
SEED=42
SAMPLE_FRAC=0.1  # set empty to disable sub-sampling

# Tokenization
MAX_LENGTH=65536 # 32768 * 2
NUM_PROC=64
OUTPUT_PATH=./tokenized_data

# ── Run pre-tokenization ─────────────────────────────────────────────────────
python pre_tokenize.py \
    --model_name_or_path "$MODEL" \
    --output_path "$OUTPUT_PATH" \
    --subsets $SUBSETS \
    --max_length "$MAX_LENGTH" \
    --num_proc "$NUM_PROC" \
    --seed "$SEED" \
    ${SAMPLE_FRAC:+--sample_frac "$SAMPLE_FRAC"}

