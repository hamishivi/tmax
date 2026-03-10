#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Sharded conversion run ───────────────────────────────────────────
#
# Splits each source across K shards so multiple SLURM jobs (or local
# background processes) can convert in parallel, then merges outputs.
#
# Usage with SLURM array:
#   sbatch --array=0-3 scripts/run_conversion_sharded.sh
#   # After all tasks finish:
#   bash scripts/run_conversion_sharded.sh --merge
#
# Usage without SLURM (runs shards sequentially, still useful for testing):
#   bash scripts/run_conversion_sharded.sh --num-shards 4
#   bash scripts/run_conversion_sharded.sh --merge
#
# Options:
#   --num-shards K      Total shards (default: $SLURM_ARRAY_TASK_COUNT or 4)
#   --sample-frac F     Fraction of each source (default: 0.01 for teaser)
#   --merge             Merge previously-written shard dirs and exit
#   --upload [--public] Push merged output to HF
#
# Output:  output/preprocessing/sharded/shard_{0..K-1}/
# Merged:  output/preprocessing/terminus2_sweagent_1pct/

NUM_WORKERS="$(nproc)"
SAMPLE_FRAC="0.01"
MAX_TURNS=999
NUM_EXAMPLES=3
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-}"
BASE_DIR="output/preprocessing/sharded"
MERGED_DIR="output/preprocessing/terminus2_sweagent_1pct"
DO_MERGE=false
UPLOAD=false
UPLOAD_FLAGS=()
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-shards)   NUM_SHARDS="$2"; shift 2 ;;
        --sample-frac)  SAMPLE_FRAC="$2"; shift 2 ;;
        --merge)        DO_MERGE=true; shift ;;
        --upload)       UPLOAD=true; shift ;;
        --public)       UPLOAD_FLAGS+=(--public); shift ;;
        --repo)         UPLOAD_FLAGS+=(--repo "$2"); shift 2 ;;
        *)              PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

# ── Merge mode ────────────────────────────────────────────────────────
if [ "${DO_MERGE}" = true ]; then
    SHARD_DIRS=()
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_DIRS+=("${BASE_DIR}/shard_${i}")
    done

    echo "=== Merging ${NUM_SHARDS} shards → ${MERGED_DIR} ==="
    python -m preprocessing.pipeline \
        --merge-shards "${SHARD_DIRS[@]}" \
        --output-dir "${MERGED_DIR}"

    if [ "${UPLOAD}" = true ]; then
        echo ""
        bash scripts/upload_data_to_hf.sh --input-dir "${MERGED_DIR}" "${UPLOAD_FLAGS[@]+"${UPLOAD_FLAGS[@]}"}"
    fi
    exit 0
fi

# ── Shard mode ────────────────────────────────────────────────────────
# When not inside a SLURM array, run all shards sequentially.
if [ -z "${SHARD_INDEX}" ]; then
    echo "=== Running ${NUM_SHARDS} shards sequentially (no SLURM array detected) ==="
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        echo ""
        echo "--- Shard ${i}/${NUM_SHARDS} ---"
        python -m preprocessing.pipeline \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${BASE_DIR}/shard_${i}" \
            --sample-frac "${SAMPLE_FRAC}" \
            --max-turns "${MAX_TURNS}" \
            --num-examples "${NUM_EXAMPLES}" \
            --shard-index "${i}" \
            --num-shards "${NUM_SHARDS}" \
            "${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}"
    done
    echo ""
    echo "=== All shards done.  Run with --merge to combine. ==="
    exit 0
fi

# Inside SLURM array: process only this task's shard.
OUTPUT_DIR="${BASE_DIR}/shard_${SHARD_INDEX}"

echo "=== Shard ${SHARD_INDEX}/${NUM_SHARDS} ==="
echo "  Workers:      ${NUM_WORKERS}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Sample frac:  ${SAMPLE_FRAC}"
echo ""

python -m preprocessing.pipeline \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --sample-frac "${SAMPLE_FRAC}" \
    --max-turns "${MAX_TURNS}" \
    --num-examples "${NUM_EXAMPLES}" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    "${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}"

echo ""
echo "=== Shard ${SHARD_INDEX} complete: ${OUTPUT_DIR}/conversion_report.json ==="
