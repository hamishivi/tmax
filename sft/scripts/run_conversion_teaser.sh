#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Teaser conversion run ─────────────────────────────────────────────
#
# Runs the conversion pipeline on 1% of each registered source dataset.
# Use this for quick iteration, testing, and sanity-checking before a
# full run.  Takes ~1-5 minutes depending on download speed and CPU count.
#
# Usage:
#   bash scripts/run_conversion_teaser.sh
#   bash scripts/run_conversion_teaser.sh --upload              # also push to HF
#   bash scripts/run_conversion_teaser.sh --upload --public     # push as public
#   bash scripts/run_conversion_teaser.sh --include-partial     # keep truncated traces (code trajectories not submitted.)
#
# Output: preprocessing/terminus2_sweagent_1pct/ (1% sample, separate from full).

NUM_WORKERS="$(nproc)"
OUTPUT_DIR="output/preprocessing/terminus2_sweagent_1pct"
SAMPLE_FRAC="0.01"
MAX_TURNS=999
NUM_EXAMPLES=3
HF_REPO="osieosie/tmax-sft-preview"
UPLOAD=false
UPLOAD_FLAGS=""

# Parse our flags, pass the rest through to pipeline.py
PIPELINE_EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)  UPLOAD=true; shift ;;
        --public)  UPLOAD_FLAGS="${UPLOAD_FLAGS} --public"; shift ;;
        --repo)    HF_REPO="$2"; shift 2 ;;
        *)         PIPELINE_EXTRA="${PIPELINE_EXTRA} $1"; shift ;;
    esac
done

echo "=== Teaser Run: ${SAMPLE_FRAC} ($(echo "${SAMPLE_FRAC} * 100" | bc)%) of each source ==="
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Max turns:  ${MAX_TURNS}"
echo "  Examples:   ${NUM_EXAMPLES} per source"
echo ""

# shellcheck disable=SC2086
python -m preprocessing.pipeline \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --sample-frac "${SAMPLE_FRAC}" \
    --max-turns "${MAX_TURNS}" \
    --num-examples "${NUM_EXAMPLES}" \
    ${PIPELINE_EXTRA}

echo ""
echo "=== Teaser complete. Full report: ${OUTPUT_DIR}/conversion_report.json ==="

if [ "${UPLOAD}" = true ]; then
    echo ""
    # shellcheck disable=SC2086
    bash scripts/upload_data_to_hf.sh --repo "${HF_REPO}" --input-dir "${OUTPUT_DIR}" ${UPLOAD_FLAGS}
else
    echo "=== To upload to HF:  bash scripts/upload_data_to_hf.sh --repo ${HF_REPO} --input-dir ${OUTPUT_DIR} ==="
fi

echo "=== To run the full pipeline:     bash scripts/run_conversion.sh ==="
