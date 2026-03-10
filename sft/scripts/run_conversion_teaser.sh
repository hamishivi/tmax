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
# (edit the variables below to tweak behaviour)
#
# Output: preprocessing/terminus2_sweagent_1pct/ (1% sample, separate from full).

NUM_WORKERS="$(nproc)"
OUTPUT_DIR="preprocessing/terminus2_sweagent_1pct"
SAMPLE_FRAC="0.01"
MAX_TURNS=999
NUM_EXAMPLES=3

echo "=== Teaser Run: ${SAMPLE_FRAC} ($(echo "${SAMPLE_FRAC} * 100" | bc)%) of each source ==="
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Max turns:  ${MAX_TURNS}"
echo "  Examples:   ${NUM_EXAMPLES} per source"
echo ""

python -m preprocessing.pipeline \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --sample-frac "${SAMPLE_FRAC}" \
    --max-turns "${MAX_TURNS}" \
    --num-examples "${NUM_EXAMPLES}" \
    "$@"

echo ""
echo "=== Teaser complete. Full report: ${OUTPUT_DIR}/conversion_report.json ==="
echo "=== To run the full pipeline:     bash scripts/run_conversion.sh ==="
