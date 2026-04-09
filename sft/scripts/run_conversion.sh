#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Full conversion pipeline ─────────────────────────────────────────
#
# Converts ALL Terminus-2 traces into SWE-agent format for SFT training.
# Sources are defined in preprocessing/config/sources.yaml.
#
# Usage:
#   bash scripts/run_conversion.sh
#   bash scripts/run_conversion.sh --upload              # also push to HF
#   bash scripts/run_conversion.sh --upload --public     # push as public
#   bash scripts/run_conversion.sh --include-partial     # keep truncated traces
#
# Output: output/preprocessing/terminus2_sweagent/

NUM_WORKERS="$(nproc)"
OUTPUT_DIR="output/preprocessing/terminus2_sweagent_full_20260403"
MAX_TURNS=999
NUM_EXAMPLES=3
HF_REPO="osieosie/tmax-sft-full-20260403"
UPLOAD=true
UPLOAD_FLAGS=""
INCLUDE_PARTIAL=true

# Parse our flags, pass the rest through to pipeline.py
PIPELINE_EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)            UPLOAD=true; shift ;;
        --public)            UPLOAD_FLAGS="${UPLOAD_FLAGS} --public"; shift ;;
        --repo)              HF_REPO="$2"; shift 2 ;;
        --include-partial)   INCLUDE_PARTIAL=true; shift ;;
        *)                   PIPELINE_EXTRA="${PIPELINE_EXTRA} $1"; shift ;;
    esac
done

PARTIAL_FLAG=""
if [ "${INCLUDE_PARTIAL}" = true ]; then
    PARTIAL_FLAG="--include-partial"
fi

echo "=== Terminus-2 -> SWE-Agent Full Conversion Pipeline ==="
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Max turns:  ${MAX_TURNS}"
echo "  Examples:   ${NUM_EXAMPLES} per source"
echo "  Partial:    ${INCLUDE_PARTIAL}"
echo ""

# shellcheck disable=SC2086
python -m preprocessing.pipeline \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-turns "${MAX_TURNS}" \
    --num-examples "${NUM_EXAMPLES}" \
    ${PARTIAL_FLAG} \
    ${PIPELINE_EXTRA}

echo ""
echo "=== Done. Full report: ${OUTPUT_DIR}/conversion_report.json ==="

if [ "${UPLOAD}" = true ]; then
    echo ""
    # shellcheck disable=SC2086
    bash scripts/upload_data_to_hf.sh --repo "${HF_REPO}" --input-dir "${OUTPUT_DIR}" ${UPLOAD_FLAGS}
else
    echo "=== To upload to HF:  bash scripts/upload_data_to_hf.sh --repo ${HF_REPO} --input-dir ${OUTPUT_DIR} ==="
fi
