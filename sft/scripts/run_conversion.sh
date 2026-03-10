#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Conversion pipeline launcher ──────────────────────────────────────
#
# Converts Terminus-2 traces into SWE-agent format for SFT training.
#
# Usage:
#   bash scripts/run_conversion.sh                        # full pipeline
# (edit the variables below to tweak behaviour; for ad-hoc overrides you
#  can still pass CLI flags like --sample or --sample-frac if needed)
#
# Sources are defined in preprocessing/config/sources.yaml.
# Output: preprocessing/terminus2_sweagent/

NUM_WORKERS="$(nproc)"
OUTPUT_DIR="preprocessing/terminus2_sweagent"
MAX_TURNS=999

echo "=== Terminus-2 -> SWE-Agent Conversion Pipeline ==="
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Max turns:  ${MAX_TURNS}"
echo ""

python -m preprocessing.pipeline \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-turns "${MAX_TURNS}" \
    "$@"

echo ""
echo "=== Done. Report: ${OUTPUT_DIR}/conversion_report.json ==="
