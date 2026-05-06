#!/usr/bin/env bash
set -euo pipefail
# Capture the absolute script dir BEFORE we cd, so we can invoke sibling
# scripts (e.g. upload_data_to_hf.sh) regardless of where the user launched
# this driver from.
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${_SCRIPT_DIR}/.."

# ── skill_tax 1k trajectory → SFT conversion ────────────────────────────
#
# Converts the per-task agent-loop trajectories produced by
# `rl_data/scripts/generate_solutions/run_generate_solutions_skill_tax_combined_2.5k.sh`
# (renamed in May 2026; previously `..._skill_tax_1k.sh`. The current script
# defaults to the 2.5k combined corpus, but this conversion driver is still
# pinned to the legacy 1k corpus path — see TASKS_DIR below.)
# Under tasks_skill_tax_20260324_1k/task_*/solutions/<MODEL_TAG>_summary.json
# into SFT parquet rows that match the existing tmax-sft-full-20260409 schema,
# then uploads them to a NEW HF dataset repo as TWO configs:
#
#   skill_tax_20260324_1k_all              -- every trajectory (8 per task)
#   skill_tax_20260324_1k_only_success     -- only the verified-passing ones
#
# Both configs live in one repo so the trainer can target either via
# `--dataset-config-name`.  The trainer code in sft/scripts/run_sft_*.sh
# already understands this format -- no code changes needed.
#
# Usage:
#   # All defaults: produces both variants and uploads.
#   bash sft/scripts/run_conversion_skill_tax_1k.sh
#
#   # Just convert; don't upload.
#   bash sft/scripts/run_conversion_skill_tax_1k.sh --no-upload
#
#   # Override the source model tag (e.g. when re-running from a different model).
#   MODEL_TAG=hosted_vllm_Qwen_Qwen3.5-9B \
#     bash sft/scripts/run_conversion_skill_tax_1k.sh
#
#   # Convert vanillux trajectories (default is bash). Reads
#   # <task>/solutions/<MODEL_TAG>_vanillux_summary.json instead of
#   # <task>/solutions/<MODEL_TAG>_summary.json.
#   HARNESS=vanillux \
#     bash sft/scripts/run_conversion_skill_tax_1k.sh
#
#   # Convert the combined-2.5k SFT corpus (vanillux trajectories from both
#   # local-Qwen and Gemini teachers) — full env override for that corpus:
#   TASKS_DIR=rl_data/output/tasks_skill_tax_combined_20260506_2.5k \
#   MODEL_TAG=hosted_vllm_Qwen_Qwen3.6-27B \
#   HARNESS=vanillux \
#   OUTPUT_DIR=output/preprocessing/skill_tax_combined_20260506_2.5k \
#   HF_REPO=osieosie/tmax-sft-skill-tax-combined-20260506-2.5k \
#   NAME_ALL=skill_tax_combined_20260506_2.5k_all \
#   NAME_ONLY_SUCCESS=skill_tax_combined_20260506_2.5k_only_success \
#     bash sft/scripts/run_conversion_skill_tax_1k.sh
#
# Requirements (for upload):
#   - hf auth login   (or HF_TOKEN env var set)

# ---- Parameters (override via env) ----
TASKS_DIR="${TASKS_DIR:-/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_20260324_1k}"
# MODEL_TAG matches the summary filename pattern:
#   solutions/<MODEL_TAG>[_<HARNESS>]_summary.json
# Default reflects the legacy 1k run (bash harness, hosted_vllm_Qwen_Qwen3.5-27B).
MODEL_TAG="${MODEL_TAG:-hosted_vllm_Qwen_Qwen3.5-27B}"
# HARNESS selects which summary file to read at conversion time. Must match
# the --harness passed to the solve script:
#   bash      -> <MODEL_TAG>_summary.json            (legacy 1k / 10k)
#   vanillux  -> <MODEL_TAG>_vanillux_summary.json   (combined 2.5k, rl_v2 5k)
HARNESS="${HARNESS:-bash}"
# Output dir under sft/output/preprocessing/ -- the timestamp is fixed in the
# default name so re-runs land in the same place (overwriting prior parquets);
# bump the suffix when you want a clean re-conversion.
OUTPUT_DIR="${OUTPUT_DIR:-output/preprocessing/skill_tax_20260324_1k}"
HF_REPO="${HF_REPO:-osieosie/tmax-sft-skill-tax-20260324-1k}"
PRIVATE_FLAG=""
UPLOAD=true

# Variant names also become HF config names (config_name = parquet stem with
# '-' -> '_'; we already use '_' so no rewriting).
NAME_ALL="${NAME_ALL:-skill_tax_20260324_1k_all}"
NAME_ONLY_SUCCESS="${NAME_ONLY_SUCCESS:-skill_tax_20260324_1k_only_success}"
# --------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-upload)  UPLOAD=false; shift ;;
        --upload)     UPLOAD=true; shift ;;
        --public)     PRIVATE_FLAG="--public"; shift ;;
        --private)    PRIVATE_FLAG="--private"; shift ;;
        --repo)       HF_REPO="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks-dir)  TASKS_DIR="$2"; shift 2 ;;
        --model-tag)  MODEL_TAG="$2"; shift 2 ;;
        --harness)    HARNESS="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== skill_tax 1k trajectory -> SFT conversion ==="
echo "  Tasks dir:   ${TASKS_DIR}"
echo "  Model tag:   ${MODEL_TAG}"
echo "  Harness:     ${HARNESS}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  HF repo:     ${HF_REPO}"
echo "  Upload:      ${UPLOAD}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ---- Variant 1: ALL trajectories ----
echo "=== Variant 1/2: all trajectories -> ${NAME_ALL}.parquet ==="
uv run python -m preprocessing.convert_trajectories \
    --tasks-dir "${TASKS_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --harness "${HARNESS}" \
    --output-dir "${OUTPUT_DIR}" \
    --name "${NAME_ALL}"

# ---- Variant 2: only-success trajectories ----
echo ""
echo "=== Variant 2/2: success-only trajectories -> ${NAME_ONLY_SUCCESS}.parquet ==="
uv run python -m preprocessing.convert_trajectories \
    --tasks-dir "${TASKS_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --harness "${HARNESS}" \
    --output-dir "${OUTPUT_DIR}" \
    --name "${NAME_ONLY_SUCCESS}" \
    --filter-success

echo ""
echo "=== Conversion done.  Parquets + reports at: ${OUTPUT_DIR} ==="
ls -la "${OUTPUT_DIR}"

if [ "${UPLOAD}" = true ]; then
    echo ""
    echo "=== Uploading to ${HF_REPO} ==="
    # shellcheck disable=SC2086
    bash "${_SCRIPT_DIR}/upload_data_to_hf.sh" \
        --repo "${HF_REPO}" \
        --input-dir "${OUTPUT_DIR}" \
        ${PRIVATE_FLAG}
else
    echo ""
    echo "=== Skipping upload (--no-upload).  To upload later (from anywhere): ==="
    echo "    bash ${_SCRIPT_DIR}/upload_data_to_hf.sh --repo ${HF_REPO} --input-dir $(pwd)/${OUTPUT_DIR}"
fi
