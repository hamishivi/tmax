#!/usr/bin/env bash
set -euo pipefail

# Build and verify all 9 domain base images.
#
# Usage:
#   bash rl_data/containers/build_bases.sh
#   bash rl_data/containers/build_bases.sh --force   # rebuild even if .sif exists

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=false

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
    esac
done

DOMAINS=(
    security
    software_engineering
    file_operations
    data_querying
    data_science
    debugging
    scientific_computing
    data_processing
    system_administration
)

PASS=0
FAIL=0
SKIP=0

for domain in "${DOMAINS[@]}"; do
    def_file="${SCRIPT_DIR}/base_${domain}.def"
    sif_file="${SCRIPT_DIR}/base_${domain}.sif"

    if [[ ! -f "$def_file" ]]; then
        echo "MISSING  $def_file"
        FAIL=$((FAIL + 1))
        continue
    fi

    if [[ -f "$sif_file" && "$FORCE" == "false" ]]; then
        echo "SKIP     base_${domain}.sif (already exists, use --force to rebuild)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  Building base_${domain}.sif ..."
    echo "================================================================"

    if apptainer build "$sif_file" "$def_file" 2>&1; then
        echo ""
        echo "  Build OK. Verifying..."

        # Verify: python3, pip, pytest available
        verify_output=$(apptainer exec "$sif_file" bash -c '
            set -e
            python3 --version
            pip3 --version
            python3 -c "import pytest; print(\"pytest \" + pytest.__version__)"
            echo "user_exists=$(id -u user 2>/dev/null || echo no)"
            echo "VERIFY_OK"
        ' 2>&1) || true

        if echo "$verify_output" | grep -q "VERIFY_OK"; then
            echo "  PASS     base_${domain}.sif"
            echo "$verify_output" | head -4 | sed 's/^/    /'
            PASS=$((PASS + 1))
        else
            echo "  FAIL     base_${domain}.sif (verification failed)"
            echo "$verify_output" | tail -5 | sed 's/^/    /'
            FAIL=$((FAIL + 1))
            rm -f "$sif_file"
        fi
    else
        echo "  FAIL     base_${domain}.sif (build failed)"
        FAIL=$((FAIL + 1))
        rm -f "$sif_file"
    fi
done

echo ""
echo "================================================================"
echo "  SUMMARY: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "================================================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
