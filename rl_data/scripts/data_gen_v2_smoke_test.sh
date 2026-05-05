#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  v2 pipeline smoke test                                                    ║
# ║                                                                            ║
# ║  Run this on a login or build node BEFORE kicking off the full v2 SFT     ║
# ║  / RL generation jobs. It exercises every stage of the new pipeline at    ║
# ║  a small scale so issues surface in seconds, not in 48-hour SBATCH jobs.  ║
# ║                                                                            ║
# ║  Stages exercised:                                                         ║
# ║    A) v2 axis sampling distribution check  (fast, no LLM, no Apptainer)   ║
# ║    B) fixture_gen materialisation smoke    (fast, no LLM)                 ║
# ║    C) base_intricate.sif build             (slow, ~5-15 min, needs net)   ║
# ║    D) end-to-end task generation, NUM_TASKS=5, CORPUS_KIND=sft_v2         ║
# ║       (slow, needs Gemini API + Apptainer build perms; ~5 min)            ║
# ║                                                                            ║
# ║  Each stage gates on the previous one; a failure aborts before the more  ║
# ║  expensive stages run.                                                    ║
# ║                                                                            ║
# ║  Usage:                                                                    ║
# ║    bash rl_data/scripts/v2_smoke_test.sh           # all stages           ║
# ║    SKIP_BUILD=1 bash ... v2_smoke_test.sh          # skip stage C         ║
# ║    SKIP_LLM=1   bash ... v2_smoke_test.sh          # skip stage D         ║
# ║                                                                            ║
# ║  Required env (only for stage D):                                         ║
# ║    GEMINI_API_KEY                                                         ║
# ║    APPTAINER_DOCKER_USERNAME / APPTAINER_DOCKER_PASSWORD                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Coloured headers help spot stage boundaries in long logs.
hdr() { printf "\n\033[1;34m== %s ==\033[0m\n" "$1"; }
ok()  { printf "\033[1;32m  OK\033[0m  %s\n" "$1"; }
fail(){ printf "\033[1;31m  FAIL\033[0m %s\n" "$1"; exit 1; }

SMOKE_OUT="/tmp/v2_smoke_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE_OUT"
echo "Smoke artefacts -> $SMOKE_OUT"

# ─────────────────────────────────────────────────────────────────────────────
hdr "Stage A — v2 axis sampling distributions"

uv run python <<'PY'
import random
from collections import Counter
from rl_data.generator.task_template_gen import (
    random_user_msg, VERIFIER_KINDS, FIXTURE_KINDS, TASK_COMPLEXITY,
)

assert len(TASK_COMPLEXITY) == 4, f"expected 4 complexity levels, got {len(TASK_COMPLEXITY)}"
assert "exact_text" in VERIFIER_KINDS and len(VERIFIER_KINDS) == 5
assert "text_only" in FIXTURE_KINDS and len(FIXTURE_KINDS) == 7

# Legacy must be byte-identical: no v2 axis values ever sampled.
random.seed(42)
legacy = [random_user_msg("legacy")[2] for _ in range(500)]
assert all(m["verifier_kind"] == "exact_text" for m in legacy)
assert all(m["fixture_kind"] == "text_only" for m in legacy)
assert all(m["base_image"] is None for m in legacy)
assert all("intricate task" not in m["task_complexity"] for m in legacy)
print("  legacy: 500 samples, all legacy defaults preserved")

# sft_v2 (M=2): P(legacy verifier) = 1/3, P(intricate) = 2/3.
random.seed(42)
n = 2000
sft = [random_user_msg("sft_v2")[2] for _ in range(n)]
v_legacy = sum(1 for m in sft if m["verifier_kind"] == "exact_text") / n
f_legacy = sum(1 for m in sft if m["fixture_kind"] == "text_only") / n
intricate = sum(1 for m in sft if "intricate task" in m["task_complexity"]) / n
print(f"  sft_v2 (M=2): P(legacy verifier)={v_legacy:.3f} (expected ~0.333)")
print(f"                P(legacy fixture)={f_legacy:.3f} (expected ~0.333)")
print(f"                P(intricate)     ={intricate:.3f} (expected ~0.667)")
assert 0.28 < v_legacy < 0.39, "sft_v2 verifier distribution off"
assert 0.28 < f_legacy < 0.39, "sft_v2 fixture distribution off"
assert 0.61 < intricate < 0.72, "sft_v2 complexity distribution off"

# rl_v2 (M=1.5): P(legacy verifier) = 1/2.5 = 0.4, P(intricate) = 1.5/2.5 = 0.6.
random.seed(42)
rl = [random_user_msg("rl_v2")[2] for _ in range(n)]
v_legacy = sum(1 for m in rl if m["verifier_kind"] == "exact_text") / n
f_legacy = sum(1 for m in rl if m["fixture_kind"] == "text_only") / n
intricate = sum(1 for m in rl if "intricate task" in m["task_complexity"]) / n
print(f"  rl_v2 (M=1.5): P(legacy verifier)={v_legacy:.3f} (expected ~0.400)")
print(f"                 P(legacy fixture)={f_legacy:.3f} (expected ~0.400)")
print(f"                 P(intricate)     ={intricate:.3f} (expected ~0.600)")
assert 0.34 < v_legacy < 0.46, "rl_v2 verifier distribution off"
assert 0.34 < f_legacy < 0.46, "rl_v2 fixture distribution off"
assert 0.54 < intricate < 0.66, "rl_v2 complexity distribution off"

print("  all distributions within tolerance")
PY
ok "Stage A passed"

# ─────────────────────────────────────────────────────────────────────────────
hdr "Stage B — fixture_gen materialisation"

FIXTURE_OUT="$SMOKE_OUT/fixtures"
uv run python -m rl_data.generator.fixture_gen --out "$FIXTURE_OUT"

# Hard-fail if any fixture kind landed only a sentinel — but only on a host
# that has the relevant tool installed. We tolerate sentinels for
# stripped_binary if gcc is missing, video if ffmpeg is missing.
check_kind() {
    local kind="$1"  required_tool="$2"
    local fdir="$FIXTURE_OUT/$kind/fixtures"
    if [ ! -d "$fdir" ]; then
        fail "fixture_gen: $kind dir missing"
    fi
    local sentinel="$fdir/${kind%%_*}.unavailable.txt"
    # Special-case: stripped_binary uses 'binary.unavailable.txt'
    if [ "$kind" = "stripped_binary" ]; then sentinel="$fdir/binary.unavailable.txt"; fi
    if [ -f "$sentinel" ]; then
        if command -v "$required_tool" >/dev/null 2>&1; then
            fail "fixture_gen: $kind sentinelled despite $required_tool available"
        else
            echo "  $kind: sentinelled (no $required_tool on host) — that's OK on a login node"
            return 0
        fi
    fi
    local n_files
    n_files=$(find "$fdir" -type f | wc -l)
    if [ "$n_files" -lt 1 ]; then
        fail "fixture_gen: $kind produced no files"
    fi
    echo "  $kind: $n_files files materialised"
}
check_kind image           tesseract  # tesseract not strictly required for image gen
check_kind audio           true       # always synthesises (stdlib only)
check_kind video           ffmpeg
check_kind stripped_binary gcc
check_kind vendored_package true
check_kind multi_service_compose true

ok "Stage B passed"

# ─────────────────────────────────────────────────────────────────────────────
hdr "Stage C — base_intricate.sif build"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "  SKIP_BUILD=1 — skipping"
else
    DEF=rl_data/containers/base_intricate.def
    SIF=rl_data/containers/base_intricate.sif
    if [ -f "$SIF" ]; then
        echo "  $SIF already exists ($(du -h "$SIF" | awk '{print $1}')) — skipping rebuild."
        echo "  Set FORCE_BUILD=1 to rebuild."
        if [ "${FORCE_BUILD:-0}" = "1" ]; then
            rm -f "$SIF"
        fi
    fi
    if [ ! -f "$SIF" ]; then
        export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/gpfs/projects/h2lab/osey/apptainer_cache}"
        export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
        mkdir -p "$APPTAINER_TMPDIR"
        echo "  Building $SIF (this typically takes 5-15 minutes)..."
        apptainer build --force "$SIF" "$DEF" 2>&1 | tail -40
        if [ ! -f "$SIF" ]; then
            fail "apptainer build did not produce $SIF"
        fi
        echo "  Built: $(du -h "$SIF" | awk '{print $1}')"
    fi
    echo "  Verifying intricate stack inside SIF..."
    apptainer exec "$SIF" bash -c '
        set -e
        for cmd in python3 gcc rustc go ffmpeg tesseract upx file xxd; do
            command -v "$cmd" >/dev/null || { echo "missing: $cmd"; exit 1; }
        done
        python3 -c "import numpy, scipy, sklearn, PIL, torch; print(\"py-stack OK\")"
    ' || fail "intricate stack missing in SIF"
fi
ok "Stage C passed"

# ─────────────────────────────────────────────────────────────────────────────
hdr "Stage D — end-to-end task generation (NUM_TASKS=5, CORPUS_KIND=sft_v2)"

if [ "${SKIP_LLM:-0}" = "1" ]; then
    echo "  SKIP_LLM=1 — skipping"
else
    : "${GEMINI_API_KEY:?Set GEMINI_API_KEY for stage D (or pass SKIP_LLM=1)}"
    : "${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME for stage D (or pass SKIP_LLM=1)}"
    : "${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD for stage D (or pass SKIP_LLM=1)}"

    SMOKE_GEN_OUT="$SMOKE_OUT/tasks_smoke_sft_v2"
    rm -rf "$SMOKE_GEN_OUT"
    mkdir -p "$SMOKE_GEN_OUT"

    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/gpfs/projects/h2lab/osey/apptainer_cache}"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
    mkdir -p "$APPTAINER_TMPDIR"

    uv run python -c "
from pathlib import Path
from rl_data.generate_tasks import AsyncBatchConfig, run_pipeline
import json

cfg = AsyncBatchConfig(
    num_tasks=5,
    out_dir=Path('$SMOKE_GEN_OUT'),
    model='gemini/gemini-3.1-pro-preview',
    max_tokens=8192,
    task_temperature=1.0,
    test_temperature=0.6,
    batch_size=5,
    max_concurrency=8,
    def_build_workers=4,
    corpus_kind='sft_v2',
    verbose=True,
)
summary = run_pipeline(cfg)
print(json.dumps(summary, indent=2))
"
    # Sanity-check that the surviving tasks have the v2 metadata fields populated.
    n_done=$(find "$SMOKE_GEN_OUT" -name "task.json" | wc -l)
    if [ "$n_done" -lt 1 ]; then
        fail "stage D produced 0 surviving tasks (expected >=1 of 5 requested)"
    fi
    echo "  $n_done surviving task(s)"
    # Inspect the first surviving task.json.
    first_task=$(find "$SMOKE_GEN_OUT" -name "task.json" | head -1)
    uv run python -c "
import json, sys
t = json.load(open('$first_task'))
need = ['verifier_kind', 'fixture_kind', 'corpus_kind', 'base_image']
for k in need:
    assert k in t, f'task.json missing v2 field: {k}'
    print(f'  {k}: {t[k]!r}')
assert t['corpus_kind'] == 'sft_v2', 'corpus_kind not propagated'
print('  v2 fields propagated through stage 4')
"
    # If a surviving task happened to roll an `intricate`-routed sample, also
    # check the def has a %files section and the fixtures dir is populated.
    intricate_task=$(uv run python -c "
import json, glob, os
for p in glob.glob('$SMOKE_GEN_OUT/task_*'):
    t = json.load(open(os.path.join(p, 'task.json')))
    if t.get('base_image') == 'intricate' and t.get('fixture_kind') != 'text_only':
        print(p); break
")
    if [ -n "$intricate_task" ]; then
        echo "  found intricate task: $intricate_task"
        if grep -q "^%files" "$intricate_task/container.def"; then
            echo "  %files section present in def"
        else
            fail "intricate task has no %files section in def"
        fi
        if [ -d "$intricate_task/fixtures" ]; then
            echo "  fixtures/ dir present: $(ls "$intricate_task/fixtures" | wc -l) entries"
        else
            fail "intricate task has no fixtures/ dir"
        fi
    else
        echo "  (none of the $n_done surviving tasks rolled an intricate fixture; that's fine for a tiny sample)"
    fi
fi
ok "Stage D passed"

hdr "All smoke stages passed"
echo "Artefacts under: $SMOKE_OUT"
