#!/bin/bash
#SBATCH --job-name=rl-gen-tasks-sft-v2-1k
#SBATCH --output=logs/gen_tasks_sft_v2_1k_%j.out
#SBATCH --error=logs/gen_tasks_sft_v2_1k_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  v2 SFT corpus — generate ~1k surviving tasks with the new axes turned on. ║
# ║                                                                            ║
# ║  Mirror of run_generate_tasks_1k.sh, three differences:                    ║
# ║    1. CORPUS_KIND=sft_v2 — enables verifier_kind / fixture_kind /          ║
# ║       intricate-complexity sampling via the bucket-upweight formula        ║
# ║       (M=2; ~67%% of tasks have a non-legacy verifier kind, etc).          ║
# ║    2. NUM_TASKS=2200 — request 2x to land ~1k surviving tasks after the    ║
# ║       4-stage pipeline-survival filter (template→init_test→final_test→     ║
# ║       SIF-build-and-smoke). This matches the legacy 1k script's pattern.   ║
# ║    3. OUT_DIR / job-name — separate output dir so the legacy 1k corpus     ║
# ║       stays untouched. Combine via plain ``cp -r`` at training time.       ║
# ║                                                                            ║
# ║  Prerequisite (one-time): build base_intricate.sif on a build node:        ║
# ║      apptainer build rl_data/containers/base_intricate.sif \               ║
# ║                      rl_data/containers/base_intricate.def                 ║
# ║  Tasks with non-legacy verifier_kind / fixture_kind / intricate complexity ║
# ║  resolve to base_intricate at solve time (see apptainer_def_gen.BASE_IMAGES ║
# ║  + env._resolve_runtime_sif).                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
NUM_TASKS=2200
OUT_DIR="rl_data/output/tasks_skill_tax_v2_20260505_1k"
MODEL="gemini/gemini-3.1-pro-preview"
MAX_TOKENS=32768
BATCH_SIZE=100
MAX_CONCURRENCY=128
DEF_BUILD_WORKERS=64
TASK_TEMPERATURE=1.0
TEST_TEMPERATURE=0.6
CORPUS_KIND=sft_v2 #used for upsampling tasks, disabled when we are generating from scartch

# ---- Resume behaviour ----
# Same as the legacy script: stages 1-3 are checkpointed to
# <OUT_DIR>/_intermediates.jsonl; stage 4 progress is in
# <OUT_DIR>/_stage4_done.jsonl. Delete those to force full regeneration.
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

uv run python -c "
from pathlib import Path
from rl_data.generate_tasks import AsyncBatchConfig, run_pipeline
import json

cfg = AsyncBatchConfig(
    num_tasks=$NUM_TASKS,
    out_dir=Path('$OUT_DIR'),
    model='$MODEL',
    max_tokens=$MAX_TOKENS,
    task_temperature=$TASK_TEMPERATURE,
    test_temperature=$TEST_TEMPERATURE,
    batch_size=$BATCH_SIZE,
    max_concurrency=$MAX_CONCURRENCY,
    def_build_workers=$DEF_BUILD_WORKERS,
    corpus_kind='$CORPUS_KIND',
    verbose=True,
)

summary = run_pipeline(cfg)
print(json.dumps(summary, indent=4))
"
