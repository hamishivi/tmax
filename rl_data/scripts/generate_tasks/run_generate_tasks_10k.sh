#!/bin/bash
#SBATCH --job-name=rl-gen-tasks-1k
#SBATCH --output=logs/gen_tasks_1k_%j.out
#SBATCH --error=logs/gen_tasks_1k_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

set -euo pipefail

# ---- Parameters (edit here) ----
# Request 2000 tasks: with ~50% pipeline survival rate (template→init_test→
# final_test→def_build), expect ~1000 surviving tasks.  Increase if survival
# is lower on your workload; decrease if you only need ~500 usable tasks.
NUM_TASKS=10000
OUT_DIR="rl_data/output/tasks_skill_tax_20260401_10k"
MODEL="gemini/gemini-3.1-pro-preview"
MAX_TOKENS=32768
# Process in batches of 100 — each batch runs stages 1-3 (LLM calls).
# All intermediates are then processed through stage 4 (def build+test)
# together, keeping build workers fully utilised.
BATCH_SIZE=250
# LLM API concurrency for all stages (templates, tests, def gen prompts).
# With a company API account, 128+ is safe.  Tune down on 429 errors.
MAX_CONCURRENCY=256
# Concurrent Apptainer build+test workers in stage 4.
# Each uses ~1 CPU + ~4 GB RAM.  With 4 GPUs (32 CPUs): 24 workers
# leaves 8 cores for Python + LLM I/O overhead.
DEF_BUILD_WORKERS=64
TASK_TEMPERATURE=1.0
TEST_TEMPERATURE=0.6

# ---- Resume behaviour ----
# The pipeline saves intermediates (stages 1-3 output) to
#   <OUT_DIR>/_intermediates.jsonl
# On restart, if that file exists, stages 1-3 are skipped and only stage 4
# (def gen + build/test) re-runs.  Delete the file to force full regeneration.
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
    verbose=True,
)

summary = run_pipeline(cfg)
print(json.dumps(summary, indent=4))
"
