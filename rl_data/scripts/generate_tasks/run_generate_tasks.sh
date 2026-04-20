#!/bin/bash
#SBATCH --job-name=rl-gen-tasks
#SBATCH --output=logs/gen_tasks_%j.out
#SBATCH --error=logs/gen_tasks_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# ---- Parameters (edit here) ----
NUM_TASKS=10
OUT_DIR="rl_data/output/tasks_skill_tax_20260327_toy"
MODEL="gemini/gemini-3-flash-preview"
#MODEL="gemini/gemini-3.1-pro-preview"
MAX_TOKENS=32768
BATCH_SIZE=10
MAX_CONCURRENCY=10
TASK_TEMPERATURE=1.0
TEST_TEMPERATURE=0.6
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/gpfs/projects/h2lab/osey/apptainer_tmp"

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
    verbose=True,
)

summary = run_pipeline(cfg)
print(json.dumps(summary, indent=4))
"
