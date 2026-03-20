#!/bin/bash
#SBATCH --job-name=rl-gen-solutions
#SBATCH --output=logs/gen_solutions_%j.out
#SBATCH --error=logs/gen_solutions_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_skill_tax_20260319"
MODEL="gemini/gemini-3.1-pro-preview"
NUM_SOLUTIONS=8
MAX_ACTIONS=16
MAX_TOKENS=65536
NUM_TASKS=10
START_AT=0
WORKERS=1
NUM_POOL_WORKERS=128
SOLUTION_TEMPERATURE=0.7
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

uv run python -c "
import sys
from pathlib import Path
from rl_data.generate_solutions import SolutionConfig, process_task
from tqdm import tqdm

cfg = SolutionConfig(
    tasks_dir='$TASKS_DIR',
    num_solutions=$NUM_SOLUTIONS,
    max_actions=$MAX_ACTIONS,
    model='$MODEL',
    solution_temperature=$SOLUTION_TEMPERATURE,
    max_tokens=$MAX_TOKENS,
    num_tasks=$NUM_TASKS,
    start_at=$START_AT,
    workers=$WORKERS,
    num_pool_workers=$NUM_POOL_WORKERS,
    verbose=True,
)

all_entries = list(Path(cfg.tasks_dir).iterdir())
task_dirs = sorted(d for d in all_entries if d.name.startswith('task_'))
task_dirs = task_dirs[cfg.start_at : min(cfg.start_at + cfg.num_tasks, len(task_dirs))]

if not task_dirs:
    print(f'No task directories found in {cfg.tasks_dir}')
    sys.exit(0)

for task_dir in tqdm(task_dirs, desc='Processing Tasks'):
    task_dir = Path(task_dir)
    max_retries = 1
    while max_retries > 0:
        result = process_task(task_dir, cfg)
        if result is None:
            print(f'Retrying task {task_dir.name}...')
            max_retries -= 1
        else:
            print(f'Pass@k: {result} for task {task_dir.name}')
            break
"
