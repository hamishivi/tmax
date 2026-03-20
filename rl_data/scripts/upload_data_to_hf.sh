#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Upload RL task dataset to Hugging Face ───────────────────────────
#
# Collects task.json + solution summaries from a tasks output directory
# and pushes as a HF dataset.
#
# Usage:
#   bash rl_data/scripts/upload_data_to_hf.sh
#   bash rl_data/scripts/upload_data_to_hf.sh --input-dir rl_data/output/tasks_v2
#   bash rl_data/scripts/upload_data_to_hf.sh --repo osieosie/tmax-rl-v2 --private
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with datasets + huggingface_hub

REPO_ID="osieosie/tmax-rl-tasks"
INPUT_DIR="rl_data/output/tasks_toy_et"
PRIVATE="true"
INCLUDE_SOLUTIONS="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)              REPO_ID="$2"; shift 2 ;;
        --input-dir)         INPUT_DIR="$2"; shift 2 ;;
        --private)           PRIVATE="true"; shift ;;
        --public)            PRIVATE="false"; shift ;;
        --no-solutions)      INCLUDE_SOLUTIONS="false"; shift ;;
        *)                   echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Upload RL Dataset to Hugging Face ==="
echo "  Repo:              ${REPO_ID}"
echo "  Input dir:         ${INPUT_DIR}"
echo "  Private:           ${PRIVATE}"
echo "  Include solutions: ${INCLUDE_SOLUTIONS}"
echo ""

uv run python - --repo "${REPO_ID}" --input-dir "${INPUT_DIR}" \
    --private "${PRIVATE}" --include-solutions "${INCLUDE_SOLUTIONS}" << 'PYTHON_EOF'
import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_tasks(tasks_dir: Path, include_solutions: bool) -> list[dict]:
    records = []
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.name.startswith("task_"):
            continue
        task_json = task_path / "task.json"
        if not task_json.exists():
            continue

        with open(task_json) as f:
            task_data = json.load(f)

        record = {
            "task_id": task_data.get("name", task_path.name),
            "domain": task_data.get("domain", task_data.get("category", "")),
            "skill_type": task_data.get("skill_type", ""),
            "primitive_skills": json.dumps(task_data.get("primitive_skills", [])),
            "task_complexity": task_data.get(
                "task_complexity", task_data.get("complexity", "")
            ),
            "command_complexity": task_data.get("command_complexity", ""),
            "scenario": task_data.get("scenario", ""),
            "description": task_data.get("description", ""),
            "truth": task_data.get("truth", ""),
        }

        init_test = task_path / "test_initial_state.py"
        final_test = task_path / "test_final_state.py"
        container_def = task_path / "container.def"
        if init_test.exists():
            record["test_initial_state"] = init_test.read_text(errors="replace")
        if final_test.exists():
            record["test_final_state"] = final_test.read_text(errors="replace")
        if container_def.exists():
            record["container_def"] = container_def.read_text(errors="replace")

        if include_solutions:
            sol_dir = task_path / "solutions"
            if sol_dir.exists():
                for sf in sorted(sol_dir.glob("*_summary.json")):
                    if sf.name == "summary.json":
                        continue
                    with open(sf) as f:
                        sol = json.load(f)
                    model_name = sf.stem.replace("_summary", "")
                    record["solution_model"] = model_name
                    record["num_runs"] = sol.get("num_runs", 0)
                    record["num_success"] = sol.get("num_success", 0)
                    record["pass_at_k"] = json.dumps(sol.get("pass_at_k", {}))
                    record["solutions"] = json.dumps(sol.get("results", []))
                    break

        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--private", default="true")
    p.add_argument("--include-solutions", default="true")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    private = args.private.lower() == "true"
    include_solutions = args.include_solutions.lower() == "true"

    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tasks from {input_dir}...")
    records = load_tasks(input_dir, include_solutions)
    if not records:
        print("No tasks found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(records)} tasks")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"  Repo ready: https://huggingface.co/datasets/{args.repo}")

    ds = Dataset.from_list(records)
    dd = DatasetDict({"train": ds})
    dd.push_to_hub(args.repo, private=private)

    print(f"\nPushed {len(records)} tasks to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
PYTHON_EOF
