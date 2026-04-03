"""Generate tasks via batched LLM calls.

Pipeline: task templates -> initial tests -> final tests -> container defs -> save.

**Limits of LLM-only ground truth:** ``truth`` and ``test_final_state.py`` are generated text.
This module does not execute setup or recompute goldens, so errors in derived quantities or
inconsistencies between setup and stated expectations can slip through. A second model writes
final tests from *truth*, so mis-copying or drift is possible. **Hardening:** add an external
validation pass (execute setup, reference solution, or automated checks) before publishing;
prompts in ``task_template_gen`` / ``completion_test_gen`` encode general principles for
consistent, reproducible *truth* and tests.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from tqdm import tqdm

from rl_data import DEFAULT_MODEL
from rl_data.generator.task_template_gen import generate_templates_batch
from rl_data.generator.initial_state_test_gen import generate_test_templates_batch as generate_initial_tests_batch
from rl_data.generator.apptainer_def_gen import iterate_def_template_batch, save_setup_artifacts
from rl_data.generator.completion_test_gen import generate_test_templates_batch as generate_final_tests_batch


@dataclass
class PipelineConfig:
    num_tasks: int
    out_dir: Path
    max_def_retries: int = 3
    max_num_completions: int = 4
    num_solutions: int = 256
    max_actions: int = 20
    model: str = DEFAULT_MODEL
    max_tokens: int = 32768
    task_temperature: float = 1.0
    test_temperature: float = 0.6
    solution_temperature: float = 1.0
    parallel_jobs: int = 1
    verbose: bool = False
    #: Max concurrent Apptainer build+test workers in stage 4 (def generation).
    #: Each worker uses ~1 CPU + ~4 GB RAM.  Safe default: 4; scale up with CPUs.
    def_build_workers: int = 4


def _safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_sif(def_path: Path, sif_path: Path) -> bool:
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.run(
            ["apptainer", "build", str(sif_path), str(def_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        return rc == 0
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False


def _format_task_dir(base: Path, idx: int, width: int = 6) -> Path:
    suffix = uuid.uuid4().hex[:8]
    return base / f"task_{idx:0{width}d}_{suffix}"


def _save_task_bundle(
    task_dir: Path,
    task_obj: Dict[str, Any],
    initial_test_code: str,
    def_text: str,
    final_test_code: str,
    summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    task_json = task_dir / "task.json"
    init_py = task_dir / "test_initial_state.py"
    final_py = task_dir / "test_final_state.py"
    def_file = task_dir / "container.def"
    sif_file = task_dir / "container.sif"
    sol_dir = task_dir / "solutions"
    sol_dir.mkdir(parents=True, exist_ok=True)

    _safe_write_text(task_json, json.dumps(task_obj, indent=4))
    _safe_write_text(init_py, initial_test_code)
    _safe_write_text(final_py, final_test_code)
    _safe_write_text(def_file, def_text)
    _safe_write_text(sol_dir / "summary.json", json.dumps(summary, indent=4))

    domain = task_obj.get("domain", "software_engineering")
    save_setup_artifacts(task_dir, def_text, domain)

    return task_json, init_py, final_py, def_file, sif_file


@dataclass
class AsyncBatchConfig(PipelineConfig):
    batch_size: int = 64
    max_concurrency: int = 64


def _generate_intermediates_batch(
    cfg: AsyncBatchConfig, batch_count: int,
) -> List[Dict[str, Any]]:
    """Run stages 1-3 (templates, initial tests, final tests) and return intermediates."""

    # 1) Task templates
    print(f"Generating {batch_count} task templates with {cfg.max_concurrency} concurrency")
    task_templates = generate_templates_batch(
        batch_count,
        model=cfg.model,
        temperature=cfg.task_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    if not task_templates:
        print("No task templates generated")
        return []

    descriptions: List[str] = [t.get("description", "").strip() for t in task_templates]
    truths: List[str] = [t.get("truth", "").strip() for t in task_templates]
    meta: List[Dict[str, Any]] = [
        {
            "domain": t.get("domain", ""),
            "skill_type": t.get("skill_type", ""),
            "primitive_skills": t.get("primitive_skills", []),
            "task_complexity": t.get("task_complexity", ""),
            "command_complexity": t.get("command_complexity", ""),
            "scenario": t.get("scenario", ""),
            "language": t.get("language", ""),
            "anchor": t.get("anchor"),
        }
        for t in task_templates
    ]

    valid_indices = [i for i, (d, tr) in enumerate(zip(descriptions, truths)) if d and tr]
    if not valid_indices:
        print("No valid task templates generated")
        return []

    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    meta = [meta[i] for i in valid_indices]

    print(f"Task templates generated: {len(descriptions)}")

    # 2) Initial tests (batch)
    print(f"Generating {len(descriptions)} initial tests with {cfg.max_concurrency} concurrency")
    init_tests = generate_initial_tests_batch(
        list(zip(descriptions, truths)),
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    valid_indices = [i for i, test in enumerate(init_tests) if test]
    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    meta = [meta[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]

    print(f"Generated {len(init_tests)} initial tests")

    # 3) Final tests (batch)
    print(f"Generating {len(descriptions)} final tests with {cfg.max_concurrency} concurrency")
    final_tests = generate_final_tests_batch(
        list(zip(descriptions, truths, init_tests)),
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    print(f"Generated {len(final_tests)} final tests")
    valid_indices = [i for i, test in enumerate(final_tests) if test]
    descriptions = [descriptions[i] for i in valid_indices]
    truths = [truths[i] for i in valid_indices]
    meta = [meta[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]
    final_tests = [final_tests[i] for i in valid_indices]

    return [
        {
            "description": descriptions[i],
            "truth": truths[i],
            "init_test": init_tests[i],
            "final_test": final_tests[i],
            "meta": meta[i],
        }
        for i in range(len(descriptions))
    ]


# ---------------------------------------------------------------------------
# Intermediate checkpoint I/O
# ---------------------------------------------------------------------------

_INTERMEDIATES_FILENAME = "_intermediates.jsonl"


def _save_intermediates(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_intermediates(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_intermediates(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Saving a single task to disk
# ---------------------------------------------------------------------------

def _save_one_task(
    item: Dict[str, Any],
    def_text: str,
    out_dir: Path,
    idx: int,
) -> Path:
    """Persist a single task (intermediate + def text) and return its directory."""
    m = item["meta"]
    desc = item["description"]
    tr = item["truth"]

    task_dir = _format_task_dir(out_dir, idx=idx)
    task_obj = {
        "name": task_dir.name,
        "domain": m["domain"],
        "skill_type": m["skill_type"],
        "primitive_skills": m["primitive_skills"],
        "task_complexity": m["task_complexity"],
        "command_complexity": m["command_complexity"],
        "scenario": m["scenario"],
        "language": m.get("language", ""),
        "anchor": m.get("anchor"),
        "description": desc,
        "truth": tr,
    }

    _save_task_bundle(
        task_dir, task_obj, item["init_test"], def_text,
        item["final_test"], summary={},
    )

    skills_str = ", ".join(m["primitive_skills"])
    summary_txt = (
        f"Task: {task_dir.name}\n"
        f"Domain: {m['domain']}\n"
        f"Skill Type: {m['skill_type']}\n"
        f"Primitive Skills: {skills_str}\n"
        f"Task Complexity: {m['task_complexity']}\n"
        f"Command Complexity: {m['command_complexity']}\n"
        f"Scenario: {m['scenario']}\n"
        f"\n{'='*60}\n"
        f"DESCRIPTION\n{'='*60}\n\n"
        f"{desc}\n"
        f"\n{'='*60}\n"
        f"GROUND TRUTH\n{'='*60}\n\n"
        f"{tr}\n"
    )
    _safe_write_text(task_dir / "task_summary.txt", summary_txt)
    return task_dir


# ---------------------------------------------------------------------------
# Stage 4 progress tracking
# ---------------------------------------------------------------------------

_STAGE4_PROGRESS_FILENAME = "_stage4_done.jsonl"


def _load_stage4_done(path: Path) -> Dict[int, str]:
    """Load completed stage-4 indices → task_dir mapping."""
    done: Dict[int, str] = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                done[entry["idx"]] = entry["task_dir"]
    return done


def _append_stage4_done(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Two-phase pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: AsyncBatchConfig) -> Dict[str, Any]:
    """Generate tasks in two phases with checkpoints at every stage.

    **Phase 1 — stages 1-3** (LLM-only: templates, initial tests, final tests):
    Fast; results are saved to ``<out_dir>/_intermediates.jsonl`` after each
    batch.  If this file already exists on entry the phase is skipped entirely.

    **Phase 2 — stage 4** (def gen + Apptainer build/test):
    CPU-heavy; runs on all intermediates at once.  Tasks are saved to disk
    **after each retry round** (streaming saves), and progress is tracked in
    ``<out_dir>/_stage4_done.jsonl``.  On restart, already-completed items
    are skipped automatically.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    intermediates_path = cfg.out_dir / _INTERMEDIATES_FILENAME
    progress_path = cfg.out_dir / _STAGE4_PROGRESS_FILENAME
    batch_size = max(1, cfg.batch_size)

    # ── Phase 1: intermediates (stages 1-3) ──
    if intermediates_path.exists():
        all_intermediates = _load_intermediates(intermediates_path)
        print(
            f"Checkpoint found: loaded {len(all_intermediates)} intermediates "
            f"from {intermediates_path} (stages 1-3 skipped)"
        )
    else:
        all_intermediates: List[Dict[str, Any]] = []
        remaining = cfg.num_tasks
        num_batches = (cfg.num_tasks + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Stages 1-3"):
            count = min(batch_size, remaining)
            items = _generate_intermediates_batch(cfg, count)
            all_intermediates.extend(items)
            _append_intermediates(intermediates_path, items)
            remaining -= count
        print(
            f"Stages 1-3 complete: {len(all_intermediates)} intermediates "
            f"(saved to {intermediates_path})"
        )

    if not all_intermediates:
        print("No intermediates to process")
        return {
            "requested": cfg.num_tasks,
            "intermediates": 0,
            "succeeded": 0,
            "success_rate": 0.0,
            "saved_dirs": [],
        }

    # ── Phase 2: def gen (stage 4) + streaming save ──
    done_map = _load_stage4_done(progress_path)
    done_indices = set(done_map.keys())
    all_saved_dirs: List[str] = list(done_map.values())
    round_stats: List[Dict[str, Any]] = []

    if done_indices:
        print(f"Stage 4 checkpoint: {len(done_indices)} items already completed, resuming")

    descriptions = [item["description"] for item in all_intermediates]
    truths = [item["truth"] for item in all_intermediates]
    init_tests = [item["init_test"] for item in all_intermediates]
    domains = [item["meta"]["domain"] for item in all_intermediates]

    n_total = len(all_intermediates)
    n_pending = n_total - len(done_indices)
    print(
        f"Stage 4: {n_pending} defs to process ({n_total} total, "
        f"{len(done_indices)} already done)\n"
        f"  build_workers={cfg.def_build_workers}, "
        f"llm_concurrency={cfg.max_concurrency}, "
        f"retries={cfg.max_def_retries}"
    )

    stage4_start = time.monotonic()
    _round_start = [stage4_start]
    _save_lock = __import__("threading").Lock()

    def _on_item_success(idx: int, def_text: str) -> None:
        """Save a single task to disk immediately when it passes build+test."""
        task_dir = _save_one_task(
            all_intermediates[idx], def_text, cfg.out_dir, idx=idx,
        )
        with _save_lock:
            all_saved_dirs.append(str(task_dir))
            done_indices.add(idx)
            _append_stage4_done(progress_path, [{"idx": idx, "task_dir": str(task_dir)}])

    def _on_round_complete(round_idx: int, newly_succeeded: Dict[int, str]) -> None:
        """Log round-level stats (saves already happened per-item)."""
        round_elapsed = time.monotonic() - _round_start[0]
        _round_start[0] = time.monotonic()

        round_stats.append({
            "round": round_idx,
            "succeeded_this_round": len(newly_succeeded),
            "cumulative_succeeded": len(all_saved_dirs),
            "remaining": n_total - len(done_indices),
            "elapsed_s": round(round_elapsed, 1),
        })

    iterate_def_template_batch(
        list(zip(descriptions, truths, init_tests)),
        domains=domains,
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
        max_retries=cfg.max_def_retries,
        max_build_workers=cfg.def_build_workers,
        skip_indices=done_indices if done_indices else None,
        on_round_complete=_on_round_complete,
        on_item_success=_on_item_success,
    )

    stage4_elapsed = time.monotonic() - stage4_start

    # ── Write generation log ──
    log_path = cfg.out_dir / "_generation_log.txt"
    n_succeeded = len(all_saved_dirs)
    n_failed = n_total - n_succeeded
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    W = 64
    sep = "=" * W
    thin = "-" * W

    log_lines = [
        sep,
        "  TASK GENERATION LOG",
        sep,
        f"  Timestamp:         {ts}",
        f"  Model:             {cfg.model}",
        f"  Max tokens:        {cfg.max_tokens}",
        f"  Task temperature:  {cfg.task_temperature}",
        f"  Test temperature:  {cfg.test_temperature}",
        "",
        sep,
        "  PHASE 1: STAGES 1-3 (LLM-only)",
        sep,
        f"  Tasks requested:   {cfg.num_tasks}",
        f"  Batch size:        {cfg.batch_size}",
        f"  LLM concurrency:   {cfg.max_concurrency}",
        f"  Intermediates:     {n_total}",
        f"  Survival rate:     {n_total / cfg.num_tasks:.1%}" if cfg.num_tasks else "",
        "",
        sep,
        "  PHASE 2: STAGE 4 (def gen + Apptainer build/test)",
        sep,
        f"  Build workers:     {cfg.def_build_workers}",
        f"  Max retries:       {cfg.max_def_retries}",
        f"  Total time:        {stage4_elapsed / 60:.1f} min",
        "",
        f"  {'Round':<8} {'Succeeded':>10} {'Cumulative':>11} {'Remaining':>10} {'Time':>10}",
        f"  {thin}",
    ]
    for rs in round_stats:
        log_lines.append(
            f"  {rs['round']:<8} {rs['succeeded_this_round']:>10} "
            f"{rs['cumulative_succeeded']:>11} {rs['remaining']:>10} "
            f"{rs['elapsed_s']:>8.1f}s"
        )
    log_lines += [
        f"  {thin}",
        "",
        sep,
        "  SUMMARY",
        sep,
        f"  Requested:         {cfg.num_tasks}",
        f"  Intermediates:     {n_total}",
        f"  Succeeded:         {n_succeeded}",
        f"  Failed:            {n_failed}",
        f"  Overall rate:      {n_succeeded / cfg.num_tasks:.1%}" if cfg.num_tasks else "",
        f"  Output dir:        {cfg.out_dir}",
        sep,
        "",
    ]

    log_text = "\n".join(log_lines)
    _safe_write_text(log_path, log_text)
    print(log_text)

    return {
        "requested": cfg.num_tasks,
        "intermediates": n_total,
        "succeeded": n_succeeded,
        "success_rate": (n_succeeded / cfg.num_tasks) if cfg.num_tasks else 0.0,
        "saved_dirs": all_saved_dirs,
    }


def parse_args(argv: Optional[List[str]] = None) -> AsyncBatchConfig:
    ap = argparse.ArgumentParser(description="Generate tasks via async-batched LLM calls.")
    ap.add_argument("--num-tasks", type=int, default=100, help="How many tasks to request")
    ap.add_argument("--out-dir", type=Path, default=Path("tasks"), help="Output directory")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--task-temperature", type=float, default=1.0)
    ap.add_argument("--test-temperature", type=float, default=0.6)
    ap.add_argument("--solution-temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=128)
    ap.add_argument(
        "--def-build-workers", type=int, default=4,
        help="Max concurrent Apptainer build+test workers in stage 4 "
             "(each uses ~1 CPU + ~4 GB RAM; default: 4)",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)
    verbose = args.verbose and not args.quiet

    return AsyncBatchConfig(
        num_tasks=args.num_tasks,
        out_dir=args.out_dir,
        model=args.model,
        task_temperature=args.task_temperature,
        test_temperature=args.test_temperature,
        solution_temperature=args.solution_temperature,
        parallel_jobs=1,
        verbose=verbose,
        batch_size=max(1, args.batch_size),
        max_concurrency=max(1, args.max_concurrency),
        def_build_workers=max(1, args.def_build_workers),
    )


if __name__ == "__main__":
    cfg = parse_args()
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=4))
