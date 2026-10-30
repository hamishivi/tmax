#!/usr/bin/env python3
"""Join the tmax-15k RL records to their on-disk task definitions.

Produces one JSONL row per task carrying everything the grader needs, so the
grading run itself does no filesystem walking and is trivially resumable.

Reconciles three sources and refuses to emit a partial index silently:

  RL records     swerl_tmax_15k_apptainer.jsonl        instruction, task_id, image
  task data      task-data.tar.gz.extracted/<task_id>  instruction.md, setup.sh, tests/test.sh
  images         sifs/<image tag>.sif                  fixtures only, no verifier

The image is recorded but not read: the SIF contains no verifier, so the task
definition is complete without it.

  python3 build_index.py --out index.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

R = os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification"))
DEFAULT_RECORDS = f"{R}/exchange/swerl_tmax_15k_apptainer.jsonl"
DEFAULT_TASKDATA = (
    f"{R}/hf-cache/hub/datasets--hamishivi--swerl-tmax-15k/snapshots/"
    "2e297abb28a9b22fe67063742f8134a745e35505/task-data.tar.gz.extracted"
)
DEFAULT_SIFS = f"{R}/sifs"

REQUIRED = ("instruction.md", "setup.sh", "tests/test.sh")


def read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def sif_name(image: str) -> str:
    """hamishi740/swerl-tmax-v3:618e344e0172 -> hamishi740__swerl-tmax-v3__618e344e0172.sif"""
    return image.replace("/", "__").replace(":", "__") + ".sif"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--task-data", default=DEFAULT_TASKDATA)
    ap.add_argument("--sifs", default=DEFAULT_SIFS)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="emit the index even if some tasks lack files (default: refuse)",
    )
    args = ap.parse_args()

    on_disk = {
        d for d in os.listdir(args.task_data) if os.path.isdir(os.path.join(args.task_data, d))
    }
    rows, missing_dir, missing_files, missing_sif, dup = [], [], [], [], []
    seen: set[str] = set()

    with open(args.records, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            env = rec.get("env_config") or {}
            tid = env.get("task_id")
            if not tid:
                continue
            if tid in seen:
                dup.append(tid)
                continue
            seen.add(tid)

            tdir = os.path.join(args.task_data, tid)
            if tid not in on_disk:
                missing_dir.append(tid)
                continue
            paths = {f: os.path.join(tdir, f) for f in REQUIRED}
            absent = [f for f, p in paths.items() if not os.path.isfile(p)]
            if absent:
                missing_files.append((tid, absent))
                continue

            image = env.get("image", "")
            sif = os.path.join(args.sifs, sif_name(image)) if image else ""
            if sif and not os.path.isfile(sif):
                missing_sif.append(tid)

            # The RL prompt's user turn wraps instruction.md; keep the file as
            # the authoritative text and record whether they diverge.
            user_turn = next(
                (m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user"),
                "",
            )
            instruction = read(paths["instruction.md"])

            rows.append(
                {
                    "task_id": tid,
                    "image": image,
                    "env_name": env.get("env_name", ""),
                    "source": rec.get("source", ""),
                    "dataset": rec.get("dataset", ""),
                    "sif_path": sif,
                    "sif_present": bool(sif) and os.path.isfile(sif),
                    "instruction": instruction,
                    "setup": read(paths["setup.sh"]),
                    "tests": read(paths["tests/test.sh"]),
                    "instruction_in_prompt_matches": instruction.strip() in user_turn,
                    "n_chars": sum(len(read(p)) for p in paths.values()),
                }
            )

    orphan_dirs = sorted(on_disk - seen)

    print(f"records read         : {len(seen)}")
    print(f"indexed              : {len(rows)}")
    print(f"missing task dir     : {len(missing_dir)}")
    print(f"missing required file: {len(missing_files)}")
    print(f"missing SIF          : {len(missing_sif)}  (recorded, not fatal: no verifier in SIF)")
    print(f"duplicate task_id    : {len(dup)}")
    print(f"task dirs not in RL set: {len(orphan_dirs)}")
    mism = sum(1 for r in rows if not r["instruction_in_prompt_matches"])
    print(f"instruction.md not verbatim inside RL prompt: {mism}")

    for label, items in (
        ("missing task dir", missing_dir[:5]),
        ("missing files", missing_files[:5]),
        ("orphan dirs", orphan_dirs[:5]),
    ):
        if items:
            print(f"  sample {label}: {items}")

    fatal = missing_dir or missing_files or dup
    if fatal and not args.allow_missing:
        print("\nREFUSING to write a partial index. Re-run with --allow-missing to override.")
        return 1

    with open(args.out, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
