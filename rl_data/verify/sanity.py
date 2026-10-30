#!/usr/bin/env python3
"""Bounded runtime sanity check over every rewritten verifier.

hamishivi 2026-09-13: "quick runtime checks of the hardened tasks, and fixes for
failures those checks reveal, are fine -- for example, checking that the rewritten
verifier itself runs and that its own build steps succeed. Please do this as one
pass over ALL rewritten samples ... This is a bounded runtime sanity-check pass,
not another audit or an iterative repair campaign."

So: run the verifier once in an untouched sandbox, with no agent work done at all,
and ask only whether the VERIFIER works -- not whether it grades correctly.

  ok_rejects_empty  ran, wrote a reward, scored 0       the verifier works
  CRASH             raised a non-assertion exception    its own code is broken
  BUILD_FAILED      its own gcc/cargo/make step failed  it cannot run here
  EXEC_BLOCKED      exit 126 / permission denied        its sandboxing is broken
  NO_REWARD         never wrote /logs/verifier/reward   harness error, not a score
  PASSES_EMPTY      scored >0 for doing nothing         it is a rubber stamp

An assertion failure is the CORRECT outcome here and is not a defect: an empty
submission should fail. Only the verifier failing on its own terms counts.

  python3 sanity.py --out sanity/ --shard 0 --num-shards 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate import run_arm  # noqa: E402
from rollout import load_corpus  # noqa: E402
from image_safety import corpus_fingerprints  # noqa: E402
from pathlib import Path
import shutil

B = os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification"))

CRASH = re.compile(
    r"(ValueError|TypeError|SyntaxError|NameError|IndexError|KeyError|AttributeError"
    r"|ImportError|ModuleNotFoundError|re\.error|ZeroDivisionError|UnboundLocalError)")
BUILD = re.compile(r"CalledProcessError[^\n]*(gcc|g\+\+|cargo|make|rustc|cmake)", re.I)
BLOCKED = re.compile(r"exit (code|status) 126|Permission denied|not executable", re.I)


def classify(rec: dict) -> str:
    if rec.get("error"):
        return "INFRA"
    obs = rec.get("observation") or ""
    reward = rec.get("reward")
    if reward is None:
        return "NO_REWARD"
    if reward > 0:
        return "PASSES_EMPTY"
    if BLOCKED.search(obs):
        return "EXEC_BLOCKED"
    if BUILD.search(obs):
        return "BUILD_FAILED"
    # An AssertionError is the correct way to fail an empty submission. Only treat
    # a non-assertion exception as the verifier breaking on its own code.
    if CRASH.search(obs):
        return "CRASH"
    return "ok_rejects_empty"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True, help="Final materialised corpus")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default="", help="json list: restrict to these ids")
    args = ap.parse_args()

    artifacts = [json.loads(line) for line in (Path(args.corpus) / 'artifacts.jsonl').read_text().splitlines()]
    tasks = sorted(r['task_id'] for r in artifacts if r['verifier_repaired'])
    fingerprints = corpus_fingerprints(Path(args.corpus))
    if args.tasks:
        want = set(json.load(open(args.tasks, encoding="utf-8")))
        tasks = [t for t in tasks if t in want]
    if args.num_shards > 1:
        tasks = tasks[args.shard :: args.num_shards]
    if args.limit:
        tasks = tasks[: args.limit]

    # The verifier may have been written against a rebuilt image; use it when present.
    images = load_corpus(args.corpus)

    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(args.out, f"_stage{args.shard}")
    os.makedirs(root, exist_ok=True)
    suffix = f"-{args.shard:02d}" if args.num_shards > 1 else ""
    path = os.path.join(args.out, f"sanity{suffix}.jsonl")
    done = set()
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
                if row.get('artifact_id') == fingerprints.get(row.get('task_id')):
                    done.add(row['task_id'])
            except Exception:  # noqa: BLE001
                pass
    todo = [t for t in tasks if t not in done and t in images]
    print(f"rewritten verifiers in this shard: {len(tasks)}  to check: {len(todo)}", flush=True)

    fh = open(path, "a", encoding="utf-8")
    lock = threading.Lock()
    counts: dict[str, int] = {}
    t0 = time.time()

    def work(t):
        try:
            td = os.path.join(root, t, 'task-data')
            shutil.copytree(os.path.join(args.corpus, 'task-data', t), os.path.join(td, t), dirs_exist_ok=True)
            arm = asyncio.run(run_arm(t, td, images[t], []))
        except Exception as exc:  # noqa: BLE001
            arm = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        verdict = classify(arm)
        rec = {"task_id": t, "verdict": verdict, "reward": arm.get("reward"),
               "artifact_id": fingerprints[t],
               "error": arm.get("error"),
               "observation": (arm.get("observation") or "")[:1500]}
        with lock:
            counts[verdict] = counts.get(verdict, 0) + 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = sum(counts.values())
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {dict(sorted(counts.items()))}  "
                      f"{n/max(time.time()-t0,1e-9)*3600:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, todo))
    fh.close()
    print(f"\n{dict(sorted(counts.items()))}\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
