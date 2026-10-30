#!/usr/bin/env python3
"""Build the correctness-filtered dataset from the rollout evidence.

A task is VERIFIED when a strong model solved it at least once. The whole point
of this pass is that "did not solve" and "was never actually attempted" are
different claims, so the export keeps them apart:

  verified       >=1 solved attempt                  the task is satisfiable
  unverified     attempted, never solved             real negative evidence
  budget_limited every attempt hit the wall-clock    negative, but budget-bound
  not_evaluable  only infra/API failures, or the     NO evidence about the task
                 route refused the content

`not_evaluable` matters more than it looks. The grading route content-filters
security-flavoured tasks -- 95% of its hard failures contained terms like
exploit/vulnerability/bypass against a 28% corpus baseline. Folding those into
"unverified" would quietly delete the security subset of the dataset, so they
are labelled, counted, and left for a different model.

  python3 export_verified.py --rollouts rollouts-wave1/ rollouts-wave2/ \
      --labels labels.jsonl --fixes fixes/ --index index.jsonl --out-dir out/
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

try:
    from .image_safety import corpus_fingerprints
except ImportError:
    from image_safety import corpus_fingerprints

SCORED = ("solved", "unsolved", "no_submit")
NO_EVIDENCE = ("error", "infra", "model_declined")


def load_rollouts(dirs: list[str], expected: dict | None = None) -> dict[str, list[dict]]:
    """Gather episodes, requiring matching final-artifact fingerprints when given.

    Directory names carry no evidentiary meaning. Historical unfingerprinted
    traces remain readable by callers that explicitly omit expected fingerprints.
    The public export CLI always requires fingerprints.
    """
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "rollouts*.jsonl"))):
            if path.endswith((".wedged", ".pre-depbake")):
                continue
            for line in Path(path).read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = r.get("task_id")
                if not tid:
                    continue
                if expected is not None and (tid not in expected or r.get('artifact_id') != expected[tid]):
                    continue
                by_task[tid].append(r)
    # Retried attempts and repeated input directories do not inflate the counts.
    for tid, rows in by_task.items():
        unique = {(r.get('model'), r['attempt']): r for r in rows}
        by_task[tid] = list(unique.values())
    return by_task


def classify(attempts: list[dict]) -> tuple[str, dict]:
    outs = [a["outcome"] for a in attempts]
    n_solved = sum(1 for o in outs if o == "solved")
    info = {
        "attempts": len(attempts),
        "n_solved": n_solved,
        "n_unsolved": sum(1 for o in outs if o == "unsolved"),
        "n_no_submit": sum(1 for o in outs if o == "no_submit"),
        "n_timeout": sum(1 for o in outs if o == "timeout"),
        "n_no_evidence": sum(1 for o in outs if o in NO_EVIDENCE),
        "best_reward": max((float(a.get("reward") or 0.0) for a in attempts), default=0.0),
        "min_steps_to_solve": min((a["steps"] for a in attempts if a["outcome"] == "solved"),
                                  default=None),
        "models": sorted({a.get("model", "") for a in attempts}),
    }
    if n_solved:
        return "verified", info
    if any(o in SCORED for o in outs):
        return "unverified", info
    if any(o == "timeout" for o in outs):
        return "budget_limited", info
    return "not_evaluable", info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corpus", required=True, help="Final materialised corpus; stale/unfingerprinted evidence is ignored")
    args = ap.parse_args()

    idx = {}
    for line in Path(args.index).read_text().splitlines():
        r = json.loads(line)
        idx[r["task_id"]] = r
    labels = {}
    for line in Path(args.labels).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label"):
            labels[r["task_id"]] = r

    expected = corpus_fingerprints(Path(args.corpus))
    artifacts = {r['task_id']: r for r in map(json.loads, (Path(args.corpus) / 'artifacts.jsonl').read_text().splitlines())}
    by_task = load_rollouts(args.rollouts, expected=expected)
    print(f"tasks with rollout evidence: {len(by_task)}")

    rows = []
    status_count = collections.Counter()
    by_label = collections.defaultdict(collections.Counter)
    for tid in sorted(expected):
        attempts = by_task.get(tid, [])
        status, info = classify(attempts)
        if not attempts:
            status = 'not_attempted'
        status_count[status] += 1
        lab = labels.get(tid, {})
        by_label[lab.get("label", "?")][status] += 1

        fixdir = os.path.join(args.fixes, tid)
        repaired = artifacts[tid]['verifier_repaired']
        rebuilt = artifacts[tid]['image_rebuilt']

        rows.append({
            "task_id": tid,
            "artifact_id": expected[tid],
            "status": status,
            "verified": status == "verified",
            "rubric_label": lab.get("label", ""),
            "rubric_reason": lab.get("reason", ""),
            "verifier_repaired": repaired,
            "image_rebuilt": rebuilt,
            "attempts": info["attempts"],
            "n_solved": info["n_solved"],
            "n_unsolved": info["n_unsolved"],
            "n_no_submit": info["n_no_submit"],
            "n_timeout": info["n_timeout"],
            "n_no_evidence": info["n_no_evidence"],
            "best_reward": info["best_reward"],
            "min_steps_to_solve": info["min_steps_to_solve"],
            "rollout_models": info["models"],
            "image": idx.get(tid, {}).get("image", ""),
        })

    n = len(rows)
    print("\n--- status ---")
    for k, v in status_count.most_common():
        print(f"  {k:16s} {v:6d}  ({100*v/n:5.1f}%)")
    scored_n = status_count["verified"] + status_count["unverified"]
    if scored_n:
        print(f"\nverified / scored : {status_count['verified']}/{scored_n} "
              f"= {100*status_count['verified']/scored_n:.1f}%")

    print("\n--- by rubric label ---")
    for lab in sorted(by_label, key=lambda k: -sum(by_label[k].values())):
        c = by_label[lab]
        tot = sum(c.values())
        ver = c["verified"]
        print(f"  {lab:26s} {ver:6d}/{tot:6d} verified  ({100*ver/tot:5.1f}%)")

    os.makedirs(args.out_dir, exist_ok=False)
    os.makedirs(os.path.join(args.out_dir, "data"))
    jl = os.path.join(args.out_dir, "verified.jsonl")
    with open(jl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {jl}")
    keep = [r["task_id"] for r in rows if r["verified"]]
    Path(args.out_dir, 'verified_task_ids.json').write_text(json.dumps(sorted(keep)) + '\n')

    try:
        import pandas as pd
    except ImportError:
        print("pandas unavailable; skipped parquet")
        return 0
    pq = os.path.join(args.out_dir, "data", "train-00000-of-00001.parquet")
    pd.DataFrame(rows).to_parquet(pq, index=False)
    print(f"wrote {pq}  ({n} rows)")

    print(f"verified task ids -> {len(keep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
