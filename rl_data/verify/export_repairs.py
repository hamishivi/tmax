#!/usr/bin/env python3
"""Export the repair pass as an HF dataset: one row per repaired task.

Deliberately NOT a tree of 11.4k directories. The repairs are text, so one row
per task with the scripts as columns is both smaller and actually browsable in
the Dataset Viewer.

Refused repairs are included with ``accepted=false`` and their refusal reason.
Dropping them would hide the guard behaviour and make the acceptance rate
unauditable from the artifact alone.

  python3 export_repairs.py --fixes fixes/ --labels labels.jsonl --out-dir out/
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys


def read(path: str) -> str:
    return open(path, encoding="utf-8", errors="replace").read() if os.path.isfile(path) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    labels = {}
    for line in open(args.labels, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label"):
            labels[r["task_id"]] = r

    rows = []
    for tid in sorted(os.listdir(args.fixes)):
        d = os.path.join(args.fixes, tid)
        meta_path = os.path.join(d, "repair.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            m = json.load(open(meta_path, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        lab = labels.get(tid, {})
        rows.append(
            {
                "task_id": tid,
                "verify_label": m.get("label", ""),
                "verify_reason": lab.get("reason", ""),
                "verify_mechanism": lab.get("verifier_validates_how", ""),
                "repair_action": m.get("action", ""),
                "accepted": bool(m.get("accepted")),
                "refusal": m.get("refusal") or "",
                "notes": m.get("notes", ""),
                "test_sh": read(os.path.join(d, "test.sh")),
                "setup_sh": read(os.path.join(d, "setup.sh")),
                "leaked_paths": m.get("leaked_paths", []),
                "oracle_to_tests": m.get("oracle_to_tests", []),
                "needs_image_rebuild": bool(m.get("needs_image_rebuild")),
                "superseded": m.get("superseded", ""),
                "repair_model": m.get("model", ""),
            }
        )

    if not rows:
        print("no repair records found")
        return 1

    acc = [r for r in rows if r["accepted"]]
    print(f"repair records : {len(rows)}")
    print(f"accepted       : {len(acc)}  ({100*len(acc)/len(rows):.1f}%)")
    print(f"needs rebuild  : {sum(1 for r in acc if r['needs_image_rebuild'])}")
    print(f"oracle extract : {sum(1 for r in acc if r['oracle_to_tests'])}")
    print("\nby source label:")
    by = collections.Counter(r["verify_label"] for r in rows)
    ok = collections.Counter(r["verify_label"] for r in acc)
    for k in by:
        print(f"  {k:26s} {ok[k]:6d} / {by[k]:6d}")

    os.makedirs(os.path.join(args.out_dir, "data"), exist_ok=True)
    jl = os.path.join(args.out_dir, "tmax15k-repairs.jsonl")
    with open(jl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {jl}")

    try:
        import pandas as pd
    except ImportError:
        print("pandas unavailable; skipped parquet")
        return 0
    pq = os.path.join(args.out_dir, "data", "train-00000-of-00001.parquet")
    pd.DataFrame(rows).to_parquet(pq, index=False)
    print(f"wrote {pq}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
