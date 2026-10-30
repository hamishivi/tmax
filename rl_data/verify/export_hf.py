#!/usr/bin/env python3
"""Join rubric labels onto the tmax-15k RL records and emit an HF-ready dataset.

NOT reusing ``rl_data/upload_to_hf.py``: it keys off each task dir's
``task.json``/``task.toml``, and the published tmax-15k task-data dirs contain
only ``instruction.md``, ``setup.sh`` and ``tests/test.sh``. Its
``_read_task_metadata`` returns None for every one of them, so it would upload
an empty parquet without complaining.

The published artifact is the ORIGINAL RL record plus verification columns, so
the dataset stays a drop-in for training and the filtering is a downstream
choice rather than something baked in:

  messages, ground_truth, dataset, env_config, source   (unchanged, verbatim)
  verify_label, verify_reason, verify_confidence,
  verify_mechanism, verify_secondary_labels, verify_model
  keep_policy_a   CLEAN + TASK-TRIVIAL
  keep_policy_b   policy A, plus VERIFIER-TOO-WEAK where the verifier does
                  validate the answer (exact-compare / recomputes /
                  fuzz-equivalence / oracle-diff) but not the prescribed method

Both flags ship because the choice between them is a judgement about whether we
want instruction fidelity or learning signal, and it should be revisable without
re-grading 14.6k tasks.

  python3 export_hf.py --records ... --labels labels.jsonl --out-dir hf_export/
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# Mechanisms where the verifier genuinely establishes the answer is correct,
# even if it never checks the artifact the instruction demanded.
ANSWER_VALIDATING = ("exact-compare", "recomputes", "fuzz-equivalence", "oracle-diff")
POLICY_A_LABELS = frozenset({"CLEAN", "TASK-TRIVIAL"})


def mechanism_validates_answer(mech: str) -> bool:
    m = (mech or "").strip().lower()
    return any(m.startswith(k) for k in ANSWER_VALIDATING)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="original tmax-15k jsonl")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="export even if some records have no label (default: refuse)",
    )
    args = ap.parse_args()

    labels: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for line in open(args.labels, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from an interrupted run
        tid = r.get("task_id")
        if not tid:
            continue
        if r.get("error"):
            errors.setdefault(tid, r["error"])
        elif r.get("label"):
            labels[tid] = r  # a later good row supersedes an earlier failure

    rows, unlabelled = [], []
    for line in open(args.records, encoding="utf-8"):
        rec = json.loads(line)
        tid = (rec.get("env_config") or {}).get("task_id")
        lab = labels.get(tid)
        if lab is None:
            unlabelled.append(tid)
            continue
        mech = lab.get("verifier_validates_how", "")
        label = lab["label"]
        a = label in POLICY_A_LABELS
        rows.append(
            {
                **rec,  # messages / ground_truth / dataset / env_config / source, verbatim
                "verify_label": label,
                "verify_reason": lab.get("reason", ""),
                "verify_confidence": lab.get("confidence"),
                "verify_mechanism": mech,
                "verify_secondary_labels": lab.get("secondary_labels", []),
                "verify_model": lab.get("model", ""),
                "keep_policy_a": a,
                "keep_policy_b": a
                or (label == "VERIFIER-TOO-WEAK" and mechanism_validates_answer(mech)),
            }
        )

    total = len(rows) + len(unlabelled)
    print(f"records            : {total}")
    print(f"labelled           : {len(rows)}")
    print(f"unlabelled         : {len(unlabelled)}  (of which {len(errors)} recorded an error)")
    if unlabelled and not args.allow_partial:
        print(f"  sample: {unlabelled[:5]}")
        print("\nREFUSING to export a partial dataset. Re-run with --allow-partial to override.")
        return 1
    if not rows:
        print("nothing to export")
        return 1

    dist = collections.Counter(r["verify_label"] for r in rows)
    ka = sum(1 for r in rows if r["keep_policy_a"])
    kb = sum(1 for r in rows if r["keep_policy_b"])
    print("\nlabel distribution:")
    for k, v in dist.most_common():
        print(f"  {v:6d}  {100*v/len(rows):5.1f}%  {k}")
    print(f"\nkeep_policy_a (CLEAN + TASK-TRIVIAL)        : {ka:6d}  {100*ka/len(rows):.1f}%")
    print(f"keep_policy_b (+ answer-validating verifiers): {kb:6d}  {100*kb/len(rows):.1f}%")

    os.makedirs(os.path.join(args.out_dir, "data"), exist_ok=True)
    jl = os.path.join(args.out_dir, "tmax15k-verified.jsonl")
    with open(jl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {jl}")

    try:
        import pandas as pd
    except ImportError:
        print("pandas unavailable; skipped parquet (HF Dataset Viewer needs data/*.parquet)")
        return 0
    df = pd.DataFrame(rows)
    # env_config is a struct and messages a list-of-struct; both survive parquet.
    pq = os.path.join(args.out_dir, "data", "train-00000-of-00001.parquet")
    df.to_parquet(pq, index=False)
    print(f"wrote {pq}  ({len(df)} rows, {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
