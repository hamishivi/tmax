#!/usr/bin/env python3
"""Repair defective tmax-15k tasks; refuse repairs that fail mechanical checks.

Resumable and concurrent, same shape as label_tasks.py. Every proposed repair is
run through `repair.check_repair` and `bash -n` before it is written, and a
refusal is recorded as a row rather than dropped, so the refusal rate is visible
instead of silently shrinking the output.

Writes one directory per repaired task:

  <out>/<task_id>/test.sh       replacement verifier   (rewrite_verifier)
  <out>/<task_id>/setup.sh      proposed build rewrite, retained as provenance
  <out>/<task_id>/repair.json   action, notes, guard result, source label

Nothing is applied in place. Producing the patch and adopting it are separate
steps on purpose -- a hardened verifier can be unsatisfiable, and that is only
provable by the rollout pass.

  python3 fix_tasks.py --index index.jsonl --labels labels.jsonl \
      --out fixes/ --limit 40
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import DEFAULT_MODEL, ProviderAuthError, RateLimiter, complete, load_api_key  # noqa: E402
from label_tasks import parse_label  # noqa: E402  (same tolerant JSON extraction)
from repair import (  # noqa: E402
    FIXABLE,
    LEAK_LABELS,
    RESPONSE_SCHEMA,
    REWRITE_LABELS,
    build_messages,
    check_repair,
    shell_syntax_error,
)

_lock = threading.Lock()


# No action/label coupling. Three times an "action must match the audited label"
# rule refused good repairs: ANSWER-LEAK pinned to remove_leak (3/20 -> 18/20 once
# relaxed), then 133 VERIFIER-TOO-WEAK tasks that also shipped a readable oracle,
# where the model correctly proposed removing it AND hardening the verifier.
# The content guards in repair.check_repair already enforce what actually matters
# -- a reward is written, no instruction path is deleted, a relocated oracle is
# really read from /tests, setup.sh no longer produces the leak. The taxonomy
# check added nothing on top and only rejected repairs for being better-informed
# than the label.


def repair_one(task: dict, label: dict, model: str, key: str, limiter, max_out: int) -> dict:
    t0 = time.time()
    resp = complete(
        build_messages(task, label),
        model=model,
        api_key=key,
        max_output_tokens=max_out,
        response_format=RESPONSE_SCHEMA,
        limiter=limiter,
    )
    rep = parse_label(resp.choices[0].message.content)
    rep = {
        "action": str(rep.get("action", "")).strip(),
        "test_sh": rep.get("test_sh") or "",
        "setup_sh": rep.get("setup_sh") or "",
        "leaked_paths": [p for p in (rep.get("leaked_paths") or []) if str(p).strip()],
        "oracle_to_tests": [p for p in (rep.get("oracle_to_tests") or []) if str(p).strip()],
        "notes": str(rep.get("notes", "")).strip(),
        # the byte-inequality guard is label-sensitive
        "_label": label["label"],
    }

    refusal = check_repair(rep, task)
    if refusal is None:
        for name, script in (("test_sh", rep["test_sh"]), ("setup_sh", rep["setup_sh"])):
            err = shell_syntax_error(script)
            if err:
                refusal = f"{name} is not valid bash: {err}"
                break

    return {
        "task_id": task["task_id"],
        "label": label["label"],
        "action": rep["action"],
        "accepted": refusal is None and rep["action"] != "skip",
        "refusal": refusal,
        "notes": rep["notes"],
        "leaked_paths": rep["leaked_paths"],
        "oracle_to_tests": rep["oracle_to_tests"],
        "needs_image_rebuild": bool(rep["leaked_paths"] or rep["setup_sh"].strip()),
        "_scripts": {"test.sh": rep["test_sh"], "setup.sh": rep["setup_sh"]},
        "model": model,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def write_repair(out_dir: str, rec: dict) -> None:
    d = os.path.join(out_dir, rec["task_id"])
    os.makedirs(d, exist_ok=True)
    scripts = rec.pop("_scripts")
    # A refused retry must not leave an older accepted script on disk.
    for name in ("test.sh", "setup.sh"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            os.unlink(p)
    if rec["accepted"]:
        for name, body in scripts.items():
            if body.strip():
                p = os.path.join(d, name)
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(body if body.endswith("\n") else body + "\n")
                os.chmod(p, 0o755)
    with open(os.path.join(d, "repair.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("VERIFY_MODEL", DEFAULT_MODEL))
    ap.add_argument("--only", default="", help="comma-separated labels to repair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    args = ap.parse_args()

    want = {s.strip().upper() for s in args.only.split(",") if s.strip()} or FIXABLE
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(args.index, encoding="utf-8")}
    labels = {}
    for line in open(args.labels, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label"):
            labels[r["task_id"]] = r

    done = set(os.listdir(args.out)) if os.path.isdir(args.out) else set()
    todo = [
        t for t, lab in labels.items()
        if lab["label"] in want and t in tasks and t not in done
    ]
    todo.sort()
    if args.limit:
        todo = todo[: args.limit]

    print(f"repairable labels : {sorted(want)}")
    print(f"already repaired  : {len(done)}")
    print(f"to repair         : {len(todo)}")
    print(f"model {args.model}  concurrency {args.concurrency}  rps {args.rps}\n")
    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(args.out, exist_ok=True)
    key = load_api_key(args.model)          # once, not once per task
    limiter = RateLimiter(args.rps)
    stats = collections.Counter()
    refusals = collections.Counter()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(repair_one, tasks[t], labels[t], args.model, key,
                        limiter, args.max_output_tokens): t
            for t in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            try:
                rec = fut.result()
            except ProviderAuthError as exc:
                print(f"\nFATAL auth/entitlement: {exc}")
                for f in futs:
                    f.cancel()
                break
            except Exception as exc:  # noqa: BLE001
                rec = {"task_id": tid, "label": labels[tid]["label"], "action": "error",
                       "accepted": False, "refusal": f"{type(exc).__name__}: {str(exc)[:200]}",
                       "notes": "", "leaked_paths": [], "oracle_to_tests": [],
                       "needs_image_rebuild": False, "_scripts": {}, "model": args.model}
            stats["accepted" if rec["accepted"] else "refused"] += 1
            if rec.get("refusal"):
                refusals[rec["refusal"].split(":")[0][:60]] += 1
            with _lock:
                write_repair(args.out, rec)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  accepted={stats['accepted']} "
                      f"refused={stats['refused']}  {i/max(time.time()-t0,1e-9):.2f}/s", flush=True)

    n = sum(stats.values()) or 1
    print(f"\naccepted {stats['accepted']}  refused {stats['refused']}  "
          f"({100*stats['accepted']/n:.0f}% accepted)")
    if refusals:
        print("\nwhy repairs were refused:")
        for k, v in refusals.most_common(10):
            print(f"  {v:4d}  {k}")
    print(f"\npatches -> {args.out}  (nothing applied in place)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
