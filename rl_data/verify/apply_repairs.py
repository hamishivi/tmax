#!/usr/bin/env python3
"""Materialise repairs: extract relocated oracles, and emit image-rebuild defs.

Two jobs, both driven off fixes/<task_id>/repair.json:

  extract   For each accepted repair declaring `oracle_to_tests`, copy that path
            out of the task's existing SIF into fixes/<task_id>/tests_assets/.
            It then ships with the tests, which upload only at verification, so
            the agent never sees it.

  defs      For each accepted repair with `needs_image_rebuild`, write an
            Apptainer def that rebuilds from the EXISTING image and removes the
            leaked paths.

Why localimage+rm rather than rebuilding from the repaired setup.sh:

  * allenai/TMax-15K carries the original `container_def`, but its %files stanzas
    point at generator-local paths, which do not exist here --
    a literal rebuild fails at the copy stage.
  * the model's replacement setup.sh has median similarity 0.59 to the original;
    asked to delete one file it rewrote ~40% of the build. Applying that is a
    much larger, unvalidatable change than the defect requires.

Removing the leaked path from the existing image closes the leak, which is the
whole point, and changes nothing else. The repaired setup_sh is retained in the
dataset as provenance, clearly marked as not-applied.

  python3 apply_repairs.py --fixes fixes/ --index index.jsonl --mode extract
  python3 apply_repairs.py --fixes fixes/ --index index.jsonl --mode defs --out defs/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shlex

try:
    from .image_safety import removal_paths
except ImportError:
    from image_safety import removal_paths


def load(fixes: str, index: str):
    idx = {json.loads(l)["task_id"]: json.loads(l) for l in open(index, encoding="utf-8")}
    out = []
    for tid in sorted(os.listdir(fixes)):
        p = os.path.join(fixes, tid, "repair.json")
        if not os.path.isfile(p):
            continue
        try:
            r = json.load(open(p, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if r.get("accepted") and tid in idx:
            out.append((r, idx[tid]))
    return out


def do_extract(items, fixes: str) -> int:
    todo = [(r, t) for r, t in items if r.get("oracle_to_tests")]
    print(f"tasks needing extraction: {len(todo)}")
    ok = miss_sif = miss_path = 0
    log = []
    for r, task in todo:
        tid = r["task_id"]
        removal_paths(r["oracle_to_tests"])
        names = [os.path.basename(p) for p in r["oracle_to_tests"]]
        if len(names) != len(set(names)) or "test.sh" in names:
            raise ValueError(f"Oracle basename collision: {tid}")
        sif = task.get("sif_path") or ""
        dest_dir = os.path.join(fixes, tid, "tests_assets")
        if not sif or not os.path.isfile(sif):
            miss_sif += 1
            log.append({"task_id": tid, "status": "sif_missing", "sif": sif})
            continue
        os.makedirs(dest_dir, exist_ok=True)
        for src in r["oracle_to_tests"]:
            dest = os.path.join(dest_dir, os.path.basename(src))
            if os.path.exists(dest):
                os.unlink(dest)  # a failed retry must not reuse an earlier extraction
            p = subprocess.run(
                ["apptainer", "exec", "--cleanenv", "--containall", "--pwd", "/", sif, "cat", "--", src],
                capture_output=True, timeout=180,
            )
            # An empty file is indistinguishable from a silent miss, so treat a
            # non-zero exit or zero bytes as a failure rather than writing it.
            if p.returncode != 0 or not p.stdout:
                miss_path += 1
                log.append({"task_id": tid, "status": "path_missing", "path": src,
                            "stderr": p.stderr.decode(errors="replace")[:160]})
                continue
            with open(dest, "wb") as fh:
                fh.write(p.stdout)
            os.chmod(dest, 0o755)
            ok += 1
            log.append({"task_id": tid, "status": "ok", "path": src,
                        "bytes": len(p.stdout), "dest": dest})
    print(f"  extracted        : {ok}")
    print(f"  SIF missing      : {miss_sif}")
    print(f"  path not in image: {miss_path}")
    with open(os.path.join(fixes, "_extract_log.jsonl"), "w", encoding="utf-8") as fh:
        for e in log:
            fh.write(json.dumps(e) + "\n")
    print(f"  log -> {os.path.join(fixes, '_extract_log.jsonl')}")
    return int(bool(miss_sif or miss_path))


def do_defs(items, out_dir: str) -> int:
    todo = [(r, t) for r, t in items if r.get("needs_image_rebuild") and r.get("leaked_paths")]
    print(f"tasks needing an image rebuild: {len(todo)}")
    os.makedirs(out_dir, exist_ok=True)
    written = skipped = 0
    for r, task in todo:
        tid = r["task_id"]
        sif = task.get("sif_path") or ""
        if not sif or not os.path.isfile(sif):
            skipped += 1
            continue
        paths = removal_paths(r["leaked_paths"])
        rms = "\n".join(f"    rm -rf -- {shlex.quote(p)}" for p in paths)
        verify = "\n".join(
            f'    test ! -e {p} || {{ echo "LEAK STILL PRESENT: {p}" >&2; exit 1; }}'
            for p in paths
        )
        body = f"""Bootstrap: localimage
From: {sif}

%post
    # Close the answer leak by removing it from the agent image. Only the
    # removal is applied; the model's wider setup.sh rewrite is retained in the
    # repair dataset as provenance but deliberately not executed.
{rms}
    # Fail the build rather than ship an image that still leaks.
{verify}

%labels
    repair.task_id {tid}
    repair.source_label {r.get('label','')}
    repair.leaked_paths {' '.join(r['leaked_paths'])}
    repair.model {r.get('model','')}
"""
        with open(os.path.join(out_dir, f"{tid}.def"), "w", encoding="utf-8") as fh:
            fh.write(body)
        written += 1
    print(f"  defs written : {written}")
    print(f"  skipped (no SIF): {skipped}")
    print(f"  -> {out_dir}")
    return 0


# Live RL jobs read the original corpus while this runs. Everything here is
# additive: the originals are inputs, never outputs. This refuses to write
# anywhere near them rather than relying on care.
PROTECTED = (
    os.path.join(os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification")), "sifs"),
    os.path.join(os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification")), "hf-cache"),
    os.path.join(os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification")), "exchange"),
)


def assert_safe_output(path: str) -> None:
    real = os.path.realpath(path)
    for p in PROTECTED:
        if real == p or real.startswith(p + "/"):
            raise SystemExit(
                f"REFUSING to write to {real}: that is the original corpus, which "
                "live RL jobs are reading. Repaired artifacts must go to a "
                "separate tree."
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--mode", choices=("extract", "defs"), required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    assert_safe_output(args.fixes)
    if args.out:
        assert_safe_output(args.out)
    items = load(args.fixes, args.index)
    print(f"accepted repairs: {len(items)}")
    if args.mode == "extract":
        return do_extract(items, args.fixes)
    if not args.out:
        ap.error("--out required for --mode defs")
    return do_defs(items, args.out)


if __name__ == "__main__":
    sys.exit(main())
