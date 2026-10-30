#!/usr/bin/env python3
"""Fix the runtime defects the sanity pass found. Mechanical, one pass, no model.

hamishivi 2026-09-13 sanctioned "fixes for failures those checks reveal ... a
bounded runtime sanity-check pass, not another audit or an iterative repair
campaign". So this applies one deterministic patch per defect class and stops.

The dominant defect is an UNDECLARED DEPENDENCY in the verifier: it does
`import requests` (662 tasks) or numpy/yaml/etc., the image does not ship it, and
pytest dies at collection before grading anything. The task then scores 0 no matter
what the agent did. This is pre-existing -- the ORIGINAL verifiers import the same
modules -- so it is a defect in the corpus, not something the rewrite introduced.

The fix is not to rewrite the verifier's logic. It is to let the verifier satisfy
its own dependency before it runs, which changes nothing about what the agent must
do and nothing about what is asserted:

    python3 -c 'import requests' 2>/dev/null || pip install --quiet requests || true

Measured in-sandbox: offline pip fails ("no matching distribution"), network pip
succeeds. `|| true` keeps a network-less environment exactly as broken as it was
rather than newly failing the script under `set -e`.

The original verifier is kept at test.sh.prepatch so the change is auditable.

  python3 fix_runtime.py --sanity sanity/ --fixes fixes/ [--apply]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys

# Only modules that are plausibly pip-installable and are the verifier's own
# dependency. A missing first-party module (the agent was meant to create it) is
# NOT a verifier defect and must not be papered over.
INSTALLABLE = {
    "requests", "numpy", "pandas", "yaml", "sklearn", "joblib", "h5py",
    "pymongo", "annoy", "scipy", "bs4", "lxml", "jinja2", "dateutil",
    "pytz", "tomli", "toml", "psutil", "cryptography", "jsonschema",
}
PIP_NAME = {"yaml": "pyyaml", "sklearn": "scikit-learn", "bs4": "beautifulsoup4",
            "dateutil": "python-dateutil", "cv2": "opencv-python-headless"}

BOOTSTRAP = """\
# --- runtime-defect fix: the verifier's own imports are not in this image ---
# Without this, pytest dies at collection with ModuleNotFoundError and the task
# scores 0 regardless of what the agent did. Pre-existing: the original verifier
# imports the same modules. `|| true` so a network-less run is no worse than before.
for _m in {mods}; do
  python3 -c "import ${{_m%%:*}}" >/dev/null 2>&1 && continue
  # Bounded and fully detached from the parent's pipes. Without `timeout` and the
  # </dev/null + >/dev/null redirects this hangs: the harness captures output, so
  # subprocess.run blocks until every writer closes the pipe, and a slow pip then
  # wedges the whole episode past its deadline. Measured: episodes stopped
  # finishing entirely until this was bounded.
  timeout 90 pip install --quiet --disable-pip-version-check --no-input \
      "${{_m##*:}}" </dev/null >/dev/null 2>&1 || true
done
# --- end fix ---
"""


def missing_modules(obs: str) -> list[str]:
    return sorted({m for m in re.findall(r"No module named '?\"?([A-Za-z0-9_]+)", obs)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanity", required=True)
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.apply:
        ap.error("Verifier-time installation is superseded. Use build_images.py with an explicit pinned dependency plan.")

    rows = []
    for f in sorted(glob.glob(os.path.join(args.sanity, "sanity*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    bad = [r for r in rows if r["verdict"] in ("CRASH", "BUILD_FAILED", "EXEC_BLOCKED")]
    print(f"sanity rows {len(rows)}  defective {len(bad)}")

    planned, skipped = [], collections.Counter()
    for r in bad:
        mods = [m for m in missing_modules(r.get("observation") or "")]
        inst = [m for m in mods if m in INSTALLABLE]
        if not mods:
            skipped["no missing module (other defect class)"] += 1
            continue
        if not inst:
            skipped[f"missing module not pip-installable: {','.join(mods[:2])}"] += 1
            continue
        planned.append((r["task_id"], inst))

    print(f"\npatchable (undeclared pip dependency): {len(planned)}")
    print("not patched here:")
    for k, n in skipped.most_common(8):
        print(f"  {n:5d}  {k}")
    mc = collections.Counter(m for _, ms in planned for m in ms)
    print(f"\nmodules to be bootstrapped: {dict(mc.most_common(8))}")

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
        return 0

    written = 0
    for tid, mods in planned:
        p = os.path.join(args.fixes, tid, "test.sh")
        if not os.path.isfile(p):
            continue
        body = open(p, encoding="utf-8").read()
        if "runtime-defect fix" in body:
            continue
        if not os.path.isfile(p + ".prepatch"):
            shutil.copyfile(p, p + ".prepatch")
        spec = " ".join(f'"{m}:{PIP_NAME.get(m, m)}"' for m in mods)
        boot = BOOTSTRAP.format(mods=spec)
        lines = body.splitlines(keepends=True)
        # keep the shebang first, insert the bootstrap immediately after it
        at = 1 if lines and lines[0].startswith("#!") else 0
        out = "".join(lines[:at]) + boot + "".join(lines[at:])
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(out)
        os.chmod(p, 0o755)
        written += 1
    print(f"\npatched {written} verifiers (original kept at test.sh.prepatch)")
    json.dump([t for t, _ in planned],
              open(os.path.join(args.sanity, "patched_tasks.json"), "w"))
    print(f"-> {os.path.join(args.sanity, 'patched_tasks.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
