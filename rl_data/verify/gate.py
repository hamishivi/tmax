#!/usr/bin/env python3
"""Two-sided gate for a replacement verifier, run before it is allowed to ship.

The audit found that 26% of the cases where a hardened verifier rejected a
previously-passing solution were defects in the verifier itself: a Python
format-string crash, assertions on source text rather than behaviour, PATH
sandboxing that made the submission exit 126, and verifier-side rebuilds that
failed on their own. A crashing verifier is strictly worse than the weak one it
replaced, because the task becomes permanently unpassable.

Every one of those would have been caught by actually running the thing. So:

  POSITIVE arm   replay a known-good solution -> the verifier MUST pass it.
                 Fails => the verifier is broken or over-specified.
  NEGATIVE arm   submit nothing at all        -> the verifier MUST fail it.
                 Passes => the verifier is still a rubber stamp.

Both arms are required. A one-sided positive gate would readmit exactly the weak
verifiers this whole exercise exists to remove, and a one-sided negative gate is
what we already had.

  python3 gate.py --tasks ids.json --verifier-dir fixes2/ --out gate/
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import SUBMIT_HINT, _import_env, load_corpus  # noqa: E402

B = os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification"))
CONTROL = f"{B}/rl15k-verify/corpus-control"
CMD_CAP = 4000


def known_good(task_id: str) -> list[str] | None:
    """Commands from an episode that passed the ORIGINAL verifier."""
    best = None
    for f in glob.glob(f"{B}/rl15k-verify/rollouts-control/rollouts-*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_id") == task_id and r.get("outcome") == "solved":
                if best is None or r["steps"] < best["steps"]:
                    best = r
    if best is None:
        return None
    cmds = []
    for t in best.get("trace", []):
        c = t.get("command")
        if not c:
            continue
        if len(c) >= CMD_CAP:
            return None
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in c:
            continue
        cmds.append(c)
    return cmds or None


def _source_dir(task_id: str) -> str:
    """Locate the task's data dir. corpus-control holds only the 800-task A/B
    sample, so looking only there fails for everything else."""
    for c in ("corpus-wave1", "corpus-wave2", "corpus-control", "corpus-control-full"):
        p = os.path.join(B, "rl15k-verify", c, "task-data", task_id)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no task-data dir for {task_id} in any corpus")


def stage(task_id: str, verifier: str, root: str) -> str:
    """Build a one-task corpus whose tests/test.sh is the verifier under test."""
    d = os.path.join(root, task_id, "task-data", task_id)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    src = _source_dir(task_id)
    shutil.copyfile(os.path.join(src, "instruction.md"), os.path.join(d, "instruction.md"))
    seeds = os.path.join(src, "environment", "seeds")
    if os.path.isdir(seeds):
        shutil.copytree(seeds, os.path.join(d, "environment", "seeds"), dirs_exist_ok=True)
    shutil.copyfile(verifier, os.path.join(d, "tests", "test.sh"))
    os.chmod(os.path.join(d, "tests", "test.sh"), 0o755)
    # any relocated oracle must travel with the tests
    assets = os.path.join(B, "rl15k-verify", "fixes", task_id, "tests_assets")
    if os.path.isdir(assets):
        for f in os.listdir(assets):
            shutil.copyfile(os.path.join(assets, f), os.path.join(d, "tests", f))
            os.chmod(os.path.join(d, "tests", f), 0o755)
    return os.path.join(root, task_id, "task-data")


async def run_arm(task_id: str, task_data: str, image: str, cmds: list[str]) -> dict:
    SWERLVanilluxSandboxEnv, EnvCall = _import_env()
    env = None
    out = {"reward": None, "error": None}
    try:
        env = SWERLVanilluxSandboxEnv(
            backend="apptainer", image=image, task_data_dir=task_data,
            test_timeout=420, timeout=90, pwd="/",
        )
        await env.setup()
        await env.reset(task_id=task_id, image=image)
        for i, c in enumerate(cmds):
            res = await env.step(EnvCall(id=f"g{i}", name="bash", args={"command": c}))
            meta = getattr(res, "metadata", None) or {}
            if meta.get("sandbox_lost") or meta.get("oom_killed"):
                out["error"] = "infra"
                return out
        res = await env.step(EnvCall(id="s", name="bash", args={"command": SUBMIT_HINT}))
        out["reward"] = float(res.reward or 0.0)
        out["observation"] = (res.result or "")[:2500]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def gate_one(task_id: str, verifier: str, images: dict, root: str) -> dict:
    rec = {"task_id": task_id, "verifier": verifier}
    good = known_good(task_id)
    if not good:
        rec["verdict"] = "UNGATEABLE (no clean known-good solution)"
        return rec
    img = images.get(task_id)
    if not img:
        rec["verdict"] = "UNGATEABLE (no image)"
        return rec
    td = stage(task_id, verifier, root)
    # A single failed positive arm is not enough to condemn a verifier: during
    # validation one good verifier failed once and then passed 6/6 on re-run, a
    # transient, not a defect. Rejecting on that would discard sound tasks --
    # the same mistake as scoring an infra failure as "unsolvable". Retry before
    # concluding; a genuine defect fails every time.
    attempts = []
    for _ in range(3):
        a = asyncio.run(run_arm(task_id, td, img, good))
        attempts.append(a)
        if (a.get("reward") or 0) > 0:
            break
    rec["positive"] = attempts[-1]
    rec["positive_attempts"] = len(attempts)
    rec["positive_rewards"] = [a.get("reward") for a in attempts]
    rec["negative"] = asyncio.run(run_arm(task_id, td, img, []))  # submit nothing
    p = rec["positive"].get("reward")
    n = rec["negative"].get("reward")
    if p is None or n is None:
        rec["verdict"] = "INCONCLUSIVE (arm errored)"
    elif p > 0 and n == 0:
        rec["verdict"] = "PASS"
    elif p == 0:
        rec["verdict"] = "REJECT: fails a known-good solution"
    elif n > 0:
        rec["verdict"] = "REJECT: passes a do-nothing submission"
    else:
        rec["verdict"] = "REJECT"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--verifier-dir", required=True,
                    help="<dir>/<task_id>/test.sh is the verifier under test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    images = load_corpus(CONTROL)
    tasks = json.load(open(args.tasks, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(args.out, "_staging")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(args.out, "gate.jsonl")
    done = set()
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = []
    for t in tasks:
        v = os.path.join(args.verifier_dir, t, "test.sh")
        if t not in done and os.path.isfile(v):
            todo.append((t, v))
    if args.limit:
        todo = todo[: args.limit]
    print(f"gating {len(todo)} verifiers (already done {len(done)})", flush=True)

    fh = open(path, "a", encoding="utf-8")
    lock = threading.Lock()
    counts: dict[str, int] = {}
    t0 = time.time()

    def work(item):
        t, v = item
        rec = gate_one(t, v, images, root)
        with lock:
            k = rec["verdict"].split(":")[0]
            counts[k] = counts.get(k, 0) + 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = sum(counts.values())
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {dict(sorted(counts.items()))}  "
                      f"{n/max(time.time()-t0,1e-9)*3600:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, todo))
    fh.close()
    print(f"\n{dict(sorted(counts.items()))}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
