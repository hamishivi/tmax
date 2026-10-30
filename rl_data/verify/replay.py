#!/usr/bin/env python3
"""Replay one candidate solution against BOTH verifiers to isolate the verifier.

hamishivi 2026-09-12: "More steps or writing real code alone does not distinguish
those explanations. Where feasible, run the same candidate solution against both
verifiers to separate verifier changes from variation between rollouts."

That is the right experiment and this is it. For a task that passed its ORIGINAL
verifier but failed the HARDENED one, we take the exact command sequence the agent
ran in the passing episode and replay it in two fresh sandboxes off the same image
-- one graded by the original verifier, one by the hardened one. The agent is out
of the loop entirely, so rollout-to-rollout variance is removed by construction and
any difference in reward is attributable to the verifier.

Two separate sandboxes rather than two verifiers in one: a verifier may mutate the
filesystem, so grading twice in the same container would let the first contaminate
the second.

Outcomes per task:
  orig=1 hard=0   the verifier change is responsible -> inspect WHY
  orig=1 hard=1   the wave-1 failure was rollout variance, not the verifier
  orig=0 ...      the replay itself did not reproduce the original pass; no evidence

Commands stored in a trace are capped at 4000 chars, so any episode containing a
clipped command is refused rather than replayed as a silent truncation.

  python3 replay.py --tasks replay_candidates.json --out replay/ --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import SUBMIT_HINT, _import_env, load_corpus  # noqa: E402

B = os.path.abspath(os.getenv("VERIFY_ROOT", "rl_data/output/verification"))
CORPORA = {
    "original": f"{B}/rl15k-verify/corpus-control",
    "hardened": f"{B}/rl15k-verify/corpus-wave1",
}
CMD_CAP = 4000


def solved_commands(task_id: str) -> list[str] | None:
    """Commands from the control episode that passed the ORIGINAL verifier."""
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
            return None  # clipped: replaying it would run a different program
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in c:
            continue      # we re-issue submit ourselves, last
        cmds.append(c)
    return cmds or None


async def grade(task_id: str, corpus: str, image: str, cmds: list[str]) -> dict:
    SWERLVanilluxSandboxEnv, EnvCall = _import_env()
    env = None
    rec = {"reward": None, "error": None, "ran": 0}
    try:
        env = SWERLVanilluxSandboxEnv(
            backend="apptainer", image=image,
            task_data_dir=os.path.join(corpus, "task-data"),
            test_timeout=420, timeout=90, pwd="/",
        )
        await env.setup()
        await env.reset(task_id=task_id, image=image)
        for i, c in enumerate(cmds):
            res = await env.step(EnvCall(id=f"r{i}", name="bash", args={"command": c}))
            rec["ran"] = i + 1
            meta = getattr(res, "metadata", None) or {}
            if meta.get("sandbox_lost") or meta.get("oom_killed") or meta.get("backend_unavailable"):
                rec["error"] = "infra during replay"
                return rec
        res = await env.step(EnvCall(id="submit", name="bash",
                                     args={"command": SUBMIT_HINT}))
        rec["reward"] = float(res.reward or 0.0)
        rec["observation"] = (res.result or "")[:3000]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass
    return rec


def one(task_id: str, images: dict) -> dict:
    cmds = solved_commands(task_id)
    out = {"task_id": task_id, "n_commands": len(cmds) if cmds else 0}
    if not cmds:
        out["skip"] = "no clean passing episode (missing, or a command was clipped)"
        return out
    for arm, corpus in CORPORA.items():
        img = images[arm].get(task_id)
        if not img:
            out[arm] = {"error": "task not in this corpus"}
            continue
        out[arm] = asyncio.run(grade(task_id, corpus, img, cmds))
    o = (out.get("original") or {}).get("reward")
    h = (out.get("hardened") or {}).get("reward")
    if o is None or h is None:
        out["verdict"] = "inconclusive (replay error)"
    elif o > 0 and h == 0:
        out["verdict"] = "VERIFIER CHANGE is responsible"
    elif o > 0 and h > 0:
        out["verdict"] = "rollout variance, not the verifier"
    elif o == 0:
        out["verdict"] = "replay did not reproduce the original pass"
    else:
        out["verdict"] = "other"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    images = {k: load_corpus(v) for k, v in CORPORA.items()}
    tasks = json.load(open(args.tasks, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "replay.jsonl")
    done = set()
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [t for t in tasks if t not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"replay candidates {len(tasks)}  already done {len(done)}  to run {len(todo)}",
          flush=True)

    lock = threading.Lock()
    fh = open(path, "a", encoding="utf-8")
    counts: dict[str, int] = {}
    t0 = time.time()

    def work(t):
        rec = one(t, images)
        with lock:
            v = rec.get("verdict", rec.get("skip", "?"))
            counts[v] = counts.get(v, 0) + 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = sum(counts.values())
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {dict(sorted(counts.items()))}  "
                      f"{n/max(time.time()-t0,1e-9)*3600:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, todo))
    fh.close()
    print(f"\nverdicts: {dict(sorted(counts.items()))}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
