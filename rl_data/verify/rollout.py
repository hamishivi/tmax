#!/usr/bin/env python3
"""Correctness filter: attempt each task N times with a strong model, keep the
tasks it can actually solve at least once.

Runs SWERLVanilluxSandboxEnv with backend="apptainer", which starts sandboxes
inside THIS job's allocation. There is no sandfleet daemon and no shared pool, so
this cannot collide with a concurrently running RL fleet.

The single most important thing here is that a zero is not a zero:

  solved          reward > 0                       -> task is satisfiable
  unsolved        agent finished, verifier said 0  -> real negative evidence
  infra           sandbox lost / OOM / backend died -> NO evidence either way
  no_submit       hit max steps without submitting  -> weak negative evidence
  error           harness threw                     -> NO evidence
  invalid_reward  verifier produced no usable score -> NO evidence

Only `solved` and `unsolved` are evidence. Counting an infra failure as "the model
could not solve it" silently deletes good tasks from the filtered set, which is the
exact failure this pass exists to prevent. Infra rows are retried, not scored.

Traces are written per attempt so the kept set is auditable rather than asserted.

  python3 rollout.py --corpus corpus-wave1/ --out rollouts/ \
      --attempts 2 --concurrency 32 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
import threading
import traceback
import math
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import ProviderAuthError, RateLimiter, complete, load_api_key  # noqa: E402
from image_safety import corpus_fingerprints  # noqa: E402
from pathlib import Path

ROLLOUT_MODEL = os.getenv("VERIFY_ROLLOUT_MODEL", "gemini/gemini-3-flash-preview")
SUBMIT_HINT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM = """\
You are a terminal agent solving a task in a Linux sandbox. You have exactly one
tool: `bash`. Work by running shell commands.

Explore first, then implement, then verify your own work before submitting.

When the task is complete, run exactly:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
That ends the episode and runs the grader. Do not run it until you are done.\
"""

# Keep the system prompt, the task statement, and the most recent exchanges.
# Without this the payload grows every step and both latency and Python-side CPU
# climb until throughput collapses.
KEEP_RECENT_TURNS = 12
OBS_CHARS = 1500


def trim(messages: list[dict]) -> list[dict]:
    if len(messages) <= 2 + KEEP_RECENT_TURNS:
        return messages
    head, tail = messages[:2], messages[2:]
    dropped = len(tail) - KEEP_RECENT_TURNS
    # Never start a retained window with an orphaned tool result. Gemini also
    # needs the originating assistant message's provider-specific signature.
    while dropped > 0 and tail[dropped].get("role") == "tool":
        dropped -= 1
    note = {"role": "user",
            "content": f"[{dropped} earlier turns elided to bound context]"}
    return head + [note] + tail[dropped:]


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the sandbox and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def step_failure(result) -> str | None:
    """Separate missing measurements and execution failures from scored zeroes."""
    metadata = getattr(result, 'metadata', None) or {}
    if any(metadata.get(name) for name in ('sandbox_lost', 'infrastructure_failure',
                                           'oom_killed', 'backend_unavailable')):
        return 'infra'
    if not result.done:
        return None
    if metadata.get('timeout'):
        return 'timeout'
    if metadata.get('invalid_reward'):
        return 'invalid_reward'
    try:
        reward = float(result.reward)
    except (TypeError, ValueError):
        return 'invalid_reward'
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        return 'invalid_reward'
    return None


def load_corpus(corpus: str) -> dict[str, str]:
    """task_id -> absolute .sif path, read from the corpus's own tasks.jsonl."""
    out: dict[str, str] = {}
    with open(os.path.join(corpus, "tasks.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ec = r.get("env_config")
            if isinstance(ec, str):
                ec = json.loads(ec)
            if ec and ec.get("task_id") and ec.get("image"):
                out[ec["task_id"]] = ec["image"]
    return out


def _import_env():
    """Import the env lazily so --help works without the training deps."""
    root = os.getenv("VERIFY_TRAINING_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training/open-instruct")))
    if root not in sys.path:
        sys.path.insert(0, root)
    from open_instruct.environments.base import EnvCall  # noqa: PLC0415
    from open_instruct.environments.swerl_vanillux_sandbox import (  # noqa: PLC0415
        SWERLVanilluxSandboxEnv,
    )

    return SWERLVanilluxSandboxEnv, EnvCall


def extract_call(msg) -> tuple[str | None, str]:
    """Return (command, assistant_text). Accepts a tool call or a bare ```bash block."""
    text = (getattr(msg, "content", None) or "") or ""
    calls = getattr(msg, "tool_calls", None) or []
    if calls:
        fn = calls[0].function
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        cmd = args.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd, text
    # Some routes ignore tools and emit a fenced block instead; accept that rather
    # than burning the episode on a format technicality.
    if "```" in text:
        seg = text.split("```", 2)
        if len(seg) >= 2:
            body = seg[1]
            if body.startswith(("bash\n", "sh\n")):
                body = body.split("\n", 1)[1]
            if body.strip():
                return body.strip(), text
    return None, text


async def run_one(task_id: str, corpus: str, image: str, model: str, key: str,
                  limiter, max_steps: int, attempt: int, deadline_sec: int = 600,
                  artifact_id: str | None = None) -> dict:
    SWERLVanilluxSandboxEnv, EnvCall = _import_env()
    t0 = time.time()
    rec = {
        "task_id": task_id, "attempt": attempt, "model": model,
        "outcome": "error", "reward": 0.0, "steps": 0, "trace": [],
        "artifact_id": artifact_id,
    }
    env = None
    try:
        env = SWERLVanilluxSandboxEnv(
            backend="apptainer",
            image=image,
            task_data_dir=os.path.join(corpus, "task-data"),
            # The 600s episode deadline is only checked BETWEEN steps, so a single
            # blocking call can overrun it: a bash command at 120s or a verifier at
            # 900s. Tightened so the cap actually binds. Safe because no solved
            # episode in 3,650 observed took over 600s end to end.
            test_timeout=int(os.getenv("SWERL_TEST_TIMEOUT", "420")),
            timeout=int(os.getenv("SWERL_CMD_TIMEOUT", "90")),
            # ApptainerBackend defaults pwd="/workspace" and exports it as
            # APPTAINER_PWD on every exec -- but these images have no /workspace,
            # so the very command that would create it cannot start:
            #   FATAL: failed to set working directory: chdir /workspace
            # Start at / instead; the vanillux bash wrapper tracks cwd itself.
            pwd="/",
        )
        await env.setup()
        obs, _tools = await env.reset(task_id=task_id, image=image)

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": obs.result}]

        empty_streak = 0
        for step in range(max_steps):
            # 4% of episodes were consuming 52% of all episode-seconds while
            # sitting on a concurrency slot. Measured over 3,650 solved episodes,
            # ZERO took longer than 600s -- so this bounds the tail without
            # costing a single solve. Recorded as its own outcome rather than
            # folded into `unsolved`: it is budget-limited, not a verdict.
            if time.time() - t0 > deadline_sec:
                rec["outcome"] = "timeout"
                rec["detail"] = f"exceeded {deadline_sec}s wall-clock at step {step}"
                break
            # 120s, not the 600s default. Observed 95 sockets stuck in CLOSE-WAIT:
            # the server hangs up, the client does not notice, and a worker thread
            # blocks on a dead socket for the full timeout -- then retries 5x. That
            # alone dropped throughput to ~67 episodes/h. A chat completion that has
            # not answered in 120s is not going to.
            resp = complete(trim(messages), model=model, api_key=key,
                            max_output_tokens=4096, tools=[BASH_TOOL], limiter=limiter,
                            timeout=120)
            choice = resp.choices[0]
            msg = choice.message
            finish = getattr(choice, "finish_reason", None)
            cmd, text = extract_call(msg)
            if cmd is None:
                # An empty completion with no tool call is usually the route
                # declining the task, not a format slip. Burning all 40 turns on
                # it wastes the budget and, worse, records it as `no_submit`,
                # which reads as negative evidence about the TASK rather than a
                # refusal by the MODEL. Count the streak and label it honestly.
                empty_streak += 1
                rec["trace"].append({"step": step, "assistant": text[:2000],
                                     "command": None, "finish_reason": finish})
                # content_filter is intermittent: a task filtered on one turn can
                # pass on the next and go on to be solved (observed: filtered at
                # step 0, then solved in 11 steps). An empty completion costs
                # almost nothing, so retry generously before calling it a refusal.
                if empty_streak >= 6 and not text.strip():
                    rec["outcome"] = "model_declined"
                    rec["detail"] = f"3 consecutive empty completions (finish_reason={finish})"
                    break
                messages.append({"role": "assistant", "content": text or "(no output)"})
                messages.append({"role": "user", "content":
                                 "Respond with a `bash` tool call. To finish, run "
                                 f"`{SUBMIT_HINT}`."})
                continue
            empty_streak = 0

            messages.append(msg.model_dump(exclude_none=True) if msg.tool_calls else {"role": "assistant", "content": text or f"```bash\n{cmd}\n```"})
            result = await env.step(EnvCall(id=f"c{step}", name="bash", args={"command": cmd}))
            rec["steps"] = step + 1
            rec["trace"].append({"step": step, "assistant": text[:2000],
                                 "command": cmd[:4000],
                                 "observation": (result.result or "")[:4000]})
            obs_text = (result.result or "")
            if len(obs_text) > OBS_CHARS:
                obs_text = obs_text[: OBS_CHARS // 2] + "\n...[elided]...\n" + obs_text[-OBS_CHARS // 2:]
            if msg.tool_calls:
                for i, call in enumerate(msg.tool_calls):
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": obs_text if i == 0 else "Not executed: issue one bash command per turn."})
            else:
                messages.append({"role": "user", "content": obs_text})

            failure = step_failure(result)
            if failure:
                rec['outcome'] = failure
                rec['detail'] = failure
                rec['verifier_metadata'] = getattr(result, 'metadata', None) or {}
                break

            if result.done:
                r = float(result.reward or 0.0)
                rec["reward"] = r
                rec["outcome"] = "solved" if r > 0 else "unsolved"
                break
        else:
            rec["outcome"] = "no_submit"

    except ProviderAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        rec["outcome"] = "infra" if "Sandbox" in type(exc).__name__ else "error"
        rec["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass
    rec["elapsed_sec"] = round(time.time() - t0, 1)
    return rec


def main_sync(args) -> int:
    """Drive episodes on a thread pool, one event loop per thread.

    run_one is `async`, but everything inside it that costs wall-clock is
    BLOCKING: litellm's completion call is sync, and the apptainer backend shells
    out with subprocess.run. Awaiting those on one shared event loop serialises
    every episode -- measured 300 episodes/h against 192 nominal slots, an
    effective concurrency of ~4. Giving each episode its own thread and its own
    loop makes the declared concurrency real.
    """
    key = load_api_key(args.model)
    limiter = RateLimiter(args.rps)
    os.makedirs(args.out, exist_ok=True)
    suffix = f"-{args.shard:02d}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.out, f"rollouts{suffix}.jsonl")

    images = load_corpus(args.corpus)
    fingerprints = corpus_fingerprints(Path(args.corpus))
    tasks = sorted(images)
    if args.only_tasks:
        want = set(json.load(open(args.only_tasks, encoding="utf-8")))
        tasks = [t for t in tasks if t in want]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.num_shards > 1:
        tasks = tasks[args.shard :: args.num_shards]

    # Resume: a scored (task, attempt) is not re-run. `infra` and `error` rows are
    # NOT scored -- they carry no evidence about the task, so they must be retried.
    done: set[tuple[str, int]] = set()
    for path in sorted(glob.glob(os.path.join(args.out, "rollouts*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("artifact_id") == fingerprints.get(r.get("task_id"))
                    and r.get("model") == args.model
                    and r.get("outcome") in ("solved", "unsolved", "no_submit", "timeout")):
                done.add((r["task_id"], r["attempt"]))
    jobs = [(t, a) for t in tasks for a in range(args.attempts) if (t, a) not in done]

    print(f"tasks {len(tasks)}  attempts {args.attempts}  already scored {len(done)}")
    print(f"to run {len(jobs)}  concurrency {args.concurrency}  model {args.model}\n", flush=True)
    if not jobs:
        print("nothing to do")
        return 0

    fh = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()
    counts: dict[str, int] = {}
    t0 = time.time()
    state = {"n": 0}

    def one(job):
        task_id, attempt = job
        try:
            rec = asyncio.run(run_one(task_id, args.corpus, images[task_id],
                                      args.model, key, limiter, args.max_steps, attempt,
                                      args.deadline, fingerprints[task_id]))
        except ProviderAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            rec = {"task_id": task_id, "attempt": attempt, "model": args.model,
                   "outcome": "error", "reward": 0.0, "steps": 0, "trace": [],
                   "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
            rec["artifact_id"] = fingerprints[task_id]
        with lock:
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            state["n"] += 1
            n = state["n"]
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                rate = n / max(el, 1e-9)
                print(f"  {n}/{len(jobs)}  {dict(sorted(counts.items()))}  "
                      f"{rate*3600:.0f}/h  eta {(len(jobs)-n)/max(rate,1e-9)/3600:.1f}h",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, jobs))
    fh.close()

    print(f"\noutcomes: {dict(sorted(counts.items()))}")
    print(f"traces -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=ROLLOUT_MODEL)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--deadline", type=int, default=600,
                    help="per-episode wall-clock cap in seconds")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--only-tasks", default="", help="json list: restrict to these task_ids")
    args = ap.parse_args()
    return main_sync(args)


if __name__ == "__main__":
    sys.exit(main())
