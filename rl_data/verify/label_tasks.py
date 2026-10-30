#!/usr/bin/env python3
"""Label tmax-15k tasks against the triage rubric via MetaGen.

Resumable by construction: every completed task is appended to the output JSONL
immediately and re-runs skip what is already there. At 14.6k calls some will
fail transiently, and restarting from zero each time is not affordable.

  # preflight: one call, verifies alias + params + structured output
  python3 label_tasks.py --smoke

  # pilot before committing the full spend
  python3 label_tasks.py --index index.jsonl --out labels.jsonl --limit 50

  # full run (resumes automatically)
  python3 label_tasks.py --index index.jsonl --out labels.jsonl --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import (  # noqa: E402
    DEFAULT_MODEL,
    ProviderAuthError,
    RateLimiter,
    complete,
    load_api_key,
)
from rubric import LABELS, RESPONSE_SCHEMA, build_messages  # noqa: E402

_write_lock = threading.Lock()


def parse_label(text: str) -> dict:
    """Extract the JSON verdict.

    The handoff warns that structured output must be verified per alias rather
    than assumed, so we tolerate a model that returns JSON in prose or a fenced
    block instead of honouring response_format.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"no JSON object in response: {text[:200]}")


def normalise(verdict: dict) -> dict:
    label = str(verdict.get("label", "")).strip().upper()
    if label not in LABELS:
        # Do not silently coerce an unknown label into a real one; that would
        # quietly corrupt the keep-set.
        raise ValueError(f"label not in rubric: {label!r}")
    sec = verdict.get("secondary_labels") or []
    if isinstance(sec, str):
        sec = [sec]
    return {
        "label": label,
        "confidence": float(verdict.get("confidence", 0.0) or 0.0),
        "reason": str(verdict.get("reason", "")).strip(),
        "verifier_validates_how": str(verdict.get("verifier_validates_how", "")).strip(),
        "secondary_labels": [s.strip().upper() for s in sec if str(s).strip().upper() in LABELS],
    }


def grade_one(task: dict, model: str, key: str, limiter, use_schema: bool, max_out: int) -> dict:
    started = time.time()
    resp = complete(
        build_messages(task),
        model=model,
        api_key=key,
        max_output_tokens=max_out,
        response_format=RESPONSE_SCHEMA if use_schema else None,
        limiter=limiter,
    )
    content = resp.choices[0].message.content
    verdict = normalise(parse_label(content))
    usage = getattr(resp, "usage", None)
    verdict.update(
        {
            "task_id": task["task_id"],
            "model": model,
            "elapsed_sec": round(time.time() - started, 2),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        }
    )
    return verdict


def load_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            if rec.get("task_id") and not rec.get("error"):
                done.add(rec["task_id"])
    return done


def smoke(model: str, max_out: int) -> int:
    """One real call. Verifies alias entitlement, output-limit param, and schema."""
    key = load_api_key(model)
    print(f"model     : {model}")
    print("provider  :", model.split("/", 1)[0])
    print("key       : loaded (not shown)")

    print("\n[1/2] plain completion ...", flush=True)
    r = complete(
        [{"role": "user", "content": "Reply with exactly: ok"}],
        model=model,
        api_key=key,
        max_output_tokens=max_out,
    )
    print(f"  -> {r.choices[0].message.content!r}")

    print("[2/2] structured output (response_format) ...", flush=True)
    try:
        r2 = complete(
            [
                {"role": "system", "content": "Return only JSON."},
                {
                    "role": "user",
                    "content": "Label this: a task whose verifier only checks the output file exists. "
                    f"Use one of {', '.join(LABELS)}.",
                },
            ],
            model=model,
            api_key=key,
            max_output_tokens=max_out,
            response_format=RESPONSE_SCHEMA,
        )
        print(f"  -> {normalise(parse_label(r2.choices[0].message.content))}")
        print("\nSMOKE OK  (schema supported)")
    except Exception as exc:  # noqa: BLE001
        print(f"  structured output unsupported/failed: {type(exc).__name__}: {str(exc)[:200]}")
        print("\nSMOKE PARTIAL  (run with --no-schema; parser falls back to prose JSON)")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index")
    ap.add_argument("--out")
    ap.add_argument("--model", default=os.getenv("VERIFY_MODEL", DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=0, help="grade only the first N ungraded")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rps", type=float, default=4.0, help="request/sec cap across workers")
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke(args.model, args.max_output_tokens)
    if not args.index or not args.out:
        ap.error("--index and --out are required unless --smoke")

    key = load_api_key(args.model)
    tasks = [json.loads(l) for l in open(args.index, encoding="utf-8")]
    done = load_done(args.out)
    todo = [t for t in tasks if t["task_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"index {len(tasks)}  already graded {len(done)}  to grade {len(todo)}")
    print(f"model {args.model}  concurrency {args.concurrency}  rps {args.rps}")
    if not todo:
        print("nothing to do")
        return 0

    limiter = RateLimiter(args.rps)
    out = open(args.out, "a", encoding="utf-8")
    ok = failed = 0
    ptok = ctok = 0
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(
                    grade_one, t, args.model, key, limiter, not args.no_schema, args.max_output_tokens
                ): t
                for t in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                task = futs[fut]
                try:
                    rec = fut.result()
                    ok += 1
                    ptok += rec.get("prompt_tokens") or 0
                    ctok += rec.get("completion_tokens") or 0
                except ProviderAuthError as exc:
                    # Entitlement problems affect every remaining call; stop
                    # rather than burn the queue against a key that cannot work.
                    print(f"\nFATAL auth/entitlement: {exc}")
                    for f in futs:
                        f.cancel()
                    break
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    rec = {
                        "task_id": task["task_id"],
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                with _write_lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - i) / rate if rate else 0
                    print(
                        f"  {i}/{len(todo)}  ok={ok} failed={failed}  "
                        f"{rate:.2f}/s  eta {eta/60:.1f}m  tok in/out {ptok}/{ctok}",
                        flush=True,
                    )
    finally:
        out.close()

    print(f"\ndone: ok={ok} failed={failed}  tokens in={ptok} out={ctok}  {(time.time()-t0)/60:.1f}m")
    print(f"output -> {args.out}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
