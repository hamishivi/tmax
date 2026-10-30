"""Repair prompt and validation guards for defective tmax-15k tasks.

Two repair actions, chosen by label:

  rewrite_verifier   VERIFIER-TOO-WEAK, INSTR-VERIFIER-MISMATCH
                     Emit a replacement tests/test.sh. Uploaded at verify time,
                     so no image rebuild.
  remove_leak        ANSWER-LEAK
                     Declare leaked paths for removal from a rebuilt image. Move
                     required oracle assets into tests uploaded at submission.

The guards matter more than the prompt. A model asked to "make the verifier
stronger" will cheerfully emit something that never writes a reward file, or
that deletes a path the instruction tells the agent to read. Both fail silently
and only surface as mangled training data, so every repair is mechanically
checked and refused rather than trusted.
"""

from __future__ import annotations

import re
import shlex

try:
    from .image_safety import removal_paths
except ImportError:
    from image_safety import removal_paths

REWRITE_LABELS = frozenset({"VERIFIER-TOO-WEAK", "INSTR-VERIFIER-MISMATCH"})
LEAK_LABELS = frozenset({"ANSWER-LEAK"})
FIXABLE = REWRITE_LABELS | LEAK_LABELS

SYSTEM_PROMPT = """\
You repair defective tasks in a reinforcement-learning dataset of terminal-agent problems.

You are given a task's complete definition and an audit finding explaining what is
wrong with it. Produce the minimal repair that fixes exactly that defect.

Hard constraints:

- Do NOT change the instruction. The agent's task must stay the same.
- Do NOT make the task harder or easier. You are closing a loophole, not redesigning.
- The repaired verifier MUST still write "1" or "0" to /logs/verifier/reward.txt,
  exactly as the original does. A verifier that does not write a reward produces a
  harness error, not a score.
- Keep the original verifier's real assertions. Add what is missing; do not replace
  a working check with a different one.
- The verifier runs AFTER the agent, as root, in the same container.

For rewrite_verifier: the defect is that the verifier can be passed without doing
the work. Close that. Typically this means actually executing the artifact the
instruction demanded, checking it exists and is the right kind of thing, and
exercising it on inputs whose answers the verifier derives rather than trusts.
If the environment ships a reference implementation, also reject a submission that
is byte-identical to it.

For an ANSWER-LEAK the fix is REMOVAL FROM THE AGENT IMAGE. Emit a replacement
setup.sh that no longer leaves the leaking file in the built image.

Do NOT try to defend a leak by having the verifier reject a byte-identical or
hash-identical copy. That does not close it: the agent copies the oracle, appends
a harmless comment, and passes both the inequality check and the behavioural
tests. Removal is the only reliable fix.

If the verifier genuinely needs the oracle, it must not live in the agent image at
all. List those paths in `oracle_to_tests`; they will be extracted and shipped with
the tests, which upload only at verification time, and your replacement test_sh
must read them from /tests/<basename> instead of the old image path.

Never remove a path the instruction tells the agent to read or write.

Return only JSON matching the requested schema.\
"""

USER_TEMPLATE = """\
=== AUDIT FINDING ===
label:     {label}
mechanism: {mechanism}
finding:   {reason}

=== INSTRUCTION (do not change) ===
{instruction}

=== ENVIRONMENT SETUP (context only; it built the image and cannot be re-run) ===
{setup}

=== CURRENT VERIFIER (tests/test.sh) ===
{tests}

=== END ===

Repair this task. Respond with JSON only:
{{
  "action": "rewrite_verifier" | "remove_leak" | "skip",
  "test_sh": "<complete replacement tests/test.sh, or empty string>",
  "setup_sh": "<complete replacement setup.sh with the leak removed, or empty string>",
  "leaked_paths": ["<absolute paths no longer present in the rebuilt image>"],
  "oracle_to_tests": ["<leaked paths the verifier still needs, shipped via /tests>"],
  "notes": "<1-3 sentences: what you closed, or why you skipped>"
}}\
"""

RESPONSE_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_repair",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "test_sh", "setup_sh", "leaked_paths",
                         "oracle_to_tests", "notes"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["rewrite_verifier", "remove_leak", "skip"],
                },
                "test_sh": {"type": "string"},
                "setup_sh": {"type": "string"},
                "leaked_paths": {"type": "array", "items": {"type": "string"}},
                "oracle_to_tests": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
        },
    },
}


def build_messages(task: dict, label: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                label=label["label"],
                mechanism=label.get("verifier_mechanism") or label.get("verifier_validates_how", ""),
                reason=label.get("reason", ""),
                instruction=task["instruction"],
                setup=task["setup"],
                tests=task["tests"],
            ),
        },
    ]


# --- guards -----------------------------------------------------------------
# Each returns a reason string when the repair must be REFUSED, else None.

_REWARD_RE = re.compile(r"/logs/verifier/reward\.txt")
_PATH_RE = re.compile(r"(?<![\w/])/(?:[\w.@+-]+/)*[\w.@+-]+")


def _paths_in(text: str) -> set[str]:
    return set(_PATH_RE.findall(text or ""))


def check_repair(repair: dict, task: dict) -> str | None:
    """Mechanically validate a proposed repair. Returns a refusal reason or None."""
    action = repair.get("action")
    if action == "skip":
        return None
    if action not in ("rewrite_verifier", "remove_leak"):
        return f"unknown action {action!r}"

    test_sh = repair.get("test_sh") or ""
    setup_sh = repair.get("setup_sh") or ""
    leaked = [p for p in (repair.get("leaked_paths") or []) if p.strip()]
    to_tests = [p for p in (repair.get("oracle_to_tests") or []) if p.strip()]
    try:
        removal_paths(leaked)
        removal_paths(to_tests)
    except ValueError as exc:
        return str(exc)
    if not set(to_tests).issubset(leaked):
        return "oracle_to_tests must name paths being removed from the image"
    bases = [p.rsplit("/", 1)[-1] for p in to_tests]
    if len(bases) != len(set(bases)) or "test.sh" in bases:
        return "oracle basename collision in tests"

    if action == "rewrite_verifier":
        if not test_sh.strip():
            return "rewrite_verifier with empty test_sh"
    # Actions describe the main repair, but hybrid proposals may change both
    # the verifier and image. Validate every supplied change independently.
    if test_sh.strip():
        if not _REWARD_RE.search(test_sh):
            return "replacement verifier never writes /logs/verifier/reward.txt"

    if action == "remove_leak":
        if not setup_sh.strip():
            return "remove_leak with empty setup_sh (the fix is removal from the image)"
        if not leaked:
            return "remove_leak declares no leaked_paths"
    if setup_sh.strip() and not leaked:
        return "image rewrite declares no leaked_paths"
    if leaked:
        # THE caveat: never remove something the instruction tells the agent to use.
        instr_paths = _paths_in(task["instruction"])
        for p in leaked:
            if any(q == p or q.startswith(p + "/") for q in instr_paths):
                return f"would remove {p}, which the instruction references"
        # The rebuilt setup must genuinely stop producing the leak.
        for p in leaked:
            if p in _paths_in(setup_sh) and not re.search(
                rf"(rm|shred|unlink)\b[^\n]*{re.escape(p)}", setup_sh
            ):
                return f"setup.sh still creates {p} and never removes it"
        # A path the verifier needs must be relocated, not merely deleted.
        verifier_text = test_sh if test_sh.strip() else task["tests"]
        verifier_paths = _paths_in(verifier_text)
        for p in leaked:
            if any(q == p or q.startswith(p + "/") for q in verifier_paths) and p not in to_tests:
                return f"verifier still reads {p} but it is not in oracle_to_tests"
        # Relocated oracles must actually be read from the tests-side path.
        for p in to_tests:
            base = p.rsplit("/", 1)[-1]
            if f"/tests/{base}" not in (test_sh or ""):
                return f"{p} relocated to tests but test_sh never reads /tests/{base}"

    # hamishivi 2026-09-11T20:00:54Z: byte/hash inequality does not close a leak --
    # copy the oracle, append a comment, and it passes. Refuse it as a sole defence.
    if repair.get("_label") == "ANSWER-LEAK" and test_sh.strip():
        anti_copy = re.search(
            r"(read\(\)\s*!=|!=\s*\w*oracle\w*|byte-identical|identical copy"
            r"|sha256\w*\s*!=|md5\w*\s*!=|filecmp\.cmp)", test_sh, re.I
        )
        removed = bool(leaked)
        if anti_copy and not removed:
            return ("relies on byte/hash inequality to defend a leak; that is "
                    "bypassable by appending a comment. Remove the file from the image.")

    return None


def shell_syntax_error(script: str) -> str | None:
    """`bash -n` the script; returns stderr on failure."""
    import subprocess

    if not script.strip():
        return None
    p = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True, timeout=30
    )
    return None if p.returncode == 0 else (p.stderr or "syntax error").strip()[:200]


__all__ = [
    "FIXABLE", "REWRITE_LABELS", "LEAK_LABELS", "RESPONSE_SCHEMA",
    "build_messages", "check_repair", "shell_syntax_error", "shlex",
]
