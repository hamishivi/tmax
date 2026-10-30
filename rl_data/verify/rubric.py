"""Rubric prompt and label schema for tmax-15k task triage.

The grader sees everything that defines a task: the instruction the agent gets,
the environment setup that builds the container, and the verifier that decides
reward. Those three files are the whole task definition --- the SIF contains
only fixtures, no verifier --- so nothing material is withheld.

Labels are exactly as specified by Oscar. We assign a label to every task and
keep the keep/drop decision as a separate, reversible filter downstream, so a
change of policy does not mean re-grading 14.6k tasks.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "CLEAN",
    "VERIFIER-TOO-WEAK",
    "INSTR-VERIFIER-MISMATCH",
    "INSTR-ENV-MISMATCH",
    "ANSWER-LEAK",
    "INSTR-AMBIGUOUS",
    "TASK-TRIVIAL",
    "OTHER",
)

# Current keep policy. Deliberately a constant, not baked into the grader.
KEEP_LABELS: frozenset[str] = frozenset({"CLEAN", "TASK-TRIVIAL"})

RUBRIC = """\
• CLEAN: Instruction is clear and self-sufficient, a correct-per-instruction solution passes, AND the verifier genuinely validates correctness: it recomputes/decodes the true answer from the input (or exact-compares against a correct hardcoded answer) rather than merely checking output shape. A hard-but-fair task is still CLEAN.
• VERIFIER-TOO-WEAK: The verifier passes without the task actually being solved: it checks only output shape/format/types/count, trusts a self-reported value, greps the source for keywords, checks a hardcoded subset, or tests only fixed inputs so the answer can be hardcoded. A fabricated, plausibly-shaped output would pass.
• INSTR-VERIFIER-MISMATCH: The verifier checks something the instruction does not imply, or contradicts it. For example, reads/writes a different path, asserts a hardcoded value that conflicts with the data, or demands a stricter format. A solution that faithfully follows the instruction still fails.
• INSTR-ENV-MISMATCH: The instruction or verifier references input paths, files, dependencies, or services the environment does not actually provide, so the task cannot be completed as written (missing data/compiler/service).
• ANSWER-LEAK: The instruction itself, or a file shipped in the environment, reveals the expected answer (lists the result, embeds the solution, or provides a ground-truth file), so the task requires no genuine work.
• INSTR-AMBIGUOUS: The instruction alone is underspecified: a necessary specification is missing, so two equally reasonable interpretations produce different outputs and the verifier accepts only one. (Merely admitting multiple valid solution paths is not enough.)
• TASK-TRIVIAL: The task itself is so easy there is no meaningful work or learning signal: write a constant, echo a value, copy a file, regardless of verifier strength.
• OTHER: A genuine defect fitting none of the above: verifier code that is buggy/crashes; a task needing network/GPU/GUI/hardware the container lacks; non-determinism; a required background service that is never started; or an internally contradictory spec.\
"""

SYSTEM_PROMPT = """\
You are auditing tasks for a reinforcement-learning dataset of terminal-agent problems.

Each task gives a coding agent a shell in a container. You are shown the complete task
definition: the instruction the agent receives, the setup script that builds the
container, and the verifier script that decides whether the agent succeeded. There is
nothing else — the container image contains only fixtures created by the setup script.

Assign exactly one label from the rubric. Judge the task as written, not as it could be
repaired.

Guidance that decides most hard cases:

- The central question for CLEAN is whether the verifier could be passed by a plausible
  fake. If an agent that never did the work could produce output that passes, the label
  is VERIFIER-TOO-WEAK, however sophisticated the task looks.
- A verifier that recomputes the answer from the input, diffs against a reference
  implementation or oracle binary, or fuzzes over many random inputs is genuinely
  validating. A verifier that only checks a file exists, matches a regex, has the right
  row count, or reports its own success is not.
- Difficulty is not a defect. A task can be very hard and still CLEAN.
- Prefer the most specific applicable label. Use OTHER only when no other fits.
- If several defects apply, pick the one that most determines the task's value as
  training data, and name the others in `secondary_labels`.

Return only JSON matching the requested schema. No prose outside it.\
"""

USER_TEMPLATE = """\
{rubric}

=== TASK METADATA ===
task_id:   {task_id}
image:     {image}
env_name:  {env_name}
source:    {source}

=== INSTRUCTION GIVEN TO THE AGENT (instruction.md) ===
{instruction}

=== ENVIRONMENT SETUP (setup.sh) ===
{setup}

=== VERIFIER (tests/test.sh) ===
{tests}

=== END OF TASK DEFINITION ===

Classify this task. Respond with JSON only:
{{
  "label": "<one of: {labels}>",
  "confidence": <float 0.0-1.0>,
  "reason": "<2-4 sentences citing the specific line, check, or path that decides it>",
  "verifier_validates_how": "<short phrase: recomputes | oracle-diff | fuzz-equivalence | exact-compare | shape-only | self-reported | keyword-grep | fixed-subset | other | none>",
  "secondary_labels": ["<any other applicable labels, or empty>"]
}}\
"""

# Enforced server-side where the alias supports it; the parser tolerates its
# absence because the handoff says structured output must be verified per alias
# rather than assumed.
RESPONSE_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_label",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "label",
                "confidence",
                "reason",
                "verifier_validates_how",
                "secondary_labels",
            ],
            "properties": {
                "label": {"type": "string", "enum": list(LABELS)},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
                "verifier_validates_how": {"type": "string"},
                "secondary_labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(LABELS)},
                },
            },
        },
    },
}


def build_messages(task: dict) -> list[dict[str, str]]:
    """Assemble the grading prompt for one task record."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                rubric=RUBRIC,
                task_id=task["task_id"],
                image=task.get("image", "(unknown)"),
                env_name=task.get("env_name", "(unknown)"),
                source=task.get("source", "(unknown)"),
                instruction=task["instruction"],
                setup=task["setup"],
                tests=task["tests"],
                labels=" | ".join(LABELS),
            ),
        },
    ]
