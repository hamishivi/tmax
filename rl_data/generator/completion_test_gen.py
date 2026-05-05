# generator/completion_test_gen.py
"""Generate a pytest *template* that validates the **final** state after the task
is completed.

This script consumes the task-template JSON produced by ``task_template_gen.py``
(which contains the description template, parameter schema and a privileged
``truth`` section).  It samples concrete values for each placeholder, renders a
full *task description* and then asks the LLM to create a single pytest file
(`test_final_state.py`) that passes **only if** the task has been solved
correctly.  The privileged ``truth`` data is forwarded to the LLM so the tests
can assert the exact expected end state.

V2 axes
-------
For tasks with a non-legacy ``verifier_kind`` (``metric_threshold`` /
``adversarial_corpus`` / ``fuzz_equivalence`` / ``multi_protocol``), the
default "stdlib + pytest only" constraint is *relaxed* — the verifier is
allowed to ``import`` a curated allow-list of third-party libs that the
``base_intricate.sif`` SIF pre-installs (numpy, scipy, scikit-learn, Pillow,
torch, biopython, etc.). The system prompt is also tailored per
``verifier_kind`` so the generated pytest matches the expected verifier
shape (metric vs corpus vs fuzz vs protocol).
"""
from __future__ import annotations

import textwrap
from typing import Optional

from rl_data import parse_python_code, check_python_code, chat_completion_batch, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# LLM prompt scaffolding — legacy (stdlib + pytest only)
# ---------------------------------------------------------------------------

SYSTEM_MSG = """You are a senior Python engineer who writes robust pytest suites.
Write a robust pytest suite that validates the **FINAL** state of the operating-system / container **after** the student has
completed the task described.
Use the privileged *truth* data to assert the exact expected end state for the task to be completed.

Rules:
* The filename must be ``test_final_state.py`` (show it in a header comment).
* Use **only** the Python standard library and ``pytest`` (no third-party libs).
* Failures must clearly explain **what is still wrong**.
* When you check for files or directories, always use their *absolute* paths exactly as given (no relative paths).
* Ensure that the the state of the OS matches the truth after the task is completed.
* Write the code in a fenced code block that can be parsed to get a single python file.

Ground-truth alignment (principled tests):
* Treat *truth* as the **intent** of the rubric, not as guaranteed-correct literals. When
  the task and setup logically determine an expected value, **derive or recompute** it in
  test code (stdlib only) instead of copying opaque constants from *truth* without checking.
* Match the **same procedures and ordering** as the described setup and task when you
  assert counts, checksums, or structured outputs—so tests stay faithful to the spec.
* Use the **strongest appropriate** assertion: prefer invariants, structure, and
  reproducible computations over brittle full-file equality when *truth* still allows the
  task to be graded fairly.
"""

USER_TEMPLATE = """The task description is: {task_description}
The truth value is: {truth}
The tests to check the initial container state, before the task is completed, are:
{initial_test_py}
Write the code in a fenced code block that can be parsed."""


# ---------------------------------------------------------------------------
# v2: per-verifier-kind allow-lists and prompt fragments
# ---------------------------------------------------------------------------
# Allow-list of third-party imports that ``base_intricate.sif`` ships
# pre-installed. Listed per verifier_kind so the prompt can be specific about
# what the verifier author may use; the SIF actually carries the union of
# these so any task can use any subset.
_VERIFIER_LIB_ALLOW_LIST: dict[str, tuple[str, ...]] = {
    # Names listed here are *import* names (what the verifier writes
    # ``import X``), not pip distribution names. base_intricate.sif provides:
    #   numpy / scipy / scikit-learn (import sklearn) / Pillow (import PIL) /
    #   torch / torchvision / biopython (import Bio) / pandas / matplotlib /
    #   imageio / beautifulsoup4 (import bs4) / lxml.
    "metric_threshold": (
        "numpy", "scipy", "sklearn", "PIL", "imageio",
        "torch", "torchvision", "Bio", "pandas", "matplotlib",
    ),
    "adversarial_corpus": (
        # Plain text matching is usually enough; a few extras for HTML/XML
        # adversarial cases.
        "bs4", "lxml",
    ),
    "fuzz_equivalence": (
        # Verifier mostly invokes binaries via subprocess; numpy is allowed
        # for input-shape generation.
        "numpy",
    ),
    "multi_protocol": (
        # All needed protocols are stdlib (http.client, smtplib, socket) but
        # `requests` is convenient.
        "requests",
    ),
}


def _format_allow_list(libs: tuple[str, ...]) -> str:
    if not libs:
        return "(none — stdlib + pytest only)"
    return ", ".join(libs)


_VERIFIER_KIND_PROMPT_FRAGMENTS: dict[str, str] = {
    "metric_threshold": """\
This task uses a METRIC-THRESHOLD verifier. The agent's output is graded
against a reference using a numerical metric, NOT exact text equality.

Your pytest MUST:
  * Implement the metric specified in *truth* (formula / tool / short
    snippet). Re-derive it in code rather than pasting opaque constants.
  * Assert the metric against the reference / threshold *truth* declares
    (e.g. ``assert ssim >= 0.95``, ``assert speedup >= 1.3``).
  * Check the agent's output file at the exact path *truth* declares.
  * Use clear, descriptive assertion messages that include the measured
    metric value AND the threshold (so the agent can see how close it got).

Allowed third-party imports (already pre-installed in the runtime SIF):
{allow_list}

Do NOT pip-install anything inside the pytest. The libs above are
available at solve time.""",
    "adversarial_corpus": """\
This task uses an ADVERSARIAL-CORPUS verifier. The agent's solution is
graded against TWO corpora that ship with the task:
  * an "evil" corpus that the solution MUST reject / sanitise / flag, and
  * a "clean" corpus that the solution MUST preserve / accept.
Pass requires both directions.

Your pytest MUST:
  * Iterate over every file in the evil corpus path declared by *truth*
    and assert the agent's solution rejects/transforms each one according
    to the criterion.
  * Iterate over every file in the clean corpus path declared by *truth*
    and assert the agent's solution leaves it unchanged / accepted.
  * Surface a clear summary on failure: "X of Y evil bypassed",
    "Z of W clean modified", listing offending file basenames.

Allowed third-party imports (already pre-installed in the runtime SIF):
{allow_list}""",
    "fuzz_equivalence": """\
This task uses a FUZZ-EQUIVALENCE verifier. The agent's program must
behave bit-exactly identically to a reference oracle program (often a
stripped binary shipped under /app/) on N random inputs.

Your pytest MUST:
  * Locate the oracle program at the exact path *truth* declares.
  * Generate N random inputs from the distribution *truth* describes
    (length / character set / shape). Use ``random`` with a fixed seed.
  * Run BOTH the oracle and the agent's program on each input and assert
    their outputs match exactly. On mismatch, surface the input + the two
    outputs in the failure message.

Allowed third-party imports (already pre-installed in the runtime SIF):
{allow_list}""",
    "multi_protocol": """\
This task uses a MULTI-PROTOCOL verifier. The agent brings up one or more
network services; the verifier issues real protocol-level requests and
checks the responses.

Your pytest MUST:
  * Connect to each service at the host:port *truth* declares.
  * Drive the request/response patterns *truth* describes (status codes /
    response bodies / RPC return values).
  * Surface clear failure messages with the actual response when an
    assertion fails.

Allowed third-party imports (already pre-installed in the runtime SIF):
{allow_list}""",
}


def _build_v2_system_msg(verifier_kind: str) -> str:
    """Return the system prompt for a v2 verifier_kind, or the legacy prompt
    when ``verifier_kind`` is unknown / ``"exact_text"``.
    """
    if verifier_kind not in _VERIFIER_KIND_PROMPT_FRAGMENTS:
        return SYSTEM_MSG
    fragment = _VERIFIER_KIND_PROMPT_FRAGMENTS[verifier_kind].format(
        allow_list=_format_allow_list(_VERIFIER_LIB_ALLOW_LIST.get(verifier_kind, ())),
    )
    # Replace the legacy "stdlib only" line with the per-template allow-list.
    lib_line = (
        "* Use the Python standard library, ``pytest``, and the libs allow-listed below."
    )
    base_msg = SYSTEM_MSG.replace(
        "* Use **only** the Python standard library and ``pytest`` (no third-party libs).",
        lib_line,
    )
    return f"{base_msg}\n\n# v2 Verifier kind: {verifier_kind}\n\n{fragment}\n"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def generate_test_templates_batch(
    items: list[tuple[str, str, str]] | list[tuple[str, str, str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 128,
) -> list[Optional[str]]:
    """Batched generation of final-state pytest templates.

    Accepted item shapes (mixed within one call is fine):
      * ``(task_description, truth, initial_test_py)`` — legacy 3-tuple.
        Uses the stdlib-only system prompt.
      * ``(task_description, truth, initial_test_py, verifier_kind)`` — v2
        4-tuple. Selects a per-verifier-kind system prompt and exposes the
        corresponding third-party allow-list.

    Returns an aligned list with ``None`` on failure (parse / compile error).
    """

    messages: list[list[dict[str, str]]] = []
    for item in items:
        if len(item) == 4:
            task_description, truth, initial_test_py, verifier_kind = item
        else:
            task_description, truth, initial_test_py = item
            verifier_kind = "exact_text"
        prompt = USER_TEMPLATE.format(
            task_description=task_description, truth=truth,
            initial_test_py=initial_test_py,
        )
        system_msg = _build_v2_system_msg(verifier_kind)
        messages.append([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ])

    responses = chat_completion_batch(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: list[Optional[str]] = []
    for resp in responses:
        if resp is None:
            results.append(None)
            continue
        try:
            content = textwrap.dedent(resp.choices[0].message.content)
            parsed = parse_python_code(content)
            if check_python_code(parsed):
                results.append(parsed)
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


