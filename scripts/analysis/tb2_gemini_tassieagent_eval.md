# Terminal-Bench 2.0 — `gemini-3-flash-preview` · VanilluxAgent (primary)

> **Note on filename.** This doc was originally TassieAgent-focused; we have since switched our default harness to **VanilluxAgent** (an in-house wrapper around upstream `swe-agent`). All sections below now treat Vanillux as the *primary* harness and use TassieAgent as a *baseline / ablation* only. The filename is left as-is to preserve git/PR history; rename when convenient.

**Primary run**: `jobs/tb2_gemini_vanillux_calls100` · 89 trials · pass@1 = **51.69 %** (46/89)  
**Harness**: VanilluxAgent (= upstream `swe-agent` v1.1.0, ATIF-v1.5 trajectory schema) · `calls=100` LLM-call budget · Daytona sandbox · 1 attempt/task  
**Model**: `gemini/gemini-3-flash-preview` via litellm  
**Auto-extracted artefacts**:

- Vanillux@100 (primary): [`out/tb2_gemini_vanillux_calls100/`](out/tb2_gemini_vanillux_calls100/) — `summary.json`, `per_trial.jsonl`, `failures.md`
- Vanillux@50 (budget ablation): [`out/tb2_gemini_vanillux_calls50/`](out/tb2_gemini_vanillux_calls50/)
- TassieAgent@50 (harness ablation): [`out/tb2_gemini_tassieagent_50turns/`](out/tb2_gemini_tassieagent_50turns/)

**Re-runner** (one-line, harness-agnostic):

```bash
uv run python scripts/analysis/analyze_tb2_eval.py \
    --job-dir jobs/<job_name> \
    --harbor-cache /gpfs/scrubbed/osey/harbor_cache \
    --label "<harness> + <model> [+ <budget>]" \
    --out scripts/analysis/out/<key>
```

> **TL;DR**. With Vanillux@100, gemini-3-flash-preview clears 51.7 % of TB 2.0. The remaining failures cluster into a small **harness-invariant** wrong-answer set (~15 tasks, the same tasks fail across all three of our (harness, budget) configurations) — these are the **capability-bound** failures. Our skill-tax 10 k corpus has the right *diversity* surface but not the same *difficulty depth*, because three structural choices in the existing generator pipeline (stdlib-only verifiers, LLM-imagined text fixtures, and author/solver model-symmetry) each cap how hard a generated task can be in practice. §6 spells out the gap; §7 proposes how to extend the existing pipeline along three new orthogonal axes (Verifier · Fixture · Calibration) without forking it.

---

## 1. Headline numbers

### 1.1 Primary run (Vanillux@100)

| metric                                  | value                       |
|----------------------------------------|-----------------------------|
| pass@1                                  | 0.5169 (46/89)              |
| `AgentTimeoutError` (per-task wall-clock cap hit) | 8 trials      |
| Other infra exceptions (`AgentSetupTimeoutError` × 2, `VerifierTimeoutError` × 1) | 3 trials |
| Submitted but failed verifier           | 19                          |
| Hit `calls=100` budget without submitting | 1 trial                  |
| Submitted (any outcome)                 | 63 / 89                     |
| Mean wall-clock / trial — pass          | 440 s (median 378 s)        |
| Mean wall-clock / trial — fail          | 557 s (median 420 s)        |

(Per-step token counts and LLM latency are not exposed in the SWE-agent ATIF dump, so they are zero in `summary.json` for Vanillux runs.)

### 1.2 Mean turns per outcome (Vanillux@100)

| group  | n  | mean | median | p25 | p75 | min | max |
|--------|---:|-----:|-------:|----:|----:|----:|----:|
| all    | 89 | 58.9 | 56     | 39  | 87  | 0   | 100 |
| pass   | 46 | 68.0 | 63.5   | 52  | 96  | 34  | 100 |
| fail   | 43 | 49.1 | 41     | 21  | 77  | 0   | 100 |

Successful Vanillux trials use *more* turns than failed ones (median 63 vs 41) — the opposite of TassieAgent. This is consistent with SWE-agent's deliberate inspect→edit→test loop: passes invest in exploration, fails are where the model bailed early or got stuck in a loop.

### 1.3 Three configurations side-by-side

Same model, same dataset (TB 2.0, 89 tasks), same Daytona sandbox. Only harness/budget changes.

| run                                              | pass@1     | n_pass | timeouts | hit_max  | submitted |
|--------------------------------------------------|-----------:|-------:|---------:|---------:|----------:|
| **VanilluxAgent · `calls=100`** (primary)        | **0.517**  | 46     | 8        | 1        | 63        |
| VanilluxAgent · `calls=50`                       | 0.438      | 39     | 3        | **45**   | 47        |
| TassieAgent · `max_steps=50` (TassieAgent baseline)        | 0.360      | 32     | 14       | 9        | 48        |

**+15.7 pp pass-rate lift from harness + budget on the same model.** Cleanly ablates:

- TassieAgent → Vanillux at the same nominal step budget: **+7.9 pp**. Pure harness improvement (mostly attributable to the `str_replace_editor` edit tool — see §3).
- Vanillux 50 calls → 100 calls: **+7.9 pp**. Pure budget improvement.

---

## 2. Pass rate by TB 2.0 task metadata

Categories and difficulty come from each task's `task.toml.metadata`.

### 2.1 By difficulty — *hard tasks plateau across budgets*

| difficulty | n  | TassieAgent@50 | Vanillux@50 | **Vanillux@100** |
|-----------|---:|---------------:|------------:|----------------:|
| easy      |  4 | 0.750 (3)      | 0.500 (2)   | **1.000 (4)**   |
| medium    | 55 | 0.400 (22)     | 0.491 (27)  | **0.582 (32)**  |
| hard      | 30 | 0.233 (7)      | 0.333 (10)  | **0.333 (10)**  |

Doubling Vanillux's call budget (50 → 100) added **+5 wins on medium** and **+2 wins on easy** but **zero new wins on hard**. The hard tail is *capability-bound*, not *iteration-bound*. This is the single most important data point in this doc for the data-gen story.

### 2.2 By category (n ≥ 3 only, Vanillux@100)

| category               | n  | n_pass | pass_rate |
|------------------------|---:|-------:|----------:|
| security               |  8 | 5      | 0.625     |
| data-science           |  8 | 5      | 0.625     |
| scientific-computing   |  8 | 5      | 0.625     |
| system-administration  |  9 | 5      | 0.556     |
| data-processing        |  4 | 3      | 0.750     |
| debugging              |  5 | 3      | 0.600     |
| file-operations        |  5 | 2      | 0.400     |
| machine-learning       |  3 | 0      | 0.000     |
| mathematics            |  4 | 2      | 0.500     |
| model-training         |  4 | 2      | 0.500     |
| software-engineering   | 26 | 12     | 0.462     |

Software-engineering (largest split, 29 % of suite) and machine-learning (3/3 fail) are the weak spots — both are dominated by tasks needing real builds, real packages, or quantitative correctness that our existing verifier-stdlib constraint (§6) cannot grade.

---

## 3. Tool usage — what the harness change really bought us

Vanillux command verbs are extracted from `tool_calls[].arguments.raw_action` in the ATIF trajectory. (For Tassie they came from `agent/timing.json`'s `cmd` field.)

| verb                   | TassieAgent@50 | Vanillux@50 | Vanillux@100 | comment |
|------------------------|--------------:|------------:|-------------:|---------|
| `str_replace_editor`   | **0** (n/a)   | 707         | 832          | proper edit tool — Vanillux only |
| `cat-write` (heredoc)  | **577**       | 0 (n/a)     | 0 (n/a)      | Tassie's only way to edit — wasteful |
| `submit`               | 0 (n/a)       | 239         | 947          | Vanillux's first-class submit action |
| `ls`                   | 145           | 491         | 624          | exploration |
| `python3` / `python`   | 89            | 486         | 565          | running scripts/tests |
| `cat`                  | 107           | 359         | 407          | reading files |
| `grep`                 | 79            | 187         | 203          | searching |
| `pip`                  | 25            | 64          | 64           | installing dependencies |

The **single most likely cause of the +7.9 pp Tassie → Vanillux lift at fixed budget** is the `str_replace_editor` tool. TassieAgent had no edit primitive — every code change was a heredoc rewrite of the whole file (`cat > foo.py << 'EOF' … EOF`), which on long tasks burns thousands of tokens re-emitting unchanged regions and is regression-prone. SWE-agent's edit tool does surgical region-replace, which is dramatically more token-efficient and more reliable.

A second differentiator: Vanillux emits an explicit `submit` action that the model treats as a first-class tool (947 calls in `calls=100`). TassieAgent's `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` convention is brittle — the model forgets to issue it on 14/57 failures (`no_submit_early_stop`). Almost none of Vanillux@100's failures are submission-discipline failures.

These observations are *about the harness*, not about the data-gen pipeline. They matter to the data-gen story only because §10/§7 argue that **the calibration filter must use the deployment harness**: if we filter against TassieAgent, we'll mistake harness-bound difficulty for capability-bound difficulty.

---

## 4. Failure mode taxonomy (Vanillux@100 primary)

Every failure is bucketed by a deterministic classifier in `analyze_tb2_eval.py`:

```
if exception == "AgentTimeoutError"           → agent_timeout              (8)
elif other_exception                          → other_error:<type>          (3)
elif not submitted and hit_max_steps          → no_submit_max_steps         (1)
elif not submitted                            → no_submit_early_stop        (14)
elif submitted and tests-failed:
    msg matches /FileNotFoundError/           → submitted_missing_artifact   (1)
    msg matches /header|format|schema/        → submitted_wrong_format       (0)
    msg matches /expected.*got|actual/        → submitted_wrong_value        (1)
    else                                      → submitted_verifier_failed_other (15)
```

| failure mode                                | TassieAgent@50 | Vanillux@50 | **Vanillux@100** |
|---------------------------------------------|---------------:|------------:|----------------:|
| `pass`                                      | 32             | 39          | **46**          |
| `submitted_verifier_failed_other`           | **17**         | **15**      | **15**          |
| `no_submit_early_stop`                      | 14             | 9           | 14              |
| `agent_timeout`                             | 13             | 3           | 8               |
| `no_submit_max_steps`                       | 9              | **22**      | 1               |
| `submitted_wrong_format`                    | 2              | 0           | 0               |
| `submitted_wrong_value`                     | 2              | 0           | 1               |
| `submitted_missing_artifact`                | 0              | 1           | 1               |
| `other_error:AgentSetupTimeoutError`        | 0              | 0           | 2               |
| `other_error:VerifierTimeoutError`          | 0              | 0           | 1               |

### Two patterns worth naming

1. **`submitted_verifier_failed_other` is invariant at 15–17 across all three runs.** Same model, different harnesses, different budgets — *the same ~15 tasks* fail in this bucket every time. This is **the model-capability bound**: tasks where the agent confidently submits a wrong answer no matter how it's run. **This set is the most useful target for the harder-task generator** — see §7.5.
2. **`no_submit_max_steps` is harness/budget-bound, not capability-bound.** It collapsed from 22 → 1 when Vanillux's call budget doubled. Conversely Tassie's 9 same-pattern failures are mostly *deep-search multi-component* tasks (`mailman`, `make-doom-for-mips`, `polyglot-rust-c`) — the agent kept trying productively and ran out of room.

### The harness-invariant capability set (15 tasks)

These are the tasks that fail with `submitted_verifier_failed_other` on Vanillux@100. Each is a "submitted-but-wrong" failure — model thought it was done, verifier disagreed:

| task                              | failure character                                                  |
|-----------------------------------|--------------------------------------------------------------------|
| `chess-best-move`                 | image OCR — read board state from PNG, picked wrong move           |
| `mteb-retrieve`                   | run a specific embedding model, rank docs by cosine — wrong line   |
| `extract-elf`                     | parse ELF binary's loaded segments — off by some bytes             |
| `model-extraction-relu-logits`    | extract NN weights via blackbox queries — failed 28/30 rows        |
| `path-tracing-reverse`            | RE a binary, reproduce as C with image_similarity ≥ 0.995 — got 0.887 |
| `dna-insert`                      | synthetic biology primer design — failed BsaI clamp                |
| `winning-avg-corewars`            | CoreWars warrior — got 32 % win rate vs 75 % required              |
| `make-mips-interpreter`           | MIPS interpreter in JS — TimeoutError on test_vm_execution         |
| `pytorch-model-cli`               | MNIST CLI tool — predicted 0, expected 2 (model loaded wrong)      |
| `qemu-startup`                    | qemu+telnet up — got `Password:` prompt instead of shell           |
| `sam-cell-seg`                    | histopath cell seg w/ Mobile-SAM — subprocess error                |
| `sanitize-git-repo`               | redact API keys, preserve everything else — false positives        |
| `torch-pipeline-parallelism`      | pipeline-parallel LLaMA training — process raised exception        |
| `video-processing`                | hurdle-jump frame detection — wrong frame                          |
| `filter-js-from-html`             | XSS sanitizer on adversarial corpus — bypasses + clean-doc damage  |

**Common pattern**: each ships a real artefact (image / binary / package / video / corpus) AND grades against a quantitative or adversarial threshold. The two structural differences from our generated tasks (§6) intersect exactly here.

---

## 5. What makes TB 2.0 hard — six dimensions of difficulty

Cross-cutting observations distilled from §4 and from reading every failed trajectory in [`out/tb2_gemini_vanillux_calls100/failures.md`](out/tb2_gemini_vanillux_calls100/failures.md).

### 5.1 Real-software anchoring (specific versions)

TB 2.0 names specific artefacts: *POV-Ray 2.2*, *BVLC Caffe 1.0.0*, *PyStan 3.10.0*, *fasttext on Yelp*, *MTEB 1.36.8*, *MobileSAM*, *QEMU 5.2.0*, *Windows 3.11 for Workgroups*, *OCaml compiler bootstrap*, *CompCert 3.13.1*. The agent has to navigate real upstream ecosystems — install procedures, ABI quirks, vendored sources.

### 5.2 Multimodal / non-text inputs

TB 2.0 includes images (`chess-best-move`, `code-from-image`, `path-tracing`, `pytorch-model-cli`, `sam-cell-seg`), videos (`extract-moves-from-video`, `video-processing`), and binary blobs (`extract-elf`, `path-tracing-reverse`, `mystery`, `gcode-to-text`).

### 5.3 Quantitative correctness with tight tolerances

TB 2.0 verifiers use thresholds like `image_similarity ≥ 0.99`, `speedup faster than reference numpy`, `model_size < 150 MB AND accuracy ≥ 0.62`, `win_rate ≥ 75 %`, `atol=1e-5`. These are *gradient-rich* signals — they tell the agent how close it got, and they make exact-match string comparison impossible to game.

### 5.4 Adversarial / hostile inputs

`filter-js-from-html` ships a curated XSS corpus + clean-HTML corpus; both must be satisfied. `sanitize-git-repo` checks both "right secrets replaced" *and* "no other files changed". `password-recovery` requires entropy-aware reasoning.

### 5.5 Multi-component / multi-service orchestration

`mailman` (postfix + mailman3 + mail flow), `install-windows-3.11` (qemu + VNC + nginx), `qemu-startup` (qemu + telnet), `kv-store-grpc` (gRPC + replication), `configure-git-webserver` (git protocol + nginx + auth).

### 5.6 Reverse engineering / forensics

`extract-elf`, `path-tracing-reverse`, `feal-linear-cryptanalysis`, `chess-best-move` (read board from image), `crack-7z-hash`, `password-recovery`, `db-wal-recovery`, `git-leak-recovery`. Our skill-tax taxonomy *names* these (`Forensics` sub-skill of `debugging`) but rarely instantiates them with a real binary blob.

### 5.7 Knowledge-cutoff / external lookup (anti-pattern, do not reproduce)

`mteb-leaderboard` ("best model as of August 2025"), `build-pov-ray` (find historical source URL), `caffe-cifar-10` (find BVLC source). These are *unsolvable* without internet access — the model spends its budget guessing URLs. Bad RL signal. We should *not* generate tasks with this pattern.

---

## 6. Gap analysis — *diversity* axes are saturated, *depth* axes are missing

This section is the central narrative of the doc. The argument has three parts.

### 6.1 The current generator's story

`rl_data/generator/task_template_gen.py` samples a tuple along seven axes per task — domain (×9), skill type (×~5), primitive skills (3–5 of ~30/domain), task complexity (×3), command complexity (×3), scenario/persona (×~10/domain), language (Python/C/Bash/C++/Rust/Go/multi/any). Optionally a real-software anchor is injected (35 % of tasks). The downstream stages — `apptainer_def_gen.py` (env), `initial_state_test_gen.py` (pre-check), `completion_test_gen.py` (post-check) — each take an LLM call to materialise the env, the initial-state pytest, and the final-state pytest.

Story so far: *Sample diverse axes → an LLM invents a novel task at every intersection → a small per-task delta over a pre-built domain base SIF runs it → a pytest verifier checks the result.*

This is great for **surface diversity**: the domain × skill × persona × language cross-product covers the same instruction space as TB 2.0 (the categories nearly map 1:1). And the skill-tax 10 k corpus reflects this — pass@1 from gemini-3-flash-preview on it is *much* higher than on TB 2.0 (the existing `quality_pass1_*.png` plots).

### 6.2 Why high diversity does not produce TB-2 difficulty

Three structural choices in the existing pipeline cap how hard a generated task can be in practice. They are not bugs — each has good reasons in the original design — but they each correspond to a missing dimension of TB 2.0 difficulty (§5).

#### 6.2.1 Verifier is stdlib + pytest only

`completion_test_gen.py` instructs the LLM:

> ```
> Use **only** the Python standard library and ``pytest`` (no third-party libs).
> ```

This is what makes our verifiers safe to run on any base image. It is also what makes them **incapable of grading any of the TB 2.0 quantitative tolerance tasks** (§5.3). Without numpy/scipy, you cannot compute SSIM. Without librosa, you cannot compare audio. Without torch, you cannot evaluate accuracy on a held-out test set. Without a domain library, you cannot reason about *anything except text equality*. So even when the LLM dreams up "agent must produce a CSV whose column has `<ε>` distance from this reference", the verifier collapses to `assert open('output.csv').read() == EXPECTED` — and the LLM, generating both the task and the expected text, lands on something that's either trivially solvable (because the expected output is something the LLM can also predict) or gibberish.

This is the single biggest lever. **Verifier sophistication is what creates TB 2.0's hardest failure mode (`submitted_verifier_failed_other`)** — see the harness-invariant 15-task list in §4.

#### 6.2.2 Fixtures are LLM-imagined text

All inputs to the agent are descriptions that the *generator* model wrote at sample time. The agent receives a paragraph that says "the file `/app/data.txt` contains a CSV of …", not the actual CSV. The LLM in `apptainer_def_gen.py`'s `%post` step generates *deterministic* setup scripts that fabricate the input data from text the same LLM authored — which means the inputs the agent encounters at solve time are sampled from the same model's prior over inputs. There is no concrete artefact (image, audio, stripped binary, real package source, real video) the model didn't itself imagine.

This explains the absence of all of §5.2 (multimodal), §5.6 (RE/forensics), and most of §5.1 (real-software anchoring) from our corpus. Even when our taxonomy *names* "Reverse engineering and disassembly" as a primitive skill, the resulting generated task has no real binary to reverse — it has a text description of one.

#### 6.2.3 Author and solver are the same model

Every stage of the existing pipeline runs on `DEFAULT_MODEL = gemini/gemini-3.1-pro-preview`, and we evaluate solutions on `gemini/gemini-3-flash-preview` (a slightly weaker sibling). The author/solver gap is small. When the generator writes a task, it cannot articulate a problem whose *solution requires capability the generator itself lacks* — and tasks the generator articulates clearly are typically tasks the generator can solve. So the difficulty distribution of our 10 k corpus is, by construction, *bounded above by what the generator can do*.

TB 2.0 was authored by **humans with hard, specific capability gaps in mind**. Author ≠ solver; author can articulate problems they can't solve, and they deliberately do.

We can mimic this asymmetry empirically without changing the author model: by **post-hoc filtering** generated tasks against a held-out solver (the same model + harness we will evaluate on later), keeping only those whose pass@k lands in a target band. This is Track C (§7.3).

### 6.3 Gap table (TB 2.0 vs current generator output)

| Dimension                          | TB 2.0           | skill-tax 10 k pipeline (current)        | Where this lives in the pipeline          | Gap |
|------------------------------------|------------------|------------------------------------------|-------------------------------------------|-----|
| Specific software/version anchoring | Most tasks       | 35 % anchor rate; anchors abstract       | `task_template_gen.py:REAL_SOFTWARE_ANCHORS` | Large |
| Multimodal inputs                  | ~10 tasks (11 %) | None                                     | `apptainer_def_gen.py` `%post` is text-only | **Total** |
| Quantitative tolerance verifiers   | ~30 tasks (34 %) | None — stdlib pytest only                | `completion_test_gen.py:SYSTEM_MSG`        | **Total** |
| Adversarial test corpora           | ~5 tasks (6 %)   | None                                     | `completion_test_gen.py` (no fixture mech) | **Total** |
| Multi-service orchestration        | ~8 tasks (9 %)   | None                                     | `apptainer_def_gen.py` is single-image     | **Total** |
| RE / binary forensics              | ~7 tasks (8 %)   | Named in taxonomy, rarely instantiated   | No fixture-deposit mechanism               | Medium |
| Pre-vendored data fixtures         | All tasks        | Generated text/JSON only                 | `apptainer_def_gen.py` no asset injection  | Large |
| Per-task wall-clock budget         | 1–200 min        | LLM-imagined, often quick                 | task metadata not exposed                  | Large |
| Difficulty self-labels             | easy/medium/hard | None                                     | not measured                                | Medium |
| Author/solver asymmetry            | Human → LLM      | Same family (3.1-pro → 3-flash)          | global config                              | Medium (closes via §7.3) |

Three structural rows are **total** gaps — not just "underused", but mechanically impossible in the current pipeline architecture. They map to the three depth axes proposed in §7.

---

## 7. Proposal — three orthogonal *depth axes*, additive over the existing pipeline

The existing pipeline samples diversity axes (domain × skill × scenario × language × …). We propose three *new orthogonal axes* that get sampled per task alongside the existing ones, each unlocking a class of TB 2.0 difficulty:

| Axis                | Values (sketch)                                                                 | Unlocks (§5)        |
|---------------------|---------------------------------------------------------------------------------|---------------------|
| **A. Verifier kind**  | `exact_text` (legacy default) · `metric_threshold` · `adversarial_corpus` · `fuzz_equivalence` · `multi_protocol` | §5.3, §5.4         |
| **B. Fixture kind**   | `text_only` (legacy default) · `image` · `audio` · `video` · `stripped_binary` · `vendored_package` · `multi_service_compose` | §5.1, §5.2, §5.5, §5.6 |
| **C. Calibration**    | post-hoc; not a generation axis but a *filter stage*: keep only pass@k ∈ [0.125, 0.875] on the deployment harness | §6.2.3            |

Critically, axes A and B are **multiplicative over the existing axes**, not replacements. The same `(security, "Algorithmic", forensics-analyst, hard, C, …)` tuple can now be instantiated with `verifier_kind=fuzz_equivalence` and `fixture_kind=stripped_binary` to produce a binary-RE task that the current pipeline literally cannot author. A generated task's metadata becomes:

```json
{
  "domain": "...", "skill_type": "...", "primitive_skills": [...],
  "task_complexity": "...", "command_complexity": "...",
  "scenario": "...", "language": "...", "anchor": "...",
  "verifier_kind": "metric_threshold",
  "fixture_kind": "image",
  "difficulty_calibration": {"solver": "vanillux+gemini-3-flash", "pass_at_8": 0.375}
}
```

### 7.0 Pipeline integration map (no fork required)

| Stage                                  | Existing role                                | Extension for new axes                                                                                              |
|----------------------------------------|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `task_template_gen.py`                 | sample axes; LLM emits `<task>` + `<truth>`  | sample `verifier_kind` + `fixture_kind` too; emit them in metadata; pass them into the system prompt so the description respects them |
| **NEW** `fixture_gen.py`               | —                                            | given `(fixture_kind, task_description, truth)` → produce the actual artefact (image/audio/binary/etc.) on the host  |
| `apptainer_def_gen.py`                 | LLM emits `.def` with text setup            | when `fixture_kind ≠ text_only`, **deposit fixture files** into the task dir at build time; `%files` section copies them into `/app` |
| `initial_state_test_gen.py`            | stdlib pytest checks env was set up         | unchanged — fixture presence is checked the same way                                                                |
| `completion_test_gen.py`               | stdlib pytest checks final state            | when `verifier_kind ≠ exact_text`, switch to a verifier-template-specific system prompt and **allow a curated allow-list of third-party libs** (numpy/scipy/Pillow/torch/…) in the verifier |
| `sample_solutions.py`                  | runs N solutions, computes pass@k            | unchanged — already supports k=8                                                                                    |
| **NEW** `calibrate_difficulty.py`      | —                                            | post-stage: read `solutions/*_summary.json`, compute pass@k, emit `keep`/`drop` decision per task; tag retained tasks with measured difficulty |

The whole proposal is therefore: **two new files, two extended call-sites, no file-rewrites in existing stages.**

### 7.1 Axis A — Verifier kind (§5.3, §5.4)

**Why first.** The `submitted_verifier_failed_other` failure mode — the harness-invariant 15-task set in §4 — is *exactly* the failure mode that quantitative and adversarial verifiers create at generation time. This axis attacks the largest currently-failing TB 2.0 bucket directly.

#### Verifier templates

- **`metric_threshold`**: agent's output is graded by a numerical metric against a reference. Verifier ships the metric harness (e.g. `metric.py` with cosine, SSIM, BLEU, accuracy-on-test-set). Generator runs the reference solution offline once, records the reference metric, then sets the agent's threshold to `(reference_metric − epsilon)` so the bar is achievable. Tasks where the reference fails to converge are dropped at generation time.
  - *TB 2.0 analogues*: `largest-eigenval`, `tune-mjcf`, `train-fasttext`, `path-tracing-reverse`, `model-extraction-relu-logits`, `pytorch-model-cli`, `winning-avg-corewars`.

- **`adversarial_corpus`**: verifier carries two corpora — `evil/` (must reject / sanitise / detect) and `clean/` (must not modify / false-positive). Pass requires both directions. Corpora can be seeded from public lists (OWASP XSS, SQL-injection wordlists, leaked-API-key formats) and amplified by an LLM mutator.
  - *TB 2.0 analogues*: `filter-js-from-html`, `sanitize-git-repo`.

- **`fuzz_equivalence`**: a reference binary `oracle` is shipped; agent must produce another implementation `student`; verifier fuzzes `oracle` and `student` with random inputs and asserts bit-exact agreement on N samples.
  - *TB 2.0 analogues*: `extract-elf`, `path-tracing-reverse`.

- **`multi_protocol`** (smaller, optional): verifier issues real protocol-level requests (HTTP, TCP, gRPC, mail) and checks responses. Used together with axis B `multi_service_compose`.
  - *TB 2.0 analogues*: `mailman`, `kv-store-grpc`, `qemu-startup`.

#### Required pipeline change in `completion_test_gen.py`

Today's `SYSTEM_MSG` hard-codes "stdlib + pytest only" (line 30). For the new templates we'd:

1. Make the system prompt template-conditional (one variant per `verifier_kind`).
2. Allow a curated, base-SIF-pre-installed allow-list per template:
   - `metric_threshold` → numpy, scipy, scikit-learn, Pillow, torch (cpu)
   - `adversarial_corpus` → no extras needed (text matching)
   - `fuzz_equivalence` → no extras needed
   - `multi_protocol` → requests, grpcio, smtplib (already stdlib)
3. Pre-build a single new base SIF `base_metric_verifier.sif` that ships these libs; tasks with these verifier kinds resolve to that base in `apptainer_def_gen._resolve_base()`.

The "stdlib only" constraint stays the default; we just stop applying it when the task's metadata explicitly opts into a richer verifier stack.

### 7.2 Axis B — Fixture kind (§5.1, §5.2, §5.5, §5.6)

**Why second.** This is *the* dimension our pipeline currently has zero coverage on. The TB 2.0 failure analysis (§4 / §5.2) shows that real artefacts are central to half the harness-invariant failures.

#### Fixture kinds

- **`image`** — generator produces a PNG procedurally (PIL/matplotlib/LaTeX → PNG). The *generation parameters* (the source code rendered, the table values, the chess position) are the hidden ground truth. The agent must use OCR (tesseract) or vision to recover them.
  - *TB 2.0 analogues*: `chess-best-move`, `code-from-image`, `path-tracing` reference image.

- **`audio`** — generator synthesises speech via espeak/edge-tts. Hidden ground truth is the spoken text. Agent must use Whisper or similar.

- **`video`** — ffmpeg-stitches a sequence of frames with timestamped events (counter increments, ball positions). Agent must detect events.

- **`stripped_binary`** — generator picks a small algorithm (CRC variant / obfuscated XOR / state machine), emits C source, compiles with `-s`, optionally runs UPX, then **discards the source**. The task ships only the binary + a few sample (input, output) pairs. Verifier `fuzz_equivalence` (axis A) randomly fuzzes oracle vs student.
  - *TB 2.0 analogues*: `mystery`, `extract-elf`, `path-tracing-reverse`.

- **`vendored_package`** — generator picks one of ~50 curated real packages (`pyknotid 0.5.4`, `pmars 0.9.4`, `MobileSAM Y`, …) that have been **pre-vendored** into a base SIF (no internet at solve time). It then samples a *perturbation* (broken Makefile flag / wrong env var / missing patch / wrong numpy ABI) and writes a description that asks the agent to make a known-good code path of the package work again.
  - *TB 2.0 analogues*: `build-pmars` (passed!), `cobol-modernization` (passed!), `modernize-scientific-stack` (passed!) — confirms agents can succeed when the package is pre-vendored. Avoids the §5.7 anti-pattern of URL-hunting.

- **`multi_service_compose`** — generator picks a `docker-compose.yaml` template from a small curated library (postgres+adminer, nginx+flask, redis+producer/consumer, kafka+consumer, smtp+mta+mailman) and a perturbation. The task asks the agent to make the broken service flow again.

#### Required pipeline changes

1. New file `rl_data/generator/fixture_gen.py`. Pure host-side code (PIL/ffmpeg/gcc/UPX/curl) that materialises the artefact bytes and writes them to a per-task `fixtures/` dir. No LLM call; deterministic given a `(fixture_kind, seed, task_description)`.
2. `apptainer_def_gen.py` learns to emit a `%files` section that copies the fixture files into the container `/app/`. Today the def has only `Bootstrap`/`From`/`%post`; adding `%files <host_path> <container_path>` lines is a one-function change.
3. `task_template_gen.py` adds `fixture_kind` to the sampled tuple and passes it to the description generator's system prompt so the task description references the artefact correctly ("the binary at `/app/mystery`", "the image at `/app/board.png`").

### 7.3 Axis C — Calibration filter (§6.2.3)

**Why post-hoc, not a generation axis.** Difficulty is an emergent property of (task, model, harness), not a controllable input. The cleanest move is to over-generate and filter empirically.

#### Algorithm

1. Generate K tasks via the existing pipeline + axes A/B.
2. Run `sample_solutions.py` with `n_solutions=8` using **the deployment harness** (currently VanilluxAgent + gemini-3-flash-preview). This is the same code path the existing `run_generate_solutions_*.sh` scripts invoke.
3. New stage `rl_data/generator/calibrate_difficulty.py` reads `<task>/solutions/<run>_summary.json`, computes pass@k on each task, writes a `<task>/difficulty.json` with `{pass_at_1, pass_at_8, decision}`.
4. Decision rule: keep tasks with `pass@8 ∈ [0.125, 0.875]`. Drop the trivially-solvable (pass@8=1) and the broken-or-impossible (pass@8=0). Tag retained tasks with a difficulty bin (`easy: pass@8 ≥ 0.625`, `medium: 0.25 ≤ pass@8 < 0.625`, `hard: 0.125 ≤ pass@8 < 0.25`).
5. Optional curriculum sampler weights downstream training mixtures by these bins.

#### Two methodological points worth flagging

- **The calibration solver must be the deployment harness.** §3 / §10 showed that switching from TassieAgent to Vanillux at fixed budget shifts pass@1 by +7.9 pp. If we calibrate against a weaker harness than we deploy on, we'll keep tasks that are merely *harness-hard* (e.g. tasks that fail because TassieAgent has no edit tool) rather than *capability-hard*. **Recommendation: run calibration against VanilluxAgent at `calls=100`.**
- **The author model should be different from (and stronger than) the calibration solver.** Currently both are gemini-3.x; this works but doesn't maximise the asymmetry. Cheap improvement: keep `DEFAULT_MODEL = gemini/gemini-3.1-pro-preview` for authoring and use `gemini-3-flash-preview` for calibration (the same model we evaluate on). The pro-preview model is strictly stronger than flash and thus can articulate harder tasks.

### 7.4 Sequencing recommendation

Working back from the technical-report story, the order should be:

1. **Track C first** (~1 week). Implement `calibrate_difficulty.py` and apply it to the *existing* skill-tax 10 k corpus. This gives an immediate empirical reading: how many of our current tasks fall in the `[0.125, 0.875]` keep band? Hypothesis: most of the 10 k will be in `pass@8 = 1.0` (too easy), which directly justifies tracks A and B with a number.
2. **Track A (verifier templates)** (~2 weeks). Most impact-per-effort. Requires the smallest extension to the existing pipeline (`completion_test_gen.py` + one new base SIF). Generate ~1 k tasks with `verifier_kind ∈ {metric_threshold, adversarial_corpus}` and re-run track C.
3. **Track B (fixture templates)** (~2–3 weeks). Higher upside but each fixture kind adds a new fixture format and verifier integration. Start with `stripped_binary` (composes naturally with track A's `fuzz_equivalence`) and `image` (largest gap in §5.2 coverage).
4. **Multi-service compose** (Track B's `multi_service_compose` value) is its own can of worms (Daytona doesn't natively support compose); defer until after the others.

Throughout, keep the existing skill-tax 10 k corpus unchanged as the easy/diversity baseline. New tracks produce a *complementary* harder-task corpus that lives alongside it.

### 7.5 The harness-invariant 15-task set as a target distribution

The 15 tasks listed in §4 are the cleanest empirical specification of "what a harder-task generator should produce more of". They cluster:

- 6 of 15 → axis A `metric_threshold` (image/perf/accuracy thresholds): `path-tracing-reverse`, `pytorch-model-cli`, `winning-avg-corewars`, `make-mips-interpreter`, `extract-elf`, `model-extraction-relu-logits`
- 4 of 15 → axis B `image`/`video` fixtures: `chess-best-move`, `path-tracing-reverse`, `video-processing`, `pytorch-model-cli`
- 2 of 15 → axis A `adversarial_corpus`: `filter-js-from-html`, `sanitize-git-repo`
- 2 of 15 → axis B `vendored_package`: `mteb-retrieve`, `sam-cell-seg`
- 1 of 15 → axis A `multi_protocol` + B `multi_service`: `qemu-startup`
- 2 of 15 → axis B `vendored_package` + dna/biology domain: `dna-insert`, `dna-assembly` (would need `biopython` in verifier allow-list)

(Several tasks count toward more than one axis. Total > 15.)

This breakdown gives an explicit target distribution for the harder-task corpus: roughly 40 % `metric_threshold`, 25 % multimodal (image/video), 15 % adversarial corpus, 15 % vendored_package, 5 % multi-protocol/service.

---

## 8. Concrete next-step recommendation

Single-track if forced to pick: **Track C (calibration filter) + a half-day reading of where the existing 10 k corpus falls in the pass@8 distribution.**

Reasoning:

- It's the cheapest ground-truth check on whether the gap analysis above is calibrated. If our current 10 k turns out to have ~50 % of tasks in the `[0.125, 0.875]` keep band, we have less of a difficulty problem than the gap argument suggests. If <10 %, the argument is settled and we know exactly how many net-new harder tasks we need.
- It's a prerequisite for Tracks A and B regardless — both depend on having the calibration loop wired up to filter their output.
- It produces a number that goes directly into the technical report: "X % of our 10 k corpus is in the empirical-difficulty keep band against (Vanillux + gemini-3-flash) at pass@8".

Two-track recommendation: **C + A**. After the calibration baseline lands, Track A (verifier templates) is the smallest pipeline change with the largest difficulty-ceiling lift, because it directly converts §6.2.1 (the stdlib-only verifier constraint) into a controllable axis.

---

## 9. Reproducibility

Run the eval (resumes if `jobs/<name>` exists):

```bash
bash scripts/run_tb2_gemini_vanillux.sh   # primary
bash scripts/run_tb2_gemini_tassie.sh     # ablation
```

Run the analyser (replace job dir / label / out for other runs):

```bash
uv run python scripts/analysis/analyze_tb2_eval.py \
    --job-dir jobs/tb2_gemini_vanillux_calls100 \
    --harbor-cache /gpfs/scrubbed/osey/harbor_cache \
    --label "VanilluxAgent + gemini-3-flash-preview, calls=100" \
    --out scripts/analysis/out/tb2_gemini_vanillux_calls100
```

Generated files (overwrite-safe, same script):

- `out/<key>/per_trial.jsonl` — one row per trial: tool histogram, verifier-test-level pass/fail, last-assistant-text, final assertion excerpts.
- `out/<key>/summary.json` — machine-readable aggregates.
- `out/<key>/failures.md` — auto-generated narrative with the assertion-text excerpt of every failed test for every failed trial.

The same script handles two harness conventions out of the box:

- **TassieAgent** — uses `agent/timing.json` (one row per step) plus an OpenAI-style `agent/trajectory.json`. Submission marker: literal `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` substring in the bash command.
- **VanilluxAgent** (= upstream `swe-agent`) — only writes `agent/trajectory.json` in **ATIF-v1.5** schema (top-level `{"steps": [...]}`, each agent step has `tool_calls[].arguments.raw_action`). No per-step timing/token counts. Submission marker: standalone `submit` token at end of bash chain (regex `(?:^|[\s|;&])submit\b\s*$`). Step-budget cap is parsed from the run-dir-name pattern `..._calls(\d+)`.

It re-runs against any harbor job dir produced by `run_tb*.sh` / `run_swebench*.sh`, so cross-(model, harness, budget) comparisons stay one-line.
