# `rl_data/` — terminal task data generation

A **simple, scalable, diverse, and difficulty-aware** pipeline for synthesising
terminal-agent tasks, solving them with LLM agents, and exporting the results
for SFT and RL training. Each task is sampled as an *independent product of
structured axes*, packaged as a self-contained Apptainer/Docker environment with
a programmatic verifier, then solved at pass@k, analysed, and uploaded to the
Hugging Face Hub.

This is the code behind the **Terminal Data Generation** section of the paper.
We use `gemini-3.1-pro-preview` as the generation model.

## Compositional generation

Instead of an expensive multi-stage validation pipeline, each task is a single
draw from a set of orthogonal axes — combinatorially many distinct task
signatures from a small sampler. All axes live in
[`generator/task_template_gen.py`](generator/task_template_gen.py)
(`random_user_msg`):

| Axis | Role | Values |
|------|------|--------|
| **domain** | coverage | 9 domains (security, software_engineering, data_science, …) |
| **skill type + primitive skill** | coverage | per-domain skill taxonomy (seeded from Pi et al. 2026) |
| **persona** | diversity | domain-tied personas (5–18 per domain) |
| **fixture kind** | diversity | `text_only` (default) + `image` / `audio` / `video` / `stripped_binary` / `vendored_package` / `multi_service_compose` |
| **task complexity** | difficulty | `short` → `moderate` → `complex` → `intricate` (≈30–60 commands) |
| **command complexity** | difficulty | `bash-only` → `bash+code` → `bash+code+services` |
| **verifier kind** | difficulty | `exact_text` (default) + `metric_threshold` / `adversarial_corpus` / `fuzz_equivalence` / `multi_protocol` |

Three design choices map directly to the paper:

- **Scalability via soft filtering.** We *skip* teacher-based correctness
  validation. The pipeline only guarantees each task is **executable** by
  building a container image for it and running its tests; task quality is left
  to RL's soft filtering (zero-pass-rate rollouts contribute no gradient and are
  dropped at train time). Base images are pre-configured per domain
  (`containers/base_<domain>.sif`) so tasks in a domain share a base, with
  task-specific deps added on top.
- **Diversity via independent sampling**, reinforced by **persona
  diversification** and **multi-modal fixtures** — a concrete artefact (PNG /
  audio / video / stripped binary / vendored package / compose stack) shipped
  inside the container. The policy stays a text-only model and inspects fixtures
  through terminal tooling (OCR, ASR, `ffmpeg`, …); see
  [`generator/fixture_gen.py`](generator/fixture_gen.py).
- **Difficulty via explicit calibration** — the two complexity axes plus
  **graded verifiers** (metric-threshold, adversarial-corpus, fuzz-equivalence,
  multi-protocol) give a continuous difficulty knob that avoids the bi-modal
  "trivial or unsolvable" task pool.

## Pipeline at a glance

```
generate_tasks  ->  generate_solutions  ->  analyze  ->  upload_to_hf
 (create tasks)      (solve at pass@k)      (stats)      (publish corpus)
```

Each stage operates on a *corpus directory* under `output/`, which holds one
`task_*/` folder per task:

```
task_000123_ab12cd34/
├── task.json              # prompt, truth, domain + sampled axes (verifier/fixture/complexity/…)
├── test_initial_state.py  # asserts the env starts in the expected state
├── test_final_state.py    # programmatic verifier (pass/fail signal)
├── container.def          # Apptainer definition for the task environment
├── setup.sh               # env setup baked into the image
├── fixtures/              # multi-modal artefacts shipped with the task (when sampled)
└── solutions/             # per-model solution summaries + pass@k results
```

`generate_tasks` runs four sub-steps per task — template → initial test → final
test → **container build + smoke test** — and keeps a task only if its image
builds and runs. This is the *executability* check described above, not a
quality filter.

## Layout

```
rl_data/
├── __init__.py            # litellm-backed LLM client (batched chat completions) + DEFAULT_MODEL
│
├── generate_tasks.py      # STAGE 1: sample axes -> template -> tests -> container build+smoke
├── generate_solutions.py  # STAGE 2: run agents against tasks, collect solutions + pass@k
├── analyze.py             # STAGE 3: composition/difficulty/balance tables + plots for a corpus
├── upload_to_hf.py        # STAGE 4: push a corpus (tasks + parquet) to the HF Hub
├── estimate_cost.py       # project API cost for a proposed generation/solve run
│
├── generator/             # building blocks called by generate_tasks/generate_solutions
│   ├── task_template_gen.py      # skill taxonomy, personas, and the compositional axis sampler
│   ├── initial_state_test_gen.py # generate test_initial_state.py
│   ├── completion_test_gen.py    # generate test_final_state.py (the graded verifier)
│   ├── apptainer_def_gen.py      # generate + iterate container.def, base-image routing
│   ├── container_def_patch.py    # inject %files sections for fixtures
│   ├── fixture_gen.py            # deterministic host-side multi-modal fixture materialisation
│   ├── sample_solutions.py       # bash agent harness (solve loop)
│   ├── vanillux_solver.py        # alternative "vanillux" solver harness
│   └── env.py                    # Apptainer/runtime helpers (fakeroot, base SIF resolution)
│
├── comparison/            # composition/difficulty/decontamination vs external baselines
│                          #   (endless-terminals, OpenThoughts, SWE-smith, r2e-gym, termigen, …)
├── decontamination/       # 13-gram overlap of corpora vs Terminal-Bench / TB-Lite
├── containers/            # prebuilt per-domain base Apptainer defs/SIFs + base_intricate
├── output/                # generated corpora live here (gitignored)
└── scripts/               # thin shell/Slurm launchers for every stage — see scripts/README.md
```

## Quickstart

All commands run from the **repo root**; the launcher scripts `cd` there
automatically. Python is run via `uv`.

```bash
# 0. (optional) estimate cost before committing to a big run
uv run python -m rl_data.estimate_cost --num-tasks 1000 --num-solutions 8

# 1. generate a small task corpus (env-overridable; legacy by default)
NUM_TASKS=10 OUT_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_tasks/run_generate_tasks.sh

# 2. solve the tasks with an LLM agent at pass@k
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_solutions/run_generate_solutions.sh

# 3. analyse pass@k + composition/balance stats (writes <TASKS_DIR>/analysis)
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/analyze/run_analyze.sh

# 4. publish to the Hugging Face Hub
bash rl_data/scripts/upload/upload_data_to_hf.sh
```

For the full set of launchers (presets, comparison pipeline, decontamination,
fixture repair, corpus combination) see [`scripts/README.md`](scripts/README.md).

## Corpus kinds

The released corpus combines two generations of the sampler, described together
in the paper. `CORPUS_KIND` controls how aggressively the newer axes (graded
verifiers, multi-modal fixtures, `intricate` complexity) are drawn:

| `CORPUS_KIND` | Behaviour |
|---------------|-----------|
| `legacy` (v1) | Original axes only: `exact_text` verifiers, `text_only` fixtures, 3 complexity buckets. |
| `sft_v2`      | Up-weights the newer verifier/fixture kinds + `intricate` complexity (M=2; ~67% intricate). |
| `rl_v2`       | Same axes, per-axis multipliers tuned so `legacy` + `rl_v2` concatenated gives a balanced final mix. |

The `legacy` and `*_v2` corpora are generated into separate `output/` dirs and
merged at training time with
[`scripts/combine/combine_corpora.py`](scripts/combine/combine_corpora.py)
(`balanced` for SFT, `union` for RL). Any `*_v2` corpus (i.e. anything that can
draw a non-legacy verifier/fixture or `intricate` complexity) resolves to
`containers/base_intricate.sif`, so build it once on a build node first:

```bash
apptainer build rl_data/containers/base_intricate.sif \
                 rl_data/containers/base_intricate.def
```

## SFT warm-start data

The same pipeline produces the SFT warm-start set: generate ~2.2k environments,
then run `generate_solutions` to roll out 8 trajectories per environment. The
successful (pass) trajectories form the SFT corpus used to warm-start RL.

## Evaluating on the published Harbor dataset

The full 15k corpus is published on the [Harbor](https://www.harborframework.com/docs/datasets)
registry as **[`tmax/TMax-15K-Harbor`](https://hub.harborframework.com/datasets/tmax/TMax-15K-Harbor/latest)**
(public). Each task ships as a self-contained Harbor environment with a
programmatic verifier, so you can evaluate any agent/model on it without
regenerating anything.

Evaluate an agent on the dataset with `harbor run -d` ([run docs](https://www.harborframework.com/docs/datasets)):

```bash
# full dataset
uv run harbor run -d "tmax/TMax-15K-Harbor@latest" \
    --agent terminus-2 --model "<model>" --env docker

# a quick subset: cap the number of tasks (-l) and/or filter by name (-i / -x)
uv run harbor run -d "tmax/TMax-15K-Harbor@latest" \
    --agent terminus-2 --model "<model>" --env docker \
    -i "task_00000*" -l 10
```

Results (per-task reward, agent/verifier logs) are written under `jobs/<job-name>/`.

**No local Docker?** Use the Daytona cloud sandbox instead of building images
locally — set `DAYTONA_API_KEY` and pass `--env daytona`:

```bash
uv run harbor run -d "tmax/TMax-15K-Harbor@latest" \
    --agent terminus-2 --model "<model>" \
    --env daytona --n-concurrent 10
```

## Requirements

- `uv` for dependency management (deps pinned in the repo-root `pyproject.toml` / `uv.lock`).
- An LLM API key for the configured model, e.g. `GEMINI_API_KEY` (default model
  is `gemini/gemini-3.1-pro-preview`). Local vLLM/Ollama/OpenAI-compatible
  endpoints are supported via env vars — see `scripts/README.md`.
- `apptainer` on PATH for building/running task containers.
- `HF_TOKEN` for the upload stage.
- To evaluate on the published `tmax/TMax-15K-Harbor` dataset: the `harbor` CLI
  (a project dep, run via `uv run harbor …`), a model API key for your chosen
  agent, and a container runtime — Docker locally, or `DAYTONA_API_KEY` for the
  Daytona sandbox on Docker-less hosts.
