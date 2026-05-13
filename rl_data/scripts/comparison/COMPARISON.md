# Dataset comparison suite

Scripts & code for comparing our RL dataset (`tasks_skill_tax_20260401_10k`)
against terminal-task baselines — currently
[`obiwan96/endless-terminals`](https://huggingface.co/datasets/obiwan96/endless-terminals),
[`open-thoughts/OpenThoughts-Agent-v1-RL`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL),
[`ucsb-mlsec/terminal-bench-env`](https://github.com/ucsb-mlsec/terminal-bench-env)
(the TermiGen / Harbor 2.0 task bank, ~3.5k tasks),
and [`m-a-p/TerminalTraj-5k-instances`](https://huggingface.co/datasets/m-a-p/TerminalTraj-5k-instances)
(the TerminalTraj 5k-instance release, TerminalBench 1.0 layout, 5,660 tasks).

## TL;DR

```bash
# Full pipeline: ingest -> classify -> solve -> compare
bash rl_data/scripts/comparison/run_comparison.sh

# Cost-bounded run: solve only 250 randomly-sampled ET tasks
SAMPLE_SIZE=250 SAMPLE_SEED=42 bash rl_data/scripts/comparison/run_comparison.sh

# Analysis only (solves already done)
SKIP_INGEST_ET=1 SKIP_INGEST_OT=1 SKIP_CLASSIFY=1 \
  SKIP_SOLVE_ET=1 SKIP_SOLVE_OT=1 \
  bash rl_data/scripts/comparison/run_comparison.sh
```

## Files

### Python package — `rl_data/comparison/`

```
rl_data/comparison/
  core.py                     # DatasetSpec + load_records + save_fig_with_data + plot helpers
  modules.py                  # all 6 analysis functions
  command_taxonomy.py         # bash-command classifier (16 categories)
  taxonomy_classifier.py      # LLM-based domain/skill_type/task_complexity/command_complexity
  cli.py                      # `python -m rl_data.comparison.cli`
  adapters/
    skill_tax.py              # identity
    endless_terminals.py      # obiwan96/endless-terminals (HF)
    openthoughts_tb.py        # open-thoughts/OpenThoughts-TB-dev (HF)
    openthoughts_agent_rl.py  # open-thoughts/OpenThoughts-Agent-v1-RL (HF, parquet tarballs)
    termigen.py               # ucsb-mlsec/terminal-bench-env (GitHub, Harbor 2.0)
    terminaltraj.py           # m-a-p/TerminalTraj-5k-instances (HF, tarball, TB 1.0)
```

### Shell launchers — `rl_data/scripts/comparison/`

```
comparison/
  run_ingest_et.sh
  run_ingest_openthoughts.sh
  run_ingest_termigen.sh
  run_ingest_terminaltraj.sh
  run_classify_taxonomy.sh
  run_generate_solutions_et.sh
  run_generate_solutions_openthoughts.sh
  run_generate_solutions_termigen.sh
  run_generate_solutions_terminaltraj.sh
  run_comparison.sh           # top-level orchestrator
  COMPARISON.md               # this file
```

## Adding a new baseline

1. Drop a new adapter in `rl_data/comparison/adapters/<my_ds>.py`:

   ```python
   from rl_data.comparison.adapters import Adapter, flatten_harbor_task, register_adapter

   class MyDatasetAdapter(Adapter):
       name = "my_ds"
       hf_repo_id = "org/my-dataset"
       default_dst = "rl_data/output/tasks_my_ds"
       def convert_one(self, src, dst_root):
           return flatten_harbor_task(src, dst_root,
               source_name="myds", source_repo=self.hf_repo_id)

   register_adapter(MyDatasetAdapter())
   ```

2. Add the import in `rl_data/comparison/adapters/__init__.py`.
3. Add a `scripts/comparison/run_ingest_my_ds.sh` shell launcher.
4. Pass it as another `--baseline` to the CLI.

Modules already accept a list of N datasets — no changes needed on the
analysis side.

### Baseline-specific quirks

- **TermiGen** (`ucsb-mlsec/terminal-bench-env`, ~3.5k tasks): lives on
  GitHub, not HF. The adapter does a `--filter=blob:none --sparse` clone of
  only `environments_harbor/` so we skip the large `termigen_env.zip`
  (TB-1.0 artifact, unused). Every upstream Dockerfile `FROM`s
  `ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624`; the solve script
  prebuilds a shared `tbench_ubuntu24_base.sif` once and the adapter
  rewrites each per-task `container.def`'s
  `Bootstrap:`/`From:` header to `localimage`/`./tbench_ubuntu24_base.sif`
  so per-task builds only layer the task-specific delta (payload files +
  any extra apt/pip beyond the common `python3-pip`/`pytest`/`pandas`/
  `scipy` that the base bakes in). The real verifier is `tests/test_outputs.py`
  (the sibling `tests/test.sh` is just a Harbor-style reward-logging wrapper
  around `pytest` on that file, which our harness runs directly).
- **OpenThoughts-Agent-v1-RL**: tarballed inside a single `tasks.parquet`;
  verifier is `tests/test.sh`, wrapped into a pytest adapter that invokes
  `bash /tests/test.sh`. See adapter docstring for the instruction rewrite
  that fixes the upstream "fabricate your own fixtures" prompt.
- **Endless-Terminals**: ships native `container.def`s with
  `From: ./ubuntu_22.04.sif`; the ET solve script prebuilds that base SIF
  with python3/pip/pytest preloaded.
- **TerminalTraj** (`m-a-p/TerminalTraj-5k-instances`, 5,660 tasks): shipped
  as a single 13 MB `5k_instances.tar.gz` on HF; the adapter uses
  `hf_hub_download` + `tarfile.extractall`, not `snapshot_download`. Uses
  the **TerminalBench 1.0** layout (task root with `Dockerfile`,
  `task.yaml`, `tests/test_outputs.py`) — not Harbor 2.0 — so
  `flatten_harbor_task` doesn't apply; we write a TB-1.0–specific converter.
  Every task `FROM`s a **unique** `yizhilll/tb_container-<md5>:tmux_asciinema_v2`
  image on Docker Hub (~400 MB each, all 5,660 distinct), so we can't
  pre-bake a shared base SIF the way we do for TermiGen/OT; per-task
  Docker pulls are unavoidable. The images span many distros (Debian/
  Ubuntu/Fedora/Alpine/…) with varying Python+pip availability, so the
  adapter injects a robust pytest bootstrap into `%post` that tries, in
  order: existing `pip3` → system package manager → `get-pip.py`.
  Separately, a few tasks (e.g. Fedora 27 with glibc<2.33) fail with
  Apptainer's bundled `fakeroot` binary; the solve script therefore
  pre-builds each per-task SIF with `--ignore-fakeroot-command` (which
  `generate_solutions.build_sif()` doesn't pass) before handing off to
  the harness. Native `category`/`difficulty`/`tags` in `task.yaml` are
  placeholder constants (`mathematics`/`easy`/`["mathematics"]`) across
  all 5,660 tasks, so domain/skill-type buckets rely entirely on the LLM
  classifier output.

## Output layout (`rl_data/output/comparison/`)

Each figure ships with a co-located `.csv` holding the raw numbers so you can
replot in a custom style.

```
main/                                    # paper body (minimal)
  fig1_difficulty_headline.png  + .csv
  fig2_command_mix_coverage.png + .csv
  fig3_composition_domain.png   + .csv
  summary_table.md
  summary_data.csv                       # machine-readable mirror
  paper_snippets.md

appendix/                                # deep-dive
  difficulty_pass_at_k_overlay.png + .csv
  difficulty_turn_cdf.png          + .csv
  command_mix_distinct_categories_hist.png + .csv
  command_mix_turn_distribution.png        + .csv
  command_mix_cooccurrence_<dataset>.png   + .csv
  composition_skill_type.png        + .csv    # second-level taxonomy axis
  composition_task_complexity.png   + .csv
  composition_command_complexity.png + .csv
  diversity_shared_clusters.png     + .csv
  diversity_clusters.csv
  realism_histograms.png            + .csv
  verifier_assertion_types.png      + .csv
  verifier_loc_asserts.png          + .csv
  per_task_metrics.csv
report.json                              # full machine-readable dump
```

## Modules

1. **difficulty** — pass@1, pass@8, turns, tokens, cost per task, Mann–Whitney U.
2. **command_mix** — every bash tool call is tagged via
   `command_taxonomy.classify_one` (16 categories: file manip, code_write,
   code_run, pkg_install, service, db, net, ...); reports coverage, per-turn
   distribution, co-occurrence heatmaps. Uses **no metadata** → fully fair.
3. **composition** — projects each dataset onto OUR taxonomy on **four** axes
   (9 domains, ~29 skill_types, task complexity, command complexity) via the
   LLM classifier. `fig3_composition_domain.png` is the main-body chart;
   skill_type / complexity charts are in the appendix.
4. **diversity** — shared TF-IDF clustering on the union corpus.
5. **realism** — description length, apt+pip package counts, services started,
   artifacts checked by verifier.
6. **verifier** — `ast`-walks `test_final_state.py`: LOC, assert count,
   assertion-type distribution.

## Cost-bounded solve (SAMPLE_SIZE)

All four `run_generate_solutions_*.sh` scripts honour a `SAMPLE_SIZE` env
var that randomly subsamples tasks before solving, useful when you want
quick-turn comparisons without spending $30+:

```bash
# Solve only 250 randomly-sampled ET tasks with seed 42
SAMPLE_SIZE=250 SAMPLE_SEED=42 \
    bash rl_data/scripts/comparison/run_generate_solutions_et.sh
```

The sample is drawn from the filtered task set and is deterministic in
`SAMPLE_SEED` so reruns pick the same subset.

## pass@k breadth (NUM_SOLUTIONS)

All four solve scripts accept `NUM_SOLUTIONS` as an env override. The
harness's `run_n_solutions()` reports `pass@k` for every k in [1..N], so
`NUM_SOLUTIONS=8` gives you both `pass@1` and `pass@8` (and 2/3/4/5/6/7) in
one run. Default is `1` (matches the 10k gemini run).

```bash
# Pass@8 ET run with the API model
NUM_SOLUTIONS=8 sbatch rl_data/scripts/comparison/run_generate_solutions_et.sh
```

For local-model pass@k runs, see "Local models (vLLM / Ollama)" below.

## Local models (vLLM / Ollama)

The solver + classifier both use `litellm`, which supports local models via
OpenAI-compatible proxies out of the box. There are now **two** supported
flows: (A) one SLURM job that serves vLLM AND solves (recommended), or
(B) split serve and solve into two separate jobs (legacy).

### A. Single-job, in-job vLLM (recommended for pass@k with a local model)

Each `run_generate_solutions_*.sh` accepts `LAUNCH_VLLM=1`, which makes the
script bring up vLLM on the same node it allocates, wait for readiness, run
the solver against `http://127.0.0.1:<auto>/v1`, and tear vLLM down on exit.
For a single dataset:

```bash
LAUNCH_VLLM=1 NUM_SOLUTIONS=8 \
    sbatch rl_data/scripts/comparison/run_generate_solutions_et.sh
```

That single command gives you both `pass@1` and `pass@8` for ET under
`Qwen/Qwen3-8B` (the helper's default), with `_summary.json` filenames keyed
by model so existing gemini summaries are not overwritten.

For all four baselines under one shared vLLM (single 8xH200 allocation,
single weight load, no idle GPUs between datasets), use the orchestrator:

```bash
APPTAINER_DOCKER_USERNAME=... APPTAINER_DOCKER_PASSWORD=... \
    sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh
```

Useful overrides:

```bash
# Subset of datasets
DATASETS="et openthoughts" sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh

# Cost-bounded (250 random tasks per baseline)
SAMPLE_SIZE=250 SAMPLE_SEED=42 sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh

# Different model (Qwen2.5-Coder-7B-Instruct)
VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
    sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh

# Different pass@k breadth
NUM_SOLUTIONS=16 sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh
```

The vLLM helper (`_vllm_local.sh`) auto-picks tensor-parallel = 1 and
data-parallel = visible GPU count, which is the throughput-optimal config
for an 8B model that fits on one H200. Override via `VLLM_TP` / `VLLM_DP`.

For Qwen3-family models the helper additionally:

- enables `--tool-call-parser hermes` (OpenAI-compatible tool calls in our
  bash-tool agent loop),
- enables `--reasoning-parser qwen3` (so `<think>` tokens go to
  `reasoning_content`, not `content`),
- sets `--chat-template-kwargs '{"enable_thinking": false}'` to skip the
  thinking pass entirely (much faster turns; opt back in by exporting
  `VLLM_DISABLE_THINKING=0`).

### B. Two-job (separate vLLM + solver) — legacy

If you'd rather keep vLLM on one node and run solvers from another, the
older two-job recipe still works:

1. Launch vLLM on a GPU node (one-liner helper):

   ```bash
   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TP=1 \
       bash rl_data/scripts/generate_solutions/launch_vllm.sh
   ```

2. Point the solve script at it:

   ```bash
   export MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct"
   export HOSTED_VLLM_API_BASE="http://<vllm-host>:8000/v1"
   bash rl_data/scripts/comparison/run_generate_solutions_et.sh
   ```

The solve scripts detect `HOSTED_VLLM_API_BASE` / `OLLAMA_API_BASE` /
`OPENAI_API_BASE` and arrange the right env for `litellm.completion`. No
code changes required.

Ollama variant:

```bash
export MODEL="ollama_chat/qwen2.5-coder:7b"
export OLLAMA_API_BASE="http://localhost:11434"
```

Note: tool-calling (our bash-tool loop) requires the local backend support
OpenAI-style `tool_calls`. vLLM supports this via `--tool-call-parser
hermes` for Qwen2.5/Qwen3 (the launcher enables this by default).

## Dependencies

```bash
uv pip install scikit-learn scipy
uv pip install vllm==0.19.1    # only if running a local server
```

scipy powers Mann-Whitney U + chi-squared; scikit-learn powers TF-IDF + KMeans
in the diversity module. Both are gated — missing them logs a warning and
those sub-modules no-op rather than crashing.

## Rough cost estimate (API flash, full-run)

- Ingest: CPU-only, ~10 min for ~2500 ET tasks + ~730 OT-Agent-RL tasks +
  ~3500 TermiGen tasks + ~5660 TerminalTraj tasks. TermiGen's first run
  does a ~350 MB sparse clone of the GitHub repo; TerminalTraj's first
  run pulls a single 13 MB tarball from HF.
- Taxonomy classifier: ~$5-8 total (~12k × one flash call each).
- Solves (NUM_SOLUTIONS=1, ~30k tokens/task): ~$30-60 for ET, <$5 for
  OT-Agent-RL, ~$40-80 for TermiGen, ~$5-10 for TerminalTraj at default
  `SAMPLE_SIZE=500` (~$60-100 for the full 5660-task run). Baseline cost
  scales linearly with `NUM_SOLUTIONS`.
- **TerminalTraj disk budget**: each per-task SIF is ~500 MB, so 500
  tasks ≈ 250 GB, 5660 ≈ 2.8 TB. Bound via `SAMPLE_SIZE`.
- Comparison post-processing: <10 min CPU on 10k + ~12k tasks.

Using `SAMPLE_SIZE=250` cuts any solve to ~10% of the above. Using a
local Qwen model cuts it to zero (modulo GPU wall time).
