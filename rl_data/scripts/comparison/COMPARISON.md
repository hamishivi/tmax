# Dataset comparison suite

Scripts & code for comparing our RL dataset (`tasks_skill_tax_20260401_10k`)
against terminal-task baselines — currently [`obiwan96/endless-terminals`](https://huggingface.co/datasets/obiwan96/endless-terminals)
and [`open-thoughts/OpenThoughts-TB-dev`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev).

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
    endless_terminals.py
    openthoughts_tb.py
```

### Shell launchers — `rl_data/scripts/comparison/`

```
comparison/
  run_ingest_et.sh
  run_ingest_openthoughts.sh
  run_classify_taxonomy.sh
  run_generate_solutions_et.sh
  run_generate_solutions_openthoughts.sh
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

Both `run_generate_solutions_et.sh` and `run_generate_solutions_openthoughts.sh`
honour a `SAMPLE_SIZE` env var that randomly subsamples tasks before solving,
useful when you want quick-turn comparisons without spending $30+:

```bash
# Solve only 250 randomly-sampled ET tasks with seed 42
SAMPLE_SIZE=250 SAMPLE_SEED=42 \
    bash rl_data/scripts/comparison/run_generate_solutions_et.sh
```

The sample is drawn from the filtered task set and is deterministic in
`SAMPLE_SEED` so reruns pick the same subset.

## Local models (vLLM / Ollama)

The solver + classifier both use `litellm`, which supports local models via
OpenAI-compatible proxies out of the box. To run with Qwen2.5-Coder-7B locally:

1. Launch vLLM on a GPU node (one-liner helper):

   ```bash
   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TP=1 \
       bash rl_data/scripts/generate_solutions/launch_vllm.sh
   ```

   For the 14B model set `MODEL=Qwen/Qwen2.5-Coder-14B-Instruct` (and `TP=2`
   if you need more memory).

2. Point the solve script at it:

   ```bash
   export MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct"
   export HOSTED_VLLM_API_BASE="http://<vllm-host>:8000/v1"
   bash rl_data/scripts/comparison/run_generate_solutions_et.sh
   ```

The solve scripts detect `HOSTED_VLLM_API_BASE` / `OLLAMA_API_BASE` /
`OPENAI_API_BASE` and arrange the right env for `litellm.completion`. No code
changes required.

Ollama variant:

```bash
export MODEL="ollama_chat/qwen2.5-coder:7b"
export OLLAMA_API_BASE="http://localhost:11434"
```

Note: tool-calling (our bash-tool loop) requires that the local backend
supports OpenAI-style `tool_calls`. vLLM supports this via
`--tool-call-parser hermes` for Qwen2.5 (the launcher enables this by default).

## Dependencies

```bash
uv pip install scikit-learn scipy
uv pip install vllm            # only if running a local server
```

scipy powers Mann-Whitney U + chi-squared; scikit-learn powers TF-IDF + KMeans
in the diversity module. Both are gated — missing them logs a warning and
those sub-modules no-op rather than crashing.

## Rough cost estimate (API flash, full-run)

- Ingest: CPU-only, ~10 min for ~2500 ET tasks + ~100 OT tasks.
- Taxonomy classifier: ~$1-2 total (~2600 × one flash call each).
- Solves: ~$30-60 for ET (2500 tasks × 8 runs × ~30k tokens).
  OpenThoughts-TB is tiny (~100 tasks); <$2.
- Comparison post-processing: <5 min CPU on 10k + 2600 tasks.

Using `SAMPLE_SIZE=250` cuts the ET solve to ~10% of the above. Using a
local Qwen model cuts it to zero (modulo GPU wall time).
