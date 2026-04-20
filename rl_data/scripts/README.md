# `rl_data/scripts/` — launcher scripts

Thin shell / Slurm launchers for every stage of the RL-data pipeline. The
python code that these scripts call lives under `rl_data/` (e.g.
`rl_data.generate_tasks`, `rl_data.generate_solutions`, `rl_data.comparison`).

## Layout

```
scripts/
├── generate_tasks/         # create new tasks from scratch
│   ├── run_generate_tasks.sh          # generic wrapper
│   ├── run_generate_tasks_1k.sh       # 1k-task preset
│   └── run_generate_tasks_10k.sh      # 10k-task preset (what `tasks_skill_tax_20260401_10k` used)
│
├── generate_solutions/     # run LLM agents against tasks to collect solutions
│   ├── run_generate_solutions.sh      # generic wrapper
│   ├── run_generate_solutions_1k.sh
│   ├── run_generate_solutions_10k.sh  # the reference "solve our 10k" launcher
│   └── launch_vllm.sh                 # spin up a local vLLM server for Qwen etc.
│
├── analyze/                # analysis + cost estimation + format conversions
│   ├── run_analyze.sh                 # runs rl_data.analyze on a solved task dir
│   ├── estimate_cost.sh               # projects cost for a proposed run
│   ├── classify_difficulty.py         # bin tasks into Frontier/Advanced+/Advanced/Core tiers
│   └── convert_to_harbor.py           # export tasks into Harbor-compatible layout
│
├── comparison/             # head-to-head vs external terminal-task baselines
│   ├── run_ingest_et.sh               # pull + flatten obiwan96/endless-terminals
│   ├── run_ingest_openthoughts.sh     # pull + flatten open-thoughts/OpenThoughts-TB-dev
│   ├── run_classify_taxonomy.sh       # LLM-classify external tasks into OUR taxonomy
│   ├── run_generate_solutions_et.sh   # solve ET tasks with the same model as our 10k
│   ├── run_generate_solutions_openthoughts.sh
│   ├── run_comparison.sh              # one-shot pipeline: ingest -> classify -> solve -> compare
│   └── COMPARISON.md                  # full reference: modules, outputs, local-model usage, costs
│
└── upload/                 # push a solved+analysed task dir to Hugging Face
    ├── upload_data_to_hf.sh
    └── upload_data_to_hf_verified.sh  # skip tasks with 0 pass@k
```

## Conventions

- Every script does `cd "$PROJECT_ROOT"` so you can run it from anywhere.
- `MODEL`, `SAMPLE_SIZE`, `SAMPLE_SEED`, and the local-model env vars
  (`HOSTED_VLLM_API_BASE` / `OLLAMA_API_BASE` / `OPENAI_API_BASE`) can be set
  as overrides without editing files.
- Slurm-ready scripts include `#SBATCH` headers and can be launched with
  `sbatch`; bare `bash` also works for interactive runs.
- See [`comparison/COMPARISON.md`](comparison/COMPARISON.md) for the
  comparison pipeline in detail.
