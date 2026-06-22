<p align="center">
  <img src="assets/tmax-banner.png" alt="TMax" width="640">
</p>

<p align="center">
  <em>Simple terminal-using agents.</em>
</p>

<p align="center">
  💻 <a href="https://github.com/hamishivi/tmax">Code</a> ·
  🤗 <a href="https://huggingface.co/collections/allenai/tmax">Models &amp; Data</a> ·
  📜 <a href="https://github.com/hamishivi/tmax/blob/master/assets/paper.pdf">Paper</a> ·
  📓 <a href="https://wai-org.com/blog/tmax/">Blog</a>
</p>

---

Tmax is our project around training simple, powerful terminal using agents. This codebase covers data generation, training, and evaluation.
We use this to train our 'tmax' series of models, with strong performance as shown below! Please refer to our [paper](https://arxiv.org/abs/xxx) for more details.


<p align="center">
  <img src="scripts/plot/output/teaser_terminal_bench.png" alt="Tmax" width="640">
</p>


Below, we give a quick overview of the codebase and how to use it.

### News
- Initial release of the codebase and models! Please read [our paper](https://arxiv.org/abs/xxx) for more details.



## What's here

The repo is organised around four stages of building a terminal agent:

| Stage | Where | What it does |
|-------|-------|--------------|
| **Data generation** | `rl_data/` | A simple, scalable, diverse, difficulty-aware pipeline for synthesising terminal-agent tasks, solving them at pass@k, analysing the corpus, and publishing it to the Hugging Face Hub. Tasks are sampled as an *independent product of structured axes* and packaged as self-contained Apptainer/Docker environments with programmatic verifiers. |
| **Agent** | `Vanillux2Agent/` | The Harbor agent used for solving and evaluation: a direct LiteLLM agent built on the vanillux prompt harness (mini-SWE-agent-derived prompts, bash tool schema, submit marker, format-error recovery, and output truncation), executing commands through Harbor's active environment. |
| **Training** | `training/open-instruct/` | A fork of [open-instruct](https://github.com/allenai/open-instruct) with fixes for Qwen 3.5 and terminal-agent training. SFT and DPPO RL launch scripts for the tmax models live under `training/open-instruct/scripts/tmax/`. |
| **Evaluation** | `scripts/` + `beaker_configs/` | Shell/Slurm launchers and a Beaker pipeline that serves a model with vLLM and runs Harbor datasets (Terminal-Bench, TB-Lite, SWE-bench) against it. |

## Quickstart

Python is run via [`uv`](https://github.com/astral-sh/uv); all commands run from
the repo root.

```bash
# Install dependencies
uv sync
```

### Generate task data

```bash
# 1. generate a small task corpus
NUM_TASKS=10 OUT_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_tasks/run_generate_tasks.sh

# 2. solve the tasks with an LLM agent at pass@k
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_solutions/run_generate_solutions.sh

# 3. analyse pass@k + composition/balance stats
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/analyze/run_analyze.sh
```

[`rl_data/README.md`](rl_data/README.md) is the entry point for the data side
of the project: it walks through the compositional sampler (the orthogonal
axes that define a task), the four-stage pipeline
(`generate_tasks → generate_solutions → analyze → upload_to_hf`), the corpus
kinds (`legacy` / `sft_v2` / `rl_v2`), the SFT warm-start set, and how to
evaluate on our published Harbor dataset. Start there for anything data-related.

### Train a model

SFT and DPPO RL are run via the open-instruct fork in `training/open-instruct/`.
Launch scripts for the tmax models live under `training/open-instruct/scripts/tmax/`
(`SFT/` and `RL/`):

```bash
# from training/open-instruct/, e.g. RL on Qwen3.5-4B
bash scripts/tmax/RL/qwen35_4b.sh <beaker-image>
```

See [`training/open-instruct/scripts/tmax/README.md`](training/open-instruct/scripts/tmax/README.md)
for how to read the scripts (`mason.py` launcher vs. the underlying training command)
and how to run them off-cluster.

### Evaluate a model

Run a Harbor dataset against a locally served model on Beaker:

```bash
./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
    --revision sft_qwen3_4b_tmax_4node \
    --name sft-4b \
    --dataset terminal-bench@2.0
```

See [`scripts/beaker/README.md`](scripts/beaker/README.md) for the full eval
pipeline, flags, and troubleshooting.

#### Evaluate manually

To run an eval yourself on a node with GPUs (no Beaker), serve the model with
vLLM and point Harbor at it:

```bash
# 1. serve the model on localhost:8008 (a separate shell/process)
uvx vllm==0.19.1 serve allenai/tmax-9b \
    --served-model-name tmax-9b \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 8 --port 8008

# set daytona key
export DAYTONA_API_KEY='xxx'

# 3. run a Harbor dataset against the local vLLM endpoint
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/tmax-9b \
  --agent-kwarg api_base=http://localhost:8008/v1 \
  --agent-kwarg max_format_errors=64 \
  --n-concurrent 16 \
  -k 5 \
  --job-name swerl-qwen32-2b-tmax-step100-vanillux2-daytona-tblite
```

Feel free to swap out daytona for your sandboxing backend of choice.

## Task data on Harbor

We also ship the full **15k** task corpus in [Harbor](https://www.harborframework.com/docs/datasets)
format so you can run it out of the box. It's published on the Harbor registry as
**[`tmax/TMax-15K-Harbor`](https://hub.harborframework.com/datasets/tmax/TMax-15K-Harbor/latest)**
(public): the legacy 10k self-contained tasks plus 5k newer *intricate*
multi-modal tasks. Every task ships as a self-contained Harbor environment with
a programmatic verifier, so you can evaluate any agent/model on it directly — no
need to regenerate or build anything from `rl_data/`.

```bash
# evaluate an agent on a 10-task subset using the Daytona cloud sandbox
# (no local Docker required — swap `--env daytona` for `--env docker` to build locally)
export DAYTONA_API_KEY='xxx'
uv run harbor run -d "tmax/TMax-15K-Harbor@latest" \
    --agent terminus-2 --model "<model>" \
    --env daytona -l 10
```

Drop `-l` to run the full set, or use `-i`/`-x` to include/exclude tasks by name.
Per-task rewards and agent/verifier logs land under `jobs/<job-name>/`. See
[`rl_data/README.md`](rl_data/README.md#evaluating-on-the-published-harbor-dataset)
for the full set of run options.

## Requirements

- **[`uv`](https://github.com/astral-sh/uv)** for dependency management (deps pinned in `pyproject.toml` / `uv.lock`).
- An **LLM API key** for the configured model (e.g. `GEMINI_API_KEY`); local
  vLLM / Ollama / OpenAI-compatible endpoints are also supported via env vars.
- (for data generation) **`apptainer`** on PATH for building and running task containers.
- (for training) A **Dockerhub login** and personal access token (PAT). In particular, you probably need a business account to pull images from Dockerhub at large scale.
- **`HF_TOKEN`** for the Hugging Face upload stage and for pulling gated models.
- (to evaluate on the published `tmax/TMax-15K-Harbor` dataset) a model API key
  for your chosen agent plus a container runtime — Docker locally, or
  `DAYTONA_API_KEY` for the Daytona sandbox on Docker-less hosts.

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](LICENSE).

# Citation

If you use this codebase or models in your research, please cite our paper:

```bibtex
@misc{ivison2026tmax,
  title={{TMAX}: A Simple Recipe for Terminal Agents},
  author={Ivison, Hamish and Yin, Junjie Oscar and Shao, Rulin and Xiao, Teng and Lambert, Nathan and Hajishirzi, Hannaneh},
  year={2026},
}
```