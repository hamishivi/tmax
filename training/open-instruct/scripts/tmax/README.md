# tmax training scripts

These are the training launch scripts for the tmax fork (Qwen 3.5 + terminal agent training).

## Scripts here

Scripts are split into two folders by training stage:

### `SFT/` — supervised finetuning (`open_instruct/finetune.py`)

| Script | What it does |
| --- | --- |
| `sft_qwen3_8b_small.sh` | SFT of Qwen3-8B on just the tmax SFT data |
| `sft_qwen3_8b_big.sh` | SFT of Qwen3-8B on the full SFT blob (all subsets) |
| `sft_qwen35_9b_small.sh` | SFT of Qwen3.5-9B on just the tmax SFT data |
| `sft_qwen35_9b_big.sh` | SFT of Qwen3.5-9B on the full SFT blob (all subsets) |

### `RL/` — DPPO RL (`open_instruct/grpo_fast.py`)

| Script | What it does |
| --- | --- |
| `qwen35_2b.sh` | DPPO RL on Qwen3.5-2B with `swerl-tmax-15k` |
| `qwen35_4b.sh` | DPPO RL on Qwen3.5-4B with `swerl-tmax-15k` |
| `qwen35_9b.sh` | DPPO RL on Qwen3.5-9B with `swerl-tmax-15k` |
| `qwen36_27b.sh` | DPPO RL on Qwen3.6-27B with `swerl-tmax-15k` |

The RL scripts mostly share the same args; the per-size differences are model name,
node/engine counts, rollout shape, and a few memory/perf flags (e.g. the 27b adds
`--gather_whole_model false` / `--deepspeed_zpg 1` to be able to fit the model during training).

## Reading a script: `mason.py` vs. the regular training command

Every script has the same two-part shape, split by a bare `--`:

```bash
uv run python mason.py \
    <mason / Beaker scheduling args> \
    -- \
    <the actual command that runs on the cluster>
```

### Part 1 — `mason.py` (the launcher)

`mason.py` is Ai2's Beaker job launcher. Everything **before** the `--` configures
*where and how* the job runs on the cluster, not the training itself. Common args:

- `--cluster` / `--workspace` / `--budget` — which Beaker cluster, workspace, and budget
- `--image` — the Docker image to run in (passed as `$1` to the script)
- `--num_nodes` / `--gpus` — how much hardware to request
- `--priority` / `--preemptible` / `--max_retries` — scheduling behavior
- `--pure_docker_mode`, `--mount_docker_socket`, `--env`, `--secret` — container/runtime setup

`mason.py` also does extra bookkeeping for known `open_instruct` commands (e.g.
`finetune.py`, `grpo_fast.py`): caching, auto-resume for GRPO, etc.

If you aren't at Ai2, then you can remove this part of the command. But note how many GPUs and nodes are requested, since you need to match that!

### Part 2 — the regular training command (after `--`)

Everything **after** the `--` is the command mason actually executes on each node.
This is the "regular script" you would run locally if you weren't using Beaker:

- SFT scripts: `accelerate launch ... open_instruct/finetune.py <training args>`
- RL scripts: `... python open_instruct/grpo_fast.py <training args>`

These args (`--model_name_or_path`, `--dataset_mixer_list`, `--learning_rate`,
`--max_seq_length`, `--response_length`, `--tools`, etc.) are the actual
hyperparameters and are documented by `finetune.py` / `grpo_fast.py` themselves.

## Running locally vs. on Beaker

To run without Beaker, drop the `uv run python mason.py ... --` prefix and run the
command after the `--` directly (adjusting `--num_processes`/node settings for your
hardware). To launch on the cluster, just run the script with a Beaker image:

```bash
bash scripts/tmax/SFT/sft_qwen3_8b_small.sh <beaker-image>
```

## General Tips

Training terminal agents is tough, painful, and requires a lot of patience. Here are some useful tips I found for this stack:

- Sometimes nodes go down because of 'running hot', since we run podman on the same nodes (unideal, but avoids paying money to big sandbox). Hence, turning off the podman janitor/lowering the pool size/lowering the start and exec concurrency can help preventing nodes from going down on restarts, if you repeatedly hit it.
- Increasing group size and output length helps a lot with stability (we show this a bit in the paper). In particular, training with ~65k output length seems quite important for the 9B and 4B model, otherwise the model has too many 'overlong negatives' and too much negative gradient.
- 2 nodes is all you need for training max, and put as much extra as you need for inference.
- Counterintuitively, training larger models can be more stable and just as fast as the smaller models, since often you are bottlenecked on sandbox memory/setup, and so have 'gpu wiggle room' for running slower, larger models.
- Instabilities seem to arise from the model getting into insane entropy states. Adding well-formedness rewards on language / tool calls may help in training stability.
