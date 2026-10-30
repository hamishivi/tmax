# Tmax training recipes

Run these scripts from `training/open-instruct/`. They use the current trainer,
including capped DPPO, valid-sibling reward handling, distributed batch padding,
and independent grading deadlines.

Start with the single-GPU debug recipe after configuring a Docker/Podman service
or a remote sandbox backend:

```bash
bash scripts/tmax/RL/qwen35_2b_1gpu.sh
```

It uses the Tmax task dataset, synchronous collection (`ASYNC_STEPS=0`), and
24 episodes in groups of eight for three learner steps. Full RL recipes retain
asynchronous collection (`ASYNC_STEPS=4`). All RL recipes explicitly set
`--dppo_ratio_cap 10.0`.

## Recipes

| Script | Model / input |
| --- | --- |
| `RL/qwen35_2b_1gpu.sh` | Qwen3.5-2B, single-GPU debug run |
| `RL/qwen35_2b.sh` | Qwen3.5-2B |
| `RL/qwen35_4b.sh` | Qwen3.5-4B |
| `RL/qwen35_9b.sh` | Qwen3.5-9B |
| `RL/qwen36_27b.sh` | Qwen3.6-27B |
| `RL/qwen3_8b.sh` | Qwen3-8B with the SFT warm start |
| `SFT/sft_qwen3_8b_small.sh` | Qwen3-8B on Tmax SFT data |
| `SFT/sft_qwen3_8b_big.sh` | Qwen3-8B on the full SFT mixture |
| `SFT/sft_qwen35_9b_small.sh` | Qwen3.5-9B on Tmax SFT data |
| `SFT/sft_qwen35_9b_big.sh` | Qwen3.5-9B on the full SFT mixture |

The full recipes keep their existing GPU layouts: the 2B/4B RL recipes request
four nodes, the 9B/27B/8B RL recipes eight, and SFT four, each with eight GPUs.
The 27B recipe also uses `--gather_whole_model false --deepspeed_zpg 1`.

## Launching full recipes

Full recipes use the existing `mason.py` Beaker launcher. Supply a container image
as the first argument and set `BEAKER_CLUSTER` in your environment.
`BEAKER_WORKSPACE` and `BEAKER_BUDGET` are optional; when omitted, Beaker uses
its configured defaults. The scripts contain no site addresses or secret names.

```bash
bash scripts/tmax/RL/qwen35_9b.sh <beaker-image>
bash scripts/tmax/SFT/sft_qwen35_9b_small.sh <beaker-image>
```

Arguments before the bare `--` configure Mason; those after it configure training.
To use another scheduler, adapt that prefix to your own launcher and preserve the
GPU/Ray topology. For SFT, Mason expands the per-node `--num_processes 8` into the
world size and sets Accelerate's node rendezvous arguments.

Each recipe passes `--no_auto_dataset_cache` before `--` so submission does not
import vLLM locally. Dataset preparation happens on the training nodes.
`OUTPUT_DIR` defaults to `/output` for full recipes; checkpoints and traces should
be placed on storage retained by your launcher. Set `ROLLOUTS_SAVE_PATH` to override
the default `$OUTPUT_DIR/rollouts` in any RL recipe. Reference-policy state is
saved alongside trainer checkpoints under `$OUTPUT_DIR/checkpoint_state`.

## RL data and execution

| Variable | Behavior |
| --- | --- |
| `DATASET` | Dataset mixer and task-data repository; defaults to `allenai/tmax-15k-open-instruct` |
| `ASYNC_STEPS` | Zero selects synchronous collection; positive values enable prefetching |
| `VLLM_SEED_PER_TURN` | Defaults to `false`; set `true` to use distinct deterministic seeds for generation turns |
| `SANDBOX_BACKEND` | Defaults to `docker`; also supports `apptainer` and `sandfleet` |
| `TOOL_CONFIGS` | Optional complete JSON override for the sandbox configuration |

Select the dataset and generation-seed setting recorded for the run you want to
reproduce. The v1.1 size/SFT-warm-start runs used per-turn seeding. Changing only
the model or dataset does not reproduce all historical run settings.

By default, the generated tool config follows `DATASET`, sets ordinary command
timeout to 120 seconds and verifier timeout to 600 seconds, and uses `/` as the
initial cwd for Apptainer/Sandfleet Tmax images. Recognized submissions receive
the verifier budget plus 120 seconds of transport/cleanup overhead, independently
of the ordinary tool and remaining rollout deadlines.

For a separately prepared local corpus, override both the mixer and task data:

```bash
DATASET=your-org/task-dataset \
TOOL_CONFIGS='{"backend":"sandfleet","task_data_dir":"/data/task-data","pwd":"/","timeout":120,"test_timeout":600}' \
    bash scripts/tmax/RL/qwen35_9b.sh <beaker-image>
```

The launcher uses the backend in `TOOL_CONFIGS` to select sandbox setup. Sandfleet
requires `SANDFLEET_URL` and a client token. Beaker recipes accept `SANDFLEET_CLIENT_TOKEN_SECRET` as the name of
a secret to expose as `SANDFLEET_CLIENT_TOKEN`; the local debug recipe reads the
token directly from its environment. `SANDFLEET_POOL` is optional. Install the
controller and workers from [Sandfleet](https://github.com/hamishivi/sandfleet)
at matching revisions; the training client uses the HTTP API directly.

The full RL recipes forward operator-provided `SANDFLEET_URL`, `SANDFLEET_POOL`,
`SANDFLEET_REGISTRY`, `SWERL_APPTAINER_SIF_DIR`, `SWERL_ROLLOUT_TIMEOUT_S`,
`SWERL_SANDBOX_PWD_MAP`, `SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP`, and
`SWERL_SANDBOX_MAX_VERIFIER_TIMEOUT_S`, plus `NO_PROXY` and `no_proxy`. Set proxy
bypasses for both the controller and the worker-agent addresses. Full and
minimal Ray runtime environments preserve sandbox connection settings. Referenced
files must be visible on the nodes that use them. A local SIF pool is required
for Apptainer/Sandfleet image resolution; missing images fail explicitly.
See [Sandfleet recovery](../../../../docs/sandfleet-restart-recovery.md) and
[optional dataset setup](../../../../docs/dataset-setup.md).

The debug recipe creates 16 environment actors. Size a named Sandfleet pool for
that lease demand and its intended CPU/RAM profile. Without a named pool, enable
declarative resources on the controller; the adapter requests two CPUs and
4096 MiB by default. Sandboxes run offline, so task dependencies belong in the
prepared images.

For Docker/Podman, full recipes retain the existing Podman setup. Supply
`DOCKERHUB_USERNAME` and optionally `DOCKER_PAT_SECRET` for authenticated image
pulls; `MIRROR_URL` is optional. The local debug recipe expects an already
configured Docker/Podman service and credentials.

## Pretokenized SFT input

SFT recipes normally use their published dataset mixtures. To load saved token
trajectories directly, set `PRETOKENIZED_JSONL` to a file visible on every training
node. This replaces the dataset mixer/config arguments:

```bash
PRETOKENIZED_JSONL=/data/trajectories.jsonl \
    bash scripts/tmax/SFT/sft_qwen35_9b_small.sh <beaker-image>
```

Each JSONL record must contain equal-length `input_ids` and `labels`. The loader
adds an all-ones attention mask and filters records exceeding `max_seq_length`,
without retokenizing tool or special-token boundaries.
