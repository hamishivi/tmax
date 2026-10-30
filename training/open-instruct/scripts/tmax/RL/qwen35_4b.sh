#!/bin/bash

# DPPO training on Qwen3.5 4B with tmax-15k data
# we used the same script for other datasets,
# but with the dataset mixer list changed ofc :)

set -eo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
ASYNC_STEPS="${ASYNC_STEPS:-4}"
VLLM_SEED_PER_TURN="${VLLM_SEED_PER_TURN:-false}"
DATASET="${DATASET:-allenai/tmax-15k-open-instruct}"
SANDBOX_BACKEND="${SANDBOX_BACKEND:-docker}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
# TOOL_CONFIGS can override the complete environment config, including local task data.
if [ -z "${TOOL_CONFIGS:-}" ]; then
    TOOL_CONFIGS="$(uv run python - "$DATASET" "$SANDBOX_BACKEND" <<'PYCONFIG'
import json
import sys

config = {"task_data_hf_repo": sys.argv[1], "backend": sys.argv[2], "timeout": 120, "test_timeout": 600}
if sys.argv[2] in {"apptainer", "sandfleet"}:
    config["pwd"] = "/"
print(json.dumps(config))
PYCONFIG
)"
fi

MASON_ARGS=(--cluster "${BEAKER_CLUSTER:?Set BEAKER_CLUSTER}")
if [ -n "${BEAKER_WORKSPACE:-}" ]; then
    MASON_ARGS+=(--workspace "$BEAKER_WORKSPACE")
fi
if [ -n "${BEAKER_BUDGET:-}" ]; then
    MASON_ARGS+=(--budget "$BEAKER_BUDGET")
fi
# Forward only operator-supplied settings; credentials use Beaker secrets.
for variable in DOCKERHUB_USERNAME MIRROR_URL SANDFLEET_URL SANDFLEET_POOL SANDFLEET_REGISTRY \
    SWERL_APPTAINER_SIF_DIR SWERL_ROLLOUT_TIMEOUT_S SWERL_SANDBOX_PWD_MAP \
    SWERL_SANDBOX_VERIFIER_TIMEOUT_MAP SWERL_SANDBOX_MAX_VERIFIER_TIMEOUT_S; do
    if [ -n "${!variable:-}" ]; then
        MASON_ARGS+=(--env "$variable=${!variable}")
    fi
done
if [ -n "${DOCKER_PAT_SECRET:-}" ]; then
    MASON_ARGS+=(--secret "DOCKER_PAT=$DOCKER_PAT_SECRET")
fi
if [ -n "${SANDFLEET_CLIENT_TOKEN_SECRET:-}" ]; then
    MASON_ARGS+=(--secret "SANDFLEET_CLIENT_TOKEN=$SANDFLEET_CLIENT_TOKEN_SECRET")
fi
SANDBOX_BACKEND="$(uv run python -c 'import json, sys; print(json.loads(sys.argv[1]).get("backend", "docker"))' "$TOOL_CONFIGS")"
SANDBOX_SETUP=:
if [ "$SANDBOX_BACKEND" = docker ]; then
    SANDBOX_SETUP="source scripts/docker/docker_login.sh"
fi

uv run python mason.py \
       "${MASON_ARGS[@]}" \
       --image "$BEAKER_IMAGE" \
       --description "tmax-15k DPPO Qwen35 4b" \
       --pure_docker_mode \
       --priority urgent \
       --preemptible \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=4 \
       --env SWERL_DOCKER_START_CONCURRENCY=64 \
       --env SWERL_DOCKER_EXEC_CONCURRENCY=256 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_PODMAN_IMAGE_JANITOR_ENABLED=1 \
       --env SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S=60 \
       --env SWERL_PODMAN_IMAGE_JANITOR_UNTIL=10m \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --gpus 8 \
       --no_auto_dataset_cache \
       --auto_output_dir_path "" \
       --auto_checkpoint_state_dir "" \
       -- "$SANDBOX_SETUP" \&\& source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$DATASET" 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 16384 \
    --response_length 65536 \
    --pack_length 67584 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 32 \
    --async_steps "$ASYNC_STEPS" \
    --model_name_or_path hamishivi/Qwen3.5-4B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 128000 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 8 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --vllm_seed_per_turn "$VLLM_SEED_PER_TURN" \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs "$TOOL_CONFIGS" \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_freq 10 \
    --checkpoint_state_dir "$OUTPUT_DIR/checkpoint_state" \
    --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --dppo_ratio_cap 10.0 \
    --rollouts_save_path "${ROLLOUTS_SAVE_PATH:-$OUTPUT_DIR/rollouts}" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name qwen35_4b_dppo \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
