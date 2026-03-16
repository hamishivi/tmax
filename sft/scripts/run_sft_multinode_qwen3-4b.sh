#!/usr/bin/env bash
#
# Multi-node SFT training for Qwen3-4B-Instruct-2507 via SLURM.
#
# Usage:
#   sbatch scripts/run_sft_multinode_qwen3-4b.sh        # submit as batch job
#   bash  scripts/run_sft_multinode_qwen3-4b.sh          # run inside an existing salloc
#
# For salloc, request nodes first (note: --gpus-per-node, NOT --gpus):
#   salloc --qos=normal --nodes=2 --gpus-per-node=8 --cpus-per-task=8 --mem=1440G --time=8:00:00
#
# ── SLURM directives (used by sbatch, ignored by bash) ───────────────────────
#SBATCH --job-name=sft-qwen3-4b
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1440G
#SBATCH --time=14:00:00
#SBATCH --output=/gpfs/scrubbed/osey/tmax/sft/output/slurm-%j.out

set -euo pipefail
cd /gpfs/scrubbed/osey/tmax/sft

# ── Environment (modules + venv) ─────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/12.9.1
export CUDA_HOME=/gpfs/software/cuda/12.9.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source /gpfs/scrubbed/osey/tmax/.venv/bin/activate

export TRITON_CACHE_DIR="/gpfs/scrubbed/osey/.triton_cache"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="/gpfs/scrubbed/osey/tmax/models/Qwen3-4B-Instruct-2507"

GPUS_PER_NODE=8
NUM_NODES="${SLURM_NNODES:-2}"
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

ACCEL_CONFIG="configs/accelerate_ds_z3_sp4_2x8xh200.yaml"

# Data
TOKENIZED_DATASET="/gpfs/scrubbed/osey/tmax/sft/data/tokenized_tbmax_terminus2_sweagent_full_20260315_qwen3_asst_loss_42"

# Subsampling (comment out to train on the full dataset)
MAX_TRAIN_SAMPLES=100000
SEED=42

# Training hyperparams
GLOBAL_BATCH_SIZE=128
MAX_LENGTH=65536
NUM_EPOCHS=2
LR=2e-5

LOGGING_STEPS=1
SAVE_STEPS=0.1
WANDB_PROJECT="tmax-sft"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_PATH="/gpfs/scrubbed/osey/tmax/sft/output"
MODEL_NAME=$(basename "$MODEL")

DATA_NAME=""
for path in $TOKENIZED_DATASET; do
    b=$(basename "$path" | sed 's/^tokenized_//')
    if [ -z "$DATA_NAME" ]; then
        DATA_NAME="$b"
    else
        DATA_NAME="${DATA_NAME}_${b}"
    fi
done

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
    DATA_NAME="${DATA_NAME}_n${MAX_TRAIN_SAMPLES}"
fi

OUTPUT_DIR="${BASE_PATH}/${MODEL_NAME}_${DATA_NAME}"
RUN_NAME="${MODEL_NAME}_${DATA_NAME}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

echo "=== Multi-node SFT ==="
echo "  Model:       $MODEL"
echo "  Nodes:       $NUM_NODES ($SLURM_JOB_NODELIST)"
echo "  GPUs:        $NUM_GPUS ($GPUS_PER_NODE/node)"
echo "  Master:      $MASTER_ADDR:$MASTER_PORT"
echo "  Config:      $ACCEL_CONFIG"
echo "  Dataset:     $TOKENIZED_DATASET"
echo "  Subsample:   ${MAX_TRAIN_SAMPLES:-full}"
echo "  Output:      $OUTPUT_DIR"
echo "  Log:         $LOG_FILE"
echo ""

# ── Build training args ──────────────────────────────────────────────────────
TRAIN_ARGS=(
    train.py
    --model_name_or_path "$MODEL"
    --output_dir "$OUTPUT_DIR"
    --tokenized_dataset_path $TOKENIZED_DATASET
    --num_gpus "$NUM_GPUS"
    --per_device_train_batch_size 1  # MUST be 1 with Ulysses SP: SP registers
                                     # attention shapes on the first forward pass,
                                     # so batch size must be constant. Effective
                                     # batch size is controlled via global_batch_size
                                     # and gradient accumulation instead.
    --max_length "$MAX_LENGTH"
    --num_train_epochs "$NUM_EPOCHS"
    --learning_rate "$LR"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$SAVE_STEPS"
    --seed "$SEED"
    --dataset_num_proc 1
    --packing
    --optim adamw_torch_fused
    --wandb_project "$WANDB_PROJECT"
    --run_name "$RUN_NAME"
)

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
    TRAIN_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

# ── Launch ───────────────────────────────────────────────────────────────────
NODE_LAUNCHER="${OUTPUT_DIR}/.node_launcher_${SLURM_JOB_ID}.sh"
cat > "$NODE_LAUNCHER" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
cd /gpfs/scrubbed/osey/tmax/sft
module load gcc/13.4.0
module load cuda/12.9.1
export CUDA_HOME=/gpfs/software/cuda/12.9.1
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
source /gpfs/scrubbed/osey/tmax/.venv/bin/activate
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
accelerate launch \\
    --config_file "$ACCEL_CONFIG" \\
    --deepspeed_multinode_launcher standard \\
    --main_process_ip "\$MASTER_ADDR" \\
    --main_process_port "\$MASTER_PORT" \\
    --machine_rank "\$SLURM_NODEID" \\
    $(printf '%q ' "${TRAIN_ARGS[@]}")
LAUNCHER
chmod +x "$NODE_LAUNCHER"

unset SLURM_CPUS_PER_TASK
srun --nodes="$NUM_NODES" --ntasks-per-node=1 \
    bash "$NODE_LAUNCHER" 2>&1 | tee "$LOG_FILE"

rm -f "$NODE_LAUNCHER"
