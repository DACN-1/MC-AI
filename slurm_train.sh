#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "Started: $(date)"

# Load modules
module load Stages/2023
module load GCC/11.3.0
module load Python/3.10.4
module load CUDA/11.7

# Activate venv
source .venv/bin/activate

# HuggingFace cache
export HF_HOME="$(pwd)/hf_cache"
export TRANSFORMERS_CACHE="$(pwd)/hf_cache/transformers"

# Performance settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs

# Configuration (override via environment variables)
DATA_DIR="${DATA_DIR:-./trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"

echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS, Batch: $BATCH_SIZE, LR: $LR"

# Run pipeline: convert videos -> train -> evaluate
python cluster_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device cuda \
    --num-workers 8

echo "Finished: $(date)"
