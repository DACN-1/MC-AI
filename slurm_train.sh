#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# ^ first run builds the LLaVA cache (~26 h); subsequent runs reuse it and
#   finish in ~30 min of head training. Drop to 04:00:00 once caches exist.
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
CACHE_DIR="${CACHE_DIR:-./caches}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Ablation-cell selectors. One job = one cell.
#   BACKBONE: llava | clip
#   TASK_FILTER: e.g. chop_a_tree, collect_dirt
#   USE_LANGUAGE: 1 | 0
BACKBONE="${BACKBONE:-llava}"
TASK_FILTER="${TASK_FILTER:?Set TASK_FILTER (e.g. chop_a_tree or collect_dirt)}"
USE_LANGUAGE="${USE_LANGUAGE:-1}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"          # large is fine on cached features
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"  # backbone forward batch
LR="${LR:-1e-3}"
PAST_ACTION_K="${PAST_ACTION_K:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"

# Tag the output dir per cell so 8 jobs don't fight over output/
LANG_TAG=$([ "$USE_LANGUAGE" = "1" ] && echo lang || echo nolang)
OUTPUT_DIR="$OUTPUT_DIR/${BACKBONE}_${TASK_FILTER}_${LANG_TAG}"

echo "Cell: backbone=$BACKBONE  task=$TASK_FILTER  use_language=$USE_LANGUAGE"
echo "Past-action K=$PAST_ACTION_K  chunk=$CHUNK_SIZE  epochs=$EPOCHS  batch=$BATCH_SIZE  lr=$LR"
echo "Data=$DATA_DIR  cache=$CACHE_DIR  output=$OUTPUT_DIR"

NO_LANGUAGE_FLAG=""
if [ "$USE_LANGUAGE" != "1" ]; then
    NO_LANGUAGE_FLAG="--no-language"
fi

# Default path: precompute (or reuse) cache, then train head fast.
# Pass --end-to-end to fall back to the legacy backbone-every-batch loop.
python cluster_pipeline.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --task-filter "$TASK_FILTER" \
    --past-action-k "$PAST_ACTION_K" \
    --chunk-size "$CHUNK_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --cache-batch-size "$CACHE_BATCH_SIZE" \
    --lr "$LR" \
    --device cuda \
    --num-workers 8 \
    $NO_LANGUAGE_FLAG

echo "Finished: $(date)"
