#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --partition=Abaki
#SBATCH --qos=abaki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
# ^ Must stay inside the weekly compvis26 reservation window
#   (Sat 06:00 -> Mon 06:00 holds every Abaki node). 48 h covers an LLaVA
#   cache build (~26 h) + head train (< 1 h) with ±50% slack. Subsequent
#   cache-hit runs take ~30 min — override with --time=04:00:00 then.
# Note: no --mem here. SLURM on LMU CIP tracks RealMemory=1 (misconfigured),
# so any --mem request is rejected. The OS hands us actual RAM regardless
# (128 GB on 1N nodes, 512 GB on 2N).
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Started: $(date)"

REPO_ROOT="${REPO_ROOT:-$HOME/BIG}"
cd "$REPO_ROOT"

# Activate venv (created once: python3.11 -m venv "$REPO_ROOT/.venv")
source "$REPO_ROOT/.venv/bin/activate"

# Performance settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs

# ---------- ablation-cell selectors (one job per cell) ----------------------
#   BACKBONE: llava | clip
#   TASK_FILTER: e.g. chop_a_tree, collect_dirt
#   USE_LANGUAGE: 1 | 0
BACKBONE="${BACKBONE:-llava}"
TASK_FILTER="${TASK_FILTER:?Set TASK_FILTER (e.g. chop_a_tree or collect_dirt)}"
USE_LANGUAGE="${USE_LANGUAGE:-1}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
PAST_ACTION_K="${PAST_ACTION_K:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"

LANG_TAG=$([ "$USE_LANGUAGE" = "1" ] && echo lang || echo nolang)

# ---------- storage layout --------------------------------------------------
# /var/tmp1 is the local NVMe on Abaki nodes (1 TB on 1N, 2 TB on 2N).
# It wipes on weekly reboot; feature_cache.precompute is resumable so a
# partial cache rebuild is the worst case.
NODE_SCRATCH="/var/tmp1/$USER"
mkdir -p "$NODE_SCRATCH"

DATA_TARBALL_DIR="$REPO_ROOT/trajectories"            # persistent tarballs
DATA_DIR="$NODE_SCRATCH/trajectories"                 # extracted, per node
CACHE_DIR="$NODE_SCRATCH/caches"                      # survives across jobs
OUTPUT_DIR="$REPO_ROOT/output/${BACKBONE}_${TASK_FILTER}_${LANG_TAG}"

export HF_HOME="$NODE_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

mkdir -p "$DATA_DIR" "$CACHE_DIR" "$OUTPUT_DIR" "$HF_HOME"

# ---------- stage trajectory data on local NVMe -----------------------------
# The tarballs wrap their contents in an extra AAA_trajectory/ directory and
# ship per-video action/info JSONLs without the consolidated all_actions.json
# that feature_cache.enumerate_samples requires. So:
#   1. --strip-components=1 drops the wrapper.
#   2. consolidate_metadata.py walks the extracted dir and writes
#      all_actions.json + all_infos.json once.
# Both steps are idempotent so reruns on the same node are no-ops.
TRAJ_SUBDIR="$DATA_DIR/trajectory_task_${TASK_FILTER}_length_3000"
if [ ! -f "$TRAJ_SUBDIR/all_actions.json" ]; then
    TARBALL=$(ls "$DATA_TARBALL_DIR"/trajectory_task_${TASK_FILTER}_*.tar.gz 2>/dev/null | head -1 || true)
    if [ -z "$TARBALL" ]; then
        echo "ERROR: no tarball matching trajectory_task_${TASK_FILTER}_*.tar.gz under $DATA_TARBALL_DIR" >&2
        exit 1
    fi
    if [ ! -d "$TRAJ_SUBDIR" ]; then
        echo "Extracting $TARBALL -> $DATA_DIR (strip AAA_trajectory/ wrapper)"
        tar -xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1
    else
        echo "Already extracted: $TRAJ_SUBDIR (missing consolidated JSON only)"
    fi
    echo "Consolidating per-video JSONLs -> all_actions.json + all_infos.json"
    python consolidate_metadata.py --data-dir "$DATA_DIR" --delete-originals
else
    echo "Trajectory + consolidated JSON already staged: $TRAJ_SUBDIR"
fi

echo "Cell: backbone=$BACKBONE  task=$TASK_FILTER  use_language=$USE_LANGUAGE"
echo "Past-action K=$PAST_ACTION_K  chunk=$CHUNK_SIZE  epochs=$EPOCHS  batch=$BATCH_SIZE  lr=$LR"
echo "Data=$DATA_DIR  cache=$CACHE_DIR  output=$OUTPUT_DIR  HF_HOME=$HF_HOME"

NO_LANGUAGE_FLAG=""
if [ "$USE_LANGUAGE" != "1" ]; then
    NO_LANGUAGE_FLAG="--no-language"
fi

# ---------- run -------------------------------------------------------------
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
