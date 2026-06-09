#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --partition=NvidiaAll
#SBATCH --qos=normal
# Avoid andesit: it has only 37 GB free in /tmp (other NvidiaAll nodes have ~90 GB).
#SBATCH --exclude=andesit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# NvidiaAll variant of slurm_train.sh — used when the Abaki QOS isn't granted to
# stud_ifi. NvidiaAll nodes are RTX 2060 SUPER 8 GB (vs A5000 24 GB on Abaki) and
# have no /var/tmp1 scratch. The trade-off:
#   - trajectory data must be PRE-EXTRACTED under ~/BIG/trajectories/ before
#     submitting (the tarball→/var/tmp1 staging block in slurm_train.sh assumed
#     1 TB local NVMe; NvidiaAll's /tmp is only ~90 GB and ephemeral).
#   - caches live on ~/BIG/caches/ directly (NFS). CLIP forward is GPU-bound
#     so the NFS write rate is not the bottleneck.
#   - HF cache + TMPDIR-style scratch use /tmp/$USER on the node.

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Started: $(date)"

REPO_ROOT="${REPO_ROOT:-$HOME/BIG}"
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- ablation-cell selectors ----------------------------------------
BACKBONE="${BACKBONE:-clip}"
USE_LANGUAGE="${USE_LANGUAGE:-1}"
TASK_FILTER="${TASK_FILTER:-}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
# CLIP-ViT-B/32 forward at batch=64 fits comfortably in 8 GB. Smaller GPU than
# Abaki's A5000 24 GB; LLaVA on these nodes would OOM at any batch — use Abaki
# (or another big-GPU partition) for LLaVA cache builds.
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
PAST_ACTION_K="${PAST_ACTION_K:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
# Round 1 recipe knobs (all cache-safe; only DataLoader/loss-side changes):
#   FRAME_WEIGHT_MULTIPLIER   WeightedRandomSampler boost for sustained-attack
#                             frames. 1.0 = uniform; 5.0 typical.
#   FRAME_WEIGHT_MIN_RUN      Minimum sustained-attack run length (ticks).
#   HISTORY_DROPOUT           Probability of zeroing past-action vector per
#                             sample during training. 0.0 = off; 0.5 typical.
#   LEARNABLE_BCE_TEMP=1      Add per-binary-action learnable temperature.
#   RECIPE_TAG                Suffix appended to OUTPUT_DIR/cell name; lets
#                             multiple recipes coexist without overwriting.
FRAME_WEIGHT_MULTIPLIER="${FRAME_WEIGHT_MULTIPLIER:-1.0}"
FRAME_WEIGHT_MIN_RUN="${FRAME_WEIGHT_MIN_RUN:-60}"
HISTORY_DROPOUT="${HISTORY_DROPOUT:-0.0}"
LEARNABLE_BCE_TEMP="${LEARNABLE_BCE_TEMP:-0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"
PAST_ACTION_SLOT_DROPOUT="${PAST_ACTION_SLOT_DROPOUT:-0.0}"
CHOP_OVERSAMPLE_WEIGHT="${CHOP_OVERSAMPLE_WEIGHT:-1.0}"
RECIPE_TAG="${RECIPE_TAG:-}"

LANG_TAG=$([ "$USE_LANGUAGE" = "1" ] && echo lang || echo nolang)
TASK_TAG=${TASK_FILTER:-combined}
STRIDE_TAG=$([ "$FRAME_STRIDE" -gt 1 ] && echo "_stride${FRAME_STRIDE}" || echo "")
RECIPE_SUFFIX=$([ -n "$RECIPE_TAG" ] && echo "_${RECIPE_TAG}" || echo "")

# ---------- storage layout -------------------------------------------------
NODE_SCRATCH="/tmp/$USER"
mkdir -p "$NODE_SCRATCH"

DATA_DIR="$REPO_ROOT/trajectories"                   # pre-extracted on BIG
CACHE_DIR="$REPO_ROOT/caches"                        # persist on BIG
OUTPUT_DIR="$REPO_ROOT/output/${BACKBONE}_${TASK_TAG}_${LANG_TAG}${STRIDE_TAG}${RECIPE_SUFFIX}"

export HF_HOME="$NODE_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$HF_HOME"

# ---------- data validation (no extraction; data must be pre-staged) -------
if [ -n "$TASK_FILTER" ]; then
    STAGE_TASKS=("$TASK_FILTER")
else
    STAGE_TASKS=()
    for d in "$DATA_DIR"/trajectory_task_*_length_*; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        task=${name#trajectory_task_}
        task=${task%_length_*}
        STAGE_TASKS+=("$task")
    done
    if [ ${#STAGE_TASKS[@]} -eq 0 ]; then
        echo "ERROR: no trajectory_task_*_length_* under $DATA_DIR — extract tarballs first" >&2
        exit 1
    fi
fi

for TASK in "${STAGE_TASKS[@]}"; do
    TRAJ_SUBDIR="$DATA_DIR/trajectory_task_${TASK}_length_3000"
    ACTIONS_JSON="$TRAJ_SUBDIR/all_actions.json"
    if [ ! -s "$ACTIONS_JSON" ]; then
        echo "ERROR: $ACTIONS_JSON missing or empty — extract + consolidate first" >&2
        exit 1
    fi
    n_stems=$(python -c "import json; print(len(json.load(open('$ACTIONS_JSON'))))")
    n_videos=$(find "$TRAJ_SUBDIR/videos" -maxdepth 1 -name 'video_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    echo "Stage check  $TASK: stems=$n_stems videos=$n_videos"
    if [ "$n_stems" -eq 0 ] || [ "$n_videos" -eq 0 ]; then
        echo "ERROR: trajectory_task_${TASK} has stems=$n_stems videos=$n_videos" >&2
        exit 1
    fi
    if [ "$n_videos" -lt "$(( n_stems * 95 / 100 ))" ]; then
        echo "ERROR: trajectory_task_${TASK} videos ($n_videos) < 95% of stems ($n_stems)" >&2
        exit 1
    fi
done

echo "Cell: backbone=$BACKBONE  task_filter=${TASK_FILTER:-<combined>}  use_language=$USE_LANGUAGE  stride=$FRAME_STRIDE  hidden_dim=$HIDDEN_DIM"
echo "Data=$DATA_DIR  cache=$CACHE_DIR  output=$OUTPUT_DIR  HF_HOME=$HF_HOME"

EXTRA_FLAGS=()
if [ "$USE_LANGUAGE" != "1" ]; then
    EXTRA_FLAGS+=("--no-language")
fi
if [ -n "$TASK_FILTER" ]; then
    EXTRA_FLAGS+=("--task-filter" "$TASK_FILTER")
fi
# Round 1 recipe knobs
if [ "$(echo "$FRAME_WEIGHT_MULTIPLIER > 1.0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_FLAGS+=("--frame-weight-multiplier" "$FRAME_WEIGHT_MULTIPLIER"
                  "--frame-weight-min-run" "$FRAME_WEIGHT_MIN_RUN")
fi
if [ "$(echo "$HISTORY_DROPOUT > 0.0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_FLAGS+=("--history-dropout" "$HISTORY_DROPOUT")
fi
if [ "$LEARNABLE_BCE_TEMP" = "1" ]; then
    EXTRA_FLAGS+=("--learnable-bce-temp")
fi
if [ "$(echo "$FOCAL_GAMMA > 0.0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_FLAGS+=("--focal-gamma" "$FOCAL_GAMMA")
fi
if [ "$(echo "$PAST_ACTION_SLOT_DROPOUT > 0.0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_FLAGS+=("--past-action-slot-dropout" "$PAST_ACTION_SLOT_DROPOUT")
fi
if [ "$(echo "$CHOP_OVERSAMPLE_WEIGHT != 1.0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_FLAGS+=("--chop-oversample-weight" "$CHOP_OVERSAMPLE_WEIGHT")
fi

# ---------- run ------------------------------------------------------------
python cluster_pipeline.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --past-action-k "$PAST_ACTION_K" \
    --chunk-size "$CHUNK_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --cache-batch-size "$CACHE_BATCH_SIZE" \
    --frame-stride "$FRAME_STRIDE" \
    --lr "$LR" \
    --device cuda \
    --num-workers 8 \
    "${EXTRA_FLAGS[@]}"

echo "Finished: $(date)"
