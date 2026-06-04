#!/bin/bash
#SBATCH --job-name=clip_nomove
#SBATCH --partition=NvidiaAll
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/clip_%j.out
#SBATCH --error=logs/clip_%j.err
#
# No-move-fix smoke test on a FREE NvidiaAll node (RTX 2060 SUPER 8GB, normal QOS).
# CLIP backbone (fits 8GB), global-pool (cacheable), single-task chop, with the
# class-balanced loss + history-dropout from no_move_fix.md. Cache + model are
# written to NFS home so they survive the node wipe AND are downloadable.
#
# Env knobs:  TASK_FILTER (default chop_a_tree), USE_LANGUAGE (1), FRAME_STRIDE (8),
#             WEIGHTED_LOSS (1), HISTORY_DROPOUT (0.5), EPOCHS (15)
set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ($(date)) ==="
REPO_ROOT="${REPO_ROOT:-$HOME/BIG}"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_FILTER="${TASK_FILTER:-chop_a_tree}"
USE_LANGUAGE="${USE_LANGUAGE:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"          # stride 8 -> ~400k frames, ~2h on a 2060
WEIGHTED_LOSS="${WEIGHTED_LOSS:-1}"
HISTORY_DROPOUT="${HISTORY_DROPOUT:-0.5}"
EPOCHS="${EPOCHS:-15}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}" # CLIP is light; 64 fits 8GB easily
BATCH_SIZE="${BATCH_SIZE:-256}"

LANG_TAG=$([ "$USE_LANGUAGE" = "1" ] && echo lang || echo nolang)

# NvidiaAll nodes expose /var/tmp (not /var/tmp1) as node-local scratch; it wipes
# on reboot. Videos are only needed during the cache build, so stage them here.
NODE_SCRATCH="/var/tmp/$USER"
DATA_DIR="$NODE_SCRATCH/trajectories"
export HF_HOME="$NODE_SCRATCH/hf_cache"
# Cache + output go to NFS home so they PERSIST and can be downloaded afterwards.
CACHE_DIR="$REPO_ROOT/caches"
OUTPUT_DIR="$REPO_ROOT/output/clip_${TASK_FILTER}_${LANG_TAG}_nomove_stride${FRAME_STRIDE}"
mkdir -p "$DATA_DIR" "$CACHE_DIR" "$OUTPUT_DIR" "$HF_HOME" logs

# ---------- stage chop tarball on node-local scratch ------------------------
TRAJ_SUBDIR="$DATA_DIR/trajectory_task_${TASK_FILTER}_length_3000"
if [ ! -f "$TRAJ_SUBDIR/all_actions.json" ]; then
    TARBALL=$(ls "$REPO_ROOT"/trajectories/trajectory_task_${TASK_FILTER}_*.tar.gz 2>/dev/null | head -1)
    [ -z "$TARBALL" ] && { echo "ERROR: no tarball for $TASK_FILTER" >&2; exit 1; }
    STAGE_TMP="$DATA_DIR/.stage_$$"; rm -rf "$STAGE_TMP"; mkdir -p "$STAGE_TMP"
    echo "Extracting $TARBALL -> $STAGE_TMP (strip AAA_trajectory/ wrapper)"
    tar -xzf "$TARBALL" -C "$STAGE_TMP" --strip-components=1
    mkdir -p "$TRAJ_SUBDIR"
    inner="$STAGE_TMP/trajectory_task_${TASK_FILTER}_length_3000"
    if [ -d "$inner" ]; then mv "$inner"/* "$TRAJ_SUBDIR"/; else mv "$STAGE_TMP"/* "$TRAJ_SUBDIR"/; fi
    rm -rf "$STAGE_TMP"
    echo "Consolidating per-video JSONLs -> all_actions.json"
    python consolidate_metadata.py --data-dir "$DATA_DIR"
fi
n_stems=$(python -c "import json; print(len(json.load(open('$TRAJ_SUBDIR/all_actions.json'))))")
n_videos=$(find "$TRAJ_SUBDIR/videos" -maxdepth 1 -name 'video_*.mp4' | wc -l | tr -d ' ')
echo "Staged $TASK_FILTER: stems=$n_stems videos=$n_videos"
[ "$n_videos" -eq 0 ] && { echo "ERROR: no videos staged" >&2; exit 1; }

EXTRA=()
[ "$USE_LANGUAGE" != "1" ] && EXTRA+=("--no-language")
[ "$WEIGHTED_LOSS" = "1" ] && EXTRA+=("--weighted-loss")

echo "Cell: clip $TASK_FILTER $LANG_TAG  stride=$FRAME_STRIDE  weighted=$WEIGHTED_LOSS  hdrop=$HISTORY_DROPOUT"
echo "Cache(persistent)=$CACHE_DIR  Output=$OUTPUT_DIR"

python cluster_pipeline.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone clip \
    --task-filter "$TASK_FILTER" \
    --frame-stride "$FRAME_STRIDE" \
    --past-action-k 8 --chunk-size 8 \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --cache-batch-size "$CACHE_BATCH_SIZE" \
    --lr 1e-3 --device cuda --num-workers 8 \
    --history-dropout "$HISTORY_DROPOUT" \
    "${EXTRA[@]}"

echo "=== Finished $(date). Cache + model under $CACHE_DIR / $OUTPUT_DIR ==="
