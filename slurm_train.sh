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

# Activate venv (created once: python3.12 -m venv "$REPO_ROOT/.venv")
source "$REPO_ROOT/.venv/bin/activate"

# Performance settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# expandable_segments helps PyTorch reclaim fragmented blocks instead of OOMing
# when the asked-for chunk doesn't fit in any single free segment. The previous
# max_split_size_mb=128 was tuned for a different cluster and made things worse
# here (LLaVA OOM'd with 4.6 MiB free while 40 MiB sat reserved but unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- ablation-cell selectors (one job per cell) ----------------------
#   BACKBONE: llava | clip          (frozen backbone)
#   USE_LANGUAGE: 1 | 0             (0 -> text prompt zeroed in the encoder)
#   TASK_FILTER (optional): substring of trajectory_task_<task>_*. If unset,
#       the cache is built on the union of every trajectory_task_* dir found
#       under DATA_DIR (the standard combined-dataset ablation cell).
BACKBONE="${BACKBONE:-llava}"
USE_LANGUAGE="${USE_LANGUAGE:-1}"
TASK_FILTER="${TASK_FILTER:-}"        # empty = combined cache across both tasks

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CACHE_BATCH_SIZE_DEFAULT=64
# Pre-2026-05-29 the LLaVA path was capped at 16 because `output_hidden_states=True`
# in VLAAgent.encode retained all 32 transformer layers (~10 GB) and batch=32/64
# OOM'd on the A5000 24 GB. The encode hook + FlashAttention-2 fixed that; batch=64
# now fits comfortably (~16 GB peak) and can be pushed higher if probing leaves room.
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-$CACHE_BATCH_SIZE_DEFAULT}"
LR="${LR:-1e-3}"
PAST_ACTION_K="${PAST_ACTION_K:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"

LANG_TAG=$([ "$USE_LANGUAGE" = "1" ] && echo lang || echo nolang)
TASK_TAG=${TASK_FILTER:-combined}

# ---------- storage layout --------------------------------------------------
# /var/tmp1 is the local NVMe on Abaki nodes (1 TB on 1N, 2 TB on 2N).
# It wipes on weekly reboot; feature_cache.precompute is resumable so a
# partial cache rebuild is the worst case.
NODE_SCRATCH="/var/tmp1/$USER"
mkdir -p "$NODE_SCRATCH"

DATA_TARBALL_DIR="$REPO_ROOT/trajectories"            # persistent tarballs
DATA_DIR="$NODE_SCRATCH/trajectories"                 # extracted, per node
CACHE_DIR="$NODE_SCRATCH/caches"                      # survives across jobs
OUTPUT_DIR="$REPO_ROOT/output/${BACKBONE}_${TASK_TAG}_${LANG_TAG}"

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
# When TASK_FILTER is empty we need BOTH tarballs staged for the combined cache.
if [ -n "$TASK_FILTER" ]; then
    STAGE_TASKS=("$TASK_FILTER")
else
    STAGE_TASKS=()
    for f in "$DATA_TARBALL_DIR"/trajectory_task_*.tar.gz; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .tar.gz)            # trajectory_task_<task>_length_3000
        task=${name#trajectory_task_}
        task=${task%_length_*}
        STAGE_TASKS+=("$task")
    done
    if [ ${#STAGE_TASKS[@]} -eq 0 ]; then
        echo "ERROR: no trajectory_task_*.tar.gz under $DATA_TARBALL_DIR" >&2
        exit 1
    fi
fi

NEED_CONSOLIDATE=0
for TASK in "${STAGE_TASKS[@]}"; do
    TRAJ_SUBDIR="$DATA_DIR/trajectory_task_${TASK}_length_3000"
    if [ ! -f "$TRAJ_SUBDIR/all_actions.json" ]; then
        TARBALL=$(ls "$DATA_TARBALL_DIR"/trajectory_task_${TASK}_*.tar.gz 2>/dev/null | head -1 || true)
        if [ -z "$TARBALL" ]; then
            echo "ERROR: no tarball matching trajectory_task_${TASK}_*.tar.gz under $DATA_TARBALL_DIR" >&2
            exit 1
        fi
        if [ ! -d "$TRAJ_SUBDIR" ]; then
            # The tarballs are inconsistent: chop_a_tree wraps content in
            # AAA_trajectory/trajectory_task_<task>_length_3000/{actions,infos,videos},
            # while collect_dirt wraps directly as AAA_trajectory/{actions,infos,videos}
            # (no per-task subdir). Extract to a TASK-specific staging dir and
            # then move the contents up into $TRAJ_SUBDIR regardless of which
            # layout the tarball used. This silently-different layout silently
            # produced a chop-only "combined" cache once — never again.
            STAGE_TMP="$DATA_DIR/.stage_${TASK}_$$"
            rm -rf "$STAGE_TMP"
            mkdir -p "$STAGE_TMP"
            echo "Extracting $TARBALL -> $STAGE_TMP (strip AAA_trajectory/ wrapper)"
            tar -xzf "$TARBALL" -C "$STAGE_TMP" --strip-components=1
            mkdir -p "$TRAJ_SUBDIR"
            inner_dir="$STAGE_TMP/trajectory_task_${TASK}_length_3000"
            if [ -d "$inner_dir" ]; then
                # chop-style: tarball already nested the task subdir
                mv "$inner_dir"/* "$TRAJ_SUBDIR"/
            else
                # dirt-style: tarball wrote actions/infos/videos at top of strip
                mv "$STAGE_TMP"/* "$TRAJ_SUBDIR"/
            fi
            rm -rf "$STAGE_TMP"
        else
            echo "Already extracted: $TRAJ_SUBDIR (missing consolidated JSON only)"
        fi
        NEED_CONSOLIDATE=1
    else
        echo "Trajectory + consolidated JSON already staged: $TRAJ_SUBDIR"
    fi
done

if [ "$NEED_CONSOLIDATE" = "1" ]; then
    echo "Consolidating per-video JSONLs -> all_actions.json + all_infos.json"
    python consolidate_metadata.py --data-dir "$DATA_DIR" --delete-originals
fi

# ---------- post-stage content validation (fail-loud) -----------------------
# A previous run silently trained on chop-only data because abakus22's
# /var/tmp1 already had `all_actions.json` for collect_dirt but its `videos/`
# directory was empty after a partial extract. The `[ ! -f all_actions.json ]`
# skip above let the script proceed; feature_cache.enumerate_samples then
# silently produced 3.2 M samples instead of 6.24 M. Validate every staged
# task has BOTH a non-empty all_actions.json AND at least one video file.
for TASK in "${STAGE_TASKS[@]}"; do
    TRAJ_SUBDIR="$DATA_DIR/trajectory_task_${TASK}_length_3000"
    ACTIONS_JSON="$TRAJ_SUBDIR/all_actions.json"
    if [ ! -s "$ACTIONS_JSON" ]; then
        echo "ERROR: $ACTIONS_JSON missing or empty after staging — refusing to train on partial data" >&2
        exit 1
    fi
    n_stems=$(python -c "import json; print(len(json.load(open('$ACTIONS_JSON'))))")
    n_videos=$(find "$TRAJ_SUBDIR/videos" -maxdepth 1 -name 'video_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    echo "Stage check  $TASK: stems=$n_stems videos=$n_videos"
    if [ "$n_stems" -eq 0 ] || [ "$n_videos" -eq 0 ]; then
        echo "ERROR: trajectory_task_${TASK} has stems=$n_stems videos=$n_videos — refusing to train on empty data" >&2
        exit 1
    fi
    # Soft check: stems and videos should be roughly equal; off by more than
    # 5% means a partial extraction. Fail loud to force investigation.
    if [ "$n_videos" -lt "$(( n_stems * 95 / 100 ))" ]; then
        echo "ERROR: trajectory_task_${TASK} videos ($n_videos) < 95% of stems ($n_stems) — partial data, refusing" >&2
        exit 1
    fi
done

echo "Cell: backbone=$BACKBONE  task_filter=${TASK_FILTER:-<combined>}  use_language=$USE_LANGUAGE"
echo "Past-action K=$PAST_ACTION_K  chunk=$CHUNK_SIZE  epochs=$EPOCHS  batch=$BATCH_SIZE  lr=$LR"
echo "Data=$DATA_DIR  cache=$CACHE_DIR  output=$OUTPUT_DIR  HF_HOME=$HF_HOME"

EXTRA_FLAGS=()
if [ "$USE_LANGUAGE" != "1" ]; then
    EXTRA_FLAGS+=("--no-language")
fi
if [ -n "$TASK_FILTER" ]; then
    EXTRA_FLAGS+=("--task-filter" "$TASK_FILTER")
fi

# ---------- run -------------------------------------------------------------
python cluster_pipeline.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --past-action-k "$PAST_ACTION_K" \
    --chunk-size "$CHUNK_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --cache-batch-size "$CACHE_BATCH_SIZE" \
    --lr "$LR" \
    --device cuda \
    --num-workers 8 \
    "${EXTRA_FLAGS[@]}"

echo "Finished: $(date)"
