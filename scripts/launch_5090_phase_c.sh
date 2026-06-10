#!/usr/bin/env bash
# Run this script ON the rented vast.ai RTX 5090 box. It does the full Phase C
# LLaVA-stride-4 cache build end-to-end:
#
#   1. Verify the environment (GPU, torch with sm_120 support, decord).
#   2. Verify trajectory data was staged at /workspace/trajectories (see
#      scripts/push_data_to_5090.sh for the cluster-side push command).
#   3. Run scripts/probe_5090_throughput.py for ~5 min to measure actual
#      samples/sec on real data.
#   4. Build llava_combined_lang_stride4.npy (cache + head train).
#   5. Build llava_combined_nolang_stride4.npy (cache + head train).
#   6. Print a one-line "fetch outputs back" command for the local side.
#
# Outputs land in /workspace/output/llava_combined_{lang,nolang}_stride4/
# (model.pt + metrics.json) and /workspace/caches/ (the two cache .npy/.json
# pairs). Pull them off the box with:
#   rsync -avz <vast_ssh>:/workspace/{caches,output}/ ~/BIG/
#
# Env knobs (all optional):
#   REPO_ROOT     repo on the box (default: /workspace/r1-va)
#   DATA_DIR      pre-extracted trajectories (default: /workspace/trajectories)
#   CACHE_DIR     where the .npy/.json land (default: /workspace/caches)
#   OUTPUT_BASE   per-cell parent (default: /workspace/output)
#   CACHE_BATCH_SIZE  starts at 32 per the 5090 sdpa+bf16 recipe
#   EPOCHS        head training epochs (default: 10)
#   SKIP_PROBE    set to 1 to skip the probe (e.g. on resume)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/r1-va}"
DATA_DIR="${DATA_DIR:-/workspace/trajectories}"
CACHE_DIR="${CACHE_DIR:-/workspace/caches}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/output}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
PAST_ACTION_K="${PAST_ACTION_K:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
LR="${LR:-1e-3}"

echo "=== Phase C LLaVA stride-4 launch on $(hostname) ==="
echo "Started: $(date)"

cd "$REPO_ROOT"

# ---------- environment ----------------------------------------------------
echo
echo "=== Step 1: environment ==="
bash scripts/setup_vastai_5090.sh

# ---------- data check (do not silently proceed) ---------------------------
echo
echo "=== Step 2: data check ==="
mkdir -p "$CACHE_DIR" "$OUTPUT_BASE"
for task in chop_a_tree collect_dirt; do
    traj="$DATA_DIR/trajectory_task_${task}_length_3000"
    if [ ! -s "$traj/all_actions.json" ]; then
        cat >&2 <<EOF
ERROR: $traj/all_actions.json missing or empty.

Stage the trajectories first. From the LMU cluster:
  bash scripts/push_data_to_5090.sh <vast_ssh_host> <vast_ssh_port>

Or rsync directly:
  rsync -avz --progress \\
      cuencanieto@remote.cip.ifi.lmu.de:~/BIG/trajectories/ \\
      $DATA_DIR/
EOF
        exit 1
    fi
    n_stems=$(python -c "import json,sys; print(len(json.load(open('$traj/all_actions.json'))))")
    n_videos=$(find "$traj/videos" -maxdepth 1 -name 'video_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $task: stems=$n_stems videos=$n_videos"
    if [ "$n_videos" -lt "$(( n_stems * 95 / 100 ))" ]; then
        echo "ERROR: $task partial staging (videos<95% of stems) — refusing to build" >&2
        exit 1
    fi
done

# ---------- throughput probe -----------------------------------------------
if [ "${SKIP_PROBE:-0}" != "1" ]; then
    echo
    echo "=== Step 3: throughput probe ==="
    python scripts/probe_5090_throughput.py \
        --data-dir "$DATA_DIR" \
        --backbone llava --use-language \
        --batch-size "$CACHE_BATCH_SIZE" \
        --num-frames 480 --price 0.513 \
    || { echo "Probe failed — fix before committing to a multi-hour build." >&2; exit 1; }
fi

# ---------- cache build + head train per cell -----------------------------
run_cell() {
    local use_language="$1"
    local tag
    tag="$([ "$use_language" = 1 ] && echo lang || echo nolang)"
    local output_dir="$OUTPUT_BASE/llava_combined_${tag}_stride4"
    local extra=()
    [ "$use_language" = 0 ] && extra+=("--no-language")
    mkdir -p "$output_dir"
    echo
    echo "=== Step 4/5: building llava_combined_${tag}_stride4 ==="
    echo "  output -> $output_dir"
    python cluster_pipeline.py \
        --data-dir "$DATA_DIR" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$output_dir" \
        --backbone llava \
        --past-action-k "$PAST_ACTION_K" \
        --chunk-size "$CHUNK_SIZE" \
        --hidden-dim "$HIDDEN_DIM" \
        --epochs "$EPOCHS" \
        --batch-size 256 \
        --cache-batch-size "$CACHE_BATCH_SIZE" \
        --frame-stride 4 \
        --lr "$LR" \
        --device cuda \
        --num-workers 8 \
        "${extra[@]}"
}

run_cell 1
run_cell 0

# ---------- summary --------------------------------------------------------
echo
echo "=== Phase C LLaVA stride-4 done ==="
echo "Finished: $(date)"
echo
echo "Outputs on this box:"
ls -lh "$CACHE_DIR"/llava_combined_*_stride4.* 2>&1 | head
echo
for tag in lang nolang; do
    ls -lh "$OUTPUT_BASE/llava_combined_${tag}_stride4/" 2>&1 | head
done
echo
echo "Pull everything back to the LMU cluster (run on the cluster login node):"
cat <<EOF
  rsync -avz --progress \\
      <vast_ssh_host>:/workspace/caches/llava_combined_*_stride4.* \\
      ~/BIG/caches/
  rsync -avz --progress \\
      <vast_ssh_host>:/workspace/output/llava_combined_*_stride4/ \\
      ~/BIG/output/
EOF
