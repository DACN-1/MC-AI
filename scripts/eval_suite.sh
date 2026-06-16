#!/usr/bin/env bash
# Behavioural eval suite for one VLA model checkpoint.
#
# Runs the same model in 4 prompt-conditions with identical env seeds across
# every condition, so per-step behaviour can be compared directly:
#
#   A_chop_nocap      env=MineRLChopATree640Fast-v0  prompt=""
#   B_chop_ood        env=MineRLChopATree640Fast-v0  prompt="Play Minecraft."
#   C_chop_task       env=MineRLChopATree640Fast-v0  prompt="chop a tree"
#   D_dirt_task       env=MineRLCollectDirt640Fast-v0  prompt="collect dirt"
#
# Each condition: 10 episodes × 1000 steps. Episode i uses seed BASE_SEED+i,
# identical across conditions and across models — so two models' steps_*.json
# can be diffed step-by-step on the SAME env state.
#
# Usage:
#   bash scripts/eval_suite.sh <model_path> [model_tag] [base_seed]
#
# Example:
#   bash scripts/eval_suite.sh output/clip_combined_lang_stride4/model.pt lang 0
#
# Outputs land at output/eval/<model_tag>/{A_chop_nocap,B_chop_ood,C_chop_task,D_dirt_task}/.
# Run scripts/eval_compare.py afterwards to aggregate.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <model_path> [model_tag] [base_seed]" >&2
    exit 1
fi

MODEL_PATH="$1"
MODEL_TAG="${2:-$(basename "$(dirname "$MODEL_PATH")")}"
BASE_SEED="${3:-0}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SERVER_PORT="${SERVER_PORT:-8765}"
DEVICE="${DEVICE:-mps}"
# Each suite-run lives under output/evaluation/<endtime>_<modeltag>/. While
# the run is in flight the dir is .running_<starttime>_<modeltag>/; on a
# successful exit it is atomically renamed to <endtime>_<modeltag>/. Override
# EVAL_ROOT_BASE to put runs elsewhere, or EVAL_ROOT to use a fixed dir
# (skips the running/final rename — useful for resuming a partially-interrupted
# suite). Legacy: setting EVAL_ROOT=output/eval/<tag> reproduces the old layout.
# Default: exploratory runs land in evaluations/test/. For the canonical 2x2
# paper cells, pass EVAL_ROOT_BASE=evaluations/paper explicitly.
EVAL_ROOT_BASE="${EVAL_ROOT_BASE:-evaluations/test}"
START_TS="$(date +%Y%m%d_%H%M%S)"
if [ -z "${EVAL_ROOT:-}" ]; then
    RUNNING_DIR="$EVAL_ROOT_BASE/.running_${START_TS}_${MODEL_TAG}"
    EVAL_ROOT="$RUNNING_DIR"
    RENAME_ON_EXIT=1
else
    RENAME_ON_EXIT=0
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$EVAL_ROOT"

# ---------- (re)start inference server with this model ---------------------
echo "=== Restarting inference server for $MODEL_TAG ==="
if lsof -t -i:"$SERVER_PORT" >/dev/null 2>&1; then
    kill "$(lsof -t -i:"$SERVER_PORT")" || true
    sleep 3
fi
mkdir -p logs
nohup python inference_server.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --port "$SERVER_PORT" \
    > "logs/inference_${MODEL_TAG}.log" 2>&1 &
SERVER_PID=$!
echo "Server PID $SERVER_PID, log logs/inference_${MODEL_TAG}.log"
# Wait for "Serving on" to appear
for _ in $(seq 1 60); do
    if grep -q "Serving on" "logs/inference_${MODEL_TAG}.log" 2>/dev/null; then
        break
    fi
    sleep 1
done
if ! grep -q "Serving on" "logs/inference_${MODEL_TAG}.log" 2>/dev/null; then
    echo "ERROR: server did not come up; check logs/inference_${MODEL_TAG}.log" >&2
    exit 1
fi
echo "Server ready."
echo

# ---------- common rollout knobs -------------------------------------------
COMMON_FLAGS=(
    --remote-agent host.docker.internal:"$SERVER_PORT"
    --episodes "$EPISODES"
    --max-steps "$MAX_STEPS"
    --sample
    --temperature 1.0
    --camera-temperature 2.0
    --color-match auto
    --seed "$BASE_SEED"
    --record-video
)

run_condition() {
    local cond_name="$1"
    local env_id="$2"
    local prompt="$3"
    local out_dir="$EVAL_ROOT/$cond_name"
    if [ -d "$out_dir" ] && ls "$out_dir"/episode_*.mp4 >/dev/null 2>&1; then
        echo "  -> $cond_name: already has videos; skipping (rm -rf to redo)"
        return
    fi
    rm -f logs/minerl_watchers/*.pid 2>/dev/null || true
    echo "  -> $cond_name on $env_id  prompt=$(printf %q "$prompt")"
    docker compose run --rm --remove-orphans minerl run_rollout.py \
        "${COMMON_FLAGS[@]}" \
        --env "$env_id" \
        --prompt "$prompt" \
        --output-dir "/workspace/$out_dir"
}

echo "=== Eval suite for $MODEL_TAG (base_seed=$BASE_SEED, $EPISODES ep × $MAX_STEPS steps) ==="
echo "Running dir: $EVAL_ROOT"
run_condition A_chop_nocap MineRLChopATree640Fast-v0   ""
run_condition B_chop_ood   MineRLChopATree640Fast-v0   "Play Minecraft."
run_condition C_chop_task  MineRLChopATree640Fast-v0   "chop a tree"
run_condition D_dirt_task  MineRLCollectDirt640Fast-v0 "collect dirt"

# ---------- manifest + rename on success ----------------------------------
END_TS="$(date +%Y%m%d_%H%M%S)"
START_EPOCH="$(date -j -f %Y%m%d_%H%M%S "$START_TS" +%s 2>/dev/null || date -d "${START_TS:0:4}-${START_TS:4:2}-${START_TS:6:2} ${START_TS:9:2}:${START_TS:11:2}:${START_TS:13:2}" +%s 2>/dev/null || echo 0)"
END_EPOCH="$(date +%s)"
DURATION_S=$(( END_EPOCH - START_EPOCH ))
cat > "$EVAL_ROOT/manifest.json" <<EOF
{
  "model_tag": "$MODEL_TAG",
  "model_path": "$MODEL_PATH",
  "model_sha1": "$(shasum -a 1 "$MODEL_PATH" | awk '{print $1}')",
  "base_seed": $BASE_SEED,
  "episodes": $EPISODES,
  "max_steps": $MAX_STEPS,
  "device": "$DEVICE",
  "start_time": "$START_TS",
  "end_time": "$END_TS",
  "duration_seconds": $DURATION_S,
  "conditions": [
    {"tag": "A_chop_nocap", "env": "MineRLChopATree640Fast-v0",   "prompt": ""},
    {"tag": "B_chop_ood",   "env": "MineRLChopATree640Fast-v0",   "prompt": "Play Minecraft."},
    {"tag": "C_chop_task",  "env": "MineRLChopATree640Fast-v0",   "prompt": "chop a tree"},
    {"tag": "D_dirt_task",  "env": "MineRLCollectDirt640Fast-v0", "prompt": "collect dirt"}
  ],
  "decode": {
    "sample": true,
    "temperature": 1.0,
    "camera_temperature": 2.0,
    "color_match": "auto"
  }
}
EOF

if [ "$RENAME_ON_EXIT" = "1" ]; then
    FINAL_DIR="$EVAL_ROOT_BASE/${END_TS}_${MODEL_TAG}"
    mv "$EVAL_ROOT" "$FINAL_DIR"
    EVAL_ROOT="$FINAL_DIR"
fi

echo
echo "Done in ${DURATION_S}s. Outputs in $EVAL_ROOT/{A_chop_nocap,B_chop_ood,C_chop_task,D_dirt_task}/"
echo "Inference server still running (PID $SERVER_PID). Kill with: kill \$(lsof -t -i:$SERVER_PORT)"
