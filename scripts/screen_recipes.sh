#!/usr/bin/env bash
# Screen CLIP recipe checkpoints by OBJECTIVE task performance (items collected),
# the reward-independent signal added in eval_logger (RewardForCollectingItems is
# broken: it watches 'log' but chopping yields 'oak_log'/'birch_log').
#
# Runs each checkpoint on the on-task chop + dirt conditions and writes to
# evaluations/test/recipe_sweep_v2/<tag>/. Re-running skips conditions that
# already have videos. Rank afterwards with scripts/rank_item_collection.py.
#
# Usage:
#   EPISODES=8 DEVICE=mps bash scripts/screen_recipes.sh
#   CONDITIONS="A_chop_nocap B_chop_ood C_chop_task D_dirt_task" ... (all 4)
set -euo pipefail

EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-C_chop_task D_dirt_task}"
EVAL_ROOT_BASE="${EVAL_ROOT_BASE:-evaluations/test/recipe_sweep_v2}"
CKPT_DIR="${CKPT_DIR:-output}"

# The 18 recipe-sweep cells dedupe to these 15 unique checkpoints (wrap_* were
# reward-rescoring re-evals of checkpoints already listed). <tag> <ckpt-subdir>.
# The baseline (clip_combined_lang_stride4) is included as the reference point.
MODELS=(
  "baseline            clip_combined_lang_stride4"
  "r1                  clip_combined_lang_stride4_r1"
  "r2                  clip_combined_lang_stride4_r2"
  "focal2              clip_combined_lang_stride4_focal2"
  "slot30              clip_combined_lang_stride4_slot30"
  "r3a_noTemp          clip_combined_lang_stride4_r3a_noTemp"
  "r3b_lowDrop         clip_combined_lang_stride4_r3b_lowDrop"
  "r3c_minrun300       clip_combined_lang_stride4_r3c_minrun300"
  "slot30_chop3        clip_combined_lang_stride4_slot30_chop3"
  "ep20                clip_combined_lang_stride4_ep20"
  "slot50              clip_combined_lang_stride4_slot50"
  "hd4096              clip_combined_lang_stride4_hd4096"
  "slot30_chop3_ep20   clip_combined_lang_stride4_slot30_chop3_ep20"
  "wloss               clip_combined_lang_stride4_wloss"
  "slot30_chop3_wloss  clip_combined_lang_stride4_slot30_chop3_wloss"
)

echo "=== Recipe screen: ${#MODELS[@]} checkpoints x [$CONDITIONS] x $EPISODES ep ==="
echo "Output base: $EVAL_ROOT_BASE   device=$DEVICE"
i=0
for row in "${MODELS[@]}"; do
  i=$((i+1))
  tag="$(echo "$row" | awk '{print $1}')"
  sub="$(echo "$row" | awk '{print $2}')"
  ckpt="$CKPT_DIR/$sub/model.pt"
  echo
  echo "############ [$i/${#MODELS[@]}] $tag  ($ckpt) ############"
  if [ ! -f "$ckpt" ]; then
    echo "  !! missing checkpoint, skipping: $ckpt"
    continue
  fi
  EVAL_ROOT="$EVAL_ROOT_BASE/$tag" \
  CONDITIONS="$CONDITIONS" EPISODES="$EPISODES" MAX_STEPS="$MAX_STEPS" DEVICE="$DEVICE" \
    bash scripts/eval_suite.sh "$ckpt" "$tag" 0
done
echo
echo "=== screen complete -> rank with: python scripts/rank_item_collection.py $EVAL_ROOT_BASE ==="
