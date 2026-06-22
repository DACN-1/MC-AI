#!/usr/bin/env bash
# Full CLIP recipe study: 10 cells (5 recipes × {lang,nolang}) evaluated in the
# canonical 4-condition suite (A/B/C/D), 10 ep × 1000 steps, sampled decode.
#
#   - Serves model_best.pt (keep-best epoch) — matches docs/llava_5090_runbook.md
#     so the recipe study stays symmetric with the LLaVA backbone comparison.
#   - DEVICE=mps (Apple Silicon GPU); inference server is (re)started per cell by
#     eval_suite.sh itself.
#   - Fixed per-cell EVAL_ROOT under evaluations/paper/ → resumable at condition
#     granularity (eval_suite skips conditions that already have videos) and the
#     dir names match the existing paper/ convention.
#   - Order: anchor first (conditioning control spine), then the screen's top
#     dirt diggers, so an interruption still leaves the most valuable cells done.
set -uo pipefail
cd /Users/diego/VSCode/r1-va
source .venv/bin/activate

export DEVICE=mps
export EPISODES=10
export MAX_STEPS=1000

rm -f logs/minerl_watchers/*.pid 2>/dev/null || true

# "<output-dir suffix>|<dir/tag stem>" — evaluated lang then nolang per recipe.
ORDER=(
  "tsplit_base|anchor"
  "slot30_chop3_tsplit|slot30chop3"
  "r3c_minrun300_tsplit|r3c"
  "slot50_tsplit|slot50"
  "lr5e4_ep20_tsplit|lr5e4ep20"
)

run_cell () {
  local dir="$1" tag="$2"
  local eroot="evaluations/paper/$tag"
  if [ -f "$eroot/manifest.json" ]; then
    echo "SKIP $tag — manifest.json present (already complete)"
    return
  fi
  local ckpt="output/$dir/model_best.pt"
  if [ ! -f "$ckpt" ]; then
    echo "FAILED $tag — checkpoint missing: $ckpt"
    return
  fi
  # Self-healing resume: eval_suite skips any condition dir that already has >=1
  # video, which would leave a crash-interrupted condition stuck at <EPISODES.
  # Drop incomplete conditions (1..EPISODES-1 videos) so they redo cleanly;
  # complete ones (==EPISODES) are kept and skipped.
  if [ -d "$eroot" ]; then
    for cdir in "$eroot"/[A-D]_*; do
      [ -d "$cdir" ] || continue
      n=$(find "$cdir" -name 'episode_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
      if [ "$n" -gt 0 ] && [ "$n" -lt "$EPISODES" ]; then
        echo "  cleaning incomplete $(basename "$cdir") ($n/$EPISODES videos)"
        rm -rf "$cdir"
      fi
    done
  fi
  echo "=== EVAL $tag  ($dir)  $(date) ==="
  EVAL_ROOT="$eroot" bash scripts/eval_suite.sh "$ckpt" "$tag" 0 \
    || echo "FAILED $tag rc=$?"
}

for cell in "${ORDER[@]}"; do
  suffix="${cell%%|*}"; rtag="${cell##*|}"
  for lang in lang nolang; do
    run_cell "clip_combined_${lang}_stride4_${suffix}" "clip_${lang}_${rtag}"
  done
done

echo "ALL_CLIP_EVALS_DONE $(date)"
