#!/bin/bash
# Box-side native eval for the A+B conditioning-fix LLaVA head (llava_fix.md).
# Mirrors eval_all.sh's do_head: native run_rollout per condition, IDENTICAL
# decode flags so it stays symmetric with the paper cells. Resumable (skips a
# condition whose run_summary.json exists). Outputs to evaluations/paper/llava_lang_fixAB/.
ulimit -n 1048576
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
cd /workspace/r1-va
source /workspace/venv/bin/activate
mkdir -p logs logs/minerl_watchers evaluations/paper
HEAD=output/llava_combined_lang_stride4_fixAB/model_best.pt
TAG=llava_lang_fixAB
ROOT=evaluations/paper
DEC="--episodes 10 --max-steps 1000 --sample --temperature 1.0 --camera-temperature 2.0 --color-match auto --seed 0"
rc(){ local cond="$1" env="$2" prompt="$3"; local out="$ROOT/$TAG/$cond"
  [ -f "$out/run_summary.json" ] && { echo "SKIP $TAG/$cond"; return 0; }
  echo "RUN $TAG/$cond $(date +%F_%H:%M:%S)"
  local att; for att in 1 2 3; do
    rm -f logs/minerl_watchers/*.pid 2>/dev/null
    xvfb-run -a python -u run_rollout.py --model-path "$HEAD" --device cuda \
      --env "$env" --prompt "$prompt" $DEC --output-dir "$out" \
      >> "logs/eval_${TAG}_${cond}.log" 2>&1
    [ -f "$out/run_summary.json" ] && { echo "DONE $TAG/$cond $(date +%H:%M:%S)"; return 0; }
    echo "RETRY $TAG/$cond ($att)"; sleep 10
  done; echo "FAILED $TAG/$cond"; return 1; }
echo "=== fixAB eval START $(date) ==="
rc A_chop_nocap MineRLChopATree640Fast-v0 ""
rc B_chop_ood   MineRLChopATree640Fast-v0 "Play Minecraft."
rc C_chop_task  MineRLChopATree640Fast-v0 "chop a tree"
rc D_dirt_task  MineRLCollectDirt640Fast-v0 "collect dirt"
echo "=== fixAB eval DONE $(date) ==="
