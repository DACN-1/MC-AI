#!/usr/bin/env bash
# Random-policy sanity-floor baseline for the thesis Method (chapter3.tex:386-389):
# action-frequency / item-collection floor to confirm trained agents diverge from
# chance. Run in the SAME envs and SAME base seed (0) as scripts/eval_suite.sh —
# the only thing that changes is the policy (random) and thus the collected
# reward/inventory. Episode i uses seed 0+i, so each episode lands on the exact
# same world state as every trained cell's episode i.
#
# No inference server (run_rollout.py samples actions directly when no model /
# no --remote-agent is given), so there is no MPS/Docker RAM contention — this
# does NOT thrash like the head evals do.
#
# Random actions are np.random-seeded per episode (run_rollout.py:374), so the
# chop-env conditions A/B/C (identical env + seed, prompt ignored by random)
# would be byte-identical → run once per ENVIRONMENT, not per prompt-condition.
set -uo pipefail
cd /Users/diego/VSCode/r1-va

EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEED="${SEED:-0}"
OUT=evaluations/paper/random_baseline

rm -f logs/minerl_watchers/*.pid 2>/dev/null || true

run_random () {
  local name="$1" env_id="$2"
  local out_dir="$OUT/$name"
  if [ -d "$out_dir" ]; then
    n=$(find "$out_dir" -name 'episode_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -ge "$EPISODES" ]; then echo "SKIP $name ($n/$EPISODES already done)"; return; fi
    [ "$n" -gt 0 ] && { echo "  cleaning incomplete $name ($n/$EPISODES)"; rm -rf "$out_dir"; }
  fi
  rm -f logs/minerl_watchers/*.pid 2>/dev/null || true
  echo "=== RANDOM $name on $env_id (seed $SEED, $EPISODES ep × $MAX_STEPS steps) $(date) ==="
  docker compose run --rm --remove-orphans minerl run_rollout.py \
    --env "$env_id" \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --color-match auto \
    --record-video \
    --output-dir "/workspace/$out_dir" \
    || echo "FAILED $name rc=$?"
}

# chop env serves trained conditions A/B/C; dirt env serves D.
run_random chop MineRLChopATree640Fast-v0
run_random dirt MineRLCollectDirt640Fast-v0
echo "RANDOM_BASELINE_DONE $(date)"
