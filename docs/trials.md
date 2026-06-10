# Trials — head-only recipe sweeps

## Camera-axis sweep (2026-06-09/10) — CLOSED, negative result

Motivated by the chop-tasks-at-reward-0 problem: HANDOFF flagged "camera-aware
training fix" as the next axis. Base recipe is `sc` = `slot30_chop3` (SOTA).
All retrains on the same `clip_combined_lang_stride4` cache, 10 epochs,
HIDDEN_DIM=2048. Quick screens were 3 eps (C4/C7); wloss cells got the full
4-condition suite; C2 is decode-only (no retrain).

| ID | Recipe | Knobs | C_chop_task | D_dirt_task (mean / max) |
|----|--------|-------|-------------|--------------------------|
| —  | `slot30_chop3` (SOTA ref) | — | 0 | **4.60 / 8** |
| C1 | forced +20° pitch env | env variant, no retrain | n/a | n/a (apples only) |
| C2 | `sc` + `--camera-temperature 4.0` | decode only | 0 | 2.60 / 4 |
| C3 | `wloss` alone | `WEIGHTED_LOSS=1` | 0 | 0.00 / 0 |
| C4 | `sc_camCE2x` | `CAM_CE_WEIGHT=2.0` | 0 | 3.33 / 9 (high variance) |
| C5 | `sc_wloss` | sc + `WEIGHTED_LOSS=1` | 0 | 0.00 / 0 |
| C7 | `sc_cwloss` | sc + `CAM_WEIGHTED_LOSS=1` | 0 | 1.67 / 3 |

Checkpoints: `output/clip_combined_lang_stride4_{wloss,slot30_chop3_wloss,slot30_chop3_camCE2x,slot30_chop3_cwloss,cwloss}/`.
Eval runs: `output/quick_C4_sc_camCE2x/`, `output/quick_C7_sc_cwloss/`,
`output/c2_slot30_chop3_camtemp4/`, `output/evaluation/20260609_232536_wrap_wloss/`,
`output/evaluation/20260610_003348_wrap_slot30_chop3_wloss/`.

**Verdict:** the lang baseline's camera prediction is already near its loss
floor (camera mae 0.46°, effectively the demo distribution's limit). Every
camera-loss-side intervention either disrupts the binary policy (the wloss
family collapses dirt to 0) or shifts variance up without improving the mean
(camCE×2). No camera intervention helped chop. The unsolved chop tasks are
NOT a "wrong camera prediction" problem — the per-step camera prediction is
right, but the *use of camera over time* during chop attempts is wrong.
That's a temporal/behavioral problem, not a loss-weighting one. **Do not
spend more compute on camera-loss recipes.** Next axis: temporal decode
mechanisms (chunk ensembling, open-loop execution, attack hysteresis).

New knobs that landed with this sweep (all default-off / legacy-identical):
`--cam-weighted-loss` (camera CE class weights without binary pos_weight),
`--cam-ce-weight` (scales the camera CE term; default 0.5 = historical), and
the matching `WEIGHTED_LOSS` / `CAM_WEIGHTED_LOSS` / `CAM_CE_WEIGHT` env vars
in `slurm_train_nvidiaall.sh`.

# Head-only recipe sweep (2026-06-08/09)

Comprehensive log of every recipe tested in the 2026-06-08/09 head-only sweep
on top of the `clip_combined_lang_stride4` cache (combined chop+dirt, stride 4,
HIDDEN_DIM=2048, past_action_k=8, chunk_size=8). All models share the same
backbone (frozen CLIP ViT-L/14 + text encoder), the same dataset, and the same
default training hyperparameters except where noted.

## Recipe summary

| # | Tag | Description | Train F1 winner | Wrapper env eval | Notes |
|---|---|---|---|---|---|
|   | `lang` | Baseline (no extra knobs) | — | dirt **0.00** / 0 | Original SOTA from 2026-06-06 |
| 0 | `decode_thr_lang` | Decode tweak only: `--binary-thresholds attack=0.005` (greedy) | n/a | reward=0 (3 ep) | Agent attacks air, no movement |
| 0 | `decode_bias_lang` | Decode tweak only: `--binary-logit-bias forward=1.5 sprint=1.0 left=0.5 right=0.5` (sample) | n/a | reward=0 (2 ep) | Forward up, attack collapses |
| 1 | `r1` | Frame-weight×5, min_run=60, history_dropout=0.5, learnable_bce_temp | (none vs lang) | pre-wrapper, reward=0 | 66 % active frames → over-corrected to sneak/sustained-attack |
| 2 | `r2` | Like r1 but `min_run=180` | (none) | pre-wrapper, reward=0 | Worse — forward/sprint collapsed even further |
| 3a | `r3a_noTemp` | r2 minus `learnable_bce_temp` | back +.060 | pre-wrapper, reward=0 | |
| 3b | `r3b_lowDrop` | r2 with `history_dropout=0.2` | right +.082, jump +.061 | pre-wrapper, reward=0 | F1 close to slot30, never wrapper-eval'd |
| 3c | `r3c_minrun300` | r2 with `min_run=300` | (modest) | pre-wrapper, reward=0 | |
| 4 | `chop3` | `CHOP_OVERSAMPLE_WEIGHT=3.0` only | jump +.088 | not env-eval'd standalone | Modest gains; subsumed by compound |
| 5 | `ep20` | `EPOCHS=20` only | right +.117, jump +.128 | **dirt 1.60 / 5** | Surprising — more epochs DOES help alone |
| 6 | `decode_bias_lang` | (see #0 above) | n/a | reward=0 | Free; no retrain |
| 7 | `hd4096` | `HIDDEN_DIM=4096` only | (modest, some regressions) | not env-eval'd | Capacity isn't the bottleneck |
| 8a | `slot30` | `PAST_ACTION_SLOT_DROPOUT=0.3` only | right +.095, jump +.139, back +.072 | **dirt 3.20 / 7** | Cleanest single-knob recipe |
| 8b | `slot50` | `PAST_ACTION_SLOT_DROPOUT=0.5` | back +.105, ~slot30 elsewhere | not env-eval'd | Wash vs slot30; don't go higher |
| 8c | `focal2` | `FOCAL_GAMMA=2.0` only | left +.059, jump +.080 | pre-wrapper, reward=0 | BROKE policy — hotbar/inventory fire 14–33 % at inference |
| ★ | **`slot30_chop3`** | slot30 + chop3 compound | **left +.088, back +.084**, jump +.128 | **dirt 4.60 / 8** ← **SOTA** | Synergistic on both F1 and reward |
| ✗ | `slot30_chop3_ep20` | SOTA + `EPOCHS=20` | forward +.010 vs SOTA, left −.037, back −.056 | **dirt 0.00 / 0** | Over-fits — DO NOT add epochs to compound |

## Final dirt-task reward leaderboard (with InventoryRewardWrapper, 5 eps × 1000 steps, seeds 0–4)

| Recipe | Mean | Max | Per-ep |
|---|---|---|---|
| lang baseline | 0.00 | 0 | [0, 0, 0, 0, 0] |
| ep20 | 1.60 | 5 | [0, 2, 1, 0, 5] |
| slot30 | 3.20 | 7 | [2, 3, 0, 7, 4] |
| **slot30_chop3** ← SOTA | **4.60** | **8** | [3, 6, 8, 2, 4] |
| slot30_chop3_ep20 | 0.00 | 0 | [0, 0, 0, 0, 0] |

All chop tasks (A_chop_nocap / B_chop_ood / C_chop_task) → reward=0 across every
recipe. The pitch policy never aims at trunks; orthogonal to the head-recipe
sweep and likely needs an aim-aware training fix or the forced-pitch env
variant for fair scoring.

## Per-action training F1 (full table)

Validation set F1 for the binary action head (camera CE in a separate column,
omitted here). 9 action columns (attack/forward/left/right/sprint/jump/back/
sneak/use); the dropped columns (hotbar.1-9, drop, inventory, ESC) sit near 0
for all models except `focal2` (which broke them upward).

| tag | bin_acc | attack | forward | left | right | sprint | jump | back | sneak | use |
|---|---|---|---|---|---|---|---|---|---|---|
| lang | 0.9825 | 0.966 | 0.599 | 0.377 | 0.164 | 0.567 | 0.296 | 0.258 | 0.608 | 0.016 |
| chop3 | 0.9826 | 0.966 | 0.600 | 0.421 | 0.153 | 0.594 | 0.384 | 0.251 | 0.586 | 0.002 |
| ep20 | 0.9825 | 0.965 | 0.627 | 0.362 | **0.281** | 0.576 | 0.425 | 0.255 | 0.618 | 0.010 |
| focal2 | 0.9819 | 0.964 | 0.605 | 0.436 | 0.175 | 0.578 | 0.376 | 0.266 | 0.563 | **0.025** |
| hd4096 | 0.9824 | 0.964 | 0.601 | 0.394 | 0.142 | 0.588 | 0.361 | 0.261 | 0.552 | 0.008 |
| r1 | 0.9820 | 0.964 | 0.574 | 0.327 | 0.129 | 0.576 | 0.350 | 0.326 | 0.600 | 0.017 |
| r2 | 0.9822 | 0.965 | 0.587 | 0.328 | 0.153 | 0.558 | 0.336 | 0.266 | 0.585 | 0.018 |
| r3a_noTemp | 0.9821 | 0.964 | 0.584 | 0.399 | 0.156 | 0.592 | 0.308 | 0.319 | 0.612 | 0.041 |
| r3b_lowDrop | 0.9824 | 0.965 | 0.618 | 0.422 | 0.246 | 0.600 | 0.357 | 0.294 | 0.590 | 0.036 |
| r3c_minrun300 | 0.9821 | 0.965 | 0.571 | 0.334 | 0.175 | 0.543 | 0.343 | 0.302 | 0.621 | 0.022 |
| slot30 | 0.9824 | 0.965 | 0.616 | 0.413 | 0.259 | 0.614 | **0.435** | 0.330 | 0.605 | **0.052** |
| **slot30_chop3** | 0.9823 | 0.965 | 0.602 | **0.465** | 0.236 | 0.588 | 0.425 | **0.342** | **0.628** | 0.038 |
| slot30_chop3_ep20 | 0.9824 | **0.966** | 0.611 | 0.428 | 0.220 | 0.604 | 0.418 | 0.286 | 0.613 | 0.026 |
| slot50 | 0.9822 | 0.965 | 0.609 | 0.430 | 0.256 | 0.605 | 0.433 | **0.363** | 0.617 | 0.038 |

Bold = per-column max across the sweep.

## Cluster commands

Reproduce on LMU NvidiaAll (replace `RECIPE_TAG` to keep multiple cells side
by side; output lands at `~/BIG/output/clip_combined_lang_stride4_<TAG>/`):

```bash
# SOTA — slot30 + chop3 compound, 10 epochs
ssh -i ~/.ssh/id_lmu cuencanieto@remote.cip.ifi.lmu.de \
  'cd ~/BIG && BACKBONE=clip USE_LANGUAGE=1 FRAME_STRIDE=4 HIDDEN_DIM=2048 \
   TASK_FILTER="" PAST_ACTION_SLOT_DROPOUT=0.3 CHOP_OVERSAMPLE_WEIGHT=3.0 \
   RECIPE_TAG=slot30_chop3 sbatch slurm_train_nvidiaall.sh'

# slot30 alone
BACKBONE=clip USE_LANGUAGE=1 ... PAST_ACTION_SLOT_DROPOUT=0.3 RECIPE_TAG=slot30 ...

# Compound + 20 epochs — DON'T DO THIS, regresses to 0 reward
BACKBONE=clip USE_LANGUAGE=1 ... PAST_ACTION_SLOT_DROPOUT=0.3 CHOP_OVERSAMPLE_WEIGHT=3.0 \
  EPOCHS=20 RECIPE_TAG=slot30_chop3_ep20 ...
```

Trained on RTX 2060 SUPER (8 GB), ~14 min per 10-epoch run, ~17 min for the
20-epoch run.

## Local eval

```bash
# Restart inference server with target ckpt
kill $(lsof -t -i:8765) 2>/dev/null
python inference_server.py --model-path output/clip_combined_lang_stride4_slot30_chop3/model.pt \
  --device mps --port 8765 > logs/inference_slot30_chop3.log 2>&1 &

# Eval suite (4 conditions × 5 eps; uses InventoryRewardWrapper for real reward)
EPISODES=5 bash scripts/eval_suite.sh \
  output/clip_combined_lang_stride4_slot30_chop3/model.pt slot30_chop3 0

# Compare against the rest
python scripts/eval_compare.py \
  --models wrap_lang wrap_slot30 wrap_slot30_chop3 wrap_ep20 \
  --csv output/evaluation/wrapper_comparison.csv \
  | tee output/evaluation/wrapper_comparison.txt
```

## Critical caveat

**Pre-2026-06-09 env eval results (rows marked "pre-wrapper, reward=0")** had
Malmo's `RewardForPossessingItem` silently failing — they may have collected
items that simply weren't logged. To get a true reward number, re-run with
`run_rollout.py` post-commit `59897b8` (the `InventoryRewardWrapper`). The top
candidates with positive F1 deltas (`r3b_lowDrop`, `chop3`, `slot50`,
`hd4096`) are the obvious candidates to re-eval if a future session has Mac
time.

## Other notes

- The `InventoryRewardWrapper` patch added `inventory` snapshot logging to each
  `steps_*.json` entry — pre-existing rollouts can also be scored post-hoc
  using `scripts/inventory_reward.py`, no re-run needed.
- `slurm_train_nvidiaall.sh` exposes new env vars: `FOCAL_GAMMA`,
  `PAST_ACTION_SLOT_DROPOUT`, `CHOP_OVERSAMPLE_WEIGHT`. All default to "off",
  so existing recipes (R1/R2) keep producing the same output.
- Camera-side recipes (different bin spacing, camera-aware loss weighting,
  forced pitch start) were NOT tested this session. UPDATE 2026-06-10: they
  were tested in the follow-up camera-axis sweep (top of this file) and the
  axis is now closed — negative result.
