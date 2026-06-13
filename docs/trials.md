# Trials — head-only recipe sweeps

## Wave 1–5 sweep + reproducibility controls (2026-06-10/11) — CLOSED; eval variance dominates

27 seeded runs (5 eps × 1000 steps, seeds 0–4, InventoryRewardWrapper) across
four fronts: decode interventions on the SOTA ckpt (Wave 1), wrapper re-evals
of orphaned ckpts (Wave 2), 9 knob retrains on the `clip_combined_lang_stride4`
cache (Waves 3/5, all with the new trajectory-level split + keep-best), three
new inference-time temporal mechanisms (Wave 4: `--chunk-ensemble`,
`--execute-steps`, `--attack-hysteresis`), and three reproducibility controls.
Raw logs: `output/eval_w/<tag>/<condition>/`. Summarize with
`python output/summarize_eval_w.py`.

### Results (dirt mean / max over 5 eps; chop where run)

| Tag | What | C_chop | D_dirt |
|---|---|---|---|
| w11_thr | SOTA + `attack=0.005` (exp2 trick) | 0 | 0.00 |
| w12_thrbias | SOTA + thr + `forward=1.0` bias | 0 | — |
| w13_sampT15 | SOTA + `--sample T=1.5` | 0 | 1.00 / 2 |
| w14_thr25 | SOTA + `attack=0.25` | 0 | — |
| w41_ensemble | SOTA + chunk ensembling | 0 | 0.00 |
| w42_openloop4 | SOTA + execute-k=4 | 0 | — |
| w43_hyst | SOTA + attack hysteresis 40:0.2 | 0 | — |
| w2_r3b / w2_slot50 | orphan ckpt re-evals | 0 / 0 | 0.00 / 0.00 |
| w31_hd02 (+_best) | sc + history_dropout 0.2 | — | 0.00 (best: 0.20 / 1) |
| w32_chop5 | slot30 + chop oversample 5 | — | 0.00 |
| w33_slot20 / w33_slot40 | slot dropout 0.2 / 0.4 + chop3 | — | 0.00 / 0.00 |
| w34_ep14 (+_best) | sc + 14 epochs | — | 0.00 / 0.00 |
| w35_lr5e4_ep20 | sc + LR 5e-4, 20 ep | — | 0.00 |
| w36_chunk4 | sc + chunk_size 4 | — | 0.00 |
| w51_best20 | sc + 20 ep, best-epoch ckpt | — | 0.00 |
| **w52_tsplit** | **sc re-trained, trajectory split** | — | **0.00** |
| **w53_replica** | **sc re-trained, EXACT original recipe (frame split)** | — | **0.00** |
| **w50_sota_rerun** | **the ORIGINAL 4.60 ckpt, re-run unchanged** | — | **0.00** |

### Controls and the verdict

1. **w53_replica = 0.00**: retraining the exact original `slot30_chop3` recipe
   does not reproduce 4.60.
2. **w50_sota_rerun = 0.00**: the *same artifact* that scored 4.60 on
   2026-06-09 scores 0.00 on the same seeds two days later.
3. **Determinism probe**: two back-to-back identical runs (same ckpt, seed,
   stack, container) produce step-0 logits differing by max |Δ| ≈ 0.23 —
   the env render/reset is not bit-deterministic even with `--seed`. Greedy
   decode usually absorbs the jitter, but when a logit sits near threshold an
   early action flips and the episode diverges chaotically.

**Verdict: 5-episode wrapper-reward screens are dominated by run-to-run env
variance, not by recipe quality.** The 4.60 "SOTA", the 3.20 slot30 and the
1.60 ep20 numbers below (2026-06-08/09 leaderboard) were high-variance draws,
not stable recipe effects — treat that entire leaderboard's ranking as
unreliable. Any future reward-based comparison needs either 20–50 episodes
per cell or a denser proxy metric (per-action firing/F1 against demos, attack-
run statistics, inventory-delta event counts) before reward is trusted.

### Chop behavioral analysis (why decode can't fix it)

From `steps_*.json` on the chop runs: under plain greedy the SOTA head **never
asserts attack** in the chop context (sigmoid lives in ~(0.005, 0.5) — 0 % of
75 k steps with thr 0.5, bimodal 0 %/100 % per episode at thr 0.25, 100 % at
thr 0.005 — and at 100 % it attacks air without aiming at trunks). Chunk
ensembling suppresses attack entirely (later chunk steps are systematically
more conservative, so averaging drags marginal logits below threshold) — it
also zeroes dirt. Open-loop execution and hysteresis change nothing because
the underlying logits are context-locked, not jittery. Chop failure is a
policy/perception gap (no tree-seeking, no aim), not a decode problem.

### Infra that landed with this sweep

- `run_rollout.py`: `--chunk-ensemble`, `--execute-steps K`,
  `--attack-hysteresis N[:THR]` (default-off, legacy byte-identical).
- Trainer: trajectory-level val/test split by stem (default; old leaky
  frame-level split behind `--frame-level-split`), per-epoch per-action val
  F1 in history, `--keep-best`/`KEEP_BEST=1` → `model_best.pt` by movement
  F1. Under the honest split, val movement-F1 is nearly flat across 1–20
  epochs (~0.43–0.47) for every recipe — training length barely matters.
- Cluster data note: `~/BIG/trajectories/` videos were deleted; head-only
  retrains need only the restored `all_actions.json` per task (see
  slurm_train_nvidiaall.sh stage-check fix).

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

## Frame-history windows (2026-06-11) — CLOSED, negative

The last untried post-cache lever: concatenate the K previous cached frame
FEATURES (stride-spaced, `--frame-history-k`, commit 8158f38) to give the
head visual motion context. Three cells on `clip_combined_lang_stride4`,
identical knobs otherwise (trajectory split, keep-best, 10 ep):

| Cell | Head input | Best val move-F1 (epoch) | Final val_loss |
|---|---|---|---|
| `tsplit_base` (K=0) | 1,536 | **0.4554** (3) | **0.6048** |
| `fhist4` (K=4) | 7,680 | 0.4402 (7) | 0.6119 |
| `fhist8` (K=8) | 13,824 | 0.4305 (7) | 0.6204 |

Monotonic negative dose-response on both F1 and loss. Interpretation:
stacked *global pooled* vectors add no usable motion signal — the global
gist barely changes across stride-4 frames, so the extra dims are mostly
redundant input to fit. Consistent with the representational thesis: the
information loss happens at pooling, and no head-side input arrangement of
pooled features recovers it. **Post-cache levers on pooled features are now
exhausted on every axis tried** (loss weighting, sampling, dropout family,
epochs/LR/capacity, decode calibration, temporal decode, visual history).
The open bet is the spatial patch cache (`_patch4`, building).

Note: `tsplit_base` (0.4554) is also the missing CLIP-lang baseline anchor
under the honest split — slot30/chop3-family cells (0.43–0.47) do NOT beat
it; the recipe knobs were within-noise on F1 too.

## Demo aiming analysis (2026-06-13) — the signal IS there, the loss drowns it

Reframes the whole chop investigation. After decode/recipe/temporal/
representation all failed, the open question was whether the demos contain a
learnable "aim at the trunk" signal at all. `scripts/analyze_chop_aiming.py`
aligns camera to sustained-attack-run onsets (300 stems, 5,365 onsets,
median chop run 4.5 s):

```
              aiming window         onset       locked-on chop
 rel frame:  -15   -9   -3     [attack starts]     +9   +15
 |camera|:   2.2   2.5   3.3        ↓ settles ↓     1.2   0.9
```

- Pre-onset mean |camera| = **2.67°**, during-chop = **1.37°** — a camera
  burst peaking ~3 frames before the agent commits, settling once locked on.
- **Directional, not noise:** pre-onset pitch signed −0.446 == abs 0.446 —
  the demonstrator consistently pitches DOWN (toward the trunk base) before
  chopping. Yaw weaker but also directional (+0.155 == 0.155).

**Conclusion flip:** the "aim then chop" pattern is clearly present and
learnable. The model fails to learn it because it is RARE and TRANSIENT (a
~5-frame burst per 4.5 s run) and drowned by the per-frame loss, where 83 %
of frames are attack=1 and 78 % are camera-still. The two representation
negatives (patch, frame-history) are consistent: better *inputs* can't help
when the *loss* never emphasizes the aiming frames. Compounding it: the
aiming pitch (~0.45°/frame) quantizes into the near-zero mu-law bins (4-6),
exactly where the still-camera majority also lives.

**Targeted post-cache fix (untried, next):** upweight the camera CE on the
pre-attack-onset aiming windows specifically — onset-aligned, NOT the global
bin-frequency reweighting `cam_weighted_loss` did (which is why that failed).
Implementable head-only on the existing pooled cache via a per-frame camera-
loss weight = high inside [onset-W, onset] of each sustained run (reuse the
run-finding logic from `compute_task_active_weights`). The aux
"is-chopping-segment" head is the structural variant of the same idea.

## CLIP patch-grid pilot (2026-06-12) — spatial features do NOT beat pooled (flat-MLP head)

The representational bet from the Wave 1–5 verdict: cache a 4×4 grid of
vision-tower patch tokens (`--patch-grid 4` → 16×1024 + 768 text = 17,152
dims, commit 4a0515a) so "where is the trunk" is representable. Pilot cell
`clip_chop_a_tree_lang_stride4_patch4` (797,685 samples, 25.5 GB, ~8.7 h
encode on a 2060) vs an exactly matched pooled control
(`pooled_chop_ctl`: combined pooled cache sliced to the same 797,685 chop
samples via `--head-stem-filter`, same split/knobs/epochs):

| | pooled ctl | patch4 |
|---|---|---|
| best val move-F1 | **0.4979** (ep7) | 0.4577 (ep2) |
| final val_loss | **0.7124** | 0.7254 |
| test cam_mae° | 0.623 | 0.630 |
| cam x/y bin acc | .8258/.8102 | .8232/.8094 |
| left / right F1 | **.410** / .239 | .273 / .203 |
| forward / sprint F1 | .677/.634 | .671/.602 |

Pooled wins or ties on every metric; **camera prediction is bit-identical**
— spatial features moved the aiming proxy not at all. Patch4 also shows the
overfit signature (lower train_loss 0.682 vs 0.700, higher val_loss).

**Interpretation (two readings, both important):**
1. Camera supervision is saturated at the demo floor from global features
   alone — the BC *targets* may simply not contain a learnable "aim at the
   trunk" signal (demonstrator camera is dominated by smooth wander). The
   bottleneck looks like the data/supervision, not the representation.
2. Caveat: this pilot pairs patch features with a FLAT MLP (16× input dims,
   same hidden). A cross-attention head over the grid could in principle use
   spatial structure the MLP dilutes — untested. But given (1), analyze the
   demos for aiming signal BEFORE buying more architecture.

Also validated here: `--head-stem-filter` slices a combined cache to exactly
the single-task composition (797,685 == patch cache sample count).

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
