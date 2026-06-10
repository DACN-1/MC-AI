> **SUPERSEDED — 2026-06-08.** This is an archived handoff. The current handoff
> is `docs/HANDOFF.md`. Specific technical details below (Phase C plan, recipe
> arguments) are still referenced from code comments (e.g. `VLAAgent.py`,
> `slurm_train.sh`, `cluster_pipeline.py`) and remain accurate, but the
> proposed timeline / pending tasks are stale.

---

# Agent handoff — Phase C in code, awaiting cluster rebuild

**Status as of 2026-06-05 PM:**
- Phase A1 (inference-side calibration knobs): commit `d534cca`.
- Env edits for usable reward (Fast variants, force-aim variant):
  commits `6a9012b`, `a2ba0ff`.
- **Phase C** (LLaVA split image/text pooling + `--hidden-dim` plumbing):
  commit `1a9966d`. **Code is landed; the cluster work — LLaVA cache rebuild
  at 8192 dim, head retrain at `hidden_dim=2048` — is what remains for next
  session.**

What's next: kick off the slurm jobs (`HIDDEN_DIM=2048 sbatch slurm_train.sh`
per cell), wait for caches (~26 h LLaVA / ~6 h CLIP per tag), train heads
(~30 min each), then aligned 4-cell rollouts on the Fast env.

## TL;DR for the receiving agent

The user has a 2×2 ablation (LLaVA/CLIP × prompt/no-prompt) that is currently
not producing reward in rollout. `docs/alignment_handoff.md` argues the cause
is encoder asymmetry between `VLAAgent.encode` and `frozen_vision_baseline.encode`
and proposes a 3-step fix. We audited the handoff and the actual rollout data
and concluded **the handoff alone is necessary but not sufficient.** Three
distinct failure modes show up in `output/**/run_summary.json`; the handoff
addresses only one. Recommended sequencing is now:

  **A** (camera collapse, cheap, inference-side) →
  **B** (recipe over-correction, ~30 min retrains) →
  **C** (encoder alignment per the handoff, ~5 h cache rebuild) →
  **D** (combined-dataset 4-cell paper-grade run)

The full plan is in `/Users/diego/.claude/plans/tender-fluttering-candy.md`.
Read it before doing anything.

## What just landed (commit d534cca)

`feat: decoding-side calibration knobs, incl. --camera-temperature`

- `action_mapping.py:map_to_minerl_action` now takes a `camera_temperature:
  Optional[float]` kwarg. When set under `sample=True`, it overrides the
  scalar `temperature` for the camera softmax (both axes) only. `None` is
  byte-identical to before.
- `run_rollout.py` exposes `--camera-temperature T` and surfaces it in the
  run banner.
- `tests/test_action_conversion.py` adds two tests: hot
  `camera_temperature=10` flattens the sampled bin distribution, and `=None`
  preserves legacy behavior. All 35 tests pass.

The commit also bundles the prior uncommitted no_move_fix follow-on knobs
(`--binary-temperatures`, `--binary-thresholds`, `--binary-logit-bias`) that
were live in the working tree but unstaged. They were already documented in
`docs/alignment_handoff.md` line 219 and are needed for Phase B.

## CORRECTION (2026-06-05 PM): camera was never collapsed on CLIP

The earlier framing of "camera collapsed across all cells" was a schema-
reading error. `EpisodeLogger.action_frequency` only logs camera when NO
binary fired — so with attack/forward firing 30-58 % of the time, camera
got undercounted as 0 %. Recomputing from per-step `minerl_action.camera`
in `steps_*.json`:

| cell | steps | true cam_x nz | true cam_y nz | x bins | y bins |
|---|---:|---:|---:|---:|---:|
| LLaVA chop greedy | 300 | **0.0%** | **0.0%** | **1/11** | **1/11** |
| CLIP nomove greedy | 4000 | 97.1% | 97.2% | 9/11 | 9/11 |
| CLIP fixed greedy | 2000 | 86.2% | 93.8% | 11/11 | 11/11 |
| CLIP indist sample | 600 | 50.0% | 47.8% | 11/11 | 11/11 |
| **CLIP exp2_thr greedy+thr_attack=0.005** | 5000 | 60.0% | 59.9% | 4/11 | 7/11 |

**Only LLaVA truly has camera collapse.** That's part of the same flat-logit
encoder problem the handoff's Step 2 targets; not a separate failure mode
on CLIP.

Reward is also unreliable — per the user, `rollout_logs/exp2_thr/episode_004.mp4`
visibly shows the agent chopping a tree, but `total_reward=0`. Don't trust
reward as the quality signal; use behavior (per-action firing rates, camera
coverage, video review).

## Strongest baseline to date: exp2_thr (greedy + threshold-attack=0.005)

`rollout_logs/exp2_thr/`: 5 ep × 1000 steps each = 5000 steps total.
Per-action firing rates from `steps_*.json`:

  attack 100% · forward 82.1% · jump 78.1% · left 59.5% · sprint 49.7% ·
  camera_x nz 60.0% · camera_y nz 59.9%

Agent sustains attack every single step, sprint-jumps forward-left across
terrain, makes fine camera adjustments (std 0.5°, mostly bins 5-6). User
reports ep004 visibly chopping a tree. The recipe-side `pos_weight[attack]
=0.05` over-correction is functionally solved at inference by dropping the
attack threshold from 0.5 to 0.005, which lets the suppressed sigmoid
(~0.025) trigger every step.

The `exp3_bias` (`--binary-logit-bias attack=+6.0`) recipe in the handoff
table is the additive-shift analogue and probably equivalent in effect.

## Updated three failure-mode picture

1. **LLaVA encoder collapse (real).** Camera + binaries all flat.
   The handoff's Step 2 (split image/text pooling + bigger feature_dim) is
   the right fix.
2. **CLIP `no_move_fix` attack collapse (already solved at inference).**
   Greedy + `--binary-thresholds attack=0.005` recovers task-relevant
   behavior. Whether to *also* retrain with a milder `pos_weight[attack]` is
   an open call.
3. ~~Camera collapse on all cells~~ — was a schema misread on my part.
   CLIP cameras have always been moving. The `--camera-temperature` knob
   committed in `d534cca` is still useful for sample-mode tuning but
   isn't the bottleneck.
4. **(NEW) Reward signal was unreachable on the default env.** Bare-handed
   log mining at vanilla Minecraft break-speed is ~3 s of sustained attack
   on the same block (~60 ticks @ 20 tps) + the agent has to *then* walk
   over the dropped log for `RewardForCollectingItems` to fire. That
   compound event almost never lands in a 1000-step rollout, so
   `total_reward = 0` everywhere regardless of behavior quality. **Fix:**
   added `break_speed_multiplier` param to `_Combined640EnvSpec` and two
   new Fast env IDs:
   - `MineRLChopATree640Fast-v0`
   - `MineRLCollectDirt640Fast-v0`

   Both are identical to their parent envs except `BreakSpeedMultiplier=5.0`
   (drops log break to ~0.6 s). Visual training distribution is unaffected
   (no axe, no HUD change — only the engine's per-tick damage rate scales).
   The Fast variants are what should be used for any rollout where reward
   needs to be a usable signal. The vanilla envs stay registered so prior
   rollouts remain comparable. See `eval_envs.py:81-130, 152-180`.

## What does the env change enable

- **Scoreable evaluation.** Reward now fires reliably when the agent
  completes a chop. The 4-cell ablation (Phase D) gains a quantitative
  metric — mean reward per cell — instead of relying on per-video review.
- **Hyperparameter sweeps become tractable.** Pick `--binary-thresholds
  attack`, `--camera-temperature`, etc. by reward instead of inspecting
  videos. Each sweep point is a ~2-min Fast rollout.
- **Phase B becomes empirically answerable.** Does relaxing
  `pos_weight[attack]` from 0.05 to 0.3 in retraining outperform the
  inference-side `--binary-thresholds attack=0.005` fix on the same model?
  Compare reward; pick the winner.
- **exp2_thr's status gets quantified.** Today it's "the strongest
  baseline by video review". On the Fast env it'll be either "task-
  completing baseline" (reward > 0) or "task-attempting but still failing
  the pickup step" (reward = 0 despite breaks happening). Both are
  actionable.

What it does *not* change:
- LLaVA still has encoder collapse. Phase C is still the substantive work.
- Vanilla break-speed semantics are no longer the default eval surface —
  any paper-grade comparison should disclose `break_speed_multiplier=5.0`
  and ideally report both env variants.

## Phase C status (commit `1a9966d`)

**Landed in code:**

- `VLAAgent.encode` now returns `[image_pool || text_pool]` of dim 8192 for
  LLaVA-1.5-7B (was 4096 mean-pool over all tokens). `_split_pool` helper
  handles the post-expansion role mapping: T_out − T_in + 1 = N image tokens,
  image positions [img_pos, img_pos+N), text everywhere else attention-valid
  and non-image. Tested on synthetic inputs (3 unit tests pass without LLaVA).
- `use_language=False` zeros text_pool at the head input — matches CLIP's
  zero-text-features. Encoder still sees the prompt text (LLaVA cross-modal
  attention can't be turned off architecturally), but the head input is
  image-only.
- `cluster_pipeline.py` exposes `--hidden-dim` (default None → falls back
  to `feature_dim`, preserving legacy). `slurm_train.sh` adds `HIDDEN_DIM`
  env var defaulting to 2048 and passes it as `--hidden-dim`. Same pinned
  head width across all 4 ablation cells.
- `agent_loader.py:54` heuristic tightened from `feature_dim < 4096` to
  `feature_dim < 2048` so it doesn't break on the new 8192-dim LLaVA caches.
- `scripts/probe_5090_throughput.py:43` updated to `EXPECTED_FEATURE_DIM["llava"]
  = 8192`.

**Still pending (cluster work):**

1. **Rebuild LLaVA caches.** The 4096→8192 feature_dim change forces
   `feature_cache.precompute` to detect a metadata mismatch and rebuild from
   scratch (`feature_cache.py:258` check). One pass per (LLaVA × lang_flag ×
   task) tag; ~26 h per LLaVA tag on A5000. Per the combined-dataset rule
   (memory: `feedback_combined_dataset_strict`), use `TASK_FILTER=""` for the
   real ablation caches; chop-only is fine for validation passes only.
2. **Retrain heads.** Once caches are rebuilt, run
   `HIDDEN_DIM=2048 BACKBONE=llava USE_LANGUAGE={0,1} sbatch slurm_train.sh`
   for the LLaVA cells. CLIP caches don't need rebuild; CLIP heads should also
   be retrained at `HIDDEN_DIM=2048` for symmetry (~30 min each on cached
   features). Total: 4 head trains × ~30 min = ~2 h.
3. **Verify integration tests on the cluster.** Before kicking off the cache
   rebuild, run `R1VA_RUN_LLAVA_INTEGRATION=1 python -m unittest
   tests.test_encode_equivalence` on a cluster node with LLaVA cached. Three
   tests cover (a) hook == output_hidden_states[-1], (b) the model's image-
   token expansion count matches `config.image_seq_length`, (c) shape +
   use_language=False text-half zero. Catch any transformers-version regressions
   before they cost a 26 h rebuild.
4. **Aligned 4-cell rollout on Fast env.** Decode at `--sample --temperature
   1.0 --camera-temperature 2.0` (or whatever you settle on as the unified
   inference recipe) on `MineRLChopATree640Fast-v0`. Score by per-action TV
   distance against contractor distribution from `all_actions.json`. Phase C
   "done" criterion: LLaVA cells become indistinguishable from CLIP cells in
   action distribution, OR if they don't, the finding is genuine ("LLaVA-as-
   encoder is worse for this BC task" is a publishable result).

## Symmetry constraint (load-bearing — read before proposing changes)

The 2×2 ablation design controls exactly two variables across cells:
**backbone** (LLaVA vs CLIP, rows) and **language access** (prompt vs
no-prompt, columns). Per the original spec: *"all four cells use
identical training hyperparameters, dataset, and action head architecture."*

Any change proposed to "improve CLIP" must also be applied identically to
LLaVA, and vice versa. The same recipe knobs, head capacity, sampling
weights, inference flags, env, and decode settings must hold across all
four cells. Otherwise the pairwise comparisons (Exp 1 vs Exp 3 etc.) stop
answering the questions the design was built to answer.

The exception is the encoder code path itself — `VLAAgent.encode` and
`frozen_vision_baseline.encode` necessarily differ because the backbones
are different. The original handoff's Phase C2 (split image/text pooling
for LLaVA) is *asymmetric on purpose*: it brings LLaVA's `encode()` into
behavioral parity with CLIP's. Once landed, future changes go back to
applying symmetrically.

Subtler architectural caveat that **cannot** be closed by code: LLaVA's
joint cross-modal transformer means image tokens have attended to text
tokens at every layer, so even after split pooling the "image-pool" still
carries text-conditioning. CLIP's dual-encoder design doesn't. This is a
structural property of LLaVA-as-encoder; the paper has to acknowledge it
when interpreting the no-text cells.

## CLIP improvements worth trying (retraining is free) — applies symmetrically to LLaVA

All proposals below are encoder-agnostic by construction: they touch the
loss, sampler, head capacity, or auxiliary supervision — none of them
modifies anything that's encoder-specific. So "CLIP improvements" is a
misnomer; these are head/recipe improvements that should be run with the
SAME settings on both backbones for any Phase D pass.

User flagged: "after move to C if there is anything that can be done to
better clip's performance? retraining clip is free, that is both the head
and cache". The CLIP cache is ~6 h on A5000 / one tag, the head retrain is
~30 min — both cheap enough to sweep. Diagnostic findings from this
session point to two concrete recipe knobs that should move the needle,
plus a few harder experiments.

### Priority 1 — recipe re-tuning (no code changes, just config sweeps)

The `no_move_fix` recipe weights are too aggressive for chop:

- **`pos_weight[attack]`**: 0.05 collapses attack to sigmoid ≈ 0.025 (the
  exp2_thr `--binary-thresholds attack=0.005` workaround proves the logit
  ordering is fine, the calibration is broken). Sweep 0.05 → 0.1 → 0.3 →
  0.5 → 1.0. Pick the value that produces ~50–80% greedy attack rate on a
  short Fast rollout without the inference-side threshold knob.
- **`cam_weight` on look-down bins**: currently `cam_weight[0°]=0.10`
  (deprioritize bin 5). Add `cam_weight[bin0..3] = 2.0–3.0` so the model
  is actively incentivized to predict look-down. Today's rollout has zero
  mass on bins 0-3 (-10° to -1.6°) — the demo prior wins by default.

These are 2-line changes in whatever computes the loss weights. The
combination should produce a model that doesn't need the `--binary-
thresholds attack=0.005` inference hack AND aims down on its own. Sweep
on the Fast env, score by mean reward across 3 episodes × 1500 steps.

### Priority 2 — frame-weighted sampling for task-active frames

Contractor frames are mostly "walk + look around"; rare frames are "agent
is actively chopping a tree, inventory increments." The current trainer
samples uniformly, washing out the high-information frames.

- Pre-compute a per-frame weight `w = 1.0 + 4.0 * is_task_active`, where
  `is_task_active` is true for frames within ±30 ticks of an inventory
  increment (or whatever marks the contractor's successful chops).
- Pass weights to the `DataLoader` via `WeightedRandomSampler` or as a
  per-sample loss multiplier.

5× upweighting on task-active frames means the head sees more "look-down +
sustained-attack" examples per epoch. This is the same kind of
recipe-tuning trick that fixed similar issues in published BC work
(e.g. CLIPort's per-frame weighting). Cost: one pass over `all_actions.json`
to compute weights, plus minor changes to the dataset class.

### Priority 3 — wider / deeper head

`HeadOnlyAgent` currently defaults to `hidden_dim = feature_dim = 1536`
for CLIP (1 hidden layer). Try:
- `hidden_dim = 2048` (already plumbed via `train_cached_head`, per
  Step 1 of the original handoff)
- Add a second hidden layer: 1536 → 2048 → 2048 → output

More capacity should help discriminate the "looking at trunk vs
looking at leaves" feature distinction CLIP's pooled image embedding
collapses by default.

### Priority 4 — auxiliary task losses

Train a small auxiliary head off the same encoder that predicts:
- "Is this frame from a chopping segment?" (binary classifier) — labels
  derivable from the inventory-change criterion above
- "Camera target bin in 8 ticks" — forces multi-step planning signal

Both push the encoder pool to encode task-state distinction. Tiny code
change; main risk is auxiliary loss balancing.

### Priority 5 — calibration-aware training

If we want a model that doesn't need any inference-side flags, train with
a temperature on the BCE loss to encourage higher-magnitude logits (so
sigmoid output sits near 0 or 1 rather than near 0.5). Equivalent to
sharpening the logit distribution.

### What I would NOT prioritize for CLIP

- Bigger CLIP backbone (ViT-L vs ViT-B) — minor wins, large cost
- More chunk_size — already at 8, returns diminish
- Adding more BASALT tasks to training — slows training, regularization
  benefit speculative
- Architecture surgery on CLIP itself (modifying `encode()`) — CLIP's
  dual-encoder split is already aligned per the original handoff; the
  bottleneck is the head, not the encoder.

### Sequence

After Phase C (LLaVA alignment) lands:

1. **Quick win pass**: run Priority 1's recipe sweep on the existing
   CLIP cache. 5 retrains × 30 min = ~2.5 h. Score on Fast env.
2. If that gives a CLIP cell with reward > 0 reliably, ship it as the
   "tuned CLIP" baseline and move to Phase D (combined-dataset 4-cell
   ablation) with this recipe.
3. If Priority 1 plateaus below the LLaVA-aligned cell, fall back to
   Priority 2 (frame weighting) — adds ~1 day of dev + 1 retrain.
4. Priorities 3-5 are paper-grade polish; do them if there's time
   budget after the 4-cell ablation has clean numbers.

## Phase A2 RESULT — camera collapse is inference-side

Sweep at `output/clip_chop_a_tree_lang_nomove_stride8/model.pt`, 1 ep ×
300 steps each, `MineRLChopATree640-v0`, `--sample --temperature 1.0
--camera-temperature <T>`. MPS inference server, Docker MineRL env.

| T_cam | reward | attack | forward | back | cam_x nz | cam_y nz | X std | Y std | X bins | Y bins |
|------:|-------:|-------:|--------:|-----:|---------:|---------:|------:|------:|-------:|-------:|
|  1.0  |   0.00 |  67.7% |   4.7%  |24.7% |   95.3%  |   96.3%  | 1.7°  | 3.2°  |  8/11  | 10/11  |
|  1.5  |   0.00 |   2.3% |  97.3%  | 0.0% |   91.3%  |   97.7%  | 1.4°  | 2.5°  |  8/11  | 10/11  |
|  2.0  |   0.00 |  21.3% |  78.0%  | 0.3% |   85.7%  |   91.7%  | 1.8°  | 2.8°  | 10/11  | 11/11  |
|  3.0  |   0.00 |  29.3% |   7.3%  |58.7% |   92.3%  |   90.3%  | 3.4°  | 3.7°  | 11/11  | 11/11  |

**Conclusion: the camera collapse seen in prior rollouts was greedy-decode-
specific.** With `--sample` enabled, camera fires 91–95% even at T_cam=1.0
across 8–10 of 11 bins. T_cam=2.0 gives slightly wider coverage (10/11, 11/11)
without much downside; T_cam=3.0 starts to thrash (std jumps to 3.4°/3.7°).
**Recommended setting going forward: `--sample --camera-temperature 2.0`** for
camera diagnostics; downstream phases can revisit.

**Reward still 0 across the board.** Confirms the plan's prediction: fixing
camera alone is necessary but not sufficient. The attack rate also varies
wildly (2.3% to 67.7%) across the four 1-episode runs, which is sampling
noise on 300 steps but also reflects that the no_move_fix recipe is over-
correcting some seeds into "walk past tree forever." **Phase B (recipe
tuning) is now the bottleneck.**

Per-T_cam step-level logs are at `output/rollout_clip_T_cam_{1.0,1.5,2.0,3.0}/`
(NOT in git — gitignored along with the other rollout dirs).

## Note: MPS inference server still running

`inference_server.py --model-path output/clip_chop_a_tree_lang_nomove_stride8/model.pt
--device mps --port 8765` is still serving in the background as of this
handoff. Kill with `lsof -i :8765` → `kill <PID>` if you're done. Otherwise
you can keep feeding it Phase B rollouts (the per-binary inference flags
`--binary-logit-bias attack=6.0` and `--binary-temperatures attack=5.0`
are already wired, so attack-restoration experiments don't need a retrain).

## What's next: Phase B — recipe over-correction

The natural next experiment is **inference-side attack restoration** to see
if combining `--camera-temperature 2.0` with `--binary-logit-bias attack=6.0`
(per the handoff doc table on the `chop FIX` model) gets reward > 0. This
is free — the MPS server is already loaded and the flag exists.

```bash
docker compose run --rm --remove-orphans minerl run_rollout.py \
  --remote-agent host.docker.internal:8765 \
  --env MineRLChopATree640-v0 \
  --episodes 3 --max-steps 500 \
  --sample --temperature 1.0 \
  --camera-temperature 2.0 \
  --binary-logit-bias attack=6.0 \
  --color-match auto \
  --output-dir /workspace/output/rollout_clip_T2_attack_bias6 \
  --record-video
```

`--record-video` is worth turning on — visual confirmation that the agent
is now looking around AND attacking will say more than action frequencies.
Increase `--episodes` to 3 and `--max-steps` to 500 to give the agent a
realistic chance at the ~5 s sustained-attack reward criterion.

If reward > 0: inference-side calibration is a complete fix; retrain is
unnecessary. Update `feedback_combined_dataset_strict` memory to note the
camera+bias eval recipe, then proceed to Phase D directly (combined-data
4-cell retrain at the same calibration settings).

If reward still 0: the recipe-trained head can't be salvaged at inference;
need to retrain with milder `pos_weight[attack]` (probably ~0.3) per the
plan's B1/B2. The CLIP cache exists; head retraining is ~30 min.

## What's next: Phase A2 — sweep T_cam in MineRL

Pick the cleanest existing ckpt (`output/clip_chop_a_tree_lang_nomove_stride8/model.pt`
— that's the model behind `rollout_clip_indist`, best binary distribution today).
Run a sweep `T_cam ∈ {1.0, 1.5, 2.0, 3.0}` with `--sample` and check whether
non-zero camera fire rates emerge.

### Local Docker path (easiest if it works on this Mac)

```bash
# Sanity-check the MineRL stack first.
docker compose run --remove-orphans minerl test_minerl.py

# Then the sweep — one episode per T_cam to start, 300 steps each.
for T_cam in 1.0 1.5 2.0 3.0; do
  docker compose run --remove-orphans minerl run_rollout.py \
    --model-path output/clip_chop_a_tree_lang_nomove_stride8/model.pt \
    --env MineRLChopATree640-v0 \
    --episodes 1 --max-steps 300 \
    --sample --temperature 1.0 --camera-temperature "$T_cam" \
    --color-match auto \
    --output-dir output/rollout_clip_T_cam_${T_cam}
done
```

Then compare `output/rollout_clip_T_cam_*/run_summary.json` action_frequency
for `camera_x` / `camera_y` (these are fraction of steps where the camera
returned a non-zero degree value after undiscretize — so nonzero means the
sampler picked a bin other than 5).

### Vast.ai inference server path (if local Docker is broken)

`scripts/setup_vastai_inference.sh` provisions an RTX 3090 spot instance
with the minimal stack (torch+cu121, transformers, no flash-attn, no decord).
After provisioning, run `inference_server.py` on the box and SSH-tunnel
8765 → host.docker.internal. Then point `run_rollout.py --remote-agent
host.docker.internal:8765` and use the same loop above. See
`docs/alignment_handoff.md` "Infrastructure notes" for the vast.ai filters.

### Decision rule for A2

- **If camera fire rate stays at ~0% even at T_cam=3.0:** the model truly
  didn't learn to look around — the head puts so much mass on bin 5 that
  flattening can't recover it. Move to A3 (training-side): verify whether
  `cam_weight[0°]=0.10` was actually applied to the saved CLIP ckpts. Read
  `output/clip_chop_a_tree_lang_nomove_stride8/metrics.json` and the ckpt's
  config dict.
- **If camera fire rate rises with T_cam (e.g. 30%+ at T_cam=2.0):** the
  camera collapse is inference-time. Pick the lowest T_cam that produces
  visible re-orientation without making the agent thrash. Note the value;
  carry it forward to Phases B, C, D.

## Critical context the next session needs

### Decisions already made

- **Chop-only is OK for this alignment-validation pass.** The stored
  combined-dataset rule (`feedback_combined_dataset_strict` in memory) still
  binds for the final paper-grade run (Phase D), but the validation chain
  A→B→C runs on chop-only for cost reasons. The memory entry should be
  updated once D is queued.
- **All 4 cells must share the `no_move_fix` recipe** (`weighted_loss=True`,
  `history_dropout=0.5`). The existing CLIP cells have it; the existing
  LLaVA chop cell does not. Phase C will retrain LLaVA with the recipe.

### Asymmetry claims in `docs/alignment_handoff.md` — all three verified

1. **Pooling**: CLIP returns `[image_proj || text_proj]` (1536-d); LLaVA
   mean-pools the full image+text token sequence (4096-d). With ~576 image
   tokens vs ~5 text tokens, text gets washed out. (Verified at
   `frozen_vision_baseline.py:67` and `VLAAgent.py:130`.)
2. **`use_language=False`**: CLIP zeros the text branch; LLaVA only sends an
   empty string (`"<image>\n"`) so text-side hidden states still leak into
   the mean pool. (Verified at `frozen_vision_baseline.py:66` and
   `VLAAgent.py:96`.)
3. **Head capacity** in end-to-end heads: CLIP 1536→768 (2×), LLaVA 4096→4096
   (1×). But cached path (which is what's actually trained) defaults both
   to 1× via `HeadOnlyAgent`'s `hidden_dim or feature_dim` at
   `feature_cache.py:471`. So the asymmetry is real for end-to-end and
   *moot for cached* — Phase C step 1 (pin `--hidden-dim 2048`) is good
   hygiene but weak as a diagnostic.

### Caveat the handoff doesn't state

Even with the proposed split image/text pooling in `VLAAgent.encode`,
LLaVA's joint cross-modal transformer means image tokens have already
attended to text tokens at every layer. After slicing, "image-pool" still
carries text-conditioning. So Phase C narrows the gap to CLIP but does not
close it; the paper has to flag this honestly. The handoff doesn't.

### Things to NOT do

- **Do NOT execute the handoff's Steps 1-3 in order without first doing
  Phase A and Phase B.** The 4-cell ablation alignment is meaningless if
  the resulting models can't perform the task at all (camera frozen +
  attack collapsed).
- **Do NOT rebuild the LLaVA cache at 8192-dim (Phase C2) before chop
  rollouts on aligned CLIP cells are producing reward.** It's ~5 h and ~$1.25
  of rented GPU; verify the recipe + camera fixes work on the cheaper
  backbone first.
- **Do NOT commit `inference_server.py`, `README.md`, `.gitignore`, or the
  staged `*.md → docs/*.md` renames** without checking with the user first.
  Those changes were in the working tree before this session.

### Files to read first

1. `/Users/diego/.claude/plans/tender-fluttering-candy.md` — the full plan
   with all four phases, decisions, and verification steps.
2. `docs/alignment_handoff.md` — the original asymmetry argument (untracked
   file; do not commit).
3. `docs/no_move_fix.md` — context on the recipe that's now blocking attack.
4. `docs/rollouts.md` — the running log of rollout findings; today's session
   should be appended to it once A2 lands.

### Memory updates pending (do these only after Phase A2 lands)

- Add a memory note under `project_` describing the Phase A→B→C→D sequencing
  decision and the rationale (handoff-only is insufficient; camera + recipe
  must move first).
- Add a memory note clarifying that chop-only validation passes are an
  allowed exception to the `feedback_combined_dataset_strict` rule, but
  Phase D must conform.

## Quick verification when picking up

```bash
# Confirm the camera-temperature commit is in HEAD.
git log --oneline -3
# Should show d534cca feat: decoding-side calibration knobs ...

# Confirm tests still pass.
source .venv/bin/activate && python -m unittest tests.test_action_conversion -v 2>&1 | tail -5

# Confirm --camera-temperature is in the parser (gym not required for grep).
grep -A 5 "camera-temperature" run_rollout.py | head -10
```
