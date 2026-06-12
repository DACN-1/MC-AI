# HANDOFF — 2026-06-12 update (canonical)

## TL;DR — 2026-06-12 (representation experiments: both negative; suspect the data)

Two experiments closed overnight (details: docs/trials.md top sections):

1. **Frame-history windows** (`--frame-history-k`): concatenating K previous
   cached frame features is monotonically WORSE (base 0.4554 > K=4 0.4402 >
   K=8 0.4305 val move-F1). Stacked global gists carry no usable motion
   signal.
2. **CLIP 4×4 patch-grid cache** (`--patch-grid 4`, 25.5 GB chop pilot, 8.7 h
   encode): vs an exactly matched pooled control (same 797,685 chop
   samples via the new `--head-stem-filter`), pooled wins or ties on every
   metric and **camera prediction is identical** (mae 0.62° both). Spatial
   features + flat MLP do not move the aiming proxy at all.

**Where this points:** with decode, recipes, temporal context, and now
representation all eliminated, the remaining suspect for chop is the
SUPERVISION itself — the demos' camera targets may contain no learnable
"aim at trunk" signal (smooth demonstrator wander dominates). Next cheap
step: demo-side analysis — around block-break events, measure camera-
movement statistics and attack onsets to quantify how much aiming signal
exists to imitate. If it's absent, no architecture on this dataset fixes
chop, and the paper should say so. (Residual untested branch: a cross-
attention head over the patch grid instead of a flat MLP — only worth it if
the demo analysis finds signal.)

LLaVA next steps are unaffected: docs/llava_5090_runbook.md.

## TL;DR — 2026-06-11 (Wave 1–5 sweep + controls: eval variance dominates)

The full post-cache roadmap (decode sweeps, temporal decode mechanisms,
orphan re-evals, 9 knob retrains, selection infra) ran overnight — 27 seeded
eval runs + 10 cluster trainings. Full table in `docs/trials.md` (top
section). Three findings, in order of importance:

1. **The 4.60 dirt "SOTA" is not reproducible — and not because of the
   recipes.** Exact-recipe retrain → 0.00. The *original artifact itself*
   re-run on the same seeds → 0.00. Back-to-back identical runs differ in
   step-0 logits by ~0.23 (env render/reset is not bit-deterministic), and
   near-threshold logits occasionally flip an early action, after which the
   episode diverges chaotically. **5-ep wrapper-reward screens are noise-
   dominated; the 2026-06-08/09 leaderboard ranking is unreliable.** Use
   20–50 eps per cell or denser proxy metrics before trusting reward.
2. **Chop is a policy/perception gap, not decode.** The head never asserts
   attack in chop contexts (sigmoid in ~(0.005, 0.5) for 75 k steps); forcing
   it (thr 0.005) gives 100 % attack-of-air. Chunk ensembling suppresses
   attack (far-horizon chunk steps are systematically conservative);
   hysteresis/open-loop change nothing. All 7 chop interventions: reward 0.
3. **Selection infra landed**: trajectory-level val split (honest; default),
   per-epoch movement-F1 + `model_best.pt` keep-best, and three new decode
   flags in `run_rollout.py`. Under the honest split, val movement-F1 is
   flat (~0.43–0.47) across 1–20 epochs for every recipe.

**Where this leaves the project:** stop tuning the CLIP head against 5-ep
reward. The two defensible directions are (a) an eval-variance fix (more
episodes per cell, and/or demo-grounded proxy metrics from steps_*.json) so
comparisons mean something, and (b) the LLaVA Phase C cells (caches exist on
cluster BIG) evaluated under that fixed protocol from day one.

## TL;DR — 2026-06-10 (camera-axis sweep: CLOSED, negative)

The camera axis proposed in the 2026-06-09 TL;DR was swept (C1–C7, see
`docs/trials.md` top section) and is **closed with a negative result**:

- Camera *prediction* is already at the demo floor (mae 0.46°). Class
  weights (`wloss`/`cwloss`) break the binary policy (dirt collapses to
  0.00–1.67); scaling the camera CE (`camCE×2`) adds variance without mean
  gains (3.33 / 9); decode-side `--camera-temperature 4.0` regresses dirt
  (2.60). No intervention moved chop off reward=0.
- **Verdict: chop failure is temporal/behavioral** — the per-step camera
  prediction is right, but its use over time during chop attempts is wrong.
  Do not spend more compute on camera-loss recipes.
- SOTA confirmed stable: `slot30_chop3` = 4.60 dirt/ep (max 8).

Next batch (see `~/.claude/plans/recipe-merry-fog.md` roadmap): decode
sweeps on the SOTA ckpt (the exp2_thr trick was never run on it), temporal
decode mechanisms in `run_rollout.py` (chunk ensembling / open-loop
execute-k / attack hysteresis), wrapper re-evals of `r3b_lowDrop`/`slot50`,
a knob-only compound-neighborhood cluster batch, and trainer selection infra
(per-trajectory val split + keep-best checkpoint).

> This is the single source of truth for picking up the project.
> Older handoffs (`agent_handoff_2026-06-05.md`, `alignment_handoff_2026-06-04.md`,
> `HANDOFF_pre-2026-06-08.md`) live under `docs/_archive/` with SUPERSEDED
> banners — their specific technical details are still referenced from code
> comments and remain accurate, but their proposed pending tasks are stale.

## TL;DR — 2026-06-09

**Two big findings on top of the 2026-06-08 handoff:**

1. **Every "reward=0" verdict before today was a false negative.** Malmo's
   `RewardForPossessingItem` handler silently fails to emit reward in our
   docker stack — `obs["inventory"]` correctly tracks gained items but
   `env.step()`'s reward stays 0. Confirmed visually:
   `output/evaluation/20260608_161320_lang_slot30/D_dirt_task/episode_002.mp4`
   has dirt in the HUD with logged `total_reward = 0`. Patched in commit
   `59897b8`: `InventoryRewardWrapper` (run_rollout.py:159) wraps the env
   and computes reward from `obs["inventory"]` delta inline. After the
   patch, eval_suite + eval_compare numbers are real. See
   [[chop-reward-unreliable]] memory for the root-cause writeup.

2. **SOTA on combined-stride-4 CLIP head: `slot30_chop3`** — mean **4.60
   dirt/ep (max 8) on D_dirt_task across 5 seeded eps.** Recipe:
   `PAST_ACTION_SLOT_DROPOUT=0.3 CHOP_OVERSAMPLE_WEIGHT=3.0` on the
   existing `clip_combined_lang_stride4` cache (no other knobs, 10 epochs,
   HIDDEN_DIM=2048). Beats slot30 alone (3.20) and ep20 alone (1.60); the
   compound is genuinely synergistic. Both training knobs landed in
   commit `59897b8`. Cluster command:
   ```bash
   BACKBONE=clip USE_LANGUAGE=1 FRAME_STRIDE=4 HIDDEN_DIM=2048 TASK_FILTER='' \
     PAST_ACTION_SLOT_DROPOUT=0.3 CHOP_OVERSAMPLE_WEIGHT=3.0 \
     RECIPE_TAG=slot30_chop3 \
     sbatch slurm_train_nvidiaall.sh
   ```

**Counter-intuitive lesson — don't add epochs to the compound.** Tested
`slot30_chop3 + EPOCHS=20` (job 154188); F1 hinted at over-fit (forward
+0.010 / sprint +0.015 but left −0.037 / back −0.056 vs 10-ep), and the
env reward COLLAPSED to 0.00 across all 5 dirt eps. ep20 alone (no
weighted sampling, no slot dropout) helps F1 modestly, but the compound
recipe is already aggressive — doubling epochs pushes the head past the
sweet spot. **Stick to 10 epochs on compound recipes.**

**Chop tasks (A_chop_nocap / B_chop_ood / C_chop_task) still reward=0
across every CLIP recipe tested**, even with the wrapper. The pitch
policy never aims at trunks; this is orthogonal to the head-recipe
sweep and likely needs an aim-aware training fix (e.g. retrain with
camera-y class weight or use `MineRLChopATree640FastAim-v0` for eval).

## Earlier (pre-2026-06-09) summary

The 2×2 ablation's "language axis" question is answered: **lang's text encoder
does drive real behavioral shifts in rollout** — confirmed with a Bonferroni-
corrected seeded eval suite. None of the three CLIP heads completes the chop
reward chain (all 120 k rollout steps ended reward=0 — BUT see today's
finding that this was the broken handler, not the model). The next problem
isn't *whether* language matters but *what makes the model actually pick up a
log*. Likely either Round 2 of recipe tuning (tighter frame-weight + Round 1's
over-correction reverse), or a decode change (greedy + threshold) that the
older `exp2_thr` baseline used.

LLaVA Phase C is fully staged for a rented 5090 (`scripts/launch_5090_phase_c.sh`
+ `scripts/push_data_to_5090.sh`), gated only on the user topping up vast.ai
balance (~$5 short).

## What "Round N" means in this project

The user and previous-session agent settled on a numbered-round scheme for
recipe-tuning passes on top of an existing cache. The conventions:

- **Baseline = pre-Round-1.** `clip_combined_{lang,nolang}_stride4` are the
  unweighted heads trained 2026-06-06 with `weighted_loss=False,
  history_dropout=0.0`, no frame-weighted sampling, no learnable BCE temp.
  These remain the SOTA reference cells (see [[current-clip-sota]]).

- **Round 1 = first cache-safe recipe pass.** Goal: push the head toward
  "task-active" frames without touching the cache. Three knobs applied
  simultaneously on the lang cell:
  1. **Frame-weighted sampling** (`--frame-weight-multiplier 5.0
     --frame-weight-min-run 60`): `WeightedRandomSampler` upweights training
     frames inside sustained-attack runs ≥60 ticks (~3 s).
  2. **History dropout** (`--history-dropout 0.5`): randomly zero the
     past-action vector in half of training samples to prevent the head from
     locking into sticky no-move recurrences.
  3. **Learnable per-action BCE temperature** (`--learnable-bce-temp`):
     `nn.Parameter(torch.ones(NUM_BINARY))` divides binary logits in
     `HeadOnlyAgent.forward`, jointly trained so sigmoid outputs sit closer
     to 0/1 at inference (less need for `--binary-thresholds`).
  Outcome 2026-06-07 evening: ckpt at
  `output/clip_combined_lang_stride4_r1/`. Per-action F1 mixed (back/jump
  +0.06/+0.05, forward/left/right −0.025/−0.05/−0.034). Eval suite shows
  attack +10-19 pp, sneak +11-44 pp, sprint −5 to −38 pp vs lang.
  **Over-corrected** — frame-weighting flagged 66 % of frames as "active"
  (intended ~5-10 %); the camera y-tilt conditioning that lang had vs
  nolang disappeared. R1 was applied to lang only — never to nolang.

- **Round 2 (proposed, not yet run).** Tighten R1's `min_run` from 60 to
  ~180 ticks (≈9 s sustained attack — closer to actual chop-completion
  windows). Either keep the other R1 knobs or ablate which one drives the
  over-correction. Expected to land at `output/clip_combined_lang_stride4_r2/`.
  See "What to do next §2" below for the submission cmd.

- **Round 3+ (speculative).** Auxiliary head losses (predict "is this a
  chopping segment?" off the cached feature) and calibration-aware BCE
  with focal loss were proposed in earlier handoffs (see
  `docs/_archive/agent_handoff.md` "CLIP improvements" section) but
  haven't been built. Only worth doing if R2 still doesn't break the
  reward chain.

- **Phase C ≠ Round N.** Phase C is the LLaVA backbone migration (split
  image/text pooling, HIDDEN_DIM=2048 plumbing, sm_120 / 5090 path). It's
  separate from CLIP recipe tuning — both progress in parallel.

## State of the world

### Models that exist (all CLIP, combined chop+dirt, stride 4, HIDDEN_DIM=2048)

| Tag | Path | Notes |
|---|---|---|
| **lang** (SOTA) | `output/clip_combined_lang_stride4/model.pt` | trained 2026-06-06 on cluster NvidiaAll. weighted_loss=False, history_dropout=0. |
| **nolang** (SOTA control) | `output/clip_combined_nolang_stride4/model.pt` | symmetric to lang, text branch zeroed. |
| **lang_r1** (Round 1 retrain) | `output/clip_combined_lang_stride4_r1/model.pt` | trained 2026-06-07; frame-weight=5×, min_run=60, history_dropout=0.5, learnable_bce_temp. **Over-corrects** — see [[lang-conditioning-confirmed]] and [[current-clip-sota]]. |

Older models in `output/` are legacy (chop-only stride-8 etc.) — do not use
for new comparisons. The legacy `exp2_thr` decode trick (greedy + `--binary-
thresholds attack=0.005` on the no_move_fix ckpt) is the only artifact that
ever produced visible chopping (`rollout_logs/exp2_thr/episode_004.mp4`).

### Eval suite results

Full per-action firing rate table + Wilson 95 % CI + pairwise Bonferroni-
corrected z-tests + Cohen's h are at:
- `output/evaluation/comparison.txt` (formatted)
- `output/evaluation/comparison.csv`

Raw episode JSONs + mp4s at
`output/evaluation/{20260607_211016_lang,20260607_231607_nolang,20260608_012831_lang_r1}/{A_chop_nocap,B_chop_ood,C_chop_task,D_dirt_task}/`.
Each cell is 10 episodes × 1000 steps on seeds [0..9], same env+seed across every
(model, prompt) cell so step-by-step diff is possible.

**Headline findings — already saved to memory** ([[lang-conditioning-confirmed]]):

1. **lang shows real prompt conditioning** (attack 82 / 71 / **29** % across
   prompts A=`""` / B=`"Play Minecraft."` / C=`"chop a tree"`, same env+seeds).
   With the *training* prompt the model goes from "constant attack" baseline
   to "wander → find → chop" cycle.
2. **nolang doesn't condition** (attack 61 / 61 / 66 % — 5 pp variation is
   the seed-determinism noise floor that survives `--seed`).
3. **Camera y-tilt separates lang from nolang** at a different signal —
   lang looks slightly *down* (+0.5 to +1.0°) across all 4 prompts, nolang
   slightly *up* (−0.6 to −0.7°).
4. **lang_r1 over-corrected**. The recipe boosted attack (+10–19 pp) and
   sneak (+11–44 pp) but dropped sprint (−5 to −38 pp). Camera y-tilt
   conditioning disappeared. Likely cause: `min_run=60` ticks flagged
   **66.1 %** of all frames as "task active" (vs the intended ~5–10 %), so
   the WeightedRandomSampler over-fit to the dominant fragment.
5. **Reward chain still broken across all three models**. 120 k seeded rollout
   steps; mean_reward=0 everywhere.

## What to do next

Sequenced by cost / payoff:

### 1. (~free) Greedy + threshold decode on today's lang ckpt

The behavioral mode that produced visible chopping (`exp2_thr`) used
greedy + `--binary-thresholds attack=0.005`. Worth testing on
`clip_combined_lang_stride4/model.pt` with the training prompt:

```bash
docker compose run --rm --remove-orphans minerl run_rollout.py \
  --remote-agent host.docker.internal:8765 \
  --env MineRLChopATree640Fast-v0 \
  --episodes 5 --max-steps 1000 \
  --prompt "chop a tree" \
  --binary-thresholds attack=0.005 \
  --color-match auto \
  --seed 0 \
  --output-dir output/eval_exp2_style_lang
```

If this produces reward > 0 we've separated "decode is the bottleneck" from
"model is the bottleneck" — and we have a working CLIP baseline cell.

### 2. (~30 min retrain) Round 2 of frame-weighted sampling

R1 had `min_run=60` (3 s sustained attack) which flagged 66 % of frames.
Tighten to `min_run=180` (≈9 s) to isolate actual chop-completion moments
(should drop active fraction to <15 %). Either retrain just lang or both
lang+nolang for symmetry. Cluster cmd:

```bash
ssh remote.cip.ifi.lmu.de
cd ~/BIG
BACKBONE=clip USE_LANGUAGE=1 FRAME_STRIDE=4 HIDDEN_DIM=2048 TASK_FILTER='' \
  FRAME_WEIGHT_MULTIPLIER=5.0 FRAME_WEIGHT_MIN_RUN=180 \
  HISTORY_DROPOUT=0.5 LEARNABLE_BCE_TEMP=1 RECIPE_TAG=r2 \
  sbatch slurm_train_nvidiaall.sh
```

After ~17 min, ckpt lands at `~/BIG/output/clip_combined_lang_stride4_r2/`.
Rsync down (~55 MB) and re-run `scripts/eval_suite.sh` to compare against
lang and lang_r1.

### 3. (~5 h) LLaVA Phase C cache build on rented 5090

Already staged. The user needs to:
1. Top up vast.ai by ~$5 (current credit $9.18 is $0.40 short of $10 estimate)
2. Rent a 5090 with `disk_space>=200 inet_down>=400`
3. Add their cluster SSH pubkey to box's `authorized_keys`
4. From cluster login node: `bash ~/BIG/scripts/push_data_to_5090.sh <host> <port>`
5. SSH to box: `cd /workspace/r1-va && bash scripts/launch_5090_phase_c.sh`
6. Pull back: `rsync -avz -e 'ssh -p <port>' root@<host>:/workspace/{caches,output}/ ~/BIG/`

Full runbook is in `CLAUDE.md §2c`. Both `scripts/launch_5090_phase_c.sh`
and `scripts/push_data_to_5090.sh` are on the cluster at `~/BIG/scripts/`.

### 4. Eval LLaVA cells once they exist

Once `llava_combined_{lang,nolang}_stride4` heads exist:
```bash
bash scripts/eval_suite.sh output/llava_combined_lang_stride4/model.pt llava_lang 0
bash scripts/eval_suite.sh output/llava_combined_nolang_stride4/model.pt llava_nolang 0
python scripts/eval_compare.py --models lang nolang lang_r1 llava_lang llava_nolang \
  --csv output/evaluation/full_2x2_comparison.csv | tee output/evaluation/full_2x2.txt
```

This is the paper-grade 2×2 + R1 ablation.

## Infrastructure summary

### Tooling that landed this session

- `run_rollout.py --seed N`: env+np+torch RNGs seeded per episode for
  cross-model step-by-step comparison.
- `scripts/eval_suite.sh <model_path> [tag] [base_seed]`: 4 prompt-
  conditions × 10 ep × 1000 steps, handles inference-server swap, writes
  to `output/evaluation/<endtime>_<tag>/` with `manifest.json` (model SHA-1,
  start/end time, decode flags, conditions).
- `scripts/eval_compare.py`: walks `output/evaluation/*`, reports per-action
  firing rates + Wilson 95 % CI + pairwise Bonferroni-corrected z-tests +
  Cohen's h. Optional `--csv` output.
- `feature_cache.HeadOnlyAgent(learnable_bce_temp=True)`: per-action
  temperature on binary logits, threaded through `cluster_pipeline.py`
  + `slurm_train_nvidiaall.sh` (`LEARNABLE_BCE_TEMP=1` env var).
- `imitation_learning.compute_task_active_weights()` + frame-weighted
  sampling plumbed through `train_cached_head` (`--frame-weight-multiplier`,
  `--frame-weight-min-run`).
- `agent_loader.py` bug fix: `cache_tag.endswith("nolang")` was missing
  the `_stride4` suffix → token-level check (`"nolang" not in
  cache_tag.split("_")`). Was silently loading nolang ckpts with
  `use_language=True` until 2026-06-07 evening.

### Cluster state (memory: [[lmu-cluster-state]])

- LMU CIP Abaki QOS still revoked for `stud_ifi` — submissions go to
  `NvidiaAll` partition via `slurm_train_nvidiaall.sh` (RTX 2060 SUPER 8 GB).
- Trajectories pre-extracted at `~/BIG/trajectories/` (~120 GB, 17 GB free
  in 144 GB BIG quota).
- Caches at `~/BIG/caches/`: 4 CLIP combined-stride-4 .npy (4.79 GB each).

### Code state on Mac

- Memory index at `~/.claude/projects/-Users-diego-VSCode-r1-va/memory/MEMORY.md`
- Plans dir: `~/.claude/plans/`
- Latest pre-session plan: `~/.claude/plans/groovy-jingling-flask.md`
  (largely superseded by this handoff)

## Open questions for the next session

1. **Does greedy + threshold decode produce reward on today's models?** —
   tests whether decode is the bottleneck. Answer in step 1 above.
2. **Round 2 retrain: tighter min_run, or revert learnable_bce_temp?** —
   the BCE temp learning may also be over-fitting. Could ablate by training
   r2_no_temp without that flag.
3. **What is the right "task completion" metric** if reward is unreliable
   on chop? Could compute "frames inside contiguous attack-runs ≥120 ticks
   per episode" as a proxy for "the agent committed to a chop attempt".

## Files to read first when picking this up

1. `docs/HANDOFF.md` — this file (canonical handoff dated 2026-06-08).
   Older superseded handoffs are in `docs/_archive/` with banners.
2. Memory index `~/.claude/projects/-Users-diego-VSCode-r1-va/memory/MEMORY.md`
   then the recent entries (`current-clip-sota`, `lang-conditioning-confirmed`,
   `lmu-cluster-state`).
3. `output/evaluation/comparison.txt` — the numbers behind the
   conditioning finding.
4. `CLAUDE.md §2c` — 5090 LLaVA launch runbook.
5. `scripts/eval_suite.sh` + `scripts/eval_compare.py` — the eval tooling
   you'll use for any new ckpt.
