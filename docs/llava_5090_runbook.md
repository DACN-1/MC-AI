# LLaVA head-training + eval runbook (rented 5090) — 2026-06-11

Self-contained instructions for an agent driving a rented RTX 5090 box.
Goal: train LLaVA heads matching the defensible CLIP recipes, then run the
eval protocol that survives the eval-nondeterminism finding. Context:
`docs/trials.md` ("Wave 1–5 sweep + reproducibility controls") and
`docs/HANDOFF.md` (2026-06-11 TL;DR).

## Why these recipes (and not the reward leaderboard)

The 2026-06-08/09 reward leaderboard (slot30_chop3 4.60 > slot30 3.20 > …)
was shown on 2026-06-11 to be noise-dominated: the same checkpoint scored
4.60 then 0.00 on the same seeds, an exact-recipe retrain scored 0.00, and
identical back-to-back runs differ in step-0 logits by ~0.23 (MineRL reset
is not bit-deterministic). Recipes below are selected by the signals that
ARE robust — high-n val F1 under the honest trajectory split and behavioral
profiles — plus the mandatory knob-free 2×2 anchor.

**Hard rules** (from project memory):
- 2×2 symmetry: across backbone×language cells, every knob other than
  backbone and use_language must be identical (HIDDEN_DIM=2048,
  past_action_k=8, chunk_size=8, stride 4, trajectory split, keep-best).
- Never rank recipes on ≤5-episode wrapper reward. Reward needs 20+
  episodes across distinct seeds; per-step firing-rate stats are the
  primary instrument.

## 0. Box + stack

- vast.ai RTX 5090, `disk_space>=120 inet_down>=400 reliability>=0.97`.
- Clone the repo (github DACN-1/MC-AI, branch main, commit >= 4a0515a).
- `bash scripts/setup_vastai_5090.sh` (torch 2.7 cu128, no flash-attn;
  VLAAgent runs bf16+sdpa — only relevant for serving, not head training).
- For rollout serving use the already-tested path from
  `scripts/setup_vastai_inference.sh` (inference_server.py on the box,
  MineRL docker client connects with `--remote-agent <host>:8765`; the
  JPEG+keep-alive transport handles the tunnel).

## 1. Stage data (push FROM the Mac; cluster no longer has the caches)

Head training needs ONLY caches + all_actions.json — no videos.

```bash
# from the Mac repo root (~60 GB total, ~30 min):
rsync -a --partial --progress -e "ssh -p <port>" \
  caches/llava_combined_lang_stride4.{npy,json} \
  caches/llava_combined_nolang_stride4.{npy,json} \
  caches/clip_combined_lang_stride4.{npy,json} \
  caches/clip_combined_nolang_stride4.{npy,json} \
  root@<host>:/workspace/r1-va/caches/
rsync -a -e "ssh -p <port>" --relative \
  trajectories/./trajectory_task_chop_a_tree_length_3000/all_actions.json \
  trajectories/./trajectory_task_collect_dirt_length_3000/all_actions.json \
  root@<host>:/workspace/r1-va/trajectories/
```

Sanity check on the box before training:

```bash
python -c "
from feature_cache import load_cache
for t in ['llava_combined_lang_stride4','llava_combined_nolang_stride4',
          'clip_combined_lang_stride4','clip_combined_nolang_stride4']:
    f,m = load_cache('caches', t); print(t, f.shape, m['backbone'], m['use_language'])"
# expect (1560750, 8192) for llava, (1560750, 1536) for clip
```

## 2. Head trainings (9 runs, each minutes-fast on a 5090)

> Executed 2026-06-12 **locally on an M1 Pro (MPS)**, not the 5090 — head
> training is a tiny MLP on cached features (~15–30 min/run, $0), and the Mac's
> 13 Mbps uplink made shipping the 57 GB of caches to a box impractical. Use
> `--device mps --num-workers 0` locally (macOS `spawn` materialises the memmap
> per worker → OOM at num_workers>0). The 5090 is now reserved for the eval phase.

All via `cluster_pipeline.py` (cache exists → it skips the build and goes
straight to `train_cached_head`; trajectory-level split is the default).
Common flags:

```bash
COMMON="--data-dir trajectories --cache-dir caches --output-dir output/<DIR> \
  --device cuda --epochs 10 --batch-size 256 --lr 1e-3 --num-workers 8 \
  --past-action-k 8 --chunk-size 8 --frame-stride 4 --hidden-dim 2048 --keep-best"
```

| # | Output dir (`output/...`) | Backbone flags | Recipe flags |
|---|---|---|---|
| 1 | `llava_combined_lang_stride4_tsplit` | `--backbone llava` | (none — 2×2 anchor) |
| 2 | `llava_combined_nolang_stride4_tsplit` | `--backbone llava --no-language` | (none — 2×2 anchor) |
| 3 | `llava_combined_lang_stride4_slot30` | `--backbone llava` | `--past-action-slot-dropout 0.3` |
| 4 | `llava_combined_lang_stride4_slot30_chop3` | `--backbone llava` | `--past-action-slot-dropout 0.3 --chop-oversample-weight 3.0` |
| 5 | `llava_combined_nolang_stride4_slot30_chop3` | `--backbone llava --no-language` | `--past-action-slot-dropout 0.3 --chop-oversample-weight 3.0` |
| 6 | `llava_combined_lang_stride4_lr5e4_ep20` | `--backbone llava` | `--lr 5e-4 --epochs 20` (overrides COMMON) |
| 7 | `clip_combined_lang_stride4_tsplit_base` | `--backbone clip` | (none — CLIP anchor under the honest split, missing from the Wave-3 batch) |
| 8 | `clip_combined_nolang_stride4_tsplit_base` | `--backbone clip --no-language` | (none) |
| 9 | `llava_combined_lang_stride4_lr5e4_cos40` | `--backbone llava` | `--lr 5e-4 --epochs 40 --lr-schedule cosine` (overrides COMMON; long-epoch probe — does LLaVA keep improving past 20 ep?) |

> **Run 9 result (2026-06-12):** No. The cosine schedule damped the constant-LR
> jitter and revealed a clean **plateau at ~0.43 val movement-F1 by epoch ~25**
> (slope over ep21–40 = +0.0006/ep; `val_loss` flat at 0.613). Peak 0.4335@ep25 —
> no better than the 20-epoch constant run (0.432) and still below CLIP's val
> (0.452–0.455). So **cap LLaVA heads at ~20 epochs**; longer training only
> asymptotes to a ceiling that sits at/under CLIP. NB on the *test* split the
> backbone ranking is within noise and flips (LLaVA lr5e4 0.433 ≈ CLIP-nolang
> 0.431 ≥ CLIP-lang 0.407 ≈ cos40 0.423) — the movement-F1 contrast is a wash,
> so the decisive comparison is the §3b conditioning suite, not F1. The
> `--lr-schedule {constant,cosine}` flag was added to `cluster_pipeline.py` /
> `train_cached_head` (default constant, so runs 1–8 are unaffected).

Example invocation (run #4):

```bash
python cluster_pipeline.py --data-dir trajectories --cache-dir caches \
  --output-dir output/llava_combined_lang_stride4_slot30_chop3 \
  --backbone llava --device cuda --epochs 10 --batch-size 256 --lr 1e-3 \
  --num-workers 8 --past-action-k 8 --chunk-size 8 --frame-stride 4 \
  --hidden-dim 2048 --keep-best \
  --past-action-slot-dropout 0.3 --chop-oversample-weight 3.0
```

Each run writes `model.pt`, `model_best.pt` (best val movement-F1 epoch) and
`metrics.json` (test per-action F1). Per-epoch `val_movement_f1` /
`val_per_action_f1` live in the checkpoint's `training_metrics`.

## 3. Evals — the 4 LLaVA rollouts (recipe × language 2×2, with control)

**Why these 4 (and only 4).** The CLIP recipe screen (`docs/recipes.md`,
2026-06-17, objective item-collection metric) established two things that scope
the LLaVA spend:
- **Chop is floor for every recipe** — no head completes a chop. Don't chase it.
- **Dirt is the only task that works**, and `slot30_chop3` is the leading dirt
  recipe (≈4.8 dirt/ep vs baseline 0 on CLIP).

So the LLaVA rollouts are a single **recipe × language 2×2** that answers both
open questions on the working task at once: does the dirt recipe beat the
knob-free anchor on the bigger backbone, and does language conditioning hold?

| # | cell tag | head (serve `model_best.pt`) | role |
|---|---|---|---|
| 1 | `llava_lang` | `llava_combined_lang_stride4_tsplit` | anchor (knob-free), lang — recipe control |
| 2 | `llava_nolang` | `llava_combined_nolang_stride4_tsplit` | **control** (no recipe, no language) |
| 3 | `llava_slot30_chop3_lang` | `llava_combined_lang_stride4_slot30_chop3` | dirt recipe, lang |
| 4 | `llava_slot30_chop3_nolang` | `llava_combined_nolang_stride4_slot30_chop3` | dirt recipe, nolang |

Cell **2 is the control**; rows 1↔3 isolate the recipe effect, {1,3}↔{2,4} isolate
language. All four heads are already trained (`model.pt` + `model_best.pt`).

### Protocol (identical across all 4)

Full `eval_suite`: 4 conditions × **10 ep × 1000 steps**, base seed 0. Serve
`model_best.pt` on the 5090 (`inference_server.py --device cuda`); the container
connects via `--remote-agent`. `eval_suite.sh` handles the server swap.

| Condition | Env | Prompt |
|---|---|---|
| A_chop_nocap | MineRLChopATree640Fast-v0 | `""` |
| B_chop_ood | MineRLChopATree640Fast-v0 | `"Play Minecraft."` |
| C_chop_task | MineRLChopATree640Fast-v0 | `"chop a tree"` |
| D_dirt_task | MineRLCollectDirt640Fast-v0 | `"collect dirt"` |

**Decode = sampled: `--sample --temperature 1.0 --camera-temperature 2.0
--color-match auto`** (hardcoded in `eval_suite.sh`, matches the CLIP cells).
⚠ Do NOT switch to greedy — it would break decode-symmetry vs the CLIP cells.

```bash
rm -f logs/minerl_watchers/*.pid
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda bash scripts/eval_suite.sh output/llava_combined_lang_stride4_tsplit/model_best.pt           llava_lang                0
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda bash scripts/eval_suite.sh output/llava_combined_nolang_stride4_tsplit/model_best.pt         llava_nolang              0
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda bash scripts/eval_suite.sh output/llava_combined_lang_stride4_slot30_chop3/model_best.pt     llava_slot30_chop3_lang   0
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda bash scripts/eval_suite.sh output/llava_combined_nolang_stride4_slot30_chop3/model_best.pt   llava_slot30_chop3_nolang 0
```

### Metrics (both come straight out of `run_summary.json`)

1. **PRIMARY — task performance (dirt):** `mean_peak_inventory["dirt"]` +
   `collect_rate`, the reward-independent item count (the env reward was broken;
   `docs/recipes.md`). `total_reward` is now also correct (wrapper fixed) and
   should agree. *Q: does `slot30_chop3` beat the anchor on dirt for LLaVA, and
   does it match/beat CLIP `slot30_chop3`?*
2. **SECONDARY — conditioning:** per-action firing rates across A/B/C via
   `eval_compare.py` (Wilson CIs, Bonferroni z-tests, camera tilt). *Q: does
   `llava_lang` shift attack across prompts while `llava_nolang` stays flat?*
   (CLIP showed attack 82/71/29 % vs flat ~61–66 %.)
3. **Chop:** expect 0 logs for all — report it as the negative result, don't tune
   for it.

Aggregate:
```bash
python scripts/eval_compare.py --models llava_lang llava_nolang llava_slot30_chop3_lang llava_slot30_chop3_nolang
python scripts/rank_item_collection.py evaluations/paper
```
For the **backbone** contrast, also pull the matched CLIP cells (`lang`, `nolang`,
`clip ... slot30_chop3`) into the same compare. *(Free pre-check: the per-action
F1 table from each cell's `metrics.json` — attack/forward/.../camera-mae — gives
an immediate read before spending rollout time.)*

## 4. Pull artifacts back to the Mac

```bash
rsync -avz -e "ssh -p <port>" \
  root@<host>:/workspace/r1-va/output/{llava_*,clip_*tsplit_base*} output/
# eval outputs land under evaluations/paper/ (llava_lang, llava_nolang):
rsync -avz -e "ssh -p <port>" root@<host>:/workspace/r1-va/evaluations/paper/ evaluations/paper/
```

## 5. Acceptance checklist

- [ ] 8/8 trainings completed with `model_best.pt` + `metrics.json`
- [ ] F1 table assembled (3a) with the three contrasts
- [ ] Conditioning suite: 10 eps × 4 conditions for cells 1, 2, 4
- [ ] eval_compare stats generated (firing rates + CIs + z-tests)
- [ ] 20-ep dirt reward for cells 1 and 4
- [ ] Attack-logit percentile analysis on chop (3d)
- [ ] Everything rsync'd back; box destroyed only after checksums verified

Budget: trainings <1 GPU-h total; evals dominated by MineRL wall-clock
(~10-14 h of episodes at LLaVA inference speed on the 5090, can run
unattended); ~$10-15 at spot prices.
