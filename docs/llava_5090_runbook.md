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

## 1. Stage data (push FROM the Mac)

This book covers **only the 2 LLaVA caches** — `{lang,nolang}`. The CLIP half of
the study (its 2 caches → 4-recipe heads → 4-way eval) runs **locally on the Mac**
(CLIP fits), so don't ship the CLIP caches to the box. Head training needs ONLY
the caches + `all_actions.json` — no videos.

```bash
# from the Mac repo root (~40 GB, ~20 min):
rsync -a --partial --progress -e "ssh -p <port>" \
  caches/llava_combined_lang_stride4.{npy,json} \
  caches/llava_combined_nolang_stride4.{npy,json} \
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
for t in ['llava_combined_lang_stride4','llava_combined_nolang_stride4']:
    f,m = load_cache('caches', t); print(t, f.shape, m['backbone'], m['use_language'])"
# expect (1560750, 8192) for each LLaVA cache
```

## 2. Head trainings — 8 LLaVA heads (2 caches × 4 recipes)

**The 4 recipes** (chosen from the CLIP item-collection screen, `docs/recipes.md`;
chop is floor for all, so these are the dirt leaders + the knob-free anchor as the
in-cell control):

| recipe | role | recipe flags |
|---|---|---|
| `anchor` (tsplit) | in-cell control (knob-free) | (none) |
| `slot30_chop3` | dirt leader | `--past-action-slot-dropout 0.3 --chop-oversample-weight 3.0` |
| `r3c_minrun300` | dirt (frame-weighted) | `--history-dropout 0.5 --frame-weight-multiplier 5.0 --frame-weight-min-run 300 --learnable-bce-temp` |
| `slot50` | dirt (heavy slot-dropout; only CLIP cell to ever complete a chop) | `--past-action-slot-dropout 0.5` |

Train each recipe on **both** LLaVA caches (`--backbone llava` lang, and
`--backbone llava --no-language` nolang) → 8 heads. **4 already exist; 4 to train:**

| head (`output/...`) | recipe | status |
|---|---|---|
| `llava_combined_lang_stride4_tsplit` | anchor · lang | ✅ exists |
| `llava_combined_nolang_stride4_tsplit` | anchor · nolang | ✅ exists |
| `llava_combined_lang_stride4_slot30_chop3` | slot30_chop3 · lang | ✅ exists |
| `llava_combined_nolang_stride4_slot30_chop3` | slot30_chop3 · nolang | ✅ exists |
| `llava_combined_lang_stride4_r3c_minrun300` | r3c · lang | ⬜ train |
| `llava_combined_nolang_stride4_r3c_minrun300` | r3c · nolang | ⬜ train |
| `llava_combined_lang_stride4_slot50` | slot50 · lang | ⬜ train |
| `llava_combined_nolang_stride4_slot50` | slot50 · nolang | ⬜ train |

> Head training is a tiny MLP on cached features (~15–30 min/run, $0) — run it
> **locally on MPS** (`--device mps --num-workers 0`; macOS `spawn` OOMs at
> workers>0) **or** on the box. **Cap epochs at ~20**: a cosine long-epoch probe
> (lr5e4 ep40) plateaued at ~0.43 val movement-F1 by ep25, no better than 20 ep
> and at/under CLIP — and test-split F1 across backbones is within noise, so the
> decisive comparison is the **rollout eval (§3), not F1**.

All via `cluster_pipeline.py` (cache present → skips build, goes straight to
`train_cached_head`; trajectory split + keep-best is the cohort default):

```bash
COMMON="--data-dir trajectories --cache-dir caches --device cuda \
  --epochs 10 --batch-size 256 --lr 1e-3 --num-workers 8 \
  --past-action-k 8 --chunk-size 8 --frame-stride 4 --hidden-dim 2048 --keep-best"

# r3c_minrun300 (lang + nolang)
python cluster_pipeline.py $COMMON --backbone llava \
  --output-dir output/llava_combined_lang_stride4_r3c_minrun300 \
  --history-dropout 0.5 --frame-weight-multiplier 5.0 --frame-weight-min-run 300 --learnable-bce-temp
python cluster_pipeline.py $COMMON --backbone llava --no-language \
  --output-dir output/llava_combined_nolang_stride4_r3c_minrun300 \
  --history-dropout 0.5 --frame-weight-multiplier 5.0 --frame-weight-min-run 300 --learnable-bce-temp

# slot50 (lang + nolang)
python cluster_pipeline.py $COMMON --backbone llava \
  --output-dir output/llava_combined_lang_stride4_slot50 \
  --past-action-slot-dropout 0.5
python cluster_pipeline.py $COMMON --backbone llava --no-language \
  --output-dir output/llava_combined_nolang_stride4_slot50 \
  --past-action-slot-dropout 0.5
```

Each run writes `model.pt`, `model_best.pt` (best val movement-F1 epoch) and
`metrics.json`. Serve `model_best.pt` for the evals.

## 3. Evals — two 4-way recipe evals (one per LLaVA cache)

Each LLaVA cache's 4 heads compete in a **4-way recipe eval**: `anchor` (control)
vs `slot30_chop3` vs `r3c_minrun300` vs `slot50`. Run it for both caches → 8
rollout runs. (The CLIP half — the same two 4-way evals on the CLIP caches — runs
locally on the Mac; together the four 4-ways give the full backbone × language ×
recipe picture.)

Why this scope: the CLIP item-collection screen (`docs/recipes.md`) found **chop
is floor for every recipe** (don't chase it) and **dirt is the only working task**;
these 4 are the dirt leaders + the knob-free control.

| 4-way eval | cache | tags (anchor / slot30_chop3 / r3c / slot50) |
|---|---|---|
| **LLaVA-lang** | `llava_combined_lang_stride4` | `llava_lang_anchor` `llava_lang_slot30_chop3` `llava_lang_r3c` `llava_lang_slot50` |
| **LLaVA-nolang** | `llava_combined_nolang_stride4` | `llava_nolang_anchor` `llava_nolang_slot30_chop3` `llava_nolang_r3c` `llava_nolang_slot50` |

### Protocol (identical across all 8 runs)

Full `eval_suite`: 4 conditions × **10 ep × 1000 steps**, base seed 0. Serve
`model_best.pt` on the 5090 (`inference_server.py --device cuda`); the container
connects via `--remote-agent`. **Decode = sampled** (`--sample --temperature 1.0
--camera-temperature 2.0 --color-match auto`, hardcoded in `eval_suite.sh`, matches
the CLIP cells — do NOT switch to greedy). Conditions: A_chop_nocap (`""`),
B_chop_ood (`"Play Minecraft."`), C_chop_task (`"chop a tree"`) on
`MineRLChopATree640Fast-v0`; D_dirt_task (`"collect dirt"`) on
`MineRLCollectDirt640Fast-v0`.

```bash
rm -f logs/minerl_watchers/*.pid
run() { EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda \
        bash scripts/eval_suite.sh "output/$1/model_best.pt" "$2" 0; }

# --- LLaVA-lang 4-way ---
run llava_combined_lang_stride4_tsplit          llava_lang_anchor
run llava_combined_lang_stride4_slot30_chop3    llava_lang_slot30_chop3
run llava_combined_lang_stride4_r3c_minrun300   llava_lang_r3c
run llava_combined_lang_stride4_slot50          llava_lang_slot50
# --- LLaVA-nolang 4-way ---
run llava_combined_nolang_stride4_tsplit        llava_nolang_anchor
run llava_combined_nolang_stride4_slot30_chop3  llava_nolang_slot30_chop3
run llava_combined_nolang_stride4_r3c_minrun300 llava_nolang_r3c
run llava_combined_nolang_stride4_slot50        llava_nolang_slot50
```

### Metrics (straight out of `run_summary.json`)

1. **PRIMARY — task performance (dirt):** `mean_peak_inventory["dirt"]` +
   `collect_rate` (reward-independent; the env reward was broken — `docs/recipes.md`).
   `total_reward` is now also correct (wrapper fixed). *Within each cache: which
   recipe digs most, and does any beat the `anchor` control?*
2. **Conditioning (cross-cache):** compare the LLaVA-lang vs LLaVA-nolang 4-ways —
   per-action firing rates across A/B/C via `eval_compare.py`. *Does lang shift
   attack across prompts while nolang stays flat?* (CLIP: 82/71/29 % vs ~61–66 %.)
3. **Chop:** expect 0 logs for all — the negative result; don't tune for it.

Aggregate (one compare per 4-way, then cross to CLIP):
```bash
python scripts/eval_compare.py --root evaluations/paper --models llava_lang_anchor llava_lang_slot30_chop3 llava_lang_r3c llava_lang_slot50
python scripts/eval_compare.py --root evaluations/paper --models llava_nolang_anchor llava_nolang_slot30_chop3 llava_nolang_r3c llava_nolang_slot50
python scripts/rank_item_collection.py evaluations/paper
```
Pre-check before rollouts: the per-action F1 table from each head's `metrics.json`
gives an immediate read (but the decision is the dirt rollout, not F1).

## 4. Pull artifacts back to the Mac

```bash
# the 4 newly-trained LLaVA heads (r3c, slot50 × lang/nolang):
rsync -avz -e "ssh -p <port>" \
  root@<host>:/workspace/r1-va/output/llava_combined_*_stride4_{r3c_minrun300,slot50} output/
# the 8 rollout-eval dirs:
rsync -avz -e "ssh -p <port>" root@<host>:/workspace/r1-va/evaluations/paper/ evaluations/paper/
```

## 5. Acceptance checklist

- [ ] 4 new heads trained (`r3c_minrun300`, `slot50` × lang/nolang) →
      `model_best.pt` + `metrics.json` (the 4 others already exist)
- [ ] 8 rollout evals done (2 caches × 4 recipes), 10 ep × 4 conditions
- [ ] Two 4-way `eval_compare` runs (LLaVA-lang, LLaVA-nolang) — dirt item counts
- [ ] `rank_item_collection.py evaluations/paper` shows the per-cache recipe ranking
- [ ] Cross-cache conditioning read (lang vs nolang firing rates A/B/C)
- [ ] Chop confirmed at floor (negative result recorded)
- [ ] Everything rsync'd back; box destroyed only after checksums verified

Budget: trainings <1 GPU-h total; evals dominated by MineRL wall-clock
(~10-14 h of episodes at LLaVA inference speed on the 5090, can run
unattended); ~$10-15 at spot prices.
