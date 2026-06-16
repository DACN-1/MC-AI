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

## 2. Head trainings (8 runs, each minutes-fast on a 5090)

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

## 3. Evals

### 3a. F1 table (free, immediate)

After training, assemble the per-action F1 comparison from each cell's
`metrics.json`: rows = the 8 new cells, columns = attack/forward/left/right/
back/jump/sprint/sneak/use F1 + camera mae. Key contrasts:
- LLaVA-lang vs LLaVA-nolang (does language help F1 with the bigger backbone?)
- LLaVA vs CLIP at matched recipe (backbone effect under identical knobs)
- recipe deltas within LLaVA (do slot30/chop3 reproduce their CLIP F1 gains?)

### 3b. Conditioning suite (primary behavioral instrument)

The **paper 2×2** = cells **#1 `llava_combined_lang_stride4_tsplit`** and
**#2 `llava_combined_nolang_stride4_tsplit`** (the knob-free anchors that match
the CLIP `clip_combined_{lang,nolang}_stride4` cells). Serve `model_best.pt` with
`python inference_server.py --model-path <ckpt> --device cuda --port 8765`, then
run 4 conditions × **10 episodes** × 1000 steps, base seed 0:

| Condition | Env | Prompt |
|---|---|---|
| A_chop_nocap | MineRLChopATree640Fast-v0 | `""` |
| B_chop_ood | MineRLChopATree640Fast-v0 | `"Play Minecraft."` |
| C_chop_task | MineRLChopATree640Fast-v0 | `"chop a tree"` |
| D_dirt_task | MineRLCollectDirt640Fast-v0 | `"collect dirt"` |

**Decode — must match the CLIP cells exactly: `--sample --temperature 1.0
--camera-temperature 2.0 --color-match auto`.** These are hardcoded in
`scripts/eval_suite.sh` (lines 97-100) and are exactly what the existing CLIP
paper cells used (their `manifest.json` records `sample=true, temp 1.0,
cam-temp 2.0`). So the simplest correct path is to run the suite, which bakes in
that decode and writes a `manifest.json` for `eval_compare.py`:

```bash
rm -f logs/minerl_watchers/*.pid
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda \
  bash scripts/eval_suite.sh output/llava_combined_lang_stride4_tsplit/model_best.pt   llava_lang   0
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda \
  bash scripts/eval_suite.sh output/llava_combined_nolang_stride4_tsplit/model_best.pt llava_nolang 0
```

⚠ **Do NOT use greedy decode here.** An earlier draft of this section said "plain
greedy, NO --sample" — that is wrong and would silently break decode-symmetry
against the already-run CLIP cells. The decode "proven harmful on CLIP" was the
*aggressive* family (low attack thresholds, logit bias, high temperature), NOT
this temp-1.0 sampling. If you bypass `eval_suite.sh` and call `run_rollout.py`
directly, pass those four flags verbatim.

Aggregate: `python scripts/eval_compare.py --models lang nolang llava_lang
llava_nolang` (per-action firing rates, Wilson CIs, Bonferroni z-tests, camera
tilt). Headline question: does LLaVA-lang shift attack across prompts A/B/C while
LLaVA-nolang stays flat (CLIP showed attack 82/71/29% vs flat ~61-66%)?

### 3c. Reward (secondary, properly powered)

D_dirt_task only, cells 1 and 4: **20 episodes** (`--episodes 20 --seed 100`,
fresh env per run if the harness allows). Report mean ± per-episode spread.
Do NOT compare against the historical 4.60 (known unreliable); compare cells
against each other within this same session.

### 3d. Chop behavioral check (no reward expected)

From C_chop_task steps_*.json: attack firing rate, mean sustained-attack run
length, camera-y distribution. Expect reward 0; what matters is whether
LLaVA's attack logit is also context-locked like CLIP's (never crosses 0.5)
— grep `action_vec[0]` (attack logit) percentiles.

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
