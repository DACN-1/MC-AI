# llava_fix.md — deploy & evaluate the A+B LLaVA conditioning head

**For:** an agent who will roll out and evaluate this checkpoint in MineRL.
**Goal of the experiment:** test whether two **post-cache** head-side fixes recover
the **language conditioning** that the plain LLaVA anchor lacks (attack-firing
must drop on the on-task prompt vs the empty/OOD prompts). It is **not** expected
to reach CLIP-level item collection — that's bounded by the frozen mean-pooled
features and is out of scope here.

---

## 1. What was trained (and why)

Frozen LLaVA-1.5-7B features are mean-pooled to `[image_pool(4096) ‖ text_pool(4096)]`
and probed by a small MLP head. The plain anchor head ignores language because
(a) the un-normalized 8192-d input is dominated by the high-variance image dims,
and (b) the head reads task-identity off the *image* (a shortcut), leaving
`text_pool` unused. Two head-only fixes address that — **no cache rebuild**:

- **Fix A — `feature_norm`**: `LayerNorm(8192)` on the cached feature before the
  MLP, equalizing image vs text per-dim scale. Active at train **and** rollout.
- **Fix B — `image_dropout=0.15`**: during training, zero the image half of the
  feature for 15% of samples so the head must read the task from `text_pool`,
  breaking the image shortcut. **Train-time only — a no-op at rollout.**

Both landed in `feature_cache.HeadOnlyAgent`, `imitation_learning.train_cached_head`,
and `cluster_pipeline.py` (flags `--feature-norm`, `--image-dropout`). The rollout
loader `agent_loader._load_cached_head_agent` rebuilds `feature_norm` from the
checkpoint config, so **deploy needs no code changes**.

### Training result (health only — not the verdict)
10 epochs, stable, monotonic. Best `val_movement_f1 = 0.4129` (epoch 10) — on par
with / slightly above the anchor (0.394), so the head is healthy and **not**
collapsed. F1 only confirms it learned to imitate; it does **not** measure
conditioning. The verdict is the rollout in §4.

> Note: an earlier attempt at `lr 1e-3` (anchor LR) + `image_dropout 0.25`
> **diverged** (LayerNorm amplifies the gradient + dropout adds variance → fell
> into the degenerate "predict no movement" basin, `move_f1→0`). The shipped head
> uses **`lr 3e-4` + cosine decay + `image_dropout 0.15`** to stay stable. The
> diverged run is archived at `output/_diverged_normdrop25_lr1e3/` — do not deploy it.

---

## 2. The checkpoint

```
output/llava_combined_lang_stride4_fixAB/model_best.pt   <- DEPLOY THIS (best epoch)
output/llava_combined_lang_stride4_fixAB/model.pt        <- last epoch (fallback)
```
- `model_best.pt` sha1 `9d5da8fbd7942c8afc2845d3e25391a0681bc5a0` (~208 MB).
- **Type:** cached-head checkpoint (carries `cache_tag`, no `llava_model` key).
  At rollout, `agent_loader` routes it to `_CachedHeadRolloutAgent`, which rebuilds
  the **frozen LLaVA-1.5-7B backbone** and runs `encode()` **live per frame**,
  then this head. `feature_norm` is rebuilt from `config["feature_norm"]=True`;
  `image_dropout` is off in eval.
- Config: `feature_dim=8192, hidden_dim=2048, past_action_k=8, chunk_size=8,
  use_language=True (cache_tag has no "nolang" token)`.

---

## 3. Hardware — READ THIS FIRST

This rolls out the **live LLaVA-7B backbone on every frame** (the feature cache
is *not* used at rollout — the agent visits new states). That is heavy:

- **Use a CUDA GPU box** (the cluster or a rented 5090) with `DEVICE=cuda`.
  Needs the HF weights `llava-hf/llava-1.5-7b-hf` (~14 GB bf16) and MineRL in Docker.
- **Do NOT run on the 16 GB Mac.** Sustained LLaVA + MPS + MineRL thrashes
  (~hours/episode; see `memory/project_mac_eval_thrash.md`). The cache `.npy` and
  `trajectories/` are **not** needed for rollout — only the checkpoint + HF model + Docker.

---

## 4. Deploy: one command (4 conditions × 10 ep × 1000 steps)

`scripts/eval_suite.sh` starts `inference_server.py` (loads the checkpoint),
then drives MineRL in Docker across the 4 prompt-conditions with identical seeds.
Decode flags (`--sample --temperature 1.0 --camera-temperature 2.0 --color-match auto`),
the stale-PID guard (`rm -f logs/minerl_watchers/*.pid`), and `manifest.json` are
**all handled inside the script** — this guarantees decode symmetry with the CLIP/LLaVA
paper cells. Do not override the decode.

```bash
cd /path/to/r1-va
# Build/refresh trajectories NOT required. MineRL Docker image must be built.
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda \
  bash scripts/eval_suite.sh \
    output/llava_combined_lang_stride4_fixAB/model_best.pt \
    llava_lang_fixAB \
    0
```

- Conditions: `A_chop_nocap` (prompt "") · `B_chop_ood` ("Play Minecraft.") ·
  `C_chop_task` ("chop a tree") · `D_dirt_task` ("collect dirt").
- Output lands at `evaluations/paper/<endtime>_llava_lang_fixAB/` (renamed from
  `.running_*` on success) with a `manifest.json`.
- The inference server keeps running after; kill with `kill $(lsof -t -i:8765)`.

---

## 5. The verdict: did conditioning come back?

```bash
python scripts/eval_compare.py --root evaluations/paper \
  --models llava_lang_fixAB llava_lang_anchor clip_lang_anchor clip_nolang_anchor
```

Read **attack-firing %** across `A_chop_nocap / B_chop_ood / C_chop_task` (table 1).
The conditioning signal is the **A→C drop** (attack should fall on the on-task
"chop a tree" prompt). Reference numbers already in `evaluations/paper`:

| cell | A | B | C | A→C Δ | conditions? |
|------|--:|--:|--:|------:|---|
| `clip_lang_anchor` | 84.9 | 76.7 | 37.0 | **−47.9 pp** | ✅ strong |
| `clip_nolang_anchor` (control) | 57.8 | 57.8 | 64.0 | +6.1 pp | flat |
| `llava_lang_anchor` (the thing we're fixing) | 34.9 | 32.6 | 29.8 | **−5.1 pp** | ❌ none |
| `llava_lang_fixAB` | ? | ? | ? | **?** | **← does this beat the ~5 pp noise floor?** |

**Success criterion:** `llava_lang_fixAB` shows an A→C attack drop **clearly beyond
the ~5 pp noise floor** (anchor showed −5 pp = none; CLIP lang anchor shows −48 pp).
Even a partial drop (e.g. −15/−20 pp) is a positive result — it means language now
moves the policy. Also sanity-check the policy isn't degenerate (inventory should
**not** be ~60% like the broken anchor; movement should look like a real agent).

Item collection (`scripts/rank_item_collection.py evaluations/paper`) is a
secondary readout — do **not** expect it to match CLIP; the frozen mean-pool
features cap task quality regardless of conditioning.

---

## 6. Caveats for interpretation

- **Not a pure A+B-only delta vs the anchor.** To stay stable, this cell also
  changed `lr 1e-3 → 3e-4` and added cosine decay (the anchor was `lr 1e-3`
  constant). The LR/schedule change is a *stability* necessity and cannot
  manufacture conditioning (it doesn't add language info), but a fully rigorous
  isolation would retrain the anchor at `lr 3e-4 cosine` too. The headline
  comparison (does the prompt move behavior at all) is unaffected.
- **Ceiling.** Best case this recovers the **conditioning** headline. CLIP-level
  item collection needs better features (last-token re-cache or LoRA), which is
  out of scope for this head-only fix.

---

## 7. Reproduce the training (optional)

```bash
python -c "
from imitation_learning import train_cached_head
train_cached_head(
    cache_dir='caches', cache_tag='llava_combined_lang_stride4',
    data_root='trajectories',
    out_weights='output/llava_combined_lang_stride4_fixAB/model.pt',
    batch_size=256, epochs=10, lr=3e-4, lr_schedule='cosine',
    device='cuda', num_workers=8,
    past_action_k=8, chunk_size=8, hidden_dim=2048, keep_best=True,
    feature_norm=True, image_dropout=0.15,
)"
```
Or via the cohort entry point: `cluster_pipeline.py ... --feature-norm --image-dropout 0.15`
(plus the COMMON anchor flags, but with `--lr 5e-4`/`3e-4 --lr-schedule cosine`).
On macOS use `num_workers=0` (the multiprocessing DataLoader path leaks
semaphores / gets killed); on a CUDA box `num_workers=8` is fine.
