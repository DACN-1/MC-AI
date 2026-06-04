# First attempt: frame-stride validation + first closed-loop rollout

## TL;DR

We validated the **frame-stride** subsampling end-to-end and got the **first
closed-loop rollout** of a trained head. The head *learns the features* fine but
**collapses to a frozen, idle agent** in rollout — the exact "no-move" failure
`no_move_fix.md` predicted. Root cause is structural, not a bug: **frozen-backbone
features that are globally pooled discard the spatial signal needed to act.** Next
attempt: stay frozen (project thesis), switch to **CLIP** (cheaper, runs free), and
fix the **pooling** (the real lever) plus the no-move loss/decode. Honest ceiling:
this can stop the freezing and study ablations, but a *competent* chopper would need
unfreezing/RL, which is out of scope by design.

---

## What happened (this session)

### 1. Frame-stride implemented + validated ✅
- Added `--frame-stride N` through `enumerate_samples` / `precompute` /
  `cluster_pipeline.py` / `slurm_train.sh`. Caches every Nth frame; targets and
  past-actions stay full-resolution (`CachedFeatureDataset` reconstructs them from
  `all_actions.json`), so the rollout policy is unchanged — only the cached
  *training* frames are thinned.
- Built a stride-4 cache for **chop · LLaVA-1.5-7b · language** on a rented RTX 5090
  (Blackwell, sdpa — no FA2 on sm_120): 798,750 = 3.195 M ÷ 4 frames, 6.09 GB,
  **measured 15.8 samples/s → ~14 h, ~$13**.
- Trained the MLP head (cached): val_loss 0.745 → 0.713 (converged ~epoch 5, no
  overfit). Eval: binary acc 0.978, **attack F1 0.957**, camera MAE **0.61°**.
- **Verdict:** the frame-stride code path is correct and the head learns from
  subsampled features. Stride-4 is viable → cheap full-run route is real.

### 2. First closed-loop rollout ❌ (informative negative)
- Rolled out the stride-4 head on a rented **RTX 3090 ($0.215/hr)** using the
  project's MineRL Dockerfile *recipe run natively* (the box's 16 GB-free Mac can't
  hold LLaVA-7B + MineRL; a 24 GB GPU can). Env `MineRLChopATree640-v0`,
  `--color-match on`, 300 steps.
- **Result: total idle.** `action_counts: {"none": 300}`, reward 0. Per-step data:
  - all binary sigmoids below the 0.5 decode threshold: `forward` 0.107 (max 0.178),
    `jump` 0.079, **`attack` 0.052 (max 0.256)** — even the 83%-demo action can't fire.
  - camera frozen at exactly 0° (argmax always the 92%-majority bin): mean/std/max = 0.
- This reproduces **all three** `no_move_fix.md` mechanisms (unweighted-loss base-rate
  collapse + greedy/threshold decode + past-action trap) on a *new* head
  (LLaVA/language/stride-4), proving the collapse isn't specific to the earlier
  CLIP/no-language cells.
- Artifacts saved locally: `output/rollout_chop_stride4/{episode_000.mp4,
  steps_000.json, run_summary.json}`.

### 3. Infra lessons
- **Abaki A5000s are QOS-blocked** (`abaki` QOS no longer in the `stud_ifi`
  association; only `normal`). The GPUs we *can* reach (`NvidiaAll`, RTX 2060 SUPER
  8 GB) are too small for LLaVA-7B (~14 GB). Restoring `abaki` QOS (admin email) would
  re-enable the free A5000 path.
- **Transatlantic transfer is slow**: cluster → California vast box capped ~13 MB/s
  (aggregate, not per-connection — 16 lftp streams only ~2× a single stream over the
  141 ms RTT). 65 GB took ~80 min. A European box, or building where the data already
  is, avoids this.
- **We lost the 6.09 GB cache** by destroying the box after only downloading
  `model.pt`. **Policy going forward: caches are the expensive reusable asset —
  always download/persist the `.npy`, never just the head.** (This is what task #12,
  the splice, is meant to protect.)

---

## What needs to change (next attempt)

### Design decisions (fixed by the user)
- **Frozen backbone stays** — it's the project thesis. This rules out the two things
  that actually make Minecraft agents competent (unfreezing/LoRA and RL fine-tuning).
  So the honest goal is *"the best agent a frozen backbone can give + clean
  ablations,"* **not** a reliable tree-chopper.
- **Switch to CLIP** (`openai/clip-vit-large-patch14`). Cheaper *and* potentially
  **free end-to-end** (probe-confirmed, see Infra below): cache/train on a free
  `NvidiaAll` node, roll out locally in Mac-Docker (CLIP + MineRL fit 16 GB).

### The real lever: POOLING (and its cache conflict)
- **Current CLIP pooling is the worst case.** `FrozenVisionAgent.encode` uses
  `clip.get_image_features(...)`, which returns the **global projected CLS embedding**
  — a single "what is this image about" vector with **no spatial layout**. (LLaVA used
  mean-pool over all tokens — also spatially lossy.) You cannot decide "turn left
  toward the tree" from a global vector. The rollout evidence — outputs nearly
  constant across frames (the head "ignores the image") — is consistent with exactly
  this.
- **Fix:** pool the **patch-token grid** instead:
  `clip.vision_model(pixel_values).last_hidden_state` → `(B, 257, 1024)` for ViT-L/14
  (256 patches + CLS), then a **learned attention-pool** (a query attending over the
  256 patches) → `(B, 1024)`. This preserves *where* things are. CLIP's clean spatial
  patches may actually suit this probe **better** than LLaVA's mean-pool — so it's a
  quality move, not just a cost move.
- **⚠️ Cache conflict (important):** a *trained* attention-pool is **head-side**, so a
  feature cache would have to store the full patch grid (256×1024 per frame ≈ **256×**
  today's cache → ~TB at full dataset). Infeasible. Options:
  1. **End-to-end CLIP, no cache** — CLIP forward is cheap (~7–15 ms) and fits 8 GB, so
     re-running it per epoch is fine, especially on a subset. **Recommended for the
     smoke test.**
  2. Cache a *bounded* representation (e.g. top-k patches, or a fixed coarse region
     pool) and attention-pool the bounded set in the head — keeps caching but caps the
     spatial fidelity.
  3. Keep global pooling and accept the spatial ceiling (status quo — known to freeze).

### The no-move fix (necessary, not sufficient)
- Apply `--weighted-loss` (class-balanced) + `--history-dropout 0.5` (already in
  `train_cached_head`/`train_vla`) to lift the sub-threshold probabilities and break
  the past-action trap.
- **Add the missing piece:** a **`--sample` decode mode** in
  `action_mapping.map_to_minerl_action` (Bernoulli-sample binaries, softmax-sample
  camera) — listed as "still to do" in `no_move_fix.md`. Threshold-0.5 decode makes
  rare actions mathematically un-selectable; sampling fixes that, but only pairs well
  with the weighted retrain.
- **This un-freezes the agent; it does not create competence.** If the features are
  frame-blind, you trade a frozen statue for a fidgeting one with reward ≈ 0.

### The decisive metric
- Not reward. After the pooling change + weighted retrain, measure **per-frame output
  variance** (std of the action logits across the rollout) and whether the camera
  **turns toward trees**.
  - Variance rises + scene-dependent → frozen-CLIP-with-spatial-pooling has real
    signal; keep pushing (temporal head, etc.).
  - Still near-constant → the **frozen backbone is the wall**, which is itself a clean,
    publishable ablation result (and would be the evidence that justifies the
    out-of-scope unfreeze/RL work).
- Make `run_rollout` report this automatically.

### Optional (composes with frozen + cache)
- Small **temporal head** (GRU / tiny transformer over a window of *frame* embeddings,
  not past-action one-hots) — Minecraft is partially observed; one frame can't tell
  you where the tree went.

---

## Concrete next experiment (≈ $0, free path)

Probe-confirmed feasible on LMU `NvidiaAll` (normal QOS) + local Mac-Docker:

1. **Pooling**: change `FrozenVisionAgent` to attention-pool `vision_model
   .last_hidden_state` patches (and decide cache vs end-to-end per above — recommend
   end-to-end CLIP for the smoke test).
2. **Decode**: add `--sample` to `action_mapping.map_to_minerl_action`.
3. **Train** two heads on a single-task (chop) CLIP cell on a free `NvidiaAll` node:
   **baseline** vs **attention-pool + weighted + history-dropout**. Minutes each.
4. **Roll out both locally** in Mac-Docker (`MineRLChopATree640-v0`, `--color-match on`,
   `--sample`), auto-reporting **per-frame action variance** + reward.
5. **Decide** from the variance metric whether frozen-CLIP has usable spatial signal.

### Free-path infra notes (probe-confirmed 2026-06-03)
- `NvidiaAll` idle node: torch 2.4.1+cu121, **RTX 2060 SUPER 8.2 GB** (fits CLIP),
  `transformers` 4.49 + `decord` in `~/BIG/.venv`, **94 GB scratch free** on `/var/tmp`,
  **HuggingFace egress OK** (model downloads on-node), `normal` QOS schedules fine.
- Caveat: TMPDIR `/tmp/user/<uid>` perm-denied warning (non-fatal; falls back to
  `/tmp`). Irrelevant to caching; *would* matter for Java/MineRL — but rollout runs on
  the Mac, so it doesn't bite here.
- Data: `~/BIG/trajectories/*.tar.gz` on NFS; extract chop (or a subset for the smoke
  test) into the node's `/var/tmp` (94 GB free fits the 65 GB chop set; a subset is
  safer and faster).

---

## Status of related work items
- Frame-stride: implemented, pushed to cluster, validated. ✅
- Cache-splice (single-task → combined): planned (task #12), gated on stride-4 going to
  production. Independent of the CLIP/pooling work.
- `model.pt` (stride-4 LLaVA/lang head) + rollout artifacts: saved under
  `output/`. The 6.09 GB feature cache was **not** saved (box destroyed) — would need a
  rebuild to iterate heads on it.
