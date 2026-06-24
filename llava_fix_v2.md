# llava_fix_v2.md — FiLM conditioning head: deploy & evaluate

**For:** an agent taking the FiLM (fix C) LLaVA head from checkpoint to a MineRL
conditioning verdict. This is **v2** of the post-cache LLaVA fix; v1 (fix A+B,
LayerNorm + image-dropout) is in `llava_fix.md`.
**Question:** can **post-cache** FiLM make the prompt move the policy, where v1's
concat + image-dropout could not?

---

## 0. Why v2 — what v1 showed

On the frozen LLaVA-lang cache (`llava_fix.md`):

- **Fix A (LayerNorm) worked, big.** The degenerate anchor (inventory-mashing
  ~60%) became a functional agent (inventory ~1%, attack ~74%, **dirt 0.0 → 2.5/ep**,
  competitive with CLIP). The collapse was a feature-normalization problem.
- **Fix B (image-dropout) did NOT recover conditioning.** Attack stayed flat
  across prompts (A→C **−0.2 pp**; anchor −5 pp; CLIP-lang anchor **−48 pp**).
  With a **concat** head the image is always present at rollout, so the head reads
  task-identity from it and zero-weights the text path; dropout only taught a
  fallback for *missing* image, not to *use* text when image is present.

**FiLM removes the "ignore text" escape hatch.** text_pool generates a per-channel
scale γ and shift β that modulate the image; the MLP consumes **only the modulated
image**:

```
modulated = (1 + gamma(text)) * image + beta(text)
```

- Text is **structurally injected** — there is no weight the head can zero to
  ignore it (unlike concat).
- **Synergy with image-dropout:** when the image is dropped in training,
  `modulated → beta(text)` alone, forcing β to encode the task — and β is added at
  rollout too, so the prompt shifts behavior on every frame.
- **Identity init** (`film_gen` weight=bias=0 → γ=1, β=0) starts the head as an
  image-only MLP (stable), then learns to modulate.

Implementation: `feature_cache.HeadOnlyAgent(film=True)`; flags `--film` /
`--image-dropout` in `cluster_pipeline.py`; rebuilt at rollout by
`agent_loader._load_cached_head_agent` from `config["film"]`/`["feature_norm"]`.
**No deploy-side code changes needed.**

---

## 1. Training result (health only — NOT the verdict)

Trained on the frozen LLaVA-lang cache, anchor recipe + `feature_norm` +
`image_dropout=0.25` + `film`, `lr 3e-4` + cosine, 10 epochs, `keep_best`. Clean,
monotonic, no divergence. Best **`val_movement_f1 = 0.4518` (epoch 8)** — the
**best of all LLaVA heads**:

| head | val_movement_f1 |
|------|--:|
| `llava_lang_anchor` (degenerate) | 0.394 |
| `llava_lang_fixAB` (LayerNorm + dropout, concat) | 0.413 |
| **`llava_lang_film` (FiLM)** | **0.452** |

> F1 only confirms the head learned to imitate and is **not** collapsed. It does
> **not** measure language conditioning. The verdict is the rollout in §4.

---

## 2. The checkpoint

```
output/llava_combined_lang_stride4_film/model_best.pt   <- DEPLOY THIS (epoch 8)
output/llava_combined_lang_stride4_film/model.pt        <- last epoch (fallback)
```
- `model_best.pt` sha1 `de5ecfbaa9fa45c131ef5f8369578da5bd264478` (~520 MB — larger
  than v1 because `film_gen` adds a 4096→8192 Linear + its optimizer state).
- **Cached-head checkpoint** (carries `cache_tag`, no `llava_model`): at rollout
  `agent_loader` rebuilds the frozen **LLaVA-1.5-7B** backbone, runs `encode()`
  **live per frame**, then this head. `feature_norm=True` + `film=True` are rebuilt
  from config (state_dict carries `film_gen.*`, `film_ln_image.*`, `film_ln_text.*`);
  `image_dropout` is a no-op in eval.
- Config: `feature_dim=8192` (image 4096 ‖ text 4096), `hidden_dim=2048`,
  `past_action_k=8`, `chunk_size=8`, `use_language=True`.

---

## 3. Hardware — READ FIRST
Rolls out the **live LLaVA-7B backbone per frame** (cache is NOT used at rollout).
- **Use a CUDA GPU box** (`DEVICE=cuda`); needs HF `llava-hf/llava-1.5-7b-hf`
  (~14 GB) + MineRL Docker. Cache `.npy` and `trajectories/` are not needed.
- **Do NOT run on the 16 GB Mac** — sustained LLaVA+MPS+MineRL thrashes
  (`memory/project_mac_eval_thrash.md`).

---

## 4. Deploy: one command

```bash
cd /path/to/r1-va
EVAL_ROOT_BASE=evaluations/paper EPISODES=10 DEVICE=cuda \
  bash scripts/eval_suite.sh \
    output/llava_combined_lang_stride4_film/model_best.pt \
    llava_lang_film \
    0
```
`eval_suite.sh` handles everything: starts `inference_server.py`, runs the 4
conditions (A `""` · B `"Play Minecraft."` · C `"chop a tree"` · D `"collect dirt"`)
× 10 ep × 1000 steps with identical seeds, the **fixed decode**
(`--sample --temperature 1.0 --camera-temperature 2.0 --color-match auto`,
symmetric with every CLIP/LLaVA paper cell — do not override), the stale-PID
guard, and `manifest.json`. Output → `evaluations/paper/<endtime>_llava_lang_film/`.
Kill the server after: `kill $(lsof -t -i:8765)`.

---

## 5. The verdict

```bash
python scripts/eval_compare.py --root evaluations/paper \
  --models llava_lang_film llava_lang_fixAB llava_lang_anchor clip_lang_anchor clip_nolang_anchor
```

Read **attack-firing %** for `A_chop_nocap / B_chop_ood / C_chop_task` (table 1).
The signal is the **A→C drop** (attack should fall on the on-task prompt).
Reference numbers already in `evaluations/paper`:

| cell | A | B | C | A→C Δ | conditions? |
|------|--:|--:|--:|------:|---|
| `clip_lang_anchor` | 84.9 | 76.7 | 37.0 | **−47.9 pp** | ✅ strong |
| `clip_nolang_anchor` (control) | 57.8 | 57.8 | 64.0 | +6.1 pp | flat |
| `llava_lang_anchor` | 34.9 | 32.6 | 29.8 | −5.1 pp | ❌ |
| `llava_lang_fixAB` (concat+dropout) | 74.5 | 75.4 | 74.3 | −0.2 pp | ❌ |
| `llava_lang_film` (A+B+C) | 86.9 | 88.6 | 89.9 | **+3.0 pp** | ❌ |

### RESULT (rollout complete, 2026-06-23): conditioning NOT recovered.
Attack is flat at ~87–90% across all prompts (no drop on the on-task C). This is
the **"still flat"** outcome. BUT FiLM is the **best LLaVA head on the objective
metrics**: sane policy (inventory ~0.1%), val F1 0.452, and **dirt 4.0/ep @ 90% —
essentially matching CLIP's best (4.3/ep)**. So FiLM clearly *used* the features
well; it just couldn't make behavior depend on the prompt.

**Conclusion — two separate root causes, both now pinned:**
- **Degenerate policy = normalization problem → FIXED post-cache** (Fix A; confirmed
  by both fixAB and FiLM un-collapsing the anchor).
- **No conditioning = feature problem → NOT fixable post-cache.** Concat+dropout (B)
  *and* FiLM (C, which structurally forces text in via β) both come back null →
  strong evidence `text_pool` (mean-pool of a 2–3-word prompt through frozen LLaMA)
  is too weak a signal. Post-cache conditioning levers are exhausted; the remaining
  lever is **feature quality**: last-token re-cache or LoRA (see `llava_fix.md` §6).

Note: A, B, C **compose cleanly** — objective metrics climb monotonically
(F1 0.394→0.413→0.452, dirt 0.0→2.5→4.0). Stacking more head tricks won't recover
conditioning; the bottleneck is upstream of the head.

---

## 6. Caveats
- **Not an A+B+C-only delta vs the anchor:** this cell also uses `lr 3e-4 + cosine`
  (anchor was `lr 1e-3` constant) — a stability necessity that cannot manufacture
  conditioning. A fully rigorous isolation retrains the anchor at `lr 3e-4 cosine`.
- **FiLM consumes the modulated image only** (no raw-text concat) — all conditioning
  flows through γ/β, by design.
- **Ceiling unchanged:** best case recovers conditioning, not CLIP-level item
  collection (frozen mean-pool features).

---

## 7. Reproduce the training
```bash
python -c "
from imitation_learning import train_cached_head
train_cached_head(
    cache_dir='caches', cache_tag='llava_combined_lang_stride4',
    data_root='trajectories',
    out_weights='output/llava_combined_lang_stride4_film/model.pt',
    batch_size=256, epochs=10, lr=3e-4, lr_schedule='cosine',
    device='cuda', num_workers=8,
    past_action_k=8, chunk_size=8, hidden_dim=2048, keep_best=True,
    feature_norm=True, image_dropout=0.25, film=True,
)"
```
Cohort form: `cluster_pipeline.py … --feature-norm --image-dropout 0.25 --film
--lr 3e-4 --lr-schedule cosine` (+ COMMON anchor flags).
On macOS use `num_workers=0` (the multiprocessing DataLoader path leaks semaphores
/ gets killed); on CUDA `num_workers=8` is fine.
