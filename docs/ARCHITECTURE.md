# r1v-a — Architecture & Methods Specification

A behaviour-cloning (BC) agent that learns to play MineRL by training **only an
MLP action head** on top of a **frozen** vision-language backbone. The research
claim: a frozen VLM (LLaVA-1.5-7B) is a competitive feature extractor for
game-playing BC, demonstrable against a CLIP image+text baseline *without ever
updating the backbone*. The only trainable surface is a 2-layer head; the
experiment is a 2×2 over **backbone** (LLaVA vs CLIP) × **language** (prompt on
vs off).

This document is a dense, source-cited reference for the thesis methods section.
Every quantitative claim carries a `file:line` citation; line numbers were
verified against the working tree on 2026-06-16. Where a docstring disagrees
with executed behaviour, the executed value is used and the discrepancy flagged.

## 0. Repo map

| File | Role |
|---|---|
| `constants.py` | Canonical action keys, `action_to_tensor`/`action_to_onehot`, size constants |
| `vpt_camera.py` | `CameraQuantizer` (mu-law), vendored slice of OpenAI VPT |
| `VLAAgent.py` | Frozen LLaVA backbone + trainable head; `.encode()` for caching |
| `frozen_vision_baseline.py` | Frozen CLIP image+text backbone + head; `.encode()` for caching |
| `feature_cache.py` | `precompute()` (resumable), `CachedFeatureDataset`, `HeadOnlyAgent`, CLI |
| `imitation_learning.py` | `TrajectoryDataset`, `vla_loss`, `train_vla`, `train_cached_head`, `evaluate`/`evaluate_cached`, CLI |
| `action_mapping.py` | Logits → MineRL action dict (decode) |
| `cluster_pipeline.py` | Default = cached path; `--end-to-end` opts into the legacy path; one job per ablation cell |
| `agent_loader.py` | Gym-free checkpoint loader (both checkpoint flavours) |
| `run_rollout.py` | MineRL runner: past-action buffer, decode, reward recompute, video |
| `inference_server.py` | HTTP server hosting a checkpoint for remote (Dockerised) rollout |
| `eval_envs.py` | Custom 640×360 MineRL envs (chop / dirt, Fast variants), R/B-swap wrapper |
| `eval_logger.py` | Per-episode / per-run rollout metrics |
| `consolidate_metadata.py` | `actions/*.jsonl` + `infos/*.json` → `all_actions.json` / `all_infos.json` |
| `scripts/` | vast.ai 5090 build/probe/launch, data push, eval suite, eval compare |
| `tests/` | Action conversion, encode equivalence, split disjointness, camera-onset, frame-history, patch-grid |

Two distinct data paths exist and share the same head architecture and loss:
- **End-to-end**: `decord` decodes frames on demand in `TrajectoryDataset`;
  the frozen backbone runs every batch (`train_vla`). Used for sanity / rollout
  validation only.
- **Cached (default)**: a one-time backbone pass writes pooled embeddings to a
  memmapped fp16 `.npy`; every head-training run reads from that
  (`feature_cache.precompute` → `train_cached_head`). This is the ablation path.

---

## 1. Action space (`constants.py`)

23 env-facing canonical keys = **21 binary** + **2 camera axes**
(`NUM_BINARY=21`, `NUM_CAMERA=2`, `NUM_ACTIONS=23`; `constants.py:58-60`). The
21 binary keys, in order (`constants.py:32-54`):

```
attack, back, forward, jump, left, right, sneak, sprint, use, drop,
inventory, hotbar.1 … hotbar.9, ESC
```

The model emits a **wider logit vector** than 23: each camera axis is a
categorical over 11 mu-law bins, not a scalar. Per chunk-step the head outputs
**`NUM_OUTPUT_LOGITS = 43`** logits (`constants.py:64`), laid out as
(`constants.py:8-13`):

```
[0  : 21)   21 binary-action logits     -> BCE
[21 : 32)   camera_x  11 bin logits     -> cross-entropy
[32 : 43)   camera_y  11 bin logits     -> cross-entropy
```

(`NUM_CAMERA_BINS = 11`, `CAMERA_NULL_BIN = 5`; `constants.py:62-63`.) The same
slice boundaries are reused in the loss (`_CAM_X_SLICE`/`_CAM_Y_SLICE`,
`imitation_learning.py:240-241`).

**Past-action encoding.** `PAST_ACTION_DIM = NUM_BINARY + 2*NUM_CAMERA_BINS = 43`
(`constants.py:70`) — coincidentally equal to `NUM_OUTPUT_LOGITS` but a separate
constant (input vs output role). `DEFAULT_PAST_ACTION_K = 8`
(`constants.py:71`).

**Two encoders** (`constants.py`):
- `action_to_tensor(action_dict) -> (23,) float32` (`:108-123`): first 21 entries
  `float(bool(v))`; entries 21,22 are camera **bin indices** (mu-law
  discretized), stored as floats and cast to `long` at loss time.
- `action_to_onehot(action_dict) -> (43,) float32` (`:126-144`): the past-action
  feature — binary 0/1, then camera_x one-hot (11) and camera_y one-hot (11).
  Layout deliberately mirrors the output logits so the head reads a vector that
  "looks like" what it produces.

**Two input formats** are accepted transparently (`constants.py:74-105`):
- Contractor (`all_actions.json`): every value wrapped in a list, camera
  double-nested — `{"attack":[1], "camera":[[x,y]], …}`.
- Env step (gym): unwrapped scalars — `{"attack":1, "camera":[x,y], …}`.

`_unwrap_scalar` (`:74-84`) peels the list wrapper (note `bool([0])` is `True`,
so the wrapper must be peeled before casting); `_camera_xy` (`:87-105`)
normalises camera to `(x, y)` floats.

---

## 2. Camera quantizer (`vpt_camera.py`)

Mu-law per-axis quantizer vendored from OpenAI VPT (only the quantize/dequantize
slice; VPT's hierarchical action joining is dropped because the action space
here is flat). The default matches the BASALT contractor recordings exactly:

```
DEFAULT_CAMERA_QUANTIZER = CameraQuantizer(camera_maxval=10, camera_binsize=2, mu=10)   # vpt_camera.py:75-77
```

Derived counts (`vpt_camera.py:33-40`): `n_bins = 2*maxval/binsize + 1 = 11`,
`null_bin = n_bins // 2 = 5` (0°). **discretize** (degrees → int64 bin;
`:42-56`) clips to ±10, mu-law-encodes, then linear-bins; **undiscretize**
(bin → float32 degrees; `:58-67`) inverts. Because the recorded camera angles
*are* bin centers of this scheme, `discretize(load)` is lossless on demos
(`vpt_camera.py:13-14`).

**Exact bin centers (degrees), executed from code** (index 5 = null/0°):

```
[-10.0, -5.8095, -3.2154, -1.6095, -0.6154, 0.0, 0.6154, 1.6095, 3.2154, 5.8095, 10.0]
```

> **Staleness flag.** The module docstring (`vpt_camera.py:11`) lists the
> ±3-bin centers as ±3.13, but the code yields **±3.2154**. The ±1.61 and ±0.62
> docstring values round correctly; only the ±3.13 entry is stale. Cite the
> executed values above in the thesis, not the docstring.

---

## 3. Models

Three `nn.Module`s share the forward signature
`forward(images, texts, past_actions=None) -> (B, chunk_size, NUM_OUTPUT_LOGITS)`
and each is a **2-Linear / 1-ReLU MLP head — no dropout, no LayerNorm anywhere
in the repo** (grep-confirmed). `VLAAgent` and `FrozenVisionAgent` both expose
`.encode(images, texts)` so `feature_cache.precompute` can drive either backbone
uniformly; `HeadOnlyAgent` consumes pre-pooled features.

| | VLAAgent (LLaVA) | FrozenVisionAgent (CLIP) | HeadOnlyAgent |
|---|---|---|---|
| Backbone id | `llava-hf/llava-1.5-7b-hf` (`VLAAgent.py:24`) | `openai/clip-vit-large-patch14` (`frozen_vision_baseline.py:51`) | none — reads cache |
| Frozen | yes (`:71-72`) | yes (`:60-61`) | n/a |
| Compute dtype | **bf16 on CUDA**, fp16 on CPU; env/kwarg override (`:40-48`) | HF default (fp32) | fp32 |
| Attention | FA2 attempt → **sdpa** fallback (`:56-69`) | HF default | n/a |
| Pooling | split image/text **mean-pool**, post-RMSNorm (`:178-226`) | pooled image proj + text proj, or patch-grid (`:88-110`) | pre-pooled in cache |
| Feature dim | **8192** = 2×4096 (`:88-89`) | **1536** = 2×768 pooled, or `G²·1024+768` grid (`:73-80`) | `feature_dim · (1+frame_history_k)` (`feature_cache.py:460-462`) |
| Head | `Linear(8192+pa → 8192) → ReLU → Linear(8192 → 43·chunk)` (`:97-101`) | `Linear(fd+pa → 768) → ReLU → Linear(768 → 43·chunk)` (`:82-86`) | `Linear(fd+pa → hidden) → ReLU → Linear(hidden → 43·chunk)` (`:523-540`) |
| Head hidden | 8192 (configurable via `head_hidden_dim`) | **768 (fixed = embed_dim)** | `feature_dim` (configurable) |
| Head precision | fp32 (features cast at boundary, `:94-96,:175-176`) | fp32 | fp32 |

### 3.1 VLAAgent — frozen LLaVA-1.5-7B + head (`VLAAgent.py`)

**Loading / freeze / dtype / attention** (`:32-72`). `LlavaProcessor` +
`LlavaForConditionalGeneration`. Compute dtype defaults to **bf16 on CUDA**
(wider exponent keeps the sdpa softmax stable at batch ~32 without FA2), fp16 on
CPU; overridable via env `R1VA_LLAVA_DTYPE={fp16,bf16}` or the `compute_dtype`
kwarg (`:40-48`). Attention tries `flash_attention_2` and falls back to sdpa on
`(ImportError, ValueError, RuntimeError)` — Blackwell sm_120 has no FA2 kernel,
so the 5090 path runs on **sdpa by design** (`:49-69`). Every backbone parameter
is `requires_grad_(False)` (`:71-72`). LLaMA hidden size = 4096 (`:74-77`).

**Prompt construction** (`:138`): each text `t` becomes `f"<image>\n{t}"`
(unless it already contains `<image>`). The actual prompt text is **always**
passed to the processor — `use_language=False` is enforced at pooling, not at
the processor (`:136-137`). The single `<image>` placeholder is expanded to
`image_seq_length = 576` tokens at forward time (`:122-127`).

**Pooling — split image/text mean-pool** (`encode`/`_split_pool`, `:112-226`).
The post-norm final hidden state is captured via a **forward hook on the LLaMA
RMSNorm** (`language_model.model.norm`) rather than `output_hidden_states=True`,
which would stash all 32 layers (~10 GB at batch 64) and force the cache build
down to batch 16 (`:147-167`). Image-role and text-role tokens are mean-pooled
**separately** with masks (`_masked_mean`, `:220-224`); the pooled feature is
`cat([image_pool, text_pool], dim=-1)` → **dim 8192** (`:88-89,:174`). When
`use_language=False`, `text_pool` is zeroed *after pooling* (`:172-173`) — the
encoder still attends to the prompt (an unavoidable LLaVA architectural
property), but the head input is image-only. This split-pool fix (landed
2026-06-05) replaced a single 4096-d joint mean that was dominated ~576:5 by
image tokens and washed out the language signal entirely (`:121-127`).

**Head** (`:90-101`): `head_hidden_dim` defaults to `feature_dim = 8192`.
The head stays **fp32**; pooled features are cast to the head's weight dtype at
the encode boundary (`:175-176`). Forward concatenates `past_actions` onto the
pooled feature when `past_action_dim > 0` (required, else `ValueError`;
`:243-250`) and reshapes the flat output to `(B, chunk_size, output_dim)`
(`:252-253`).

### 3.2 FrozenVisionAgent — frozen CLIP image+text baseline (`frozen_vision_baseline.py`)

`CLIPModel` + `CLIPProcessor`, all params frozen, HF-default fp32, no
`device_map` (device inferred from `next(clip.parameters()).device`)
(`:58-61,:91`). `embed_dim = projection_dim = 768` for ViT-L/14 (`:63`).

Two feature modes (`:69-80`):
- **Pooled (default, `patch_grid==0`)**: `get_image_features` → `(B,768)`;
  `feature_dim = 2*embed_dim = 1536` (`:80,:98`).
- **Spatial grid (`patch_grid>0`)**: run `vision_model`, drop CLS, average-pool
  the patch tokens (`vision_hidden = 1024`) into a `G×G` grid via
  `pool_patch_grid` (`:17-38,:93-96`); `feature_dim = G²·1024 + 768` (`:73-74`).
  Motivated by the chop task needing *where* the trunk is — a global mean can't
  decode direction-to-target (`:20-26`).

Text branch: `get_text_features` → `(B,768)` when `use_language`, else a **zero
vector** of width `projection_dim` (`:99-109`). The concat
`[image_features ‖ text_features]` is always emitted so the head input width is
constant across the language ablation cells (`:75-79,:110`). Head hidden is
**fixed at `embed_dim = 768`** (not configurable, unlike VLAAgent) (`:82-86`).

### 3.3 HeadOnlyAgent — cached-feature head (`feature_cache.py:510-564`)

Structurally identical to `VLAAgent.action_head`:
`Sequential(Linear(feature_dim + past_action_dim, hidden_dim), ReLU,
Linear(hidden_dim, output_dim*chunk_size))`, `hidden_dim` defaults to
`feature_dim` (`:523-540`). For a LLaVA cache `feature_dim = 8192`
(× `(1+frame_history_k)` if frame history is concatenated, `:460-462`).

**Optional learnable BCE temperature** (`learnable_bce_temp`, off by default):
a per-binary-action `Parameter(ones(21))` that **divides only the binary
logits** (clamped ≥1e-3) in forward, sharpening/softening the inference sigmoid
without decode-time threshold flags (`:530,:544-563`).

---

## 4. Temporal context (two orthogonal levers, both default-off-able)

| Lever | Default | Effect | Mechanism |
|---|---|---|---|
| `--past-action-k K` | 8 | Head sees last K actions | K one-hot (43-d) actions concatenated to the head input |
| `--chunk-size N` | 1 | Head predicts next N actions | Output widens to `43·N`; loss flattens the chunk axis |

**Past-action concat.** Built per frame as a `(K, 43)` block, **zero-padded at
the trajectory start** and **right-aligned so the most-recent action is in the
last slot** (stable positional reading), then flattened to `(K·43,)`
(`imitation_learning.py:208-219`; cache side `feature_cache.py:464-473`). The
trainer sets the model's `past_action_dim = K · PAST_ACTION_DIM`, so K=8 →
`8·43 = 344`, and the head's first Linear in-features become `8192 + 344 = 8536`
for LLaVA. K=0 → empty `(0,)` tensor (no history at all).

**Chunking.** The head predicts the next N actions in one forward; targets are
`targets[idx : idx+N]` shape `(N, 23)`. The last `N-1` frames per trajectory are
dropped (no full target); the dataset hard-fails if `chunk_size > len(actions)`
(`imitation_learning.py:127-150`). At inference only the **first** chunk step is
executed by default and the agent replans every tick (see §10).

The full temporal recipe used across the ablation is `--past-action-k 8
--chunk-size 8`.

---

## 5. Loss (`vla_loss`, `imitation_learning.py:244-335`)

```
total = BCE(binary) + cam_ce_weight · ( CE(camera_x) + CE(camera_y) )      # cam_ce_weight = 0.5 default (:250,:333)
```

Returns `(total_loss, bce_value, camera_ce_value)` (`:335`).

- **3D → 2D flatten** (`:286-289`): chunked logits/targets `(B, N, D)` are
  reshaped to `(-1, D)`, so every chunk step contributes equally.
- **Slicing** (`:295-301`): `binary = logits[:, :21]`,
  `cam_x = logits[:, 21:32]`, `cam_y = logits[:, 32:43]`; camera targets are
  columns 21,22 cast `.long()`.
- **Binary** (`:308-319`): default `binary_cross_entropy_with_logits` (mean
  reduction), optional `pos_weight`. If `focal_gamma > 0`: focal-weighted BCE,
  `FL = (1-p_t)^γ · BCE_elem`, composes with `pos_weight`.
- **Camera CE** (`:320-333`): `cross_entropy` per axis (mean), optional
  `cam_weight` (shared 11-bin class weights). With `cam_sample_weight` (per-frame
  `(B,)`/`(B,N)`) it uses a weighted mean `Σw·ce / Σw` on the camera CE only
  (used for pre-attack aiming windows).

**Why categorical mu-law.** Camera was previously MSE-regressed on raw degrees,
which collapsed to ~0 on the heavily zero-inflated demonstrator distribution
(~90% at the 0° bin). Categorical CE on mu-law bins represents the bimodal
"stay still vs. turn ±X°" structure.

**Optional class weighting** (`compute_class_weights`, `:338-385`): sqrt-
compressed inverse-frequency. `pos_weight[i] = clip(sqrt(#neg/#pos), 0.5, 4.0)`;
`cam_weight[b] = clip(sqrt(mean_count/count[b]), 0.3, 5.0)`. The sqrt + tight
clip is deliberate — full `#neg/#pos` over-corrected catastrophically in rollout
(rare keys saturated, attack suppressed). All of `pos_weight`, `cam_weight`,
`focal_gamma`, `cam_sample_weight` are **off by default**.

---

## 6. Data

**Two BC tasks**, each a separate trajectory directory; every paper-grade cell
trains on the **combined** chop+dirt set (strict rule):
- `trajectories/trajectory_task_chop_a_tree_length_3000/`
- `trajectories/trajectory_task_collect_dirt_length_3000/`

| Task | Trajectories | Frames/traj | Raw frames |
|---|---|---|---|
| chop_a_tree | 1065 | 3000 | 3,195,000 |
| collect_dirt | 1016 | 3000 | 3,048,000 |
| **Combined** | **2081** | 3000 | **6,243,000** |

At the production **frame-stride 4**, cached samples = 6,243,000 / 4 =
**1,560,750** (matches the runbook sanity check `(1560750, 8192)` LLaVA /
`(1560750, 1536)` CLIP).

**Provenance.** The videos are **real MineRL POV** rendered by a MineDreamer
`play` run (a STEVE-1 / VPT agent driving the real engine), **not** diffusion
output (`eval_envs.py:4-7,283-292`). Each `info_*.json` records the generation
config (`in_model: vpt/2x.model`, `in_weights: steve1.weights`, `text_prompt`,
`cond_scale 6.0`, `freq 25`, `gameplay_length 3000`, plus a `results` block of
mined/inventory counts). Resolution **640×360 at 20 fps** (`run_rollout.py:407`,
`eval_envs.py:6,81`).

**R/B channel swap (load-bearing).** The training videos were saved through a
cv2 BGR↔RGB bug that **swapped red and blue**, so the model learned on
R↔B-swapped frames. At eval the POV is therefore R↔B-swapped to match the
training distribution (`eval_envs.py:18-23,283-322`; the swap is `pov[…, ::-1]`
in `ReinhardColorWrapper.observation`). This swap is the *only* systematic
difference between train and eval frames — there is no texture/diffusion gap.
The optional per-task Reinhard LAB colour transfer is **off by default**
(`--color-match auto`).

**On-disk schema.** `consolidate_metadata.py` folds per-video
`actions/action_*.jsonl` + `infos/info_*.json` into two files per task to cut
inode pressure: `all_actions.json` = `{video_stem: [action_dict, …]}`
(`:18-44`) and `all_infos.json` = `{video_stem: info_dict}` (`:54-88`).
`TrajectoryDataset` prefers the consolidated files and falls back to the legacy
per-stem layout; `all_infos.json` supplies `text_prompt` (default
`"play minecraft"`).

```
trajectories/
└── trajectory_task_<task>_length_3000/
    ├── all_actions.json            { stem: [action_dict, ...] }   3000 frames/stem
    ├── all_infos.json              { stem: { text_prompt, ... } }
    ├── videos/video_<stem>.mp4     640×360, 20 fps, R/B-swapped
    ├── actions/action_<stem>.jsonl (legacy fallback)
    └── infos/info_<stem>.json      (legacy fallback)
```

---

## 7. Training pipelines

### 7.1 Cached path (default — the ablation path)

**Step 1 — `feature_cache.precompute()`** (`feature_cache.py:200-350`).
Enumerates samples deterministically (sorted; `frame_stride` keeps
`range(0, len, stride)`), builds the frozen backbone in `.eval()`, probes
`feature_dim` on one sample, then encodes in batches and writes to an
`np.memmap` of dtype **float16**, shape `(N, feature_dim)`
(`:313-320,:338-339`). fp16 storage halves disk for negligible BC impact and
keeps the on-disk format identical to historical A5000 fp16 caches even though
the LLaVA backbone now computes in bf16. Defaults: `batch_size=32`,
`frame_stride=1`, `progress_interval=100` (`:200-213`).

*Cache identity* — tag `f"{backbone}_{task_part}_{lang|nolang}"`, with
`_stride{N}` / `_patch{G}` suffixes when set (`task_part = task_filter or "all"`;
`cluster_pipeline._cache_tag` uses `"combined"` when no task filter). Metadata
`<tag>.json` stores `backbone, llava_id, use_language, task_filter, frame_stride,
patch_grid, n_samples, feature_dim, dtype="float16", samples=[[stem, frame_idx],…]`
(`:293-307`).

**Step 2 — `train_cached_head()`** (`imitation_learning.py:929-1334`). Builds a
`HeadOnlyAgent` over `CachedFeatureDataset` and optimizes only the head.
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=lr)` (`:1150`) —
  default betas (0.9, 0.999), eps 1e-8, **no weight decay**.
- **Defaults**: `lr=1e-3`, `batch_size=256`, `epochs=10`,
  `lr_schedule="constant"` (`"cosine"` → `CosineAnnealingLR(T_max=epochs)`,
  stepped per epoch), `past_action_k=8`, `chunk_size=1` (`:929-962,:1158-1165`).
- **No** gradient clipping, **no** AMP/autocast, **no** LR warmup.
- **Loss**: `vla_loss(..., focal_gamma, cam_ce_weight, cam_sample_weight)`
  (`:1249-1253`). Validation loss is intentionally **unweighted** for
  comparable model selection (`:1276-1282`).
- **Split**: trajectory-level, stem-grouped (`split_indices_by_stem(..., seed=42)`)
  so every frame of a stem lands in exactly one of train/val/test — no temporal
  leakage (`:1036-1059,:873-921`). `--frame-level-split` reverts to the legacy
  frame-level `random_split(seed=42)`.
- **Keep-best** (`--keep-best`): snapshots the epoch with max validation
  *movement-F1* (mean F1 over back/forward/jump/left/right/sprint — attack F1
  saturates ~0.96 and isn't discriminative) to `<name>_best` (`:1218-1229`).

### 7.2 End-to-end path (`train_vla`, `imitation_learning.py:653-867`)

Runs the frozen backbone every batch over `TrajectoryDataset`; ~53× slower than
cached, used for sanity / rollout validation only (`cluster_pipeline.py
--end-to-end`).
- **Optimizer**: `torch.optim.Adam(model.action_head.parameters(), lr=lr)`
  (`:745`) — only the head; same Adam defaults, no weight decay.
- **Defaults**: `lr=1e-4`, `batch_size=8`, `epochs=2`, `past_action_k=8`,
  `chunk_size=1`, `use_language=True` (`:653-670`). No scheduler, no clipping,
  no AMP.
- **Split**: frame-level `random_split([train,val,test], seed=42)`; test indices
  saved to `test_set_<unix>.json` (`:702-718`).

### 7.3 Data pipeline `TrajectoryDataset` (`imitation_learning.py:36-234`)

One `decord.VideoReader(path, num_threads=1)` per worker per file in an LRU
`OrderedDict` bounded by `decoder_cache_size=64` (`:185-206`). `__getitem__`
decodes one frame on demand and returns
`(PIL.Image, task_text, target_chunk (N,23), past (K·43,))` (`:226-231`);
`_collate` stacks to `(list[img], list[text], (B,N,23), (B,K·43))`
(`:568-576`). Actions are read from `all_actions.json` (+ optional
`all_infos.json` for `text_prompt`) or the legacy per-stem files
(`:152-183`). No frame stride here — `TrajectoryDataset` uses every frame
(stride lives only in the cached path).

### 7.4 Past-action regularizers (both off by default)

Two functional masks applied **only to the past-action vector at train time**
(there is no `nn.Dropout`/`nn.LayerNorm` anywhere in the repo):
- `apply_history_dropout(pasts, p)` (`:523-537`): with prob `p`, zeros the
  **entire** `(K·43,)` vector for a sample — mimics start-of-trajectory zero-
  padding to force the head to read the frame. CLI `--history-dropout`
  (default **0.0**); applied in both training loops.
- `apply_past_action_slot_dropout(pasts, p, K)` (`:540-565`): **per-slot
  independent** Bernoulli — reshapes to `(B,K,43)`, draws `keep=(rand(B,K,1)≥p)`,
  zeros each slot independently. Models the noisy rollout buffer where the head
  consumes its own drifting predictions. CLI `--past-action-slot-dropout`
  (default **0.0**; 0.3 is the recipe found cleanest empirically); applied in
  `train_cached_head` only.

### 7.5 `CachedFeatureDataset` (`feature_cache.py:371-507`)

Opens the fp16 memmap read-only and indexes `cache_index[stem][frame_idx] → row`
(`:411-418`); reconstructs targets / one-hots from `all_actions.json` so one
cache serves any `chunk_size`/`past_action_k` (`:420-458`). `__getitem__` casts
the feature to **fp32** (`:482`), optionally prepends K stride-spaced previous
frame features (`frame_history_k`, zero-padded at start, oldest-first;
`:483-496`), slices the target `(N,23)`, and returns
`(feat, target, past, cam_w)` (`:478-507`). `feature_dim() = base · (1 +
frame_history_k)` (`:460-462`).

### 7.6 Resumability (both layers, atomic tmp+rename)

- `precompute` writes `<tag>.progress` (next-sample cursor) atomically every
  100 batches after `mm.flush()`; on restart it validates `<tag>.json` against
  the request (n_samples, feature_dim, backbone, language, task_filter, llava_id,
  frame_stride, patch_grid) and resumes the memmap in `r+` from the recorded
  sample, else rebuilds. Worst-case crash loss ≈ 100·batch_size samples.
- `train_vla`/`train_cached_head` write a full checkpoint after every epoch via
  `_atomic_save` (`:589-594`) carrying `state_dict + optimizer_state + epoch +
  training_metrics + config`; auto-resume from `epoch+1` unless `--restart`.

### 7.7 `cluster_pipeline.py` orchestration

One invocation = one ablation cell `(backbone, task_filter, use_language)`
(+ stride/patch). Default path: `_ensure_cache` builds/resumes the cache, then
`train_cached_head`, then `evaluate_cached` (`:448-542`). `--end-to-end` runs
`train_vla` + `evaluate`; `--skip-train` evaluates an existing `model.pt`.
Output layout: `<output-dir>/{model.pt, model_best.pt, metrics.json}` and the
persistent `<cache-dir>/<tag>.{npy,json,progress}`.

---

## 8. CLI defaults (cite-ready hyperparameters)

**`cluster_pipeline.py`** (`:136-372`) — the canonical entry point:
`--epochs 10`, `--batch-size 256`, `--lr 1e-3`, `--lr-schedule constant`,
`--cache-batch-size 32`, `--past-action-k 8`, `--chunk-size 1`,
`--hidden-dim None`, `--frame-stride 1`, `--patch-grid 0`, `--cam-ce-weight 0.5`,
`--backbone llava`, `--task-filter None` (None ⇒ combined). All
regularizer/sampler knobs default off: `--history-dropout 0.0`,
`--past-action-slot-dropout 0.0`, `--focal-gamma 0.0`,
`--frame-weight-multiplier 1.0`, `--chop-oversample-weight 1.0`,
`--camera-onset-weight 1.0`, `--frame-history-k 0`. Flags: `--no-language`,
`--weighted-loss`, `--cam-weighted-loss`, `--learnable-bce-temp`, `--keep-best`,
`--frame-level-split`, `--end-to-end`, `--restart`, `--skip-train`.

**`imitation_learning.py`** (`:1529-1581`, drives `train_vla`): `--epochs 2`,
`--batch-size 8`, `--lr 1e-4`, `--past-action-k 8`, `--chunk-size 1`,
`--num-workers 2`, `--val-split 0.1`, `--test-split 0.1`, plus `--no-language`,
`--restart`, `--evaluate-after`, `--weighted-loss`, `--history-dropout 0.0`.

**`feature_cache.py`** (`:570-608`): `--backbone {llava,clip}` (req),
`--task-filter` (req), `--batch-size 16`, `--frame-stride 1`, `--patch-grid 0`,
`--use-language`/`--no-language` (default True), `--llava-id llava-hf/llava-1.5-7b-hf`.

> **The 2×2 symmetry rule.** In every paper cell only **backbone** and
> **use_language** vary; everything else is held identical — hidden 2048, k=8,
> chunk=8, frame-stride 4, trajectory split, `--keep-best`, and the decode flags
> below. Cite the held-constant recipe in methods, not in the ablation matrix.

---

## 9. Checkpoint format

Saved atomically (tmp + rename, `imitation_learning.py:590`). **Two schemas**,
routed by keys at load time:

**End-to-end (`train_vla`, `:749-779`)** — has `"llava_model"` (backbone id);
`"state_dict"` = **action head only** (`model.action_head.state_dict()`);
`"config"` = `num_actions=23, num_binary=21, num_camera=2, num_camera_bins=11,
num_output_logits=43, past_action_k, past_action_dim, chunk_size, use_language,
epochs, batch_size, lr, val_split, test_split, weighted_loss, history_dropout,
camera_quantizer{camera_maxval, camera_binsize, mu}`.

**Cached (`train_cached_head`, `:1167-1216`)** — has `"cache_tag"` (no
`llava_model`); `"state_dict"` = **full HeadOnlyAgent** (`model.state_dict()`,
includes the learnable BCE temperature if enabled); `"config"` adds
`feature_dim, hidden_dim` and the recipe knobs `frame_weight_multiplier,
frame_weight_min_run, learnable_bce_temp, focal_gamma, past_action_slot_dropout,
chop_oversample_weight, cam_weighted_loss, cam_ce_weight, split_by_trajectory,
keep_best, frame_history_k, stem_filter, lr_schedule, camera_onset_weight,
camera_onset_window`.

Both also carry `"optimizer_state"`, `"epoch"`, `"training_metrics"` (per-epoch
`val_loss`, `val_movement_f1`, `val_per_action_f1`, …). Pre-temporal / pre-resume
checkpoints load with `past_action_k=0`, `chunk_size=1`, no resume.

---

## 10. Inference / rollout

**Loader** (`agent_loader.load_agent`, `agent_loader.py:115-156`;
`run_rollout._load_agent` delegates here). `torch.load(map_location="cpu")`,
requires a `"state_dict"` key, routes by flavour:
1. End-to-end (`"llava_model"`): rebuild `VLAAgent(...)`, load only the head;
   backbone is the pretrained frozen LLaVA (`:138-156`).
2. Cached (`"cache_tag"` / `config["feature_dim"]`): rebuild the matching frozen
   backbone (CLIP if `cache_tag` starts `clip` or `feature_dim<2048`, else
   LLaVA) + `HeadOnlyAgent`, wrapped in `_CachedHeadRolloutAgent` that runs
   `backbone.encode()` live per frame then the head (`:43-112`). `use_language`
   parsed from the `nolang` tag token; `patch<G>` parsed too. `frame_history_k>0`
   is **not** supported for rollout serving (`:52-59`).

Both return `(agent, {past_action_k, chunk_size, use_language})` so rollout
matches training-time conditioning.

**Decode** (`action_mapping.map_to_minerl_action`, `action_mapping.py:23-122`):
- **Greedy (default)**: binary = `sigmoid(logit) ≥ threshold` (default 0.5);
  camera = argmax per 11-bin block → `undiscretize` → degrees (`:107-116`).
- **Stochastic (`--sample`)**: binary = `Bernoulli(sigmoid(logit/T))`; camera =
  `multinomial(softmax(logits/T))` per axis (`:95-106`). Default `T=1.0`; a
  separate `camera_temperature` (eval suite uses **2.0**) flattens the dominant
  0° bin so the agent looks around (`:102-104`).
- Default-off, byte-identical-to-legacy calibration knobs: per-action
  `--binary-temperatures/-thresholds/-logit-bias`, and `--attack-hysteresis`
  (greedy: lower attack threshold after it fires) (`run_rollout.py:239-256,
  295-306`).
- Unpredicted keys (`pickItem`, `swapHands`) are merged from
  `env.action_space.no_op()` captured once (`run_rollout.py:333`).

**Temporal handling at rollout** (`run_rollout.py:387-473`). Per episode a
`past_buffer = deque(maxlen=K)` is zero-padded at start; each step appends
`action_to_onehot(minerl_action)` (43-d). Chunking: model returns
`(1, chunk_size, 43)`; by default `logits = plan[0]` (first step) and the loop
**replans every tick**. Opt-in modes: `--execute-steps K` (open-loop, run first
K chunk steps) and `--chunk-ensemble` (ACT-style temporal ensembling — average
the plans previously predicted for step t).

**Reward recompute (`InventoryRewardWrapper`, `run_rollout.py:161-236`).** In
this Docker/Malmo stack `RewardForCollectingItems` silently fails to fire on
item gains, so reward is recomputed from the inventory delta. Critical detail:
chopping yields **per-wood item names** (`oak_log`/`birch_log`/…), never a bare
`"log"` — which is why chop reward stayed 0 until this fix; dirt is a real item
name and already worked.

**Envs (`eval_envs.py`).** Custom ids registered on import:
`MineRLChopATree640-v0`, `MineRLCollectDirt640-v0`, plus `…Fast-v0` /
`…FastAim-v0` (`:189-236`). Built on `HumanControlEnvSpec` (full near-human
action space matching the 23 keys), 640×360 render (`:81-100`); per-task fixed
biome (forest / plains), bare-handed start, midday with time frozen,
`RewardForCollectingItems` (`:45-177`). Fast variants set
`BreakSpeedMultiplier=5.0` so a log breaks (~3 s → ~0.6 s) within a 1000-step
rollout (`:88-100`). MineRL is installed **from git HEAD** in Docker with
**Java 8** (`Dockerfile:6-22`); gym/minerl are deliberately not in
`requirements.txt` (rollout-only). Remote rollout: `inference_server.py` hosts
the checkpoint (MPS/CUDA/CPU) and `run_rollout._RemoteAgent` POSTs JPEG frames
over a keep-alive connection from the Dockerised env.

---

## 11. Evaluation

**(a) Offline test-set metrics** (`imitation_learning.evaluate` → `metrics.json`,
`:1439-1516`): per-action **precision / recall / F1** for all 21 binary keys
(sklearn `precision_recall_fscore_support(average="binary", zero_division=0)`),
`binary_accuracy` (sigmoid>0.5), `camera_x_bin_accuracy`,
`camera_y_bin_accuracy` (argmax), and `camera_mae_degrees` (mean |pred−target|
after undiscretize). Chunk axis flattened so every step counts. A real combined
stride-4 cell reports e.g. `test_samples 1,246,928`, `binary_accuracy 0.982`,
`cam_x_acc 0.822`, `cam_y_acc 0.812`, `camera_mae_degrees 0.729`.

**(b) Closed-loop rollout metrics** (`eval_logger.EpisodeLogger`). Per step:
the **dominant action** (first of attack/forward/back/left/right/jump/use, else
camera if |camera|>0.1, else none), reward, per-item inventory peak
(`:45-71`). Per episode → `episode_NNN.json`: `total_reward`, `steps`,
`success` (default `total_reward>0`), `action_counts`, `peak_inventory`
(running max per item — the **reward-independent task-completion signal**,
since the in-env reward handler is unreliable) (`:77-111`). Per run →
`run_summary.json`: `mean_reward`, `success_rate`, `action_frequency`,
`mean_peak_inventory`, `collect_rate` (`:117-186`). `run_rollout.py` also writes
per-step `steps_NNN.json` with the full `minerl_action` and the raw 43-d logit
vector — the input `eval_compare.py` consumes.

**Eval suite (`scripts/eval_suite.sh`).** Hardcoded `COMMON_FLAGS`: `--sample
--temperature 1.0 --camera-temperature 2.0 --color-match auto --record-video`,
default **10 episodes × 1000 steps**, fixed base seed. Four conditions, **same
seeds across conditions and models** so two models' `steps_*.json` are diffable
step-by-step:

| Cond | Env | Prompt |
|---|---|---|
| A_chop_nocap | MineRLChopATree640Fast-v0 | `""` |
| B_chop_ood | MineRLChopATree640Fast-v0 | `"Play Minecraft."` |
| C_chop_task | MineRLChopATree640Fast-v0 | `"chop a tree"` |
| D_dirt_task | MineRLCollectDirt640Fast-v0 | `"collect dirt"` |

**Stats (`scripts/eval_compare.py`).** Recomputes per-action **firing rate**
(fraction of steps each binary fires) + camera nonzero-%/std/signed-mean from
`steps_*.json`, with **Wilson 95% CIs**, a **two-proportion z-test** (pooled
variance, **Bonferroni**-corrected over n_actions × n_conditions × n_pairs) and
**Cohen's h** effect size; significance markers combine adjusted p with an |h|
floor (`:58-104`).

> **Eval nondeterminism caveat (cite this).** MineRL `reset` is **not
> bit-deterministic**: the same checkpoint scored reward 4.60 then 0.00 on
> identical seeds, and back-to-back runs differ in step-0 logits by ~0.23
> (`docs/llava_5090_runbook.md:9-26`). Hard rules: never rank recipes on
> ≤5-episode wrapper reward (reward needs 20+ episodes across distinct seeds);
> the **per-step firing-rate statistics are the primary instrument**, not
> episode reward. Per-episode seeding of env/numpy/torch
> (`run_rollout.py:362-381`) does not make reset fully deterministic.

---

## 12. Experimental design (2×2)

The headline grid is **backbone × language**: {LLaVA-1.5-7B, CLIP} ×
{use_language on/off}. The knob-free anchor cells are
`llava_combined_{lang,nolang}_stride4_tsplit` matched to
`clip_combined_{lang,nolang}_stride4`. Trained on the **combined** dataset; held
constant: hidden 2048, past-action k=8, chunk=8, frame-stride 4, trajectory
split, `--keep-best`, and the decode flags `--sample --temperature 1.0
--camera-temperature 2.0 --color-match auto`. Only `backbone` and `use_language`
differ.

The broader intervention space is **2×2×2** (language × past-action × chunking),
since `--no-language`, `--past-action-k`, and `--chunk-size` each compose
cleanly; `--past-action-k 0 --chunk-size 1` is the no-temporal baseline.

`use_language` semantics differ by backbone but are matched at the head:
- LLaVA zeros `text_pool` after split-pooling (head input image-only); the
  encoder still attends to the prompt (architectural, unavoidable).
- CLIP zeros the text branch (replaced by a zero vector of matching width).

Both pool as `[image ‖ text]` so language-on vs language-off compare
apples-to-apples (feature dims 8192 vs 1536). The headline behavioural question:
does LLaVA-lang shift attack firing across prompts A/B/C while LLaVA-nolang stays
flat — CLIP showed attack **82 / 71 / 29%** (nocap / ood / task) vs a flat
~61–66% (`docs/llava_5090_runbook.md`).

What each axis tests:
- **Backbone** — does the joint vision-language backbone earn its keep over a
  contrastive image+text encoder?
- **Language** — does the text pathway carry BC-relevant signal, or is the
  visual feature doing all the work?

---

## 13. Compute budget & infrastructure

**Canonical hardware: rented RTX 5090 (Blackwell, sm_120)**, bf16 + sdpa +
batch 32 (no FA2). The A5000 / LMU-Abaki path is deprecated (QOS revoked
2026-06-06) but kept for reference. Per-sample forward ≈ 15 ms LLaVA / 3.5 ms
CLIP (5090 estimates — measure with `scripts/probe_5090_throughput.py` before
any long rental).

| Phase | Cost |
|---|---|
| LLaVA cache build, stride 1 | ~25 h / tag (≈ ¼ at stride 4) |
| CLIP cache build, stride 1 | ~6 h / tag |
| Head training (cached, 10 ep) | ~30 min – 2 h / cell |
| End-to-end (no cache) | ~53× the cached cost |

Cache storage ≈ 50 GB / LLaVA stride-1 tag (fp16 × 8192 dims × 6.24 M) +
~10 GB / CLIP tag; ÷4 at stride 4. At ~$0.70/h 5090 spot, a full 8-cache stride-1
build is ~$90; the stride-4 Phase-C two-cache LLaVA build targets ~$10
end-to-end.

**Scripts (`scripts/`).** `setup_vastai_5090.sh` (idempotent: torch 2.7 from the
cu128 index + `requirements-blackwell.txt`, no flash-attn, verifies sm_120);
`probe_5090_throughput.py` (times the real `precompute` encode loop, prints
samples/sec + extrapolated $ cost); `launch_5090_phase_c.sh` (verifies stack +
staged data, probes, then builds both stride-4 LLaVA caches and trains both
heads); `push_data_to_5090.sh` (rsync code + ~120 GB trajectories from the LMU
login node); `setup_vastai_inference.sh` (minimal inference-only stack);
`eval_suite.sh` / `eval_compare.py` (§11). Legacy SLURM: `slurm_train.sh`
(A5000/Abaki), `slurm_train_nvidiaall.sh` (NvidiaAll), env-var driven
(`BACKBONE / USE_LANGUAGE / FRAME_STRIDE / HIDDEN_DIM / TASK_FILTER`).

**Dependencies.** `requirements.txt` = A5000/Turing-Ada (torch 2.2–2.5 + pinned
FlashAttention-2 `v2.6.3+cu123torch2.4`); `requirements-blackwell.txt` = 5090
(torch 2.7 cu128, **no** flash-attn). MineRL v1.0 + gym 0.23.1 only for the
Docker rollout path.

---

## 14. Tests (`python -m unittest discover tests`)

| Test file | Covers |
|---|---|
| `test_action_conversion.py` | `action_to_tensor`/`action_to_onehot`/`map_to_minerl_action`: contractor list-unwrapping, camera flat/numpy/clip, threshold decode, argmax+undiscretize, base-action merge, camera-temperature flattening |
| `test_encode_equivalence.py` | `VLAAgent` split image/text pooling (`_split_pool` masks/padding); opt-in LLaVA integration verifying the RMSNorm hook == `hidden_states[-1]`, image-token count == `image_seq_length`, no-language zeros `text_pool` |
| `test_split.py` | Trajectory-level split is exact, disjoint, deterministic, fraction-honouring |
| `test_camera_onset.py` | Onset-windowed camera-CE weighting (uniform == unweighted; weight shifts loss; backward runs) |
| `test_frame_history.py` | `frame_history_k` window order / padding (k=0 == legacy) |
| `test_patch_grid.py` | `pool_patch_grid` block-average shapes; rejects non-square patch counts |

---

## 15. Key design decisions

1. **Backbone freezing** is the project's premise — the head is the only
   trainable surface (`VLAAgent.py:71-72`, `frozen_vision_baseline.py:60-61`).
   Feature caching is the natural consequence.
2. **Caching is the default path.** Without it the 2×2 (×2 task) ablation is
   ~53 days of GPU time vs ~5; the one-time cost pays for itself on the first
   re-run, and one cache serves the whole `past_action_k`/`chunk_size`/recipe
   sweep (cache is invariant to head architecture).
3. **Split image/text mean-pooling** (LLaVA dim 8192 = 2×4096), mirroring CLIP's
   `[image ‖ text]`, so the language ablation compares apples-to-apples and the
   text signal isn't washed out by ~576 image tokens.
4. **Always-3-D head output** `(B, chunk_size, 43)` even at `chunk_size=1`,
   unifying loss / eval / inference; index `[:, 0, :]` for the 2-D shape.
5. **Past-action as one-hot concat at the head, not as prompt text** — a direct
   signal to the trainable parameters, not diluted across 580+ LLaVA tokens.
6. **`use_language` is a runtime flag on a constant-shape head**, not a separate
   architecture; one checkpoint can serve either ablation cell.
7. **Categorical mu-law camera** (CE on 11 bins) instead of MSE on degrees, to
   represent the bimodal stay/turn structure of the zero-inflated demos.
8. **bf16 + sdpa on Blackwell** (no FA2 sm_120 kernel); fp16 storage at the cache
   boundary keeps the on-disk format identical to the legacy A5000 caches.
9. **Atomic checkpoints everywhere** (`<tag>.progress` and `model.pt` via
   tmp + POSIX rename; memmap flushed before progress is recorded, so progress
   can only under-report — never over-report — data on disk).
10. **Firing rates, not reward, are the primary rollout instrument** — MineRL
    reset nondeterminism plus a previously-broken Malmo reward handler make
    episode reward unreliable at the episode counts that are affordable.
