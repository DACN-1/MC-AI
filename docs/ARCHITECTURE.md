# r1v-a Architecture Summary

A behaviour-cloning agent that learns to play MineRL by training only an MLP action head on top of a frozen vision-language backbone. The thesis: a frozen VLM is a competitive feature extractor for game-playing BC, and you can show this against a CLIP baseline without ever updating the backbone.

## Data flow

```
trajectories/                          contractor demonstrations on disk
└── trajectory_task_<task>_length_3000/
    ├── all_actions.json               { stem: [action_dict, ...] }   3000 frames per stem
    ├── all_infos.json                 { stem: { text_prompt, ... } }
    └── videos/video_<stem>.mp4        20 fps, 360×640, ~80 MB / 3000 frames
```

No PNG extraction, no HDF5 pre-processing. End-to-end path: `decord.VideoReader` decodes frames on demand in `TrajectoryDataset.__getitem__`. Cached path: a one-time backbone pass writes pooled embeddings to a memmapped `.npy` and every subsequent training run reads from that.

## Action space (`constants.py`)

```
NUM_BINARY        = 21    binary actions (attack, forward, hotbar.1..9, ...)   BCE loss
NUM_CAMERA        = 2     camera axes (x, y)
NUM_CAMERA_BINS   = 11    mu-law bins per axis                                  CE loss per axis
NUM_OUTPUT_LOGITS = 43    per-chunk-step head logits = 21 + 2*11
PAST_ACTION_DIM   = 43    per-action one-hot feature dim (input to head)
```

Camera bins match BASALT contractor format exactly — recorded angles are bin centers, so discretization is lossless. `action_to_tensor` / `action_to_onehot` are the two encoders; both accept env-format scalars and contractor `[v]`-wrapped lists.

## Model variants

All three share `forward(images, texts, past_actions=None) -> (B, chunk_size, NUM_OUTPUT_LOGITS)`. Both VLAAgent and FrozenVisionAgent expose `.encode(images, texts)` so `feature_cache.precompute` can call either backbone uniformly.

| Model | Backbone | Feature dim | Use case |
|---|---|---|---|
| `VLAAgent` | Frozen LLaVA-1.5-7B (FP16) | 4096 (Llama hidden) | Primary model |
| `FrozenVisionAgent` | Frozen CLIP-L (image + text encoders) | 1536 (2 × 768, concat) | Baseline; `use_language=False` zeros the text branch but keeps shape |
| `HeadOnlyAgent` | none — accepts pre-pooled features | varies (loaded from cache meta) | Cached-feature training |

**Backbone freezing.** LLaVA / CLIP weights are wrapped in `requires_grad_(False)`. Only `action_head` trains. Heads stay in FP32 for optimizer stability.

## The action head

```python
nn.Sequential(
    nn.Linear(feature_dim + past_action_dim, hidden),   # widens with past-action concat
    nn.ReLU(),
    nn.Linear(hidden, NUM_OUTPUT_LOGITS * chunk_size),  # widens with chunking
)
```

Output reshaped to `(B, chunk_size, NUM_OUTPUT_LOGITS)`. For `chunk_size=1` callers can index `[:, 0, :]` for the legacy 2-D shape.

## Temporal context (two orthogonal levers)

| Flag | Effect | Mechanism |
|---|---|---|
| `--past-action-k K` (default 8) | Head sees last K actions | One-hot encoded, flattened, concat to head input. Zero-padded for trajectory start. Most-recent action at the last slot. |
| `--chunk-size N` (default 1) | Head predicts next N actions | Output widens to `N × NUM_OUTPUT_LOGITS`; loss averages over chunk axis; last N−1 frames per trajectory dropped from training. |

For `collect_wood` / `collect_dirt` the recommended recipe is `--past-action-k 8 --chunk-size 8`: past-action collapses attack-run stickiness uncertainty; chunking reduces compounding error on long action runs.

## Loss (`vla_loss`)

```
BCE(binary_logits, binary_targets) + 0.5 * (CE(cam_x_bins) + CE(cam_y_bins))
```

Accepts `(B, NUM_OUTPUT_LOGITS)` or `(B, N, NUM_OUTPUT_LOGITS)` — chunk axis is flattened so every step contributes equally.

## Two training paths

### Default: cached features (`cluster_pipeline.py`)

```
[once per (backbone, task, use_language)]                  [per ablation cell]
  enumerate_samples (sorted)                                 train_cached_head
       ↓                                                          ↓
  backbone.encode() in batches  ──→  cache_dir/<tag>.npy  ──→  CachedFeatureDataset
       ↓                                  (FP16 memmap)            ↓
  flush every 100 batches                  + <tag>.json         HeadOnlyAgent
       ↓                                   + <tag>.progress     ↓
  cache_dir/<tag>.progress                                     output/<cell>/model.pt
```

The cache is invariant to `past_action_k` and `chunk_size` — sweep those without re-caching.

### Fallback: end-to-end (`--end-to-end`)

Runs backbone forward every batch. Same `VLAAgent` + `TrajectoryDataset` + `train_vla`. Used for sanity-check runs and rollout validation; not for the ablation grid (53× slower than cached).

## Resumability (both layers)

- **`feature_cache.precompute`**: writes `<tag>.progress` atomically (tmp + rename) every 100 batches. Validates the existing `<tag>.json` metadata on restart (sample count, feature_dim, backbone, language flag); mismatch triggers a clean rebuild, otherwise the memmap reopens in `r+` mode and encoding resumes from the recorded sample. Worst case loses ~100 × `batch_size` samples on crash.
- **`train_vla` / `train_cached_head`**: full checkpoint after every epoch via `_atomic_save` (tmp + rename). Stores `state_dict + optimizer_state + epoch + training_metrics + config`. Auto-resumes from epoch+1 unless `--restart` is passed. Legacy final-only checkpoints (no `epoch` key) start fresh.

Both survive SLURM time limits, OOM kills, and node reboots without manual cleanup.

## Inference / rollout (`run_rollout.py`)

`_load_agent` reads `past_action_k`, `chunk_size`, `use_language` from the checkpoint config and rebuilds the model. Per-episode state:
- `past_buffer: deque` of last K one-hot actions, zero-init.
- Each step: encode frame + prompt, concat past-buffer, forward. From `(1, N, 43)` chunk output take `[:, 0, :]` (first step) and execute. Append the executed action's one-hot. **Re-plan every tick** — no temporal ensembling yet.

`map_to_minerl_action` handles logits → env action dict (sigmoid threshold for binary, argmax + undiscretize for camera, merge over `env.action_space.no_op()` for `pickItem`/`swapHands` we don't predict).

## Checkpoint format

```python
{
    # end-to-end checkpoint also has: "llava_model": <hf id>
    "cache_tag": "llava_chop_a_tree_lang",   # cached path only
    "state_dict": <head params only>,
    "optimizer_state": <torch optim state>,  # for resume
    "epoch": <last completed epoch>,         # for resume
    "training_metrics": {train_loss, val_loss, ...},
    "config": {
        "feature_dim",                       # cached path only — head input dim
        "past_action_k", "past_action_dim",
        "chunk_size",
        "use_language",
        "hidden_dim",
        "num_output_logits",
        "camera_quantizer": {...},
        ...
    },
}
```

Pre-temporal / pre-resume checkpoints load with the missing fields defaulted (0, 1, no resume).

## Experimental design

2×2 ablation per task, full temporal recipe held on for every cell:

|  | With prompt | Without prompt |
|---|---|---|
| **LLaVA + temporal** | Exp 1 | Exp 3 |
| **CLIP + temporal** | Exp 2 | Exp 4 |

× 2 tasks (`chop_a_tree`, `collect_dirt`) = **8 head-training runs**. Each cell is selected via env vars to `slurm_train.sh`:

```bash
for backbone in llava clip; do
  for task in chop_a_tree collect_dirt; do
    for lang in 1 0; do
      BACKBONE=$backbone TASK_FILTER=$task USE_LANGUAGE=$lang sbatch slurm_train.sh
    done
  done
done
```

Output is auto-tagged: `output/<backbone>_<task>_<lang|nolang>/{model.pt, metrics.json}`. Caches live at `caches/<backbone>_<task>_<lang|nolang>.{npy,json,progress}` and persist across runs.

What each axis tests:
- **Backbone** (LLaVA vs CLIP): does the joint vision-language backbone earn its keep over a contrastive image+text encoder?
- **Language** (prompt vs no prompt): does the text-pathway signal matter, or is the visual feature carrying all the BC information?

Temporal is held on across all cells so neither backbone is handicapped. The recipe (`--past-action-k 8 --chunk-size 8`) goes in the methods section, not the ablation matrix.

## Compute budget

| Path | One-time cost | Per cell | 8-cell ablation (2 tasks) |
|---|---|---|---|
| End-to-end, 10 epochs | — | ~160 h LLaVA / ~60 h CLIP | **~1280 h (~53 days)** |
| Cached, build phase | ~26 h LLaVA / ~6 h CLIP per cache | — | **~128 h (~5.3 days)** |
| Cached, head training | — | ~30 min | **~4 h** |
| **Cached total** | | | **~5 days** |

Cache storage: ~25 GB / LLaVA tag + ~10 GB / CLIP tag = **~140 GB total** on the 2 TB NVMe.

Per-sample times are public-benchmark estimates for A5000 + FP16; expect ±50%. Measure on one mini-run before launching all 8.

## Repo map

```
constants.py                 Canonical keys, action_to_tensor, action_to_onehot, sizes
vpt_camera.py                CameraQuantizer (mu-law) — vendored from VPT
VLAAgent.py                  Frozen LLaVA + head; .encode() for caching
frozen_vision_baseline.py    Frozen CLIP image+text + head; .encode() for caching
feature_cache.py             precompute() (resumable), CachedFeatureDataset, HeadOnlyAgent, CLI
imitation_learning.py        TrajectoryDataset, vla_loss, train_vla, train_cached_head,
                             evaluate, evaluate_cached, _atomic_save, _try_resume_training, CLI
action_mapping.py            logits -> MineRL action dict
run_rollout.py               MineRL runner with past-action buffer + chunk-first-step
cluster_pipeline.py          Default: cached path. --end-to-end opts into legacy.
                             --restart forces fresh training.
slurm_train.sh               BACKBONE/TASK_FILTER/USE_LANGUAGE env-var driven; auto-tags output
tests/                       Action encoding, loss shapes, HeadOnlyAgent contract,
                             atomic save, progress roundtrip, resume against snapshot
```

## Key design decisions

1. **MP4 stays on disk; decode on demand.** Dropped HDF5 — 14× storage blowup for an inode problem that no longer applies.
2. **Always-3-D head output, even at `chunk_size=1`.** Unifies loss / eval / inference; index `[:, 0, :]` if you want the 2-D shape.
3. **Past-action as one-hot concat at the head, not as prompt text.** Direct signal to the trainable parameters; not diluted across 580+ LLaVA tokens.
4. **Cache invariance to head architecture.** Past-action / chunking knobs don't affect cached features, so one cache serves the entire sweep.
5. **`use_language` as a runtime flag on a constant-shape head**, not a separate model. Lets one head architecture serve both ablation cells.
6. **Caching is the default path.** Without it, the ablation is 53 days of GPU time vs 5; one-time cost pays for itself on the very first re-run.
7. **Backbone freezing.** The project's value depends on this — the head is the only trainable surface. Caching is the natural consequence.
8. **Atomic checkpoints everywhere.** Both `<tag>.progress` and `model.pt` written via tmp + POSIX rename. Memmap flushed before progress is recorded, so progress can only underreport — never overreport — actual data on disk.

## Commit history (latest first)

```
80e9e42  feat: resumable cache builds + per-epoch training checkpoints
7d1496d  feat: default cluster pipeline to cached training, one job per cell
c7e90ae  feat: feature caching + CLIP text encoder for ablation
3ef90d5  feat: past-action conditioning + action chunking
0d54754  refactor: decode MP4s on the fly, drop offline HDF5 conversion
d851362  refactor: categorical mu-law camera, vendor VPT slice, drop submodule
```
