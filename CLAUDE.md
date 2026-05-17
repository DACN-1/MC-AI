# r1v-a: Minecraft Vision-Language-Action Agent

## Project Overview

Research codebase for training a lightweight action head on top of a frozen
LLaVA-1.5-7B backbone, predicting MineRL gameplay actions from RGB frames via
behavioural cloning. The mu-law camera quantizer originally from OpenAI VPT is
vendored in `vpt_camera.py`; the VPT submodule itself is no longer part of the
project.

## Action Space

23 env-facing canonical action keys, but the model emits a wider logit vector:

- 21 **binary** actions (BCEWithLogitsLoss):
  `attack, back, forward, jump, left, right, sneak, sprint, use, drop, inventory,
   hotbar.1..9, ESC`
- 2 **camera axes** (`camera_x`, `camera_y`), each predicted as a categorical
  over `NUM_CAMERA_BINS = 11` mu-law bins (cross-entropy per axis).

Sizes from `constants.py`:

```
NUM_BINARY        = 21        binary actions
NUM_CAMERA        = 2         camera axes
NUM_ACTIONS       = 23        canonical action keys (env-facing)
NUM_CAMERA_BINS   = 11        bins per camera axis
NUM_OUTPUT_LOGITS = 43        per-chunk-step head logits = 21 + 2 * 11
PAST_ACTION_DIM   = 43        per-action feature dim (binary + 2 one-hots)
```

## Temporal context (optional)

Two independent flags add temporal context to the head, both default off so
that single-frame / single-step is the baseline:

- `--past-action-k K` (`DEFAULT_PAST_ACTION_K = 8`): concatenate the last K
  actions, one-hot encoded (`PAST_ACTION_DIM` each), to the head input. Zero-
  padded for the first K frames of each trajectory. Most-recent action is
  always at the last slot so the head's positional reading is stable.
- `--chunk-size N`: the head predicts the next N actions in one forward.
  Output tensor shape becomes `(B, N, NUM_OUTPUT_LOGITS)`. Loss averages over
  the chunk axis. Last `N-1` frames per trajectory are dropped (no full
  target). Inference (`run_rollout.py`) executes only the first predicted step
  and re-plans next tick.

Both interventions compose cleanly with `--no-language` (zeroes the text
prompt), forming a 2×2×2 ablation space: language × past-action × chunking.

The camera quantizer matches the BASALT contractor recordings exactly:
`CameraQuantizer(camera_maxval=10, camera_binsize=2, mu=10)`. The recorded
camera angles in `all_actions.json` *are* bin centers from this scheme — so
discretizing demos is lossless. Bin centers (degrees):

```
[-10.0, -5.81, -3.13, -1.61, -0.62, 0.0, 0.62, 1.61, 3.13, 5.81, 10.0]
```

## Files

```
r1v-a/
├── constants.py             Canonical keys + action_to_tensor() + size constants
├── vpt_camera.py            CameraQuantizer (mu-law) — vendored from VPT
├── VLAAgent.py              Frozen LLaVA backbone + trainable MLP head
├── frozen_vision_baseline.py  CLIP image+text baseline (text branch zeroed when use_language=False)
├── feature_cache.py         Precompute LLaVA/CLIP embeddings + CachedFeatureDataset + HeadOnlyAgent
├── imitation_learning.py    TrajectoryDataset, vla_loss, train_vla, train_cached_head, evaluate, CLI
├── action_mapping.py        Logits -> MineRL action dict (argmax + undiscretize)
├── chunk_frames.py          (deprecated) MP4/PNG -> HDF5 chunked frames
├── consolidate_metadata.py  actions/*.jsonl + infos/*.json -> all_actions.json
├── cluster_pipeline.py      Train -> evaluate (calls into above)
├── run_rollout.py           Run trained agent (or random) in MineRL
├── eval_logger.py           Per-episode/per-run rollout metrics
├── slurm_train.sh           SLURM job that invokes cluster_pipeline.py
├── Dockerfile, run_minerl.sh, docker-compose.yml, test_minerl.py
└── tests/test_action_conversion.py
```

## Data Layout

`TrajectoryDataset` reads frames directly from MP4 files (via `decord`) and
actions from either consolidated JSON or legacy per-video files:

```
trajectories/
└── trajectory_task_<task>_length_<N>/
    ├── all_actions.json            # preferred — {"<stem>": [action, ...], ...}
    ├── all_infos.json              # optional  — {"<stem>": {info_dict}, ...}
    ├── videos/video_<stem>.mp4     # decoded on demand at __getitem__ time
    ├── actions/action_<stem>.jsonl # legacy fallback
    └── infos/info_<stem>.json      # legacy fallback
```

Frames are no longer pre-extracted to HDF5. One `decord.VideoReader` is cached
per DataLoader worker per file (bounded LRU, default 64 files per worker).

`all_actions.json` (BASALT contractor format) wraps every value in a
single-element list:

```jsonc
{
  "<stem>": [
    {"attack":[1], "forward":[0], "camera":[[1.609, -10.0]], ...},
    ...
  ]
}
```

The env step format unwraps the lists: `{"attack": 1, "camera": [x, y], ...}`.
`constants.action_to_tensor` accepts both.

## Common Workflows

### 1. Train (reads MP4s directly — no conversion step)
```bash
python imitation_learning.py \
    --data-dir ./trajectories \
    --out-weights ./models/vla.pt \
    --epochs 10 --batch-size 16 --lr 1e-4 \
    --num-workers 8 \
    --past-action-k 8 --chunk-size 8 \
    --evaluate-after
```

Add `--no-language` to zero the text prompt; set `--past-action-k 0` and
`--chunk-size 1` for the no-temporal baseline.

### 2. Full pipeline (train + evaluate)
```bash
python cluster_pipeline.py \
    --data-dir ./trajectories \
    --output-dir ./output \
    --epochs 10 --batch-size 16 \
    --past-action-k 8 --chunk-size 8
```

### 2b. Feature-cached training (recommended for ablation runs)

End-to-end training repeats the frozen-backbone forward every epoch, which
dominates wall time. Cache the pooled features once, then train MLP heads at
~50 µs/sample:

```bash
# Step 1 — precompute one cache per (backbone × use_language × task).
#         Cache file: <cache-dir>/<backbone>_<task>_<lang|nolang>.npy + .json
python feature_cache.py --data-dir ./trajectories --cache-dir ./caches \
    --backbone llava --task-filter chop_a_tree --use-language
python feature_cache.py --data-dir ./trajectories --cache-dir ./caches \
    --backbone llava --task-filter chop_a_tree --no-language
python feature_cache.py --data-dir ./trajectories --cache-dir ./caches \
    --backbone clip  --task-filter chop_a_tree --use-language
python feature_cache.py --data-dir ./trajectories --cache-dir ./caches \
    --backbone clip  --task-filter chop_a_tree --no-language
# ... repeat for collect_dirt ...

# Step 2 — train the MLP head against any cache (fast, ~30 min per condition).
#         Uses train_cached_head() in imitation_learning.py.
python -c "
from imitation_learning import train_cached_head
train_cached_head(
    cache_dir='./caches', cache_tag='llava_chop_a_tree_lang',
    data_root='./trajectories', out_weights='./models/exp1_llava_lang.pt',
    epochs=10, batch_size=256, past_action_k=8, chunk_size=8,
)"
```

One cache pass costs ~26 h per LLaVA tag / ~6 h per CLIP tag on A5000; all 8
head-training runs together fit in a few hours. See "Compute budget" below.

### 3. Roll out a trained agent in MineRL
```bash
python run_rollout.py \
    --model-path ./models/vla.pt \
    --env MineRLBasaltFindCave-v0 \
    --episodes 5 --max-steps 500 \
    --device cuda \
    --record-video
```

### 4. SLURM (one job per ablation cell)

`slurm_train.sh` runs `cluster_pipeline.py` for one (backbone × task × language)
cell. Selectors come from env vars; the first job for a given cell builds the
cache (~26 h LLaVA, ~6 h CLIP), every subsequent job for the same cell reuses
it and finishes in ~30 min of head training. Drop `--time` to 04:00:00 once
all caches exist.

```bash
# Build cache + train one cell (LLaVA + chop_a_tree + prompt)
BACKBONE=llava TASK_FILTER=chop_a_tree USE_LANGUAGE=1 sbatch slurm_train.sh

# All 8 cells of the 2x2 x 2 tasks ablation:
for backbone in llava clip; do
  for task in chop_a_tree collect_dirt; do
    for lang in 1 0; do
      BACKBONE=$backbone TASK_FILTER=$task USE_LANGUAGE=$lang sbatch slurm_train.sh
    done
  done
done
```

Output is auto-tagged: `output/<backbone>_<task>_<lang|nolang>/{model.pt, metrics.json}`.

### 5. MineRL inside Docker (Apple Silicon-friendly)
```bash
docker compose run --remove-orphans minerl test_minerl.py     # sanity check
docker compose run --remove-orphans minerl run_rollout.py --episodes 1
```

## Loss

`imitation_learning.vla_loss(logits, targets)` returns
`(BCE_on_binary + 0.5*(CE_on_camera_x + CE_on_camera_y), bce_value, camera_ce_value)`.

Accepts both 2-D `(B, NUM_OUTPUT_LOGITS)` and 3-D `(B, chunk_size, NUM_OUTPUT_LOGITS)`
logits — the chunk axis is flattened, so every chunk step contributes equally.
Targets follow the same convention.

Camera was previously regressed with MSE on raw degrees, which collapsed to
near-zero predictions on the heavily zero-inflated demonstrator distribution.
Categorical cross-entropy on mu-law bins lets the model represent the bimodal
"stay still vs. turn ±X°" structure of the data.

## Inference: logits → MineRL action

`action_mapping.map_to_minerl_action(logits, threshold=0.5, base_action=None)`:

- Binary entries: `int(sigmoid(logit) >= threshold)`
- Camera entries: argmax each 11-way axis -> bin index -> `vpt_camera.undiscretize`
  -> `(2,) float32` numpy array under the `"camera"` key
- If `base_action` is supplied (e.g. `env.action_space.no_op()`), unpredicted
  keys (`pickItem`, `swapHands`) are inherited from it.

`run_rollout.py` captures `env.action_space.no_op()` once and merges over it.

## Checkpoint Format

```python
th.save({
    "llava_model": <hf id>,
    "state_dict":  <action_head state dict>,        # only the head, not LLaVA
    "training_metrics": {...},
    "config": {
        "num_actions": ..., "num_binary": ..., "num_camera": ...,
        "num_camera_bins": ..., "num_output_logits": ...,
        "past_action_k": ..., "past_action_dim": ..., "chunk_size": ...,
        "use_language": ...,
        "camera_quantizer": {"camera_maxval": 10, "camera_binsize": 2, "mu": 10},
        ...
    }
}, path)
```

`run_rollout._load_agent(path, device)` is the canonical loader. The backbone
is rebuilt from `llava_model`; head input dim is `hidden + past_action_dim`
and head output is `NUM_OUTPUT_LOGITS * chunk_size`. Pre-temporal checkpoints
(no `past_action_k` / `chunk_size` in config) load with defaults 0 and 1.

## Known Limitations

- Temporal context is *off by default*. With `--past-action-k 0 --chunk-size 1`
  the model has no history at all — useful as a baseline cell but expect
  weak performance on sticky-action tasks (attack runs, walking, camera tracking).
  Turn on `--past-action-k 8 --chunk-size 8` for the full temporal model.
- LLaMA has no CLS token, so the head pools the **mean** of the last hidden
  state across all (image + text) tokens.
- Camera bin choices are fixed at training time. If the contractor data ever
  uses a different `camera_maxval`/`mu`, update `vpt_camera.DEFAULT_CAMERA_QUANTIZER`
  before training (the checkpoint stores the values for traceability).
- `decord.VideoReader` handles are cached lazily per-DataLoader-worker (LRU,
  default 64 files per worker). `num_workers > 0` is fine but each worker holds
  its own readers — keep `decoder_cache_size * num_workers` below your fd limit.
- Random MP4 seeks land on the nearest keyframe and decode forward, so
  per-frame latency is ~3–5 ms (higher than the old HDF5 ~0.5 ms). For LLaVA-7B
  forward at 30–100 ms/sample this is invisible under prefetching workers,
  but if you ever swap in a much cheaper backbone, profile the dataloader.

## Dependencies

Pinned in `requirements.txt` for the JURECA cluster (Python 3.10, torch 2.1.0
+ CUDA 11.7, transformers <4.45). MineRL v1.0 + gym 0.23.1 are required for the
rollout path; everything else only needs the ML stack + `decord` for frame
decoding.

## Resumability

Both cache builds and head training survive SLURM job kills / reboots:

- **`feature_cache.precompute`** writes a `<tag>.progress` sidecar atomically
  every 100 batches (default `progress_interval=100`). On restart it
  validates the existing `<tag>.json` metadata against the current request —
  same sample count, feature dim, backbone, language flag — and resumes the
  memmap in `r+` mode from the recorded sample. Mismatches trigger a
  rebuild from scratch. Worst-case loss from a crash: ~100 × `batch_size`
  encoded samples.
- **`train_vla` / `train_cached_head`** write a full checkpoint after every
  epoch via `_atomic_save` (tmp + rename, never partial). The checkpoint
  carries `state_dict` + `optimizer_state` + `epoch` + `training_metrics` +
  `config`. On restart the trainer auto-resumes from epoch+1; pass
  `--restart` to ignore the existing checkpoint and start over.

Both layers handle the typical SLURM failure modes (OOM kill, time limit,
node reboot) without manual intervention.

## Compute budget

For the 2×2 ablation (LLaVA/CLIP × language on/off) on both tasks with the
full temporal recipe (past-action K=8, chunk N=8), 10 epochs each:

|                  | Per-sample fwd | Per epoch / task | All 8 runs |
|------------------|---------------:|-----------------:|-----------:|
| End-to-end LLaVA |          ~30 ms |             26 h |     ~520 h |
| End-to-end CLIP  |           ~7 ms |              6 h |     ~120 h |
| **Total end-to-end (1 task)** |     |              | **~640 h** |
| **Total end-to-end (2 tasks)** |    |              | **~1280 h (~53 days)** |

With `feature_cache.precompute` running first:

|                       | One-time cache build | Head training (10 ep) |
|-----------------------|---------------------:|----------------------:|
| 4 LLaVA caches × 26 h |               ~104 h |                  ~2 h |
| 4 CLIP caches × 6 h   |                ~24 h |                  ~2 h |
| **Total with caching, 2 tasks** | **~128 h (~5.3 days)** | **~4 h** |

Cache storage: ~25 GB per LLaVA tag (FP16 × 4096 dims) + ~10 GB per CLIP tag
(FP16 × ~1536 dims) ≈ **140 GB total** — fits the 2 TB NVMe easily.

Per-sample times are public-benchmark estimates for A5000 + FP16; expect
±50%. Measure on one mini-run before committing.

## Tests

```bash
python -m unittest discover tests
```

Covers `action_to_tensor` for both formats (env scalar + contractor `[v]`),
`CameraQuantizer` round-trip, `map_to_minerl_action` argmax + base-action
merge, and a `vla_loss` shape/backward sanity check.
