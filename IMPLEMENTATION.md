# Implementation Notes

Engineering reference for the r1v-a codebase. Tracks what each module does,
how data flows through training and rollout, and the rationale behind the
non-obvious design choices (camera quantization, pooling strategy, loss
shape, checkpoint contract). Pair with `CLAUDE.md` (project conventions for
agents) and `README.md` (user-facing quickstart).

---

## 1. System overview

The model is a behavioural-cloning policy for the MineRL BASALT v1.0 action
space. The architecture is intentionally minimal:

```
RGB frame  ──►  LLaVA-1.5-7B (frozen)  ──►  mean-pool last hidden state  ──►  MLP head  ──►  43 logits
                                                                                              │
                                                                                              ├── 21 binary action logits
                                                                                              ├── 11 logits → camera_x bin
                                                                                              └── 11 logits → camera_y bin
```

Only the MLP head trains. The LLaVA backbone is loaded in fp16 and frozen at
init time (`for p in self.llava.parameters(): p.requires_grad_(False)`). At
inference, the head argmaxes each camera axis and decodes back to degrees via
the mu-law inverse.

The motivating constraint is that the contractor recordings (BASALT v1
demonstrations) contain a heavily zero-inflated camera distribution: most
frames have `camera = (0, 0)` and the rest are concentrated around small
adjustments. Regression losses (MSE) collapse to the mean on this kind of
distribution; categorical cross-entropy on mu-law bins matches the data shape
and lets the model represent the bimodal "stay still vs. turn" structure.

---

## 2. Action space and camera quantization

### 2.1 Canonical keys

Defined once in `constants.py`:

```python
BINARY_ACTION_KEYS = [
    "attack", "back", "forward", "jump", "left", "right",
    "sneak", "sprint", "use", "drop", "inventory",
    "hotbar.1" .. "hotbar.9",
    "ESC",
]                                        # 21 keys
CAMERA_ACTION_KEYS = ["camera_x", "camera_y"]  # 2 keys
CANONICAL_ACTION_KEYS = BINARY_ACTION_KEYS + CAMERA_ACTION_KEYS  # 23 total
```

There are two distinct sizes the rest of the code cares about:

| Constant            | Value | Meaning                                          |
| ------------------- | ----- | ------------------------------------------------ |
| `NUM_ACTIONS`       | 23    | env-facing canonical key count                   |
| `NUM_BINARY`        | 21    | binary action keys                               |
| `NUM_CAMERA`        | 2     | camera axes                                      |
| `NUM_CAMERA_BINS`   | 11    | mu-law bins per camera axis                      |
| `CAMERA_NULL_BIN`   | 5     | bin index for "no rotation" (0°)                 |
| `NUM_OUTPUT_LOGITS` | 43    | model output dim = `NUM_BINARY + 2 * NUM_CAMERA_BINS` |

The action head's last `nn.Linear` has `out_features = NUM_OUTPUT_LOGITS`.

### 2.2 Logit layout

The 43 logits are laid out as a single tensor:

```
[ 0 .. 21 )    binary logits (BCEWithLogits)
[21 .. 32)    camera_x bin logits (CrossEntropy)
[32 .. 43)    camera_y bin logits (CrossEntropy)
```

`imitation_learning._CAM_X_SLICE` and `_CAM_Y_SLICE` index into this tensor.
`action_mapping` does the same with `_CAM_X_START`, `_CAM_X_END`, `_CAM_Y_END`.

### 2.3 Mu-law quantizer (`vpt_camera.py`)

Vendored from VPT/lib/actions.py:CameraQuantizer. Defaults match BASALT
contractor recordings exactly:

```python
CameraQuantizer(camera_maxval=10.0, camera_binsize=2.0, mu=10.0)
```

This produces 11 bins per axis with centers (degrees):

```
[-10.0, -5.81, -3.13, -1.61, -0.62, 0.0, 0.62, 1.61, 3.13, 5.81, 10.0]
```

Mu-law spacing concentrates resolution near 0° (where the demonstrator spends
most of their time) and coarsens toward the clipping endpoints.

The forward (`discretize`) transform:

```python
xy = clip(xy, -maxval, maxval)
xy = xy / maxval
xy = sign(xy) * log(1 + mu * |xy|) / log(1 + mu)        # mu-law encode
xy = xy * maxval
bin_idx = round((xy + maxval) / binsize)                 # → integer bin
```

The inverse (`undiscretize`) is symmetric. We verified empirically that
contractor camera values like `1.6094986352788734` and `-5.809483127522302`
are *exactly* bin centers from this scheme — i.e. discretize-then-undiscretize
is lossless on the demos.

### 2.4 Two input formats

Demonstrator and runtime data look different. `action_to_tensor` accepts both:

| Source                                    | Binary entries | Camera                 |
| ----------------------------------------- | -------------- | ---------------------- |
| **Contractor** (`all_actions.json`)       | `"attack":[1]` | `"camera":[[x, y]]`    |
| **Env** (`env.action_space.no_op()`+step) | `"attack": 1`  | `"camera":[x, y]`      |

`_unwrap_scalar` peels the single-element list when present (the bug-prone
case is `bool([0]) == True` because the list is non-empty, so we explicitly
extract `value[0]` first). `_camera_xy` handles both `[[x, y]]` and `[x, y]`
plus numpy-array variants.

There's also no need for `pickItem` and `swapHands` (gym v1 BASALT keys we
don't model) at training time — `action_dict.get(key, 0)` returns `0` for
unknown binary keys, and the env supplies them via `no_op()` at rollout time.

---

## 3. Module reference

```
constants.py             canonical keys, size constants, action_to_tensor()
vpt_camera.py            CameraQuantizer (mu-law) — vendored from VPT
VLAAgent.py              frozen LLaVA + trainable MLP head
frozen_vision_baseline.py  CLIP-only baseline (same forward signature)
imitation_learning.py    TrajectoryDataset, vla_loss, train_vla, evaluate, CLI
action_mapping.py        43 logits → MineRL action dict
chunk_frames.py          MP4/PNG → HDF5 chunked frames (gzip, buffered writes)
consolidate_metadata.py  per-video JSONLs → all_actions.json / all_infos.json
cluster_pipeline.py      convert → train → evaluate orchestration
run_rollout.py           run trained or random agent in MineRL
eval_logger.py           per-episode and per-run rollout metrics
slurm_train.sh           SLURM job script wrapping cluster_pipeline.py
Dockerfile + run_minerl.sh + docker-compose.yml + test_minerl.py
tests/test_action_conversion.py  20 unit tests
```

### 3.1 `constants.py`

- Defines the canonical action keys and all derived sizes.
- `_unwrap_scalar(value)` and `_camera_xy(camera)` do the format normalization.
- `action_to_tensor(action_dict) -> th.Tensor` returns shape
  `(NUM_BINARY + NUM_CAMERA,) = (23,)` float32. The last two entries are bin
  indices stored as floats; the loss casts them to long for cross-entropy.
- Imports `DEFAULT_CAMERA_QUANTIZER` from `vpt_camera`.

### 3.2 `vpt_camera.py`

- `CameraQuantizer(maxval, binsize, mu)` with `discretize` / `undiscretize` /
  `bin_centers`. All three are vectorized over numpy arrays.
- `DEFAULT_CAMERA_QUANTIZER` is the module-level instance used everywhere.
- Running this file directly prints bin centers and a round-trip sanity
  check on a contractor sample.

### 3.3 `VLAAgent.py`

- Single class `VLAAgent(output_dim, backbone="llava-hf/llava-1.5-7b-hf",
  use_language=True)`.
- Backbone: `LlavaForConditionalGeneration.from_pretrained(..., torch_dtype=fp16)`,
  frozen.
- Head: `Linear(hidden, hidden) → ReLU → Linear(hidden, output_dim)` in fp32.
  `pooled` features are cast to fp32 before the head; this trades a small
  amount of compute for stable optimizer updates.
- `forward(images, texts)`:
  1. If `use_language=False`, replace texts with `""` so only image features
     drive the head.
  2. Ensure each prompt contains the `<image>` placeholder LLaVA's processor
     requires.
  3. Process inputs, run LLaVA with `output_hidden_states=True`.
  4. Mean-pool the last hidden state across the full (image + text) token
     sequence. **Why mean-pool**: LLaMA has no CLS token. The original code
     pooled `hidden_states[-1][:, 0]` (the BOS position), which is
     prompt-prefix-dominated and a poor visual summary. Mean over the joint
     (vision + text) sequence is a cheap, mask-agnostic alternative that
     surfaces both modalities.

### 3.4 `imitation_learning.py`

The biggest module. Owns:

- **`TrajectoryDataset`** — reads frames from HDF5 and actions from either
  consolidated JSON or legacy per-video files.
  - `_h5(path)` opens the HDF5 lazily, **per worker**, and caches the handle.
    Avoids the open-and-close-per-`__getitem__` overhead the original code
    had. Each DataLoader worker holds its own dict because h5py file handles
    aren't fork-safe.
  - `__getitem__` returns `(PIL.Image, str, th.Tensor)` so we can feed images
    directly to the LLaVA processor without a re-conversion.

- **`vla_loss(logits, targets)`** — composite loss:
  ```
  total = BCE(binary_logits, binary_targets)
        + 0.5 * (CE(cam_x_logits, cam_x_bin) + CE(cam_y_logits, cam_y_bin))
  ```
  Returns `(loss, bce_value, cam_ce_value)` so the training loop can log each
  component without re-evaluating the loss. The `0.5 *` averages across the
  two camera axes so the camera term has roughly the same magnitude as the
  binary BCE early in training.

- **`train_vla(...)`** — full training loop:
  1. Build dataset; split with seeded `random_split` into 80/10/10 train/val/test.
  2. Save `test_set_<timestamp>.json` so the held-out indices are
     reproducible across re-runs.
  3. Build DataLoaders with `persistent_workers=True` when `num_workers > 0`
     (keeps the H5 cache warm across epochs).
  4. Train with Adam over `model.action_head.parameters()` only.
  5. Per epoch: loss components + binary accuracy + camera bin accuracy.
  6. Save checkpoint (see §6).
  7. Return `(model, test_subset, history)` so the orchestrator can call
     `evaluate` without re-loading.

- **`evaluate(model, test_set, device, output_dir, ...)`** — per-action F1,
  binary accuracy, camera bin accuracy per axis, and **decoded MAE in
  degrees**. The decoded MAE is the human-meaningful number — it tells you
  how off the model's predicted angle is from the demonstrator in the original
  units, instead of an opaque cross-entropy value.

### 3.5 `action_mapping.py`

`map_to_minerl_action(logits, threshold=0.5, base_action=None) -> dict`:

- Validates shape (`logits.numel() == NUM_OUTPUT_LOGITS`).
- `sigmoid(binary) >= threshold` → int 0/1 for each binary key.
- `argmax` each camera axis → bin index → `DEFAULT_CAMERA_QUANTIZER.undiscretize`
  → `(2,) float32` numpy array under `"camera"`.
- If `base_action` is provided, builds the result on top of it. This is how
  unpredicted env keys (`pickItem`, `swapHands`) keep their no-op defaults at
  rollout time — `run_rollout.py` captures `env.action_space.no_op()` once per
  episode and merges over it each step.

### 3.6 `chunk_frames.py`

Converts MP4 / PNG sequences to HDF5 with a chunked, gzip-compressed `frames`
dataset of shape `(N, H, W, 3)` uint8.

- **Buffered chunk writes**: original code wrote frames one at a time, which
  forces h5py to re-encode the chunk on every write (chunked + compressed
  datasets aren't append-friendly). New code buffers a full `frames_per_chunk`
  worth of frames in a numpy array and writes the whole chunk in one shot.
- **`maxshape=(None, H, W, 3)`** on the dataset so we can `dset.resize(...)`
  if `cap.read()` fails before reaching the reported `total_frames`. Without
  resize the trailing rows would silently stay all-zero.
- Compression is `gzip` (level 4). The previous `lz4` option silently fell
  back to gzip while reporting `lz4` in the metadata; that's been removed.

### 3.7 `cluster_pipeline.py`

Three-step orchestration: `convert_videos` (parallel MP4 → H5 with
multiprocessing) → `train_vla` → `evaluate`. The training and evaluation
logic is *not* duplicated here — it's imported from `imitation_learning`. Only
the parallel video-conversion lives in this file.

### 3.8 `run_rollout.py`

- `_load_agent(model_path, device)` rebuilds the LLaVA backbone from
  `ckpt["llava_model"]`, sizes the head to `NUM_OUTPUT_LOGITS`, loads
  `ckpt["state_dict"]` into `agent.action_head`, and moves the whole agent
  to the requested device.
- The episode loop captures `env.action_space.no_op()` once at start, then
  passes that as `base_action` into `map_to_minerl_action` each step.
- `--prompt` is configurable; defaults to `"Play Minecraft."`.
- `--device` defaults to `cuda if available else cpu`.
- `--record-video` saves an MP4 per episode using OpenCV.
- Per-step JSON logs include the raw 43-vector logits in `action_vec` for
  offline analysis.

### 3.9 `eval_logger.py`

Stateless-looking but stateful per-episode aggregator. `log_step(action_key,
reward)` accumulates step rewards and "dominant action" counts; `end_episode`
flushes per-episode summary JSON; `finalize` writes a run summary with
mean/std reward, success rate, and top-5 action frequency. Imports
`CANONICAL_ACTION_KEYS` from `constants.py` so it's a thin module that
doesn't pull in transformers.

---

## 4. Data flow: training

```
trajectories/                       (input)
├── trajectory_task_*/
│   ├── all_actions.json            consolidated demos
│   ├── all_infos.json              optional metadata (text_prompt, ...)
│   └── videos/*.mp4                source recordings (deleted after convert)
└── (frames_chunked/ created by chunk_frames.py)

           │
           │ chunk_frames.py / cluster_pipeline.convert_videos
           ▼

trajectories/frames_chunked/
└── video_<stem>.h5                 (N, H, W, 3) uint8, gzip-chunked

           │
           │ TrajectoryDataset.__init__
           ▼

samples: list of (h5_path, frame_idx, prompt, action_dict)

           │
           │ DataLoader(workers=N, collate=_collate)
           │   └── per-worker H5 cache (lazy)
           │   └── action_to_tensor turns each dict into (23,) float32
           ▼

batch: (PIL.Image list, prompt list, (B, 23) float32)

           │
           │ VLAAgent.forward
           ▼

logits: (B, 43) fp32

           │
           │ vla_loss
           ▼

(loss, bce_value, cam_ce_value) → backward → Adam.step()
```

Targets layout:

- `targets[:, :21]` — binary 0/1 floats
- `targets[:, 21]` — `camera_x` bin index (stored as float, cast to long in loss)
- `targets[:, 22]` — `camera_y` bin index

The model output and target tensors have *different shapes* — 43 vs 23 — and
the loss function is the only place that needs to know the layout.

---

## 5. Data flow: inference (rollout)

```
env.reset()          → obs.pov: (H, W, 3) uint8
env.action_space     → no_op_template (used for pickItem/swapHands defaults)

for each step:
    img = PIL.Image.fromarray(obs["pov"])
    logits = agent([img], [prompt])[0]              # (43,) fp32

    # action_mapping.map_to_minerl_action
    binary_probs = sigmoid(logits[:21])             # → 0/1 per binary key
    cam_x_bin    = argmax(logits[21:32])
    cam_y_bin    = argmax(logits[32:43])
    cam_x, cam_y = undiscretize([cam_x_bin, cam_y_bin])  # → degrees

    minerl_action = {**no_op_template,
                     **{key: int(p >= 0.5) for key, p in zip(BIN_KEYS, binary_probs)},
                     "camera": np.array([cam_x, cam_y], dtype=float32)}

    obs, reward, done, info = env.step(minerl_action)
```

The `**no_op_template` merge is critical — without it, MineRL would receive
an action dict missing `pickItem` and `swapHands` keys, and depending on the
env wrapper that can either error or silently drop into a malformed step.

---

## 6. Loss decomposition

```
vla_loss(logits, targets) =
    BCE(σ(logits[:, 0:21]), targets[:, 0:21])             # binary
    + 0.5 * CE(logits[:, 21:32], targets[:, 21].long())   # camera_x
    + 0.5 * CE(logits[:, 32:43], targets[:, 22].long())   # camera_y
```

`BCEWithLogits` applies sigmoid internally; we don't pass `sigmoid()` of the
logits. Cross-entropy expects raw logits and integer class indices.

The historical alternative was `BCE(binary) + 0.01 * MSE(camera_degrees)`.
The MSE term failed because:
1. Camera targets are mostly 0 (zero-inflated): MSE-optimal output is near 0.
2. A 1° error and a 100° error are both "small" relative to the data
   distribution if the model just outputs 0 — gradient from rare large
   movements is negligible.
3. Single-modal regression can't represent the bimodal "stay still vs.
   actively turn" structure. Categorical cross-entropy can: separate logits
   for the null bin and each turn bin, sampled or argmaxed independently.

---

## 7. Checkpoint format

```python
torch.save({
    "llava_model": "llava-hf/llava-1.5-7b-hf",      # str — backbone identifier
    "state_dict":  model.action_head.state_dict(),   # dict — only the head
    "training_metrics": {
        "train_loss": [...], "train_bce": [...], "train_cam_ce": [...],
        "val_loss":   [...], "val_bce":   [...], "val_cam_ce":   [...],
        "val_binary_acc": [...], "val_cam_bin_acc": [...],
    },
    "config": {
        "num_actions": 23, "num_binary": 21, "num_camera": 2,
        "num_camera_bins": 11, "num_output_logits": 43,
        "epochs": int, "batch_size": int, "lr": float,
        "val_split": float, "test_split": float,
        "camera_quantizer": {"camera_maxval": 10, "camera_binsize": 2, "mu": 10},
    },
}, path)
```

The backbone weights are *not* saved — they're rebuilt from
`ckpt["llava_model"]` on load. The `state_dict` is the head only, so a
checkpoint is small (~hundreds of MB worth of MLP rather than 14 GB of LLaVA).

`run_rollout._load_agent` validates shape on load and raises if the
checkpoint isn't a dict with `"state_dict"`.

---

## 8. Performance considerations

- **HDF5 file handles are cached per worker.** The previous implementation
  opened and closed the H5 file on every `__getitem__`; with `num_workers=0`
  that's a serial bottleneck. The cache is keyed by path because a dataset
  may span multiple H5 files (one per source video).
- **`persistent_workers=True`** keeps that cache alive across epochs.
- **Frames are buffered into chunk-sized numpy arrays before being written
  to HDF5.** Random per-frame writes to a chunked, compressed dataset force
  h5py to decode-and-re-encode the chunk on each write — buffered writes
  give you 1 encode per chunk instead of 100.
- **Action head stays in fp32**, even though LLaVA runs fp16. Optimizer
  updates on fp16 weights with Adam are fragile; the cost of casting
  pooled features to fp32 once per forward pass is negligible compared to
  the LLaVA forward itself.

---

## 9. Key design decisions

| Decision                                  | Why                                                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Frozen LLaVA backbone                     | Keeps trainable params at MLP scale (~few M); makes single-GPU training feasible.                  |
| Mean-pool last hidden state               | LLaMA has no CLS token; mean is a cheap, mask-agnostic summary of the joint (image + text) sequence. |
| Categorical camera, mu-law spacing        | Matches the BASALT data distribution exactly; lossless on demos; supports multimodal predictions.   |
| Composite BCE + CE loss (no scalar weight) | Camera term is naturally scaled by averaging the two CE values; no hand-tuned weight to maintain.  |
| Save head only in checkpoints              | Backbone is reproducible from the HF id; checkpoints stay small and portable.                       |
| `base_action` merge at rollout            | Forward-compatible with future MineRL env keys we don't model.                                      |
| Per-worker H5 cache                        | Huge throughput win without needing to share file handles across processes.                         |
| Drop the VPT submodule                     | Only `CameraQuantizer` was useful; everything else is the VPT transformer policy stack we don't need. The slice is now in `vpt_camera.py`. |

---

## 10. Tests

`tests/test_action_conversion.py` covers:

- `action_to_tensor` for both env (scalar) and contractor (`[v]`) formats
- The `bool([0]) == True` regression case explicitly
- `CameraQuantizer`: zero → null bin, extremes clip, bin-center idempotency,
  full round-trip
- `map_to_minerl_action`: shape validation, threshold behaviour, argmax +
  undiscretize, `base_action` merge
- A full round-trip: contractor demo → tensor → bin → argmax-style logits →
  back to tensor (lossless)
- `vla_loss` shape sanity + backward pass

Run: `python -m unittest discover tests`. All 20 tests are expected to pass
on a CPU-only environment (no GPU required).

---

## 11. Pointers to source

| Concern                  | Code                                                               |
| ------------------------ | ------------------------------------------------------------------ |
| Adding a new binary key  | Append to `BINARY_ACTION_KEYS` in `constants.py` and retrain.       |
| Changing camera bins     | Edit `vpt_camera.DEFAULT_CAMERA_QUANTIZER` constructor args.        |
| Adjusting loss weighting | `imitation_learning.vla_loss` — currently `0.5 * (CE_x + CE_y)`.   |
| Different pooling        | `VLAAgent.forward` — replace `out.hidden_states[-1].mean(dim=1)`.   |
| Different prompt at rollout | `run_rollout.py --prompt "..."`                                  |
| Dataset format change    | `TrajectoryDataset._load_consolidated` / `_load_individual_files`. |
| Checkpoint compatibility | `run_rollout._load_agent`; `imitation_learning.train_vla` save block. |
