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
NUM_OUTPUT_LOGITS = 43        VLAAgent output dim = 21 + 2 * 11
```

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
├── frozen_vision_baseline.py  CLIP-only baseline (same forward signature)
├── imitation_learning.py    TrajectoryDataset, vla_loss, train_vla, evaluate, CLI
├── action_mapping.py        Logits -> MineRL action dict (argmax + undiscretize)
├── chunk_frames.py          MP4/PNG -> HDF5 chunked frames
├── consolidate_metadata.py  actions/*.jsonl + infos/*.json -> all_actions.json
├── cluster_pipeline.py      Convert -> train -> evaluate (calls into above)
├── run_rollout.py           Run trained agent (or random) in MineRL
├── eval_logger.py           Per-episode/per-run rollout metrics
├── slurm_train.sh           SLURM job that invokes cluster_pipeline.py
├── Dockerfile, run_minerl.sh, docker-compose.yml, test_minerl.py
└── tests/test_action_conversion.py
```

## Data Layout

`TrajectoryDataset` reads frames from HDF5 chunks and actions from either
consolidated JSON or legacy per-video files:

```
trajectories/
├── trajectory_task_<task>_length_<N>/
│   ├── all_actions.json            # preferred — {"<stem>": [action, ...], ...}
│   ├── all_infos.json              # optional  — {"<stem>": {info_dict}, ...}
│   ├── actions/action_<stem>.jsonl # legacy fallback
│   └── infos/info_<stem>.json      # legacy fallback
└── frames_chunked/
    └── video_<stem>.h5             # produced by chunk_frames.py
```

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

### 1. Convert MP4 → HDF5
```bash
python chunk_frames.py --data-dir ./trajectories --skip-existing
```

### 2. Train (assumes HDF5 already present)
```bash
python imitation_learning.py \
    --data-dir ./trajectories \
    --out-weights ./models/vla.pt \
    --epochs 10 --batch-size 16 --lr 1e-4 \
    --num-workers 2 \
    --evaluate-after
```

### 3. Full pipeline (convert + train + evaluate)
```bash
python cluster_pipeline.py \
    --data-dir ./trajectories \
    --output-dir ./output \
    --epochs 10 --batch-size 16
```

### 4. Roll out a trained agent in MineRL
```bash
python run_rollout.py \
    --model-path ./models/vla.pt \
    --env MineRLBasaltFindCave-v0 \
    --episodes 5 --max-steps 500 \
    --device cuda \
    --record-video
```

### 5. SLURM
```bash
sbatch slurm_train.sh   # wraps cluster_pipeline.py with module loads
```

### 6. MineRL inside Docker (Apple Silicon-friendly)
```bash
docker compose run --remove-orphans minerl test_minerl.py     # sanity check
docker compose run --remove-orphans minerl run_rollout.py --episodes 1
```

## Loss

`imitation_learning.vla_loss(logits, targets)` returns
`(BCE_on_binary + 0.5*(CE_on_camera_x + CE_on_camera_y), bce_value, camera_ce_value)`.

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
        "camera_quantizer": {"camera_maxval": 10, "camera_binsize": 2, "mu": 10},
        ...
    }
}, path)
```

`run_rollout._load_agent(path, device)` is the canonical loader. The backbone
is rebuilt from `llava_model`; the head is sized to `NUM_OUTPUT_LOGITS`.

## Known Limitations

- Frames are processed independently — the model has no temporal context.
- LLaMA has no CLS token, so the head pools the **mean** of the last hidden
  state across all (image + text) tokens.
- Camera bin choices are fixed at training time. If the contractor data ever
  uses a different `camera_maxval`/`mu`, update `vpt_camera.DEFAULT_CAMERA_QUANTIZER`
  before training (the checkpoint stores the values for traceability).
- HDF5 file handles are cached lazily per-DataLoader-worker; `num_workers > 0`
  is fine but each worker holds its own handles.

## Dependencies

Pinned in `requirements.txt` for the JURECA cluster (Python 3.10, torch 2.1.0
+ CUDA 11.7, transformers <4.45). MineRL v1.0 + gym 0.23.1 are required for the
rollout path; everything else (training, conversion) only needs the ML stack.

## Tests

```bash
python -m unittest discover tests
```

Covers `action_to_tensor` for both formats (env scalar + contractor `[v]`),
`CameraQuantizer` round-trip, `map_to_minerl_action` argmax + base-action
merge, and a `vla_loss` shape/backward sanity check.
