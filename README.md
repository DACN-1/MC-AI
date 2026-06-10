# R1V-A: Vision-Language-Action Agent for Minecraft

A research codebase for training a lightweight action head on top of a frozen
LLaVA-1.5-7B backbone, predicting MineRL actions from RGB frames via
behavioural cloning. The 11-bin mu-law camera quantizer originally from OpenAI
VPT is vendored in `vpt_camera.py` — the upstream submodule is no longer part
of the project.

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the engineering reference.

## Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For MineRL (rollouts), use the bundled Docker image — it sets up Java 8, Xvfb,
and the right `gym`/`minerl` versions:

```bash
docker compose run --remove-orphans minerl test_minerl.py        # sanity check
docker compose run --remove-orphans minerl run_rollout.py --episodes 1
```

## Pipeline

```
MP4 + actions  --feature_cache-->  *.npy  --imitation_learning-->  model.pt
                                                                    │
                                                  run_rollout / cluster_pipeline
```

Frames are decoded on demand from the MP4 files via `decord`; the frozen
backbone's pooled features are cached once per (backbone × task × language)
cell so head training runs at MLP speed.

Locally:

```bash
# 1. Precompute backbone features (one-time per ablation cell)
python feature_cache.py --data-dir ./trajectories --cache-dir ./caches \
    --backbone llava --task-filter chop_a_tree --use-language

# 2. Train the head + evaluate
python cluster_pipeline.py \
    --data-dir ./trajectories --cache-dir ./caches --output-dir ./output \
    --backbone llava --task-filter chop_a_tree \
    --past-action-k 8 --chunk-size 8 \
    --epochs 10 --batch-size 256

# 3. Roll out the trained agent (in Docker for MineRL)
python run_rollout.py \
    --model-path ./output/model.pt \
    --env MineRLBasaltFindCave-v0 \
    --episodes 5 --device cuda --record-video
```

End-to-end on a SLURM cluster: `BACKBONE=llava TASK_FILTER=chop_a_tree USE_LANGUAGE=1 sbatch slurm_train.sh`.

## Action Space

23 env-facing canonical keys, but the model output is wider:

- **21 binary actions** (BCEWithLogitsLoss):
  `attack, back, forward, jump, left, right, sneak, sprint, use, drop,
  inventory, hotbar.1..9, ESC`
- **2 camera axes** (`camera_x`, `camera_y`), each as a categorical over
  **11 mu-law bins** (cross-entropy per axis). Bin centers (degrees):

  ```
  [-10.0, -5.81, -3.13, -1.61, -0.62, 0.0, 0.62, 1.61, 3.13, 5.81, 10.0]
  ```

  This matches the BASALT contractor recordings — recorded camera values are
  already bin centers from this exact scheme.

`VLAAgent` therefore emits `21 + 2 * 11 = 43` logits per sample.

At inference, `action_mapping.map_to_minerl_action` thresholds the binary
block, argmaxes each camera axis, and undiscretizes back to degrees.

## Repository Layout

```
constants.py             canonical action keys + action_to_tensor() + size constants
vpt_camera.py            CameraQuantizer (mu-law) — vendored from OpenAI VPT
VLAAgent.py              frozen LLaVA backbone + trainable MLP head
frozen_vision_baseline.py  CLIP-only baseline with the same forward signature
feature_cache.py         precompute backbone embeddings + CachedFeatureDataset + HeadOnlyAgent
imitation_learning.py    TrajectoryDataset, vla_loss, train_vla, train_cached_head, evaluate, CLI
action_mapping.py        logits -> MineRL action dict
consolidate_metadata.py  collapse per-video JSONL files into all_actions.json
cluster_pipeline.py      cache -> train -> evaluate orchestration
run_rollout.py           run trained or random agent in MineRL
eval_logger.py           per-episode / per-run metrics
slurm_train.sh           SLURM job script
Dockerfile + run_minerl.sh + docker-compose.yml
tests/                   unit tests (action conversion + camera quantizer)
```

## Data Format

The contractor data wraps every value in a single-element list, while the
gym env emits unwrapped scalars. `action_to_tensor` accepts both:

```jsonc
// Contractor (all_actions.json)
{"attack":[1], "forward":[0], "camera":[[1.609, -10.0]]}

// Env step
{"attack": 1, "forward": 0, "camera": [1.609, -10.0]}
```

## Tests

```bash
python -m unittest discover tests
```

## Hardware

- Training: NVIDIA GPU with ≥16 GB VRAM recommended (LLaVA-7B in fp16).
  CPU works but is much slower.
- Rollout: any CUDA GPU; `--device cpu` is supported but slow.

## Citation

```bibtex
@misc{r1va,
  title  = {R1V-A: Vision-Language-Action Agent for Minecraft Imitation Learning},
  author = {LMU DBS},
  year   = {2026}
}
```

MIT License.
