# Rollout instructions per cell

How to run a MineRL rollout for each ablation cell after training completes.
All commands assume you're at the repo root (`/Users/diego/VSCode/r1-va`) on
Mac, with cells synced under `output_cluster/<cell>/model.pt`.

## Prerequisites (one-time)

- MineRL only runs inside the Docker image (gym 0.23.1 + Java 8 + MineRL v1.0,
  linux/amd64). The host venv doesn't have it — `run_rollout.py` calls
  `gym.make(...)` which needs the Docker stack.
- First rollout downloads the backbone weights from HF Hub into the container's
  HF cache. CLIP-ViT-L/14 ≈ 570 MB. LLaVA-1.5-7B ≈ 14 GB. Subsequent rollouts
  reuse the cache.

```bash
docker compose run --remove-orphans minerl python test_minerl.py  # sanity check
```

## Cell summary

| Cell | Backbone | use_language | Trained on | Checkpoint |
|---|---|---|---|---|
| `clip_combined_nolang` | CLIP-ViT-L/14 | False | chop_a_tree only* | `output_cluster/clip_combined_nolang/model.pt` |
| `clip_combined_lang`   | CLIP-ViT-L/14 | True  | chop_a_tree only* | `output_cluster/clip_combined_lang/model.pt` (pending) |
| `llava_combined_lang`  | LLaVA-1.5-7B  | True  | chop_a_tree only* | `output_cluster/llava_combined_lang/model.pt` (pending) |
| `llava_combined_nolang`| LLaVA-1.5-7B  | False | chop_a_tree only* | `output_cluster/llava_combined_nolang/model.pt` (pending) |

\* "combined" tag is misleading — abakus22's `/var/tmp1` only staged the
chop_a_tree tarball, so these cells trained on ~3.2 M chop-only samples
instead of the 6.24 M combined. This affects what env you should roll out
into (treechop-equivalent envs, not generic exploration).

## How `--prompt` and `use_language` interact

The training-time prompts were task names like `"chop a tree"` and
`"collect dirt"` (looked up per trajectory from `all_actions.json`). At
rollout time:

- **lang cells**: pass `--prompt` matching what you want the agent to attempt
  ("chop a tree" → tree-chopping behaviour). The LLaVA / CLIP text encoder
  conditions the head on this string.
- **nolang cells**: `use_language=False` is stored in the checkpoint config;
  `VLAAgent.encode` zeroes the prompt regardless of what `--prompt` says.
  Pass `--prompt "Play Minecraft."` (the default) — it's discarded internally.

## Per-cell commands

### `clip_combined_nolang` (ready now)

```bash
docker compose run --remove-orphans minerl python run_rollout.py \
  --model-path /workspace/output_cluster/clip_combined_nolang/model.pt \
  --env MineRLBasaltFindCave-v0 \
  --episodes 5 --max-steps 500 \
  --device cpu \
  --record-video
```

`--device cpu`: CLIP forward at ~1 s/sample on M-series; tolerable for short
rollouts. If a CUDA GPU is wired into the container, switch to `--device cuda`.

Treechop-style env for the chop-only training:
```bash
docker compose run --remove-orphans minerl python run_rollout.py \
  --model-path /workspace/output_cluster/clip_combined_nolang/model.pt \
  --env MineRLTreechop-v0 \
  --episodes 5 --max-steps 1000 \
  --device cpu --record-video
```

### `clip_combined_lang` (pending — 152167)

Same as above but with a task-matching prompt:
```bash
docker compose run --remove-orphans minerl python run_rollout.py \
  --model-path /workspace/output_cluster/clip_combined_lang/model.pt \
  --env MineRLTreechop-v0 \
  --prompt "chop a tree" \
  --episodes 5 --max-steps 1000 \
  --device cpu --record-video
```

### `llava_combined_lang` (pending — 152170)

LLaVA-7B on the Mac is **rough**: 14 GB fp16 weights + 30–100 ms/sample
forward on a CUDA box → minutes/sample on Mac CPU. Two practical options:

**A. Remote inference** — keep MineRL local, push the agent forward to a
GPU box you control (cluster login node won't accept rollouts: the cluster
Python excludes gym 0.23.1):

```bash
# On the GPU box: start the inference server
python inference_server.py --model-path ./model.pt --port 5005

# On the Mac: rollout against it
docker compose run --remove-orphans minerl python run_rollout.py \
  --remote-agent gpu-host.local:5005 \
  --env MineRLTreechop-v0 \
  --prompt "chop a tree" \
  --episodes 5 --max-steps 1000 --record-video
```

**B. Local CPU** — short rollout, expect ~1 min/step:
```bash
docker compose run --remove-orphans minerl python run_rollout.py \
  --model-path /workspace/output_cluster/llava_combined_lang/model.pt \
  --env MineRLTreechop-v0 \
  --prompt "chop a tree" \
  --episodes 1 --max-steps 50 \
  --device cpu --record-video
```

### `llava_combined_nolang` (pending — 152171)

Same as `llava_combined_lang` but drop the `--prompt` (or use the default).
Use the same remote-agent setup for tractable throughput:

```bash
docker compose run --remove-orphans minerl python run_rollout.py \
  --remote-agent gpu-host.local:5005 \
  --env MineRLTreechop-v0 \
  --episodes 5 --max-steps 1000 --record-video
```

## In-distribution eval envs (640×360 + biome + color match)

The cells train on diffusion-*generated* video at **640×360** (`trajectories_un/
…_cond_scale_6.0_…mp4`). Two failure modes of naïve rollout:

- `MineRLTreechop-v0` renders at **64×64** — wrong resolution.
- The real engine is brighter/yellower than the generated training frames — wrong hue.

`eval_envs.py` (imported automatically by `run_rollout.py`) registers two custom
ids that fix both:

| Env id | Resolution | Spawn biome | Start item | Reward |
|---|---|---|---|---|
| `MineRLChopATree640-v0` | 640×360 | forest | iron_axe | +1 / log |
| `MineRLCollectDirt640-v0` | 640×360 | plains | iron_shovel | +1 / dirt |

Both subclass `HumanControlEnvSpec`, so the action space is the full near-human set
(ESC / inventory / hotbar.1-9 / drop) the models were trained on — identical to BASALT.

`--color-match {auto,on,off}` (default `auto`) corrects the real-engine POV into the
generated-training color domain before the model (and recorded video) see it. Two parts:

1. **R↔B channel swap** (primary). The generated training videos have red and blue
   channels swapped vs the real engine — confirmed because the HUD hearts (red
   in-engine) render *blue* in the training data. The model learned on those swapped
   frames, so rollout frames must be swapped to match.
2. **Reinhard LAB transfer** (secondary) to the per-task target (`eval_envs.LAB_TARGETS`,
   from `color_targets.json` with a hardcoded fallback), aligning brightness/contrast
   (generated frames are dimmer).

`auto` wraps iff the env id has a target. Verified: a corrected engine frame lands at LAB
≈ the chop target and the hearts render blue. Does **not** fix the generated-vs-real
texture-sharpness gap.

```bash
# chop_a_tree cell, in-distribution eval, color-matched, via the MPS split server
docker compose run --remove-orphans minerl run_rollout.py \
  --remote-agent host.docker.internal:8765 \
  --env MineRLChopATree640-v0 \
  --episodes 5 --max-steps 500 --record-video

# collect_dirt
docker compose run --remove-orphans minerl run_rollout.py \
  --remote-agent host.docker.internal:8765 \
  --env MineRLCollectDirt640-v0 \
  --episodes 5 --max-steps 500 --record-video
```

## Env quick reference

- `MineRLBasaltFindCave-v0` — open exploration, no specific goal. Easiest to
  inspect behaviour visually.
- `MineRLTreechop-v0` — reward = wood collected. Right match for chop-trained
  models.
- `MineRLObtainDiamond-v0` — long-horizon, full Minecraft tech tree. Way past
  what these BC models can do; useful only to confirm they don't immediately
  fail.

## Output layout

Each rollout writes to `./rollout_logs/<timestamp>/`:
- `summary.json` — aggregated per-run + per-episode metrics
- `episode_*.json` — per-episode action distribution, reward, length
- `episode_*.mp4` — recorded gameplay (if `--record-video`)

`eval_logger.py` is the single source of truth for the schema.
