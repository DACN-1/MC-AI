# The "agent doesn't move" failure — diagnosis & fix

## Symptom

In closed-loop rollout the BC agents (`clip_combined_nolang` etc.) never play the
task. Each episode collapses into **one constant behaviour** held for all 500–2000
steps — either fully idle (`none ×N`), stuck attacking (`attack ×N`), or, rarely,
walking forward forever (`forward ×N`). The camera **never turns** and the player
**never purposefully uses WASD**, so it can't orient toward a tree or look down to
dig. Reward is 0 on every task/env, even after fixing the env (640×360 chop/dirt),
the resolution, and the R↔B color domain (see `rollouts.md`, `eval_envs.py`).

## Root cause

The contractor demos are extremely imbalanced, and the **unweighted** loss +
**greedy** decoding + **past-action conditioning** combine to collapse the heads
onto the marginal action distribution.

### Evidence

Demo action distribution (3M frames each):

| | attack ON | forward | left | right | camera = 0° |
|---|---|---|---|---|---|
| chop_a_tree | 83% | 16% | 10% | 6% | 82% / 80% (x/y) |
| collect_dirt | 95% | 7% | 6% | 2% | 91% / 90% |

Model output at rollout (per-step logits from `steps_*.json`):

- **Camera softmax is collapsed**: `P(0° bin)` ≈ **0.92** (≈ the data's 91.6% marginal),
  entropy 0.39 of a max 2.40, and std across frames ≈ **0.047** — i.e. the head emits a
  near-constant distribution and **ignores the image**. `argmax` → 0° on 2000/2000 steps.
- **Movement probs are flat and below base rate**: `forward` p ≈ **0.065** (demo rate 0.164),
  std ≈ 0.014 across 2000 frames, max 0.105 — never near the 0.5 threshold.
- The data itself is *valid* (verified: when an action says "look down 10°", the generated
  video does pitch down), so this is a model/training failure, not bad labels.

### Why (three compounding mechanisms)

1. **Unweighted loss on imbalanced demos.** `vla_loss` was plain `BCEWithLogits` + `CE`
   with no class weighting. The loss-minimising solution for weak features is to predict
   the **base rate**: attack≈1, movement≈0.07, camera≈0°. The heads regressed to the prior.
2. **Past-action feedback trap.** With `--past-action-k 8` the head learns
   `P(action | recent actions, frame)`. Movement is autocorrelated in demos, so at rollout
   — where the buffer is full of "no movement" — the conditional movement prob is pushed
   *below* the marginal → moves even less → buffer stays empty. A self-reinforcing no-move
   attractor (and the mirror-image "walk forever" attractor once forward fires by chance).
3. **Greedy decoding** (`action_mapping.map_to_minerl_action`): `argmax` on the camera
   always returns the 92%-majority 0° bin; the 0.5 threshold makes any rare binary action
   (base rate ≤16%) **mathematically un-selectable**. This is inference-only, but it is the
   final nail.

Mechanisms 1–2 are **trained-in** (baked into the weights); 3 is decoding-only.

## Fix (this change — training-side)

Implemented in `imitation_learning.py`:

1. **Class-balanced loss** — `vla_loss(logits, targets, pos_weight, cam_weight)`:
   - `pos_weight` (per binary action) = `#neg/#pos`, clipped to [0.05, 50]. Upweights rare
     positives (forward → ~9×) and downweights the over-common attack (→ 0.05×).
   - `cam_weight` (per camera bin) = inverse frequency, clipped to [0.1, 50]. Downweights the
     0° bin (→ 0.10×) and upweights the look-down/turn bins (→ ~2.3×).
   - `compute_class_weights(targets_by_stem)` derives both from the demo distribution.
2. **History dropout** — `apply_history_dropout(pasts, p)`: with probability `p`, zero the
   *entire* past-action vector for a training sample (mimicking the start-of-trajectory
   zero-padding the model already sees). Forces the head to read the frame instead of leaning
   on action autocorrelation, breaking the no-move feedback trap.

Both are off by default (baseline unchanged) and wired through `train_cached_head` and
`train_vla` (params `weighted_loss`, `history_dropout`), the training CLI
(`--weighted-loss`, `--history-dropout P`), and recorded in the checkpoint `config`.

### Usage

```bash
# end-to-end
python imitation_learning.py --data-dir ./trajectories --out-weights ./models/vla.pt \
    --past-action-k 8 --chunk-size 8 --weighted-loss --history-dropout 0.5

# cached-head (recommended)
python -c "
from imitation_learning import train_cached_head
train_cached_head(cache_dir='./caches', cache_tag='clip_combined_nolang',
    data_root='./trajectories', out_weights='./models/clip_nolang_balanced.pt',
    epochs=10, batch_size=256, past_action_k=8, chunk_size=8,
    weighted_loss=True, history_dropout=0.5)"
```

### Verified

- `compute_class_weights` on the real imbalance → `pos_weight[forward]=9.0`,
  `pos_weight[attack]=0.05`, `cam_weight[0°]=0.10`, `cam_weight[look-down]=2.3`.
- Weighted `vla_loss` (2-D and 3-D logits) backprops; `apply_history_dropout(p=0.5)` zeros
  ~51% of rows, `p=0` is identity. All 34 unit tests pass.

## Still to do

- **Retrain** a cell with `--weighted-loss --history-dropout 0.5` and re-roll-out in the
  in-distribution envs (`MineRLChopATree640-v0` / `MineRLCollectDirt640-v0`, color-matched)
  to measure whether the heads now assert movement/look-down.
- **Decoding complement** (not in this change): a `--sample` mode in
  `action_mapping.map_to_minerl_action` (Bernoulli-sample binaries, softmax-sample camera).
  Necessary but insufficient on its own — sampling a collapsed softmax barely helps; it pairs
  with the retrain above.
- Optional: data resampling (oversample movement/look-down frames); more head capacity.
