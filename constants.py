"""Canonical action space for the VLA agent (shared by training and inference).

Two distinct sizes matter:

  NUM_ACTIONS         = 23  (env-facing canonical action keys: 21 binary + 2 camera axes)
  NUM_OUTPUT_LOGITS   = 43  (model output dim: 21 binary + 2 * NUM_CAMERA_BINS camera bins)

VLAAgent emits NUM_OUTPUT_LOGITS values per sample, laid out as:

  [0 .. NUM_BINARY)                                     -> binary action logits   (BCE)
  [NUM_BINARY .. NUM_BINARY + NUM_CAMERA_BINS)          -> camera_x bin logits     (CE)
  [NUM_BINARY + NUM_CAMERA_BINS .. NUM_OUTPUT_LOGITS)   -> camera_y bin logits     (CE)

`action_to_tensor` returns a (NUM_BINARY + NUM_CAMERA,) float32 tensor where the
last two entries are *bin indices* stored as floats; the loss function casts
them to long for cross-entropy.

The dataset has two flavours we must accept:

  Contractor (all_actions.json):  {"attack":[1], "camera":[[x,y]], ...}
  Env step:                       {"attack": 1,   "camera": [x,y],  ...}

Both are handled by `_unwrap_scalar` and `_camera_xy`.
"""

import numpy as np
import torch as th

from vpt_camera import DEFAULT_CAMERA_QUANTIZER


BINARY_ACTION_KEYS = [
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "ESC",
]
CAMERA_ACTION_KEYS = ["camera_x", "camera_y"]
CANONICAL_ACTION_KEYS = BINARY_ACTION_KEYS + CAMERA_ACTION_KEYS

NUM_BINARY = len(BINARY_ACTION_KEYS)
NUM_CAMERA = len(CAMERA_ACTION_KEYS)
NUM_ACTIONS = NUM_BINARY + NUM_CAMERA  # 23 — env-facing key count

NUM_CAMERA_BINS = DEFAULT_CAMERA_QUANTIZER.n_bins  # 11
CAMERA_NULL_BIN = DEFAULT_CAMERA_QUANTIZER.null_bin  # 5
NUM_OUTPUT_LOGITS = NUM_BINARY + NUM_CAMERA * NUM_CAMERA_BINS  # 43

# Past-action conditioning: a single action represented for the action-head
# input. Binary actions stay as 0/1 (NUM_BINARY dims); each camera axis becomes
# a one-hot over NUM_CAMERA_BINS. Coincidentally equal to NUM_OUTPUT_LOGITS but
# named separately because it serves a different role (input vs. output).
PAST_ACTION_DIM = NUM_BINARY + 2 * NUM_CAMERA_BINS  # 43
DEFAULT_PAST_ACTION_K = 8


def _unwrap_scalar(value):
    """Contractor data wraps every value in a single-element list (`[1]` / `[0]`).

    `bool([0])` is True (non-empty list), so we have to peel the wrapper before
    casting to bool. Env data is already a scalar — return as-is.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else 0
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    return value


def _camera_xy(camera) -> tuple[float, float]:
    """Normalize a MineRL camera value into (x, y) floats.

    Accepts:
      - contractor format `[[x, y]]`
      - env format `[x, y]`
      - numpy arrays of either shape
      - anything malformed (returns 0, 0)
    """
    if isinstance(camera, np.ndarray):
        camera = camera.tolist()
    if not isinstance(camera, (list, tuple)) or len(camera) == 0:
        return 0.0, 0.0
    first = camera[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return float(first[0]), float(first[1])
    if len(camera) >= 2:
        return float(camera[0]), float(camera[1])
    return 0.0, 0.0


def action_to_tensor(action_dict) -> th.Tensor:
    """Convert a MineRL action dict into a (NUM_BINARY + NUM_CAMERA,) float32 tensor.

    The trailing two entries are camera bin indices (mu-law discretized) stored
    as floats — the loss function casts them to long for cross-entropy.
    """
    vec = np.zeros(NUM_BINARY + NUM_CAMERA, dtype=np.float32)
    for i, key in enumerate(BINARY_ACTION_KEYS):
        v = _unwrap_scalar(action_dict.get(key, 0))
        vec[i] = float(bool(v))

    cam_x, cam_y = _camera_xy(action_dict.get("camera", [[0.0, 0.0]]))
    bins = DEFAULT_CAMERA_QUANTIZER.discretize(np.array([cam_x, cam_y]))
    vec[NUM_BINARY] = float(bins[0])
    vec[NUM_BINARY + 1] = float(bins[1])
    return th.from_numpy(vec)


def action_to_onehot(action_dict) -> np.ndarray:
    """Encode an action as a flat one-hot feature for past-action conditioning.

    Layout (PAST_ACTION_DIM = NUM_BINARY + 2 * NUM_CAMERA_BINS = 43):
      [0 .. NUM_BINARY)                                     binary actions (0/1)
      [NUM_BINARY .. NUM_BINARY + NUM_CAMERA_BINS)          camera_x one-hot
      [NUM_BINARY + NUM_CAMERA_BINS .. PAST_ACTION_DIM)     camera_y one-hot

    Identical layout to the model's output logits — by design, so the head can
    learn from a vector that "looks like" what it produces.
    """
    base = action_to_tensor(action_dict).numpy()
    cam_x_bin = int(base[NUM_BINARY])
    cam_y_bin = int(base[NUM_BINARY + 1])
    out = np.zeros(PAST_ACTION_DIM, dtype=np.float32)
    out[:NUM_BINARY] = base[:NUM_BINARY]
    out[NUM_BINARY + cam_x_bin] = 1.0
    out[NUM_BINARY + NUM_CAMERA_BINS + cam_y_bin] = 1.0
    return out
