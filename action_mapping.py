"""Convert VLA model logits to a MineRL-compatible action dict."""

import json
from typing import Optional

import numpy as np
import torch

from constants import (
    BINARY_ACTION_KEYS,
    NUM_BINARY,
    NUM_CAMERA_BINS,
    NUM_OUTPUT_LOGITS,
)
from vpt_camera import DEFAULT_CAMERA_QUANTIZER


_CAM_X_START = NUM_BINARY
_CAM_X_END = NUM_BINARY + NUM_CAMERA_BINS
_CAM_Y_END = NUM_BINARY + 2 * NUM_CAMERA_BINS  # = NUM_OUTPUT_LOGITS


def map_to_minerl_action(
    logits: torch.Tensor,
    threshold: float = 0.5,
    base_action: Optional[dict] = None,
    sample: bool = False,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    binary_temperatures: Optional[torch.Tensor] = None,
    binary_thresholds: Optional[torch.Tensor] = None,
    binary_logit_bias: Optional[torch.Tensor] = None,
    camera_temperature: Optional[float] = None,
) -> dict:
    """Convert a (NUM_OUTPUT_LOGITS,) logit tensor from VLAAgent to a MineRL action dict.

    Greedy decode (default):
    - First NUM_BINARY entries: sigmoid + threshold -> int {0, 1}
    - Two NUM_CAMERA_BINS blocks: argmax -> camera_x / camera_y bin index

    Stochastic decode (``sample=True``):
    - Binary entries: Bernoulli-sample at p=sigmoid(logit/T).
    - Camera blocks: sample each axis from softmax(logits/T).
      This breaks the greedy failure mode where the 92%-majority 0° bin is always
      argmaxed and any rare binary (base rate <= threshold) is un-selectable — see
      no_move_fix.md. Pairs with class-balanced training; sampling a *collapsed*
      distribution barely helps on its own.

    `temperature` > 1 flattens, < 1 sharpens. Both bin indices are mu-law-
    undiscretized to degrees.

    Per-binary-action overrides (decoding-side calibration for a model whose
    output marginal has been shifted off the demo prior, e.g. by a class-balanced
    training loss — see no_move_fix.md):
    - ``binary_temperatures``: (NUM_BINARY,) tensor of per-action temperatures.
      Replaces ``temperature`` for the binary block when ``sample=True``.
      Lets a single rare action (e.g. ``attack`` on a fix-trained chop model
      whose logit mean sits at -4.5) be flattened back toward 0.5 without
      blowing up the calibration on other actions.
    - ``binary_thresholds``: (NUM_BINARY,) tensor of per-action thresholds.
      Replaces ``threshold`` for the binary block in greedy mode. Same goal
      as above without re-introducing sampling RNG.
    - ``binary_logit_bias``: (NUM_BINARY,) tensor added to the binary logits
      before sigmoid. Used in both modes. Equivalent at inference time to
      retraining without the class-balanced loss — restores the demo prior
      by shifting (not just scaling) the model's output distribution.

    All three are independent and compose; defaults are scalar / zero so the
    legacy behaviour is byte-identical when none are supplied.

    Camera-side calibration:
    - ``camera_temperature``: scalar override for the camera softmax (both axes)
      under ``sample=True``. Default ``None`` falls back to ``temperature``,
      matching legacy behaviour. Lets the camera be sampled hotter than the
      binaries when the demo distribution's bin-5 (0°) majority would otherwise
      collapse argmax / soft-sample mass onto "stay still" regardless of head
      confidence elsewhere.

    If `base_action` is provided (e.g. `env.action_space.no_op()`), the returned
    dict is built on top of it so any keys MineRL expects but the model doesn't
    predict (`pickItem`, `swapHands`, …) keep their no-op defaults.
    """
    if logits.numel() != NUM_OUTPUT_LOGITS:
        raise ValueError(
            f"Expected logits of size {NUM_OUTPUT_LOGITS}, got {tuple(logits.shape)}"
        )

    logits = logits.detach().float().cpu()
    T = max(temperature, 1e-6)

    bin_logits = logits[:NUM_BINARY]
    if binary_logit_bias is not None:
        bin_logits = bin_logits + binary_logit_bias.float().cpu()

    if sample:
        if binary_temperatures is not None:
            bin_T = binary_temperatures.float().cpu().clamp(min=1e-6)
        else:
            bin_T = torch.full((NUM_BINARY,), T)
        binary_p = torch.sigmoid(bin_logits / bin_T)
        binary_vals = torch.bernoulli(binary_p, generator=generator).int().numpy()
        cam_T = max(camera_temperature, 1e-6) if camera_temperature is not None else T
        cam_x_p = torch.softmax(logits[_CAM_X_START:_CAM_X_END] / cam_T, dim=-1)
        cam_y_p = torch.softmax(logits[_CAM_X_END:_CAM_Y_END] / cam_T, dim=-1)
        cam_x_bin = int(torch.multinomial(cam_x_p, 1, generator=generator).item())
        cam_y_bin = int(torch.multinomial(cam_y_p, 1, generator=generator).item())
    else:
        if binary_thresholds is not None:
            thr = binary_thresholds.float().cpu()
        else:
            thr = torch.full((NUM_BINARY,), float(threshold))
        binary_vals = (torch.sigmoid(bin_logits) >= thr).int().numpy()
        cam_x_bin = int(logits[_CAM_X_START:_CAM_X_END].argmax().item())
        cam_y_bin = int(logits[_CAM_X_END:_CAM_Y_END].argmax().item())

    cam_xy = DEFAULT_CAMERA_QUANTIZER.undiscretize(np.array([cam_x_bin, cam_y_bin]))

    action: dict = dict(base_action) if base_action is not None else {}
    for i, key in enumerate(BINARY_ACTION_KEYS):
        action[key] = int(binary_vals[i])
    action["camera"] = np.asarray(cam_xy, dtype=np.float32)
    return action


def build_per_action_vector(
    overrides: dict[str, float], default: float
) -> torch.Tensor:
    """Build a (NUM_BINARY,) tensor from a {action_key: value} dict + a default.

    Used to translate CLI-style `--temperature-attack 5.0` overrides into the
    per-action vectors `map_to_minerl_action` expects. Unknown keys raise.
    """
    vec = torch.full((NUM_BINARY,), float(default))
    for key, value in overrides.items():
        if key not in BINARY_ACTION_KEYS:
            raise KeyError(
                f"unknown binary action key {key!r}; valid keys: {BINARY_ACTION_KEYS}"
            )
        vec[BINARY_ACTION_KEYS.index(key)] = float(value)
    return vec


if __name__ == "__main__":
    torch.manual_seed(0)
    test_logits = torch.randn(NUM_OUTPUT_LOGITS)
    result = map_to_minerl_action(test_logits)
    printable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    print(json.dumps(printable, indent=2))
