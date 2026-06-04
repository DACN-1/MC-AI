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

    if sample:
        binary_p = torch.sigmoid(logits[:NUM_BINARY] / T)
        binary_vals = torch.bernoulli(binary_p, generator=generator).int().numpy()
        cam_x_p = torch.softmax(logits[_CAM_X_START:_CAM_X_END] / T, dim=-1)
        cam_y_p = torch.softmax(logits[_CAM_X_END:_CAM_Y_END] / T, dim=-1)
        cam_x_bin = int(torch.multinomial(cam_x_p, 1, generator=generator).item())
        cam_y_bin = int(torch.multinomial(cam_y_p, 1, generator=generator).item())
    else:
        binary_vals = (torch.sigmoid(logits[:NUM_BINARY]) >= threshold).int().numpy()
        cam_x_bin = int(logits[_CAM_X_START:_CAM_X_END].argmax().item())
        cam_y_bin = int(logits[_CAM_X_END:_CAM_Y_END].argmax().item())

    cam_xy = DEFAULT_CAMERA_QUANTIZER.undiscretize(np.array([cam_x_bin, cam_y_bin]))

    action: dict = dict(base_action) if base_action is not None else {}
    for i, key in enumerate(BINARY_ACTION_KEYS):
        action[key] = int(binary_vals[i])
    action["camera"] = np.asarray(cam_xy, dtype=np.float32)
    return action


if __name__ == "__main__":
    torch.manual_seed(0)
    test_logits = torch.randn(NUM_OUTPUT_LOGITS)
    result = map_to_minerl_action(test_logits)
    printable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    print(json.dumps(printable, indent=2))
