"""Convert VLA model logits to a MineRL-compatible action dict."""

import json
import numpy as np
import torch

from imitation_learning import CANONICAL_ACTION_KEYS

_CAMERA_X_IDX = CANONICAL_ACTION_KEYS.index("camera_x")
_CAMERA_Y_IDX = CANONICAL_ACTION_KEYS.index("camera_y")


def map_to_minerl_action(logits: torch.Tensor, threshold: float = 0.5) -> dict:
    """Convert a (23,) logit tensor from VLAAgent to a MineRL action dict.

    Binary actions are sigmoid-thresholded (int 0 or 1).
    Camera actions preserve their sigmoid magnitude as a float32 numpy array [x, y].

    Args:
        logits: Tensor of shape (23,) — raw output from VLAAgent.
        threshold: Sigmoid threshold for binary actions.

    Returns:
        dict with all 23 canonical action keys in MineRL BASALT format.
    """
    probs = torch.sigmoid(logits.float()).detach().cpu().numpy()

    action = {}
    for i, key in enumerate(CANONICAL_ACTION_KEYS):
        if key == "camera_x":
            continue  # handled below as a pair
        if key == "camera_y":
            action["camera"] = np.array(
                [probs[_CAMERA_X_IDX], probs[_CAMERA_Y_IDX]], dtype=np.float32
            )
        else:
            action[key] = int(probs[i] >= threshold)

    return action


if __name__ == "__main__":
    torch.manual_seed(0)
    test_logits = torch.randn(23)
    result = map_to_minerl_action(test_logits)
    # Convert numpy arrays for JSON serialisation
    printable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    print(json.dumps(printable, indent=2))
