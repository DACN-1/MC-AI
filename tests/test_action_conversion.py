"""Round-trip tests for action_to_tensor / map_to_minerl_action."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_mapping import map_to_minerl_action  # noqa: E402
from constants import (  # noqa: E402
    BINARY_ACTION_KEYS,
    CAMERA_ACTION_KEYS,
    CAMERA_NULL_BIN,
    NUM_BINARY,
    NUM_CAMERA_BINS,
    NUM_OUTPUT_LOGITS,
    PAST_ACTION_DIM,
    action_to_onehot,
    action_to_tensor,
)
from vpt_camera import DEFAULT_CAMERA_QUANTIZER  # noqa: E402


# ---------------------------------------------------------------------
# action_to_tensor — env (scalar) format
# ---------------------------------------------------------------------
class ActionToTensorEnvFormatTests(unittest.TestCase):
    def test_zero_action(self):
        vec = action_to_tensor({}).numpy()
        self.assertEqual(vec.shape, (NUM_BINARY + 2,))
        # binary part: all zeros
        self.assertTrue(np.all(vec[:NUM_BINARY] == 0.0))
        # camera defaults to bin centered at 0 -> null bin
        self.assertEqual(int(vec[NUM_BINARY]), CAMERA_NULL_BIN)
        self.assertEqual(int(vec[NUM_BINARY + 1]), CAMERA_NULL_BIN)

    def test_binary_action_set(self):
        vec = action_to_tensor({"forward": 1, "jump": 1}).numpy()
        idx_forward = BINARY_ACTION_KEYS.index("forward")
        idx_jump = BINARY_ACTION_KEYS.index("jump")
        self.assertEqual(vec[idx_forward], 1.0)
        self.assertEqual(vec[idx_jump], 1.0)
        on_indices = np.where(vec[:NUM_BINARY] > 0)[0].tolist()
        self.assertEqual(sorted(on_indices), sorted([idx_forward, idx_jump]))

    def test_camera_flat_format(self):
        vec = action_to_tensor({"camera": [1.6094986352788734, -5.809483127522302]}).numpy()
        # These are the centers of bins 7 and 1 in the default quantizer
        self.assertEqual(int(vec[NUM_BINARY]), 7)
        self.assertEqual(int(vec[NUM_BINARY + 1]), 1)

    def test_camera_numpy_array(self):
        vec = action_to_tensor({"camera": np.array([0.0, 0.0], dtype=np.float32)}).numpy()
        self.assertEqual(int(vec[NUM_BINARY]), CAMERA_NULL_BIN)
        self.assertEqual(int(vec[NUM_BINARY + 1]), CAMERA_NULL_BIN)

    def test_camera_clipped_to_max(self):
        vec = action_to_tensor({"camera": [50.0, -100.0]}).numpy()
        self.assertEqual(int(vec[NUM_BINARY]), NUM_CAMERA_BINS - 1)  # max bin
        self.assertEqual(int(vec[NUM_BINARY + 1]), 0)                # min bin


# ---------------------------------------------------------------------
# action_to_tensor — contractor (wrapped) format
# ---------------------------------------------------------------------
class ActionToTensorContractorFormatTests(unittest.TestCase):
    """Contractor data wraps every value in a single-element list:
    "attack":[1], "forward":[0], "camera":[[x, y]]
    """

    def test_binary_zero_in_list_decodes_to_zero(self):
        # Regression test for the bool([0]) == True bug.
        vec = action_to_tensor({"forward": [0]}).numpy()
        idx_forward = BINARY_ACTION_KEYS.index("forward")
        self.assertEqual(vec[idx_forward], 0.0)

    def test_binary_one_in_list_decodes_to_one(self):
        vec = action_to_tensor({"attack": [1]}).numpy()
        idx_attack = BINARY_ACTION_KEYS.index("attack")
        self.assertEqual(vec[idx_attack], 1.0)

    def test_full_contractor_action(self):
        action = {
            "attack": [1], "back": [0], "forward": [0], "jump": [0], "left": [0],
            "right": [0], "sneak": [0], "sprint": [0], "use": [0], "drop": [0],
            "inventory": [0],
            "hotbar.1": [0], "hotbar.2": [0], "hotbar.3": [0], "hotbar.4": [0],
            "hotbar.5": [0], "hotbar.6": [0], "hotbar.7": [0], "hotbar.8": [0],
            "hotbar.9": [0],
            "ESC": [0],
            "camera": [[1.6094986352788734, -10.0]],
        }
        vec = action_to_tensor(action).numpy()
        idx_attack = BINARY_ACTION_KEYS.index("attack")
        # Only attack is on
        on_indices = np.where(vec[:NUM_BINARY] > 0)[0].tolist()
        self.assertEqual(on_indices, [idx_attack])
        # Camera bins
        self.assertEqual(int(vec[NUM_BINARY]), 7)      # 1.609... -> bin 7
        self.assertEqual(int(vec[NUM_BINARY + 1]), 0)  # -10.0    -> bin 0


# ---------------------------------------------------------------------
# CameraQuantizer
# ---------------------------------------------------------------------
class CameraQuantizerTests(unittest.TestCase):
    def test_n_bins_matches_constants(self):
        self.assertEqual(DEFAULT_CAMERA_QUANTIZER.n_bins, NUM_CAMERA_BINS)
        self.assertEqual(DEFAULT_CAMERA_QUANTIZER.null_bin, CAMERA_NULL_BIN)

    def test_zero_maps_to_null_bin(self):
        bins = DEFAULT_CAMERA_QUANTIZER.discretize(np.array([0.0, 0.0]))
        np.testing.assert_array_equal(bins, [CAMERA_NULL_BIN, CAMERA_NULL_BIN])

    def test_extremes_clip_to_endpoints(self):
        bins = DEFAULT_CAMERA_QUANTIZER.discretize(np.array([100.0, -100.0]))
        self.assertEqual(bins[0], NUM_CAMERA_BINS - 1)
        self.assertEqual(bins[1], 0)

    def test_bin_centers_are_idempotent(self):
        # Discretizing a bin center should return that same bin.
        centers = DEFAULT_CAMERA_QUANTIZER.bin_centers()
        bins = DEFAULT_CAMERA_QUANTIZER.discretize(centers)
        np.testing.assert_array_equal(bins, np.arange(NUM_CAMERA_BINS))

    def test_undiscretize_then_discretize_roundtrip(self):
        for b in range(NUM_CAMERA_BINS):
            value = DEFAULT_CAMERA_QUANTIZER.undiscretize(np.array([b]))
            recovered = DEFAULT_CAMERA_QUANTIZER.discretize(value)
            self.assertEqual(int(recovered[0]), b)


# ---------------------------------------------------------------------
# map_to_minerl_action
# ---------------------------------------------------------------------
class MapToMineRLActionTests(unittest.TestCase):
    def test_output_keys(self):
        logits = torch.zeros(NUM_OUTPUT_LOGITS)
        out = map_to_minerl_action(logits)
        for key in BINARY_ACTION_KEYS:
            self.assertIn(key, out)
        self.assertIn("camera", out)
        for key in CAMERA_ACTION_KEYS:
            self.assertNotIn(key, out)

    def test_binary_thresholding(self):
        logits = torch.full((NUM_OUTPUT_LOGITS,), -10.0)
        out = map_to_minerl_action(logits)
        for key in BINARY_ACTION_KEYS:
            self.assertEqual(out[key], 0)

        logits = torch.full((NUM_OUTPUT_LOGITS,), -10.0)
        idx_attack = BINARY_ACTION_KEYS.index("attack")
        logits[idx_attack] = 10.0
        out = map_to_minerl_action(logits)
        self.assertEqual(out["attack"], 1)

    def test_camera_argmax_then_undiscretize(self):
        logits = torch.zeros(NUM_OUTPUT_LOGITS)
        # Pick bin 7 for x, bin 1 for y by setting their logits to a large value.
        cam_x_start = NUM_BINARY
        cam_y_start = NUM_BINARY + NUM_CAMERA_BINS
        logits[cam_x_start + 7] = 10.0
        logits[cam_y_start + 1] = 10.0

        out = map_to_minerl_action(logits)
        self.assertIsInstance(out["camera"], np.ndarray)
        self.assertEqual(out["camera"].dtype, np.float32)

        expected = DEFAULT_CAMERA_QUANTIZER.undiscretize(np.array([7, 1]))
        np.testing.assert_allclose(out["camera"], expected, rtol=1e-5)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            map_to_minerl_action(torch.zeros(NUM_OUTPUT_LOGITS - 1))

    def test_base_action_supplies_unpredicted_keys(self):
        # MineRL env emits pickItem / swapHands which we don't predict; they
        # must survive into the output dict via base_action.
        base = {"pickItem": 0, "swapHands": 0}
        logits = torch.full((NUM_OUTPUT_LOGITS,), -10.0)
        out = map_to_minerl_action(logits, base_action=base)
        self.assertEqual(out["pickItem"], 0)
        self.assertEqual(out["swapHands"], 0)
        # And our predicted keys still override
        self.assertEqual(out["attack"], 0)

    def test_round_trip_contractor_action(self):
        """Lossless: contractor demo -> action_to_tensor -> bin -> argmax logits ->
        map_to_minerl_action -> action_to_tensor recovers the same tensor."""
        original = {
            "attack": [1], "forward": [0], "jump": [1],
            "camera": [[1.6094986352788734, -5.809483127522302]],
        }
        target = action_to_tensor(original)

        logits = torch.full((NUM_OUTPUT_LOGITS,), -10.0)
        for i, key in enumerate(BINARY_ACTION_KEYS):
            v = original.get(key, 0)
            if isinstance(v, list):
                v = v[0] if v else 0
            if v:
                logits[i] = 10.0
        cam_x_bin = int(target[NUM_BINARY].item())
        cam_y_bin = int(target[NUM_BINARY + 1].item())
        logits[NUM_BINARY + cam_x_bin] = 10.0
        logits[NUM_BINARY + NUM_CAMERA_BINS + cam_y_bin] = 10.0

        action = map_to_minerl_action(logits)
        recovered = action_to_tensor(action)
        torch.testing.assert_close(recovered, target)


# ---------------------------------------------------------------------
# action_to_onehot — past-action feature encoding
# ---------------------------------------------------------------------
class ActionToOnehotTests(unittest.TestCase):
    def test_shape(self):
        oh = action_to_onehot({})
        self.assertEqual(oh.shape, (PAST_ACTION_DIM,))
        self.assertEqual(oh.dtype, np.float32)

    def test_zero_action_camera_one_hot_at_null_bin(self):
        oh = action_to_onehot({})
        # binary portion is all zero
        self.assertTrue(np.all(oh[:NUM_BINARY] == 0.0))
        # camera_x one-hot peaked at CAMERA_NULL_BIN
        cam_x = oh[NUM_BINARY : NUM_BINARY + NUM_CAMERA_BINS]
        self.assertEqual(int(cam_x.argmax()), CAMERA_NULL_BIN)
        self.assertEqual(cam_x.sum(), 1.0)
        # camera_y one-hot peaked at CAMERA_NULL_BIN
        cam_y = oh[NUM_BINARY + NUM_CAMERA_BINS :]
        self.assertEqual(int(cam_y.argmax()), CAMERA_NULL_BIN)
        self.assertEqual(cam_y.sum(), 1.0)

    def test_binary_and_camera_encoding(self):
        action = {"attack": [1], "forward": [1], "camera": [[1.6094986352788734, -10.0]]}
        oh = action_to_onehot(action)
        idx_attack = BINARY_ACTION_KEYS.index("attack")
        idx_forward = BINARY_ACTION_KEYS.index("forward")
        self.assertEqual(oh[idx_attack], 1.0)
        self.assertEqual(oh[idx_forward], 1.0)
        # camera_x = 1.609 -> bin 7,  camera_y = -10.0 -> bin 0
        cam_x = oh[NUM_BINARY : NUM_BINARY + NUM_CAMERA_BINS]
        cam_y = oh[NUM_BINARY + NUM_CAMERA_BINS :]
        self.assertEqual(int(cam_x.argmax()), 7)
        self.assertEqual(int(cam_y.argmax()), 0)


# ---------------------------------------------------------------------
# vla_loss shape sanity
# ---------------------------------------------------------------------
class VLALossShapeTests(unittest.TestCase):
    def test_loss_runs_and_returns_scalar(self):
        from imitation_learning import vla_loss

        B = 4
        logits = torch.randn(B, NUM_OUTPUT_LOGITS, requires_grad=True)
        # build a target: random binary + valid camera bin indices
        targets = torch.zeros(B, NUM_BINARY + 2)
        targets[:, :NUM_BINARY] = (torch.rand(B, NUM_BINARY) > 0.7).float()
        targets[:, NUM_BINARY] = torch.randint(0, NUM_CAMERA_BINS, (B,)).float()
        targets[:, NUM_BINARY + 1] = torch.randint(0, NUM_CAMERA_BINS, (B,)).float()

        loss, bce, cam_ce = vla_loss(logits, targets)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertGreaterEqual(bce, 0.0)
        self.assertGreaterEqual(cam_ce, 0.0)
        loss.backward()  # should not raise

    def test_loss_accepts_chunked_inputs(self):
        from imitation_learning import vla_loss

        B, N = 4, 8
        logits = torch.randn(B, N, NUM_OUTPUT_LOGITS, requires_grad=True)
        targets = torch.zeros(B, N, NUM_BINARY + 2)
        targets[:, :, :NUM_BINARY] = (torch.rand(B, N, NUM_BINARY) > 0.7).float()
        targets[:, :, NUM_BINARY] = torch.randint(0, NUM_CAMERA_BINS, (B, N)).float()
        targets[:, :, NUM_BINARY + 1] = torch.randint(0, NUM_CAMERA_BINS, (B, N)).float()

        loss, bce, cam_ce = vla_loss(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(bce, 0.0)
        self.assertGreaterEqual(cam_ce, 0.0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
