"""Tests for onset-windowed camera-CE weighting."""

import unittest

import numpy as np
import torch as th

from constants import NUM_BINARY, NUM_OUTPUT_LOGITS, NUM_CAMERA_BINS
from imitation_learning import vla_loss


def _rand_batch(b=64, seed=0):
    g = th.Generator().manual_seed(seed)
    logits = th.randn(b, NUM_OUTPUT_LOGITS, generator=g)
    targets = th.zeros(b, NUM_BINARY + 2)
    targets[:, :NUM_BINARY] = (th.rand(b, NUM_BINARY, generator=g) > 0.5).float()
    targets[:, NUM_BINARY] = th.randint(0, NUM_CAMERA_BINS, (b,), generator=g).float()
    targets[:, NUM_BINARY + 1] = th.randint(0, NUM_CAMERA_BINS, (b,), generator=g).float()
    return logits, targets


class TestCameraOnsetWeight(unittest.TestCase):
    def test_uniform_weight_matches_unweighted(self):
        logits, targets = _rand_batch()
        base = vla_loss(logits, targets)
        w = th.ones(logits.size(0))
        weighted = vla_loss(logits, targets, cam_sample_weight=w)
        # cam term identical (weighted mean with uniform w == plain mean);
        # bce identical; total identical.
        self.assertAlmostEqual(base[0].item(), weighted[0].item(), places=5)
        self.assertAlmostEqual(base[2], weighted[2], places=5)

    def test_weight_shifts_camera_loss_toward_high_weight_frames(self):
        logits, targets = _rand_batch()
        # Make the first half have huge camera error, second half ~0 error by
        # pointing logits at the target bin.
        for i in range(logits.size(0)):
            tb_x = int(targets[i, NUM_BINARY])
            tb_y = int(targets[i, NUM_BINARY + 1])
            if i >= logits.size(0) // 2:  # second half: confident-correct
                logits[i, NUM_BINARY:NUM_BINARY + NUM_CAMERA_BINS] = -5
                logits[i, NUM_BINARY + tb_x] = 10
                logits[i, NUM_BINARY + NUM_CAMERA_BINS:] = -5
                logits[i, NUM_BINARY + NUM_CAMERA_BINS + tb_y] = 10
        # Weight that emphasizes the high-error first half -> larger cam CE
        # than weight emphasizing the low-error second half.
        w_hi = th.cat([th.full((32,), 10.0), th.ones(32)])
        w_lo = th.cat([th.ones(32), th.full((32,), 10.0)])
        cam_hi = vla_loss(logits, targets, cam_sample_weight=w_hi)[2]
        cam_lo = vla_loss(logits, targets, cam_sample_weight=w_lo)[2]
        self.assertGreater(cam_hi, cam_lo)

    def test_backward_runs(self):
        logits, targets = _rand_batch()
        logits.requires_grad_(True)
        w = th.cat([th.full((32,), 5.0), th.ones(32)])
        loss, _, _ = vla_loss(logits, targets, cam_sample_weight=w)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
