"""Tests for the CLIP patch-grid spatial pooling used by the patch caches."""

import unittest

import torch as th

from frozen_vision_baseline import pool_patch_grid


class TestPoolPatchGrid(unittest.TestCase):
    def test_shape(self):
        tokens = th.randn(2, 256, 1024)  # ViT-L/14 @224: 16x16 patches
        out = pool_patch_grid(tokens, 4)
        self.assertEqual(out.shape, (2, 4 * 4 * 1024))

    def test_block_average_values(self):
        # 4x4 patch layout, H=2, pooled to 2x2 -> each cell is the mean of a
        # 2x2 block. Build a tensor where each patch's value = its row index
        # so the expected block means are easy to compute.
        side, H = 4, 2
        rows = th.arange(side, dtype=th.float32).repeat_interleave(side)  # (16,)
        tokens = rows.view(1, side * side, 1).repeat(1, 1, H)  # (1, 16, 2)
        out = pool_patch_grid(tokens, 2).view(2, 2, H)
        # Top cells average rows {0,1} -> 0.5; bottom cells rows {2,3} -> 2.5.
        expected = th.tensor([[[0.5] * H, [0.5] * H], [[2.5] * H, [2.5] * H]])
        self.assertTrue(th.allclose(out, expected))

    def test_identity_grid(self):
        # grid == side: pooling is a no-op reordering, values preserved.
        tokens = th.randn(3, 16, 8)
        out = pool_patch_grid(tokens, 4)
        self.assertTrue(th.allclose(out, tokens.reshape(3, -1)))

    def test_rejects_non_square_patch_count(self):
        with self.assertRaises(ValueError):
            pool_patch_grid(th.randn(1, 15, 8), 2)

    def test_rejects_bad_grid(self):
        with self.assertRaises(ValueError):
            pool_patch_grid(th.randn(1, 16, 8), 5)
        with self.assertRaises(ValueError):
            pool_patch_grid(th.randn(1, 16, 8), 0)


if __name__ == "__main__":
    unittest.main()
