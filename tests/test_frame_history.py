"""Tests for CachedFeatureDataset frame-history windows (frame_history_k)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from feature_cache import CachedFeatureDataset


def _make_mini_cache(root: Path, n_frames=12, stride=2, feat_dim=3):
    """One stem, cached every `stride` frames, feature value = frame index."""
    stem = "chop_a_tree_test_seed_1"
    traj = root / "trajectories" / "trajectory_task_chop_a_tree_length_3000"
    traj.mkdir(parents=True)
    noop = {k: [0] for k in ("attack", "forward")}
    actions = [dict(noop, camera=[[0.0, 0.0]]) for _ in range(n_frames)]
    (traj / "all_actions.json").write_text(json.dumps({stem: actions}))

    cache_dir = root / "caches"
    cache_dir.mkdir()
    frames = list(range(0, n_frames, stride))
    feats = np.zeros((len(frames), feat_dim), dtype=np.float16)
    for row, fidx in enumerate(frames):
        feats[row] = fidx  # recognizable per-frame value
    np.save(cache_dir / "tag.npy", feats)
    # np.save adds a header; rewrite as a raw memmap like precompute does.
    mm = np.memmap(
        cache_dir / "tag.npy", dtype=np.float16, mode="w+", shape=feats.shape
    )
    mm[:] = feats
    mm.flush()
    meta = {
        "tag": "tag",
        "backbone": "clip",
        "use_language": True,
        "task_filter": None,
        "frame_stride": stride,
        "n_samples": len(frames),
        "feature_dim": feat_dim,
        "dtype": "float16",
        "samples": [[stem, fidx] for fidx in frames],
    }
    (cache_dir / "tag.json").write_text(json.dumps(meta))
    return root / "trajectories", cache_dir


class TestFrameHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_root, self.cache_dir = _make_mini_cache(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _ds(self, k):
        return CachedFeatureDataset(
            cache_dir=self.cache_dir,
            tag="tag",
            data_root=self.data_root,
            past_action_k=0,
            chunk_size=1,
            frame_history_k=k,
        )

    def test_k0_is_legacy(self):
        ds = self._ds(0)
        self.assertEqual(ds.feature_dim(), 3)
        feat = ds[2][0]  # frame 4
        self.assertTrue(th.equal(feat, th.full((3,), 4.0)))

    def test_window_order_and_padding(self):
        ds = self._ds(2)
        self.assertEqual(ds.feature_dim(), 9)  # (2+1) * 3
        # Sample at frame 0: both history slots zero-padded, current last.
        feat0 = ds[0][0].view(3, 3)
        self.assertTrue(th.equal(feat0[0], th.zeros(3)))
        self.assertTrue(th.equal(feat0[1], th.zeros(3)))
        self.assertTrue(th.equal(feat0[2], th.zeros(3)))  # frame 0 value = 0
        # Sample at frame 4 (idx 2): history = frames 0, 2; current = 4.
        feat4 = ds[2][0].view(3, 3)
        self.assertTrue(th.equal(feat4[0], th.full((3,), 0.0)))
        self.assertTrue(th.equal(feat4[1], th.full((3,), 2.0)))
        self.assertTrue(th.equal(feat4[2], th.full((3,), 4.0)))
        # Sample at frame 2 (idx 1): one pad, then frame 0, then frame 2.
        feat2 = ds[1][0].view(3, 3)
        self.assertTrue(th.equal(feat2[0], th.zeros(3)))
        self.assertTrue(th.equal(feat2[1], th.full((3,), 0.0)))
        self.assertTrue(th.equal(feat2[2], th.full((3,), 2.0)))


if __name__ == "__main__":
    unittest.main()
