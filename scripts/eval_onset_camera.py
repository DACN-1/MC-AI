#!/usr/bin/env python3
"""Onset-frame-restricted camera evaluation.

The aggregate camera accuracy in metrics.json is dominated by the ~78%
still-camera majority, so a fix that only touches the 5% pre-attack-onset
aiming frames is invisible there. This evaluates camera bin accuracy on
those flagged frames SPECIFICALLY — the metric that tests whether
onset-windowed weighting taught the model to aim.

Run on the Mac (cache + checkpoints local):
  python scripts/eval_onset_camera.py <ckpt_dir> [<ckpt_dir> ...]
Each dir needs model.pt; compares against the chop test split of
caches/clip_combined_lang_stride4.
"""
import sys
from pathlib import Path

import numpy as np
import torch as th

from constants import NUM_BINARY, NUM_CAMERA_BINS
from feature_cache import CachedFeatureDataset, HeadOnlyAgent
from imitation_learning import compute_camera_onset_weights, split_indices_by_stem

CACHE_DIR = "caches"
CACHE_TAG = "clip_combined_lang_stride4"
DATA_ROOT = "trajectories"
_CAM_X = slice(NUM_BINARY, NUM_BINARY + NUM_CAMERA_BINS)
_CAM_Y = slice(NUM_BINARY + NUM_CAMERA_BINS, NUM_BINARY + 2 * NUM_CAMERA_BINS)


def _load_head(ckpt_path, feature_dim):
    ck = th.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    head = HeadOnlyAgent(
        feature_dim=feature_dim,
        output_dim=cfg["num_output_logits"],
        past_action_dim=cfg["past_action_dim"],
        chunk_size=cfg["chunk_size"],
        hidden_dim=cfg.get("hidden_dim"),
        learnable_bce_temp=cfg.get("learnable_bce_temp", False),
    )
    head.load_state_dict(ck["state_dict"])
    return head.eval()


def main(ckpt_dirs):
    onset = compute_camera_onset_weights(
        DATA_ROOT, task_filter="chop_a_tree", pre_window=8, multiplier=5.0
    )
    ds = CachedFeatureDataset(
        CACHE_DIR, CACHE_TAG, DATA_ROOT,
        past_action_k=8, chunk_size=8, stem_filter="chop_a_tree",
    )
    _, _, test_idx = split_indices_by_stem(ds.samples, 0.1, 0.1, seed=42)
    # Restrict to test samples whose FIRST chunk step is a flagged onset frame.
    onset_test, all_test = [], []
    for i in test_idx:
        stem, fidx = ds.samples[i]
        all_test.append(i)
        w = onset.get(stem)
        if w is not None and fidx < len(w) and w[fidx] > 1.0:
            onset_test.append(i)
    print(f"chop test: {len(all_test):,} frames, {len(onset_test):,} onset-window frames")

    def cam_acc(head, idxs):
        cx = cy = n = 0
        with th.no_grad():
            for batch_start in range(0, len(idxs), 512):
                chunk = idxs[batch_start : batch_start + 512]
                feats = th.stack([ds[i][0] for i in chunk])
                tgts = th.stack([ds[i][1] for i in chunk])[:, 0]  # first chunk step
                pasts = th.stack([ds[i][2] for i in chunk])
                lg = head(feats, pasts)[:, 0]
                cx += (lg[:, _CAM_X].argmax(-1) == tgts[:, NUM_BINARY].long()).sum().item()
                cy += (lg[:, _CAM_Y].argmax(-1) == tgts[:, NUM_BINARY + 1].long()).sum().item()
                n += len(chunk)
        return cx / n, cy / n

    fd = ds.feature_dim()
    print(f"\n{'cell':<26} {'camx_all':>9} {'camy_all':>9} {'camx_onset':>11} {'camy_onset':>11}")
    for cd in ckpt_dirs:
        head = _load_head(Path(cd) / "model.pt", fd)
        ax, ay = cam_acc(head, all_test)
        ox, oy = cam_acc(head, onset_test)
        print(f"{Path(cd).name.split('stride4_')[-1]:<26} {ax:>9.4f} {ay:>9.4f} {ox:>11.4f} {oy:>11.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
