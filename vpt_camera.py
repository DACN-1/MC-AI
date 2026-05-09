"""Mu-law camera discretization (vendored slice from OpenAI VPT).

Adapted from VPT/lib/actions.py:CameraQuantizer. We keep only the per-axis
quantize/dequantize logic — the rest of VPT's hierarchical action mapping
(button-camera meta-action joining) is not relevant here because we have a
flat 21-binary action space.

Defaults match the BASALT contractor recordings:
  camera_maxval=10, camera_binsize=2, mu=10  ->  11 bins per axis with
  centers at approximately:
    [-10.0, -5.81, -3.13, -1.61, -0.62, 0.0, 0.62, 1.61, 3.13, 5.81, 10.0]

The recorded camera values in trajectories/.../all_actions.json are bin
centers from this exact scheme, so discretize(load) is lossless on demos.
"""

import numpy as np


class CameraQuantizer:
    """Per-axis mu-law quantizer for camera angles in degrees."""

    def __init__(
        self,
        camera_maxval: float = 10.0,
        camera_binsize: float = 2.0,
        mu: float = 10.0,
    ):
        self.camera_maxval = float(camera_maxval)
        self.camera_binsize = float(camera_binsize)
        self.mu = float(mu)

    @property
    def n_bins(self) -> int:
        return int(2 * self.camera_maxval / self.camera_binsize) + 1

    @property
    def null_bin(self) -> int:
        """Bin index corresponding to camera angle 0 (no rotation)."""
        return self.n_bins // 2

    def discretize(self, xy) -> np.ndarray:
        """Continuous degrees -> bin indices (int64), elementwise.

        Accepts a scalar, list, or numpy array. Output shape matches input.
        """
        xy = np.asarray(xy, dtype=np.float64)
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        # Mu-law encode (concentrate resolution near zero)
        xy = xy / self.camera_maxval
        xy = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
        xy = xy * self.camera_maxval

        # Linear bin
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, bins) -> np.ndarray:
        """Bin indices -> continuous degrees (float32), elementwise."""
        xy = np.asarray(bins, dtype=np.float64) * self.camera_binsize - self.camera_maxval

        # Mu-law decode
        xy = xy / self.camera_maxval
        xy = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
        xy = xy * self.camera_maxval

        return xy.astype(np.float32)

    def bin_centers(self) -> np.ndarray:
        """Return the float32 angle (degrees) at the center of each bin."""
        return self.undiscretize(np.arange(self.n_bins))


# Quantizer matching the BASALT contractor data conventions.
DEFAULT_CAMERA_QUANTIZER = CameraQuantizer(
    camera_maxval=10.0, camera_binsize=2.0, mu=10.0
)


if __name__ == "__main__":
    q = DEFAULT_CAMERA_QUANTIZER
    print(f"n_bins={q.n_bins}  null_bin={q.null_bin}")
    print("bin centers:", q.bin_centers().tolist())
    # Round-trip sanity check on a contractor sample
    sample = np.array([1.6094986352788734, -5.809483127522302])
    bins = q.discretize(sample)
    decoded = q.undiscretize(bins)
    print(f"sample={sample.tolist()} -> bins={bins.tolist()} -> decoded={decoded.tolist()}")
