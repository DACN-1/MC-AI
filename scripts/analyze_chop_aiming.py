#!/usr/bin/env python3
"""Demo-side analysis: does the chop demonstrator AIM before chopping?

Reframes the chop investigation. After decode, recipe, temporal, and
representation interventions all failed (docs/trials.md), the open question
was whether the demos even contain a learnable "aim at the trunk" signal.

This aligns camera movement to sustained-attack-run onsets and tests whether
the pre-onset camera burst is (a) present and (b) directional (real aiming)
vs symmetric noise. Action-only — no video needed.

Run: python scripts/analyze_chop_aiming.py [n_stems]
"""
import json
import sys
from pathlib import Path

import numpy as np

ACTIONS = Path(
    "trajectories/trajectory_task_chop_a_tree_length_3000/all_actions.json"
)
W = 15          # +-frames around onset
MIN_RUN = 30    # min sustained-attack run to count as a "chop commit" (~1.5s)


def _uw(v):
    return v[0] if isinstance(v, list) else v


def _cam(a):
    c = a["camera"][0] if isinstance(a["camera"][0], list) else a["camera"]
    return c[0], c[1]


def main(n_stems: int = 300) -> None:
    d = json.load(ACTIONS.open())
    stems = sorted(d)[:n_stems]
    abs_p = np.zeros(2 * W + 1)
    abs_y = np.zeros(2 * W + 1)
    sgn_p = np.zeros(2 * W + 1)
    sgn_y = np.zeros(2 * W + 1)
    n = 0
    runlens = []
    for s in stems:
        acts = d[s]
        att = np.array([int(_uw(a["attack"])) for a in acts])
        cams = np.array([_cam(a) for a in acts])
        T = len(att)
        i = 0
        while i < T:
            if att[i] and (i == 0 or not att[i - 1]):
                j = i
                while j < T and att[j]:
                    j += 1
                if j - i >= MIN_RUN:
                    runlens.append(j - i)
                    if i - W >= 0 and i + W < T:
                        win = cams[i - W : i + W + 1]
                        abs_p += np.abs(win[:, 0])
                        abs_y += np.abs(win[:, 1])
                        sgn_p += win[:, 0]
                        sgn_y += win[:, 1]
                        n += 1
                i = j
            else:
                i += 1
    abs_p /= n; abs_y /= n; sgn_p /= n; sgn_y /= n
    pre = slice(W - 6, W)
    post = slice(W + 1, 2 * W + 1)
    print(f"stems={len(stems)}  onsets={n}  median_run={int(np.median(runlens))}f")
    print(f"|cam| pre-onset (aiming) {(abs_p[pre]+abs_y[pre]).mean():.3f}  "
          f"post-onset (chop) {(abs_p[post]+abs_y[post]).mean():.3f}")
    print(f"pitch signed/abs pre-onset {sgn_p[pre].mean():+.3f}/{abs_p[pre].mean():.3f} "
          f"(equal => directional aiming, down=trunk base)")
    print(f"yaw   signed/abs pre-onset {sgn_y[pre].mean():+.3f}/{abs_y[pre].mean():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
