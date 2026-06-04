"""Sample a distribution of real-engine frames from the custom eval env and
report the swapped-LAB stats, to compare against the training distribution."""
import cv2
import gym
import numpy as np

import minerl  # noqa: F401
import eval_envs  # noqa: F401  registers MineRLCollectDirt640-v0

env = gym.make("MineRLCollectDirt640-v0")
obs = env.reset()
rng = np.random.RandomState(0)

Ls, As, Bs, Vs = [], [], [], []
for t in range(200):
    a = env.action_space.no_op()
    # Wander so the view isn't stuck on one block: jitter camera, walk, occasional jump.
    a["camera"] = np.array([rng.uniform(-8, 8), rng.uniform(-15, 15)], dtype=np.float32)
    a["forward"] = int(rng.rand() < 0.6)
    a["jump"] = int(rng.rand() < 0.1)
    obs, _, done, _ = env.step(a)
    if t >= 20 and t % 3 == 0:  # skip warmup, then sample every 3rd step
        rgb = obs["pov"][:, :, ::-1]            # real engine -> swap R<->B (training space)
        swapped = np.ascontiguousarray(rgb)
        lab = cv2.cvtColor(swapped, cv2.COLOR_RGB2LAB).astype(np.float32)
        Ls.append(lab[:, :, 0].mean())
        As.append(lab[:, :, 1].mean())
        Bs.append(lab[:, :, 2].mean())
        Vs.append(cv2.cvtColor(swapped, cv2.COLOR_RGB2HSV)[:, :, 2].mean())
    if done:
        obs = env.reset()
env.close()

Ls, As, Bs, Vs = map(np.array, (Ls, As, Bs, Vs))
print(f"N frames sampled: {len(Ls)}")
print(f"NOON eval (swapped) distribution of per-frame means:")
print(f"  L: mean={Ls.mean():.1f}  std={Ls.std():.1f}  range=[{Ls.min():.0f},{Ls.max():.0f}]")
print(f"  a: mean={As.mean():.1f}  std={As.std():.1f}")
print(f"  b: mean={Bs.mean():.1f}  std={Bs.std():.1f}")
print(f"  HSV V: mean={Vs.mean():.0f}")
print(f"TRAINING avg: L=80.8(frame-std 32.5)  a=119.4  b=123.4")
