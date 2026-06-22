"""Reconstruct a rollout video by replaying recorded MineRL actions through the env.

No agent, no GPU — pure env replay. Reads a `steps_NNN.json` produced by
`run_rollout.py`/`eval_logger`, re-seeds the env, and re-steps the exact recorded
`minerl_action` sequence, capturing each `obs['pov']` frame to an MP4.

Caveat: MineRL is not bit-deterministic (see docs / project memory), so a replay
may diverge from the original episode. This is a best-effort action playback in
the same seeded world, not a guaranteed pixel-identical recording.

Usage:
  xvfb-run -a python replay_video.py \
      --steps evaluations/paper/llava_lang_slot30_chop3/C_chop_task/steps_000.json \
      --cond C_chop_task --seed 0 --out replay.mp4
"""
import argparse
import json

import numpy as np
import minerl  # noqa: F401 — registers MineRL envs
import gym
import eval_envs  # noqa: F401 — registers custom 640x360 envs
import cv2

CHOP = "MineRLChopATree640Fast-v0"
DIRT = "MineRLCollectDirt640Fast-v0"


def env_for_condition(cond: str) -> str:
    c = (cond or "").lower()
    return DIRT if c.startswith("d_") or "dirt" in c else CHOP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", required=True, help="path to steps_NNN.json")
    p.add_argument("--env", default=None, help="env id (else inferred from --cond)")
    p.add_argument("--cond", default=None, help="condition name, e.g. C_chop_task / D_dirt_task")
    p.add_argument("--seed", type=int, required=True, help="episode seed (base_seed + episode_idx)")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--fps", type=int, default=20)
    a = p.parse_args()

    env_id = a.env or env_for_condition(a.cond)
    with open(a.steps) as f:
        steps = json.load(f)
    recorded = [s["minerl_action"] for s in steps]
    print(f"replay: env={env_id} seed={a.seed} steps={len(recorded)} -> {a.out}")

    env = gym.make(env_id)
    try:
        env.seed(a.seed)
    except Exception as e:
        print(f"  (env.seed failed, continuing: {e})")
    noop = env.action_space.no_op()

    def to_action(rec):
        act = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in noop.items()}
        for k, v in rec.items():
            if k not in act:
                continue
            act[k] = np.asarray(v, dtype=np.float32) if k == "camera" else v
        return act

    obs = env.reset()
    frames = [np.asarray(obs["pov"])]
    for i, rec in enumerate(recorded):
        obs, _r, done, _info = env.step(to_action(rec))
        frames.append(np.asarray(obs["pov"]))
        if done:
            print(f"  env done at step {i}")
            break
    env.close()

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"WROTE {a.out}  frames={len(frames)}  {w}x{h}")


if __name__ == "__main__":
    main()
