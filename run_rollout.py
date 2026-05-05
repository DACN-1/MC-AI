"""Run MineRL rollout episodes with VLAAgent or a random policy."""

import argparse
import json
import os

import cv2
import gym
import minerl  # noqa: F401 — registers MineRL envs
import numpy as np
import torch
from PIL import Image

from action_mapping import map_to_minerl_action
from eval_logger import EpisodeLogger
from imitation_learning import NUM_ACTIONS


def _load_agent(model_path: str):
    from VLAAgent import VLAAgent

    agent = VLAAgent(NUM_ACTIONS=NUM_ACTIONS)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()
    return agent


def _obs_to_pil(obs: dict) -> Image.Image:
    pov = obs["pov"]  # uint8 (H, W, 3)
    return Image.fromarray(pov)


def _random_action(env):
    """Return a no-op action with random camera perturbation."""
    action = env.action_space.no_op()
    action["camera"] = np.array(
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)], dtype=np.float32
    )
    return action


def _dominant_action_key(minerl_action: dict) -> str:
    """Return the canonical key with the highest activation for logging."""
    for key in ["attack", "forward", "back", "left", "right", "jump", "use"]:
        if minerl_action.get(key, 0):
            return key
    camera = minerl_action.get("camera", np.zeros(2))
    if np.linalg.norm(camera) > 0.1:
        return "camera"
    return "none"


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    logger = EpisodeLogger(output_dir=args.output_dir)

    agent = _load_agent(args.model_path) if args.model_path else None

    env = gym.make(args.env)
    print(f"Environment: {args.env}")
    print(f"Policy: {'VLAAgent from ' + args.model_path if agent else 'random'}")
    print(f"Episodes: {args.episodes}   Max steps: {args.max_steps}")
    print()

    episode_rows: list[dict] = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        step_log = []

        # Set up video writer for this episode if recording is enabled
        writer = None
        if args.record_video:
            first_frame = obs["pov"]
            h, w = first_frame.shape[:2]
            video_path = os.path.join(args.output_dir, f"episode_{ep_idx:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

        for step in range(args.max_steps):
            if agent is not None:
                img = _obs_to_pil(obs)
                with torch.no_grad():
                    logits = agent([img], ["Play Minecraft"])[0]  # (23,)
                minerl_action = map_to_minerl_action(logits)
                action_vec = logits.cpu().tolist()
            else:
                minerl_action = _random_action(env)
                action_vec = []

            obs, reward, done, _info = env.step(minerl_action)
            total_reward += float(reward)

            if writer is not None:
                writer.write(cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))

            dominant = _dominant_action_key(minerl_action)
            logger.log_step(dominant, reward)

            step_log.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "done": bool(done),
                    "action_vec": action_vec,
                    "minerl_action": {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in minerl_action.items()
                    },
                }
            )

            if done:
                break

        if writer is not None:
            writer.release()
            print(f"    video → {os.path.join(args.output_dir, f'episode_{ep_idx:03d}.mp4')}")

        ep_summary = logger.end_episode(ep_idx)
        episode_rows.append(
            {"episode": ep_idx, "reward": ep_summary["total_reward"], "steps": ep_summary["steps"]}
        )

        # Write per-step log
        step_log_path = os.path.join(args.output_dir, f"steps_{ep_idx:03d}.json")
        with open(step_log_path, "w") as f:
            json.dump(step_log, f)

        print(
            f"  Episode {ep_idx:>3d}  reward={ep_summary['total_reward']:>8.3f}"
            f"  steps={ep_summary['steps']:>5d}"
            f"  success={ep_summary['success']}"
        )

    env.close()

    print()
    logger.finalize(run_name=os.path.basename(args.output_dir))

    # Final summary table
    print("\nEpisode Summary")
    print(f"{'Episode':>8}  {'Reward':>10}  {'Steps':>7}")
    print("-" * 32)
    for row in episode_rows:
        print(f"{row['episode']:>8d}  {row['reward']:>10.3f}  {row['steps']:>7d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MineRL rollout episodes.")
    parser.add_argument("--model-path", default=None, help="Path to VLAAgent weights (.pt)")
    parser.add_argument("--env", default="MineRLBasaltFindCave-v0", help="MineRL environment ID")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="./rollout_logs")
    parser.add_argument("--record-video", action="store_true", help="Save each episode as an MP4")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
