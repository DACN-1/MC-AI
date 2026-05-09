"""Run MineRL rollout episodes with a trained VLAAgent or a random policy."""

import argparse
import json
import os

import cv2
import gym
import minerl  # noqa: F401 — registers MineRL envs
import numpy as np
import torch
from PIL import Image

from VLAAgent import VLAAgent
from action_mapping import map_to_minerl_action
from constants import NUM_OUTPUT_LOGITS
from eval_logger import EpisodeLogger


def _load_agent(model_path: str, device: str) -> VLAAgent:
    """Load a VLA checkpoint produced by imitation_learning.train_vla."""
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint at {model_path} is not in the expected format "
            "(missing 'state_dict' key). Expected a dict produced by train_vla()."
        )
    backbone = ckpt.get("llava_model", "llava-hf/llava-1.5-7b-hf")
    agent = VLAAgent(output_dim=NUM_OUTPUT_LOGITS, backbone=backbone)
    agent.action_head.load_state_dict(ckpt["state_dict"])
    agent = agent.to(device)
    agent.eval()
    return agent


def _obs_to_pil(obs: dict) -> Image.Image:
    pov = obs["pov"]  # uint8 (H, W, 3)
    return Image.fromarray(pov)


def _random_action(env, binary_prob: float = 0.1):
    """Return a no-op action with random binary flips and camera perturbation."""
    action = env.action_space.no_op()
    action["camera"] = np.array(
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)], dtype=np.float32
    )
    _SKIP = {"ESC", "inventory"}  # these end/pause the episode
    binary_keys = [
        k for k in action
        if k != "camera" and k not in _SKIP and isinstance(action[k], (int, np.integer))
    ]
    for key in binary_keys:
        if np.random.random() < binary_prob:
            action[key] = 1
    return action


def _dominant_action_key(minerl_action: dict) -> str:
    for key in ["attack", "forward", "back", "left", "right", "jump", "use"]:
        if minerl_action.get(key, 0):
            return key
    camera = minerl_action.get("camera", np.zeros(2))
    if np.linalg.norm(camera) > 0.1:
        return "camera"
    return "none"


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    logger = EpisodeLogger(output_dir=args.output_dir)

    device = _resolve_device(args.device)
    agent = _load_agent(args.model_path, device) if args.model_path else None

    env = gym.make(args.env)
    # Captured once: gives us the canonical key set including pickItem/swapHands,
    # which the model doesn't predict. We merge it under predicted actions per step.
    no_op_template = env.action_space.no_op()
    print(f"Environment: {args.env}")
    print(f"Policy: {'VLAAgent on ' + device + ' from ' + args.model_path if agent else 'random'}")
    print(f"Episodes: {args.episodes}   Max steps: {args.max_steps}")
    print()

    episode_rows: list[dict] = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        step_log = []

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
                    logits = agent([img], [args.prompt])[0]
                minerl_action = map_to_minerl_action(logits, base_action=no_op_template)
                action_vec = logits.detach().cpu().tolist()
            else:
                minerl_action = _random_action(env)
                action_vec = []

            obs, reward, done, _info = env.step(minerl_action)

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

    print("\nEpisode Summary")
    print(f"{'Episode':>8}  {'Reward':>10}  {'Steps':>7}")
    print("-" * 32)
    for row in episode_rows:
        print(f"{row['episode']:>8d}  {row['reward']:>10.3f}  {row['steps']:>7d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MineRL rollout episodes.")
    parser.add_argument("--model-path", default=None, help="Path to VLAAgent checkpoint (.pt)")
    parser.add_argument("--env", default="MineRLBasaltFindCave-v0", help="MineRL environment ID")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="./rollout_logs")
    parser.add_argument("--record-video", action="store_true", help="Save each episode as an MP4")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the agent on (cuda/cpu)",
    )
    parser.add_argument(
        "--prompt",
        default="Play Minecraft.",
        help="Text prompt fed to the VLA backbone each step",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
