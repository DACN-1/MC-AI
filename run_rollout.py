"""Run MineRL rollout episodes with a trained VLAAgent or a random policy."""

import argparse
import base64
import json
import os
from collections import deque

import cv2
import gym
import minerl  # noqa: F401 — registers MineRL envs
import numpy as np
import torch
from PIL import Image

import eval_envs  # noqa: F401 — registers custom 640x360 envs + color matching
from action_mapping import map_to_minerl_action
from agent_loader import load_agent as _load_local_agent
from constants import (
    NUM_OUTPUT_LOGITS,
    PAST_ACTION_DIM,
    action_to_onehot,
)
from eval_logger import EpisodeLogger


class _RemoteAgent:
    """Thin client that calls an inference_server.py running on another host.

    Mirrors the in-process agent's call signature —
    `agent(images, texts, past_actions) -> (1, chunk_size, NUM_OUTPUT_LOGITS)` —
    but POSTs the frame + past-action vector to a native inference server (e.g.
    macOS + MPS) and wraps the returned logits back into a tensor. This lets the
    Docker container run only the MineRL env while the GPU does inference.
    """

    def __init__(self, endpoint: str, chunk_size: int):
        # endpoint like "host.docker.internal:8765"
        self._url = f"http://{endpoint}/predict"
        self._chunk_size = chunk_size

    def __call__(self, images, texts, past_actions=None):
        import urllib.request

        img = images[0]
        arr = np.asarray(img, dtype=np.uint8)
        if past_actions is not None:
            past = np.asarray(past_actions, dtype=np.float32).reshape(-1).tolist()
        else:
            past = []
        payload = json.dumps(
            {
                "pov": base64.b64encode(arr.tobytes()).decode("ascii"),
                "shape": list(arr.shape),
                "prompt": texts[0],
                "past": past,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        # Server returns (chunk_size, NUM_OUTPUT_LOGITS); add the batch axis.
        return torch.tensor(out["logits"], dtype=torch.float32).unsqueeze(0)


def _remote_config(endpoint: str) -> dict:
    """Fetch past_action_k / chunk_size / use_language from the server."""
    import urllib.request

    with urllib.request.urlopen(f"http://{endpoint}/config", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_agent(model_path: str, device: str):
    """Local load — delegates to the gym-free agent_loader."""
    return _load_local_agent(model_path, device)


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
    if args.remote_agent:
        agent_cfg = _remote_config(args.remote_agent)
        agent = _RemoteAgent(args.remote_agent, agent_cfg["chunk_size"])
    elif args.model_path:
        agent, agent_cfg = _load_agent(args.model_path, device)
    else:
        agent, agent_cfg = None, {"past_action_k": 0, "chunk_size": 1, "use_language": True}
    past_action_k = agent_cfg["past_action_k"]

    env = gym.make(args.env)
    env, color_applied = eval_envs.maybe_wrap_color(env, args.env, args.color_match)
    # Captured once: gives us the canonical key set including pickItem/swapHands,
    # which the model doesn't predict. We merge it under predicted actions per step.
    no_op_template = env.action_space.no_op()
    print(f"Environment: {args.env}")
    print(f"Color match: {'on (' + args.env + ' LAB target)' if color_applied else 'off'}")
    if agent is None:
        policy_desc = "random"
    elif args.remote_agent:
        policy_desc = f"remote inference server @ {args.remote_agent}"
    else:
        policy_desc = f"{type(agent).__name__} on {device} from {args.model_path}"
    print(f"Policy: {policy_desc}")
    print(f"Episodes: {args.episodes}   Max steps: {args.max_steps}")
    print()

    episode_rows: list[dict] = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        step_log = []

        # Per-episode past-action buffer (zero-padded at start). One-hot
        # encodings (PAST_ACTION_DIM each) of the most recent K env actions.
        past_buffer: deque = deque(
            [np.zeros(PAST_ACTION_DIM, dtype=np.float32) for _ in range(past_action_k)],
            maxlen=past_action_k,
        )

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
                if past_action_k > 0:
                    past_tensor = torch.from_numpy(
                        np.concatenate(list(past_buffer))
                    ).unsqueeze(0)
                else:
                    past_tensor = torch.zeros(1, 0)
                with torch.no_grad():
                    # Model returns (1, chunk_size, NUM_OUTPUT_LOGITS); execute
                    # only the first predicted step and replan next tick.
                    chunk_logits = agent([img], [args.prompt], past_tensor)
                    logits = chunk_logits[0, 0]
                minerl_action = map_to_minerl_action(
                    logits,
                    base_action=no_op_template,
                    sample=args.sample,
                    temperature=args.temperature,
                )
                action_vec = logits.detach().cpu().tolist()
                if past_action_k > 0:
                    past_buffer.append(action_to_onehot(minerl_action))
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
    parser.add_argument(
        "--remote-agent",
        default=None,
        metavar="HOST:PORT",
        help="Run inference on a remote inference_server.py (e.g. "
        "host.docker.internal:8765) instead of loading the model in-process. "
        "Lets the container run only the MineRL env while a native host (e.g. "
        "macOS + MPS) does the model forward.",
    )
    parser.add_argument("--env", default="MineRLBasaltFindCave-v0", help="MineRL environment ID")
    parser.add_argument(
        "--color-match",
        choices=("auto", "on", "lab", "off"),
        default="auto",
        help="Correct the POV channel order to the training distribution. The "
        "training videos have R/B swapped (cv2 save bug), so the model needs an "
        "R↔B swap at eval. 'auto' swaps iff the env id is a known custom env "
        "(default); 'on' requires one; 'lab' additionally applies a per-task "
        "Reinhard LAB transfer (opt-in, usually unnecessary); 'off' disables. "
        "Affects model input and recorded video.",
    )
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
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Stochastic decode: Bernoulli-sample binaries + softmax-sample camera "
        "instead of threshold/argmax. Breaks the greedy no-move collapse (see "
        "no_move_fix.md); pair with a class-balanced-trained head.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Decode temperature for --sample (>1 flattens, <1 sharpens).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
