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
from action_mapping import build_per_action_vector, map_to_minerl_action
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

    Uses a persistent http.client.HTTPConnection so the underlying TCP
    connection (and any SSH-tunneled session over a high-latency link) is
    reused across all steps. Opening a new connection per /predict call adds
    ~1.5 s of TCP+SSH handshake overhead per step over a remote tunnel — that
    overhead dominated wall-clock time at 5000 steps. With keep-alive, only
    the first request pays it.
    """

    def __init__(self, endpoint: str, chunk_size: int):
        # endpoint like "host.docker.internal:8765"
        host, _, port = endpoint.partition(":")
        self._host = host
        self._port = int(port) if port else 80
        self._chunk_size = chunk_size
        self._conn = None  # opened lazily on first call

    def _get_conn(self):
        import http.client

        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=120)
        return self._conn

    def __call__(self, images, texts, past_actions=None):
        import http.client
        import io

        img = images[0]
        if past_actions is not None:
            past = np.asarray(past_actions, dtype=np.float32).reshape(-1).tolist()
        else:
            past = []
        # JPEG-compress the 640x360 frame (~5-15 kB on real Minecraft frames,
        # vs ~922 kB raw base64). The server prefers `pov_jpeg` over `pov`;
        # the raw `pov` + `shape` path is kept for backward compat.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        payload = json.dumps(
            {
                "pov_jpeg": base64.b64encode(buf.getvalue()).decode("ascii"),
                "prompt": texts[0],
                "past": past,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}

        # Retry once on dropped connection (server restart, idle timeout, etc.).
        for attempt in (1, 2):
            try:
                conn = self._get_conn()
                conn.request("POST", "/predict", body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise http.client.HTTPException(f"status {resp.status}: {data[:200]!r}")
                out = json.loads(data.decode("utf-8"))
                break
            except (http.client.HTTPException, ConnectionError, OSError):
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                if attempt == 2:
                    raise

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


def _parse_kv_overrides(items: list[str] | None) -> dict[str, float]:
    """Parse a list of ``key=value`` strings into a {key: float} dict.

    Used by the per-binary-action calibration flags (--binary-temperatures,
    --binary-thresholds, --binary-logit-bias). Empty list / None returns {}.
    """
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"expected key=value, got {item!r}; e.g. attack=5.0"
            )
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


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

    # Per-binary-action decoding calibration (no_move_fix.md follow-on): shift
    # / flatten the binary head outputs at inference time without retraining.
    # All three are None when the user supplied no overrides — map_to_minerl_action
    # then falls back to its scalar threshold/temperature, byte-identical to the
    # legacy code path.
    temp_overrides = _parse_kv_overrides(args.binary_temperatures)
    thr_overrides = _parse_kv_overrides(args.binary_thresholds)
    bias_overrides = _parse_kv_overrides(args.binary_logit_bias)
    binary_temperatures = (
        build_per_action_vector(temp_overrides, args.temperature) if temp_overrides else None
    )
    binary_thresholds = (
        build_per_action_vector(thr_overrides, 0.5) if thr_overrides else None
    )
    binary_logit_bias = (
        build_per_action_vector(bias_overrides, 0.0) if bias_overrides else None
    )

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
    if temp_overrides:
        print(f"Binary temperatures (overrides): {temp_overrides}")
    if thr_overrides:
        print(f"Binary thresholds (overrides): {thr_overrides}")
    if bias_overrides:
        print(f"Binary logit bias (overrides): {bias_overrides}")
    if args.camera_temperature is not None:
        print(f"Camera temperature: {args.camera_temperature}")
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
                    binary_temperatures=binary_temperatures,
                    binary_thresholds=binary_thresholds,
                    binary_logit_bias=binary_logit_bias,
                    camera_temperature=args.camera_temperature,
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
    parser.add_argument(
        "--binary-temperatures",
        nargs="*",
        metavar="ACTION=T",
        help="Per-action temperature overrides for the binary head under --sample. "
        "Other binary actions inherit --temperature. Example: "
        "--binary-temperatures attack=5.0  flattens just the (suppressed) attack "
        "sigmoid back toward 0.5 on a no_move_fix-trained model.",
    )
    parser.add_argument(
        "--binary-thresholds",
        nargs="*",
        metavar="ACTION=THR",
        help="Per-action threshold overrides for the binary head in GREEDY mode "
        "(ignored under --sample). Other binary actions inherit threshold=0.5. "
        "Example: --binary-thresholds attack=0.005 lets a fix-trained model "
        "fire attack often despite its mean logit being ~-4.5.",
    )
    parser.add_argument(
        "--binary-logit-bias",
        nargs="*",
        metavar="ACTION=B",
        help="Additive bias on binary logits before sigmoid (applies in BOTH modes). "
        "Equivalent at inference time to retraining without the class-balanced "
        "loss — shifts the model's output distribution back to the demo prior. "
        "Example: --binary-logit-bias attack=6.0 on a chop fix model brings "
        "attack sigmoid mean from ~0.025 back to ~0.82.",
    )
    parser.add_argument(
        "--camera-temperature",
        type=float,
        default=None,
        help="Override temperature for the camera softmax under --sample (both axes). "
        "Default None inherits --temperature. Higher T (e.g. 2.0–3.0) flattens the "
        "11-way categorical so the demo-majority 0° bin stops dominating, letting "
        "the agent actually look around. Pairs with greedy or hot binaries.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
