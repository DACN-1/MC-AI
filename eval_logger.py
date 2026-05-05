"""Structured per-episode and per-run logging for MineRL rollouts."""

import json
import math
import os
from collections import defaultdict
from typing import Callable, Optional

from imitation_learning import CANONICAL_ACTION_KEYS


def _default_success(ep: dict) -> bool:
    return ep["total_reward"] > 0


class EpisodeLogger:
    """Log rollout metrics per episode and aggregate across a full run.

    Args:
        output_dir: Directory where JSON logs are written.
        success_fn: Called with the episode summary dict; returns bool.
                    Defaults to total_reward > 0.
    """

    def __init__(
        self,
        output_dir: str,
        success_fn: Optional[Callable[[dict], bool]] = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.success_fn = success_fn or _default_success

        self._episodes: list[dict] = []
        self._step_rewards: list[float] = []
        self._step_action_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------

    def log_step(self, action_key: str, reward: float) -> None:
        """Record one environment step.

        Args:
            action_key: The canonical action key that was active this step
                        (pass the highest-probability binary action, or "camera"
                        if camera movement dominated).
            reward: Scalar reward returned by the environment.
        """
        self._step_rewards.append(float(reward))
        self._step_action_counts[action_key] += 1

    # ------------------------------------------------------------------
    # Per-episode API
    # ------------------------------------------------------------------

    def end_episode(self, episode_id: int) -> dict:
        """Finalise the current episode and write its JSON file.

        Returns the episode summary dict.
        """
        total_reward = sum(self._step_rewards)
        steps = len(self._step_rewards)
        action_counts = dict(self._step_action_counts)

        summary = {
            "episode_id": episode_id,
            "total_reward": total_reward,
            "steps": steps,
            "success": self.success_fn({"total_reward": total_reward, "steps": steps}),
            "action_counts": action_counts,
        }

        path = os.path.join(self.output_dir, f"episode_{episode_id:03d}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        self._episodes.append(summary)
        self._step_rewards = []
        self._step_action_counts = defaultdict(int)
        return summary

    # ------------------------------------------------------------------
    # Run-level API
    # ------------------------------------------------------------------

    def finalize(self, run_name: str = "run") -> dict:
        """Aggregate all episodes and write run_summary.json.

        Prints a formatted table and returns the summary dict.
        """
        if not self._episodes:
            return {}

        rewards = [e["total_reward"] for e in self._episodes]
        lengths = [e["steps"] for e in self._episodes]
        n = len(self._episodes)

        mean_reward = sum(rewards) / n
        std_reward = math.sqrt(sum((r - mean_reward) ** 2 for r in rewards) / n)
        mean_length = sum(lengths) / n
        success_rate = sum(1 for e in self._episodes if e["success"]) / n

        # Per-action frequency across all steps
        total_steps = sum(lengths)
        combined_counts: dict[str, int] = defaultdict(int)
        for ep in self._episodes:
            for k, v in ep["action_counts"].items():
                combined_counts[k] += v
        action_frequency = {
            k: combined_counts[k] / total_steps if total_steps else 0.0
            for k in CANONICAL_ACTION_KEYS
        }

        run_summary = {
            "run_name": run_name,
            "episodes": n,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_length,
            "success_rate": success_rate,
            "action_frequency": action_frequency,
            "episode_rewards": rewards,
        }

        path = os.path.join(self.output_dir, "run_summary.json")
        with open(path, "w") as f:
            json.dump(run_summary, f, indent=2)

        self._print_table(run_summary)
        return run_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_table(self, summary: dict) -> None:
        bar = "-" * 52
        print(bar)
        print(f"  Run: {summary['run_name']}   Episodes: {summary['episodes']}")
        print(bar)
        print(f"  {'Metric':<28} {'Value':>10}")
        print(bar)
        print(f"  {'Mean reward':<28} {summary['mean_reward']:>10.4f}")
        print(f"  {'Std reward':<28} {summary['std_reward']:>10.4f}")
        print(f"  {'Mean episode length':<28} {summary['mean_episode_length']:>10.1f}")
        print(f"  {'Success rate':<28} {summary['success_rate']:>10.2%}")
        print(bar)
        print("  Top-5 action frequencies:")
        sorted_actions = sorted(
            summary["action_frequency"].items(), key=lambda x: x[1], reverse=True
        )
        for key, freq in sorted_actions[:5]:
            print(f"    {key:<24} {freq:>8.4f}")
        print(bar)
