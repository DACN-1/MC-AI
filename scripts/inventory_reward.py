#!/usr/bin/env python3
"""Compute true reward as inventory-delta from `steps_*.json` produced post the
2026-06-08 run_rollout.py patch. Use this when MineRL's
RewardForPossessingItem failed to fire (see project_chop_reward_unreliable.md).

Usage:
  python scripts/inventory_reward.py <dir_with_steps_json>
  # optional filter to specific item types (default: ['log','dirt'])
  python scripts/inventory_reward.py <dir> --items dirt log oak_log
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def episode_reward(steps: list[dict], items: list[str] | None = None) -> tuple[float, dict[str, int]]:
    """Sum positive deltas in inventory across the episode. Returns
    (total_reward, per_item_count). When `items` is set, only those items
    contribute to the reward."""
    items_set = set(items) if items else None
    prev = {}
    total = 0
    by_item: dict[str, int] = defaultdict(int)
    for s in steps:
        inv = s.get("inventory") or {}
        if not isinstance(inv, dict):
            continue
        for k, v in inv.items():
            if items_set is not None and k not in items_set:
                continue
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            delta = iv - prev.get(k, 0)
            if delta > 0:
                total += delta
                by_item[k] += delta
        prev = {k: int(v) for k, v in inv.items() if isinstance(v, (int, float))}
    return total, dict(by_item)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="dir containing steps_*.json (rollout output)")
    p.add_argument(
        "--items",
        nargs="*",
        default=None,
        help="Items to count toward reward (default: all positive deltas). "
        "Example: --items log dirt",
    )
    a = p.parse_args()

    pattern = os.path.join(a.path, "steps_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No steps_*.json under {a.path}", file=sys.stderr)
        sys.exit(1)

    print(f"{'episode':>8} {'logged_reward':>14} {'inv_reward':>11}  {'items'}")
    print("-" * 70)
    totals = []
    inv_totals = []
    for f in files:
        ep_id = os.path.basename(f).removeprefix("steps_").removesuffix(".json")
        steps = json.load(open(f))
        logged = sum(float(s.get("reward", 0.0)) for s in steps)
        true_r, by_item = episode_reward(steps, a.items)
        totals.append(logged)
        inv_totals.append(true_r)
        items_str = ", ".join(f"{k}={v}" for k, v in sorted(by_item.items(), key=lambda kv: -kv[1])[:6])
        print(f"{ep_id:>8} {logged:>14.3f} {true_r:>11d}  {items_str}")
    n = len(files)
    print("-" * 70)
    print(
        f"{'mean':>8} {sum(totals) / n:>14.3f} {sum(inv_totals) / n:>11.3f}"
    )


if __name__ == "__main__":
    main()
