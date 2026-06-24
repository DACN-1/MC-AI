#!/usr/bin/env python3
"""Demonstrator anchor: absolute reference statistics from the training data.

Every number in the thesis is relative to our own trained cells. This computes
the contractor demonstrator's own statistics so the eval numbers have an
absolute anchor, and so the counterintuitive "chop a tree prompt suppresses
attack" result can be checked against the demonstrator: if the chop demos
themselves attack at a modest rate, the model's prompt-driven suppression is
faithful imitation, not malfunction (last_refinements.md T5; examiners #6/#7).

Two signals, two sources:
  * Action firing rates (incl. the headline `attack` rate) and camera activity,
    per task. Read from the per-stem `actions/action_*.jsonl` files (each a
    standalone JSONL, one action dict per line) so we never `json.load` the
    ~900 MB `all_actions.json` (memory). Reuses constants._unwrap_scalar /
    _camera_xy so the contractor `[v]` / `[[x,y]]` wrapping is handled exactly
    as in training.
  * Dirt collection (collect_dirt only). Read from `all_infos.json`
    (~1.3 MB, safe to load whole): per-trajectory ground-truth
    `results.inventory_dis_values.dirt` (net inventory) and
    `results.mine_block_values.dirt` (blocks mined). chop_a_tree has no
    all_infos.json, so no collection ground truth exists for chop in the
    training data — reported as absent.

Episodes are 3000 steps; figures are reported both per-episode and per-1000
steps so they line up with the eval "dirt per episode" basis.

Run: python scripts/demonstrator_anchor.py [--n-stems N] [--task chop_a_tree|collect_dirt] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

# Make repo-root imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import BINARY_ACTION_KEYS, _camera_xy, _unwrap_scalar  # noqa: E402

TRAJ_ROOT = Path("trajectories")
TASKS = ["chop_a_tree", "collect_dirt"]
EPISODE_LEN = 3000  # contractor recordings are fixed-length


def task_dir(task: str) -> Path:
    return TRAJ_ROOT / f"trajectory_task_{task}_length_{EPISODE_LEN}"


def action_firing_rates(task: str, n_stems: int | None):
    """Stream per-stem jsonl; return (firing_rates, camera_active_frac, totals)."""
    files = sorted((task_dir(task) / "actions").glob("action_*.jsonl"))
    if n_stems is not None:
        files = files[:n_stems]
    counts = {k: 0 for k in BINARY_ACTION_KEYS}
    camera_active = 0
    total_steps = 0
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                a = json.loads(line)
                total_steps += 1
                for k in BINARY_ACTION_KEYS:
                    if bool(_unwrap_scalar(a.get(k, 0))):
                        counts[k] += 1
                cx, cy = _camera_xy(a.get("camera", [[0.0, 0.0]]))
                if cx != 0.0 or cy != 0.0:
                    camera_active += 1
    rates = {k: (counts[k] / total_steps if total_steps else 0.0) for k in BINARY_ACTION_KEYS}
    cam_frac = camera_active / total_steps if total_steps else 0.0
    return rates, cam_frac, {"n_stems": len(files), "total_steps": total_steps}


def dirt_collection(n_stems: int | None):
    """collect_dirt only: aggregate per-trajectory dirt ground truth."""
    infos_path = task_dir("collect_dirt") / "all_infos.json"
    if not infos_path.exists():
        return None
    infos = json.load(open(infos_path))
    stems = sorted(infos)
    if n_stems is not None:
        stems = stems[:n_stems]
    inv, mined, grass = [], [], []
    for s in stems:
        r = infos[s].get("results", {})
        idv = r.get("inventory_dis_values", {}) or {}
        mbv = r.get("mine_block_values", {}) or {}
        inv.append(idv.get("dirt", 0))
        mined.append(mbv.get("dirt", 0))
        grass.append(mbv.get("grass", 0))

    def stats(xs):
        n = len(xs)
        mean = sum(xs) / n if n else 0.0
        return {
            "n": n,
            "mean_per_episode": mean,
            "mean_per_1000_steps": mean / (EPISODE_LEN / 1000.0),
            "min": min(xs) if xs else 0,
            "max": max(xs) if xs else 0,
        }

    return {
        "inventory_dirt": stats(inv),   # net dirt held at episode end
        "mined_dirt": stats(mined),     # dirt blocks broken
        "mined_grass": stats(grass),    # grass blocks broken (yield dirt)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-stems", type=int, default=None,
                    help="cap stems per task (default: all)")
    ap.add_argument("--task", choices=TASKS, default=None,
                    help="restrict to one task (default: both)")
    ap.add_argument("--json", type=str, default=None,
                    help="optional path to dump the results as JSON")
    args = ap.parse_args()

    tasks = [args.task] if args.task else TASKS
    out = {"episode_len": EPISODE_LEN, "tasks": {}}

    for task in tasks:
        rates, cam_frac, totals = action_firing_rates(task, args.n_stems)
        out["tasks"][task] = {
            "totals": totals,
            "camera_active_frac": cam_frac,
            "firing_rates": rates,
        }

    dirt = dirt_collection(args.n_stems)
    if dirt is not None and ("collect_dirt" in tasks):
        out["tasks"]["collect_dirt"]["dirt_collection"] = dirt

    # ---- human-readable report ----
    print("## Demonstrator anchor (training data)\n")
    print(f"Episode length: {EPISODE_LEN} steps. Firing rate = fraction of steps "
          f"with the action active.\n")

    # Headline attack rate side-by-side.
    print("### Headline: attack rate\n")
    print("| task | stems | steps | attack rate | camera-active |")
    print("|------|------:|------:|------------:|--------------:|")
    for task in tasks:
        t = out["tasks"][task]
        print(f"| {task} | {t['totals']['n_stems']} | {t['totals']['total_steps']:,} "
              f"| {t['firing_rates']['attack']:.3f} | {t['camera_active_frac']:.3f} |")
    print()

    # Full firing-rate table.
    print("### Full firing rates (all 21 binary actions)\n")
    header = "| action | " + " | ".join(tasks) + " |"
    print(header)
    print("|" + "------|" * (len(tasks) + 1))
    for k in BINARY_ACTION_KEYS:
        row = f"| {k} | " + " | ".join(
            f"{out['tasks'][task]['firing_rates'][k]:.3f}" for task in tasks) + " |"
        print(row)
    print()

    if dirt is not None and ("collect_dirt" in tasks):
        print("### Dirt collection (collect_dirt demonstrator)\n")
        print("| signal | mean/episode | mean/1000 steps | min | max | n |")
        print("|--------|------------:|----------------:|----:|----:|--:|")
        for label, key in [("net inventory dirt", "inventory_dirt"),
                            ("dirt blocks mined", "mined_dirt"),
                            ("grass blocks mined", "mined_grass")]:
            s = dirt[key]
            print(f"| {label} | {s['mean_per_episode']:.2f} | "
                  f"{s['mean_per_1000_steps']:.2f} | {s['min']} | {s['max']} | {s['n']} |")
        print()
    elif "chop_a_tree" in tasks:
        print("### Dirt/wood collection (chop_a_tree)\n")
        print("_No `all_infos.json` exists for chop_a_tree, so the training data "
              "carries no item-collection ground truth for this task; only the "
              "action stream (attack rate, camera) is available._\n")

    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outp, "w"), indent=2)
        print(f"[wrote {outp}]")


if __name__ == "__main__":
    main()
