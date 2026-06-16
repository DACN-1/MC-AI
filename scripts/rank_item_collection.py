#!/usr/bin/env python3
"""Rank recipe checkpoints by OBJECTIVE task performance — items actually
collected — reading the mean_peak_inventory / collect_rate that eval_logger now
writes into each condition's run_summary.json. Reward-independent (the env's
RewardForCollectingItems is broken: it watches 'log' but chopping yields
'oak_log'/'birch_log').

Usage:
    python scripts/rank_item_collection.py evaluations/test/recipe_sweep_v2
"""
import json
import os
import sys

# Items that count as "did the task". Chopping yields per-wood log names; dirt
# digging yields dirt/coarse_dirt/grass_block.
LOG_ITEMS = ("oak_log", "birch_log", "spruce_log", "jungle_log",
             "dark_oak_log", "acacia_log", "log", "log2")
DIRT_ITEMS = ("dirt", "coarse_dirt", "grass_block")

CHOP_CONDS = ("C_chop_task", "A_chop_nocap", "B_chop_ood")
DIRT_CONDS = ("D_dirt_task",)


def _sum_items(d: dict, items) -> float:
    return sum(float(d.get(it, 0)) for it in items)


def _load(cond_dir: str):
    f = os.path.join(cond_dir, "run_summary.json")
    if not os.path.isfile(f):
        return None
    return json.load(open(f))


def main(root: str) -> None:
    tags = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    rows = []
    for tag in tags:
        tdir = os.path.join(root, tag)
        chop_logs = chop_rate = dirt_ct = dirt_rate = None
        n_chop = n_dirt = 0
        for c in CHOP_CONDS:
            s = _load(os.path.join(tdir, c))
            if s:
                mpi = s.get("mean_peak_inventory", {})
                cr = s.get("collect_rate", {})
                lv = _sum_items(mpi, LOG_ITEMS)
                rr = max((cr.get(it, 0.0) for it in LOG_ITEMS), default=0.0)
                chop_logs = (chop_logs or 0) + lv
                chop_rate = max(chop_rate or 0.0, rr)
                n_chop += 1
        for c in DIRT_CONDS:
            s = _load(os.path.join(tdir, c))
            if s:
                mpi = s.get("mean_peak_inventory", {})
                cr = s.get("collect_rate", {})
                dirt_ct = (dirt_ct or 0) + _sum_items(mpi, DIRT_ITEMS)
                dirt_rate = max((cr.get(it, 0.0) for it in DIRT_ITEMS), default=0.0)
                n_dirt += 1
        rows.append((tag, chop_logs, chop_rate, dirt_ct, dirt_rate, n_chop, n_dirt))

    def fmt(v):
        return "  -  " if v is None else f"{v:5.2f}"

    print(f"\n{'tag':<22} {'logs/ep':>8} {'chop%':>7} {'dirt/ep':>8} {'dirt%':>7}")
    print("-" * 56)
    # Sort by logs collected (primary task), then dirt.
    for r in sorted(rows, key=lambda x: (x[1] or -1, x[3] or -1), reverse=True):
        tag, cl, crate, dc, drate, nc, nd = r
        crate_s = "  -  " if crate is None else f"{crate*100:5.0f}%"
        drate_s = "  -  " if drate is None else f"{drate*100:5.0f}%"
        print(f"{tag:<22} {fmt(cl):>8} {crate_s:>7} {fmt(dc):>8} {drate_s:>7}")
    print("\nlogs/ep, dirt/ep = mean peak items collected per episode "
          "(summed across chop / dirt conditions).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluations/test/recipe_sweep_v2")
