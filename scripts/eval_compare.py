#!/usr/bin/env python3
"""Aggregate eval_suite outputs across models × conditions with statistical tests.

Walks `output/evaluation/*` (default) to find runs; one suite-run per dir, dir
named `<endtime>_<modeltag>`. Each run contains `manifest.json` and the four
condition sub-dirs `A_chop_nocap/`, `B_chop_ood/`, `C_chop_task/`, `D_dirt_task/`
(matching `scripts/eval_suite.sh`). Per-action firing rates are computed from
each cell's `steps_*.json`.

Pairwise statistical comparison between models per (condition, action):

  * **Wilson 95 % CI** on each rate (no normal approx — exact small-sample-safe).
  * **Two-proportion z-test** with pooled variance for each pair of models,
    Bonferroni-corrected over (n_actions × n_conditions × n_model_pairs).
  * **Cohen's h** effect size for proportion differences:
    h = 2 * (arcsin(sqrt(p1)) − arcsin(sqrt(p2))).
    |h|<0.2 small, 0.2–0.5 medium, >0.5 large.

Significance markers in the comparison table:
    `***` Bonferroni-adjusted p<0.001 with |h|>=0.5
    `** `                          p<0.01  with |h|>=0.2
    `*  `                          p<0.05
    (blank)                        not significant after correction

Backward compat: also reads legacy `output/eval/<modeltag>/<cond>/`
(pre-evaluation/timestamp layout).

Usage:
  python scripts/eval_compare.py                                # auto-discover
  python scripts/eval_compare.py --root output/evaluation
  python scripts/eval_compare.py --models lang nolang lang_r1
  python scripts/eval_compare.py --condition C_chop_task        # focus 1 cond
  python scripts/eval_compare.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

BINARY = [
    "attack", "back", "forward", "jump", "left", "right",
    "sneak", "sprint", "use", "drop", "inventory",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", "ESC",
]
CONDITIONS = ["A_chop_nocap", "B_chop_ood", "C_chop_task", "D_dirt_task"]

# ---------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion. Symmetric around adjusted
    centre, well-behaved for small n / extreme p (unlike Wald)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test, pooled variance. Returns (z, two-sided p-value).
    p-value computed from the standard normal via erf (no scipy dependency)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_two = 2 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p_two)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def signif_marker(p_adj: float, h: float) -> str:
    """Three-tier significance flag combining adjusted p-value with effect size.
    Bonferroni controls the family-wise error rate so we don't flag minute but
    statistically detectable differences on 200 k samples; the |h| floor
    enforces a practically meaningful effect."""
    h_abs = abs(h)
    if p_adj < 0.001 and h_abs >= 0.5:
        return "***"
    if p_adj < 0.01 and h_abs >= 0.2:
        return "** "
    if p_adj < 0.05:
        return "*  "
    return "   "


# ---------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------

def cell_stats(cell_dir: Path) -> dict | None:
    steps_files = sorted(cell_dir.glob("steps_*.json"))
    if not steps_files:
        return None
    n_steps = 0
    fires: Counter = Counter()
    cam_x_nz = cam_y_nz = 0
    cam_x_vals: list[float] = []
    cam_y_vals: list[float] = []
    rewards: list[float] = []
    for f in steps_files:
        for s in json.loads(f.read_text()):
            ma = s.get("minerl_action", {})
            n_steps += 1
            for k in BINARY:
                v = ma.get(k, 0)
                if isinstance(v, (list, tuple)):
                    v = v[0]
                if int(v) == 1:
                    fires[k] += 1
            cam = ma.get("camera", [0.0, 0.0])
            cx, cy = float(cam[0]), float(cam[1])
            if abs(cx) > 1e-6:
                cam_x_nz += 1
            if abs(cy) > 1e-6:
                cam_y_nz += 1
            cam_x_vals.append(cx)
            cam_y_vals.append(cy)
    summary_file = cell_dir / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        # The logger uses `episodes` as a *count* (int) and `episode_rewards`
        # as the list — earlier versions of this script assumed `episodes`
        # was the list. Read whichever shape exists, defaulting to per-episode
        # JSONs in the same dir.
        ep_rewards = summary.get("episode_rewards")
        if isinstance(ep_rewards, list):
            rewards.extend(float(r) for r in ep_rewards)
        else:
            for ep_json in sorted(cell_dir.glob("episode_*.json")):
                try:
                    rewards.append(float(json.loads(ep_json.read_text()).get("reward", 0.0)))
                except (ValueError, json.JSONDecodeError):
                    pass
    return {
        "n_steps": n_steps,
        "fires": fires,
        "cam_x_nz": cam_x_nz,
        "cam_y_nz": cam_y_nz,
        "cam_x_std": st.stdev(cam_x_vals) if len(cam_x_vals) > 1 else 0.0,
        "cam_y_std": st.stdev(cam_y_vals) if len(cam_y_vals) > 1 else 0.0,
        "cam_y_mean_signed": sum(cam_y_vals) / max(len(cam_y_vals), 1),
        "mean_reward": sum(rewards) / max(len(rewards), 1) if rewards else 0.0,
        "n_episodes": len(rewards),
    }


def discover_runs(root: Path) -> dict[str, Path]:
    """Map model_tag -> evaluation root dir. Picks the most-recent run per tag
    (by mtime), since the layout is `<endtime>_<modeltag>/`."""
    runs: dict[str, tuple[float, Path]] = {}
    if root.is_dir():
        for d in root.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            manifest = d / "manifest.json"
            if manifest.exists():
                meta = json.loads(manifest.read_text())
                tag = meta.get("model_tag", d.name)
            else:
                # legacy fallback: <endtime>_<tag> filename split
                parts = d.name.split("_", 1)
                tag = parts[1] if len(parts) == 2 and parts[0].isdigit() else d.name
            mt = d.stat().st_mtime
            if tag not in runs or runs[tag][0] < mt:
                runs[tag] = (mt, d)
    legacy = Path("output/eval")
    if legacy.is_dir():
        for d in legacy.iterdir():
            if d.is_dir() and d.name not in runs:
                runs[d.name] = (d.stat().st_mtime, d)
    return {tag: path for tag, (_, path) in runs.items()}


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="output/evaluation",
                   help="Eval root (default output/evaluation; legacy output/eval/ also scanned)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model tags to compare; default = every run discovered under --root")
    p.add_argument("--conditions", nargs="*", default=CONDITIONS,
                   help="Condition tags (column order); default = all 4")
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    args = p.parse_args()

    runs = discover_runs(Path(args.root))
    if not runs:
        raise SystemExit(f"No model runs found under {args.root}/ or output/eval/")
    model_tags = args.models or sorted(runs.keys())
    runs = {tag: runs[tag] for tag in model_tags if tag in runs}
    if not runs:
        raise SystemExit(f"None of the requested models exist: {model_tags}")

    print(f"Discovered runs:")
    for tag, path in runs.items():
        print(f"  {tag}: {path}")
    print()

    # Load all cells
    all_stats: dict[tuple[str, str], dict] = {}
    for tag, run_dir in runs.items():
        for cond in args.conditions:
            # Legacy path: output/eval/<tag>/<cond>/
            # New path:    <run_dir>/<cond>/  (or <run_dir>/<tag>/<cond>/ for legacy mixed)
            for candidate in (run_dir / cond, run_dir / tag / cond):
                s = cell_stats(candidate)
                if s is not None:
                    all_stats[(tag, cond)] = s
                    break

    # ---------- table 1: per-action firing % per (model × condition) -------
    headers = [f"{tag}|{cond}" for tag in model_tags for cond in args.conditions
               if (tag, cond) in all_stats]
    print("=" * (12 + 12 * len(headers)))
    print(f"{'Action':<11s} " + " ".join(f"{h:>11s}" for h in headers))
    print("-" * (12 + 12 * len(headers)))
    rows: list[list[str]] = []
    for action in BINARY:
        row = [action]
        any_visible = False
        for h in headers:
            tag, cond = h.split("|")
            s = all_stats[(tag, cond)]
            rate = s["fires"][action] / s["n_steps"] * 100
            if rate > 0.3:
                any_visible = True
            row.append(f"{rate:7.2f}")
        if any_visible:
            rows.append(row)
            print(f"{row[0]:<11s} " + " ".join(f"{x:>11s}" for x in row[1:]))

    # ---------- table 2: camera + reward summary ---------------------------
    print()
    print(f"{'Metric':<13s} " + " ".join(f"{h:>11s}" for h in headers))
    print("-" * (14 + 12 * len(headers)))
    for label, key, fmt in [
        ("cam_x_nz%", "cam_x_nz", lambda s: f"{s[key]/s['n_steps']*100:7.2f}"),
        ("cam_y_nz%", "cam_y_nz", lambda s: f"{s[key]/s['n_steps']*100:7.2f}"),
        ("cam_x_std°", "cam_x_std", lambda s: f"{s[key]:7.2f}"),
        ("cam_y_std°", "cam_y_std", lambda s: f"{s[key]:7.2f}"),
        ("cam_y_mean°", "cam_y_mean_signed", lambda s: f"{s[key]:+7.2f}"),
        ("mean_reward", "mean_reward", lambda s: f"{s[key]:7.3f}"),
        ("n_episodes", "n_episodes", lambda s: f"{s[key]:>7d}"),
        ("n_steps", "n_steps", lambda s: f"{s[key]:>7d}"),
    ]:
        row = [label]
        for h in headers:
            tag, cond = h.split("|")
            row.append(fmt(all_stats[(tag, cond)]))
        print(f"{row[0]:<13s} " + " ".join(f"{x:>11s}" for x in row[1:]))

    # ---------- table 3: pairwise statistical tests -----------------------
    pairs = [(model_tags[i], model_tags[j])
             for i in range(len(model_tags))
             for j in range(i + 1, len(model_tags))]
    if len(pairs) == 0:
        print("\n(only one model present — no pairwise tests)")
    else:
        m_tests = sum(1 for _ in BINARY) * len(args.conditions) * len(pairs)
        bonf_factor = max(m_tests, 1)
        print()
        print(f"=== Pairwise tests (Bonferroni factor = {bonf_factor})  ***p<.001+|h|≥.5  **p<.01+|h|≥.2  *p<.05 ===")
        for cond in args.conditions:
            cond_has_data = any((tag, cond) in all_stats for tag in model_tags)
            if not cond_has_data:
                continue
            print(f"\n[ {cond} ]")
            head = f"  {'Action':<11s}"
            for a, b in pairs:
                head += f"  {a:<6s}→{b:<6s} (Δpp / h / p_adj)"
            print(head)
            for action in BINARY:
                line = f"  {action:<11s}"
                emitted = False
                for a, b in pairs:
                    if (a, cond) not in all_stats or (b, cond) not in all_stats:
                        line += " " * 36
                        continue
                    sa, sb = all_stats[(a, cond)], all_stats[(b, cond)]
                    ka, na = sa["fires"][action], sa["n_steps"]
                    kb, nb = sb["fires"][action], sb["n_steps"]
                    pa, pb = ka / na, kb / nb
                    _, p_raw = two_proportion_z(kb, nb, ka, na)
                    p_adj = min(1.0, p_raw * bonf_factor)
                    h = cohens_h(pb, pa)
                    marker = signif_marker(p_adj, h)
                    if pa > 0.005 or pb > 0.005:
                        emitted = True
                    line += f"  Δ{(pb-pa)*100:+5.1f}pp h={h:+.2f} p={p_adj:.1e}{marker}"
                if emitted:
                    print(line)

    # ---------- optional CSV ---------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["model", "condition", "action", "k", "n", "rate",
                        "wilson_lo", "wilson_hi"])
            for (tag, cond), s in all_stats.items():
                n = s["n_steps"]
                for action in BINARY:
                    k = s["fires"][action]
                    lo, hi = wilson_ci(k, n)
                    w.writerow([tag, cond, action, k, n, k / n, lo, hi])
        print(f"\nCSV written: {args.csv}")


if __name__ == "__main__":
    main()
