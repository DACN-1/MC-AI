#!/usr/bin/env python3
"""
Consolidate action and info files to reduce inode count.

Converts:
  actions/action_*.jsonl  ->  all_actions.json
  infos/info_*.json       ->  all_infos.json

This reduces inodes from O(num_videos * 2) to O(2) per task directory.
"""

import argparse
import json
from pathlib import Path
import shutil


def consolidate_actions(actions_dir: Path, output_file: Path, delete_originals: bool = False) -> int:
    """
    Consolidate all action_*.jsonl files into a single JSON file.

    Format: {"video_stem": [action1, action2, ...], ...}

    Returns number of files consolidated.
    """
    consolidated = {}
    action_files = list(actions_dir.glob("action_*.jsonl"))

    for action_path in action_files:
        # Extract stem: action_chop_a_tree_freq_25_..._seed_1.jsonl -> chop_a_tree_freq_25_..._seed_1
        stem = action_path.stem[len("action_"):]

        with action_path.open("r", encoding="utf-8") as fp:
            actions = [json.loads(line) for line in fp if line.strip()]

        consolidated[stem] = actions

    # Atomic write: tmp + rename. A mid-write crash would otherwise leave a
    # half-written JSON that downstream readers (feature_cache.enumerate_samples)
    # would silently skip, producing a partial training set.
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(consolidated, fp, separators=(',', ':'))
    tmp.replace(output_file)

    if delete_originals and action_files:
        for f in action_files:
            f.unlink()
        print(f"  Deleted {len(action_files)} original action files")

    return len(action_files)


def consolidate_infos(infos_dir: Path, output_file: Path, delete_originals: bool = False) -> int:
    """
    Consolidate all info_*.json files into a single JSON file.

    Format: {"video_stem": {info_dict}, ...}

    Returns number of files consolidated.
    """
    consolidated = {}
    info_files = list(infos_dir.glob("info_*.json")) + list(infos_dir.glob("info_*.jsonl"))

    for info_path in info_files:
        # Extract stem: info_chop_a_tree_freq_25_..._seed_1.json -> chop_a_tree_freq_25_..._seed_1
        stem = info_path.stem[len("info_"):]

        with info_path.open("r", encoding="utf-8") as fp:
            if info_path.suffix == ".jsonl":
                info_data = json.loads(fp.readline())
            else:
                info_data = json.load(fp)

        consolidated[stem] = info_data

    # Atomic write: tmp + rename (see consolidate_actions for rationale).
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(consolidated, fp, separators=(',', ':'))
    tmp.replace(output_file)

    if delete_originals and info_files:
        for f in info_files:
            f.unlink()
        print(f"  Deleted {len(info_files)} original info files")

    return len(info_files)


def consolidate_trajectory(traj_dir: Path, delete_originals: bool = False) -> dict:
    """Consolidate all metadata in a trajectory directory."""
    actions_dir = traj_dir / "actions"
    infos_dir = traj_dir / "infos"

    stats = {"actions": 0, "infos": 0}

    if actions_dir.exists():
        output_file = traj_dir / "all_actions.json"
        if not output_file.exists():
            stats["actions"] = consolidate_actions(actions_dir, output_file, delete_originals)
            print(f"  Consolidated {stats['actions']} action files -> all_actions.json")
        else:
            print(f"  all_actions.json already exists, skipping")

    if infos_dir.exists():
        output_file = traj_dir / "all_infos.json"
        if not output_file.exists():
            stats["infos"] = consolidate_infos(infos_dir, output_file, delete_originals)
            print(f"  Consolidated {stats['infos']} info files -> all_infos.json")
        else:
            print(f"  all_infos.json already exists, skipping")

    # Remove empty directories after deletion
    if delete_originals:
        for subdir in [actions_dir, infos_dir]:
            if subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()
                print(f"  Removed empty directory: {subdir.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Consolidate action/info files to reduce inodes")
    parser.add_argument("--data-dir", required=True, help="Root directory containing trajectory_task_* folders")
    parser.add_argument("--delete-originals", action="store_true",
                        help="Delete original files after consolidation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    root = Path(args.data_dir)

    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return 1

    total_actions = 0
    total_infos = 0

    # Find all trajectory directories
    traj_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("trajectory_task_")])

    if not traj_dirs:
        print(f"No trajectory_task_* directories found in {root}")
        return 1

    print(f"Found {len(traj_dirs)} trajectory directories")
    print(f"Delete originals: {args.delete_originals}")
    print()

    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
        for traj_dir in traj_dirs:
            actions_dir = traj_dir / "actions"
            infos_dir = traj_dir / "infos"

            action_count = len(list(actions_dir.glob("action_*.jsonl"))) if actions_dir.exists() else 0
            info_count = len(list(infos_dir.glob("info_*.json"))) + len(list(infos_dir.glob("info_*.jsonl"))) if infos_dir.exists() else 0

            print(f"{traj_dir.name}:")
            print(f"  Would consolidate {action_count} action files")
            print(f"  Would consolidate {info_count} info files")
            total_actions += action_count
            total_infos += info_count
    else:
        for traj_dir in traj_dirs:
            print(f"Processing: {traj_dir.name}")
            stats = consolidate_trajectory(traj_dir, args.delete_originals)
            total_actions += stats["actions"]
            total_infos += stats["infos"]
            print()

    print("=" * 50)
    print(f"Total actions consolidated: {total_actions}")
    print(f"Total infos consolidated: {total_infos}")
    print(f"Inodes saved: {total_actions + total_infos - len(traj_dirs) * 2}")

    return 0


if __name__ == "__main__":
    exit(main())
