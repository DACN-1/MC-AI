#!/usr/bin/env python3
"""Cluster pipeline: convert videos -> train model -> evaluate.

Wraps the shared functions in `imitation_learning` and adds a parallel
MP4 -> HDF5 conversion step. Intended for SLURM jobs (see slurm_train.sh).

Usage::

    python cluster_pipeline.py --data-dir ./trajectories --output-dir ./output

Expected input layout::

    trajectories/
    ├── trajectory_task_*/
    │   ├── all_actions.json
    │   ├── all_infos.json
    │   └── videos/*.mp4
    └── (frames_chunked/ created during conversion)

Outputs ./output/model.pt and ./output/metrics.json.
"""

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch as th
from torch.utils.data import random_split
from tqdm import tqdm

from VLAAgent import VLAAgent
from chunk_frames import FrameChunker
from constants import NUM_OUTPUT_LOGITS
from imitation_learning import TrajectoryDataset, evaluate, train_vla


# ---------------------------------------------------------------------
# Video conversion
# ---------------------------------------------------------------------
def _convert_one(args):
    video_path, output_file, frames_per_chunk, delete_after = args
    try:
        FrameChunker(frames_per_chunk=frames_per_chunk, compression="gzip").chunk_video(
            Path(video_path), Path(output_file), verbose=False
        )
        if delete_after:
            Path(video_path).unlink()
        return True, video_path
    except Exception as e:  # noqa: BLE001 — surfaced through the progress bar
        return False, f"{video_path}: {e}"


def convert_videos(
    data_dir: Path,
    frames_per_chunk: int = 100,
    delete_videos: bool = True,
    num_workers: int | None = None,
) -> None:
    output_dir = data_dir / "frames_chunked"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files: list[Path] = []
    for traj_dir in sorted(data_dir.glob("trajectory_task_*")):
        videos_dir = traj_dir / "videos"
        if videos_dir.exists():
            video_files.extend(sorted(videos_dir.glob("*.mp4")))

    if not video_files:
        print("No videos to convert")
        return

    tasks = []
    for video_path in video_files:
        stem = video_path.stem
        if stem.startswith("video_"):
            stem = stem[len("video_") :]
        output_file = output_dir / f"video_{stem}.h5"
        if output_file.exists():
            continue
        tasks.append((str(video_path), str(output_file), frames_per_chunk, delete_videos))

    if not tasks:
        print("All videos already converted")
        return

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"Converting {len(tasks)} videos to HDF5 with {num_workers} workers…")
    converted = failed = 0
    with Pool(num_workers) as pool:
        for success, msg in tqdm(
            pool.imap_unordered(_convert_one, tasks),
            total=len(tasks),
            desc="Converting",
        ):
            if success:
                converted += 1
            else:
                failed += 1
                print(f"Failed: {msg}")

    if delete_videos:
        for traj_dir in data_dir.glob("trajectory_task_*"):
            videos_dir = traj_dir / "videos"
            if videos_dir.exists() and not any(videos_dir.iterdir()):
                videos_dir.rmdir()

    print(f"Conversion complete: {converted} converted, {failed} failed")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster pipeline: convert -> train -> evaluate")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with trajectory data")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--convert-workers",
        type=int,
        default=None,
        help="Workers for parallel MP4 -> HDF5 conversion",
    )
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Load existing model and run eval only")
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if not args.skip_convert:
        print("\n" + "=" * 60)
        print("Step 1: Converting Videos to HDF5")
        print("=" * 60)
        convert_videos(args.data_dir, delete_videos=True, num_workers=args.convert_workers)

    model_path = args.output_dir / "model.pt"

    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 2: Training Model")
        print("=" * 60)
        model, test_set, _ = train_vla(
            data_root=str(args.data_dir),
            backbone=args.llava_model,
            out_weights=str(model_path),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:
        print(f"Loading existing model from {model_path}")
        ckpt = th.load(model_path, map_location=args.device)
        model = VLAAgent(NUM_OUTPUT_LOGITS, ckpt["llava_model"]).to(args.device)
        model.action_head.load_state_dict(ckpt["state_dict"])
        dataset = TrajectoryDataset(str(args.data_dir))
        test_size = int(len(dataset) * 0.1)
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size - test_size
        _, _, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=th.Generator().manual_seed(42),
        )

    print("\n" + "=" * 60)
    print("Step 3: Evaluating Model")
    print("=" * 60)
    evaluate(
        model,
        test_set,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
