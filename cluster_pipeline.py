#!/usr/bin/env python3
"""Cluster pipeline: train model -> evaluate.

Thin wrapper around `imitation_learning.train_vla` / `evaluate`. Intended for
SLURM jobs (see slurm_train.sh). Frames are decoded directly from the MP4s
in each trajectory's `videos/` folder via `decord`; there is no offline
conversion step.

Usage::

    python cluster_pipeline.py --data-dir ./trajectories --output-dir ./output

Expected input layout::

    trajectories/
    └── trajectory_task_*/
        ├── all_actions.json
        ├── all_infos.json
        └── videos/*.mp4

Outputs ./output/model.pt and ./output/metrics.json.
"""

import argparse
import time
from pathlib import Path

import torch as th
from torch.utils.data import random_split

from VLAAgent import VLAAgent
from constants import NUM_OUTPUT_LOGITS
from imitation_learning import TrajectoryDataset, evaluate, train_vla


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster pipeline: train -> evaluate")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with trajectory data")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--skip-train", action="store_true", help="Load existing model and run eval only")
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    model_path = args.output_dir / "model.pt"

    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 1: Training Model")
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
    print("Step 2: Evaluating Model")
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
