#!/usr/bin/env python3
"""
Lean cluster pipeline: Convert videos -> Train model -> Run metrics

Usage:
    python cluster_pipeline.py --data-dir ./trajectories --output-dir ./output

Expected input structure:
    trajectories/
    ├── trajectory_task_*/
    │   ├── all_actions.json   # Consolidated actions
    │   ├── all_infos.json     # Consolidated infos
    │   └── videos/*.mp4       # Raw videos
    └── (frames_chunked/ created during conversion)

Output:
    output/
    ├── model.pt               # Trained model weights
    └── metrics.json           # Evaluation metrics
"""

import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from VLAAgent import VLAAgent
from chunk_frames import FrameChunker
from imitation_learning import TrajectoryDataset, CANONICAL_ACTION_KEYS, NUM_ACTIONS


def convert_single_video(args):
    """Convert a single video to HDF5. Used for parallel processing."""
    video_path, output_file, frames_per_chunk, delete_after = args
    try:
        chunker = FrameChunker(frames_per_chunk=frames_per_chunk, compression="gzip")
        chunker.chunk_video(Path(video_path), Path(output_file), verbose=False)
        if delete_after:
            Path(video_path).unlink()
        return True, video_path
    except Exception as e:
        return False, f"{video_path}: {e}"


def convert_videos(data_dir: Path, frames_per_chunk: int = 100, delete_videos: bool = True, num_workers: int = None):
    """Convert all MP4 videos to HDF5 format using parallel processing."""
    output_dir = data_dir / "frames_chunked"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    video_files = []
    for traj_dir in sorted(data_dir.glob("trajectory_task_*")):
        videos_dir = traj_dir / "videos"
        if videos_dir.exists():
            video_files.extend(sorted(videos_dir.glob("*.mp4")))

    if not video_files:
        print("No videos to convert")
        return

    print(f"Converting {len(video_files)} videos to HDF5...")

    # Prepare arguments for parallel processing
    tasks = []
    for video_path in video_files:
        stem = video_path.stem
        if stem.startswith("video_"):
            stem = stem[len("video_"):]
        output_file = output_dir / f"video_{stem}.h5"

        if output_file.exists():
            continue  # Skip existing
        tasks.append((str(video_path), str(output_file), frames_per_chunk, delete_videos))

    if not tasks:
        print("All videos already converted")
        return

    # Use multiprocessing for parallel conversion
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    converted, failed = 0, 0
    with Pool(num_workers) as pool:
        for success, msg in tqdm(pool.imap_unordered(convert_single_video, tasks), total=len(tasks), desc="Converting"):
            if success:
                converted += 1
            else:
                failed += 1
                print(f"Failed: {msg}")

    # Clean up empty video directories
    if delete_videos:
        for traj_dir in data_dir.glob("trajectory_task_*"):
            videos_dir = traj_dir / "videos"
            if videos_dir.exists() and not any(videos_dir.iterdir()):
                videos_dir.rmdir()

    print(f"Conversion complete: {converted} converted, {failed} failed")


def train(data_dir: Path, output_dir: Path, llava_model: str, epochs: int, batch_size: int, lr: float, device: str):
    """Train the VLA model."""
    print(f"\n{'='*60}")
    print("Training VLA Model")
    print(f"{'='*60}")

    # Load dataset
    dataset = TrajectoryDataset(str(data_dir))
    print(f"Dataset: {len(dataset)} samples")

    # Split: 80% train, 10% val, 10% test
    test_size = int(len(dataset) * 0.1)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=th.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    # DataLoaders
    collate = lambda b: list(zip(*b))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Model
    model = VLAAgent(NUM_ACTIONS, llava_model).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.action_head.parameters(), lr=lr)

    # Training loop
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for imgs, texts, acts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            acts = th.stack(acts).to(device)
            optimizer.zero_grad()
            logits = model(list(imgs), texts)
            loss = loss_fn(logits, acts)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        val_loss, all_preds, all_targets = 0, [], []
        with th.no_grad():
            for imgs, texts, acts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                acts = th.stack(acts).to(device)
                logits = model(list(imgs), texts)
                val_loss += loss_fn(logits, acts).item()
                preds = (th.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(acts.cpu())

        val_losses.append(val_loss / len(val_loader))
        all_preds = th.cat(all_preds)
        all_targets = th.cat(all_targets)
        val_acc = (all_preds == all_targets).float().mean().item()
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, val_acc={val_acc:.4f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    th.save({
        "llava_model": llava_model,
        "state_dict": model.action_head.state_dict(),
        "training_metrics": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accs,
        },
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr},
    }, model_path)
    print(f"Model saved: {model_path}")

    return model, test_set, device


def evaluate(model, test_set, device: str, output_dir: Path, batch_size: int = 8):
    """Evaluate model and compute metrics."""
    print(f"\n{'='*60}")
    print("Evaluating Model")
    print(f"{'='*60}")

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                             collate_fn=lambda b: list(zip(*b)))

    model.eval()
    all_preds, all_targets = [], []

    with th.no_grad():
        for imgs, texts, acts in tqdm(test_loader, desc="Evaluating"):
            acts = th.stack(acts).to(device)
            logits = model(list(imgs), texts)
            preds = (th.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(acts.cpu())

    all_preds = th.cat(all_preds).numpy()
    all_targets = th.cat(all_targets).numpy()

    # Compute metrics
    accuracy = (all_preds == all_targets).mean()

    # Per-action metrics
    per_action = {}
    for i, action in enumerate(CANONICAL_ACTION_KEYS):
        p, r, f1, _ = precision_recall_fscore_support(
            all_targets[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        per_action[action] = {"precision": float(p), "recall": float(r), "f1": float(f1)}

    metrics = {
        "test_accuracy": float(accuracy),
        "test_samples": len(all_preds),
        "per_action_metrics": per_action,
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Print summary
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nPer-action F1 scores:")
    for action, m in sorted(per_action.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {action:15s}: {m['f1']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cluster pipeline: convert -> train -> evaluate")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with trajectory data")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf", help="LLaVA model name")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-convert", action="store_true", help="Skip video conversion")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (eval only)")
    parser.add_argument("--num-workers", type=int, default=None, help="Workers for video conversion")
    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not th.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    start_time = time.time()

    # Step 1: Convert videos to HDF5
    if not args.skip_convert:
        print(f"\n{'='*60}")
        print("Step 1: Converting Videos to HDF5")
        print(f"{'='*60}")
        convert_videos(args.data_dir, delete_videos=True, num_workers=args.num_workers)

    # Step 2: Train model
    if not args.skip_train:
        print(f"\n{'='*60}")
        print("Step 2: Training Model")
        print(f"{'='*60}")
        model, test_set, device = train(
            args.data_dir, args.output_dir, args.llava_model,
            args.epochs, args.batch_size, args.lr, args.device
        )
    else:
        # Load existing model for evaluation
        model_path = args.output_dir / "model.pt"
        checkpoint = th.load(model_path, map_location=args.device)
        model = VLAAgent(NUM_ACTIONS, checkpoint["llava_model"]).to(args.device)
        model.action_head.load_state_dict(checkpoint["state_dict"])
        dataset = TrajectoryDataset(str(args.data_dir))
        _, _, test_set = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)],
                                       generator=th.Generator().manual_seed(42))
        device = args.device

    # Step 3: Evaluate
    print(f"\n{'='*60}")
    print("Step 3: Evaluating Model")
    print(f"{'='*60}")
    evaluate(model, test_set, device, args.output_dir, args.batch_size)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
