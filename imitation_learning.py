from argparse import ArgumentParser
import pickle
import time
# import minerl
import torch as th
import numpy as np
from pathlib import Path
# from VPT.agent import PI_HEAD_KWARGS, MineRLAgent  # Not needed for action head training
# from VPT.lib.tree_util import tree_map  # Not needed for action head training
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
)
import json
from VLAAgent import VLAAgent
import h5py

# ---------------------------------------------------------------------
# Canonical action mapping
# ---------------------------------------------------------------------
CANONICAL_ACTION_KEYS = [
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "camera_x",
    "camera_y",
    "ESC"
]
NUM_ACTIONS = len(CANONICAL_ACTION_KEYS)


def action_to_tensor(action_dict):
    """Convert a canonical‑ordered action dict into a tensor with camera movement preserved."""
    vec = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for i, k in enumerate(CANONICAL_ACTION_KEYS):
        if k == "camera_x":
            camera = action_dict.get("camera", [[0, 0]])
            # Handle nested format [[x, y]] -> x
            if isinstance(camera, list) and len(camera) > 0:
                if isinstance(camera[0], list) and len(camera[0]) > 0:
                    vec[i] = float(camera[0][0])
                else:
                    vec[i] = float(camera[0]) if len(camera) > 0 else 0.0
            else:
                vec[i] = 0.0
        elif k == "camera_y":
            camera = action_dict.get("camera", [[0, 0]])
            # Handle nested format [[x, y]] -> y  
            if isinstance(camera, list) and len(camera) > 0:
                if isinstance(camera[0], list) and len(camera[0]) > 1:
                    vec[i] = float(camera[0][1])
                else:
                    vec[i] = float(camera[1]) if len(camera) > 1 else 0.0
            else:
                vec[i] = 0.0
        else:
            v = action_dict.get(k, 0)
            vec[i] = float(bool(v))
    return th.from_numpy(vec)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    Dataset for VLA agent training using HDF5 chunked frames.

    Expected structure (consolidated format - preferred for low inode count):
        root_dir/
        ├── trajectory_task_<task>_length_<N>/
        │   ├── all_actions.json   # {"stem": [action1, action2, ...], ...}
        │   └── all_infos.json     # {"stem": {info_dict}, ...}
        └── frames_chunked/
            └── video_<stem>.h5

    Legacy structure (individual files - still supported):
        root_dir/
        ├── trajectory_task_<task>_length_<N>/
        │   ├── actions/action_<stem>.jsonl
        │   └── infos/info_<stem>.json
        └── frames_chunked/
            └── video_<stem>.h5

    HDF5 files are created by chunk_frames.py or build_dataset.py.
    Use consolidate_metadata.py to convert individual files to consolidated format.
    """
    def __init__(self, root_dir: str, use_h5: bool = True):
        """
        Initialize TrajectoryDataset.

        Args:
            root_dir: Root directory containing trajectory_task_* folders
            use_h5: Must be True (PNG support removed for simplicity)
        """
        if not use_h5:
            raise ValueError(
                "PNG frame support has been removed. Please convert videos to HDF5 using:\n"
                "  python chunk_frames.py --data-dir <root_dir>\n"
                "Or use the complete pipeline:\n"
                "  python train_pipeline.py --input-videos <videos> --output-dir <output>"
            )

        self.samples: list[tuple[str, int, str, dict]] = []  # (h5_path, frame_idx, text, action)
        self.h5_samples = 0

        root = Path(root_dir)
        h5_root = root / "frames_chunked"

        if not h5_root.exists():
            raise FileNotFoundError(
                f"HDF5 frames directory not found: {h5_root}\n"
                f"Please run: python chunk_frames.py --data-dir {root_dir}"
            )

        # Only process directories, not files (important for cluster environments)
        for item in root.iterdir():
            if not item.is_dir():
                continue  # Skip files in trajectories root
            if not item.name.startswith("trajectory_task_"):
                continue  # Skip non-trajectory directories

            traj_dir = item

            # Check for consolidated files first (preferred - fewer inodes)
            all_actions_path = traj_dir / "all_actions.json"
            all_infos_path = traj_dir / "all_infos.json"

            if all_actions_path.exists():
                # Use consolidated format
                self._load_consolidated(traj_dir, all_actions_path, all_infos_path, h5_root)
            else:
                # Fall back to individual files (legacy format)
                self._load_individual_files(traj_dir, h5_root)

    def _load_consolidated(self, traj_dir: Path, all_actions_path: Path, all_infos_path: Path, h5_root: Path):
        """Load from consolidated all_actions.json and all_infos.json files."""
        # Load all actions
        with all_actions_path.open("r", encoding="utf-8") as fp:
            all_actions = json.load(fp)

        # Load all infos (optional)
        all_infos = {}
        if all_infos_path.exists():
            with all_infos_path.open("r", encoding="utf-8") as fp:
                all_infos = json.load(fp)

        for stem, actions in all_actions.items():
            # Get task text from infos
            info_data = all_infos.get(stem, {})
            task_text = info_data.get("text_prompt", "play minecraft")

            # Load HDF5 file (required)
            h5_file = h5_root / f"video_{stem}.h5"
            if not h5_file.exists():
                print(f"Warning: Missing HDF5 file for {stem}, skipping")
                continue

            try:
                with h5py.File(h5_file, 'r') as f:
                    total_frames = f.attrs.get('total_frames', f['frames'].shape[0])
                    if total_frames != len(actions):
                        print(f"Warning: Frame/action mismatch for {stem} ({total_frames} frames vs {len(actions)} actions), skipping")
                        continue

                    # Add HDF5 samples
                    for idx in range(len(actions)):
                        self.samples.append((
                            str(h5_file),
                            idx,
                            task_text,
                            actions[idx]
                        ))
                    self.h5_samples += len(actions)
            except Exception as e:
                print(f"Error: Failed to read H5 file {h5_file}: {e}")

    def _load_individual_files(self, traj_dir: Path, h5_root: Path):
        """Load from individual action_*.jsonl and info_*.json files (legacy format)."""
        actions_dir = traj_dir / "actions"
        infos_dir = traj_dir / "infos"

        if not actions_dir.exists():
            return

        for action_path in actions_dir.glob("action_*.jsonl"):
            stem = action_path.stem[len("action_"):]  # remove 'action_'

            # Load actions
            with action_path.open("r", encoding="utf-8") as fp:
                actions = [json.loads(ln) for ln in fp]

            # Get task text from corresponding info file
            info_path_jsonl = infos_dir / f"info_{stem}.jsonl"
            info_path_json = infos_dir / f"info_{stem}.json"
            info_data = {}
            if info_path_jsonl.exists():
                with info_path_jsonl.open("r", encoding="utf-8") as fp:
                    info_data = json.loads(fp.readline())
            elif info_path_json.exists():
                with info_path_json.open("r", encoding="utf-8") as fp:
                    info_data = json.load(fp)

            task_text = info_data.get("text_prompt", "play minecraft")

            # Load HDF5 file (required)
            h5_file = h5_root / f"video_{stem}.h5"
            if not h5_file.exists():
                print(f"Warning: Missing HDF5 file for {stem}, skipping")
                continue

            try:
                with h5py.File(h5_file, 'r') as f:
                    total_frames = f.attrs.get('total_frames', f['frames'].shape[0])
                    if total_frames != len(actions):
                        print(f"Warning: Frame/action mismatch for {stem} ({total_frames} frames vs {len(actions)} actions), skipping")
                        continue

                    # Add HDF5 samples
                    for idx in range(len(actions)):
                        self.samples.append((
                            str(h5_file),
                            idx,
                            task_text,
                            actions[idx]
                        ))
                    self.h5_samples += len(actions)
            except Exception as e:
                print(f"Error: Failed to read H5 file {h5_file}: {e}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, frame_idx, task_text, act_dict = self.samples[idx]

        # Load from HDF5 (only format supported)
        with h5py.File(h5_path, 'r') as f:
            frame_array = f['frames'][frame_idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray(frame_array)

        return image, task_text, action_to_tensor(act_dict)

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            'total_samples': len(self.samples),
            'h5_samples': self.h5_samples,
        }

def train_vla(
    data_root: str,
    backbone: str,
    out_weights: str,
    batch_size: int = 8,
    epochs: int = 2,
    lr: float = 1e-4,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
    val_split: float = 0.1,
    test_split: float = 0.1,
    train_data_only: bool = False,
    use_h5: bool = True,
):
    # GPU/Device Setup
    print("\n" + "=" * 70)
    print("🖥️  Device Configuration")
    print("=" * 70)

    if device == "cuda":
        if not th.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = "cpu"
        else:
            gpu_count = th.cuda.device_count()
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                props = th.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            print(f"   Using device: {device}")

            # Set default device
            th.cuda.set_device(0)
    else:
        print(f"Using device: {device}")

    print("=" * 70 + "\n")

    dataset = TrajectoryDataset(data_root, use_h5=use_h5)

    # Print dataset statistics
    stats = dataset.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  HDF5 chunks: {stats['h5_samples']:,} frames")
    print(f"  ✓ Using HDF5 chunked format for optimal performance")
    print()
    
    if train_data_only:
        # Use all data for training (when data_root points to train split)
        train_dataset = dataset
        val_dataset = dataset  # Use same data for validation (not ideal but for compatibility)
        print(f"Using all {len(dataset)} samples for training (train_data_only=True)")
    else:
        # Split dataset into train/validation/test
        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=th.Generator().manual_seed(42)  # Reproducible splits
        )

        # Save test set for later evaluation
        test_indices = test_dataset.indices
        test_samples = [dataset.samples[i] for i in test_indices]

        test_info = {
            "test_indices": test_indices,
            "test_size": test_size,
            "test_samples": len(test_samples),
        }

        # Save test set info with model
        test_path = Path(out_weights).parent / f"test_set_{int(time.time())}.json"
        with test_path.open("w") as f:
            json.dump(test_info, f, indent=2)
        print(f"Test set info saved to: {test_path}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: list(zip(*b)),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: list(zip(*b)),
    )
    
    if train_data_only:
        print(f"Dataset: {len(dataset)} samples (train_data_only mode)")
    else:
        print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples")

    model  = VLAAgent(NUM_ACTIONS,backbone).to(device)
    loss_f = nn.BCEWithLogitsLoss()
    optim  = th.optim.Adam(model.action_head.parameters(), lr=lr)

    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for ep in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for idx, (imgs, texts, acts) in enumerate(train_loader):
            print(f"Training batch {idx+1}/{len(train_loader)}", end="\r")
            imgs  = list(imgs)
            acts  = th.stack(acts).to(device)
            
            optim.zero_grad()
            logits = model(imgs, texts)
            loss   = loss_f(logits, acts)
            loss.backward()
            optim.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with th.no_grad():
            for idx, (imgs, texts, acts) in enumerate(val_loader):
                print(f"Validation batch {idx+1}/{len(val_loader)}", end="\r")
                imgs = list(imgs)
                acts = th.stack(acts).to(device)
                
                logits = model(imgs, texts)
                loss = loss_f(logits, acts)
                val_loss += loss.item()
                
                # Convert logits to predictions (sigmoid > 0.5)
                preds = (th.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(acts.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation accuracy
        all_preds = th.cat(all_preds, dim=0)
        all_targets = th.cat(all_targets, dim=0)
        val_acc = (all_preds == all_targets).float().mean().item()
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch {ep+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print("-" * 40)

    # Save model with training metrics
    th.save(
        {
            "llava_model": backbone,
            "state_dict":  model.action_head.state_dict(),
            "training_metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "final_val_acc": val_accuracies[-1] if val_accuracies else 0.0
            },
            "config": {
                "num_actions": NUM_ACTIONS,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_split": val_split
            }
        },
        out_weights,
    )
    
    print(f"\nTraining completed!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Model saved to: {out_weights}")
# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--out-weights", required=True)
    p.add_argument("--epochs",      type=int, default=2)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--test-split",  type=float, default=0.1)
    p.add_argument("--train-data-only", action="store_true",
                   help="Use all data for training (when using pre-split datasets)")
    a = p.parse_args()

    train_vla(
        data_root   = a.data_dir,
        backbone    = a.llava_model,
        out_weights = a.out_weights,
        batch_size  = a.batch_size,
        epochs      = a.epochs,
        lr          = a.lr,
        device      = a.device,
        val_split   = a.val_split,
        test_split  = a.test_split,
        train_data_only = a.train_data_only,
        use_h5      = True,  # HDF5 is now mandatory
    )

