"""VLA imitation learning: dataset, training loop, and evaluation."""

from argparse import ArgumentParser
from collections import OrderedDict
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from VLAAgent import VLAAgent
from constants import (
    BINARY_ACTION_KEYS,
    CANONICAL_ACTION_KEYS,
    DEFAULT_PAST_ACTION_K,
    NUM_ACTIONS,
    NUM_BINARY,
    NUM_CAMERA,
    NUM_CAMERA_BINS,
    NUM_OUTPUT_LOGITS,
    PAST_ACTION_DIM,
    action_to_onehot,
    action_to_tensor,
)
from vpt_camera import DEFAULT_CAMERA_QUANTIZER


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Dataset over (frame, prompt, target_chunk, past_actions) tuples.

    Expected layout::

        root_dir/
        └── trajectory_task_<task>_length_<N>/
            ├── all_actions.json           (preferred — consolidated)
            ├── all_infos.json             (optional)
            ├── videos/video_<stem>.mp4
            └── (legacy)
                ├── actions/action_<stem>.jsonl
                └── infos/info_<stem>.json

    Frames are decoded on demand with `decord.VideoReader`. One reader is
    cached per worker per file, bounded by `decoder_cache_size` (LRU).

    `chunk_size > 1` makes each sample's target the *next* `chunk_size` actions
    (action-chunking, ACT-style). Frames within `chunk_size - 1` of the end of
    a trajectory are dropped so every sample has a full target.

    `past_action_k > 0` adds a flattened (K * PAST_ACTION_DIM,) past-action
    feature vector to every sample, zero-padded for early frames. Encoded as
    binary + per-axis one-hot camera bins (see `constants.action_to_onehot`).
    """

    def __init__(
        self,
        root_dir: str,
        decoder_cache_size: int = 64,
        past_action_k: int = 0,
        chunk_size: int = 1,
    ):
        self.samples: list[tuple[str, int, str, str]] = []
        self.decoder_cache_size = decoder_cache_size
        self.past_action_k = past_action_k
        self.chunk_size = chunk_size

        # Per-stem caches built once at init. Memory: 6M actions × 23 floats
        # ≈ 552 MB for targets, ≈ 1 GB for onehots — fits 512 GB easily.
        self.targets_by_stem: dict[str, np.ndarray] = {}
        self.onehots_by_stem: dict[str, np.ndarray] = {}

        # opened lazily per worker; never share across processes
        self._decoder_cache: OrderedDict | None = None
        self._VideoReader = None

        root = Path(root_dir)

        for item in root.iterdir():
            if not item.is_dir() or not item.name.startswith("trajectory_task_"):
                continue

            all_actions_path = item / "all_actions.json"
            if all_actions_path.exists():
                self._load_consolidated(item, all_actions_path, item / "all_infos.json")
            else:
                self._load_individual_files(item)

    def _add_samples(
        self, stem: str, actions: list, task_text: str, videos_dir: Path
    ) -> None:
        mp4_file = videos_dir / f"video_{stem}.mp4"
        if not mp4_file.exists():
            print(f"Warning: missing MP4 for {stem}, skipping")
            return

        # Precompute per-stem target tensors (cam bin indices, NOT one-hots).
        self.targets_by_stem[stem] = np.stack(
            [action_to_tensor(a).numpy() for a in actions]
        ).astype(np.float32)

        if self.past_action_k > 0:
            self.onehots_by_stem[stem] = np.stack(
                [action_to_onehot(a) for a in actions]
            ).astype(np.float32)

        # Drop the last (chunk_size - 1) frames per trajectory — they don't
        # have a full target chunk.
        last_valid = len(actions) - self.chunk_size
        for idx in range(last_valid + 1):
            self.samples.append((str(mp4_file), idx, task_text, stem))

    def _load_consolidated(
        self, traj_dir: Path, all_actions_path: Path, all_infos_path: Path
    ) -> None:
        with all_actions_path.open("r", encoding="utf-8") as fp:
            all_actions = json.load(fp)
        all_infos: dict = {}
        if all_infos_path.exists():
            with all_infos_path.open("r", encoding="utf-8") as fp:
                all_infos = json.load(fp)
        videos_dir = traj_dir / "videos"
        for stem, actions in all_actions.items():
            task_text = all_infos.get(stem, {}).get("text_prompt", "play minecraft")
            self._add_samples(stem, actions, task_text, videos_dir)

    def _load_individual_files(self, traj_dir: Path) -> None:
        actions_dir = traj_dir / "actions"
        infos_dir = traj_dir / "infos"
        videos_dir = traj_dir / "videos"
        if not actions_dir.exists():
            return
        for action_path in actions_dir.glob("action_*.jsonl"):
            stem = action_path.stem[len("action_") :]
            with action_path.open("r", encoding="utf-8") as fp:
                actions = [json.loads(ln) for ln in fp if ln.strip()]
            info_data: dict = {}
            for candidate in (infos_dir / f"info_{stem}.jsonl", infos_dir / f"info_{stem}.json"):
                if candidate.exists():
                    with candidate.open("r", encoding="utf-8") as fp:
                        info_data = json.loads(fp.readline()) if candidate.suffix == ".jsonl" else json.load(fp)
                    break
            task_text = info_data.get("text_prompt", "play minecraft")
            self._add_samples(stem, actions, task_text, videos_dir)

    def _decoder(self, path: str):
        if self._decoder_cache is None:
            self._decoder_cache = OrderedDict()
            try:
                from decord import VideoReader
            except ImportError as e:
                raise ImportError(
                    "decord is required for MP4 frame decoding. "
                    "Install with: pip install decord"
                ) from e
            self._VideoReader = VideoReader
        dec = self._decoder_cache.get(path)
        if dec is None:
            # num_threads=1: each DataLoader worker is already its own process,
            # so per-reader threading would oversubscribe the CPUs.
            dec = self._VideoReader(path, num_threads=1)
            self._decoder_cache[path] = dec
            if len(self._decoder_cache) > self.decoder_cache_size:
                self._decoder_cache.popitem(last=False)
        else:
            self._decoder_cache.move_to_end(path)
        return dec

    def _past_actions(self, stem: str, frame_idx: int) -> th.Tensor:
        if self.past_action_k == 0:
            return th.zeros(0, dtype=th.float32)
        onehots = self.onehots_by_stem[stem]
        past = np.zeros((self.past_action_k, PAST_ACTION_DIM), dtype=np.float32)
        start = max(0, frame_idx - self.past_action_k)
        n_valid = frame_idx - start
        if n_valid > 0:
            # Place most-recent action at the last slot so the position-in-vector
            # is stable regardless of how much history is available.
            past[self.past_action_k - n_valid :] = onehots[start:frame_idx]
        return th.from_numpy(past.reshape(-1))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx):
        mp4_path, frame_idx, task_text, stem = self.samples[idx]
        frame_array = self._decoder(mp4_path)[frame_idx].asnumpy()
        target_chunk = th.from_numpy(
            self.targets_by_stem[stem][frame_idx : frame_idx + self.chunk_size]
        )  # (chunk_size, NUM_BINARY + NUM_CAMERA)
        past = self._past_actions(stem, frame_idx)
        return Image.fromarray(frame_array), task_text, target_chunk, past

    def get_stats(self) -> dict:
        return {"total_samples": len(self.samples)}


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
_CAM_X_SLICE = slice(NUM_BINARY, NUM_BINARY + NUM_CAMERA_BINS)
_CAM_Y_SLICE = slice(NUM_BINARY + NUM_CAMERA_BINS, NUM_BINARY + 2 * NUM_CAMERA_BINS)


def vla_loss(logits: th.Tensor, targets: th.Tensor) -> tuple[th.Tensor, float, float]:
    """BCE on binary actions + cross-entropy on each camera axis, averaged over
    the chunk axis when present.

    Args:
        logits:  (B, NUM_OUTPUT_LOGITS) or (B, N, NUM_OUTPUT_LOGITS)
        targets: (B, NUM_BINARY + NUM_CAMERA) or (B, N, NUM_BINARY + NUM_CAMERA)
                 last 2 entries are bin indices (stored as float32 for collate
                 uniformity; cast to long here).

    Returns (total_loss, bce_value, camera_ce_value).
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    if targets.dim() == 3:
        targets = targets.reshape(-1, targets.size(-1))

    binary_logits = logits[:, :NUM_BINARY]
    binary_targets = targets[:, :NUM_BINARY]

    cam_x_logits = logits[:, _CAM_X_SLICE]
    cam_y_logits = logits[:, _CAM_Y_SLICE]
    cam_x_targets = targets[:, NUM_BINARY].long()
    cam_y_targets = targets[:, NUM_BINARY + 1].long()

    bce = nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    ce_x = nn.functional.cross_entropy(cam_x_logits, cam_x_targets)
    ce_y = nn.functional.cross_entropy(cam_y_logits, cam_y_targets)
    cam_ce = 0.5 * (ce_x + ce_y)

    return bce + cam_ce, bce.item(), cam_ce.item()


def _collate(batch):
    """DataLoader collate.

    Returns (imgs_list, texts_list, targets_BxNxD, past_actions_BxP). If the
    dataset's past_action_k==0, past_actions has zero columns (B, 0); the model
    ignores it.
    """
    imgs, texts, targets, pasts = zip(*batch)
    return list(imgs), list(texts), th.stack(targets), th.stack(pasts)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device == "cuda" and not th.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        return "cpu"
    return device


def _print_device_info(device: str) -> None:
    print("\n" + "=" * 70)
    print("🖥️  Device Configuration")
    print("=" * 70)
    if device == "cuda":
        for i in range(th.cuda.device_count()):
            props = th.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
    print(f"   Using device: {device}")
    print("=" * 70 + "\n")


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
    num_workers: int = 2,
    past_action_k: int = DEFAULT_PAST_ACTION_K,
    chunk_size: int = 1,
    use_language: bool = True,
):
    """Train the VLA action head and return (model, test_subset, history).

    `past_action_k` and `chunk_size` are the two temporal-context levers (see
    CLAUDE.md). `use_language=False` zeros the text prompt (ablation channel).
    """
    device = _resolve_device(device)
    _print_device_info(device)

    dataset = TrajectoryDataset(
        data_root, past_action_k=past_action_k, chunk_size=chunk_size
    )
    print(
        f"Dataset: {len(dataset):,} samples  "
        f"(past_action_k={past_action_k}  chunk_size={chunk_size})"
    )

    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=th.Generator().manual_seed(42),
    )
    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    out_path = Path(out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_info = {"test_indices": list(test_set.indices), "test_size": test_size}
    test_info_path = out_path.parent / f"test_set_{int(time.time())}.json"
    with test_info_path.open("w") as f:
        json.dump(test_info, f, indent=2)
    print(f"Test set info saved to: {test_info_path}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )

    past_action_dim = past_action_k * PAST_ACTION_DIM
    model = VLAAgent(
        NUM_OUTPUT_LOGITS,
        backbone,
        use_language=use_language,
        past_action_dim=past_action_dim,
        chunk_size=chunk_size,
    ).to(device)
    optim = th.optim.Adam(model.action_head.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_bce": [], "train_cam_ce": [],
        "val_loss": [], "val_bce": [], "val_cam_ce": [],
        "val_binary_acc": [], "val_cam_bin_acc": [],
    }

    def _flat_step_logits(logits_3d: th.Tensor) -> th.Tensor:
        """(B, N, NUM_OUTPUT_LOGITS) -> (B*N, NUM_OUTPUT_LOGITS) for metric slicing."""
        return logits_3d.reshape(-1, logits_3d.size(-1))

    def _flat_step_targets(targets_3d: th.Tensor) -> th.Tensor:
        return targets_3d.reshape(-1, targets_3d.size(-1))

    for ep in range(epochs):
        model.train()
        train_total = train_bce = train_cam = 0.0
        for idx, (imgs, texts, acts, pasts) in enumerate(train_loader):
            print(f"Training batch {idx + 1}/{len(train_loader)}", end="\r")
            acts = acts.to(device)
            pasts = pasts.to(device)
            optim.zero_grad()
            logits = model(imgs, texts, pasts)
            loss, bce_v, cam_v = vla_loss(logits, acts)
            loss.backward()
            optim.step()
            train_total += loss.item()
            train_bce += bce_v
            train_cam += cam_v

        n_train = len(train_loader)
        history["train_loss"].append(train_total / n_train)
        history["train_bce"].append(train_bce / n_train)
        history["train_cam_ce"].append(train_cam / n_train)

        model.eval()
        val_total = val_bce = val_cam = 0.0
        binary_correct = binary_seen = 0
        cam_correct = cam_seen = 0
        with th.no_grad():
            for idx, (imgs, texts, acts, pasts) in enumerate(val_loader):
                print(f"Validation batch {idx + 1}/{len(val_loader)}", end="\r")
                acts = acts.to(device)
                pasts = pasts.to(device)
                logits = model(imgs, texts, pasts)
                loss, bce_v, cam_v = vla_loss(logits, acts)
                val_total += loss.item()
                val_bce += bce_v
                val_cam += cam_v

                logits_flat = _flat_step_logits(logits)
                acts_flat = _flat_step_targets(acts)
                preds = (th.sigmoid(logits_flat[:, :NUM_BINARY]) > 0.5).float()
                binary_correct += (preds == acts_flat[:, :NUM_BINARY]).sum().item()
                binary_seen += preds.numel()

                cam_x_pred = logits_flat[:, _CAM_X_SLICE].argmax(dim=-1)
                cam_y_pred = logits_flat[:, _CAM_Y_SLICE].argmax(dim=-1)
                cam_correct += (cam_x_pred == acts_flat[:, NUM_BINARY].long()).sum().item()
                cam_correct += (cam_y_pred == acts_flat[:, NUM_BINARY + 1].long()).sum().item()
                cam_seen += 2 * cam_x_pred.numel()

        n_val = max(len(val_loader), 1)
        history["val_loss"].append(val_total / n_val)
        history["val_bce"].append(val_bce / n_val)
        history["val_cam_ce"].append(val_cam / n_val)
        history["val_binary_acc"].append(binary_correct / max(binary_seen, 1))
        history["val_cam_bin_acc"].append(cam_correct / max(cam_seen, 1))

        print(f"\nEpoch {ep + 1}/{epochs}:")
        print(
            f"  Train  loss={history['train_loss'][-1]:.4f}"
            f"  bce={history['train_bce'][-1]:.4f}"
            f"  cam_ce={history['train_cam_ce'][-1]:.4f}"
        )
        print(
            f"  Val    loss={history['val_loss'][-1]:.4f}"
            f"  bce={history['val_bce'][-1]:.4f}"
            f"  cam_ce={history['val_cam_ce'][-1]:.4f}"
            f"  bin_acc={history['val_binary_acc'][-1]:.4f}"
            f"  cam_bin_acc={history['val_cam_bin_acc'][-1]:.4f}"
        )
        print("-" * 60)

    th.save(
        {
            "llava_model": backbone,
            "state_dict": model.action_head.state_dict(),
            "training_metrics": history,
            "config": {
                "num_actions": NUM_ACTIONS,
                "num_binary": NUM_BINARY,
                "num_camera": NUM_CAMERA,
                "num_camera_bins": NUM_CAMERA_BINS,
                "num_output_logits": NUM_OUTPUT_LOGITS,
                "past_action_k": past_action_k,
                "past_action_dim": past_action_dim,
                "chunk_size": chunk_size,
                "use_language": use_language,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_split": val_split,
                "test_split": test_split,
                "camera_quantizer": {
                    "camera_maxval": DEFAULT_CAMERA_QUANTIZER.camera_maxval,
                    "camera_binsize": DEFAULT_CAMERA_QUANTIZER.camera_binsize,
                    "mu": DEFAULT_CAMERA_QUANTIZER.mu,
                },
            },
        },
        out_weights,
    )
    print(f"\nTraining complete. Model saved to: {out_weights}")
    return model, test_set, history


# ---------------------------------------------------------------------
# Cached-feature training (used after `feature_cache.precompute`)
# ---------------------------------------------------------------------
def _cached_collate(batch):
    feats, targets, pasts = zip(*batch)
    return th.stack(feats), th.stack(targets), th.stack(pasts)


def train_cached_head(
    cache_dir: str,
    cache_tag: str,
    data_root: str,
    out_weights: str,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 2,
    past_action_k: int = DEFAULT_PAST_ACTION_K,
    chunk_size: int = 1,
    hidden_dim: int | None = None,
):
    """Train an MLP head on precomputed backbone features.

    The cache must have been produced by `feature_cache.precompute(...)` with
    the matching task_filter and use_language settings. Past-action / chunk
    knobs are head-only and don't need cache regeneration.

    Returns (head, test_subset, history).
    """
    from feature_cache import CachedFeatureDataset, HeadOnlyAgent

    device = _resolve_device(device)
    _print_device_info(device)

    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        tag=cache_tag,
        data_root=data_root,
        past_action_k=past_action_k,
        chunk_size=chunk_size,
    )
    print(
        f"Cached dataset: {len(dataset):,} samples  "
        f"(tag={cache_tag}  past_action_k={past_action_k}  chunk_size={chunk_size})"
    )

    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=th.Generator().manual_seed(42),
    )

    out_path = Path(out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_cached_collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_cached_collate,
        persistent_workers=num_workers > 0,
    )

    past_action_dim = past_action_k * PAST_ACTION_DIM
    model = HeadOnlyAgent(
        feature_dim=dataset.feature_dim(),
        output_dim=NUM_OUTPUT_LOGITS,
        past_action_dim=past_action_dim,
        chunk_size=chunk_size,
        hidden_dim=hidden_dim,
    ).to(device)
    optim = th.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_bce": [], "train_cam_ce": [],
        "val_loss": [], "val_bce": [], "val_cam_ce": [],
        "val_binary_acc": [], "val_cam_bin_acc": [],
    }

    for ep in range(epochs):
        model.train()
        train_total = train_bce = train_cam = 0.0
        for feats, acts, pasts in train_loader:
            feats = feats.to(device)
            acts = acts.to(device)
            pasts = pasts.to(device)
            optim.zero_grad()
            logits = model(feats, pasts)
            loss, bce_v, cam_v = vla_loss(logits, acts)
            loss.backward()
            optim.step()
            train_total += loss.item()
            train_bce += bce_v
            train_cam += cam_v
        n_train = len(train_loader)
        history["train_loss"].append(train_total / n_train)
        history["train_bce"].append(train_bce / n_train)
        history["train_cam_ce"].append(train_cam / n_train)

        model.eval()
        val_total = val_bce = val_cam = 0.0
        binary_correct = binary_seen = 0
        cam_correct = cam_seen = 0
        with th.no_grad():
            for feats, acts, pasts in val_loader:
                feats = feats.to(device)
                acts = acts.to(device)
                pasts = pasts.to(device)
                logits = model(feats, pasts)
                loss, bce_v, cam_v = vla_loss(logits, acts)
                val_total += loss.item()
                val_bce += bce_v
                val_cam += cam_v
                logits_flat = logits.reshape(-1, logits.size(-1))
                acts_flat = acts.reshape(-1, acts.size(-1))
                preds = (th.sigmoid(logits_flat[:, :NUM_BINARY]) > 0.5).float()
                binary_correct += (preds == acts_flat[:, :NUM_BINARY]).sum().item()
                binary_seen += preds.numel()
                cam_x_pred = logits_flat[:, _CAM_X_SLICE].argmax(dim=-1)
                cam_y_pred = logits_flat[:, _CAM_Y_SLICE].argmax(dim=-1)
                cam_correct += (cam_x_pred == acts_flat[:, NUM_BINARY].long()).sum().item()
                cam_correct += (cam_y_pred == acts_flat[:, NUM_BINARY + 1].long()).sum().item()
                cam_seen += 2 * cam_x_pred.numel()
        n_val = max(len(val_loader), 1)
        history["val_loss"].append(val_total / n_val)
        history["val_bce"].append(val_bce / n_val)
        history["val_cam_ce"].append(val_cam / n_val)
        history["val_binary_acc"].append(binary_correct / max(binary_seen, 1))
        history["val_cam_bin_acc"].append(cam_correct / max(cam_seen, 1))

        print(
            f"Epoch {ep + 1}/{epochs}  "
            f"train_loss={history['train_loss'][-1]:.4f}  "
            f"val_loss={history['val_loss'][-1]:.4f}  "
            f"bin_acc={history['val_binary_acc'][-1]:.4f}  "
            f"cam_acc={history['val_cam_bin_acc'][-1]:.4f}"
        )

    th.save(
        {
            "cache_tag": cache_tag,
            "state_dict": model.state_dict(),
            "training_metrics": history,
            "config": {
                "feature_dim": dataset.feature_dim(),
                "num_actions": NUM_ACTIONS,
                "num_binary": NUM_BINARY,
                "num_camera": NUM_CAMERA,
                "num_camera_bins": NUM_CAMERA_BINS,
                "num_output_logits": NUM_OUTPUT_LOGITS,
                "past_action_k": past_action_k,
                "past_action_dim": past_action_dim,
                "chunk_size": chunk_size,
                "hidden_dim": model.hidden_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_split": val_split,
                "test_split": test_split,
                "camera_quantizer": {
                    "camera_maxval": DEFAULT_CAMERA_QUANTIZER.camera_maxval,
                    "camera_binsize": DEFAULT_CAMERA_QUANTIZER.camera_binsize,
                    "mu": DEFAULT_CAMERA_QUANTIZER.mu,
                },
            },
        },
        out_weights,
    )
    print(f"Cached-head training complete. Saved to: {out_weights}")
    return model, test_set, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(
    model: VLAAgent,
    test_set,
    device: str,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 2,
) -> dict:
    """Run the model on test_set and write per-action metrics to output_dir/metrics.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )
    model.eval()
    binary_preds: list[th.Tensor] = []
    binary_targets: list[th.Tensor] = []
    cam_x_pred_bins: list[th.Tensor] = []
    cam_y_pred_bins: list[th.Tensor] = []
    cam_x_target_bins: list[th.Tensor] = []
    cam_y_target_bins: list[th.Tensor] = []

    with th.no_grad():
        for imgs, texts, acts, pasts in loader:
            acts = acts.to(device)
            pasts = pasts.to(device)
            logits = model(imgs, texts, pasts)
            # Flatten the chunk axis: every chunk step contributes equally to
            # the per-action metrics.
            logits_flat = logits.reshape(-1, logits.size(-1))
            acts_flat = acts.reshape(-1, acts.size(-1))
            binary_preds.append((th.sigmoid(logits_flat[:, :NUM_BINARY]) > 0.5).float().cpu())
            binary_targets.append(acts_flat[:, :NUM_BINARY].cpu())
            cam_x_pred_bins.append(logits_flat[:, _CAM_X_SLICE].argmax(dim=-1).cpu())
            cam_y_pred_bins.append(logits_flat[:, _CAM_Y_SLICE].argmax(dim=-1).cpu())
            cam_x_target_bins.append(acts_flat[:, NUM_BINARY].long().cpu())
            cam_y_target_bins.append(acts_flat[:, NUM_BINARY + 1].long().cpu())

    binary_preds_t = th.cat(binary_preds).numpy()
    binary_targets_t = th.cat(binary_targets).numpy()
    cam_x_pred = th.cat(cam_x_pred_bins).numpy()
    cam_y_pred = th.cat(cam_y_pred_bins).numpy()
    cam_x_target = th.cat(cam_x_target_bins).numpy()
    cam_y_target = th.cat(cam_y_target_bins).numpy()

    binary_accuracy = float((binary_preds_t == binary_targets_t).mean())
    cam_x_acc = float((cam_x_pred == cam_x_target).mean())
    cam_y_acc = float((cam_y_pred == cam_y_target).mean())

    cam_x_pred_deg = DEFAULT_CAMERA_QUANTIZER.undiscretize(cam_x_pred)
    cam_y_pred_deg = DEFAULT_CAMERA_QUANTIZER.undiscretize(cam_y_pred)
    cam_x_target_deg = DEFAULT_CAMERA_QUANTIZER.undiscretize(cam_x_target)
    cam_y_target_deg = DEFAULT_CAMERA_QUANTIZER.undiscretize(cam_y_target)
    camera_mae_degrees = float(
        0.5 * (
            np.abs(cam_x_pred_deg - cam_x_target_deg).mean()
            + np.abs(cam_y_pred_deg - cam_y_target_deg).mean()
        )
    )

    per_action: dict[str, dict] = {}
    for i, key in enumerate(BINARY_ACTION_KEYS):
        p, r, f1, _ = precision_recall_fscore_support(
            binary_targets_t[:, i], binary_preds_t[:, i], average="binary", zero_division=0
        )
        per_action[key] = {"precision": float(p), "recall": float(r), "f1": float(f1)}

    metrics = {
        "test_samples": int(binary_preds_t.shape[0]),
        "binary_accuracy": binary_accuracy,
        "camera_x_bin_accuracy": cam_x_acc,
        "camera_y_bin_accuracy": cam_y_acc,
        "camera_mae_degrees": camera_mae_degrees,
        "per_action_metrics": per_action,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    print(f"  Binary accuracy: {binary_accuracy:.4f}")
    print(f"  Camera bin accuracy:  x={cam_x_acc:.4f}  y={cam_y_acc:.4f}")
    print(f"  Camera MAE (degrees): {camera_mae_degrees:.4f}")
    print("\nPer-action F1:")
    for key, m in sorted(per_action.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {key:<12s} {m['f1']:.3f}")
    return metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--out-weights", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--past-action-k",
        type=int,
        default=DEFAULT_PAST_ACTION_K,
        help="Number of past actions concatenated to the head input (0 = off)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of future actions the head predicts per forward (1 = off)",
    )
    p.add_argument(
        "--no-language",
        action="store_true",
        help="Zero the text prompt — for the language-pathway ablation cell",
    )
    p.add_argument(
        "--evaluate-after",
        action="store_true",
        help="Run evaluate() on the held-out test split after training",
    )
    a = p.parse_args()

    model, test_set, _ = train_vla(
        data_root=a.data_dir,
        backbone=a.llava_model,
        out_weights=a.out_weights,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        device=a.device,
        val_split=a.val_split,
        test_split=a.test_split,
        num_workers=a.num_workers,
        past_action_k=a.past_action_k,
        chunk_size=a.chunk_size,
        use_language=not a.no_language,
    )
    if a.evaluate_after and len(test_set) > 0:
        evaluate(
            model,
            test_set,
            device=_resolve_device(a.device),
            output_dir=Path(a.out_weights).parent,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
        )
