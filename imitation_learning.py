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
        per_dir_counts: dict[str, int] = {}

        traj_dirs = [
            item for item in sorted(root.iterdir())
            if item.is_dir() and item.name.startswith("trajectory_task_")
        ]
        if not traj_dirs:
            raise RuntimeError(
                f"TrajectoryDataset: no trajectory_task_* dirs under {root}"
            )

        for item in traj_dirs:
            before = len(self.samples)
            all_actions_path = item / "all_actions.json"
            if all_actions_path.exists():
                self._load_consolidated(item, all_actions_path, item / "all_infos.json")
            else:
                self._load_individual_files(item)
            per_dir_counts[item.name] = len(self.samples) - before

        # Fail loud on partial loads — a silent miss here previously trained
        # a "combined" cell on chop-only data.
        print("[TrajectoryDataset] per-task-dir sample counts:")
        for name, n in per_dir_counts.items():
            print(f"  {name}: {n:,}")
        empty_dirs = [name for name, n in per_dir_counts.items() if n == 0]
        if empty_dirs:
            raise RuntimeError(
                f"TrajectoryDataset: trajectory dirs contributed 0 samples: "
                f"{empty_dirs} — likely missing all_actions.json / actions/ / videos/"
            )

    def _add_samples(
        self, stem: str, actions: list, task_text: str, videos_dir: Path
    ) -> None:
        # Missing MP4 = partial dataset. Silent skip previously masked an
        # incomplete extraction; fail loud instead.
        mp4_file = videos_dir / f"video_{stem}.mp4"
        if not mp4_file.exists():
            raise RuntimeError(
                f"missing video file {mp4_file} for stem={stem!r} — "
                "refusing to train on a partially extracted trajectory dir"
            )
        # range(last_valid + 1) is empty when chunk_size > len(actions), which
        # silently drops the stem. For the BASALT 3000-frame trajectories with
        # chunk_size <= 8 this is purely defensive.
        if self.chunk_size > len(actions):
            raise RuntimeError(
                f"chunk_size={self.chunk_size} exceeds trajectory length "
                f"{len(actions)} for stem={stem!r}; would silently drop it"
            )

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


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
_CAM_X_SLICE = slice(NUM_BINARY, NUM_BINARY + NUM_CAMERA_BINS)
_CAM_Y_SLICE = slice(NUM_BINARY + NUM_CAMERA_BINS, NUM_BINARY + 2 * NUM_CAMERA_BINS)


def vla_loss(
    logits: th.Tensor,
    targets: th.Tensor,
    pos_weight: th.Tensor | None = None,
    cam_weight: th.Tensor | None = None,
    focal_gamma: float = 0.0,
    cam_ce_weight: float = 0.5,
    cam_sample_weight: th.Tensor | None = None,
) -> tuple[th.Tensor, float, float]:
    """BCE on binary actions + cross-entropy on each camera axis, averaged over
    the chunk axis when present.

    Args:
        logits:  (B, NUM_OUTPUT_LOGITS) or (B, N, NUM_OUTPUT_LOGITS)
        targets: (B, NUM_BINARY + NUM_CAMERA) or (B, N, NUM_BINARY + NUM_CAMERA)
                 last 2 entries are bin indices (stored as float32 for collate
                 uniformity; cast to long here).
        pos_weight: optional (NUM_BINARY,) BCE positive-class weights. Upweights
                 rare positives (e.g. movement keys) so the head learns to assert
                 them instead of collapsing to the ~0 base rate. See
                 `compute_class_weights`.
        cam_weight: optional (NUM_CAMERA_BINS,) cross-entropy class weights shared
                 by both camera axes. Downweights the dominant 0° bin so the head
                 doesn't collapse onto the majority class.
        focal_gamma: if > 0, replace plain BCE with focal-weighted BCE on the
                 binary head: FL = (1 - p_t)^gamma * BCE_elementwise. gamma=2.0
                 is typical. Heavily upweights hard examples (low p_t for
                 positive targets, high p_t for negatives) so the head doesn't
                 collapse onto easy-to-predict actions (e.g. attack at 96% F1)
                 while leaving rare movement keys at 15-60% F1. Composes with
                 pos_weight when both are set, though focal alone is usually
                 enough.
        cam_sample_weight: optional (B,) or (B, N) per-frame weight on the
                 camera CE only. A weighted mean (Σw·ce / Σw) replaces the
                 plain mean, so the camera head focuses on the high-weight
                 frames without changing the overall bce/cam scale. Used to
                 upweight the pre-attack-onset aiming windows (see
                 `compute_camera_onset_weights` + the 2026-06-13 demo
                 analysis). Defaults to a uniform mean (legacy).

    Returns (total_loss, bce_value, camera_ce_value).
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    if targets.dim() == 3:
        targets = targets.reshape(-1, targets.size(-1))
    if cam_sample_weight is not None:
        cam_sample_weight = cam_sample_weight.reshape(-1).to(
            logits.device, logits.dtype
        )

    binary_logits = logits[:, :NUM_BINARY]
    binary_targets = targets[:, :NUM_BINARY]

    cam_x_logits = logits[:, _CAM_X_SLICE]
    cam_y_logits = logits[:, _CAM_Y_SLICE]
    cam_x_targets = targets[:, NUM_BINARY].long()
    cam_y_targets = targets[:, NUM_BINARY + 1].long()

    if pos_weight is not None:
        pos_weight = pos_weight.to(binary_logits.device, binary_logits.dtype)
    if cam_weight is not None:
        cam_weight = cam_weight.to(cam_x_logits.device, cam_x_logits.dtype)

    if focal_gamma > 0:
        bce_elem = nn.functional.binary_cross_entropy_with_logits(
            binary_logits, binary_targets, pos_weight=pos_weight, reduction="none"
        )
        p = th.sigmoid(binary_logits)
        p_t = p * binary_targets + (1.0 - p) * (1.0 - binary_targets)
        focal_w = (1.0 - p_t).clamp(min=1e-8).pow(focal_gamma)
        bce = (focal_w * bce_elem).mean()
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(
            binary_logits, binary_targets, pos_weight=pos_weight
        )
    if cam_sample_weight is not None:
        ce_x_elem = nn.functional.cross_entropy(
            cam_x_logits, cam_x_targets, weight=cam_weight, reduction="none"
        )
        ce_y_elem = nn.functional.cross_entropy(
            cam_y_logits, cam_y_targets, weight=cam_weight, reduction="none"
        )
        wsum = cam_sample_weight.sum().clamp_min(1e-8)
        ce_x = (cam_sample_weight * ce_x_elem).sum() / wsum
        ce_y = (cam_sample_weight * ce_y_elem).sum() / wsum
    else:
        ce_x = nn.functional.cross_entropy(cam_x_logits, cam_x_targets, weight=cam_weight)
        ce_y = nn.functional.cross_entropy(cam_y_logits, cam_y_targets, weight=cam_weight)
    cam_ce = cam_ce_weight * (ce_x + ce_y)

    return bce + cam_ce, bce.item(), cam_ce.item()


def compute_class_weights(
    targets_by_stem: dict,
    pos_weight_clip: tuple[float, float] = (0.5, 4.0),
    cam_weight_clip: tuple[float, float] = (0.3, 5.0),
) -> tuple[th.Tensor, th.Tensor]:
    """Derive BCE pos_weight + camera CE class-weights from the demo distribution.

    The contractor demos are heavily imbalanced (attack ~83% on, movement <16%,
    camera ~90% at the 0° bin), and unweighted BCE/CE collapse the heads onto the
    marginal. This computes a *sqrt-compressed* inverse-frequency reweighting:

      * pos_weight[i] = sqrt(#neg / #pos) for binary action i, clipped.
      * cam_weight[b] = sqrt(mean_count / count[b]) over the combined x+y bin
        histogram, clipped.

    The sqrt + tight clip is deliberate. Full #neg/#pos (clipped 0.05–50) was
    tried first and over-corrected catastrophically in rollout: rare actions
    (hotbar, drop) saturated to sigmoid ~0.93 from the 50× upweight while the
    *most common, most essential* action — attack (the one that breaks blocks) —
    was suppressed to ~0.05 by the 0.05 lower clip. The agent fired every key at
    once (forward+back+left+right cancel) and never attacked. sqrt keeps the
    rebalancing direction but tames the magnitude; the (0.5, 4.0) clip caps
    residual extremes so no action is forced on or off by the weight alone.

    Returns (pos_weight (NUM_BINARY,), cam_weight (NUM_CAMERA_BINS,)).
    """
    bin_pos = np.zeros(NUM_BINARY, dtype=np.float64)
    n_frames = 0
    cam_counts = np.zeros(NUM_CAMERA_BINS, dtype=np.float64)
    for arr in targets_by_stem.values():  # (T, NUM_BINARY + NUM_CAMERA)
        bin_pos += arr[:, :NUM_BINARY].sum(axis=0)
        n_frames += arr.shape[0]
        for axis in (NUM_BINARY, NUM_BINARY + 1):
            idx = arr[:, axis].astype(np.int64)
            cam_counts += np.bincount(idx, minlength=NUM_CAMERA_BINS)[:NUM_CAMERA_BINS]

    bin_pos = np.clip(bin_pos, 1.0, None)  # avoid div-by-zero for never-on actions
    pos_weight = np.sqrt((n_frames - bin_pos) / bin_pos)
    pos_weight = np.clip(pos_weight, *pos_weight_clip)

    cam_counts = np.clip(cam_counts, 1.0, None)
    cam_weight = np.sqrt(cam_counts.mean() / cam_counts)
    cam_weight = np.clip(cam_weight, *cam_weight_clip)

    return (
        th.tensor(pos_weight, dtype=th.float32),
        th.tensor(cam_weight, dtype=th.float32),
    )


def compute_task_active_weights(
    data_root: str | Path,
    task_filter: str | None = None,
    min_run_length: int = 60,
    multiplier: float = 5.0,
) -> dict[str, np.ndarray]:
    """For each demo frame, return weight = `multiplier` if it sits inside a
    sustained-attack run of length ≥ `min_run_length`, else 1.0.

    Rationale: chop/dirt tasks reward the *closure* (sustained attack →
    block-breaks → walk-onto-drop) but those frames are rare in the demos
    (most frames are "walk around, look at terrain"). Without upweighting,
    the BC head can minimize loss without learning the closure pattern —
    which is exactly the failure we see at rollout (model attacks 70 %,
    walks forward, but never completes a chop chain). Reweighting the
    sampler so closure-context frames are sampled 5× as often gives the
    head a real gradient signal on those rare transitions.

    Per-frame inventory is NOT in `all_infos.json` (only the trajectory-level
    final inventory under `results.inventory_dis_values.log`); the inventory-
    delta proxy from the original handoff is therefore not available. The
    sustained-attack-run proxy is close enough in practice — every successful
    chop in the demos is the tail of a long attack run.

    Returns: `{stem: weights_per_frame (n,) float32}`.
    """
    root = Path(data_root)
    out: dict[str, np.ndarray] = {}
    for traj_dir in _iter_trajectory_dirs(root, task_filter):
        actions_path = traj_dir / "all_actions.json"
        if not actions_path.exists():
            continue
        with actions_path.open("r", encoding="utf-8") as fp:
            all_actions = json.load(fp)
        for stem, actions in all_actions.items():
            n = len(actions)
            attack = np.zeros(n, dtype=bool)
            for i, a in enumerate(actions):
                v = a.get("attack", 0)
                if isinstance(v, list):
                    v = v[0]
                attack[i] = int(v) == 1
            weights = np.ones(n, dtype=np.float32)
            i = 0
            while i < n:
                if attack[i]:
                    j = i
                    while j < n and attack[j]:
                        j += 1
                    if j - i >= min_run_length:
                        weights[i:j] = multiplier
                    i = j
                else:
                    i += 1
            out[stem] = weights
    if not out:
        raise RuntimeError(
            f"compute_task_active_weights: no trajectories under {root} "
            f"(task_filter={task_filter!r}) — check the data path"
        )
    return out


def compute_camera_onset_weights(
    data_root: str | Path,
    task_filter: str | None = None,
    pre_window: int = 8,
    min_run_length: int = 30,
    multiplier: float = 5.0,
) -> dict[str, np.ndarray]:
    """Per-frame CAMERA-loss weight = `multiplier` inside the aiming window
    just before a sustained-attack-run onset, else 1.0.

    Motivated by the 2026-06-13 demo analysis (scripts/analyze_chop_aiming.py):
    the demonstrator aims — camera bursts (pitch down toward the trunk base)
    in the ~6 frames before committing to a sustained chop, then settles. That
    signal is real and directional but RARE/TRANSIENT, so the per-frame camera
    CE — dominated by the 78 % still-camera majority — averages it out. This
    flags `[onset - pre_window, onset)` of every attack run ≥ `min_run_length`
    so `vla_loss` can upweight the camera CE there specifically (unlike
    `compute_task_active_weights`, which feeds the SAMPLER and reweights binary
    losses too, and unlike `cam_weighted_loss`, which reweights by global bin
    frequency rather than these transient moments).

    Returns: `{stem: weights_per_frame (n,) float32}`.
    """
    root = Path(data_root)
    out: dict[str, np.ndarray] = {}
    for traj_dir in _iter_trajectory_dirs(root, task_filter):
        actions_path = traj_dir / "all_actions.json"
        if not actions_path.exists():
            continue
        with actions_path.open("r", encoding="utf-8") as fp:
            all_actions = json.load(fp)
        for stem, actions in all_actions.items():
            n = len(actions)
            attack = np.zeros(n, dtype=bool)
            for i, a in enumerate(actions):
                v = a.get("attack", 0)
                if isinstance(v, list):
                    v = v[0]
                attack[i] = int(v) == 1
            weights = np.ones(n, dtype=np.float32)
            i = 0
            while i < n:
                if attack[i] and (i == 0 or not attack[i - 1]):
                    j = i
                    while j < n and attack[j]:
                        j += 1
                    if j - i >= min_run_length:
                        weights[max(0, i - pre_window) : i] = multiplier
                    i = j
                else:
                    i += 1
            out[stem] = weights
    if not out:
        raise RuntimeError(
            f"compute_camera_onset_weights: no trajectories under {root} "
            f"(task_filter={task_filter!r}) — check the data path"
        )
    return out


def _iter_trajectory_dirs(root: Path, task_filter: str | None):
    """Inline copy of feature_cache._iter_trajectory_dirs to avoid a cyclic
    import (imitation_learning is imported by cluster_pipeline before
    feature_cache.precompute runs in the same process).
    """
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("trajectory_task_"):
            continue
        if task_filter is None or task_filter in d.name:
            yield d


def apply_history_dropout(pasts: th.Tensor, p: float) -> th.Tensor:
    """Randomly zero the entire past-action vector for a fraction `p` of samples.

    Movement/camera are highly autocorrelated in the demos, so a model with the
    past-action feature learns ``P(action | recent actions)`` and at rollout —
    where the buffer is dominated by "no movement" — suppresses movement below
    even its base rate (a self-reinforcing no-move attractor). Zeroing the whole
    history for a fraction of training samples (mimicking the start-of-trajectory
    zero-padding the model already sees) forces it to read the *frame* instead.
    No-op when p<=0 or there are no past-action columns.
    """
    if p <= 0.0 or pasts.size(-1) == 0:
        return pasts
    keep = (th.rand(pasts.size(0), 1, device=pasts.device) >= p).to(pasts.dtype)
    return pasts * keep


def apply_past_action_slot_dropout(
    pasts: th.Tensor, p: float, past_action_k: int
) -> th.Tensor:
    """Per-slot Bernoulli dropout: with prob `p`, zero each of the K past-action
    slots INDEPENDENTLY. Finer-grained than `apply_history_dropout` (which zeros
    the whole vector or nothing).

    Why: at rollout the past_action buffer is the head's *own* recent predictions,
    which drift from the demo distribution. history_dropout (whole-vector) is a
    blunt instrument — when applied, it removes ALL temporal context; when not
    applied, the head is trusted blindly. Per-slot dropout teaches robustness to
    any subset of slots being corrupted — closer to the noisy-buffer scenario
    that actually happens at inference. No-op when p<=0 or past_action_k==0.

    Args:
        pasts: (B, K * PAST_ACTION_DIM)
        p: per-slot dropout probability
        past_action_k: K (number of slots)
    """
    if p <= 0.0 or pasts.size(-1) == 0 or past_action_k <= 0:
        return pasts
    B = pasts.size(0)
    D = pasts.size(-1) // past_action_k
    pasts_3d = pasts.view(B, past_action_k, D)
    keep = (th.rand(B, past_action_k, 1, device=pasts.device) >= p).to(pasts.dtype)
    return (pasts_3d * keep).view(B, past_action_k * D)


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


def _atomic_save(obj, path: str | Path) -> None:
    """torch.save via tmp + rename so a crash mid-save can't corrupt the file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    th.save(obj, tmp)
    tmp.replace(path)


def _empty_history() -> dict:
    return {
        "train_loss": [], "train_bce": [], "train_cam_ce": [],
        "val_loss": [], "val_bce": [], "val_cam_ce": [],
        "val_binary_acc": [], "val_cam_bin_acc": [],
    }


def _try_resume_training(
    out_weights: str,
    model: nn.Module,
    optim: th.optim.Optimizer,
    device: str,
    restart: bool,
) -> tuple[int, dict]:
    """Look for an existing checkpoint at `out_weights` and load model+optim
    state if found. Returns (start_epoch, history). On `restart=True` or
    missing checkpoint, returns (0, fresh_history).
    """
    out_path = Path(out_weights)
    if restart or not out_path.exists():
        return 0, _empty_history()
    ckpt = th.load(out_path, map_location=device)
    if "epoch" not in ckpt or "optimizer_state" not in ckpt:
        # Legacy / final-only checkpoint — no resume possible.
        print(f"[resume] {out_path} exists but has no epoch/optimizer_state — starting fresh")
        return 0, _empty_history()
    try:
        # Heads come in two shapes: HeadOnlyAgent (top-level state_dict) and
        # VLAAgent (only action_head saved). The caller passes the right model.
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError:
        # VLAAgent path — checkpoint stored only the action_head weights.
        model.action_head.load_state_dict(ckpt["state_dict"])
    optim.load_state_dict(ckpt["optimizer_state"])
    history = ckpt.get("training_metrics", _empty_history())
    start_epoch = int(ckpt["epoch"]) + 1
    val_suffix = (
        f" (prev val_loss={history['val_loss'][-1]:.4f})" if history.get("val_loss") else ""
    )
    print(f"[resume] {out_path}: continuing from epoch {start_epoch}{val_suffix}")
    return start_epoch, history


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
    restart: bool = False,
    weighted_loss: bool = False,
    history_dropout: float = 0.0,
):
    """Train the VLA action head and return (model, test_subset, history).

    `past_action_k` and `chunk_size` are the two temporal-context levers (see
    CLAUDE.md). `use_language=False` zeros the text prompt (ablation channel).
    Checkpoints are written atomically after each epoch; if `out_weights`
    already exists with epoch/optimizer state, training resumes from the next
    epoch unless `restart=True`.
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

    pos_weight = cam_weight = None
    if weighted_loss:
        pos_weight, cam_weight = compute_class_weights(dataset.targets_by_stem)
        pos_weight = pos_weight.to(device)
        cam_weight = cam_weight.to(device)
        print(
            f"Weighted loss ON — pos_weight[forward]={pos_weight[2]:.1f} "
            f"pos_weight[attack]={pos_weight[0]:.2f}  cam_weight[0deg]={cam_weight[5]:.2f}"
        )
    if history_dropout > 0:
        print(f"History dropout ON — p={history_dropout}")

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

    start_epoch, history = _try_resume_training(out_weights, model, optim, device, restart)

    def _save_checkpoint(epoch_just_completed: int) -> None:
        _atomic_save(
            {
                "llava_model": backbone,
                "state_dict": model.action_head.state_dict(),
                "optimizer_state": optim.state_dict(),
                "epoch": epoch_just_completed,
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
                    "weighted_loss": weighted_loss,
                    "history_dropout": history_dropout,
                    "camera_quantizer": {
                        "camera_maxval": DEFAULT_CAMERA_QUANTIZER.camera_maxval,
                        "camera_binsize": DEFAULT_CAMERA_QUANTIZER.camera_binsize,
                        "mu": DEFAULT_CAMERA_QUANTIZER.mu,
                    },
                },
            },
            out_weights,
        )

    def _flat_step_logits(logits_3d: th.Tensor) -> th.Tensor:
        """(B, N, NUM_OUTPUT_LOGITS) -> (B*N, NUM_OUTPUT_LOGITS) for metric slicing."""
        return logits_3d.reshape(-1, logits_3d.size(-1))

    def _flat_step_targets(targets_3d: th.Tensor) -> th.Tensor:
        return targets_3d.reshape(-1, targets_3d.size(-1))

    if start_epoch >= epochs:
        print(f"Already trained for {start_epoch} epochs >= requested {epochs}; nothing to do.")
        return model, test_set, history

    for ep in range(start_epoch, epochs):
        model.train()
        train_total = train_bce = train_cam = 0.0
        for idx, (imgs, texts, acts, pasts) in enumerate(train_loader):
            print(f"Training batch {idx + 1}/{len(train_loader)}", end="\r")
            acts = acts.to(device)
            pasts = apply_history_dropout(pasts.to(device), history_dropout)
            optim.zero_grad()
            logits = model(imgs, texts, pasts)
            loss, bce_v, cam_v = vla_loss(logits, acts, pos_weight, cam_weight)
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
                loss, bce_v, cam_v = vla_loss(logits, acts, pos_weight, cam_weight)
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
        _save_checkpoint(ep)

    print(f"\nTraining complete. Model saved to: {out_weights}")
    return model, test_set, history


# ---------------------------------------------------------------------
# Cached-feature training (used after `feature_cache.precompute`)
# ---------------------------------------------------------------------
def split_indices_by_stem(
    samples: list[tuple[str, int]],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Group-aware split: every sample of a trajectory stem lands in exactly
    one of (train, val, test).

    The frame-level `random_split` this replaces leaked temporally adjacent,
    near-identical frames of the same video into train AND val, so val
    metrics were optimistic and noisy — bad for exactly the best-epoch
    selection they're used for. Stems are shuffled deterministically
    (`seed`), then assigned to test until its sample budget is met, then to
    val, remainder to train, so the realized fractions track the requested
    ones as closely as whole-trajectory granularity allows.

    Returns (train_indices, val_indices, test_indices) into `samples`.
    """
    counts: dict[str, int] = {}
    for stem, _ in samples:
        counts[stem] = counts.get(stem, 0) + 1
    stems = sorted(counts)
    rng = np.random.RandomState(seed)
    rng.shuffle(stems)

    n_total = len(samples)
    test_target = n_total * test_split
    val_target = n_total * val_split
    test_stems: set[str] = set()
    val_stems: set[str] = set()
    test_seen = val_seen = 0
    for stem in stems:
        if test_seen < test_target:
            test_stems.add(stem)
            test_seen += counts[stem]
        elif val_seen < val_target:
            val_stems.add(stem)
            val_seen += counts[stem]

    train_idx, val_idx, test_idx = [], [], []
    for i, (stem, _) in enumerate(samples):
        if stem in test_stems:
            test_idx.append(i)
        elif stem in val_stems:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def _cached_collate(batch):
    feats, targets, pasts, cam_w = zip(*batch)
    return th.stack(feats), th.stack(targets), th.stack(pasts), th.stack(cam_w)


def train_cached_head(
    cache_dir: str,
    cache_tag: str,
    data_root: str,
    out_weights: str,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    lr_schedule: str = "constant",
    device: str = "cuda" if th.cuda.is_available() else "cpu",
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 2,
    past_action_k: int = DEFAULT_PAST_ACTION_K,
    chunk_size: int = 1,
    hidden_dim: int | None = None,
    restart: bool = False,
    weighted_loss: bool = False,
    history_dropout: float = 0.0,
    frame_weight_multiplier: float = 1.0,
    frame_weight_min_run: int = 60,
    learnable_bce_temp: bool = False,
    focal_gamma: float = 0.0,
    past_action_slot_dropout: float = 0.0,
    chop_oversample_weight: float = 1.0,
    cam_weighted_loss: bool = False,
    cam_ce_weight: float = 0.5,
    split_by_trajectory: bool = True,
    keep_best: bool = False,
    frame_history_k: int = 0,
    stem_filter: str | None = None,
    camera_onset_weight: float = 1.0,
    camera_onset_window: int = 8,
):
    """Train an MLP head on precomputed backbone features.

    The cache must have been produced by `feature_cache.precompute(...)` with
    the matching task_filter and use_language settings. Past-action / chunk
    knobs are head-only and don't need cache regeneration.

    Checkpoints are written atomically after each epoch; if `out_weights`
    already exists with epoch/optimizer state, training resumes from the next
    epoch unless `restart=True`.

    Returns (head, test_subset, history).
    """
    from feature_cache import CachedFeatureDataset, HeadOnlyAgent

    device = _resolve_device(device)
    _print_device_info(device)

    use_camera_onset = camera_onset_weight > 1.0
    camera_onset_weights = None
    if use_camera_onset:
        camera_onset_weights = compute_camera_onset_weights(
            data_root,
            task_filter=None,  # whatever stems landed in the cache
            pre_window=camera_onset_window,
            multiplier=camera_onset_weight,
        )
        n_flagged = sum(int((w > 1.0).sum()) for w in camera_onset_weights.values())
        n_total = sum(int(w.size) for w in camera_onset_weights.values())
        print(
            f"Camera-onset CE weighting ON — multiplier={camera_onset_weight} "
            f"window={camera_onset_window}; {n_flagged:,}/{n_total:,} frames "
            f"flagged ({100 * n_flagged / max(n_total, 1):.1f}%)"
        )

    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        tag=cache_tag,
        data_root=data_root,
        past_action_k=past_action_k,
        chunk_size=chunk_size,
        frame_history_k=frame_history_k,
        stem_filter=stem_filter,
        camera_onset_weights=camera_onset_weights,
    )
    print(
        f"Cached dataset: {len(dataset):,} samples  "
        f"(tag={cache_tag}  past_action_k={past_action_k}  chunk_size={chunk_size}"
        f"  frame_history_k={frame_history_k}  stem_filter={stem_filter!r})"
    )

    pos_weight = cam_weight = None
    if weighted_loss or cam_weighted_loss:
        # cam_weighted_loss: keep only the camera CE class weights, drop the
        # binary BCE pos_weight. Tests whether the camera fix in isolation
        # gives the F1 gains without disrupting the binary policy.
        full_pos, full_cam = compute_class_weights(dataset.targets_by_stem)
        if weighted_loss:
            pos_weight = full_pos.to(device)
        if weighted_loss or cam_weighted_loss:
            cam_weight = full_cam.to(device)
        bce_str = (
            f"pos_weight[forward]={pos_weight[2]:.1f} "
            f"pos_weight[attack]={pos_weight[0]:.2f}  "
            if pos_weight is not None
            else "pos_weight=OFF  "
        )
        print(
            f"Weighted loss ON — {bce_str}"
            f"cam_weight[0deg]={cam_weight[5]:.2f} cam_weight[max]={cam_weight.max():.1f}"
        )
    if history_dropout > 0:
        print(f"History dropout ON — p={history_dropout}")

    if split_by_trajectory:
        train_idx, val_idx, test_idx = split_indices_by_stem(
            dataset.samples, val_split, test_split, seed=42
        )
        train_set = th.utils.data.Subset(dataset, train_idx)
        val_set = th.utils.data.Subset(dataset, val_idx)
        test_set = th.utils.data.Subset(dataset, test_idx)
        n_stems = len({s for s, _ in dataset.samples})
        print(
            f"Trajectory-level split — {n_stems} stems; "
            f"train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}"
        )
    else:
        # Legacy frame-level split: leaks adjacent frames train<->val (val
        # metrics optimistic + noisy); kept only for comparability with
        # pre-2026-06-10 checkpoints.
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

    # Frame-weighted and/or task-weighted sampling.
    sampler = None
    train_shuffle = True
    use_weighted_sampler = (
        frame_weight_multiplier > 1.0 or chop_oversample_weight != 1.0
    )
    if use_weighted_sampler:
        per_sample = np.ones(len(dataset), dtype=np.float64)
        # Frame-level weighting (sustained-attack runs).
        if frame_weight_multiplier > 1.0:
            stem_weights = compute_task_active_weights(
                data_root,
                task_filter=None,  # cache_tag's task_filter is encoded in the cache;
                # at this stage we want weights for whatever stems landed in the cache
                min_run_length=frame_weight_min_run,
                multiplier=frame_weight_multiplier,
            )
            unmatched = 0
            for idx, (stem, fidx) in enumerate(dataset.samples):
                w = stem_weights.get(stem)
                if w is None or fidx >= len(w):
                    unmatched += 1
                    continue
                per_sample[idx] = float(w[fidx])
        # Task-level weighting (chop vs dirt). Composes multiplicatively with
        # frame_weight when both are set. Stems are named "chop_a_tree_*" and
        # "collect_dirt_*"; everything not chop is treated as 1.0.
        if chop_oversample_weight != 1.0:
            n_chop = 0
            for idx, (stem, _) in enumerate(dataset.samples):
                if stem.startswith("chop_a_tree"):
                    per_sample[idx] *= chop_oversample_weight
                    n_chop += 1
            print(
                f"Chop oversample ON — weight={chop_oversample_weight}; "
                f"{n_chop:,}/{len(dataset):,} chop samples weighted"
            )
        train_weights = per_sample[list(train_set.indices)]
        sampler = th.utils.data.WeightedRandomSampler(
            weights=th.from_numpy(train_weights),
            num_samples=len(train_weights),
            replacement=True,
        )
        train_shuffle = False
        if frame_weight_multiplier > 1.0:
            active = int((per_sample > 1.0).sum())
            print(
                f"Frame-weighted sampling ON — multiplier={frame_weight_multiplier} "
                f"min_run={frame_weight_min_run}; active_frames={active:,}/{len(dataset):,} "
                f"({100 * active / len(dataset):.1f}%); unmatched_stems={unmatched}"
            )

    if learnable_bce_temp:
        print("Learnable per-action BCE temperature ON (init=1.0)")
    if focal_gamma > 0:
        print(f"Focal BCE ON — gamma={focal_gamma}")
    if past_action_slot_dropout > 0:
        print(f"Past-action per-slot dropout ON — p={past_action_slot_dropout}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
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
        learnable_bce_temp=learnable_bce_temp,
    ).to(device)
    optim = th.optim.Adam(model.parameters(), lr=lr)

    start_epoch, history = _try_resume_training(out_weights, model, optim, device, restart)

    # Optional cosine LR decay (lr -> 0 over `epochs`), stepped once per epoch.
    # Damps the late-epoch val-F1 jitter seen at constant LR. On resume, fast-
    # forward the schedule to the already-completed epoch count so the curve is
    # identical whether or not the run was interrupted.
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Cosine LR schedule ON — lr {lr:.1e} -> 0 over {epochs} epochs")
    elif lr_schedule != "constant":
        raise ValueError(f"unknown lr_schedule={lr_schedule!r} (expected 'constant' or 'cosine')")

    def _save_checkpoint(epoch_just_completed: int) -> None:
        _atomic_save(
            {
                "cache_tag": cache_tag,
                "state_dict": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "epoch": epoch_just_completed,
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
                    "weighted_loss": weighted_loss,
                    "history_dropout": history_dropout,
                    "frame_weight_multiplier": frame_weight_multiplier,
                    "frame_weight_min_run": frame_weight_min_run,
                    "learnable_bce_temp": learnable_bce_temp,
                    "focal_gamma": focal_gamma,
                    "past_action_slot_dropout": past_action_slot_dropout,
                    "chop_oversample_weight": chop_oversample_weight,
                    "cam_weighted_loss": cam_weighted_loss,
                    "cam_ce_weight": cam_ce_weight,
                    "split_by_trajectory": split_by_trajectory,
                    "keep_best": keep_best,
                    "frame_history_k": frame_history_k,
                    "stem_filter": stem_filter,
                    "lr_schedule": lr_schedule,
                    "camera_onset_weight": camera_onset_weight,
                    "camera_onset_window": camera_onset_window,
                    "camera_quantizer": {
                        "camera_maxval": DEFAULT_CAMERA_QUANTIZER.camera_maxval,
                        "camera_binsize": DEFAULT_CAMERA_QUANTIZER.camera_binsize,
                        "mu": DEFAULT_CAMERA_QUANTIZER.mu,
                    },
                },
            },
            out_weights,
        )

    # Keep-best: snapshot the epoch with the highest val movement-F1 to
    # model_best.pt next to the rolling last-epoch checkpoint. Motivated by
    # the ep10-vs-ep20 finding (docs/trials.md): more epochs helps some
    # recipes and collapses others, so "last epoch wins" leaves performance
    # on the table either way. Movement keys are where recipes actually
    # differ — attack F1 saturates at ~.96 for every recipe.
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)
    movement_idx = th.tensor(
        [BINARY_ACTION_KEYS.index(k)
         for k in ("back", "forward", "jump", "left", "right", "sprint")]
    )
    best_movement_f1 = max(history.get("val_movement_f1") or [float("-inf")])

    if start_epoch >= epochs:
        print(f"Already trained for {start_epoch} epochs >= requested {epochs}; nothing to do.")
        return model, test_set, history

    for ep in range(start_epoch, epochs):
        model.train()
        train_total = train_bce = train_cam = 0.0
        for feats, acts, pasts, cam_w in train_loader:
            feats = feats.to(device)
            acts = acts.to(device)
            pasts = pasts.to(device)
            cam_w = cam_w.to(device) if use_camera_onset else None
            pasts = apply_history_dropout(pasts, history_dropout)
            pasts = apply_past_action_slot_dropout(
                pasts, past_action_slot_dropout, past_action_k
            )
            optim.zero_grad()
            logits = model(feats, pasts)
            loss, bce_v, cam_v = vla_loss(
                logits, acts, pos_weight, cam_weight,
                focal_gamma=focal_gamma, cam_ce_weight=cam_ce_weight,
                cam_sample_weight=cam_w,
            )
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
        tp = th.zeros(NUM_BINARY, device=device)
        fp = th.zeros(NUM_BINARY, device=device)
        fn = th.zeros(NUM_BINARY, device=device)
        with th.no_grad():
            for feats, acts, pasts, _cam_w in val_loader:
                feats = feats.to(device)
                acts = acts.to(device)
                pasts = pasts.to(device)
                # Val loss stays UNWEIGHTED so val_loss / movement-F1 remain
                # comparable across cells for model selection.
                logits = model(feats, pasts)
                loss, bce_v, cam_v = vla_loss(
                    logits, acts, pos_weight, cam_weight,
                    focal_gamma=focal_gamma, cam_ce_weight=cam_ce_weight,
                )
                val_total += loss.item()
                val_bce += bce_v
                val_cam += cam_v
                logits_flat = logits.reshape(-1, logits.size(-1))
                acts_flat = acts.reshape(-1, acts.size(-1))
                preds = (th.sigmoid(logits_flat[:, :NUM_BINARY]) > 0.5).float()
                bin_targets = acts_flat[:, :NUM_BINARY]
                binary_correct += (preds == bin_targets).sum().item()
                binary_seen += preds.numel()
                tp += (preds * bin_targets).sum(dim=0)
                fp += (preds * (1.0 - bin_targets)).sum(dim=0)
                fn += ((1.0 - preds) * bin_targets).sum(dim=0)
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

        f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1.0)).cpu()
        movement_f1 = float(f1[movement_idx].mean())
        history.setdefault("val_per_action_f1", []).append(
            {BINARY_ACTION_KEYS[i]: round(float(f1[i]), 4) for i in range(NUM_BINARY)}
        )
        history.setdefault("val_movement_f1", []).append(movement_f1)

        print(
            f"Epoch {ep + 1}/{epochs}  "
            f"train_loss={history['train_loss'][-1]:.4f}  "
            f"val_loss={history['val_loss'][-1]:.4f}  "
            f"bin_acc={history['val_binary_acc'][-1]:.4f}  "
            f"cam_acc={history['val_cam_bin_acc'][-1]:.4f}  "
            f"move_f1={movement_f1:.4f}"
        )
        _save_checkpoint(ep)
        if keep_best and movement_f1 > best_movement_f1:
            best_movement_f1 = movement_f1
            ckpt = th.load(out_weights, map_location="cpu", weights_only=False)
            ckpt["best_epoch"] = ep
            ckpt["best_val_movement_f1"] = movement_f1
            _atomic_save(ckpt, best_path)
            print(f"  ↳ new best movement_f1={movement_f1:.4f} → {best_path}")
        if scheduler is not None:
            scheduler.step()

    print(f"Cached-head training complete. Saved to: {out_weights}")
    return model, test_set, history


def evaluate_cached(
    model,
    test_set,
    device: str,
    output_dir: Path,
    batch_size: int = 256,
    num_workers: int = 2,
) -> dict:
    """Run per-action metrics on a HeadOnlyAgent + CachedFeatureDataset split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_cached_collate,
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
        for feats, acts, pasts, _cam_w in loader:
            feats = feats.to(device)
            acts = acts.to(device)
            pasts = pasts.to(device)
            logits = model(feats, pasts)
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
    return metrics


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
        "--restart",
        action="store_true",
        help="Ignore any existing checkpoint at --out-weights and start from epoch 0",
    )
    p.add_argument(
        "--evaluate-after",
        action="store_true",
        help="Run evaluate() on the held-out test split after training",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Class-balance the loss: BCE pos_weight on rare binary actions + "
        "inverse-frequency camera-bin weights (fixes head collapse onto the marginal)",
    )
    p.add_argument(
        "--history-dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing the whole past-action vector per training "
        "sample (breaks the no-move feedback trap; e.g. 0.5)",
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
        restart=a.restart,
        weighted_loss=a.weighted_loss,
        history_dropout=a.history_dropout,
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
