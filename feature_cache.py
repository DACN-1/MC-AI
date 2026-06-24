"""Feature caching: precompute frozen-backbone embeddings once, train MLP many times.

The backbones (LLaVA, CLIP) are frozen during behaviour-cloning training, so
their per-sample pooled output is a deterministic function of
(frame, text, use_language). Running the backbone every epoch wastes ~95%+ of
the GPU time per the timing breakdown in CLAUDE.md. This module:

  1. Iterates the dataset in a *deterministic* sorted order.
  2. Runs the backbone's `encode(images, texts)` once per sample.
  3. Writes the pooled features to a flat memmap so a thin head-only model
     can train at MLP speed.

The on-disk format is intentionally minimal:

  <cache_dir>/<tag>.npy           — (N_samples, feature_dim) float16 memmap
  <cache_dir>/<tag>.json          — {"tag", "backbone", "use_language",
                                     "task_filter", "n_samples", "feature_dim",
                                     "samples": [[stem, frame_idx], ...]}

Cache lookup is by tag — typically `<backbone>_<task>_<lang|nolang>`. The
sidecar JSON pins the sample ordering so `CachedFeatureDataset` can rebuild
the (target, past-action) features at training time without re-decoding video.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image
from tqdm import tqdm

from constants import (
    DEFAULT_PAST_ACTION_K,
    NUM_BINARY,
    NUM_CAMERA,
    PAST_ACTION_DIM,
    action_to_onehot,
    action_to_tensor,
)


def _iter_trajectory_dirs(root: Path, task_filter: str | None):
    """Yield trajectory_task_* dirs in sorted order, optionally filtered."""
    for item in sorted(root.iterdir()):
        if not item.is_dir() or not item.name.startswith("trajectory_task_"):
            continue
        if task_filter and task_filter not in item.name:
            continue
        yield item


def enumerate_samples(
    data_root: str | Path, task_filter: str | None = None, frame_stride: int = 1
) -> list[tuple[str, int, str, str]]:
    """Return (mp4_path, frame_idx, task_text, stem) tuples in cache-order.

    Mirrors TrajectoryDataset's loading logic but with deterministic sorting so
    cache writes and reads agree on indexing. Logs a per-task-dir breakdown and
    raises if any directory contributes zero samples — silent partial loads
    previously wasted an entire training run.

    `frame_stride > 1` keeps only every Nth frame per trajectory (0, N, 2N, ...),
    cutting the cached-feature count — and thus frozen-backbone compute — ~N-fold.
    CachedFeatureDataset still reconstructs targets and past-actions at full
    resolution from all_actions.json, so this thins only the *training* frames:
    the policy and `run_rollout.py` stay at native rate (no action-repeat needed).
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    root = Path(data_root)
    samples: list[tuple[str, int, str, str]] = []
    per_dir_counts: dict[str, int] = {}
    skipped_no_actions: list[str] = []
    skipped_no_video: list[str] = []
    for traj_dir in _iter_trajectory_dirs(root, task_filter):
        all_actions_path = traj_dir / "all_actions.json"
        if not all_actions_path.exists():
            skipped_no_actions.append(traj_dir.name)
            per_dir_counts[traj_dir.name] = 0
            continue
        with all_actions_path.open("r", encoding="utf-8") as fp:
            all_actions = json.load(fp)
        all_infos: dict = {}
        all_infos_path = traj_dir / "all_infos.json"
        if all_infos_path.exists():
            with all_infos_path.open("r", encoding="utf-8") as fp:
                all_infos = json.load(fp)
        videos_dir = traj_dir / "videos"
        dir_n = 0
        for stem in sorted(all_actions.keys()):
            mp4_file = videos_dir / f"video_{stem}.mp4"
            if not mp4_file.exists():
                skipped_no_video.append(stem)
                continue
            task_text = all_infos.get(stem, {}).get("text_prompt", "play minecraft")
            actions = all_actions[stem]
            for idx in range(0, len(actions), frame_stride):
                samples.append((str(mp4_file), idx, task_text, stem))
                dir_n += 1
        per_dir_counts[traj_dir.name] = dir_n

    print("[enumerate_samples] per-task-dir sample counts:")
    for name, n in per_dir_counts.items():
        print(f"  {name}: {n:,}")
    print(f"  TOTAL: {len(samples):,}")
    if skipped_no_actions:
        raise RuntimeError(
            "enumerate_samples: trajectory dirs without all_actions.json: "
            f"{skipped_no_actions} — refusing to silently train on partial data"
        )
    empty_dirs = [name for name, n in per_dir_counts.items() if n == 0]
    if empty_dirs:
        raise RuntimeError(
            f"enumerate_samples: empty trajectory dirs (zero samples): {empty_dirs} "
            f"under {data_root} — likely missing videos/ contents"
        )
    if skipped_no_video:
        # Hard fail rather than silent skip: missing MP4s mean partial data,
        # which silently changes the dataset composition.
        raise RuntimeError(
            f"enumerate_samples: {len(skipped_no_video)} stems had no matching "
            f"video_<stem>.mp4 under videos/ (e.g. {skipped_no_video[:3]}) — "
            "refusing to train on partial trajectories"
        )
    return samples


def _build_backbone(
    backbone: str, llava_id: str, use_language: bool, device: str, patch_grid: int = 0
):
    """Construct a frozen backbone with no action head usage. The head is
    initialised at minimal size (chunk_size=1, past_action_dim=0) but we never
    forward through it — we only call `encode()`.
    """
    if backbone == "llava":
        from VLAAgent import VLAAgent

        if patch_grid > 0:
            raise NotImplementedError(
                "patch_grid caching is CLIP-only for now — the LLaVA encode "
                "path pools over joint image+text tokens and needs its own "
                "spatial design"
            )
        agent = VLAAgent(
            output_dim=1,
            backbone=llava_id,
            use_language=use_language,
            past_action_dim=0,
            chunk_size=1,
        )
    elif backbone == "clip":
        from frozen_vision_baseline import FrozenVisionAgent

        agent = FrozenVisionAgent(
            output_dim=1,
            use_language=use_language,
            past_action_dim=0,
            chunk_size=1,
            patch_grid=patch_grid,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")
    return agent.to(device).eval()


def _decoder(path: str, cache: OrderedDict, VideoReader, cache_size: int = 64):
    dec = cache.get(path)
    if dec is None:
        dec = VideoReader(path, num_threads=1)
        cache[path] = dec
        if len(cache) > cache_size:
            cache.popitem(last=False)
    else:
        cache.move_to_end(path)
    return dec


def _read_progress(progress_path: Path) -> int:
    """Atomic-safe read; returns 0 on missing/partial file."""
    if not progress_path.exists():
        return 0
    try:
        return int(progress_path.read_text().strip() or "0")
    except (ValueError, OSError):
        return 0


def _write_progress(progress_path: Path, value: int) -> None:
    """POSIX-atomic write so the file is never half-written on crash."""
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(str(value))
    tmp.replace(progress_path)


def precompute(
    data_root: str | Path,
    cache_dir: str | Path,
    backbone: str,
    use_language: bool,
    task_filter: str | None = None,
    llava_id: str = "llava-hf/llava-1.5-7b-hf",
    batch_size: int = 32,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
    tag: str | None = None,
    frame_stride: int = 1,
    progress_interval: int = 100,
    patch_grid: int = 0,
) -> Path:
    """Compute and write a feature cache. Returns the .npy path.

    Resumable: if the cache files exist from a previous run *and* metadata
    matches the current request (same backbone, use_language, sample list,
    feature_dim), encoding resumes from the last checkpointed sample. The
    `<tag>.progress` sidecar records that cursor; it's written atomically
    every `progress_interval` batches so a crash loses at most that many
    batches of work.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = enumerate_samples(data_root, task_filter=task_filter, frame_stride=frame_stride)
    if not samples:
        raise RuntimeError(f"No samples found under {data_root} (filter={task_filter!r})")

    if tag is None:
        task_part = (task_filter or "all").replace("/", "_")
        lang_part = "lang" if use_language else "nolang"
        tag = f"{backbone}_{task_part}_{lang_part}"
        if frame_stride > 1:
            tag += f"_stride{frame_stride}"
        if patch_grid > 0:
            tag += f"_patch{patch_grid}"
    cache_path = cache_dir / f"{tag}.npy"
    meta_path = cache_dir / f"{tag}.json"
    progress_path = cache_dir / f"{tag}.progress"

    try:
        from decord import VideoReader
    except ImportError as e:
        raise ImportError("decord required for video decoding. pip install decord") from e

    agent = _build_backbone(backbone, llava_id, use_language, device, patch_grid=patch_grid)
    # Probe feature_dim by encoding one sample (cheap; also needed to validate
    # resume against any prior cache file).
    with th.no_grad():
        mp4_path, frame_idx, task_text, _stem = samples[0]
        vr = VideoReader(mp4_path, num_threads=1)
        probe_img = Image.fromarray(vr[frame_idx].asnumpy())
        probe_feat = agent.encode([probe_img], [task_text])
    feature_dim = int(probe_feat.shape[-1])

    # Decide whether to resume or start fresh. The metadata file must exist
    # AND match the requested cell exactly — any mismatch (different sample
    # ordering, different feature_dim, ...) means the on-disk cache is stale.
    resume = False
    start_batch = 0
    if cache_path.exists() and meta_path.exists():
        with meta_path.open() as fp:
            meta_existing = json.load(fp)
        # llava_id is checked only for the LLaVA backbone — CLIP does not store
        # an HF model id in its metadata. Without this check we would happily
        # resume against a cache built with a different LLaVA checkpoint.
        expected_llava_id = llava_id if backbone == "llava" else None
        matches = (
            meta_existing.get("n_samples") == len(samples)
            and meta_existing.get("feature_dim") == feature_dim
            and meta_existing.get("backbone") == backbone
            and meta_existing.get("use_language") == use_language
            and meta_existing.get("task_filter") == task_filter
            and meta_existing.get("llava_id") == expected_llava_id
            and meta_existing.get("frame_stride", 1) == frame_stride
            and meta_existing.get("patch_grid", 0) == patch_grid
        )
        if matches:
            start_batch = _read_progress(progress_path)
            if start_batch >= len(samples):
                print(f"[{tag}] already complete ({start_batch:,}/{len(samples):,})")
                return cache_path
            resume = True
            print(
                f"[{tag}] resuming from sample {start_batch:,}/{len(samples):,} "
                f"({100 * start_batch / len(samples):.1f}% done)"
            )
        else:
            print(f"[{tag}] cache metadata mismatch — rebuilding from scratch")

    if not resume:
        # Write metadata up-front so a future resume can validate against it.
        meta = {
            "tag": tag,
            "backbone": backbone,
            "llava_id": llava_id if backbone == "llava" else None,
            "use_language": use_language,
            "task_filter": task_filter,
            "frame_stride": frame_stride,
            "patch_grid": patch_grid,
            "n_samples": len(samples),
            "feature_dim": feature_dim,
            "dtype": "float16",
            "samples": [[stem, frame_idx] for (_, frame_idx, _, stem) in samples],
        }
        with meta_path.open("w") as fp:
            json.dump(meta, fp)
        _write_progress(progress_path, 0)
        start_batch = 0

    print(f"[{tag}] N={len(samples):,}  feature_dim={feature_dim}  device={device}")

    # FP16 storage: half the disk for negligible BC accuracy impact.
    # 'r+' on resume preserves existing contents; 'w+' truncates for a fresh build.
    mm = np.memmap(
        cache_path,
        dtype=np.float16,
        mode="r+" if resume else "w+",
        shape=(len(samples), feature_dim),
    )
    decoder_cache: OrderedDict = OrderedDict()

    with th.no_grad():
        batches_since_flush = 0
        for batch_start in tqdm(
            range(start_batch, len(samples), batch_size),
            desc=f"encoding {tag}",
            initial=start_batch // batch_size,
            total=(len(samples) + batch_size - 1) // batch_size,
        ):
            batch = samples[batch_start : batch_start + batch_size]
            imgs: list[Image.Image] = []
            texts: list[str] = []
            for mp4_path, frame_idx, task_text, _stem in batch:
                vr = _decoder(mp4_path, decoder_cache, VideoReader)
                imgs.append(Image.fromarray(vr[frame_idx].asnumpy()))
                texts.append(task_text)
            feats = agent.encode(imgs, texts).detach().cpu().to(th.float16).numpy()
            mm[batch_start : batch_start + len(batch)] = feats
            batches_since_flush += 1
            if batches_since_flush >= progress_interval:
                mm.flush()
                _write_progress(progress_path, batch_start + len(batch))
                batches_since_flush = 0

    mm.flush()
    del mm
    _write_progress(progress_path, len(samples))
    print(f"[{tag}] wrote {cache_path}  ({cache_path.stat().st_size / 1024 ** 3:.2f} GB)")
    return cache_path


def load_cache(cache_dir: str | Path, tag: str) -> tuple[np.ndarray, dict]:
    """Open a cache memmap (read-only) plus its metadata sidecar."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / f"{tag}.json"
    with meta_path.open("r") as fp:
        meta = json.load(fp)
    mm = np.memmap(
        cache_dir / f"{tag}.npy",
        dtype=np.float16,
        mode="r",
        shape=(meta["n_samples"], meta["feature_dim"]),
    )
    return mm, meta


# ---------------------------------------------------------------------
# Head-only training pieces
# ---------------------------------------------------------------------
class CachedFeatureDataset(th.utils.data.Dataset):
    """Dataset over (cached_feature, target_chunk, past_actions) tuples.

    Reads pooled features from a memmap produced by `precompute`, and
    reconstructs targets / past-actions from the original all_actions.json
    files (so the same cache serves any chunk_size / past_action_k combination).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        tag: str,
        data_root: str | Path,
        past_action_k: int = 0,
        chunk_size: int = 1,
        frame_history_k: int = 0,
        stem_filter: str | None = None,
        camera_onset_weights: dict | None = None,
    ):
        self.past_action_k = past_action_k
        self.chunk_size = chunk_size
        # Per-frame camera-CE weight (1.0 default). When supplied (keyed by
        # stem -> (n,) array), __getitem__ returns the chunk slice so the
        # trainer can upweight the camera loss on pre-attack-onset aiming
        # windows. See imitation_learning.compute_camera_onset_weights.
        self.camera_onset_weights = camera_onset_weights or {}
        # Train-time task slice: keep only stems starting with this prefix
        # (e.g. "chop_a_tree") without rebuilding the cache. Lets a combined
        # cache serve single-task control cells (sample composition matches a
        # single-task cache exactly; only the memmap rows read differ).
        self.stem_filter = stem_filter
        # Visual temporal context: concatenate the K previous CACHED frame
        # features (stride-spaced, zero-padded at trajectory start, oldest
        # first / current last) in front of the current frame's feature.
        # Post-cache lever: a single frame + past *actions* carries no motion
        # signal — the head can't tell "rotating toward the target" from
        # "away" — but the cache already holds every sampled frame's feature,
        # so the window is assembled dataset-side for free.
        self.frame_history_k = frame_history_k

        self._features, self.meta = load_cache(cache_dir, tag)

        # Group cached samples by stem and load per-stem action tables.
        # `cache_index[stem][frame_idx]` -> row in the memmap.
        cache_index: dict[str, dict[int, int]] = {}
        for row, (stem, frame_idx) in enumerate(self.meta["samples"]):
            cache_index.setdefault(stem, {})[int(frame_idx)] = row
        self.cache_index = cache_index

        self.targets_by_stem: dict[str, np.ndarray] = {}
        self.onehots_by_stem: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        root = Path(data_root)
        # task_filter=None scans every trajectory_task_* dir (combined cache).
        for traj_dir in _iter_trajectory_dirs(root, self.meta.get("task_filter")):
            all_actions_path = traj_dir / "all_actions.json"
            if not all_actions_path.exists():
                continue
            with all_actions_path.open("r", encoding="utf-8") as fp:
                all_actions = json.load(fp)
            for stem in sorted(all_actions.keys()):
                if stem not in cache_index:
                    continue
                if stem_filter and not stem.startswith(stem_filter):
                    continue
                actions = all_actions[stem]
                # range(last_valid + 1) is empty when chunk_size > len(actions),
                # which silently drops the trajectory. Fail loud instead — for
                # the BASALT 3000-frame trajectories with chunk_size<=8 this
                # is purely a defensive check.
                if chunk_size > len(actions):
                    raise RuntimeError(
                        f"chunk_size={chunk_size} exceeds trajectory length "
                        f"{len(actions)} for stem={stem!r}; would silently drop "
                        "this stem from the training set"
                    )
                self.targets_by_stem[stem] = np.stack(
                    [action_to_tensor(a).numpy() for a in actions]
                ).astype(np.float32)
                if past_action_k > 0:
                    self.onehots_by_stem[stem] = np.stack(
                        [action_to_onehot(a) for a in actions]
                    ).astype(np.float32)
                last_valid = len(actions) - chunk_size
                for idx in range(last_valid + 1):
                    if idx in cache_index[stem]:
                        self.samples.append((stem, idx))

    def feature_dim(self) -> int:
        """Effective head input width — (1 + frame_history_k) cache slots."""
        return int(self.meta["feature_dim"]) * (1 + self.frame_history_k)

    def _past(self, stem: str, frame_idx: int) -> th.Tensor:
        if self.past_action_k == 0:
            return th.zeros(0, dtype=th.float32)
        onehots = self.onehots_by_stem[stem]
        past = np.zeros((self.past_action_k, PAST_ACTION_DIM), dtype=np.float32)
        start = max(0, frame_idx - self.past_action_k)
        n_valid = frame_idx - start
        if n_valid > 0:
            past[self.past_action_k - n_valid :] = onehots[start:frame_idx]
        return th.from_numpy(past.reshape(-1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        stem, frame_idx = self.samples[idx]
        row = self.cache_index[stem][frame_idx]
        # Cast features to float32 — the head trains in fp32.
        feat = th.from_numpy(np.asarray(self._features[row], dtype=np.float32))
        if self.frame_history_k > 0:
            stride = int(self.meta.get("frame_stride", 1))
            base_dim = int(self.meta["feature_dim"])
            window = np.zeros(
                ((self.frame_history_k + 1), base_dim), dtype=np.float32
            )
            for j in range(self.frame_history_k, 0, -1):
                prev_row = self.cache_index[stem].get(frame_idx - j * stride)
                if prev_row is not None:
                    window[self.frame_history_k - j] = np.asarray(
                        self._features[prev_row], dtype=np.float32
                    )
            window[-1] = feat.numpy()
            feat = th.from_numpy(window.reshape(-1))
        target = th.from_numpy(
            self.targets_by_stem[stem][frame_idx : frame_idx + self.chunk_size]
        )
        cw = self.camera_onset_weights.get(stem)
        if cw is None:
            cam_w = th.ones(self.chunk_size, dtype=th.float32)
        else:
            cam_w = th.from_numpy(
                cw[frame_idx : frame_idx + self.chunk_size].astype(np.float32)
            )
        return feat, target, self._past(stem, frame_idx), cam_w


class HeadOnlyAgent(th.nn.Module):
    """Trainable MLP head that consumes pre-pooled features.

    Mirrors `VLAAgent.action_head` exactly, so a checkpoint trained on cached
    features is structurally identical to the head trained end-to-end.

    Optional `learnable_bce_temp` adds a per-binary-action scalar that divides
    the binary logits in forward(). At training time the temperature is learned
    alongside the head; at inference time it sharpens (or softens) the sigmoid
    so the rollout doesn't need decode-time `--binary-thresholds` or
    `--binary-logit-bias` flags. Camera CE logits are untouched.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        past_action_dim: int = 0,
        chunk_size: int = 1,
        hidden_dim: int | None = None,
        learnable_bce_temp: bool = False,
        feature_norm: bool = False,
        image_dropout: float = 0.0,
        image_feature_dim: int | None = None,
        film: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.past_action_dim = past_action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim or feature_dim
        self.learnable_bce_temp = learnable_bce_temp
        self.film = film
        # Split point of the cached [image_pool || text_pool] feature. Defaults to
        # the first half (LLaVA split-pool: 4096 image + 4096 text = 8192).
        self.image_feature_dim = (
            image_feature_dim if image_feature_dim is not None else feature_dim // 2
        )
        self.text_feature_dim = feature_dim - self.image_feature_dim
        # Post-cache fix B: modality dropout on the image sub-block. With prob
        # `image_dropout`, zero image_pool for a sample so the head must predict
        # from text — breaking the "read the task off the image" shortcut. Train-
        # time only (gated by self.training); a no-op at rollout. In FiLM mode it
        # is what forces `beta(text)` to carry the task signal (see below).
        self.image_dropout = float(image_dropout)

        if self.film:
            # Post-cache fix C: FiLM conditioning. text_pool generates a per-channel
            # scale (gamma) and shift (beta) that modulate image_pool:
            #     modulated = (1 + gamma(text)) * image + beta(text)
            # and the MLP consumes ONLY the modulated image. Unlike concat (which
            # the head can zero-weight to ignore text), text is structurally injected
            # into the image representation. Synergistic with image_dropout: when the
            # image is dropped, modulated collapses to beta(text), forcing beta to
            # carry the full task signal — and beta is added at rollout too, so the
            # prompt shifts behavior. Identity init (gen weight=bias=0 -> gamma=1,
            # beta=0) starts the head as an image-only MLP for stability, then learns.
            self.feature_norm = None
            if feature_norm:
                # Per-modality LayerNorm: equalize image/text scale before FiLM
                # (stable gamma/beta from text; clean image to modulate).
                self.film_ln_image = th.nn.LayerNorm(self.image_feature_dim)
                self.film_ln_text = th.nn.LayerNorm(self.text_feature_dim)
            else:
                self.film_ln_image = self.film_ln_text = None
            self.film_gen = th.nn.Linear(
                self.text_feature_dim, 2 * self.image_feature_dim
            )
            th.nn.init.zeros_(self.film_gen.weight)
            th.nn.init.zeros_(self.film_gen.bias)
            head_in = self.image_feature_dim + past_action_dim
        else:
            # Post-cache fix A: LayerNorm the whole frozen feature before the probe.
            # LLaVA's 8192-d mean-pooled features arrive on very different per-dim
            # scales (image_pool is a mean over ~576 tokens, text_pool over a few),
            # so an un-normalized first Linear is dominated by the high-variance image
            # dims and effectively ignores text. A LayerNorm with learnable affine
            # puts the halves on equal footing. Active at train AND rollout, so it is
            # reconstructed from `config["feature_norm"]` by agent_loader.
            self.feature_norm = th.nn.LayerNorm(feature_dim) if feature_norm else None
            head_in = feature_dim + past_action_dim

        self.action_head = th.nn.Sequential(
            th.nn.Linear(head_in, self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim, output_dim * chunk_size),
        )
        if learnable_bce_temp:
            # Per-binary-action temperature, init to 1.0. Stored as raw scalars
            # (clamped > 1e-3 at use site) so the optimizer can move it freely.
            self.bce_temperature = th.nn.Parameter(th.ones(NUM_BINARY))

    def _drop_image(self, img: th.Tensor) -> th.Tensor:
        """Whole-modality Bernoulli dropout: zero the image block for a random
        subset of the batch (train only, no 1/(1-p) rescale). Out-of-place so
        autograd is happy after a (non-leaf) LayerNorm."""
        if not (self.training and self.image_dropout > 0.0):
            return img
        keep = (
            th.rand(img.size(0), 1, device=img.device, dtype=img.dtype)
            >= self.image_dropout
        ).to(img.dtype)
        return img * keep

    def forward(self, features: th.Tensor, past_actions: th.Tensor | None = None) -> th.Tensor:
        if self.film:
            img = features[:, : self.image_feature_dim]
            txt = features[:, self.image_feature_dim :]
            if self.film_ln_image is not None:
                img = self.film_ln_image(img)
                txt = self.film_ln_text(txt)
            img = self._drop_image(img)
            gb = self.film_gen(txt)
            gamma = gb[:, : self.image_feature_dim]
            beta = gb[:, self.image_feature_dim :]
            x = (1.0 + gamma) * img + beta  # when img dropped -> x = beta(text)
        else:
            x = features
            if self.feature_norm is not None:
                x = self.feature_norm(x)
            if self.training and self.image_dropout > 0.0 and self.image_feature_dim > 0:
                keep = (
                    th.rand(x.size(0), 1, device=x.device, dtype=x.dtype)
                    >= self.image_dropout
                ).to(x.dtype)
                mask = th.ones_like(x)
                mask[:, : self.image_feature_dim] = keep
                x = x * mask
        if self.past_action_dim > 0:
            if past_actions is None:
                raise ValueError("past_actions required when past_action_dim > 0")
            past = past_actions.to(x.device).to(x.dtype)
            x = th.cat([x, past], dim=-1)
        flat = self.action_head(x)
        logits = flat.view(flat.size(0), self.chunk_size, self.output_dim)
        if self.learnable_bce_temp:
            tau = self.bce_temperature.clamp_min(1e-3).view(1, 1, NUM_BINARY)
            logits = th.cat(
                [logits[..., :NUM_BINARY] / tau, logits[..., NUM_BINARY:]],
                dim=-1,
            )
        return logits


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Precompute frozen-backbone features for fast head training."
    )
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--backbone", required=True, choices=("llava", "clip"))
    p.add_argument(
        "--task-filter",
        required=True,
        help="Substring matched against trajectory_task_* dir name (e.g. 'chop_a_tree')",
    )
    p.add_argument("--use-language", action="store_true")
    p.add_argument("--no-language", dest="use_language", action="store_false")
    p.set_defaults(use_language=True)
    p.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--device",
        default="cuda" if th.cuda.is_available() else "cpu",
    )
    p.add_argument("--tag", default=None, help="Override the default cache tag")
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Cache every Nth frame per trajectory (N>1 cuts cost ~N-fold; "
        "targets/past-actions stay full-resolution, rollout unchanged)",
    )
    p.add_argument(
        "--patch-grid",
        type=int,
        default=0,
        help="CLIP only: store a GxG average-pooled grid of vision-tower patch "
        "tokens instead of the pooled global vector (plus the usual text "
        "feature). 0 = legacy pooled mode. Cache tag gets a _patchG suffix. "
        "Feature dim becomes G*G*1024 + 768 for ViT-L/14.",
    )
    args = p.parse_args()

    precompute(
        data_root=args.data_dir,
        cache_dir=args.cache_dir,
        backbone=args.backbone,
        use_language=args.use_language,
        task_filter=args.task_filter,
        llava_id=args.llava_id,
        batch_size=args.batch_size,
        device=args.device,
        tag=args.tag,
        frame_stride=args.frame_stride,
        patch_grid=args.patch_grid,
    )


if __name__ == "__main__":
    main()
