#!/usr/bin/env python3
"""Measure frozen-backbone encode throughput, then extrapolate full-run cost.

Run this on a freshly-rented GPU (e.g. a vast.ai RTX 5090) *before* committing
to a multi-day cache build. It mirrors the exact encode loop in
`feature_cache.precompute` — same `_build_backbone()`, same decord decode, same
`agent.encode(imgs, texts)` — so the samples/sec it reports is what the real
cache build will see. It writes nothing to the cache; it only times the forward.

Example (LLaVA, language on, on a 5090 priced at $0.686/hr):

    python scripts/probe_5090_throughput.py \
        --data-dir ./trajectories --backbone llava --use-language \
        --batch-size 24 --num-frames 480 --price 0.686

Then read off "estimated full-run cost" and decide whether the rental is worth
it / whether to bisect batch-size up or down before the real run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch as th
from PIL import Image

# Allow running as `python scripts/probe_5090_throughput.py` from anywhere —
# the repo root (parent of scripts/) must be importable for feature_cache etc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_cache import _build_backbone, enumerate_samples  # noqa: E402

# Combined chop+dirt dataset size — every ablation cell trains on the full set
# (strict combined-dataset rule). Override with --full-samples if your run scope
# differs. The cache is built once per (backbone x use_language) condition.
DEFAULT_FULL_SAMPLES = 6_240_000

# Expected pooled feature dim per backbone — a mismatch here means a cache built
# now would be rejected by feature_cache resume-validation when merged later.
# LLaVA: 2 * hidden_size (split image/text pooling per Phase C step 2).
# CLIP:  2 * projection_dim (image_proj || text_proj).
EXPECTED_FEATURE_DIM = {"llava": 8192, "clip": 1536}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="trajectories root")
    p.add_argument("--backbone", choices=["llava", "clip"], default="llava")
    p.add_argument("--task-filter", default=None, help="e.g. chop_a_tree (default: all)")
    lang = p.add_mutually_exclusive_group()
    lang.add_argument("--use-language", dest="use_language", action="store_true")
    lang.add_argument("--no-language", dest="use_language", action="store_false")
    p.set_defaults(use_language=True)
    p.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--batch-size", type=int, default=24,
                   help="start at 24 on a 32GB 5090 (no FA2); bisect up/down on nvidia-smi")
    p.add_argument("--num-frames", type=int, default=480,
                   help="frames to time over (after warmup)")
    p.add_argument("--warmup-batches", type=int, default=2,
                   help="batches to run untimed first (GPU warmup / lazy init)")
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--price", type=float, default=0.686, help="rental $/hr for the cost estimate")
    p.add_argument("--full-samples", type=int, default=DEFAULT_FULL_SAMPLES,
                   help="samples in one full cache (per backbone x lang condition)")
    args = p.parse_args()

    if args.device == "cuda":
        if not th.cuda.is_available():
            raise SystemExit("CUDA not available — rent a GPU box or pass --device cpu")
        cap = th.cuda.get_device_capability()
        print(f"GPU: {th.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
              f"torch {th.__version__} (cuda {th.version.cuda})")
        if cap[0] >= 12:
            print("  -> Blackwell (sm_120): running on sdpa attention by design (no flash-attn).")

    try:
        from decord import VideoReader
    except ImportError as e:  # decord is the highest-risk wheel on a fresh box
        raise SystemExit("decord not importable — check the install (pip install decord)") from e

    samples = enumerate_samples(args.data_dir, task_filter=args.task_filter)
    if not samples:
        raise SystemExit(f"No samples under {args.data_dir} (filter={args.task_filter!r})")
    n_total = args.warmup_batches * args.batch_size + args.num_frames
    samples = samples[:n_total]
    print(f"probing {len(samples)} frames "
          f"({args.warmup_batches} warmup batches + {args.num_frames} timed) "
          f"batch_size={args.batch_size} backbone={args.backbone} "
          f"lang={'on' if args.use_language else 'off'}")

    agent = _build_backbone(args.backbone, args.llava_id, args.use_language, args.device)

    def load_batch(batch):
        imgs, texts = [], []
        for mp4_path, frame_idx, task_text, _stem in batch:
            vr = VideoReader(mp4_path, num_threads=1)
            imgs.append(Image.fromarray(vr[frame_idx].asnumpy()))
            texts.append(task_text)
        return imgs, texts

    feature_dim = None
    timed_frames = 0
    warmup_frames = args.warmup_batches * args.batch_size
    t0 = None

    with th.no_grad():
        for batch_start in range(0, len(samples), args.batch_size):
            batch = samples[batch_start : batch_start + args.batch_size]
            imgs, texts = load_batch(batch)
            # Start the clock once warmup is done; sync so timing excludes warmup.
            if batch_start >= warmup_frames and t0 is None:
                if args.device == "cuda":
                    th.cuda.synchronize()
                t0 = time.perf_counter()
            feats = agent.encode(imgs, texts)
            if args.device == "cuda":
                th.cuda.synchronize()
            feature_dim = int(feats.shape[-1])
            if batch_start >= warmup_frames:
                timed_frames += len(batch)

    elapsed = time.perf_counter() - t0
    sps = timed_frames / elapsed
    print()
    print(f"feature_dim = {feature_dim}", end="")
    exp = EXPECTED_FEATURE_DIM.get(args.backbone)
    if exp is not None and feature_dim != exp:
        print(f"  !! expected {exp} for {args.backbone} — a cache built now may not "
              f"merge with existing caches")
    else:
        print("  (ok)")
    print(f"timed {timed_frames} frames in {elapsed:.1f}s")
    print(f"throughput  = {sps:.1f} samples/sec   ({1000 / sps:.2f} ms/sample)")

    hours = args.full_samples / sps / 3600
    print()
    print(f"--- extrapolation to one full cache ({args.full_samples:,} samples) ---")
    print(f"  ~{hours:.1f} h  ->  ~${hours * args.price:.2f} @ ${args.price:.3f}/hr")
    print(f"  full 2x2 ablation builds 2 {args.backbone} caches (lang on/off): "
          f"~{2 * hours:.1f} h -> ~${2 * hours * args.price:.2f}")
    print("  (CLIP caches + head training add a few more hours; see CLAUDE.md budget)")
    if args.device == "cuda":
        free, total = th.cuda.mem_get_info()
        print(f"  VRAM in use now: {(total - free) / 1024**3:.1f} / {total / 1024**3:.1f} GB "
              f"at batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
