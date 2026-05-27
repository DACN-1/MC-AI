#!/usr/bin/env python3
"""Cluster pipeline: precompute backbone features -> train head -> evaluate.

Default path uses the feature cache (frozen backbones run once per cell,
heads train at MLP speed). Pass `--end-to-end` to fall back to the legacy
path where the backbone runs every batch.

Usage::

    python cluster_pipeline.py --data-dir ./trajectories \\
        --task-filter chop_a_tree --backbone llava

Each invocation trains one ablation cell. A cell is identified by
(backbone, task_filter, use_language). The cache file is built lazily on
first request and reused by every subsequent run for the same cell.

Expected input layout::

    trajectories/
    └── trajectory_task_*/
        ├── all_actions.json
        ├── all_infos.json
        └── videos/*.mp4

Outputs:
    <output-dir>/model.pt        head weights + config
    <output-dir>/metrics.json    per-action F1, binary acc, camera MAE
    <cache-dir>/<tag>.npy        backbone embeddings (persists across runs)
    <cache-dir>/<tag>.json       cache metadata
"""

import argparse
import time
from pathlib import Path

import torch as th
from torch.utils.data import random_split

from VLAAgent import VLAAgent
from constants import DEFAULT_PAST_ACTION_K, NUM_OUTPUT_LOGITS, PAST_ACTION_DIM
from imitation_learning import (
    TrajectoryDataset,
    evaluate,
    evaluate_cached,
    train_cached_head,
    train_vla,
)


def _cache_tag(backbone: str, task_filter: str | None, use_language: bool) -> str:
    task_part = task_filter if task_filter else "combined"
    return f"{backbone}_{task_part}_{'lang' if use_language else 'nolang'}"


def _ensure_cache(
    cache_dir: Path,
    backbone: str,
    task_filter: str | None,
    use_language: bool,
    data_dir: Path,
    llava_id: str,
    cache_batch_size: int,
    device: str,
) -> str:
    """Build (or resume) the feature cache, returning the tag.

    `precompute` writes the full-size .npy memmap and metadata up-front and
    only fills the memmap incrementally — so "files exist" is NOT a complete-
    cache signal. We have to consult the .progress cursor too, otherwise a
    job killed mid-build leaves a partial cache that the next run would
    silently treat as done and train a head on 99% zeros.
    """
    import json as _json
    from feature_cache import precompute  # lazy import — only needed for cached path

    tag = _cache_tag(backbone, task_filter, use_language)
    cache_npy = cache_dir / f"{tag}.npy"
    cache_meta = cache_dir / f"{tag}.json"
    progress_path = cache_dir / f"{tag}.progress"

    if cache_npy.exists() and cache_meta.exists():
        try:
            n_samples = int(_json.loads(cache_meta.read_text()).get("n_samples", 0))
        except (ValueError, OSError, KeyError):
            n_samples = 0
        progress = 0
        if progress_path.exists():
            try:
                progress = int(progress_path.read_text().strip() or "0")
            except (ValueError, OSError):
                progress = 0
        if n_samples > 0 and progress >= n_samples:
            print(
                f"[cache] {tag}: reusing complete cache "
                f"({cache_npy.stat().st_size / 1024 ** 3:.2f} GB, {progress:,}/{n_samples:,})"
            )
            return tag
        print(f"[cache] {tag}: partial ({progress:,}/{n_samples:,}) — resuming")
    else:
        print(f"[cache] {tag}: not found — building (one-time, ~hours for LLaVA)")

    precompute(
        data_root=data_dir,
        cache_dir=cache_dir,
        backbone=backbone,
        use_language=use_language,
        task_filter=task_filter,
        llava_id=llava_id,
        batch_size=cache_batch_size,
        device=device,
        tag=tag,
    )
    return tag


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster pipeline: cache -> train -> evaluate")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with trajectory data")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./caches"))
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Substring matched against trajectory_task_* dir name (e.g. 'chop_a_tree'). "
             "Omit to build a combined cache across every task dir (recommended default).",
    )
    parser.add_argument(
        "--backbone",
        default="llava",
        choices=("llava", "clip"),
        help="Frozen backbone used to produce pooled features",
    )
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Head-training batch size (large is fine on cached features)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=16,
        help="Batch size for the one-time backbone forward pass during cache build",
    )
    parser.add_argument(
        "--past-action-k",
        type=int,
        default=DEFAULT_PAST_ACTION_K,
        help="Past actions concatenated to head input (0 = off)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Future actions predicted per forward (1 = off)",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Zero the text prompt (language-pathway ablation cell)",
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Skip caching; train backbone+head end-to-end (legacy path).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any existing checkpoint at <output-dir>/model.pt and retrain from epoch 0",
    )
    parser.add_argument("--skip-train", action="store_true", help="Load existing model and run eval only")
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    use_language = not args.no_language
    start_time = time.time()
    model_path = args.output_dir / "model.pt"

    if args.end_to_end:
        # ----- legacy path: backbone runs every batch ----------------------
        if not args.skip_train:
            print("\n" + "=" * 60)
            print(f"End-to-end training: backbone={args.backbone}  use_language={use_language}")
            print("=" * 60)
            if args.backbone != "llava":
                raise NotImplementedError(
                    "End-to-end path only supports backbone='llava' currently. "
                    "Use --backbone clip with the cached path."
                )
            model, test_set, _ = train_vla(
                data_root=str(args.data_dir),
                backbone=args.llava_model,
                out_weights=str(model_path),
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                num_workers=args.num_workers,
                past_action_k=args.past_action_k,
                chunk_size=args.chunk_size,
                use_language=use_language,
                restart=args.restart,
            )
        else:
            print(f"Loading existing model from {model_path}")
            ckpt = th.load(model_path, map_location=args.device)
            cfg = ckpt.get("config", {})
            past_action_k = cfg.get("past_action_k", 0)
            chunk_size = cfg.get("chunk_size", 1)
            model = VLAAgent(
                NUM_OUTPUT_LOGITS,
                ckpt["llava_model"],
                use_language=cfg.get("use_language", True),
                past_action_dim=past_action_k * PAST_ACTION_DIM,
                chunk_size=chunk_size,
            ).to(args.device)
            model.action_head.load_state_dict(ckpt["state_dict"])
            dataset = TrajectoryDataset(
                str(args.data_dir),
                past_action_k=past_action_k,
                chunk_size=chunk_size,
            )
            test_size = int(len(dataset) * 0.1)
            val_size = int(len(dataset) * 0.1)
            train_size = len(dataset) - val_size - test_size
            _, _, test_set = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=th.Generator().manual_seed(42),
            )

        print("\n" + "=" * 60)
        print("Evaluating end-to-end model")
        print("=" * 60)
        evaluate(
            model,
            test_set,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        # ----- default path: precompute once, train head fast --------------
        tag = _ensure_cache(
            cache_dir=args.cache_dir,
            backbone=args.backbone,
            task_filter=args.task_filter,
            use_language=use_language,
            data_dir=args.data_dir,
            llava_id=args.llava_model,
            cache_batch_size=args.cache_batch_size,
            device=args.device,
        )

        if not args.skip_train:
            print("\n" + "=" * 60)
            print(f"Training head on cache {tag}")
            print("=" * 60)
            model, test_set, _ = train_cached_head(
                cache_dir=str(args.cache_dir),
                cache_tag=tag,
                data_root=str(args.data_dir),
                out_weights=str(model_path),
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                num_workers=args.num_workers,
                past_action_k=args.past_action_k,
                chunk_size=args.chunk_size,
                restart=args.restart,
            )
        else:
            from feature_cache import CachedFeatureDataset, HeadOnlyAgent

            print(f"Loading existing head from {model_path}")
            ckpt = th.load(model_path, map_location=args.device)
            cfg = ckpt.get("config", {})
            past_action_k = cfg.get("past_action_k", 0)
            chunk_size = cfg.get("chunk_size", 1)
            dataset = CachedFeatureDataset(
                cache_dir=str(args.cache_dir),
                tag=tag,
                data_root=str(args.data_dir),
                past_action_k=past_action_k,
                chunk_size=chunk_size,
            )
            model = HeadOnlyAgent(
                feature_dim=cfg["feature_dim"],
                output_dim=NUM_OUTPUT_LOGITS,
                past_action_dim=cfg.get("past_action_dim", 0),
                chunk_size=chunk_size,
                hidden_dim=cfg.get("hidden_dim"),
            ).to(args.device)
            model.load_state_dict(ckpt["state_dict"])
            test_size = int(len(dataset) * 0.1)
            val_size = int(len(dataset) * 0.1)
            train_size = len(dataset) - val_size - test_size
            _, _, test_set = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=th.Generator().manual_seed(42),
            )

        print("\n" + "=" * 60)
        print("Evaluating cached-head model")
        print("=" * 60)
        evaluate_cached(
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
