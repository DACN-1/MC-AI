#!/usr/bin/env python3
"""T4 counterfactual prompt-swap probe (last_refinements.md T4).

Tests the asserted mechanism behind the conditioning result: CLIP's separate
text encoder gives an image-independent prompt axis the head can read, whereas
LLaVA's pooled text-half is computed inside the joint decoder (text tokens
attend to the image) and therefore barely moves when only the prompt changes at
a fixed frame. The cached 99.7%% linear probe could not show this because it is
prompt<->biome confounded; this re-encodes a *fixed* set of frames under all
four eval prompts and measures how far the pooled **text-half** moves.

For each backbone it reports, across the frame set:
  * text-half prompt sensitivity  — cosine + relative-L2 between prompts at a
    fixed frame. LLaVA near cosine 1 / relL2 0 ("barely moves") vs CLIP much
    lower cosine / higher relL2 ("jumps") demonstrates the mechanism.
  * image-half prompt sensitivity — the confound control. CLIP's image-half is
    prompt-invariant by construction (cosine 1.0); any LLaVA image-half movement
    quantifies how much the prompt leaks into the image tokens.
  * text-half frame-invariance    — mean cosine of a fixed prompt's text-half
    across different frames. CLIP == 1 (frame-independent text encoder); LLaVA
    < 1 directly measures the image-entanglement.

Cache-incompatible by design (the on-disk caches are mean-pooled, one prompt per
frame) — this must re-encode. LLaVA needs a GPU (7B forward); CLIP runs anywhere.

Run (on a CUDA box):
  python scripts/prompt_swap_probe.py --backbone both --n-frames 500 --device cuda
  python scripts/prompt_swap_probe.py --backbone clip --n-frames 32 --device cpu   # smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch as th  # noqa: E402
from PIL import Image  # noqa: E402

from feature_cache import _build_backbone, enumerate_samples  # noqa: E402

# Match eval_suite.sh conditions A/B/C/D exactly.
PROMPTS = {
    "A_empty": "",
    "B_ood": "Play Minecraft.",
    "C_chop": "chop a tree",
    "D_dirt": "collect dirt",
}


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if th.cuda.is_available():
        return "cuda"
    if getattr(th.backends, "mps", None) is not None and th.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frames(data_dir: str, n_frames: int):
    """Evenly subsample n_frames across BOTH tasks and decode them to PIL once."""
    from decord import VideoReader

    samples = enumerate_samples(data_dir)  # (mp4, idx, task_text, stem), both tasks
    if n_frames < len(samples):
        step = len(samples) / n_frames
        picks = [samples[int(i * step)] for i in range(n_frames)]
    else:
        picks = samples
    print(f"[probe] decoding {len(picks)} frames (evenly spaced over {len(samples):,})")
    readers: dict[str, object] = {}
    images = []
    for mp4_path, idx, _txt, _stem in picks:
        vr = readers.get(mp4_path)
        if vr is None:
            vr = VideoReader(mp4_path, num_threads=1)
            readers[mp4_path] = vr
            if len(readers) > 64:  # bound open handles
                readers.pop(next(iter(readers)))
        images.append(Image.fromarray(vr[idx].asnumpy()))
    return images


@th.no_grad()
def encode_all(agent, images, prompt, batch_size, device):
    """Encode every frame under one prompt; return (N, feature_dim) float32."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        feats = agent.encode(batch, [prompt] * len(batch))
        out.append(feats.detach().to("cpu", th.float32).numpy())
    return np.concatenate(out, axis=0)


def _cos(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den


def _rel_l2(a, b):
    num = np.linalg.norm(a - b, axis=1)
    den = 0.5 * (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)) + 1e-8
    return num / den


def half_stats(by_prompt_half):
    """Pairwise prompt cosine/relL2 (mean over frames) + frame-invariance."""
    names = list(by_prompt_half)
    pairs = {}
    cos_vals, rel_vals = [], []
    for i, p in enumerate(names):
        for q in names[i + 1 :]:
            c = float(_cos(by_prompt_half[p], by_prompt_half[q]).mean())
            r = float(_rel_l2(by_prompt_half[p], by_prompt_half[q]).mean())
            pairs[f"{p}|{q}"] = {"cosine": c, "rel_l2": r}
            cos_vals.append(c)
            rel_vals.append(r)
    # Frame-invariance: for each prompt, mean cosine between random frame pairs.
    frame_inv = {}
    for p, M in by_prompt_half.items():
        n = M.shape[0]
        if n >= 2:
            perm = np.roll(np.arange(n), 1)
            frame_inv[p] = float(_cos(M, M[perm]).mean())
    return {
        "pairwise": pairs,
        "mean_pair_cosine": float(np.mean(cos_vals)),
        "mean_pair_rel_l2": float(np.mean(rel_vals)),
        "frame_invariance_cosine": frame_inv,
        "mean_frame_invariance": float(np.mean(list(frame_inv.values()))) if frame_inv else None,
    }


def run_backbone(backbone, images, llava_id, device, batch_size):
    agent = _build_backbone(backbone, llava_id, use_language=True, device=device)
    feats = {name: encode_all(agent, images, text, batch_size, device)
             for name, text in PROMPTS.items()}
    dim = next(iter(feats.values())).shape[1]
    half = dim // 2  # [image_half || text_half], equal halves for both backbones
    text = {p: f[:, half:] for p, f in feats.items()}
    image = {p: f[:, :half] for p, f in feats.items()}
    del agent
    if device == "cuda":
        th.cuda.empty_cache()
    return {
        "feature_dim": dim,
        "half_dim": half,
        "text_half": half_stats(text),
        "image_half": half_stats(image),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=["llava", "clip", "both"], default="both")
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--data-dir", default="./trajectories")
    ap.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", default="evaluations/paper/prompt_swap_probe.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[probe] device={device}  n_frames={args.n_frames}  backbone={args.backbone}")
    images = load_frames(args.data_dir, args.n_frames)

    backbones = ["clip", "llava"] if args.backbone == "both" else [args.backbone]
    out = {"n_frames": len(images), "prompts": PROMPTS, "backbones": {}}
    for bb in backbones:
        print(f"\n[probe] === {bb} ===")
        out["backbones"][bb] = run_backbone(bb, images, args.llava_id, device, args.batch_size)

    # ---- report ----
    print("\n## Prompt-swap probe (pooled text-half movement across prompts)\n")
    print("| backbone | text mean-pair cosine | text mean-pair relL2 | text frame-inv | image mean-pair cosine |")
    print("|----------|----------------------:|---------------------:|---------------:|-----------------------:|")
    for bb, r in out["backbones"].items():
        t, im = r["text_half"], r["image_half"]
        print(f"| {bb} | {t['mean_pair_cosine']:.3f} | {t['mean_pair_rel_l2']:.3f} | "
              f"{t['mean_frame_invariance']:.3f} | {im['mean_pair_cosine']:.3f} |")
    print("\nInterpretation: lower text cosine / higher text relL2 = the prompt moves "
          "the text-half more (CLIP). text frame-inv < 1 = text-half is image-entangled "
          "(LLaVA). image cosine < 1 = prompt leaks into the image-half (the confound).\n")
    print("### text-half: chop-vs-empty (C|A) and dirt-vs-empty (D|A)\n")
    for bb, r in out["backbones"].items():
        pw = r["text_half"]["pairwise"]
        ca = pw.get("A_empty|C_chop", {})
        da = pw.get("A_empty|D_dirt", {})
        print(f"- {bb}: C|A cosine={ca.get('cosine'):.3f} relL2={ca.get('rel_l2'):.3f}  "
              f"D|A cosine={da.get('cosine'):.3f} relL2={da.get('rel_l2'):.3f}")

    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outp, "w"), indent=2)
        print(f"\n[wrote {outp}]")


if __name__ == "__main__":
    main()
