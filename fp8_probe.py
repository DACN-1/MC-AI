"""Measure fp8 vs fp16 throughput for the frozen LLaVA forward on Blackwell.
Drops A5000 fp16 parity (cluster retired) — all cells would use fp8 consistently."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import torch as th
import numpy as np
from PIL import Image
from feature_cache import _build_backbone


def timed(agent, imgs, texts, iters=10):
    with th.no_grad():
        agent.encode(imgs, texts)  # warmup (kernels, autotune)
        agent.encode(imgs, texts)
        th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            feat = agent.encode(imgs, texts)
        th.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, feat


B = 24
imgs = [Image.fromarray((np.random.rand(640, 640, 3) * 255).astype("uint8")) for _ in range(B)]
texts = ["chop a tree"] * B

agent = _build_backbone("llava", "llava-hf/llava-1.5-7b-hf", True, "cuda")
t16, f16 = timed(agent, imgs, texts)
print(f"fp16: {B / t16:5.1f} samples/sec  ({1000 * t16 / B:5.1f} ms/sample)", flush=True)

# fp8 dynamic activation + weight on the 7B language model (the compute bottleneck)
from torchao.quantization import quantize_
try:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    cfg = Float8DynamicActivationFloat8WeightConfig()
except ImportError:
    from torchao.quantization import float8_dynamic_activation_float8_weight
    cfg = float8_dynamic_activation_float8_weight()

lm = getattr(agent.llava, "language_model", None) or agent.llava.model.language_model
quantize_(lm, cfg)
print("quantized language model to fp8", flush=True)

t8, f8 = timed(agent, imgs, texts)
print(f"fp8 : {B / t8:5.1f} samples/sec  ({1000 * t8 / B:5.1f} ms/sample)   SPEEDUP {t16 / t8:.2f}x", flush=True)

# how different are the features? (cosine sim of fp16 vs fp8 pooled feature)
cos = th.nn.functional.cosine_similarity(f16.float(), f8.float(), dim=-1).mean().item()
print(f"feature cosine-sim fp16 vs fp8: {cos:.4f}  (dim={f8.shape[-1]})", flush=True)
free, total = th.cuda.mem_get_info()
print(f"VRAM after fp8: {(total - free) / 1024**3:.1f}/{total / 1024**3:.1f} GB at B={B}", flush=True)
