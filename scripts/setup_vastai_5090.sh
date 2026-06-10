#!/usr/bin/env bash
# Set up the r1v-a training stack on a rented RTX 5090 (Blackwell / sm_120) box.
#
# Blackwell needs torch >= 2.7 + CUDA 12.8 (cu128) — the cluster's pinned
# torch 2.4 / flash-attn wheel has no sm_120 kernel. This installs the additive
# Blackwell path (requirements-blackwell.txt), deliberately WITHOUT flash-attn:
# VLAAgent falls back to sdpa attention.
#
# Compute dtype: VLAAgent auto-defaults to bf16 on sm_120 (override via
# R1VA_LLAVA_DTYPE). bf16's wider exponent stabilises sdpa softmax at the
# 32-batch+ regime where fp16 used to clip — that ~2x speedup over the prior
# fp16+batch-24 setup. Cache storage stays fp16 (feature_cache.py:324), so
# 5090 bf16 builds remain shape-mergeable with A5000 fp16 builds; per-feature
# drift is ~3rd-decimal noise and not material for the downstream BC head.
#
# Works on the vast.ai "PyTorch (Vast)" or "NVIDIA CUDA" templates (or any
# CUDA base whose driver reports Max CUDA >= 12.8 — pip's cu128 torch wheels
# bundle their own CUDA runtime, so only the driver must be new enough).
#
# Idempotent. Env knobs:
#   PYTHON_BIN=python3      python to use (default: python3; 3.10-3.12 all fine)
#   CREATE_VENV=1           isolate into $REPO_ROOT/.venv (default: use container python)
# Usage:   bash scripts/setup_vastai_5090.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
CU_INDEX="https://download.pytorch.org/whl/cu128"

echo "=== GPU / driver ==="
nvidia-smi || { echo "nvidia-smi failed — no GPU visible. Aborting."; exit 1; }

echo
echo "=== Python ==="
# No flash-attn on Blackwell -> the old cp312-only constraint is gone. torch
# cu128, transformers, numpy<2 and decord all ship cp310/311/312 wheels, so any
# Python 3.10-3.12 the image provides works.
PY="${PYTHON_BIN:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: '$PY' not found on this box."; exit 1; }
if [ "${CREATE_VENV:-0}" = "1" ]; then
    [ -d "$VENV" ] || "$PY" -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PY=python
fi
"$PY" --version
"$PY" - <<'PY'
import sys
maj, mnr = sys.version_info[:2]
assert maj == 3 and 10 <= mnr <= 12, f"need Python 3.10-3.12, got {maj}.{mnr}"
PY
# Some base images (vast/debian) ship pip/setuptools/wheel that can't be
# uninstalled ("RECORD file not found"). --ignore-installed installs fresh
# copies without trying to remove the debian-managed ones.
"$PY" -m pip install --upgrade --ignore-installed pip setuptools wheel

echo
echo "=== torch (Blackwell / sm_120 needs torch>=2.7 + cu128) ==="
# Reuse a preinstalled torch only if it already sees the GPU at sm_120 (e.g. the
# vast "PyTorch" image with a recent cu128 build); otherwise install torch 2.7.
if "$PY" - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12
    print("existing torch", torch.__version__, "| cuda", torch.version.cuda, "| usable:", ok)
    sys.exit(0 if ok else 1)
except Exception as e:
    print("no usable torch yet:", e)
    sys.exit(1)
PY
then
    echo "Reusing preinstalled torch (already Blackwell-capable)."
else
    echo "Installing torch 2.7 from cu128 index..."
    "$PY" -m pip install "torch>=2.7,<2.9" torchvision torchaudio --index-url "$CU_INDEX"
fi

echo
echo "=== rest of the stack (NO flash-attn — sdpa fallback by design) ==="
"$PY" -m pip install -r "$REPO_ROOT/requirements-blackwell.txt"

echo
echo "=== verify ==="
"$PY" - <<'PY'
import torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
print("torch         :", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device cap    :", cap, "(expect (12, 0) on a 5090)")
assert torch.cuda.is_available(), "CUDA not available after install"
if cap and cap[0] >= 12:
    print(" -> Blackwell sm_120 detected; running sdpa attention (no flash-attn).")
PY

"$PY" - <<'PY'
# decord is the highest-risk wheel on a fresh box — confirm it imports.
import torchvision, transformers, decord, numpy as np
print("transformers  :", transformers.__version__)
print("numpy         :", np.__version__, "(must be <2)")
print("decord        :", getattr(decord, "__version__", "ok (imported)"))
assert np.__version__ < "2", "numpy must be <2 for this stack"
PY

echo
echo "Setup OK. Next: probe throughput before the real run, e.g."
echo "  $PY scripts/probe_5090_throughput.py --data-dir ./trajectories \\"
echo "      --backbone llava --use-language --batch-size 32 --price 0.802"
echo "  (bf16 is the default on sm_120; add --compute-dtype fp16 to A/B against legacy)"
