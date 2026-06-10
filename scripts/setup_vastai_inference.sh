#!/usr/bin/env bash
# Minimal inference-only stack for a rented CUDA box (vast.ai or similar).
# Installs just what `inference_server.py` needs to host a VLAAgent forward —
# no MineRL, no decord, no flash-attn, no training deps. sdpa attention is
# fine at batch=1, so we deliberately skip the flash-attn wheel (none for
# Volta T4 / sm_75 in our pinned set anyway, and Ampere/Ada gain little from
# FA2 at batch=1).
#
# Works on any vast.ai "PyTorch (Vast)" or "NVIDIA CUDA" template whose driver
# reports Max CUDA >= 12.1. Tested target: RTX 3090 / A4000 / L4 / T4.
#
# Idempotent. Env knobs:
#   PYTHON_BIN=python3       (default python3; 3.10-3.12 fine)
#   CREATE_VENV=1            isolate into $REPO_ROOT/.venv
# Usage:  bash scripts/setup_vastai_inference.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
CU_INDEX="https://download.pytorch.org/whl/cu121"

echo "=== GPU / driver ==="
nvidia-smi || { echo "nvidia-smi failed — no GPU visible. Aborting."; exit 1; }

echo
echo "=== Python ==="
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
"$PY" -m pip install --upgrade --ignore-installed pip setuptools wheel

echo
echo "=== torch (cu121; sdpa attention is enough for batch=1 inference) ==="
if "$PY" - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available() and torch.version.cuda is not None
    print("existing torch", torch.__version__, "| cuda", torch.version.cuda, "| usable:", ok)
    sys.exit(0 if ok else 1)
except Exception as e:
    print("no usable torch yet:", e)
    sys.exit(1)
PY
then
    echo "Reusing preinstalled torch."
else
    echo "Installing torch 2.4 from cu121 index..."
    "$PY" -m pip install "torch>=2.2,<2.5" torchvision torchaudio --index-url "$CU_INDEX"
fi

echo
echo "=== inference-only deps (transformers + image loaders, no flash-attn) ==="
"$PY" -m pip install \
    "transformers>=4.45,<4.50" \
    "numpy<2" \
    pillow \
    accelerate \
    sentencepiece \
    safetensors

echo
echo "=== verify ==="
"$PY" - <<'PY'
import torch, transformers, numpy as np, PIL
print("torch        :", torch.__version__, "| cuda", torch.version.cuda)
print("transformers :", transformers.__version__)
print("numpy        :", np.__version__, "(must be <2)")
print("pillow       :", PIL.__version__)
assert torch.cuda.is_available(), "CUDA not available after install"
print("device       :", torch.cuda.get_device_name(0))
print("vram (GB)    :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo
echo "Setup OK. Next steps on this box:"
echo "  1) Upload the trained head:  scp model.pt <user>@<host>:~/r1-va/output/llava_chop_a_tree_lang_stride4/model.pt"
echo "  2) Start server (HF Hub downloads LLaVA-7B base on first run, ~14 GB):"
echo "       $PY inference_server.py \\"
echo "           --model-path output/llava_chop_a_tree_lang_stride4/model.pt \\"
echo "           --device cuda --host 0.0.0.0 --port 8765"
echo "  3) From the Mac, SSH-tunnel + run rollout (see docs/rollouts.md)."
