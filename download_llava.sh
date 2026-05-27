#!/bin/bash
# Pre-download LLaVA weights into the HF cache so the first SLURM job
# doesn't spend an hour pulling from huggingface.co on the compute node.
# Run from the login node (it has internet; compute nodes do too on LMU CIP,
# so this script is optional — it just avoids the first-job download hit).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/BIG}"
HF_HOME="${HF_HOME:-/var/tmp1/$USER/hf_cache}"
MODEL="${LLAVA_MODEL:-llava-hf/llava-1.5-7b-hf}"

echo "Host:    $(hostname)"
echo "Repo:    $REPO_ROOT"
echo "HF_HOME: $HF_HOME"
echo "Model:   $MODEL"

if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "ERROR: venv not found at $REPO_ROOT/.venv" >&2
    echo "Create it once:  python3.11 -m venv $REPO_ROOT/.venv && source $REPO_ROOT/.venv/bin/activate && pip install -r $REPO_ROOT/requirements.txt" >&2
    exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

if ! curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    echo "ERROR: huggingface.co not reachable from $(hostname)" >&2
    exit 1
fi

echo "Downloading $MODEL into $HF_HOME (~13 GB, takes 10-15 min) …"
python - <<PY
from transformers import LlavaProcessor, LlavaForConditionalGeneration
LlavaProcessor.from_pretrained("$MODEL")
LlavaForConditionalGeneration.from_pretrained("$MODEL")
print("OK")
PY

echo "Done. Cache size: $(du -sh "$HF_HOME" | cut -f1)"
