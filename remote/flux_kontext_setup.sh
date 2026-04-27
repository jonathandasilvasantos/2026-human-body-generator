#!/usr/bin/env bash
# One-time install of FLUX.1 Kontext [dev] on the bender remote.
#
# Run on bender (192.168.15.99) as user `bender` after copying
# remote/flux_kontext_infer.py to /home/bender/projects/flux_kontext/infer.py.
#
#   ssh bender@192.168.15.99
#   mkdir -p /home/bender/projects/flux_kontext
#   # (scp infer.py and this script in)
#   bash flux_kontext_setup.sh
#
# Notes:
# - black-forest-labs/FLUX.1-Kontext-dev is gated. Accept the license at
#   https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev once with the
#   same HF account whose token is at ~/.cache/huggingface/token.
# - Lives in its own conda env `flux` so it doesn't touch the existing `hy3d`
#   env (which has pinned torch 2.6.0+cu124 for Hunyuan3D-2mv).
# - License is "FLUX.1 [dev] Non-Commercial". Fine for research; if commercial
#   use is needed, swap to Qwen-Image-Edit-2509 (Apache 2.0).

set -euo pipefail

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx flux; then
    conda create -y -n flux python=3.11
fi
conda activate flux

pip install --upgrade pip
pip install \
    "torch==2.4.1" "torchvision==0.19.1" \
    --index-url https://download.pytorch.org/whl/cu124
pip install \
    "diffusers>=0.32" \
    "transformers>=4.45" \
    "accelerate>=1.0" \
    "sentencepiece" \
    "protobuf" \
    "pillow"

# Pre-download the weights so first inference doesn't pay the download cost.
HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")" \
    python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Kontext-dev",
    token=os.environ["HF_TOKEN"],
)
print("FLUX.1-Kontext-dev cached")
PY

echo "flux env ready. Smoke test:"
echo "  conda activate flux"
echo "  cd /home/bender/projects/flux_kontext"
echo "  python infer.py --image sample.png --prompt 'photoreal portrait' --output out.png"
