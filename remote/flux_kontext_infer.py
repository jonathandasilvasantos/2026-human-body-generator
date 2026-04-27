"""FLUX.1 Kontext [dev] image-to-image edit, intended to live at
/home/bender/projects/flux_kontext/infer.py on the bender remote (RTX 3090 Ti, 24GB).

Replaces the Gemini "Nano Banana Pro" call from Stage B of the enhancement
pipeline. Same input/output shape: takes one reference image and an instruction,
writes one edited image.

Run locally on bender; the local machine drives this via
scripts/flux_kontext_remote.py (scp+ssh wrapper).

Usage:
    python infer.py --image cond.png --prompt "..." --output out.png \
        [--seed 42] [--steps 28] [--guidance 2.5] [--height 1024] [--width 1024]
"""

from __future__ import annotations

import argparse
import os

import torch
from diffusers import FluxKontextPipeline
from PIL import Image


_PIPE = None


def _load_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    _PIPE = pipe
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width must be divisible by 16")

    pipe = _load_pipe()
    src = Image.open(args.image).convert("RGB").resize((args.width, args.height))

    out = pipe(
        image=src,
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.save(args.output)
    print(f"wrote {args.output} ({out.width}x{out.height}, seed={args.seed}, steps={args.steps})")


if __name__ == "__main__":
    main()
