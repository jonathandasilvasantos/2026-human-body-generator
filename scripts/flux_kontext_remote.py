"""Local CLI for Stage B (procedural -> photoreal) using FLUX.1 Kontext [dev]
running on the bender remote (RTX 3090 Ti, 24GB).

Drop-in replacement for the deprecated scripts/nano_banana_pro.py. Same
positional shape:

    python3 scripts/flux_kontext_remote.py \\
        --image  screenshots/triview/cN_front.png \\
        --output screenshots/triview/cN_front_photoreal.png \\
        --prompt "Transform into photorealistic person, same pose..."

Behind the scenes: scp the input to bender, ssh into the `flux` conda env to
run remote/flux_kontext_infer.py (installed at
/home/bender/projects/flux_kontext/infer.py), scp the result back.

bender host details mirror the existing Hunyuan3D Stage C wiring; see
enhance-pipeline.txt for SSH / install context.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import uuid


BENDER = os.environ.get("BENDER_HOST", "bender@192.168.15.99")
REMOTE_DIR = "/home/bender/projects/flux_kontext"


def _run(cmd: list[str]):
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="local procedural PNG to edit")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True, help="local destination PNG")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"input image not found: {args.image}")

    tag = uuid.uuid4().hex[:10]
    remote_in = f"/tmp/flux_in_{tag}.png"
    remote_out = f"/tmp/flux_out_{tag}.png"

    try:
        _run(["scp", args.image, f"{BENDER}:{remote_in}"])

        remote_cmd = (
            "source /home/bender/miniconda3/etc/profile.d/conda.sh && "
            "conda activate flux && "
            f"cd {shlex.quote(REMOTE_DIR)} && "
            "HF_TOKEN=$(cat ~/.cache/huggingface/token) "
            "PYTHONNOUSERSITE=1 "
            "python infer.py "
            f"--image {shlex.quote(remote_in)} "
            f"--output {shlex.quote(remote_out)} "
            f"--prompt {shlex.quote(args.prompt)} "
            f"--seed {args.seed} --steps {args.steps} --guidance {args.guidance} "
            f"--height {args.height} --width {args.width}"
        )
        _run(["ssh", BENDER, remote_cmd])

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        _run(["scp", f"{BENDER}:{remote_out}", args.output])
        print(f"wrote {args.output}")
    finally:
        # best-effort cleanup of remote temp files
        subprocess.run(
            ["ssh", BENDER, f"rm -f {shlex.quote(remote_in)} {shlex.quote(remote_out)}"],
            check=False,
        )


if __name__ == "__main__":
    main()
