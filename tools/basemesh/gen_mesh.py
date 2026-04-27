"""Stage 2 — Hy3D 2.1 on bender turns reference images into raw GLB.

Submits each archetype's front-view reference to a Hy3D 2.1 process running
on the bender host. Back-view is used for cross-validation only (Hy3D takes
a single front image as input).

Operator must:
  1) Have a working SSH config entry `bender` reaching the GPU host.
  2) Have Hy3D 2.1 already installed there (env activates via `human/hy3d-env`).
  3) Have run `gen_reference.py` so reference images exist.

This script is a thin wrapper around `ssh bender ...`. It does not invoke
the Hy3D Python API directly.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from .archetypes import CATALOG

BENDER_HOST = "bender"
BENDER_HY3D_RUN = "~/human/hy3d-env/bin/python -m hy3d.cli generate"


def run_one(slug: str, ref_dir: Path, out_dir: Path) -> int:
    src = ref_dir / f"{slug}_front.png"
    if not src.exists():
        print(f"SKIP {slug}: missing {src}")
        return 1
    remote_in = f"/tmp/human_ref_{slug}.png"
    remote_out = f"/tmp/human_raw_{slug}.glb"
    local_out = out_dir / f"{slug}_raw.glb"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmds = [
        ["scp", str(src), f"{BENDER_HOST}:{remote_in}"],
        ["ssh", BENDER_HOST, f"{BENDER_HY3D_RUN} --image {shlex.quote(remote_in)} --out {shlex.quote(remote_out)} --quality high"],
        ["scp", f"{BENDER_HOST}:{remote_out}", str(local_out)],
    ]
    for c in cmds:
        print("$", " ".join(shlex.quote(x) for x in c))
        rc = subprocess.call(c)
        if rc != 0:
            print(f"FAIL {slug}: rc={rc}")
            return rc
    print(f"OK {slug} -> {local_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-dir", type=Path, default=Path("assets/basemesh/reference"))
    ap.add_argument("--out-dir", type=Path, default=Path("assets/basemesh/raw_glb"))
    ap.add_argument("--archetype", help="only this slug (default: all)")
    args = ap.parse_args()

    targets = [a for a in CATALOG if not args.archetype or a.slug == args.archetype]
    fails = 0
    for a in targets:
        rc = run_one(a.slug, args.ref_dir, args.out_dir)
        fails += (rc != 0)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
