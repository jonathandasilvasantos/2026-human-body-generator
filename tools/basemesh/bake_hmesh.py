"""Stage 4 — convert a cleaned, rigged GLB into a portable .hmesh.

Reads the cleaned GLB (Mixamo-rigged, single mesh, single UV atlas),
extracts:
  - vertex positions, normals, uvs
  - bone hierarchy with bind-local matrices
  - skinning ids/weights (4-influence cap)
and writes the .hmesh format defined in core/human.h via the C engine's
binary writer (we go through Python: build a synthetic Human in memory,
populate fields, then save).

Currently a documented stub — final shape depends on whether we use
pygltflib or trimesh. Decision: use pygltflib (smaller dep, lower-level,
gives direct access to the skin and animation data we care about).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--in",  dest="in_path",  type=Path, default=None)
    ap.add_argument("--out", dest="out_path", type=Path, default=None)
    _args = ap.parse_args()
    raise NotImplementedError(
        "bake_hmesh.py: implement once cleanup.py produces a rigged GLB. "
        "Plan: parse with pygltflib, build a Human-equivalent struct in C "
        "via a new human_create_from_arrays() entry point, then human_save()."
    )


if __name__ == "__main__":
    raise SystemExit(main())
