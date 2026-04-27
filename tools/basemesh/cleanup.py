"""Stage 3 — Blender headless cleanup + Mixamo skeleton bind.

Run this *inside* Blender (4.0+) headless:

  blender -b -P tools/basemesh/cleanup.py -- --slug adult_m

Steps performed:
  1. Import the raw Hy3D GLB.
  2. Decimate to the archetype's target tri budget (preserve UV seams).
  3. Symmetrize across X (Hy3D output is mostly symmetric but not exactly).
  4. Auto-rig to the standard Mixamo skeleton (heuristic landmarks + IK).
  5. UV unwrap (smart project, single atlas).
  6. Export cleaned GLB to assets/basemesh/cleaned/<slug>.glb.

The Mixamo skeleton template lives at assets/basemesh/mixamo_template.blend
and must be present.

This file is *not* importable as a Python module outside of Blender — it
relies on bpy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- guard: only run inside Blender ---
try:
    import bpy  # type: ignore[import-not-found]
except ImportError:
    print("ERROR: this script must be run with `blender -b -P ...`", file=sys.stderr)
    sys.exit(2)


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--in",  dest="in_path",  type=Path, default=None)
    ap.add_argument("--out", dest="out_path", type=Path, default=None)
    return ap.parse_args(argv)


def main() -> int:
    args = parse_argv()
    in_path  = args.in_path  or Path(f"assets/basemesh/raw_glb/{args.slug}_raw.glb")
    out_path = args.out_path or Path(f"assets/basemesh/cleaned/{args.slug}.glb")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(in_path))

    # TODO(human v2 phase 4): implement decimate + symmetrize + Mixamo rig.
    # Keeping this as a documented stub until first archetype lands.
    raise NotImplementedError(
        "cleanup.py: implement decimate + symmetrize + Mixamo bind once "
        "the first raw GLB exists. See notes/v2_rewrite_plan.md."
    )


if __name__ == "__main__":
    sys.exit(main())
