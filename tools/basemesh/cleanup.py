"""Stage 3 — Blender headless cleanup + Mixamo armature bind.

Run via:
  blender -b -P tools/basemesh/cleanup.py -- --slug adult_m

Steps:
  1. Import the raw Hy3D GLB.
  2. Center on origin, align bbox upright (assume +Y up, facing +Z).
  3. Decimate to the archetype's target triangle budget.
  4. Symmetrize across X (Hy3D output is mostly symmetric but not exactly).
  5. Smart UV unwrap.
  6. Build a Mixamo-named armature at canonical fractional positions of bbox.
  7. Parent mesh to armature with automatic envelope weights.
  8. Export cleaned GLB to assets/basemesh/cleaned/<slug>.glb.

The auto-rig is approximate: heads at canonical Y fractions of the bbox.
Operator should open the result in Blender and tweak bone positions before
running bake_hmesh.py. A real auto-rigger (landmark detection or ML) would
replace step 6 in a future iteration.

This file is *not* importable outside Blender — it relies on bpy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- guard: only run inside Blender ---
try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:
    print("ERROR: this script must be run with `blender -b -P ...`", file=sys.stderr)
    sys.exit(2)


# ---------- repo-relative imports without packaging ----------
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))
from tools.basemesh.archetypes import by_slug, Archetype  # noqa: E402
from tools.basemesh.mixamo_skeleton import BONES         # noqa: E402


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--in",  dest="in_path",  type=Path, default=None)
    ap.add_argument("--out", dest="out_path", type=Path, default=None)
    ap.add_argument("--no-decimate", action="store_true")
    ap.add_argument("--no-symmetrize", action="store_true")
    return ap.parse_args(argv)


def import_glb(path: Path) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh in GLB")
    if len(meshes) > 1:
        # join everything into one
        for o in meshes[1:]:
            o.select_set(True)
        meshes[0].select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "BaseMesh"
    return obj


def center_on_origin(obj: bpy.types.Object) -> None:
    """Translate so feet sit on Y=0 and the figure is centered on X=Z=0."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    # axis-aligned bbox in world space
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
    cx = (min(xs) + max(xs)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    feet_y = min(ys)
    obj.location.x -= cx
    obj.location.y -= feet_y
    obj.location.z -= cz
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def decimate_to(obj: bpy.types.Object, target_tris: int) -> None:
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    n = sum(len(p.vertices) - 2 for p in obj.data.polygons)  # tri-equivalent
    if n <= 0:
        return
    mod.ratio = max(0.05, min(1.0, target_tris / max(1, n)))
    bpy.ops.object.modifier_apply(modifier=mod.name)


def symmetrize_x(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.symmetrize(direction="POSITIVE_X")
    bpy.ops.object.mode_set(mode="OBJECT")


def smart_uv(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def build_armature(obj: bpy.types.Object) -> bpy.types.Object:
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    ys = [v.y for v in bb]; xs = [v.x for v in bb]
    y0, y1 = min(ys), max(ys)
    height = y1 - y0
    half_w = (max(xs) - min(xs)) * 0.5
    if height <= 0 or half_w <= 0:
        raise RuntimeError("degenerate bbox")

    arm_data = bpy.data.armatures.new("Armature")
    armature = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    created: dict[str, bpy.types.EditBone] = {}
    for b in BONES:
        eb_obj = eb.new(b.name)
        eb_obj.head = Vector((b.head_x * half_w, y0 + b.head_y * height, 0.0))
        eb_obj.tail = Vector((b.tail_x * half_w, y0 + b.tail_y * height, 0.0))
        if b.parent and b.parent in created:
            eb_obj.parent = created[b.parent]
        created[b.name] = eb_obj
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def parent_with_auto_weights(obj: bpy.types.Object, armature: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def export_glb(obj: bpy.types.Object, armature: bpy.types.Object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=False,
        export_yup=True,
        export_skins=True,
        export_animations=False,
    )


def run(slug: str, in_path: Path, out_path: Path, *, no_decimate: bool, no_symmetrize: bool) -> int:
    arch = by_slug(slug)
    obj = import_glb(in_path)
    center_on_origin(obj)
    if not no_decimate:
        decimate_to(obj, arch.target_tris)
    if not no_symmetrize:
        symmetrize_x(obj)
    smart_uv(obj)
    armature = build_armature(obj)
    parent_with_auto_weights(obj, armature)
    export_glb(obj, armature, out_path)
    print(f"OK {slug} -> {out_path}")
    return 0


def main() -> int:
    args = parse_argv()
    in_path  = args.in_path  or Path(f"assets/basemesh/raw_glb/{args.slug}_raw.glb")
    out_path = args.out_path or Path(f"assets/basemesh/cleaned/{args.slug}.glb")
    return run(args.slug, in_path, out_path,
               no_decimate=args.no_decimate, no_symmetrize=args.no_symmetrize)


if __name__ == "__main__":
    sys.exit(main())
