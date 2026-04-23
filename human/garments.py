"""Procedural garments: tops, bottoms, dresses, shoes.

Design principles (informed by GarmentCode / CAPE / MakeHuman):

- **Tight garments** (t-shirt, tank, long-sleeve, pants, shorts, sneakers)
  are built with :func:`mesh.build_selected` -- capsule shells over the
  bones that the garment covers, inflated slightly outward from the skin.
  They inherit the body's skinning weights, so they deform with the rig.
- **Loose garments** (skirt, dress) are built as **cone shells rigidly
  skinned to the pelvis bone** (dresses additionally cover the upper
  torso with a capsule layer). This matches how a real skirt / A-line
  dress behaves: the skirt follows the pelvis but legs swing freely
  inside. No per-vertex cloth simulation is attempted.

Each builder returns a :class:`mesh.SkinnedMesh`.
"""

import numpy as np

from . import mathx
from . import mesh as mesh_mod
from . import primitives as prim
from .mesh import SkinnedMesh, _empty_mesh, _find


# --- garment catalogs --------------------------------------------------------

TOP_BONE_SETS = {
    # Tops stop at the natural waist instead of wrapping the whole pelvis.
    # That keeps the torso from becoming a single block while bottoms still
    # overlap enough to hide seams in motion.
    "tank":       ["chest", "spine", "pelvis", "clav_L", "clav_R"],
    "tshirt":     ["chest", "spine", "pelvis", "clav_L", "clav_R",
                   "uarm_L", "uarm_R"],
    "longsleeve": ["chest", "spine", "pelvis", "clav_L", "clav_R",
                   "uarm_L", "uarm_R", "farm_L", "farm_R"],
}


BOTTOM_BONE_SETS = {
    "pants":  ["pelvis", "thigh_L", "thigh_R", "shin_L", "shin_R"],
    "shorts": ["pelvis", "thigh_L", "thigh_R"],
}


# --- tight garments (tops + pants/shorts) -----------------------------------

def build_top(bones, style, inflate=0.022, length_scale=1.0) -> SkinnedMesh:
    """T-shirt / tank / long-sleeve. Dresses handled by :func:`build_dress`."""
    if style == "dress":
        return _empty_mesh()
    names = TOP_BONE_SETS.get(style)
    if not names:
        return _empty_mesh()
    return mesh_mod.build_selected(bones, names, inflate, length_scale)


def build_bottom(bones, style, inflate=0.020, length_scale=1.0) -> SkinnedMesh:
    """Pants / shorts. Skirts handled by :func:`build_skirt`."""
    if style in ("none", "skirt"):
        return _empty_mesh()
    names = BOTTOM_BONE_SETS.get(style)
    if not names:
        return _empty_mesh()
    # tuck pants/shorts up slightly to sit below the garment top
    return mesh_mod.build_selected(bones, names, inflate, length_scale)


# --- shoes -------------------------------------------------------------------

def build_shoes(bones, style) -> SkinnedMesh:
    if style == "barefoot":
        return _empty_mesh()
    if style == "sneakers":
        # sneakers are low-cut: modest inflate, barely past the toe
        inflate = 0.010
        length_scale = 1.03
    else:
        # boots are bulkier and have a small shaft that hugs the ankle
        inflate = 0.016
        length_scale = 1.07
    return mesh_mod.build_selected(bones, ["foot_L", "foot_R"], inflate, length_scale)


# --- skirt -------------------------------------------------------------------

def build_skirt(bones, length_frac=0.9, flare=1.5) -> SkinnedMesh:
    """Flared skirt hanging from the pelvis.

    ``length_frac`` is a multiple of thigh length: 0.4 for a mini-skirt,
    1.0 for a knee-length skirt. ``flare`` >1 makes the hem wider than the
    waist.
    """
    pelvis_idx = _find(bones, "pelvis")
    thigh_idx = _find(bones, "thigh_L")
    if pelvis_idx < 0 or thigh_idx < 0:
        return _empty_mesh()

    _, _, _, pelvis_tip, pelvis_r = bones[pelvis_idx]
    _, _, thigh_head, thigh_tip, _ = bones[thigh_idx]

    # Cone top: hug the hips. Slightly wider than pelvis but narrow enough
    # that the arms (at ~pelvis_r + 0.04 from centre) hang outside.
    top_radius = pelvis_r * 0.92
    # Cone bottom: length_frac * thigh length below the hip
    thigh_len = float(np.linalg.norm(np.asarray(thigh_tip, dtype=np.float32)))
    total_drop = (float(np.asarray(pelvis_tip)[1]) * -1.0  # pelvis height component
                  if False else 0.0) + thigh_len * length_frac
    bottom_radius = top_radius * flare
    top_center = (0.0, 0.01, 0.0)

    chunks = [prim.cone_shell(
        top_center, top_radius,
        height=total_drop,
        bottom_radius=bottom_radius,
        bone_index=pelvis_idx,
        parent_index=-1,
        rings=8, radial=28,
    )]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


# --- dress -------------------------------------------------------------------

def build_dress(bones, length_frac=1.0, flare=1.4) -> SkinnedMesh:
    """Dress = tight upper (chest + shoulders) + flared cone starting at
    the natural waist. The cone begins at spine-head level (not above
    pelvis) with a radius matched to the spine capsule, so the waist
    transition is width-continuous instead of showing a belt-like seam.
    """
    # Upper tight piece includes the pelvis so the dress reads as one
    # continuous garment even when the cone shell intersects at the waist.
    upper = mesh_mod.build_selected(
        bones,
        ["chest", "spine", "pelvis", "clav_L", "clav_R"],
        radius_inflate=0.014,
        length_scale=0.98,
    )
    # Custom cone: top at waist (y = spine head ~0.08), matching spine
    # capsule radius so the two surfaces meet flush. Flares to hem.
    pelvis_idx = mesh_mod._find(bones, "pelvis")
    thigh_idx = mesh_mod._find(bones, "thigh_L")
    if pelvis_idx < 0 or thigh_idx < 0:
        return upper
    pelvis_r = bones[pelvis_idx][4]
    thigh_tip = bones[thigh_idx][3]
    thigh_len = float(np.linalg.norm(np.asarray(thigh_tip, dtype=np.float32)))

    waist_y = 0.03           # overlap the upper shell so the dress reads as one piece
    waist_r = pelvis_r * 0.98   # matches the slimmer upper shell better and
                                # inflate, so the cone's top ring lines up
                                # with the capsule silhouette (no seam)
    hem_drop = 0.10 + thigh_len * length_frac
    hem_r = waist_r * flare

    cone = prim.cone_shell(
        (0.0, waist_y, 0.0), waist_r,
        height=hem_drop, bottom_radius=hem_r,
        bone_index=pelvis_idx, parent_index=-1,
        rings=10, radial=28,
    )
    lower_parts = [cone]
    lower_v, lower_n, lower_ba, lower_bb, lower_w, lower_idx = prim.merge(lower_parts)
    lower = SkinnedMesh(lower_v, lower_n,
                        np.stack([lower_ba, lower_bb], axis=1).astype(np.int32),
                        lower_w, lower_idx)
    if upper.indices.size == 0 and lower.indices.size == 0:
        return _empty_mesh()

    # Merge
    parts = []
    if upper.indices.size:
        parts.append((upper.positions, upper.normals,
                      upper.bones[:, 0], upper.bones[:, 1],
                      upper.weights, upper.indices))
    if lower.indices.size:
        parts.append((lower.positions, lower.normals,
                      lower.bones[:, 0], lower.bones[:, 1],
                      lower.weights, lower.indices))
    v, n, ba, bb, w, idx = prim.merge(parts)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)
