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


def _merge_meshes(meshes) -> SkinnedMesh:
    parts = []
    for m in meshes:
        if m is None or m.indices.size == 0:
            continue
        parts.append((m.positions, m.normals,
                      m.bones[:, 0], m.bones[:, 1],
                      m.weights, m.indices))
    if not parts:
        return _empty_mesh()
    v, n, ba, bb, w, idx = prim.merge(parts)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def _mesh_from_chunks(chunks) -> SkinnedMesh:
    v, n, ba, bb, w, idx = prim.merge(chunks)
    if idx.size == 0:
        return _empty_mesh()
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


# --- garment catalogs --------------------------------------------------------

TOP_BONE_SETS = {
    # Tops stop at the natural waist instead of wrapping the whole pelvis.
    # That keeps the torso from becoming a single block while bottoms still
    # overlap enough to hide seams in motion. `neck` is included so the
    # collar follows head/neck motion: without it, tilting or rotating the
    # head exposes the underside of the jaw / back of the beard through the
    # chest's static collar dome.
    "tank":       ["chest", "spine", "pelvis", "neck", "clav_L", "clav_R"],
    "tshirt":     ["chest", "spine", "pelvis", "neck", "clav_L", "clav_R",
                   "uarm_L", "uarm_R"],
    "longsleeve": ["chest", "spine", "pelvis", "neck", "clav_L", "clav_R",
                   "uarm_L", "uarm_R", "farm_L", "farm_R"],
}


BOTTOM_BONE_SETS = {
    "pants":  ["pelvis", "thigh_L", "thigh_R", "shin_L", "shin_R"],
    "shorts": ["pelvis", "thigh_L", "thigh_R"],
}


# --- tight garments (tops + pants/shorts) -----------------------------------

def build_top(bones, style, inflate=0.022, length_scale=1.0,
              gender="neutral", shape=None) -> SkinnedMesh:
    """T-shirt / tank / long-sleeve. Dresses handled by :func:`build_dress`."""
    if style == "dress":
        return _empty_mesh()
    names = TOP_BONE_SETS.get(style)
    if not names:
        return _empty_mesh()
    return mesh_mod.build_selected(bones, names, inflate, length_scale,
                                   gender, shape)


# Bone sets for the sleeve underlayer: a thin fabric-colored shell that
# sits just outside the body skin but inside the sleeve. When the sleeve
# momentarily clips during animation (pose interpolation pushing the skin
# past the cloth surface for a frame or two), the pixel shown is still
# garment-colored instead of bare skin, which the eye reads as "cloth
# stretched tight" rather than "body poking through cloth".
_UNDERLAYER_BONES = {
    "tshirt":     ["clav_L", "clav_R", "uarm_L", "uarm_R"],
    "longsleeve": ["clav_L", "clav_R", "uarm_L", "uarm_R",
                   "farm_L", "farm_R"],
}


def build_sleeve_underlayer(bones, style, top_inflate=0.022,
                            length_scale=1.0, gender="neutral") -> SkinnedMesh:
    """Fabric-colored shell nested between skin and sleeve.

    Radius sits at ~35% of the way from skin (r) to sleeve (r+top_inflate),
    so it is hidden under the sleeve in normal poses but catches any
    shoulder/upper-arm clip-through with the shirt's color. Only built for
    t-shirt / long-sleeve tops (tanks and dresses don't cover the arms).
    """
    names = _UNDERLAYER_BONES.get(style)
    if not names:
        return _empty_mesh()
    # Thin shell: enough clearance over skin to avoid z-fighting, but
    # well inside the sleeve surface.
    underlayer_inflate = max(0.004, top_inflate * 0.35)
    # Shrink slightly so the underlayer doesn't poke past the sleeve cuff
    # at wrist/elbow.
    return mesh_mod.build_selected(bones, names, underlayer_inflate,
                                   length_scale * 0.985, gender)


def build_bottom(bones, style, inflate=0.020, length_scale=1.0,
                 gender="neutral") -> SkinnedMesh:
    """Pants / shorts. Skirts handled by :func:`build_skirt`."""
    if style in ("none", "skirt"):
        return _empty_mesh()
    names = BOTTOM_BONE_SETS.get(style)
    if not names:
        return _empty_mesh()
    # tuck pants/shorts up slightly to sit below the garment top
    return mesh_mod.build_selected(bones, names, inflate, length_scale, gender)


# --- bust fabric overlay -----------------------------------------------------

def build_bust_overlay(bones, shape, top_inflate=0.022) -> SkinnedMesh:
    """Two fabric ellipsoids that sit just outside the female bust skin so
    a top stretches over the breast shape instead of letting bare skin
    poke through the chest cylinder.

    Built only when the character has a non-zero bust amount AND a top;
    the caller is expected to gate that. Skinned to the chest bone like
    the bust skin layer so it deforms identically.
    """
    chest_idx = _find(bones, "chest")
    if chest_idx < 0 or shape is None:
        return _empty_mesh()
    bust = float(getattr(shape, "bust", 0.0))
    if bust < 0.01:
        return _empty_mesh()
    proj = float(getattr(shape, "bust_proj", 0.6))

    _, _, _, tip, radius = bones[chest_idx]
    tip_vec = np.asarray(tip, dtype=np.float32)
    R = mathx.align_y_to(tip_vec)
    length = float(np.linalg.norm(tip_vec))
    sep = radius * 0.40
    y = length * 0.55
    # Push slightly closer to the chest centre so the overlay base sits
    # inside the tank surface (less visible seam) while still projecting.
    z = radius * (0.45 + 0.50 * proj)
    base = radius * (0.32 + 0.22 * bust) + max(0.004, top_inflate * 0.60)
    # Wider X base so the overlay fairs into the chest tank silhouette
    # instead of showing a hard ring where ellipsoid meets cylinder.
    rxyz = (base * 1.00, base * 0.80, base * (1.10 + 0.55 * proj))
    chunks = [
        prim.ellipsoid((+sep, y, z), rxyz, chest_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
        prim.ellipsoid((-sep, y, z), rxyz, chest_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx)
              for (v, n, ba, bb, w, idx) in chunks]
    return _mesh_from_chunks(chunks)


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

    band_h = min(0.018, max(0.010, thigh_len * 0.035))
    hem_y = top_center[1] - total_drop

    chunks = [prim.cone_shell(
        top_center, top_radius,
        height=total_drop,
        bottom_radius=bottom_radius,
        bone_index=pelvis_idx,
        parent_index=-1,
        rings=8, radial=28,
    )]
    # Raised waistband and hem add pattern-like garment boundaries and make
    # the otherwise infinitely thin cone shell read as constructed cloth.
    chunks.append(prim.cone_shell(
        (0.0, top_center[1] + band_h * 0.5, 0.0),
        top_radius * 1.035,
        height=band_h,
        bottom_radius=top_radius * 1.02,
        bone_index=pelvis_idx,
        parent_index=-1,
        rings=2, radial=28,
    ))
    chunks.append(prim.cone_shell(
        (0.0, hem_y + band_h * 0.5, 0.0),
        bottom_radius * 1.015,
        height=band_h,
        bottom_radius=bottom_radius * 1.04,
        bone_index=pelvis_idx,
        parent_index=-1,
        rings=2, radial=28,
    ))
    return _mesh_from_chunks(chunks)


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

    band_h = min(0.020, max(0.011, thigh_len * 0.035))
    hem_y = waist_y - hem_drop
    cone = prim.cone_shell(
        (0.0, waist_y, 0.0), waist_r,
        height=hem_drop, bottom_radius=hem_r,
        bone_index=pelvis_idx, parent_index=-1,
        rings=10, radial=28,
    )
    lower = _mesh_from_chunks([
        cone,
        prim.cone_shell(
            (0.0, waist_y + band_h * 0.45, 0.0),
            waist_r * 1.025,
            height=band_h,
            bottom_radius=waist_r * 1.01,
            bone_index=pelvis_idx, parent_index=-1,
            rings=2, radial=28,
        ),
        prim.cone_shell(
            (0.0, hem_y + band_h * 0.5, 0.0),
            hem_r * 1.015,
            height=band_h,
            bottom_radius=hem_r * 1.04,
            bone_index=pelvis_idx, parent_index=-1,
            rings=2, radial=28,
        ),
    ])
    return _merge_meshes([upper, lower])
