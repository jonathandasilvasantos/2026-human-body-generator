"""Procedural skinned mesh: one capsule per bone, merged into a single VBO.

Each vertex carries two bone indices and a 2-weight vector (summing to 1).
Vertices near a bone's head smoothstep-blend to the parent bone, so joints
bend without the usual LBS candy-wrapper artefact.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from . import mathx
from . import primitives as prim


def _find(bones, name):
    for i, b in enumerate(bones):
        if b[0] == name:
            return i
    return -1


@dataclass
class SkinnedMesh:
    positions: np.ndarray   # (N, 3) float32, bone-local
    normals:   np.ndarray   # (N, 3) float32
    bones:     np.ndarray   # (N, 2) int32
    weights:   np.ndarray   # (N, 2) float32, rows sum to 1
    indices:   np.ndarray   # (M,)   uint32


def _capsule(bone_index, parent_index, length, radius, radial=16, rings=6,
             top_cap_scale=1.0, bottom_cap_scale=1.0):
    """Capsule along +Y from y=0 to y=length in bone-local space.

    ``top_cap_scale`` / ``bottom_cap_scale`` shrink the hemisphere caps
    vertically without affecting the cylinder radius, useful for the
    chest (whose top should not balloon over the neck).
    """
    verts, norms, b_a, b_b, w = [], [], [], [], []

    def add(p, n, wa):
        verts.append(p)
        norms.append(n / (np.linalg.norm(n) + 1e-8))
        b_a.append(bone_index)
        b_b.append(parent_index if parent_index >= 0 else bone_index)
        w.append((wa, 1.0 - wa))

    blend_zone = 0.35 * max(length, 1e-3)
    has_parent = parent_index >= 0

    # bottom hemisphere (y <= 0)
    for i in range(rings + 1):
        phi = (math.pi * 0.5) * (i / rings)
        y = -math.sin(phi) * radius * bottom_cap_scale
        rr = math.cos(phi) * radius
        wa = 0.0 if has_parent else 1.0
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            x, z = math.cos(th) * rr, math.sin(th) * rr
            add(np.array([x, y, z], np.float32), np.array([x, y, z], np.float32), wa)

    # cylinder
    cyl_rings = max(2, rings)
    for i in range(cyl_rings + 1):
        y = (i / cyl_rings) * length
        if not has_parent:
            wa = 1.0
        else:
            t = min(1.0, y / max(blend_zone, 1e-4))
            wa = t * t * (3 - 2 * t)  # smoothstep
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            x, z = math.cos(th) * radius, math.sin(th) * radius
            add(np.array([x, y, z], np.float32), np.array([x, 0.0, z], np.float32), wa)

    # top hemisphere
    for i in range(rings + 1):
        phi = (math.pi * 0.5) * (i / rings)
        y = length + math.sin(phi) * radius * top_cap_scale
        rr = math.cos(phi) * radius
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            x, z = math.cos(th) * rr, math.sin(th) * rr
            add(np.array([x, y, z], np.float32), np.array([x, y - length, z], np.float32), 1.0)

    total_rings = (rings + 1) + (cyl_rings + 1) + (rings + 1)
    indices = []
    for r in range(total_rings - 1):
        for j in range(radial):
            a = r * radial + j
            b = r * radial + (j + 1) % radial
            c = (r + 1) * radial + j
            d = (r + 1) * radial + (j + 1) % radial
            indices.extend([a, c, b, b, c, d])

    return (
        np.asarray(verts, np.float32),
        np.asarray(norms, np.float32),
        np.asarray(b_a, np.int32),
        np.asarray(b_b, np.int32),
        np.asarray(w, np.float32),
        np.asarray(indices, np.uint32),
    )


def _profiled_capsule(bone_index, parent_index, length, radius, profile,
                      radial=16, rings=6, top_cap_scale=1.0,
                      bottom_cap_scale=1.0):
    """Capsule with an anatomical radius profile along the bone.

    ``profile(t)`` returns a radius multiplier for ``t`` in [0, 1], where
    t=0 is the proximal joint and t=1 is the distal joint. This keeps the
    cheap capsule topology but avoids perfectly cylindrical limbs.
    """
    verts, norms, b_a, b_b, w = [], [], [], [], []

    def add(p, n, wa):
        verts.append(p)
        norms.append(n / (np.linalg.norm(n) + 1e-8))
        b_a.append(bone_index)
        b_b.append(parent_index if parent_index >= 0 else bone_index)
        w.append((wa, 1.0 - wa))

    blend_zone = 0.35 * max(length, 1e-3)
    has_parent = parent_index >= 0
    r0 = radius * profile(0.0)
    r1 = radius * profile(1.0)

    # bottom hemisphere
    for i in range(rings + 1):
        phi = (math.pi * 0.5) * (i / rings)
        y = -math.sin(phi) * r0 * bottom_cap_scale
        rr = math.cos(phi) * r0
        wa = 0.0 if has_parent else 1.0
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            x, z = math.cos(th) * rr, math.sin(th) * rr
            add(np.array([x, y, z], np.float32), np.array([x, y, z], np.float32), wa)

    # profiled cylinder
    cyl_rings = max(3, rings + 2)
    for i in range(cyl_rings + 1):
        t = i / cyl_rings
        y = t * length
        rr = radius * profile(t)
        if not has_parent:
            wa = 1.0
        else:
            bt = min(1.0, y / max(blend_zone, 1e-4))
            wa = bt * bt * (3 - 2 * bt)
        # Finite-difference slope for normals on tapered sections.
        t0 = max(0.0, t - 1.0 / cyl_rings)
        t1_ = min(1.0, t + 1.0 / cyl_rings)
        drdy = (radius * profile(t1_) - radius * profile(t0)) / max((t1_ - t0) * length, 1e-5)
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            cp, sp = math.cos(th), math.sin(th)
            x, z = cp * rr, sp * rr
            add(np.array([x, y, z], np.float32),
                np.array([cp, -drdy, sp], np.float32), wa)

    # top hemisphere
    for i in range(rings + 1):
        phi = (math.pi * 0.5) * (i / rings)
        y = length + math.sin(phi) * r1 * top_cap_scale
        rr = math.cos(phi) * r1
        for j in range(radial):
            th = (j / radial) * 2 * math.pi
            x, z = math.cos(th) * rr, math.sin(th) * rr
            add(np.array([x, y, z], np.float32), np.array([x, y - length, z], np.float32), 1.0)

    total_rings = (rings + 1) + (cyl_rings + 1) + (rings + 1)
    indices = []
    for r in range(total_rings - 1):
        for j in range(radial):
            a = r * radial + j
            b = r * radial + (j + 1) % radial
            c = (r + 1) * radial + j
            d = (r + 1) * radial + (j + 1) % radial
            indices.extend([a, c, b, b, c, d])

    return (
        np.asarray(verts, np.float32),
        np.asarray(norms, np.float32),
        np.asarray(b_a, np.int32),
        np.asarray(b_b, np.int32),
        np.asarray(w, np.float32),
        np.asarray(indices, np.uint32),
    )


def _limb_profile(name):
    """Return a radius profile for exposed anatomical skin limbs."""
    if name.startswith("uarm_"):
        return lambda t: 1.00 - 0.22 * t + 0.04 * math.sin(math.pi * t)
    if name.startswith("farm_"):
        return lambda t: 0.98 - 0.30 * t + 0.08 * math.sin(math.pi * t)
    if name.startswith("thigh_"):
        return lambda t: 1.00 - 0.24 * t + 0.04 * math.sin(math.pi * t)
    if name.startswith("shin_"):
        return lambda t: 0.84 - 0.22 * t + 0.30 * math.sin(math.pi * t)
    return None


_GARMENT_CAP_SCALE = {
    # Torso bones: shallow TOP caps (so the shirt doesn't balloon over the
    # neck) and tall BOTTOM caps (shirt + pants overlap at the waist with
    # no visible belt-seam). Pelvis top cap is modest so the pants/shorts
    # stop at hip level and the shirt clearly covers the midriff above.
    "chest":  (0.14, 0.72),
    "spine":  (0.18, 0.76),
    "pelvis": (0.22, 0.48),
    # Feet: shrink the HEEL cap so the shoe doesn't bulge up the shin.
    "foot_L": (0.90, 0.25),
    "foot_R": (0.90, 0.25),
    # Forearm sleeve cuff: the PRE-ROTATION top cap is the one that extends
    # past the wrist in world space (these bones point along -Y, so the
    # capsule is mirrored by align_y_to). Shrinking the top cap stops the
    # longsleeve at the wrist instead of swallowing the hand.
    "farm_L": (0.25, 1.00),
    "farm_R": (0.25, 1.00),
    # Shin cuff: same reasoning. Shrink pre-rotation top cap so the
    # trousers stop at the ankle and the shoe sits underneath cleanly.
    "shin_L": (0.30, 1.00),
    "shin_R": (0.30, 1.00),
}


def build_selected(bones, bone_names, radius_inflate=0.02, length_scale=1.0) -> SkinnedMesh:
    """Build a capsule mesh that only covers the named bones.

    Used for procedural garments: inflate the capsule radius by
    ``radius_inflate`` (meters) so the cloth sits outside the skin, and
    optionally shorten via ``length_scale`` to get sleeve/pants cutoffs.

    This is a light-weight analytic cousin of CAPE's per-vertex displacement
    (Ma et al., CVPR 2020): instead of a learned offset per SMPL vertex,
    we add a capsule-shell offset on chosen bones.
    """
    wanted = set(bone_names)
    chunks_v, chunks_n, chunks_ba, chunks_bb, chunks_w, chunks_i = [], [], [], [], [], []
    offset = 0
    for i, (name, parent, _head, tip, r) in enumerate(bones):
        if name not in wanted:
            continue
        tip_vec = np.asarray(tip, dtype=np.float32)
        full_len = float(np.linalg.norm(tip_vec))
        length = full_len * length_scale
        if length < 1e-5:
            continue
        top_scale, bot_scale = _GARMENT_CAP_SCALE.get(name, (1.0, 1.0))
        v, n, ba, bb, w, idx = _capsule(i, parent, length, r + radius_inflate,
                                         top_cap_scale=top_scale,
                                         bottom_cap_scale=bot_scale)
        R = mathx.align_y_to(tip_vec)
        v = v @ R.T
        n = n @ R.T
        chunks_v.append(v); chunks_n.append(n)
        chunks_ba.append(ba); chunks_bb.append(bb); chunks_w.append(w)
        chunks_i.append(idx + offset)
        offset += v.shape[0]

    # Small shoulder blend piece so tops meet the upper-arm shell without
    # the oversized "puff sleeve" look from the previous version.
    for name, u_idx in (("uarm_L", _find(bones, "uarm_L")),
                        ("uarm_R", _find(bones, "uarm_R"))):
        if name not in wanted or u_idx < 0:
            continue
        _, u_parent, _, u_tip, u_r = bones[u_idx]
        u_tip = np.asarray(u_tip, dtype=np.float32)
        R = mathx.align_y_to(u_tip)
        r_sh = (u_r + radius_inflate) * 0.72
        v, n, ba, bb, w, idx = prim.ellipsoid(
            (0.0, -0.005, 0.0),
            (r_sh * 0.95, r_sh * 0.72, r_sh * 0.90),
            u_idx, u_parent, weight_self=0.72,
            rings=8, radial=12,
        )
        chunks_v.append(v @ R.T); chunks_n.append(n @ R.T)
        chunks_ba.append(ba); chunks_bb.append(bb); chunks_w.append(w)
        chunks_i.append(idx + offset)
        offset += v.shape[0]

    if not chunks_v:
        empty_f32 = np.zeros((0, 3), dtype=np.float32)
        empty_i32 = np.zeros((0, 2), dtype=np.int32)
        return SkinnedMesh(empty_f32, empty_f32, empty_i32,
                           np.zeros((0, 2), dtype=np.float32),
                           np.zeros((0,), dtype=np.uint32))

    positions = np.concatenate(chunks_v, axis=0)
    normals = np.concatenate(chunks_n, axis=0)
    bones_pair = np.stack(
        [np.concatenate(chunks_ba), np.concatenate(chunks_bb)], axis=1
    ).astype(np.int32)
    weights = np.concatenate(chunks_w, axis=0)
    indices = np.concatenate(chunks_i, axis=0)
    return SkinnedMesh(positions, normals, bones_pair, weights, indices)


def build(bones, shape=None) -> SkinnedMesh:
    """Stitch a capsule per bone + compound head + compound hands +
    optional bust/glutes into one merged skinned mesh. The head and hand
    bones get dedicated compound meshes instead of plain capsules."""
    head_idx = _find(bones, "head")
    chest_idx = _find(bones, "chest")
    hand_idx_L = _find(bones, "hand_L")
    hand_idx_R = _find(bones, "hand_R")
    chunks = []

    for i, (name, parent, _, tip, r) in enumerate(bones):
        if i == head_idx or i == hand_idx_L or i == hand_idx_R:
            continue  # compound meshes below
        tip_vec = np.asarray(tip, dtype=np.float32)
        length = float(np.linalg.norm(tip_vec))
        if length < 1e-5:
            continue
        # Torso bones need shallow top/bottom caps so the chest doesn't
        # balloon over the neck and the pelvis doesn't bulge under the
        # thighs. Limbs keep the default rounded caps.
        if name == "chest":
            top_scale, bot_scale = 0.16, 0.30
        elif name == "spine":
            top_scale, bot_scale = 0.20, 0.30
        elif name == "pelvis":
            top_scale, bot_scale = 0.32, 0.32
        elif name in ("foot_L", "foot_R"):
            top_scale, bot_scale = 0.90, 0.25
        else:
            top_scale, bot_scale = 1.0, 1.0
        profile = _limb_profile(name)
        if profile is None:
            v, n, ba, bb, w, idx = _capsule(i, parent, length, r,
                                             top_cap_scale=top_scale,
                                             bottom_cap_scale=bot_scale)
        else:
            v, n, ba, bb, w, idx = _profiled_capsule(
                i, parent, length, r, profile,
                top_cap_scale=top_scale,
                bottom_cap_scale=bot_scale,
            )
        R = mathx.align_y_to(tip_vec)
        v = v @ R.T
        n = n @ R.T
        chunks.append((v, n, ba, bb, w, idx))

    for h_idx, side in ((hand_idx_L, +1), (hand_idx_R, -1)):
        if h_idx < 0:
            continue
        _, parent, _, tip, r = bones[h_idx]
        chunks.extend(_hand_compound(h_idx, parent, np.asarray(tip, np.float32), r, side))

    # Deltoid bulges at each shoulder (rounds the silhouette and avoids the
    # visible wedge between the chest and upper-arm capsules). Slightly
    # smaller than before so garments with standard inflate still cover it.
    for uarm_name in ("uarm_L", "uarm_R"):
        u_idx = _find(bones, uarm_name)
        if u_idx < 0:
            continue
        _, u_parent, _, u_tip, u_r = bones[u_idx]
        u_tip = np.asarray(u_tip, dtype=np.float32)
        R = mathx.align_y_to(u_tip)
        r_shoulder = u_r * 0.98
        v, n, ba, bb, w, idx = prim.ellipsoid(
            (0.0, -0.01, 0.0),
            (r_shoulder * 1.00, r_shoulder * 0.88, r_shoulder * 0.96),
            u_idx, u_parent, weight_self=0.75,
            rings=12, radial=16,
        )
        chunks.append((v @ R.T, n @ R.T, ba, bb, w, idx))

    if head_idx >= 0:
        head_parent = bones[head_idx][1]
        head_tip = np.asarray(bones[head_idx][3], dtype=np.float32)
        head_r = bones[head_idx][4]
        gender = getattr(shape, "gender", "neutral") if shape else "neutral"
        chunks.extend(_head_compound(head_idx, head_parent, head_tip, head_r, gender))

    if shape is not None and getattr(shape, "bust", 0.0) > 0.01 and chest_idx >= 0:
        chunks.extend(_bust(chest_idx, bones[chest_idx], shape.bust))

    if shape is not None and getattr(shape, "gender", "neutral") == "female":
        pelvis_idx = _find(bones, "pelvis")
        if pelvis_idx >= 0:
            chunks.extend(_glutes(pelvis_idx, bones[pelvis_idx],
                                  getattr(shape, "hip_w", 1.0)))

    v, n, ba, bb, w, idx = prim.merge(chunks)
    bones_pair = np.stack([ba, bb], axis=1).astype(np.int32)
    return SkinnedMesh(v, n, bones_pair, w, idx)


# --- compound head -----------------------------------------------------------

# Head anthropometric landmark fractions (y as fraction of head bone length).
# Based on the classical "rule of thirds" vertical facial canon: the distances
# chin->nose-base, nose-base->brow, brow->hairline are approximately equal.
# Eyes sit near the vertical midpoint of the head including skull top.
# References: Leonardo canon; Farkas facial anthropometry; FLAME landmarks.
H_CHIN      = 0.08
H_MOUTH     = 0.28
H_NOSE_BASE = 0.40
H_EYE       = 0.55
H_BROW      = 0.63
H_HAIRLINE  = 0.80
H_TOP       = 1.00


def _head_compound(head_idx, parent_idx, tip, radius, gender):
    """Build a compound head with anthropometrically proportioned features.

    The head bone's local +Y points from the neck joint to the top of the
    skull. We place landmarks as fractions of that length and size features
    from the same length (not from the bone radius, which was too generous
    and gave ping-pong-ball proportions).
    """
    length = float(np.linalg.norm(tip))
    R = mathx.align_y_to(tip) if length > 1e-6 else np.eye(3, dtype=np.float32)

    # Skull dimensions. Real-world adult head: ~23cm tall, ~15cm wide,
    # ~21cm deep -> ratios ~1.0 : 0.65 : 0.90.
    w_factor = 0.34 * (1.06 if gender == "male" else 0.98)
    d_factor = 0.38
    head_w = length * w_factor
    head_d = length * d_factor
    skull_cy = length * (H_TOP + H_CHIN) * 0.5
    skull_cy_offset = length * 0.02           # slight upward push for rounder crown
    skull_r = (head_w, length * (H_TOP - H_CHIN) * 0.5, head_d)

    chunks = [
        prim.ellipsoid((0.0, skull_cy + skull_cy_offset, 0.0), skull_r,
                       head_idx, parent_idx, weight_self=1.0, rings=18, radial=28),
    ]

    # Nose: bridge (longer ellipsoid) + tip (small).
    nose_y_bridge = length * (H_EYE + H_NOSE_BASE) * 0.5
    nose_y_tip    = length * (H_NOSE_BASE + 0.04)
    # Single unified nose: one elongated ellipsoid from brow down to tip,
    # protruding forward. Using one shape avoids the "two bumps" artefact
    # of a separate bridge + tip.
    nose_cy = length * (H_NOSE_BASE + (H_EYE - H_NOSE_BASE) * 0.55)
    chunks.append(prim.ellipsoid(
        (0.0, nose_cy, head_d * 0.95),
        (length * 0.05,
         length * (H_EYE - H_NOSE_BASE) * 0.75,
         length * 0.08),
        head_idx, parent_idx, rings=12, radial=14,
    ))

    # Brow ridge: thin wide ellipsoid just above eye line.
    brow_radii = (head_w * 0.60, length * 0.025, head_d * 0.12)
    brow_z = head_d * 0.80 * (1.05 if gender == "male" else 1.0)
    brow_radii = (brow_radii[0], brow_radii[1] * (1.4 if gender == "male" else 1.0), brow_radii[2])
    chunks.append(prim.ellipsoid(
        (0.0, length * H_BROW, brow_z), brow_radii,
        head_idx, parent_idx, rings=8, radial=18,
    ))

    # Cheekbones: gentle swells flanking the nose.
    cheek_y = length * (H_NOSE_BASE + 0.04)
    cheek_r = (length * 0.055, length * 0.05, length * 0.035)
    chunks += [
        prim.ellipsoid((+head_w * 0.55, cheek_y, head_d * 0.60), cheek_r,
                       head_idx, parent_idx, rings=8, radial=14),
        prim.ellipsoid((-head_w * 0.55, cheek_y, head_d * 0.60), cheek_r,
                       head_idx, parent_idx, rings=8, radial=14),
    ]

    # Chin: small ellipsoid under the mouth to give a chin point.
    chin_radii = (head_w * 0.30, length * 0.05, head_d * 0.35)
    if gender == "male":
        chin_radii = (chin_radii[0] * 1.15, chin_radii[1] * 1.2, chin_radii[2] * 1.05)
    chunks.append(prim.ellipsoid(
        (0.0, length * (H_CHIN + 0.05), head_d * 0.55), chin_radii,
        head_idx, parent_idx, rings=8, radial=14,
    ))

    # Ears: thin, hugging the sides of the skull so they don't read as
    # circular dots on the face from a distance.
    ear_r = (length * 0.012, length * 0.055, length * 0.030)
    chunks += [
        prim.ellipsoid((+head_w * 1.02, length * (H_EYE - 0.02), -head_d * 0.02), ear_r,
                       head_idx, parent_idx, rings=8, radial=12),
        prim.ellipsoid((-head_w * 1.02, length * (H_EYE - 0.02), -head_d * 0.02), ear_r,
                       head_idx, parent_idx, rings=8, radial=12),
    ]

    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- female bust -------------------------------------------------------------

def _bust(chest_idx, chest_bone, amount):
    """Two small ellipsoids attached to the chest bone for female characters."""
    _, _, _, tip, radius = chest_bone
    tip_vec = np.asarray(tip, dtype=np.float32)
    R = mathx.align_y_to(tip_vec)
    length = float(np.linalg.norm(tip_vec))
    sep = radius * 0.45
    y = length * 0.55
    z = radius * 0.55
    size = radius * (0.18 + 0.18 * amount)
    rxyz = (size * 1.05, size * 0.85, size)
    chunks = [
        prim.ellipsoid((+sep, y, z), rxyz, chest_idx, -1, weight_self=1.0),
        prim.ellipsoid((-sep, y, z), rxyz, chest_idx, -1, weight_self=1.0),
    ]
    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- hands -------------------------------------------------------------------

def _hand_compound(hand_idx, parent_idx, tip, radius, side: int):
    """Palm + thumb + fused finger block, all rigidly skinned to the hand bone.

    Built in the bone's pre-rotation frame (+Y along tip direction). The
    ``side`` argument is +1 for the left hand, -1 for the right so the thumb
    ends up on anatomically matching sides after R is applied.

    Inspired by MANO (Romero et al., SIGGRAPH Asia 2017) but collapsed to a
    single rigid pose -- finger articulation would require child bones per
    finger joint (not in this rig).
    """
    length = float(np.linalg.norm(tip))
    if length < 1e-5:
        return []
    R = mathx.align_y_to(tip)

    palm_len   = length * 0.68
    hand_w     = radius * 1.18
    palm_t     = radius * 0.52
    finger_r   = radius * 0.18

    # The hand bone's tip points along -Y (it hangs from the wrist), so
    # ``align_y_to`` produces R = diag(1, -1, -1) -- it flips Y and Z from
    # the pre-rotation frame. Account for that when placing the thumb and
    # finger centres so the final world-space orientation is anatomical:
    #   - thumb sits on the INNER side of the hand (toward body midline)
    #     and on the PALM FRONT (character's +Z in rest pose)
    #   - fingers extend downward from the palm tip (no Z offset) and are
    #     slightly cupped forward
    chunks = [
        # Palm: wide + thin (no Z offset, centred on the palm plane).
        prim.ellipsoid(
            (0.0, palm_len * 0.26, 0.0),
            (hand_w * 1.02, palm_len * 0.48, palm_t * 1.02),
            hand_idx, parent_idx, weight_self=0.85,
            rings=10, radial=16,
        ),
        # Thumb: pre-R X is NEGATIVE on the side multiplier so that after
        # the R flip (which leaves X untouched) the thumb ends up medial.
        # Pre-R Z is NEGATIVE so after flip it lands at +Z (palm front).
        prim.ellipsoid(
            (-hand_w * 0.74 * side, palm_len * 0.24, -palm_t * 0.22),
            (finger_r * 1.05, length * 0.10, finger_r),
            hand_idx, parent_idx, rings=8, radial=10,
        ),
    ]

    chunks.append(prim.ellipsoid(
        (0.0, palm_len * 0.66, -palm_t * 0.04),
        (hand_w * 0.84, length * 0.16, finger_r * 1.15),
        hand_idx, parent_idx,
        rings=8, radial=12,
    ))

    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- glutes ------------------------------------------------------------------

def _glutes(pelvis_idx, pelvis_bone, hip_w):
    """Two ellipsoids on the back of the pelvis (female). Larger when hips
    are wider."""
    _, _, _, tip, radius = pelvis_bone
    tip_vec = np.asarray(tip, dtype=np.float32)
    R = mathx.align_y_to(tip_vec)
    length = float(np.linalg.norm(tip_vec))
    sep = radius * 0.50 * hip_w
    y = -length * 0.2       # below pelvis head, at seat level
    z = -radius * 0.55      # behind the body
    size = radius * (0.48 + 0.14 * hip_w)
    rxyz = (size * 0.95, size * 0.85, size)
    chunks = [
        prim.ellipsoid((+sep, y, z), rxyz, pelvis_idx, -1, weight_self=1.0),
        prim.ellipsoid((-sep, y, z), rxyz, pelvis_idx, -1, weight_self=1.0),
    ]
    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- eyes --------------------------------------------------------------------

def _head_info(bones):
    """Shared helper: head bone index, parent, rest rotation, length, depth."""
    head_idx = _find(bones, "head")
    if head_idx < 0:
        return None
    parent = bones[head_idx][1]
    tip = np.asarray(bones[head_idx][3], dtype=np.float32)
    r = bones[head_idx][4]
    R = mathx.align_y_to(tip)
    length = float(np.linalg.norm(tip))
    head_d = length * 0.38
    head_w = length * 0.34
    return head_idx, parent, R, length, r, head_w, head_d


def build_eyes(bones) -> SkinnedMesh:
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    # Eye line is at y = length*H_EYE; eyes sit on the front of the skull.
    sep  = length * 0.115
    y    = length * H_EYE
    z    = head_d * 0.78
    eye_r = length * 0.060
    radii = (eye_r, eye_r * 0.95, eye_r * 0.9)
    chunks = [
        prim.ellipsoid((+sep, y, z), radii, head_idx, parent, rings=12, radial=18),
        prim.ellipsoid((-sep, y, z), radii, head_idx, parent, rings=12, radial=18),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_iris(bones) -> SkinnedMesh:
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep  = length * 0.115
    y    = length * H_EYE
    z    = head_d * 0.78
    eye_r = length * 0.060
    # iris disc sits on the front surface of the eye white
    iris_r = eye_r * 0.55
    radii = (iris_r, iris_r, iris_r * 0.3)
    chunks = [
        prim.ellipsoid((+sep, y, z + eye_r * 0.75), radii, head_idx, parent,
                       rings=8, radial=14),
        prim.ellipsoid((-sep, y, z + eye_r * 0.75), radii, head_idx, parent,
                       rings=8, radial=14),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_pupils(bones) -> SkinnedMesh:
    """Small dark pupil in the centre of each iris."""
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep  = length * 0.115
    y    = length * H_EYE
    z    = head_d * 0.78
    eye_r = length * 0.060
    pr = eye_r * 0.22
    radii = (pr, pr, pr * 0.3)
    chunks = [
        prim.ellipsoid((+sep, y, z + eye_r * 0.85), radii, head_idx, parent,
                       rings=6, radial=10),
        prim.ellipsoid((-sep, y, z + eye_r * 0.85), radii, head_idx, parent,
                       rings=6, radial=10),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_eyelids(bones) -> SkinnedMesh:
    """Skin-colored upper/lower eyelid folds that mask the spherical eyeballs.

    This keeps the existing simple eye spheres, but exposes them through a
    narrower almond aperture instead of showing full round balls.
    """
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep = length * 0.115
    y = length * H_EYE
    z = head_d * 0.78
    eye_r = length * 0.060

    # Slightly in front of the sclera/iris so depth testing naturally hides
    # the upper/lower poles of the eye sphere.
    lid_z = z + eye_r * 0.92
    hx = length * 0.070
    upper_hy = length * 0.016
    lower_hy = length * 0.010
    upper_y = y + eye_r * 0.64
    lower_y = y - eye_r * 0.58

    chunks = []
    for side in (+1.0, -1.0):
        x = side * sep
        chunks.append(prim.flat_patch(
            (x, upper_y, lid_z),
            (hx, upper_hy),
            head_idx, parent,
            normal=(0.0, 0.10, 1.0),
            subdiv=(8, 2),
            thickness=0.0025,
        ))
        chunks.append(prim.flat_patch(
            (x, lower_y, lid_z - eye_r * 0.04),
            (hx * 0.92, lower_hy),
            head_idx, parent,
            normal=(0.0, -0.08, 1.0),
            subdiv=(8, 2),
            thickness=0.002,
        ))

    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_lips(bones) -> SkinnedMesh:
    """Upper + lower lip as two thin squashed ellipsoids on the face."""
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    y_mouth = length * H_MOUTH
    z = head_d * 0.88
    # upper lip: wider + slight dip
    # Thin, mostly-flat lips pressed against the face plane. Keeping both
    # the vertical half-extent (ry) small and the depth (rz) shallow so the
    # lips read as a mouth line rather than two stacked domes.
    upper = prim.ellipsoid(
        (0.0, y_mouth + length * 0.012, z),
        (length * 0.09, length * 0.008, length * 0.012),
        head_idx, parent, rings=5, radial=18,
    )
    lower = prim.ellipsoid(
        (0.0, y_mouth - length * 0.006, z),
        (length * 0.085, length * 0.010, length * 0.014),
        head_idx, parent, rings=5, radial=18,
    )
    chunks = [upper, lower]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def _empty_mesh() -> SkinnedMesh:
    z_f = np.zeros((0, 3), np.float32)
    return SkinnedMesh(
        z_f, z_f,
        np.zeros((0, 2), np.int32),
        np.zeros((0, 2), np.float32),
        np.zeros((0,), np.uint32),
    )
