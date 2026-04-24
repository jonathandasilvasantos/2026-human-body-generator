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


def _torso_profile(name, shape):
    """Per-bone radial taper for torso bones so the chest -> waist ->
    pelvis transition is continuous.

    The chest bone tapers from full radius at the top down to roughly the
    waist radius (`shape.waist`) at the bottom; the spine bone holds at
    the waist radius then opens up at the pelvis. Without this profile the
    cylindrical chest mesh ended at full chest radius and the narrow spine
    started immediately after, producing a visible step / "skirt" ring at
    the chest bottom whenever waist < 1.
    """
    if shape is None:
        return None
    waist = float(getattr(shape, "waist", 1.0))
    if name == "chest":
        # Smoothly taper from r at t=0 to ~waist at t=1.
        return lambda t, w=waist: 1.0 - (1.0 - w) * t
    if name == "spine":
        # Narrow at top (waist), open up slightly at the bottom toward the
        # pelvis so the join with the pelvis cap is continuous.
        # spine radius is already `bulk*waist`; profile divides BACK out to
        # land at ~bulk by t=1.
        if waist < 1e-3:
            return None
        return lambda t, w=waist: 1.0 + ((1.0 / w) - 1.0) * t
    return None


def _limb_profile(name, gender="neutral"):
    """Return a radius profile for exposed anatomical skin limbs.

    Profiles encode the *anatomical* radius taper from limb head -> tip
    (t in [0,1]). Sex differences:
      * Male upper-arm and calf carry a stronger muscle belly bulge
        (deltoid+biceps; gastrocnemius) that we model via a larger sin
        amplitude in the middle of the bone.
      * Female thighs taper LESS toward the knee (subcutaneous fat on
        the quads/inner thigh). Female forearms are slightly thinner.
    """
    male = (gender == "male")
    female = (gender == "female")
    if name.startswith("uarm_"):
        # Mid-bone sin bulge: bigger for males (biceps/triceps mass).
        bulge = 0.10 if male else (0.04 if female else 0.05)
        taper = 0.22 if not female else 0.18
        return lambda t, b=bulge, k=taper: 1.00 - k * t + b * math.sin(math.pi * t)
    if name.startswith("farm_"):
        scale = 1.00 if not female else 0.94
        return lambda t, s=scale: s * (0.98 - 0.30 * t + 0.08 * math.sin(math.pi * t))
    if name.startswith("thigh_"):
        # Female thigh: less taper, fuller at the head (gluteal/hip fat)
        # and broader at the knee (quad fat). Male thigh keeps a clearer
        # vastus lateralis bulge mid-thigh.
        if female:
            return lambda t: 1.10 - 0.14 * t + 0.06 * math.sin(math.pi * t)
        if male:
            return lambda t: 1.00 - 0.26 * t + 0.08 * math.sin(math.pi * t)
        return lambda t: 1.00 - 0.24 * t + 0.04 * math.sin(math.pi * t)
    if name.startswith("shin_"):
        # Calf bulge: pronounced + slightly higher for males.
        if male:
            return lambda t: 0.86 - 0.22 * t + 0.36 * math.sin(math.pi * t)
        if female:
            return lambda t: 0.82 - 0.22 * t + 0.24 * math.sin(math.pi * t)
        return lambda t: 0.84 - 0.22 * t + 0.30 * math.sin(math.pi * t)
    return None


_GARMENT_CAP_SCALE = {
    # Torso bones: the top cap on `chest` forms the collar dome of the
    # shirt. Previously this was very shallow (0.14) which produced an
    # almost flat disc; when the character bent forward (bow/punch), the
    # head's large skull ellipsoid punched through this disc and the face
    # rendered on the shirt front. A taller dome (0.55) wraps the base of
    # the neck and stays clear of the head under typical flexion. The
    # bottom cap stays tall so shirt + pants overlap at the waist.
    "chest":  (0.55, 0.72),
    "spine":  (0.18, 0.76),
    "pelvis": (0.22, 0.48),
    # Neck is included in the top bone set so the collar follows the head.
    # Shallow top cap so the collar rim stops below the jawline instead of
    # swallowing the skull; deeper bottom cap blends into the chest dome.
    "neck":   (0.15, 0.60),
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


def build_selected(bones, bone_names, radius_inflate=0.02, length_scale=1.0,
                   gender="neutral", shape=None) -> SkinnedMesh:
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
        # For sleeve/trouser cuff bones, the underlying body limb tapers
        # (see `_limb_profile`). A uniform-radius cloth capsule then ends
        # in a wide "candle" cuff that the much smaller hand/foot pokes
        # out of. Run a matching profiled capsule for those bones so the
        # sleeve hugs the wrist / the trouser leg hugs the ankle.
        body_profile = _limb_profile(name, gender)
        # Cloth covering chest/spine should also follow the torso taper so
        # a cinched waist doesn't show a chest-bottom skirt-ring.
        if body_profile is None:
            body_profile = _torso_profile(name, shape)
        if body_profile is not None and name in ("farm_L", "farm_R",
                                                  "shin_L", "shin_R",
                                                  "uarm_L", "uarm_R",
                                                  "thigh_L", "thigh_R"):
            # Garment profile = body taper, with the extra clearance
            # contributing more at the proximal end (loose where the limb
            # is fat) and tapering near the cuff so the seam meets the
            # wrist/ankle without slack.
            body_r_at = lambda t, p=body_profile, br=r: br * p(t)
            cuff_extra = max(0.004, radius_inflate * 0.35)
            def garment_profile(t, body_r_at=body_r_at, r=r,
                                infl=radius_inflate, cuff=cuff_extra):
                # Linearly fade clearance from full at elbow/knee to cuff
                # at the wrist/ankle so the cloth converges on the limb.
                clearance = infl * (1.0 - t) + cuff * t
                return (body_r_at(t) + clearance) / r
            v, n, ba, bb, w, idx = _profiled_capsule(
                i, parent, length, r, garment_profile,
                top_cap_scale=top_scale,
                bottom_cap_scale=bot_scale,
            )
        elif body_profile is not None and name in ("chest", "spine"):
            # Torso cloth follows the chest -> waist taper at uniform
            # clearance so the bottom of the chest shell meets the spine
            # shell without a step.
            body_r_at = lambda t, p=body_profile, br=r: br * p(t)
            def torso_garment_profile(t, body_r_at=body_r_at, r=r,
                                       infl=radius_inflate):
                return (body_r_at(t) + infl) / r
            v, n, ba, bb, w, idx = _profiled_capsule(
                i, parent, length, r, torso_garment_profile,
                top_cap_scale=top_scale,
                bottom_cap_scale=bot_scale,
            )
        else:
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

    # Shoulder blend piece: must fully envelop the body's deltoid bulge
    # under arbitrary arm rotation. The body deltoid (built in `build`) is
    # an ellipsoid sized r_shoulder=u_r*0.98 with axes (1.00, 0.88, 0.96)
    # offset y=-0.01, skinned weight_self=0.75. To prevent skin pop-through
    # when the arm raises/rotates, the garment ellipsoid here is derived
    # directly from the body deltoid extents + radius_inflate clearance,
    # and uses the *same* uarm/clav blend weight so both deform identically.
    body_r_shoulder = 0.98  # multiplier on u_r used by build()'s deltoid
    body_axes = (1.00, 0.88, 0.96)
    body_offset_y = -0.01
    body_weight_self = 0.75
    for name, u_idx in (("uarm_L", _find(bones, "uarm_L")),
                        ("uarm_R", _find(bones, "uarm_R"))):
        if name not in wanted or u_idx < 0:
            continue
        _, u_parent, _, u_tip, u_r = bones[u_idx]
        u_tip = np.asarray(u_tip, dtype=np.float32)
        R = mathx.align_y_to(u_tip)
        # Body deltoid extents in bone-local space
        body_rx = u_r * body_r_shoulder * body_axes[0]
        body_ry = u_r * body_r_shoulder * body_axes[1]
        body_rz = u_r * body_r_shoulder * body_axes[2]
        # Sleeve must clear body deltoid by full radius_inflate on every axis.
        # The downward (y) axis is where the armpit crease lives, so bias the
        # clearance there: when the arm raises, LBS rotates the deltoid
        # ellipsoid away from the chest and skin pops through the seam on the
        # underside. Extra y-clearance keeps the sleeve wrapping past the
        # skin's silhouette through the full range of arm elevation.
        clear = max(radius_inflate + 0.004, 0.016)
        rx = body_rx + clear
        ry = body_ry + clear + 0.006
        rz = body_rz + clear
        v, n, ba, bb, w, idx = prim.ellipsoid(
            (0.0, body_offset_y, 0.0),
            (rx, ry, rz),
            u_idx, u_parent, weight_self=body_weight_self,
            rings=10, radial=16,
        )
        chunks_v.append(v @ R.T); chunks_n.append(n @ R.T)
        chunks_ba.append(ba); chunks_bb.append(bb); chunks_w.append(w)
        chunks_i.append(idx + offset)
        offset += v.shape[0]

    # Clavicle-anchored shoulder yoke: bridges the armpit gap that opens up
    # when the arm raises. Without it, the deltoid ellipsoid above (skinned
    # mostly to the upper-arm) rotates away from the chest shell as the arm
    # lifts, exposing skin under the armpit. By skinning this yoke to the
    # clavicle bone (which barely rotates with arm motion), the patch stays
    # planted on the lateral chest and keeps the cloth continuous through
    # the full arm-raise range.
    chest_idx = _find(bones, "chest")
    for clav_name in ("clav_L", "clav_R"):
        if clav_name not in wanted:
            continue
        c_idx = _find(bones, clav_name)
        if c_idx < 0 or chest_idx < 0:
            continue
        _, c_parent, _, c_tip, c_r = bones[c_idx]
        c_tip_v = np.asarray(c_tip, dtype=np.float32)
        c_len = float(np.linalg.norm(c_tip_v))
        # Build in the clavicle's pre-rotation frame where +Y is along the
        # bone's length (head -> tip). Place the ellipsoid centred ~85%
        # along the clavicle (right under the shoulder joint) and slightly
        # depressed so it caps the armpit. R rotates it onto the actual
        # clavicle axis (lateral) before merging.
        Rc = mathx.align_y_to(c_tip_v)
        ext_lat = c_len * 0.48 + radius_inflate * 0.7   # along clav (lateral)
        ext_dn  = c_r * 2.1 + radius_inflate            # downward (axillary)
        ext_fb  = c_r * 1.9 + radius_inflate            # front-back
        v, n, ba, bb, w, idx = prim.ellipsoid(
            (0.0, c_len * 0.85, -c_r * 0.4),
            (ext_dn, ext_lat, ext_fb),
            c_idx, chest_idx, weight_self=0.85,
            rings=8, radial=14,
        )
        v = v @ Rc.T
        n = n @ Rc.T
        chunks_v.append(v); chunks_n.append(n)
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
    gender = getattr(shape, "gender", "neutral") if shape else "neutral"
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
        profile = _limb_profile(name, gender)
        if profile is None:
            profile = _torso_profile(name, shape)
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

    # Deltoid bulges at each shoulder. Sized by gender: male shoulders
    # carry visible deltoid mass, female shoulders are softer. The cap
    # still has to fit inside the garment radius_inflate so it doesn't
    # poke through the shirt -- keep the male axes <= 1.06 of arm radius.
    delt_x = 1.04 if gender == "male" else (0.92 if gender == "female" else 1.00)
    delt_y = 0.78 if gender == "male" else (0.74 if gender == "female" else 0.78)
    delt_z = 0.98 if gender == "male" else (0.90 if gender == "female" else 0.96)
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
            (r_shoulder * delt_x, r_shoulder * delt_y, r_shoulder * delt_z),
            u_idx, u_parent, weight_self=0.75,
            rings=12, radial=16,
        )
        chunks.append((v @ R.T, n @ R.T, ba, bb, w, idx))

    if head_idx >= 0:
        head_parent = bones[head_idx][1]
        head_tip = np.asarray(bones[head_idx][3], dtype=np.float32)
        head_r = bones[head_idx][4]
        chunks.extend(_head_compound(head_idx, head_parent, head_tip, head_r,
                                     gender, shape))

    if shape is not None and getattr(shape, "bust", 0.0) > 0.01 and chest_idx >= 0:
        chunks.extend(_bust(chest_idx, bones[chest_idx], shape.bust,
                            projection=getattr(shape, "bust_proj", 0.6)))

    if shape is not None and getattr(shape, "gender", "neutral") == "female":
        pelvis_idx = _find(bones, "pelvis")
        if pelvis_idx >= 0:
            chunks.extend(_glutes(pelvis_idx, bones[pelvis_idx],
                                  getattr(shape, "hip_w", 1.0),
                                  lower_bulk=getattr(shape, "lower_bulk", 1.0)))

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


def _skull_shell(cy, half_h, profile, bone_index, parent_index,
                 rings=28, radial=32, weight_self=1.0, asymmetry=0.0):
    """Single closed mesh for a skull/face shell.

    Cross-section at height-parameter ``t`` in [-1, 1] (chin -> crown) is
    an ellipse whose X/Z radii and centre-Z offset come from
    ``profile(t) -> (rx, rz, cz)``. Like a classic UV sphere but with
    Y-varying radii, so the silhouette is a proper head shape rather
    than a ball of bumps.
    """
    v, n, ba, bb, w = [], [], [], [], []
    profile_cache = []
    for i in range(rings + 1):
        theta = math.pi * (i / rings)
        t = math.cos(theta)
        p = profile(t)
        if len(p) == 3:
            p = (p[0], p[1], p[1], p[2])   # (rx, rz_front, rz_back, cz)
        profile_cache.append(p)

    for i in range(rings + 1):
        theta = math.pi * (i / rings)
        ct = math.cos(theta)
        st = math.sin(theta)
        t = ct
        rx, rz_f, rz_b, cz = profile_cache[i]
        for j in range(radial):
            phi = 2.0 * math.pi * (j / radial)
            sp, cp = math.sin(phi), math.cos(phi)
            # Choose front/back depth based on which hemisphere this vertex
            # is on; smoothly blend near the equator (sp~0) to avoid a
            # visible seam at +/-X.
            blend = 0.5 * (1.0 + sp)        # 0 at back pole, 1 at front pole
            blend_s = blend * blend * (3 - 2 * blend)
            rz = rz_b + (rz_f - rz_b) * blend_s
            # Subtle left/right asymmetry -- peaks in the face band (t near
            # the cheekbone/eye line) and fades toward chin and crown so the
            # skull silhouette stays plausible.
            if asymmetry != 0.0:
                face_band = max(0.0, 1.0 - abs(t - 0.1) * 2.2)
                rx_eff = rx * (1.0 + asymmetry * cp * face_band)
            else:
                rx_eff = rx
            px = st * cp * rx_eff
            py = cy + t * half_h
            pz = st * sp * rz + cz
            v.append((px, py, pz))
            if i == 0:
                nx, ny, nz = 0.0, 1.0, 0.0
            elif i == rings:
                nx, ny, nz = 0.0, -1.0, 0.0
            else:
                nx = (st * cp) / max(rx, 1e-6)
                ny = t / max(half_h, 1e-6)
                nz = (st * sp) / max(rz, 1e-6)
                inv = 1.0 / (math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
                nx *= inv; ny *= inv; nz *= inv
            n.append((nx, ny, nz))
            ba.append(bone_index)
            bb.append(parent_index if parent_index >= 0 else bone_index)
            w.append((weight_self, 1.0 - weight_self))

    idx = []
    for i in range(rings):
        for j in range(radial):
            a = i * radial + j
            b = i * radial + (j + 1) % radial
            c = (i + 1) * radial + j
            d = (i + 1) * radial + (j + 1) % radial
            idx.extend([a, c, b, b, c, d])

    return (np.asarray(v, np.float32), np.asarray(n, np.float32),
            np.asarray(ba, np.int32), np.asarray(bb, np.int32),
            np.asarray(w, np.float32), np.asarray(idx, np.uint32))


def _head_compound(head_idx, parent_idx, tip, radius, gender, shape=None):
    """Build a compound head with anthropometrically proportioned features.

    The head bone's local +Y points from the neck joint to the top of the
    skull. We place landmarks as fractions of that length and size features
    from the same length (not from the bone radius, which was too generous
    and gave ping-pong-ball proportions).

    Head is assembled from several overlapping ellipsoids rather than one
    sphere, so the silhouette shows the cranium/face/jaw distinction:
      - cranial vault (upper back, deeper than wide, occiput pushed back)
      - face plate (narrower front-lower, slight forward lean)
      - mandible (jaw block tapering from angles to chin)
    """
    length = float(np.linalg.norm(tip))
    R = mathx.align_y_to(tip) if length > 1e-6 else np.eye(3, dtype=np.float32)

    # Reference half-widths / depths. Real-world adult head proportions
    # (height : width : depth) ~= 1.0 : 0.65 : 0.90.
    w_factor = 0.34 * (1.06 if gender == "male" else 0.98)
    d_factor = 0.40
    head_w = length * w_factor
    head_d = length * d_factor

    chunks = []

    # --- Single profiled skull shell --------------------------------------
    # t in [-1, 1]: -1 = chin apex, 0 ~ cheekbone/zygomatic line, +1 = crown.
    # We scale rx/rz with t so the silhouette shows:
    #   * a rounded crown narrower than the temples
    #   * max cheekbone width near t ~ 0.1
    #   * a tapered jawline narrowing smoothly toward the chin
    #   * occiput pushed back in Z at the crown, forehead slightly forward
    # Gender dimorphism: male has wider jaw (less taper) and more forward
    # chin; female has more pronounced jaw taper.
    jaw_taper = 0.38 if gender == "male" else 0.50   # chin width reduction
    # Skull depth is not front/back symmetric: the occipital bulge sits
    # behind the head so much further than the forehead protrudes. We
    # split the depth into a front half and a back half and handle them
    # independently in the primitive below.

    def profile(t):
        # Width (rx) profile ------------------------------------------------
        if t >= 0:
            rx_scale = 1.0 - 0.08 * t * t
        else:
            u = -t
            rx_scale = 1.0 - jaw_taper * (u * u * (3 - 2 * u))
            rx_scale = max(rx_scale, 0.28)
        rx = head_w * rx_scale

        # Asymmetric front / back depth ------------------------------------
        # Front (face plane) stays close to the base head_d at the midface
        # and tapers a bit toward chin and crown. Back (occiput) is rounder
        # and a bit deeper at the crown so the profile shows a true skull.
        if t >= 0:
            rz_f = head_d * (1.00 - 0.16 * t * t)     # forehead slopes back
        else:
            rz_f = head_d * max(0.96 + 0.14 * t, 0.72)  # chin shallower
        if t >= 0:
            rz_b = head_d * (1.08 + 0.06 * t - 0.14 * t * t)
        else:
            rz_b = head_d * max(1.06 + 0.28 * t, 0.78)

        # Forward shift of the lower face: build a chin that pushes the
        # skull forward below the mouth, peaking at the chin tip.
        if t < -0.3:
            u = (-t - 0.3) / 0.7
            chin_push = 0.14 * u * u * (3 - 2 * u)
        else:
            chin_push = 0.0
        cz = head_d * chin_push
        return rx, rz_f, rz_b, cz

    # Skull spans from chin (y=length*H_CHIN) to crown (y=length*H_TOP).
    skull_cy = length * (H_TOP + H_CHIN) * 0.5
    skull_half = length * (H_TOP - H_CHIN) * 0.5
    face_asym = _shape_trait(shape, "face_asymmetry", 0.0)
    chunks.append(_skull_shell(
        skull_cy, skull_half, profile,
        head_idx, parent_idx, rings=32, radial=40, weight_self=1.0,
        asymmetry=face_asym,
    ))

    # Chin protuberance is carried by the skull shell profile (forward cz
    # offset at t<-0.5). No floating blob is added here.

    # --- Nose: bridge + tip + wings ---------------------------------------
    # Bridge: a narrow ellipsoid from the radix (between the eyebrows) down
    # to just above the tip. Narrower in X than before, and longer in Y so
    # the nose has a visible bridge line.
    bridge_cy = length * (H_NOSE_BASE + (H_EYE - H_NOSE_BASE) * 0.70)
    bridge_rx = length * (0.030 if gender == "male" else 0.026)
    bridge_ry = length * (H_EYE - H_NOSE_BASE) * 0.60
    bridge_rz = length * 0.060
    chunks.append(prim.ellipsoid(
        (0.0, bridge_cy, head_d * 0.90),
        (bridge_rx, bridge_ry, bridge_rz),
        head_idx, parent_idx, rings=10, radial=12,
    ))

    # Nose tip: rounded bulb at the base of the nose, protruding forward.
    tip_cy = length * (H_NOSE_BASE + 0.02)
    tip_rx = length * (0.050 if gender == "male" else 0.044)
    tip_ry = length * 0.040
    tip_rz = length * 0.055
    chunks.append(prim.ellipsoid(
        (0.0, tip_cy, head_d * 0.95),
        (tip_rx, tip_ry, tip_rz),
        head_idx, parent_idx, rings=10, radial=14,
    ))

    # Nostril wings (alae): two small lobes flanking the tip, implying
    # nostrils without modelling a cavity. Placed slightly behind the tip
    # so the tip still reads as the forwardmost point.
    wing_rx = length * 0.025
    wing_ry = length * 0.024
    wing_rz = length * 0.032
    wing_sep = length * (0.040 if gender == "male" else 0.034)
    wing_cy = length * (H_NOSE_BASE + 0.005)
    wing_cz = head_d * 0.87
    chunks += [
        prim.ellipsoid((+wing_sep, wing_cy, wing_cz),
                       (wing_rx, wing_ry, wing_rz),
                       head_idx, parent_idx, rings=8, radial=12),
        prim.ellipsoid((-wing_sep, wing_cy, wing_cz),
                       (wing_rx, wing_ry, wing_rz),
                       head_idx, parent_idx, rings=8, radial=12),
    ]

    # --- Brow ridge -------------------------------------------------------
    # Subtle supraorbital ridge: two short arched swells above each orbit
    # (not a single horizontal bar). Male's is more prominent. Tucked
    # deep enough that only a soft highlight pokes through the shell.
    brow_ry = length * (0.012 if gender == "male" else 0.008)
    brow_rz = length * (0.020 if gender == "male" else 0.015)
    brow_rx = length * 0.070
    brow_y = length * (H_BROW - 0.015)
    brow_sep = length * 0.105
    brow_z = head_d * 0.65
    chunks += [
        prim.ellipsoid((+brow_sep, brow_y, brow_z), (brow_rx, brow_ry, brow_rz),
                       head_idx, parent_idx, rings=6, radial=14),
        prim.ellipsoid((-brow_sep, brow_y, brow_z), (brow_rx, brow_ry, brow_rz),
                       head_idx, parent_idx, rings=6, radial=14),
    ]

    # Cheekbones are now implicit in the skull shell's profile; no extra
    # blobs are added here. Future cycles may re-introduce them as subtle
    # inset swells, but the prior-cycle floating discs are removed.

    # --- Ears -------------------------------------------------------------
    # Ear top aligned roughly with the brow line, bottom around the nose
    # base -- the canonical anatomical positioning. Built from two pieces
    # per side: a helix (outer rim) and a lobe. They hug the skull and
    # tilt back slightly so they read correctly in profile.
    ear_top_y = length * (H_BROW - 0.02)
    ear_bot_y = length * (H_NOSE_BASE - 0.02)
    ear_cy = 0.5 * (ear_top_y + ear_bot_y)
    ear_half_h = 0.5 * (ear_top_y - ear_bot_y)
    ear_cz = -head_d * 0.10          # pushed back: ear canal sits behind eye line
    ear_x = head_w * 0.98
    for side in (+1.0, -1.0):
        # Helix: tall narrow vertical capsule that forms the outer rim.
        chunks.append(prim.ellipsoid(
            (side * ear_x, ear_cy + ear_half_h * 0.05, ear_cz),
            (length * 0.020, ear_half_h * 1.10, length * 0.060),
            head_idx, parent_idx, rings=10, radial=14,
        ))
        # Lobe: small bulb at the bottom, slightly protruding forward.
        chunks.append(prim.ellipsoid(
            (side * ear_x, ear_bot_y - length * 0.005, ear_cz + length * 0.010),
            (length * 0.024, length * 0.026, length * 0.038),
            head_idx, parent_idx, rings=8, radial=12,
        ))

    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- female bust -------------------------------------------------------------

def _bust(chest_idx, chest_bone, amount, projection=0.6):
    """Two ellipsoids attached to the chest bone for female characters.

    Geometry:
      * Sit on the upper half of the chest bone (pectoral plane), centred
        roughly on the nipple line at ~60% of chest length.
      * Separation ~ chest radius * 0.40 (one fingerbreadth at the sternum).
      * Forward projection scales with `projection` (0..1) -- this is the
        knob that makes the bust visible above a tank top.
      * Slight downward teardrop axis (Y < X, Z big) so the silhouette
        from the side reads as breast-shaped, not as a sphere.

    `amount` controls overall mass; `projection` controls how far the
    breast tissue sticks out forward (Z axis) relative to its base size.
    """
    _, _, _, tip, radius = chest_bone
    tip_vec = np.asarray(tip, dtype=np.float32)
    R = mathx.align_y_to(tip_vec)
    length = float(np.linalg.norm(tip_vec))
    sep = radius * 0.40
    y = length * 0.55
    z = radius * (0.45 + 0.50 * projection)
    base = radius * (0.30 + 0.22 * amount)
    # Slightly narrower than the cloth overlay so the skin layer sits
    # inside the fabric surface; matched X so the overlay doesn't show
    # a step at the side silhouette.
    rxyz = (base * 0.96, base * 0.78, base * (1.05 + 0.55 * projection))
    chunks = [
        prim.ellipsoid((+sep, y, z), rxyz, chest_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
        prim.ellipsoid((-sep, y, z), rxyz, chest_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
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

def _glutes(pelvis_idx, pelvis_bone, hip_w, lower_bulk=1.0):
    """Two ellipsoids on the back of the pelvis (female). Larger when hips
    are wider OR when lower_bulk indicates more gynoid fat distribution."""
    _, _, _, tip, radius = pelvis_bone
    tip_vec = np.asarray(tip, dtype=np.float32)
    R = mathx.align_y_to(tip_vec)
    length = float(np.linalg.norm(tip_vec))
    # Glute centres sit just inside the pelvis cap so the cheeks blend
    # into the hip silhouette rather than sticking out as separate balls.
    sep = radius * 0.34 * (0.5 + 0.5 * hip_w)
    y = -length * 0.10
    z = -radius * 0.55
    size = radius * (0.46 + 0.12 * hip_w) * (0.88 + 0.22 * lower_bulk)
    # X axis narrow (so the cheeks don't push past the hip silhouette),
    # Y a bit squashed (vertical), Z deep (the buttock projects backward).
    rxyz = (size * 0.86, size * 0.92, size * 1.10)
    chunks = [
        prim.ellipsoid((+sep, y, z), rxyz, pelvis_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
        prim.ellipsoid((-sep, y, z), rxyz, pelvis_idx, -1, weight_self=1.0,
                       rings=12, radial=14),
    ]
    return [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]


# --- eyes --------------------------------------------------------------------

def _shape_trait(shape, name, default=1.0):
    """Read an optional numeric knob off the Shape; fall back when absent."""
    if shape is None:
        return default
    return float(getattr(shape, name, default))


def _head_info(bones, shape=None):
    """Shared helper: head bone index, parent, rest rotation, length, depth.

    When ``shape`` is provided the head width picks up the gender bias so
    feature primitives line up with the gender-dimorphic skull shell.
    """
    head_idx = _find(bones, "head")
    if head_idx < 0:
        return None
    parent = bones[head_idx][1]
    tip = np.asarray(bones[head_idx][3], dtype=np.float32)
    r = bones[head_idx][4]
    R = mathx.align_y_to(tip)
    length = float(np.linalg.norm(tip))
    gender = getattr(shape, "gender", "neutral") if shape is not None else "neutral"
    head_d = length * 0.40
    head_w = length * 0.34 * (1.06 if gender == "male" else 0.98)
    return head_idx, parent, R, length, r, head_w, head_d


def build_eyes(bones, shape=None) -> SkinnedMesh:
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    # Eye line at y = length*H_EYE. Pushed deeper into the orbit (lower z,
    # smaller radius) so the eyelids read as real folds over a set-in
    # eyeball rather than a surface disc.
    sep  = length * 0.108
    y    = length * H_EYE
    z    = head_d * 0.70
    eye_r = length * 0.050
    radii = (eye_r, eye_r * 0.95, eye_r * 0.9)
    chunks = [
        prim.ellipsoid((+sep, y, z), radii, head_idx, parent, rings=12, radial=18),
        prim.ellipsoid((-sep, y, z), radii, head_idx, parent, rings=12, radial=18),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_iris(bones, shape=None) -> SkinnedMesh:
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep  = length * 0.108
    y    = length * H_EYE
    z    = head_d * 0.70
    eye_r = length * 0.050
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


def build_pupils(bones, shape=None) -> SkinnedMesh:
    """Small dark pupil in the centre of each iris."""
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep  = length * 0.108
    y    = length * H_EYE
    z    = head_d * 0.70
    eye_r = length * 0.050
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


def build_eyelids(bones, shape=None, expression="neutral") -> SkinnedMesh:
    """Skin-colored upper/lower eyelid folds that mask the spherical eyeballs.

    This keeps the existing simple eye spheres, but exposes them through a
    narrower almond aperture instead of showing full round balls. The
    aperture is widened or narrowed according to ``expression`` so squints
    and surprise read at the eye level as well as the mouth level.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep = length * 0.115
    y = length * H_EYE
    z = head_d * 0.78
    eye_r = length * 0.060

    # Slightly in front of the sclera/iris so depth testing naturally hides
    # the upper/lower poles of the eye sphere. Upper lid is thicker and
    # tilts outward-up for an almond shape; lower lid is thinner.
    lid_z = z + eye_r * 0.96
    hx = length * 0.068
    upper_hy = length * (0.022 if expression != "surprised" else 0.016)
    lower_hy = length * 0.012
    aperture = 1.0
    if expression == "squint":
        aperture = 0.62
    elif expression == "surprised":
        aperture = 1.22
    upper_y = y + eye_r * (0.70 * aperture)
    lower_y = y - eye_r * (0.60 * aperture)

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


def build_lips(bones, shape=None, expression="neutral") -> SkinnedMesh:
    """Upper + lower lip with a subtle cupid's bow and expression offsets.

    Upper lip is built from two halves (left + right peak) with a small
    central dip between them, giving a readable cupid's bow. The lower
    lip is a single fuller pillow. ``expression`` slides the mouth corners
    up (smile) or down (frown), and separates the lips vertically when
    ``surprised`` so the mouth reads as open.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    y_mouth = length * H_MOUTH
    z = head_d * 0.86
    fullness = _shape_trait(shape, "lip_fullness", 1.0)

    corner_lift = 0.0
    mouth_open = 0.0
    if expression == "smile":
        corner_lift = length * 0.014
    elif expression == "frown":
        corner_lift = -length * 0.012
    elif expression == "surprised":
        mouth_open = length * 0.014

    # Upper lip: two symmetric lobes, narrow, peaked slightly off-center.
    up_half_sep = length * 0.028
    up_rx = length * 0.048
    up_ry = length * 0.011 * fullness
    up_rz = length * 0.014 * fullness
    up_y = y_mouth + length * 0.010 + mouth_open * 0.35
    upper_L = prim.ellipsoid(
        (+up_half_sep, up_y, z),
        (up_rx, up_ry, up_rz),
        head_idx, parent, rings=6, radial=14,
    )
    upper_R = prim.ellipsoid(
        (-up_half_sep, up_y, z),
        (up_rx, up_ry, up_rz),
        head_idx, parent, rings=6, radial=14,
    )
    # Mouth corners -- slide up/down with smile/frown.
    corner_rx = length * 0.014
    corner_ry = length * 0.010
    corner_rz = length * 0.010
    corner_x = length * 0.075
    corner_y = y_mouth + length * 0.001 + corner_lift
    corner_L = prim.ellipsoid(
        (+corner_x, corner_y, z - length * 0.003),
        (corner_rx, corner_ry, corner_rz),
        head_idx, parent, rings=5, radial=10,
    )
    corner_R = prim.ellipsoid(
        (-corner_x, corner_y, z - length * 0.003),
        (corner_rx, corner_ry, corner_rz),
        head_idx, parent, rings=5, radial=10,
    )
    # Lower lip: wider, fuller pillow. Pushed further down when surprised.
    lower = prim.ellipsoid(
        (0.0, y_mouth - length * 0.010 - mouth_open * 0.65, z + length * 0.002),
        (length * 0.082, length * 0.014 * fullness, length * 0.016 * fullness),
        head_idx, parent, rings=6, radial=20,
    )
    chunks = [upper_L, upper_R, corner_L, corner_R, lower]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_age_detail(bones, shape=None, age_group="adult") -> SkinnedMesh:
    """Subtle forehead, nasolabial and eye-corner creases for mature faces.

    Rendered as very thin flat patches laid over the skull/face surface.
    Returns an empty mesh for ``young`` faces so young characters stay
    smooth. ``strength`` scales the crease thickness with age group.
    """
    if age_group not in ("adult", "elder"):
        return _empty_mesh()
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info
    strength = 1.0 if age_group == "elder" else 0.45
    z = head_d * 0.93
    chunks = []

    # Forehead: three horizontal creases spread across the brow band.
    for off in (0.0, -0.030, -0.060):
        chunks.append(prim.flat_patch(
            (0.0, length * (H_BROW + 0.080 + off), z),
            (head_w * 0.25, length * 0.0023 * strength),
            head_idx, parent, normal=(0, 0.05, 1),
            subdiv=(8, 1), thickness=0.0012,
        ))

    # Eye-corner crows'-feet and nasolabial fold, mirrored L/R.
    for side in (+1.0, -1.0):
        chunks.append(prim.flat_patch(
            (side * head_w * 0.38, length * (H_EYE - 0.010), head_d * 0.91),
            (length * 0.030, length * 0.0045 * strength),
            head_idx, parent, normal=(side * 0.18, 0.00, 1),
            subdiv=(4, 1), thickness=0.0012,
        ))
        chunks.append(prim.flat_patch(
            (side * head_w * 0.20, length * (H_NOSE_BASE + 0.015), head_d * 0.88),
            (length * 0.006 * strength, length * 0.048),
            head_idx, parent, normal=(side * 0.12, -0.08, 1),
            subdiv=(1, 5), thickness=0.0011,
        ))

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
