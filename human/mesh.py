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
H_CHIN      = 0.06
H_MOUTH     = 0.31
H_NOSE_BASE = 0.43
H_EYE       = 0.55
H_BROW      = 0.63
H_HAIRLINE  = 0.77
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
    w_factor = 0.335 * (1.05 if gender == "male" else 0.99)
    d_factor = 0.385
    head_w = length * w_factor
    head_d = length * d_factor
    age_group = getattr(shape, "age_group", "adult") if shape is not None else "adult"
    age_young = 1.0 if age_group == "young" else 0.0
    age_elder = 1.0 if age_group == "elder" else 0.0
    cheekbone = _shape_trait(shape, "cheekbone", 1.0)
    jaw_width = _shape_trait(shape, "jaw_width", 1.0)
    chin_proj = _shape_trait(shape, "chin_proj", 1.0)
    nose_width = _shape_trait(shape, "nose_width", 1.0)
    nose_proj = _shape_trait(shape, "nose_proj", 1.0)
    nose_bridge = _shape_trait(shape, "nose_bridge", 1.0)
    brow_prom = _shape_trait(shape, "brow_prominence", 1.0)

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
    jaw_taper = (0.22 if gender == "male" else 0.30) / max(jaw_width, 0.75)
    # Skull depth is not front/back symmetric: the occipital bulge sits
    # behind the head so much further than the forehead protrudes. We
    # split the depth into a front half and a back half and handle them
    # independently in the primitive below.

    def profile(t):
        # Width (rx) profile ------------------------------------------------
        if t >= 0:
            rx_scale = 1.0 - 0.11 * t * t
            # Young heads keep a little more rounded cranium; elder heads
            # lose a touch of temporal fullness.
            rx_scale += 0.020 * age_young * max(t, 0.0)
            rx_scale -= 0.025 * age_elder * max(t - 0.25, 0.0)
        else:
            u = -t
            rx_scale = 1.0 - jaw_taper * (u * u * (3 - 2 * u))
            # Cheekbone variation is broad but subtle; it peaks high in the
            # midface and fades before it turns into a caricatured jaw.
            cheek_band = max(0.0, 1.0 - abs(t - 0.06) * 3.8)
            rx_scale *= 1.0 + (cheekbone - 1.0) * 0.11 * cheek_band
            rx_scale *= 1.0 + 0.030 * age_elder * u
            rx_scale = max(rx_scale, 0.46 if gender == "male" else 0.40)
        rx = head_w * rx_scale

        # Asymmetric front / back depth ------------------------------------
        # Front (face plane) stays close to the base head_d at the midface
        # and tapers a bit toward chin and crown. Back (occiput) is rounder
        # and a bit deeper at the crown so the profile shows a true skull.
        if t >= 0:
            rz_f = head_d * (0.98 - 0.18 * t * t)     # forehead slopes back
            forehead_plane = max(0.0, 1.0 - abs(t - 0.38) * 3.2)
            rz_f *= 1.0 - 0.050 * forehead_plane
        else:
            rz_f = head_d * max(0.95 + 0.12 * t, 0.74)  # chin shallower
        if t >= 0:
            rz_b = head_d * (1.06 + 0.07 * t - 0.12 * t * t)
        else:
            rz_b = head_d * max(1.06 + 0.28 * t, 0.78)

        # Forward shift of the lower face: build a chin that pushes the
        # skull forward below the mouth, peaking at the chin tip.
        if t < -0.3:
            u = (-t - 0.3) / 0.7
            chin_push = 0.10 * chin_proj * u * u * (3 - 2 * u)
        else:
            chin_push = 0.0
        # Elder lower faces settle slightly forward/down around the mouth and
        # chin; keeping this in the skull shell avoids floating detail pieces.
        if t < -0.15:
            chin_push += 0.018 * age_elder * (-t - 0.15)
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
    # offset at t<-0.5). Surface detail below adds the mentolabial cue.

    # --- Nose: bridge + tip + alar cartilages ------------------------------
    # The nasal silhouette is built from a narrow bony bridge, paired upper
    # lateral cartilages, tip dome, alar wings, and a small columella. Keeping
    # those roles separate avoids the stacked-sphere look while preserving the
    # lightweight procedural mesh.
    bridge_cy = length * (H_NOSE_BASE + (H_EYE - H_NOSE_BASE) * 0.72)
    bridge_rx = length * (0.016 if gender == "male" else 0.014) * nose_width
    bridge_ry = length * (H_EYE - H_NOSE_BASE) * 0.68
    bridge_rz = length * 0.042 * nose_bridge * nose_proj
    chunks.append(prim.ellipsoid(
        (0.0, bridge_cy, head_d * (0.83 + 0.035 * nose_proj)),
        (bridge_rx, bridge_ry, bridge_rz),
        head_idx, parent_idx, rings=10, radial=12,
    ))

    # Tip dome: smaller and lower than the former bulb, so the alar wings and
    # bridge define the nose instead of a single round bead.
    tip_cy = length * (H_NOSE_BASE + 0.018)
    tip_rx = length * (0.027 if gender == "male" else 0.025) * nose_width
    tip_ry = length * 0.025
    tip_rz = length * 0.045 * nose_proj
    chunks.append(prim.ellipsoid(
        (0.0, tip_cy, head_d * (0.895 + 0.058 * nose_proj)),
        (tip_rx, tip_ry, tip_rz),
        head_idx, parent_idx, rings=10, radial=14,
    ))

    # Nostril wings (alae): wider, flatter lateral cartilages wrapping around
    # the nostril plane. They sit behind the tip rather than forming two balls.
    wing_rx = length * 0.020 * nose_width
    wing_ry = length * 0.012
    wing_rz = length * 0.018 * nose_proj
    wing_sep = length * (0.030 if gender == "male" else 0.027) * nose_width
    wing_cy = length * (H_NOSE_BASE + 0.005)
    wing_cz = head_d * (0.852 + 0.036 * nose_proj)
    chunks += [
        prim.ellipsoid((+wing_sep, wing_cy, wing_cz),
                       (wing_rx, wing_ry, wing_rz),
                       head_idx, parent_idx, rings=8, radial=12),
        prim.ellipsoid((-wing_sep, wing_cy, wing_cz),
                       (wing_rx, wing_ry, wing_rz),
                       head_idx, parent_idx, rings=8, radial=12),
    ]

    # Columella: central soft-tissue strut between nostrils, visible in
    # three-quarter/profile and anchoring the tip to the philtrum.
    chunks.append(prim.ellipsoid(
        (0.0, length * (H_NOSE_BASE - 0.012),
         head_d * (0.865 + 0.036 * nose_proj)),
        (length * 0.0060 * nose_width, length * 0.011,
         length * 0.009 * nose_proj),
        head_idx, parent_idx, rings=7, radial=10,
    ))

    # --- Brow ridge -------------------------------------------------------
    # Subtle supraorbital ridge: two short arched swells above each orbit
    # (not a single horizontal bar). Male's is more prominent. Tucked
    # deep enough that only a soft highlight pokes through the shell.
    brow_ry = length * (0.008 if gender == "male" else 0.006) * brow_prom
    brow_rz = length * (0.014 if gender == "male" else 0.010) * brow_prom
    brow_rx = length * 0.070
    brow_y = length * (H_BROW - 0.015)
    brow_sep = length * 0.105
    brow_z = head_d * 0.60
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
    # base -- the canonical anatomical positioning. Built from landmark
    # pieces: outer helix, inner antihelix/concha ridge, tragus, and lobe.
    # The parts hug the skull and tilt back slightly so profile reads as a
    # pinna rather than a flat oval.
    ear_top_y = length * (H_BROW - 0.01)
    ear_bot_y = length * (H_NOSE_BASE - 0.01)
    ear_cy = 0.5 * (ear_top_y + ear_bot_y)
    ear_half_h = 0.5 * (ear_top_y - ear_bot_y)
    ear_cz = -head_d * 0.04          # canal sits behind eye line without floating
    ear_x = head_w * 0.93
    for side in (+1.0, -1.0):
        # Helix: tall narrow vertical capsule that forms the outer rim.
        chunks.append(prim.ellipsoid(
            (side * ear_x, ear_cy + ear_half_h * 0.05, ear_cz),
            (length * 0.012, ear_half_h * 1.04, length * 0.045),
            head_idx, parent_idx, rings=10, radial=14,
        ))
        # Antihelix / concha rim: smaller raised inner fold set slightly
        # forward and inward from the helix.
        chunks.append(prim.ellipsoid(
            (side * (ear_x - length * 0.012), ear_cy + ear_half_h * 0.06,
             ear_cz + length * 0.008),
            (length * 0.0065, ear_half_h * 0.62, length * 0.020),
            head_idx, parent_idx, rings=8, radial=10,
        ))
        # Tragus: small knob in front of the concha, near the canal.
        chunks.append(prim.ellipsoid(
            (side * (ear_x - length * 0.018), ear_cy - ear_half_h * 0.22,
             ear_cz + length * 0.020),
            (length * 0.008, length * 0.012, length * 0.011),
            head_idx, parent_idx, rings=6, radial=10,
        ))
        # Lobe: small bulb at the bottom, slightly protruding forward.
        chunks.append(prim.ellipsoid(
            (side * ear_x, ear_bot_y - length * 0.002, ear_cz + length * 0.006),
            (length * 0.014, length * 0.017, length * 0.024),
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


def _head_age(shape):
    if shape is None:
        return "adult"
    age = getattr(shape, "age_group", "adult")
    return age if age in ("young", "adult", "elder") else "adult"


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
    head_d = length * 0.385
    head_w = length * 0.335 * (1.05 if gender == "male" else 0.99)
    return head_idx, parent, R, length, r, head_w, head_d


def build_ear_detail(bones, shape=None) -> SkinnedMesh:
    """Soft concha shadows for the procedural pinna.

    The main head mesh carries raised ear geometry. These very thin patches
    add the recessed concha/canal value change that geometry alone cannot
    express at this low polygon budget.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    ear_top_y = length * (H_BROW - 0.01)
    ear_bot_y = length * (H_NOSE_BASE - 0.01)
    ear_cy = 0.5 * (ear_top_y + ear_bot_y)
    ear_half_h = 0.5 * (ear_top_y - ear_bot_y)
    ear_cz = -head_d * 0.04
    ear_x = head_w * 0.93
    chunks = []
    for side in (+1.0, -1.0):
        chunks.append(prim.flat_patch(
            (side * (ear_x + length * 0.006), ear_cy - ear_half_h * 0.02,
             ear_cz + length * 0.007),
            (length * 0.014, ear_half_h * 0.30),
            head_idx, parent,
            normal=(side, 0.0, 0.0),
            subdiv=(4, 6),
            thickness=0.0008,
        ))

    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_chin_detail(bones, shape=None) -> SkinnedMesh:
    """Static lower-face crease cue for the mental pad.

    A low-poly skull shell cannot express the small mentolabial break between
    lower lip and chin without either adding many rings or a floating chin
    blob. This skin-toned shadow patch provides that anatomical landmark
    while staying flush to the head surface.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, _head_w, head_d = info

    age = _head_age(shape)
    strength = 0.80 if age == "elder" else (0.52 if age == "adult" else 0.34)
    chunk = prim.flat_patch(
        (0.0, length * (H_MOUTH - 0.058), head_d * 0.858),
        (length * 0.034, length * 0.0032 * strength),
        head_idx, parent,
        normal=(0.0, -0.12, 1.0),
        subdiv=(6, 1),
        thickness=0.0008,
    )
    v, n, ba, bb, w, idx = chunk
    v, n = v @ R.T, n @ R.T
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


# Eye geometry constants (factored so eyeball / iris / pupil / limbus /
# catchlight stay in perfect concentric agreement and convergence is
# applied identically everywhere). All scales are relative to head length.
EYE_SEP_X      = 0.098   # half-distance between eye centers
EYE_RADIUS     = 0.037   # eyeball radius (true sphere)
IRIS_RATIO     = 0.55    # iris radius as fraction of eyeball radius
PUPIL_RATIO    = 0.30    # pupil radius as fraction of iris radius (~3mm)
LIMBUS_RATIO   = 1.10    # limbal-ring outer radius as fraction of iris
# Convergence: each iris is shifted inward so both gaze axes meet ~50 cm
# in front of the head (typical portrait distance). Without this the eyes
# read as a parallel "thousand-yard stare" and any tiny skull asymmetry
# tips the brain toward registering exotropia.
GAZE_TARGET_M  = 0.50    # focal distance (meters) -- portrait range
# Eye is set ~0.7*head_d back from face surface; iris sits 0.75*eye_r
# ahead of the eyeball center. Convergence shift is computed per-build
# from the actual offsets so the math stays consistent if those move.


def _eye_metrics(length, head_d):
    """Return shared eye placement: (sep, y, z_back, eye_r, conv_shift).

    ``z_back`` is the eyeball center; iris/pupil/limbus/catchlight all
    sit forward of that on +Z. ``conv_shift`` is the inward x-offset to
    apply to anything sitting at iris-depth so both eyes converge on
    GAZE_TARGET_M instead of staring parallel.
    """
    sep   = length * EYE_SEP_X
    y     = length * H_EYE
    z     = head_d * 0.56
    eye_r = length * EYE_RADIUS
    # iris depth ahead of eyeball center
    iris_z_off = eye_r * 0.75
    # convergence: tiny similar-triangles toe-in. Real head_length is
    # already in meter-ish units; sep is the lateral half-separation.
    # angle = atan(sep / GAZE_TARGET_M); inward shift at iris depth
    # = iris_z_off * tan(angle).
    if GAZE_TARGET_M > 0:
        import math as _m
        ang = _m.atan(sep / GAZE_TARGET_M)
        conv_shift = iris_z_off * _m.tan(ang)
    else:
        conv_shift = 0.0
    return sep, y, z, eye_r, conv_shift


def _eye_metrics_for_shape(length, head_d, shape=None):
    """Eye placement with rest-identity traits applied consistently."""
    sep, y, z, eye_r, conv = _eye_metrics(length, head_d)
    sep *= _shape_trait(shape, "eye_sep", 1.0)
    eye_r *= _shape_trait(shape, "eye_size", 1.0)
    # Larger value means deeper-set eyes. The center moves back, but the iris
    # layers still project forward from that same center, preserving rig sync.
    z *= 1.0 / max(_shape_trait(shape, "eye_depth", 1.0), 0.75)
    if _head_age(shape) == "young":
        eye_r *= 1.03
    elif _head_age(shape) == "elder":
        eye_r *= 0.96
        y -= length * 0.006
    iris_z_off = eye_r * 0.75
    if GAZE_TARGET_M > 0:
        import math as _m
        conv = iris_z_off * _m.tan(_m.atan(sep / GAZE_TARGET_M))
    else:
        conv = 0.0
    return sep, y, z, eye_r, conv


def _gaze_offset(eye_r, weights):
    """Per-eye (dx, dy) iris offset from the four eyeLook* channels.

    Returned as ``(dx_left, dy_left, dx_right, dy_right)``. Side-naming
    follows ARKit / character POV: side=+1 in ``_eye_metrics`` is the
    character's LEFT eye. ``In`` is medial (toward nose), ``Out`` is
    lateral (toward temple).
    """
    from . import face_anim as _fa
    if weights is None:
        return 0.0, 0.0, 0.0, 0.0
    amp_x = eye_r * 0.45
    amp_y = eye_r * 0.40
    # Left eye: nose is at -X from its position, so In = -X, Out = +X.
    dx_l = amp_x * (_fa.w(weights, "eyeLookOutLeft")
                    - _fa.w(weights, "eyeLookInLeft"))
    dy_l = amp_y * (_fa.w(weights, "eyeLookUpLeft")
                    - _fa.w(weights, "eyeLookDownLeft"))
    # Right eye: nose is at +X from its position, so In = +X, Out = -X.
    dx_r = amp_x * (_fa.w(weights, "eyeLookInRight")
                    - _fa.w(weights, "eyeLookOutRight"))
    dy_r = amp_y * (_fa.w(weights, "eyeLookUpRight")
                    - _fa.w(weights, "eyeLookDownRight"))
    return dx_l, dy_l, dx_r, dy_r


def build_eyes(bones, shape=None) -> SkinnedMesh:
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, _ = _eye_metrics_for_shape(length, head_d, shape)
    # Truly spherical eyeball -- previous (rx, ry*0.95, rz*0.9) gave a
    # squashed ovoid that read as a flat disc once the lids covered the
    # poles. A real sclera is near-perfectly spherical.
    radii = (eye_r, eye_r, eye_r)
    chunks = [
        prim.ellipsoid((+sep, y, z), radii, head_idx, parent, rings=14, radial=20),
        prim.ellipsoid((-sep, y, z), radii, head_idx, parent, rings=14, radial=20),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_limbus(bones, shape=None) -> SkinnedMesh:
    """Dark limbal ring drawn behind the iris.

    Reads as the corneoscleral junction in real eyes and is the single
    biggest cue that an iris is part of a 3D ball rather than a printed
    decal. We render it as a slightly-larger dark disc placed *behind*
    the iris (smaller +Z offset) so the iris hides its center and only
    the outer annulus shows.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, conv = _eye_metrics_for_shape(length, head_d, shape)
    iris_r = eye_r * IRIS_RATIO
    limb_r = iris_r * LIMBUS_RATIO
    # Sits a hair behind the iris so only the outer ring peeks through.
    z_lim = z + eye_r * 0.74
    radii = (limb_r, limb_r, limb_r * 0.18)
    chunks = [
        prim.ellipsoid((+sep - conv, y, z_lim), radii, head_idx, parent,
                       rings=6, radial=18),
        prim.ellipsoid((-sep + conv, y, z_lim), radii, head_idx, parent,
                       rings=6, radial=18),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_iris(bones, shape=None, weights=None) -> SkinnedMesh:
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, conv = _eye_metrics_for_shape(length, head_d, shape)
    iris_r = eye_r * IRIS_RATIO
    radii = (iris_r, iris_r, iris_r * 0.42)
    z_iris = z + eye_r * 0.75
    dx_l, dy_l, dx_r, dy_r = _gaze_offset(eye_r, weights)
    chunks = [
        prim.ellipsoid((+sep - conv + dx_l, y + dy_l, z_iris),
                       radii, head_idx, parent, rings=8, radial=16),
        prim.ellipsoid((-sep + conv + dx_r, y + dy_r, z_iris),
                       radii, head_idx, parent, rings=8, radial=16),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_pupils(bones, shape=None, weights=None) -> SkinnedMesh:
    """Small dark pupil in the centre of each iris.

    Pupil diameter is ~30% of the iris (≈3mm of a 12mm iris under typical
    indoor light); larger reads as a cartoon stare.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, conv = _eye_metrics_for_shape(length, head_d, shape)
    iris_r = eye_r * IRIS_RATIO
    pr = iris_r * PUPIL_RATIO
    radii = (pr, pr, pr * 0.4)
    z_pup = z + eye_r * 0.86
    dx_l, dy_l, dx_r, dy_r = _gaze_offset(eye_r, weights)
    chunks = [
        prim.ellipsoid((+sep - conv + dx_l, y + dy_l, z_pup),
                       radii, head_idx, parent, rings=6, radial=12),
        prim.ellipsoid((-sep + conv + dx_r, y + dy_r, z_pup),
                       radii, head_idx, parent, rings=6, radial=12),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_collarette(bones, shape=None, weights=None) -> SkinnedMesh:
    """Faint darker ring inside the iris -- the iris collarette.

    Real irises have a visible boundary between the pupillary zone and
    the ciliary zone (the collarette) that breaks the colored disc into
    two concentric textures. Without it the iris reads as a single flat
    swatch of color. We render it as a slightly-darker disc placed in
    front of the iris but behind the pupil; the pupil hides its center
    so only the ring shows.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, conv = _eye_metrics_for_shape(length, head_d, shape)
    iris_r = eye_r * IRIS_RATIO
    coll_r = iris_r * 0.58
    radii = (coll_r, coll_r, coll_r * 0.16)
    z_col = z + eye_r * 0.81
    dx_l, dy_l, dx_r, dy_r = _gaze_offset(eye_r, weights)
    chunks = [
        prim.ellipsoid((+sep - conv + dx_l, y + dy_l, z_col),
                       radii, head_idx, parent, rings=5, radial=14),
        prim.ellipsoid((-sep + conv + dx_r, y + dy_r, z_col),
                       radii, head_idx, parent, rings=5, radial=14),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_caruncle(bones, shape=None) -> SkinnedMesh:
    """Small flesh-pink lacrimal caruncle at the medial canthus.

    The fleshy bump where the upper and lower eyelids meet at the inner
    eye corner. Tiny but very recognisable -- its absence is one of the
    things that makes simple CG eyes look mannequin-like.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, _conv = _eye_metrics_for_shape(length, head_d, shape)
    # Sit just medial of the eyeball, slightly forward of the eye center.
    cx = sep - eye_r * 0.95
    cy = y - eye_r * 0.05
    cz = z + eye_r * 0.55
    cr = eye_r * 0.18
    radii = (cr * 0.55, cr * 0.55, cr * 0.45)
    chunks = [
        prim.ellipsoid((+cx, cy, cz), radii, head_idx, parent, rings=4, radial=10),
        prim.ellipsoid((-cx, cy, cz), radii, head_idx, parent, rings=4, radial=10),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_catchlights(bones, shape=None, weights=None) -> SkinnedMesh:
    """Small specular catchlights on the upper-outer iris of each eye.

    A single bright speck per eye fakes the corneal highlight that real
    photography always picks up; without it the eyes read as dead glass.
    Position is the same on both eyes (upper-outer, *not* mirrored on the
    medial side) so it's read as one consistent off-axis light source --
    mirrored highlights look uncanny.
    """
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, conv = _eye_metrics_for_shape(length, head_d, shape)
    iris_r = eye_r * IRIS_RATIO
    cr = iris_r * 0.18
    radii = (cr, cr, cr * 0.5)
    # Upper-outer placement: shift up and to the lateral side per eye.
    dy = iris_r * 0.42
    dx = iris_r * 0.30
    z_cat = z + eye_r * 0.92
    gx_l, gy_l, gx_r, gy_r = _gaze_offset(eye_r, weights)
    chunks = [
        prim.ellipsoid((+sep - conv + dx + gx_l, y + dy + gy_l, z_cat), radii,
                       head_idx, parent, rings=4, radial=10),
        prim.ellipsoid((-sep + conv - dx + gx_r, y + dy + gy_r, z_cat), radii,
                       head_idx, parent, rings=4, radial=10),
    ]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_eyelids(bones, shape=None, weights=None) -> SkinnedMesh:
    """Skin-colored upper/lower eyelid folds that mask the spherical eyeballs.

    Driven by ARKit weights:
      - ``eyeBlink{Left,Right}``  -> upper lid drops to meet lower (full
        closure at w=1).
      - ``eyeWide{Left,Right}``   -> aperture x (1 + 0.30*w).
      - ``eyeSquint{Left,Right}`` -> aperture x (1 - 0.45*w), via lower
        lid raise (squint is a *lower-lid* tightening in FACS AU7).
      - ``cheekSquint{Left,Right}`` -> additional lower-lid lift (AU6,
        Duchenne marker; layered on top of squint).
    """
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    sep, y, z, eye_r, _conv = _eye_metrics_for_shape(length, head_d, shape)

    lid_z = z + eye_r * 0.88
    hx = eye_r * 1.36

    muscles = _fa.muscle_activations(weights)
    chunks = []
    # side=+1 is character's LEFT eye (camera right). ARKit naming is
    # from the *character's* point of view, matching this convention.
    for side, suffix in ((+1.0, "Left"), (-1.0, "Right")):
        x = side * sep
        blink   = _fa.w(weights, f"eyeBlink{suffix}")
        wide    = _fa.w(weights, f"eyeWide{suffix}")
        squint  = _fa.w(weights, f"eyeSquint{suffix}")
        cheek_s = _fa.w(weights, f"cheekSquint{suffix}")
        orbicularis = muscles[f"orbicularis_oculi_{suffix.lower()}"]

        # base aperture multiplier from wide/squint (excluding blink).
        aperture = 1.0 + 0.30 * wide - 0.38 * squint - 0.12 * orbicularis
        if aperture < 0.05:
            aperture = 0.05

        upper_y = y + eye_r * (0.58 * aperture)
        lower_y = y - eye_r * (0.50 * aperture)
        # cheekSquint pulls the *lower* lid up (orbicularis oculi, pars
        # orbitalis) without touching the upper lid -- the Duchenne tell.
        # Bigger amplitude makes the eye-narrowing read clearly when
        # paired with a smile.
        lower_y += eye_r * (0.28 * cheek_s + 0.14 * orbicularis)

        # Blink: upper lid sweeps down to meet lower; lower stays put.
        if blink > 0.0:
            target_y = lower_y + eye_r * 0.05
            upper_y = upper_y * (1.0 - blink) + target_y * blink

        upper_hy = eye_r * (0.46 - 0.08 * wide)
        lower_hy = eye_r * (0.26 + 0.08 * (squint + cheek_s) + 0.04 * orbicularis)

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
        # Canthus/socket cover: small skin-colored corner pads that hide the
        # exposed side of the spherical eyeball in profile while leaving the
        # iris/pupil and ARKit gaze offsets visible.
        chunks.append(prim.flat_patch(
            (x + side * hx * 0.88, y + eye_r * 0.02, lid_z - eye_r * 0.02),
            (eye_r * 0.14, eye_r * 0.36),
            head_idx, parent,
            normal=(side * 0.28, 0.00, 1.0),
            subdiv=(2, 4),
            thickness=0.0022,
        ))
        chunks.append(prim.flat_patch(
            (x - side * hx * 0.88, y - eye_r * 0.01, lid_z - eye_r * 0.02),
            (eye_r * 0.11, eye_r * 0.30),
            head_idx, parent,
            normal=(-side * 0.18, 0.00, 1.0),
            subdiv=(2, 4),
            thickness=0.0022,
        ))

    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_lips(bones, shape=None, weights=None) -> SkinnedMesh:
    """Upper + lower lip with a subtle cupid's bow and ARKit-driven offsets.

    Driven channels:
      - ``jawOpen``                          -> lip vertical separation.
      - ``mouthSmile{Left,Right}``           -> per-side corner lift.
      - ``mouthFrown{Left,Right}``           -> per-side corner drop.
      - ``mouthDimple{Left,Right}``          -> corner pulled back/in.
      - ``mouthStretch{Left,Right}``         -> corner widened sideways.
      - ``mouthPress{Left,Right}``           -> upper/lower lips squeezed
        together + thinner ry.
      - ``mouthPucker``                      -> lips pulled forward + in.
      - ``mouthFunnel``                      -> lips pushed forward, ring
        opens (additive with jawOpen).
      - ``mouthLeft`` / ``mouthRight``       -> whole-mouth lateral shift.
      - ``mouthUpperUp{Left,Right}``         -> per-side upper lip raise.
      - ``mouthLowerDown{Left,Right}``       -> per-side lower lip drop.
      - ``noseSneer{Left,Right}``            -> per-side upper lip raise
        (adds to mouthUpperUp; AU9 anchors the snarl).
    """
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, r, head_w, head_d = info
    muscles = _fa.muscle_activations(weights)
    y_mouth = length * H_MOUTH
    z = head_d * 0.84
    fullness = _shape_trait(shape, "lip_fullness", 1.0)
    if _head_age(shape) == "young":
        fullness *= 1.04
    elif _head_age(shape) == "elder":
        fullness *= 0.82

    jaw_open    = _fa.w(weights, "jawOpen")
    mouth_close = _fa.w(weights, "mouthClose")
    jaw_left    = _fa.w(weights, "jawLeft")
    jaw_right   = _fa.w(weights, "jawRight")
    jaw_forward = _fa.w(weights, "jawForward")
    pucker      = _fa.w(weights, "mouthPucker")
    funnel      = _fa.w(weights, "mouthFunnel")
    mouth_left  = _fa.w(weights, "mouthLeft")
    mouth_right = _fa.w(weights, "mouthRight")
    lateral     = ((mouth_left - mouth_right) * length * 0.020
                   + (jaw_left - jaw_right) * length * 0.010)

    smile_l  = _fa.w(weights, "mouthSmileLeft")
    smile_r  = _fa.w(weights, "mouthSmileRight")
    frown_l  = _fa.w(weights, "mouthFrownLeft")
    frown_r  = _fa.w(weights, "mouthFrownRight")
    dimple_l = _fa.w(weights, "mouthDimpleLeft")
    dimple_r = _fa.w(weights, "mouthDimpleRight")
    stretch_l = _fa.w(weights, "mouthStretchLeft")
    stretch_r = _fa.w(weights, "mouthStretchRight")
    press_l  = _fa.w(weights, "mouthPressLeft")
    press_r  = _fa.w(weights, "mouthPressRight")
    upUp_l   = _fa.w(weights, "mouthUpperUpLeft")
    upUp_r   = _fa.w(weights, "mouthUpperUpRight")
    lowDn_l  = _fa.w(weights, "mouthLowerDownLeft")
    lowDn_r  = _fa.w(weights, "mouthLowerDownRight")
    sneer_l  = _fa.w(weights, "noseSneerLeft")
    sneer_r  = _fa.w(weights, "noseSneerRight")
    roll_upper = _fa.w(weights, "mouthRollUpper")
    roll_lower = _fa.w(weights, "mouthRollLower")
    shrug_upper = _fa.w(weights, "mouthShrugUpper")
    shrug_lower = _fa.w(weights, "mouthShrugLower")
    avg_smile = 0.5 * (smile_l + smile_r)
    avg_press = 0.5 * (press_l + press_r)
    mouth_tension = muscles["mouth_tension"]
    orbicularis_oris = muscles["orbicularis_oris"]
    jaw_drop = max(0.0, jaw_open - 0.45 * mouth_close)

    # Vertical separation: jawOpen drops lower lip, mouthClose pulls
    # upper down to meet it (handled implicitly via reduced upper Y).
    open_amt = length * 0.060 * jaw_drop + length * 0.016 * funnel

    # Upper lip: two symmetric lobes; per-side raise from upUp + sneer.
    up_half_sep = length * 0.023
    lip_volume = 1.0 + 0.08 * pucker + 0.05 * funnel
    press_thin = 1.0 - 0.30 * avg_press - 0.10 * mouth_close
    up_rx = length * (0.037 - 0.009 * pucker
                      + 0.008 * (stretch_l + stretch_r) * 0.5
                      - 0.004 * orbicularis_oris)
    # Roll tucks the upper lip inward (hides the red vermillion zone over
    # the teeth); shrug pushes it forward+up (AU17 mentalis for the lower,
    # AU16 levator for the upper).
    up_ry = length * 0.0068 * fullness * max(0.40, press_thin - 0.25 * roll_upper)
    up_rz = length * (0.0063 + 0.006 * pucker + 0.004 * funnel
                      + 0.0025 * avg_press
                      - 0.003 * roll_upper) * fullness * lip_volume
    up_y = (y_mouth + length * 0.010 + open_amt * 0.35
            + length * 0.004 * avg_smile - length * 0.010 * mouth_close
            - length * 0.014 * roll_upper
            + length * 0.008 * shrug_upper)
    lip_z_forward = length * (0.005 * pucker + 0.006 * funnel
                              + 0.006 * jaw_forward
                              - 0.012 * roll_upper
                              + 0.006 * shrug_upper)

    up_L_y = up_y + length * (0.014 * (upUp_l + 0.60 * sneer_l)
                              + 0.010 * smile_l - 0.004 * frown_l)
    up_R_y = up_y + length * (0.014 * (upUp_r + 0.60 * sneer_r)
                              + 0.010 * smile_r - 0.004 * frown_r)
    upper_L = prim.ellipsoid(
        (+up_half_sep + lateral, up_L_y, z + lip_z_forward),
        (up_rx, up_ry, up_rz),
        head_idx, parent, rings=6, radial=14,
    )
    upper_R = prim.ellipsoid(
        (-up_half_sep + lateral, up_R_y, z + lip_z_forward),
        (up_rx, up_ry, up_rz),
        head_idx, parent, rings=6, radial=14,
    )

    # Mouth corners: per-side smile / frown / dimple / stretch.
    base_corner_x = length * 0.060
    corner_rx = length * (0.010 + 0.0025 * mouth_tension)
    corner_ry = length * (0.0068 * max(0.58, 1.0 - 0.25 * avg_press))
    corner_rz = length * (0.0048 + 0.0016 * mouth_tension)

    def _corner(side_sign, smile, frown, dimple, stretch):
        # Wider amplitude than the legacy preset so a w=0.7 smile reads
        # plainly in a thumbnail-sized render. Frown amplitude is kept
        # smaller because mouth-corner-down past a few mm starts to
        # caricature.
        lift = length * (0.038 * smile - 0.030 * frown + 0.006 * jaw_drop)
        # Dimple pulls corner back (-Z) and slightly inward (X toward 0).
        # Smile also pulls the corner laterally/upward along zygomaticus,
        # while pucker/funnel recruit orbicularis oris and pull inward.
        x = side_sign * (base_corner_x + length * 0.014 * stretch
                         + length * 0.010 * smile
                         - length * 0.006 * dimple
                         - length * 0.012 * orbicularis_oris)
        y = y_mouth + length * 0.001 + lift
        z_off = (-length * 0.002 - length * 0.006 * dimple
                 + length * 0.004 * (pucker + funnel + jaw_forward))
        return prim.ellipsoid(
            (x + lateral, y, z + z_off),
            (corner_rx, corner_ry, corner_rz),
            head_idx, parent, rings=5, radial=10,
        )

    corner_L = _corner(+1.0, smile_l, frown_l, dimple_l, stretch_l)
    corner_R = _corner(-1.0, smile_r, frown_r, dimple_r, stretch_r)

    # Lower lip: wider pillow. jawOpen + lowerDown drop it; pucker
    # narrows and pushes forward; press thins it.
    avg_lowDn = 0.5 * (lowDn_l + lowDn_r)
    low_y = (y_mouth - length * 0.010 - open_amt * 0.65
             - length * 0.014 * avg_lowDn + length * 0.003 * avg_smile
             + length * 0.012 * mouth_close
             + length * 0.016 * roll_lower
             + length * 0.008 * shrug_lower)
    low_rx = length * (0.070 - 0.018 * pucker
                       + 0.011 * (stretch_l + stretch_r) * 0.5
                       - 0.006 * orbicularis_oris)
    low_ry = length * 0.0088 * fullness * max(
        0.40, 1.0 - 0.34 * avg_press - 0.28 * roll_lower)
    low_rz = length * (0.0072 + 0.005 * pucker + 0.004 * funnel
                       + 0.0025 * avg_press
                       - 0.004 * roll_lower) * fullness * lip_volume
    lower = prim.ellipsoid(
        (lateral, low_y, z + length * 0.002 + length * 0.008 * pucker
         + length * 0.004 * funnel + length * 0.006 * jaw_forward
         - length * 0.012 * roll_lower
         + length * 0.006 * shrug_lower),
        (low_rx, low_ry, low_rz),
        head_idx, parent, rings=6, radial=20,
    )
    chunks = [upper_L, upper_R, corner_L, corner_R, lower]
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_expression_folds(bones, shape=None, weights=None) -> SkinnedMesh:
    """Subtle dynamic crease/shadow patches for active facial muscles."""
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, _head_w, head_d = info
    m = _fa.muscle_activations(weights)
    brow_compress = 0.5 * (m["corrugator_left"] + m["corrugator_right"])
    brow_raise = 0.5 * (m["frontalis_left"] + m["frontalis_right"])
    if max(m["smile"], m["frown"], m["sneer"], m["mouth_tension"],
           m["chin_tension"], brow_compress, brow_raise) < 0.04:
        return _empty_mesh()

    chunks = []
    for side, suffix in ((+1.0, "left"), (-1.0, "right")):
        zyg = m[f"zygomatic_{suffix}"]
        dep = m[f"depressor_anguli_{suffix}"]
        sneer = m[f"nasalis_{suffix}"]
        corr = m[f"corrugator_{suffix}"]
        orb = m[f"orbicularis_oculi_{suffix}"]
        fold = max(0.0, 0.65 * zyg + 0.55 * sneer + 0.30 * dep)
        if fold > 0.045:
            for x_base, y_base, seg_len, normal_y in (
                (0.046, 0.382, 0.022, -0.06),
                (0.062, 0.342, 0.020, -0.20),
            ):
                chunks.append(prim.flat_patch(
                    (side * length * x_base,
                     length * (y_base - 0.018 * dep + 0.010 * zyg),
                     head_d * 0.872),
                    (length * 0.0026,
                     length * (seg_len + 0.004 * fold)),
                    head_idx, parent,
                    normal=(side * 0.42, normal_y, 1.0),
                    subdiv=(1, 4),
                    thickness=0.00075,
                ))
        mouth_pin = max(dep, m[f"lip_press_{suffix}"], m["mouth_tension"])
        if mouth_pin > 0.055:
            chunks.append(prim.flat_patch(
                (side * length * 0.066,
                 length * (0.305 - 0.020 * dep),
                 head_d * 0.864),
                (length * 0.0026, length * 0.014),
                head_idx, parent,
                normal=(side * 0.30, -0.18, 1.0),
                subdiv=(1, 3),
                thickness=0.0007,
            ))
        if corr > 0.12:
            chunks.append(prim.flat_patch(
                (side * length * 0.030,
                 length * (H_BROW - 0.030),
                head_d * 0.795),
                (length * 0.0025, length * 0.018),
                head_idx, parent,
                normal=(side * 0.12, -0.25, 1.0),
                subdiv=(1, 3),
                thickness=0.0007,
            ))
        if orb > 0.10:
            # AU6/AU7 orbicularis oculi: short radiating crow's-feet at the
            # outer canthus. These are expression wrinkles, so young faces get
            # them only while smiling/squinting, unlike static age lines.
            for yoff, normal_y, scale in (
                (0.012,  0.14, 0.82),
                (0.000,  0.00, 1.00),
                (-0.012, -0.16, 0.78),
            ):
                chunks.append(prim.flat_patch(
                    (side * length * 0.198,
                     length * (H_EYE + yoff),
                     head_d * 0.902),
                    (length * (0.008 + 0.004 * orb) * scale,
                     length * 0.0018),
                    head_idx, parent,
                    normal=(side * 0.30, normal_y, 1.0),
                    subdiv=(4, 1),
                    thickness=0.00065,
                ))

    if brow_compress > 0.08:
        # AU4 corrugator/procerus: short vertical glabellar furrows plus a
        # shallow horizontal fold just above the nasal root. Wrinkles form
        # perpendicular to compression, so brow squeeze creates vertical
        # folds near midline and a transverse root crease.
        for x in (-0.010, 0.010):
            chunks.append(prim.flat_patch(
                (x * length,
                 length * (H_BROW - 0.050),
                 head_d * 0.830),
                (length * 0.0022, length * (0.018 + 0.006 * brow_compress)),
                head_idx, parent,
                normal=(0.0, -0.20, 1.0),
                subdiv=(1, 4),
                thickness=0.0007,
            ))
        chunks.append(prim.flat_patch(
            (0.0,
             length * (H_BROW - 0.018),
             head_d * 0.852),
            (length * (0.034 + 0.012 * brow_compress),
             length * 0.0025),
            head_idx, parent,
            normal=(0.0, -0.08, 1.0),
            subdiv=(6, 1),
            thickness=0.0007,
        ))

    if m["chin_tension"] > 0.06:
        chunks.append(prim.flat_patch(
            (0.0, length * (0.236 - 0.012 * m["jaw_open"]), head_d * 0.850),
            (length * 0.022, length * 0.004),
            head_idx, parent,
            normal=(0.0, -0.16, 1.0),
            subdiv=(4, 1),
            thickness=0.0007,
        ))

    if not chunks:
        return _empty_mesh()
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_mouth_cavity(bones, shape=None, weights=None) -> SkinnedMesh:
    """Dark oral aperture behind the lips.

    The lips are separate ellipsoids, so without a darker aperture the ARKit
    jaw and smile channels move visible pieces but do not read as an opening.
    This patch stays behind the lip primitives and scales from a neutral slit
    to an oval opening for ``jawOpen`` / ``mouthFunnel``.
    """
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, _head_w, head_d = info

    jaw_open = _fa.w(weights, "jawOpen")
    funnel = _fa.w(weights, "mouthFunnel")
    pucker = _fa.w(weights, "mouthPucker")
    smile = 0.5 * (_fa.w(weights, "mouthSmileLeft")
                   + _fa.w(weights, "mouthSmileRight"))
    stretch = 0.5 * (_fa.w(weights, "mouthStretchLeft")
                     + _fa.w(weights, "mouthStretchRight"))
    press = 0.5 * (_fa.w(weights, "mouthPressLeft")
                   + _fa.w(weights, "mouthPressRight"))
    close = _fa.w(weights, "mouthClose")

    openness = max(0.0, jaw_open + 0.45 * funnel - 0.60 * close - 0.45 * press)
    half_x = length * (0.050 + 0.014 * smile + 0.020 * stretch - 0.018 * pucker)
    half_x = max(length * 0.028, half_x)
    half_y = length * (0.0025 + 0.026 * openness + 0.004 * funnel)
    center_y = length * H_MOUTH - length * 0.004 - length * 0.010 * jaw_open
    center_z = head_d * (0.835 + 0.010 * pucker)

    chunk = prim.flat_patch(
        (0.0, center_y, center_z),
        (half_x, half_y),
        head_idx, parent,
        normal=(0.0, 0.0, 1.0),
        subdiv=(8, 3),
        thickness=0.0015,
    )
    v, n, ba, bb, w, idx = chunk
    v, n = v @ R.T, n @ R.T
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

    # Eye-corner crow's-feet and nasolabial fold, mirrored L/R.
    for side in (+1.0, -1.0):
        age_lines = (
            (0.014,  0.14, 0.74),
            (0.000,  0.00, 0.95),
            (-0.014, -0.16, 0.70),
        )
        for yoff, normal_y, scale in age_lines:
            chunks.append(prim.flat_patch(
                (side * length * 0.190,
                 length * (H_EYE + yoff),
                 head_d * 0.905),
                (length * (0.010 + 0.006 * strength) * scale,
                 length * 0.0018 * strength),
                head_idx, parent,
                normal=(side * 0.26, normal_y, 1.0),
                subdiv=(4, 1),
                thickness=0.0008,
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


def build_teeth(bones, shape=None, weights=None) -> SkinnedMesh:
    """Upper + lower dental arches behind the lips.

    Background: plausible speech visually *requires* visible teeth on open
    vowels and fricatives (Ezzat & Poggio 2002; Edwards et al. 2016 "JALI").
    Without them the mouth reads as an empty dark hole regardless of how
    accurate the lip blendshapes are.

    Geometry: two horseshoe-shaped bands of incisor-shaped flat patches.
    - upper arch is rigidly skinned to the head bone (so it tracks the skull,
      not the jaw) and positioned behind the upper lip
    - lower arch translates downward by ``jawOpen`` so the bite opens as the
      jaw drops, and laterally by ``jawLeft/Right`` for off-axis jaw motion
    - visibility is gated by lip state (see build_lips for the reciprocal):
      heavy mouthClose / mouthPress culls the bite-edge exposure further back
      so the teeth stop poking through a pressed lip

    The teeth are intentionally *slightly* recessed (z toward the skull) so
    on a neutral mouth only a thin white bite line can appear, never a
    toothy grimace.
    """
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, _head_w, head_d = info

    jaw_open = _fa.w(weights, "jawOpen")
    mouth_close = _fa.w(weights, "mouthClose")
    press = 0.5 * (_fa.w(weights, "mouthPressLeft")
                   + _fa.w(weights, "mouthPressRight"))
    jaw_left = _fa.w(weights, "jawLeft")
    jaw_right = _fa.w(weights, "jawRight")
    jaw_forward = _fa.w(weights, "jawForward")
    roll_upper = _fa.w(weights, "mouthRollUpper")
    roll_lower = _fa.w(weights, "mouthRollLower")
    funnel = _fa.w(weights, "mouthFunnel")
    pucker = _fa.w(weights, "mouthPucker")

    # Effective bite opening: jawOpen drops the mandible, mouthClose pulls
    # it back up, mouthPress seals the bite.
    bite = max(0.0, jaw_open - 0.75 * mouth_close - 0.35 * press)

    # Upper-lip exposure: mouthUpperUp* / noseSneer* raise the upper lip and
    # reveal the incisors even without jaw motion (classic "snarl"). Lower
    # lip analogue lets mouthLowerDown* drop the lip and expose the lower
    # bite.
    upper_reveal = 0.5 * (_fa.w(weights, "mouthUpperUpLeft")
                          + _fa.w(weights, "mouthUpperUpRight")) + \
                   0.30 * (_fa.w(weights, "noseSneerLeft")
                           + _fa.w(weights, "noseSneerRight"))
    lower_reveal = 0.5 * (_fa.w(weights, "mouthLowerDownLeft")
                          + _fa.w(weights, "mouthLowerDownRight"))
    smile_reveal = 0.5 * (_fa.w(weights, "mouthSmileLeft")
                          + _fa.w(weights, "mouthSmileRight"))
    stretch_reveal = 0.5 * (_fa.w(weights, "mouthStretchLeft")
                            + _fa.w(weights, "mouthStretchRight"))
    # A sealed bite with no independent lip raise should hide the teeth
    # entirely (they sit behind the closed lip surface but any z-fighting
    # or near-contact artefact reads as a poke-through at this primitive
    # density).
    # roll_upper/roll_lower *by themselves* mean the lip rolls over the
    # bite -- the teeth should stay hidden. They only reveal teeth when
    # combined with jaw_open (handled via the bite-line y shift above).
    visible = (bite + 0.8 * upper_reveal + 0.7 * lower_reveal
               + 0.6 * smile_reveal + 0.6 * stretch_reveal)
    if visible < 0.075:
        return _empty_mesh()
    if mouth_close > 0.5 and visible < 0.12:
        return _empty_mesh()

    # Overall arch geometry. The dental arch in a human skull is a
    # horseshoe whose width is roughly 0.70x the mouth width and whose
    # depth (front-to-back) is ~1.0x its width. Numbers here are scaled
    # fractions of head_length to match the rest of the face primitives.
    y_mouth = length * H_MOUTH
    z_front = head_d * 0.78   # slightly recessed vs lips (lips sit at 0.84)
    arch_w = length * 0.062   # half-width of the arch at the molars
    arch_d = length * 0.055   # front-to-back depth of the horseshoe

    # Lateral jaw shift (AU26 asymmetric jaw motion).
    lower_lateral = (jaw_left - jaw_right) * length * 0.012
    forward_shift = jaw_forward * length * 0.008

    # Upper arch y: just below the upper lip line. mouthRollUpper tucks the
    # upper lip under, which *reveals* upper incisor edge -- so roll adds to
    # exposure. pucker/funnel narrow the arch laterally (cheeks pull in).
    up_y = y_mouth + length * (0.004 - 0.008 * roll_upper)
    up_z = z_front + forward_shift - length * 0.002 * (pucker + funnel)

    # Lower arch y: tracks the jaw -- drops with jawOpen, tucks with
    # mouthRollLower (hides lower bite, e.g. FV viseme biting the lip).
    low_drop = length * (0.034 * bite + 0.005 * funnel)
    low_y = y_mouth - length * 0.004 - low_drop - length * 0.012 * roll_lower
    low_z = z_front + forward_shift - length * 0.002 * (pucker + funnel)

    # Number of "tooth" segments per arch -- front 10 in humans, but
    # visually 8-9 segments reads cleanly at this primitive density.
    n_teeth = 9
    # Half-sweep angle across the horseshoe (radians). Wider angle -> the
    # arch looks more like a semicircle.
    half_sweep = math.radians(68.0)

    def _tooth_placements(jaw_scale_x):
        out = []
        for i in range(n_teeth):
            u = (i + 0.5) / n_teeth * 2.0 - 1.0   # -1..+1
            theta = u * half_sweep
            # Horseshoe: x = w*sin(theta); z = -d*(1 - cos(theta))
            x = math.sin(theta) * arch_w * jaw_scale_x
            z_off = -(1.0 - math.cos(theta)) * arch_d
            # Individual tooth radii -- front incisors slightly larger, rear
            # teeth taper down (canines -> premolars -> molars).
            front = 1.0 - 0.55 * (u * u)
            out.append((x, z_off, front))
        return out

    # Upper arch narrows slightly with pucker/funnel (cheek compression),
    # lower arch follows jaw_forward -> forward cant.
    up_scale  = 1.0 - 0.15 * pucker - 0.10 * funnel
    low_scale = 1.0 - 0.12 * pucker - 0.08 * funnel

    chunks: list = []

    for x_off, z_off, front in _tooth_placements(up_scale):
        # Each upper incisor: small flattened ellipsoid. The bite edge sits
        # at -y relative to the tooth center; the gum end is at +y.
        tooth_ry = length * (0.011 + 0.002 * front)
        tooth_rx = length * (0.008 + 0.0015 * front)
        tooth_rz = length * (0.008 + 0.002 * front)
        chunks.append(prim.ellipsoid(
            (x_off, up_y - tooth_ry * 0.55, up_z + z_off),
            (tooth_rx, tooth_ry, tooth_rz),
            head_idx, parent, rings=4, radial=8,
        ))

    for x_off, z_off, front in _tooth_placements(low_scale):
        tooth_ry = length * (0.010 + 0.002 * front)
        tooth_rx = length * (0.0075 + 0.0015 * front)
        tooth_rz = length * (0.0075 + 0.002 * front)
        chunks.append(prim.ellipsoid(
            (x_off + lower_lateral, low_y + tooth_ry * 0.55,
             low_z + z_off),
            (tooth_rx, tooth_ry, tooth_rz),
            head_idx, parent, rings=4, radial=8,
        ))

    if not chunks:
        return _empty_mesh()
    chunks = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def build_tongue(bones, shape=None, weights=None) -> SkinnedMesh:
    """Tongue body visible behind the teeth on open vowels and lingual
    consonants (L, TH, N, D, T).

    Driven channels:
      - ``jawOpen``    -> tongue descends and flattens along with the floor
        of the mouth (it is anchored to the mandible hyoid anatomically,
        which tracks the jaw).
      - ``tongueOut``  -> tongue tip pushes forward past the incisal edge
        (AU19 / Preston-Blair "L" and "TH" visemes).

    The tongue is a single flattened ellipsoid: accurate enough to read
    as volume behind the teeth without introducing a deformable mesh.
    """
    from . import face_anim as _fa
    info = _head_info(bones, shape)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, _head_w, head_d = info

    jaw_open = _fa.w(weights, "jawOpen")
    tongue_out = _fa.w(weights, "tongueOut")
    mouth_close = _fa.w(weights, "mouthClose")

    # Fully hide the tongue if the bite is closed -- it should never poke
    # through sealed lips.
    bite = max(0.0, jaw_open - 0.80 * mouth_close)
    if bite < 0.03 and tongue_out < 0.02:
        return _empty_mesh()

    y_mouth = length * H_MOUTH
    # Tongue sits on the floor of the mouth, just above the lower incisors.
    cy = y_mouth - length * (0.018 + 0.022 * bite)
    # Forward position: sits back from the bite line unless tongue_out is
    # nonzero (the L/TH visemes push the tip to / past the incisal edge).
    cz = head_d * (0.77 + 0.04 * tongue_out) - length * 0.004

    rx = length * (0.036 + 0.003 * bite)
    ry = length * (0.009 + 0.004 * bite)   # flatter when mouth is wide open
    rz = length * (0.036 + 0.020 * tongue_out)

    chunk = prim.ellipsoid(
        (0.0, cy, cz), (rx, ry, rz),
        head_idx, parent, rings=6, radial=14,
    )
    v, n, ba, bb, w, idx = chunk
    v, n = v @ R.T, n @ R.T
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def _empty_mesh() -> SkinnedMesh:
    z_f = np.zeros((0, 3), np.float32)
    return SkinnedMesh(
        z_f, z_f,
        np.zeros((0, 2), np.int32),
        np.zeros((0, 2), np.float32),
        np.zeros((0,), np.uint32),
    )
