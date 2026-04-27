"""Lofted torso mesh — phase 1 of the dense-mesh migration.

Replaces the chest+spine+pelvis capsule trio with a single watertight
mesh whose rings carry per-bone weights and whose cross-sections are
non-circular (X stretch from shoulder_w/hip_w, narrower Z so the
torso reads as a human chest rather than a tube). The loft samples a
chain of N rings between the pelvis bottom and the chest top, with
each ring assigned a smooth blend of two adjacent bones (LBS).

Design choices documented in `notes/dense_mesh_migration_plan.md`.
"""

from __future__ import annotations

import math

import numpy as np

from . import mathx
from .mesh import SkinnedMesh, _find


# Number of rings sampled between pelvis bottom and chest top. More =
# smoother silhouette but more verts. 28 rings × 32 segments = 896 verts
# for the side wall plus two end caps; comfortable for our shader.
RINGS_PELVIS = 6
RINGS_SPINE = 8
RINGS_CHEST = 10
RADIAL = 32

# Cross-section aspect: X (lateral) is wider than Z (front-back). Humans
# in T-pose seen from above are an oval, not a circle. The numbers below
# are 1.0 (full lateral radius) and 0.65 (front-back depth as fraction of
# lateral). The chest can stretch X further by shoulder_w; pelvis by
# hip_w; spine sits in between.
ASPECT_Z = 0.65


RADIAL_LIMB = 16
RINGS_PER_LIMB_BONE = 6


# Pants loft (cycle 54). Tunable per-ring counts kept low so the pants
# add ~1k extra verts on top of the body, well within shader budget.
RADIAL_PANTS = 16
RINGS_PANTS_THIGH = 7
RINGS_PANTS_SHIN = 7
RINGS_PANTS_WAIST = 4


def _limb_profile(bone_name, gender):
    """Per-bone radius profile along the bone's length (t in [0,1]).
    Mirrors the shape of `human/mesh.py::_limb_profile` so the loft's
    silhouette matches the previous capsule version with anatomical
    bulges (biceps, calf gastrocnemius)."""
    male = (gender == "male")
    female = (gender == "female")
    if bone_name.startswith("uarm_"):
        bulge = 0.10 if male else (0.04 if female else 0.05)
        taper = 0.22 if not female else 0.18
        return lambda t, b=bulge, k=taper: 1.00 - k * t + b * math.sin(math.pi * t)
    if bone_name.startswith("farm_"):
        scale = 1.00 if not female else 0.94
        return lambda t, s=scale: s * (0.98 - 0.30 * t + 0.08 * math.sin(math.pi * t))
    if bone_name.startswith("thigh_"):
        if female:
            return lambda t: 1.10 - 0.14 * t + 0.06 * math.sin(math.pi * t)
        if male:
            return lambda t: 1.00 - 0.26 * t + 0.08 * math.sin(math.pi * t)
        return lambda t: 1.00 - 0.24 * t + 0.04 * math.sin(math.pi * t)
    if bone_name.startswith("shin_"):
        if male:
            return lambda t: 0.86 - 0.22 * t + 0.36 * math.sin(math.pi * t)
        if female:
            return lambda t: 0.82 - 0.22 * t + 0.24 * math.sin(math.pi * t)
        return lambda t: 0.84 - 0.22 * t + 0.30 * math.sin(math.pi * t)
    return lambda t: 1.0


def build_limb_loft(bones, shape, chain_names) -> SkinnedMesh | None:
    """Continuous lofted limb across a chain of bones (e.g.
    ``['uarm_L', 'farm_L']`` or ``['thigh_L', 'shin_L']``).

    Cycle 55 rewrite: the previous version generated the loft in the
    first bone's local +Y space and relied on each subsequent bone's
    matrix to rotate it into place — but the rig's bone matrices at
    rest are translation-only (no rotation, see
    ``skeleton.compute_bone_matrices``), so the chain's rings ended up
    stacked along world +Y instead of along the limb axis ("flat
    plates"). Two fixes:

    1. Build each ring's center in WORLD space using each bone's actual
       axis (``tip / |tip|``).
    2. Store every vertex as ``world_pos - weighted_ref`` where
       ``weighted_ref = wa * head_a_world + wb * head_b_world``. At rest
       the LBS shader sums ``wa*M_a*v + wb*M_b*v`` and recovers
       ``world_pos`` exactly. At a bent joint ``R_a != R_b`` and the
       blend produces smooth deformation through the seam zone — the
       same trick ``build_torso_loft`` uses for pelvis/spine/chest.

    All bones in the chain are assumed aligned (T-pose limbs) so a
    single perpendicular cross-section frame works for every ring.
    """
    chain = []
    for name in chain_names:
        idx = _find(bones, name)
        if idx < 0:
            return None
        chain.append((idx, bones[idx]))
    if not chain:
        return None
    gender = getattr(shape, "gender", "neutral") if shape else "neutral"

    chain_meta = []
    for idx, (name, parent, head, tip, r) in chain:
        tip_v = np.asarray(tip, dtype=np.float32)
        length = float(np.linalg.norm(tip_v))
        if length < 1e-5:
            continue
        head_world = _world_pos(bones, idx)
        # In rest pose with no bone rotations, world tip = world head + tip.
        axis = tip_v / length
        chain_meta.append({
            "idx": idx, "name": name, "length": length,
            "radius": float(r), "axis": axis,
            "head_world": head_world,
            "profile": _limb_profile(name, gender),
        })

    if not chain_meta:
        return None

    # Cross-section frame: one set of perpendicular vectors orthogonal to
    # the chain's primary axis. Aligned-chain assumption → same frame
    # works for every ring.
    primary_axis = chain_meta[0]["axis"]
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(float(np.dot(up, primary_axis))) > 0.9:
        up = np.array([1, 0, 0], dtype=np.float32)
    perp1 = np.cross(primary_axis, up)
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
    perp2 = np.cross(primary_axis, perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

    verts = []
    norms = []
    bones_a_list = []
    bones_b_list = []
    weights_list = []
    n_rings_total = 0
    seam_zone = 0.30

    for k, meta in enumerate(chain_meta):
        idx = meta["idx"]
        length = meta["length"]
        radius = meta["radius"]
        axis = meta["axis"]
        profile = meta["profile"]
        head_world = meta["head_world"]
        prev_meta = chain_meta[k - 1] if k > 0 else None
        next_meta = chain_meta[k + 1] if k < len(chain_meta) - 1 else None

        # Proximal cap on the FIRST bone only (closes the shoulder / hip
        # seam against the torso loft).
        if k == 0:
            r0 = radius * profile(0.0)
            cap_rings = 4
            for ci in range(cap_rings + 1):
                phi_cap = 0.5 * math.pi * (ci / cap_rings)
                # Inset along -axis (toward the torso) by sin(phi)*r*0.45
                cap_offset = -math.sin(phi_cap) * r0 * 0.45
                rr_cap = math.cos(phi_cap) * r0
                ring_center_world = head_world + cap_offset * axis
                for j in range(RADIAL_LIMB):
                    th = (j / RADIAL_LIMB) * 2 * math.pi
                    cp, sp = math.cos(th), math.sin(th)
                    v_world = ring_center_world + rr_cap * (cp * perp1 + sp * perp2)
                    n_world = (cp * perp1 + sp * perp2 - math.sin(phi_cap) * axis)
                    verts.append(tuple(v_world - head_world))
                    norms.append(tuple(n_world))
                    bones_a_list.append(idx)
                    bones_b_list.append(idx)
                    weights_list.append((1.0, 0.0))
                n_rings_total += 1

        # Cylindrical body of this bone, ring centers in world along
        # head_world → tip_world.
        for ri in range(RINGS_PER_LIMB_BONE + 1):
            t = ri / RINGS_PER_LIMB_BONE
            ring_center_world = head_world + (t * length) * axis
            rr = radius * profile(t)
            if t < seam_zone and prev_meta is not None:
                wa = 0.5 + 0.5 * (t / seam_zone)
                anchor, blend_b = idx, prev_meta["idx"]
                ref_world = wa * head_world + (1.0 - wa) * prev_meta["head_world"]
            elif t > 1.0 - seam_zone and next_meta is not None:
                wa = 0.5 + 0.5 * ((1.0 - t) / seam_zone)
                anchor, blend_b = idx, next_meta["idx"]
                ref_world = wa * head_world + (1.0 - wa) * next_meta["head_world"]
            else:
                wa, anchor, blend_b = 1.0, idx, idx
                ref_world = head_world

            # Finite-diff radius slope for the axial component of the normal.
            t0 = max(0.0, t - 1.0 / RINGS_PER_LIMB_BONE)
            t1 = min(1.0, t + 1.0 / RINGS_PER_LIMB_BONE)
            drdt = (radius * profile(t1) - radius * profile(t0)) / max(
                (t1 - t0) * length, 1e-5)

            for j in range(RADIAL_LIMB):
                th = (j / RADIAL_LIMB) * 2 * math.pi
                cp, sp = math.cos(th), math.sin(th)
                v_world = ring_center_world + rr * (cp * perp1 + sp * perp2)
                # Outward radial normal minus axial slope (same construction
                # as the original; sign of axis component depends on whether
                # the bone narrows or widens with t).
                n_world = (cp * perp1 + sp * perp2 - drdt * axis)
                verts.append(tuple(v_world - ref_world))
                norms.append(tuple(n_world))
                bones_a_list.append(anchor)
                bones_b_list.append(blend_b)
                weights_list.append((wa, 1.0 - wa))
            n_rings_total += 1

        # Distal cap on the LAST bone (closes the wrist / ankle).
        if k == len(chain_meta) - 1:
            r1 = radius * profile(1.0)
            tip_world = head_world + length * axis
            cap_rings = 4
            for ci in range(1, cap_rings + 1):
                phi_cap = 0.5 * math.pi * (ci / cap_rings)
                cap_offset = math.sin(phi_cap) * r1 * 0.6
                rr_cap = math.cos(phi_cap) * r1
                ring_center_world = tip_world + cap_offset * axis
                for j in range(RADIAL_LIMB):
                    th = (j / RADIAL_LIMB) * 2 * math.pi
                    cp, sp = math.cos(th), math.sin(th)
                    v_world = ring_center_world + rr_cap * (cp * perp1 + sp * perp2)
                    n_world = (cp * perp1 + sp * perp2 + math.sin(phi_cap) * axis)
                    verts.append(tuple(v_world - head_world))
                    norms.append(tuple(n_world))
                    bones_a_list.append(idx)
                    bones_b_list.append(idx)
                    weights_list.append((1.0, 0.0))
                n_rings_total += 1

    if not verts:
        return None

    verts_a = np.asarray(verts, dtype=np.float32)
    norms_a = np.asarray(norms, dtype=np.float32)
    norms_a /= np.linalg.norm(norms_a, axis=1, keepdims=True) + 1e-8

    # All rings have RADIAL_LIMB segments. Strip-triangulate the side wall.
    indices = []
    for r_i in range(n_rings_total - 1):
        for j in range(RADIAL_LIMB):
            a = r_i * RADIAL_LIMB + j
            b = r_i * RADIAL_LIMB + (j + 1) % RADIAL_LIMB
            c = (r_i + 1) * RADIAL_LIMB + j
            d = (r_i + 1) * RADIAL_LIMB + (j + 1) % RADIAL_LIMB
            indices.extend([a, c, b, b, c, d])

    bones_pair = np.stack(
        [np.asarray(bones_a_list, dtype=np.int32),
         np.asarray(bones_b_list, dtype=np.int32)], axis=1,
    )
    return SkinnedMesh(
        positions=verts_a,
        normals=norms_a,
        bones=bones_pair,
        weights=np.asarray(weights_list, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
    )


def _world_y(bones, idx):
    """Recover the world-space Y of a bone's head by walking the chain.
    The rest pose has no rotations, so each bone's world Y is the sum of
    its parent's world Y and the bone's local head Y.
    """
    if idx < 0:
        return 0.0
    name, parent, head, tip, r = bones[idx]
    return _world_y(bones, parent) + float(head[1])


def _world_y_tip(bones, idx):
    head_y = _world_y(bones, idx)
    return head_y + float(bones[idx][3][1])


def _world_pos(bones, idx):
    """Full 3D world-space head position. Same chain-walk as _world_y."""
    if idx < 0:
        return np.zeros(3, dtype=np.float32)
    name, parent, head, tip, r = bones[idx]
    return _world_pos(bones, parent) + np.asarray(head, dtype=np.float32)


def _merge_skinned(parts):
    """Concatenate a list of SkinnedMesh into one, offsetting indices.
    Mirror of garments._merge_meshes but local so body_loft can self-merge.
    """
    from . import primitives as prim
    chunks = []
    for m in parts:
        if m is None or m.indices.size == 0:
            continue
        chunks.append((m.positions, m.normals,
                       m.bones[:, 0], m.bones[:, 1],
                       m.weights, m.indices))
    if not chunks:
        return None
    v, n, ba, bb, w, idx = prim.merge(chunks)
    return SkinnedMesh(
        positions=v, normals=n,
        bones=np.stack([ba, bb], axis=1).astype(np.int32),
        weights=w, indices=idx,
    )


def build_pants_cuffs(bones, shape, inflate=0.020,
                      cuff_taper=1.10) -> "SkinnedMesh | None":
    """Cycle 54 deliverable: small flared cuff ring at the bottom of each
    shin so the pants hem reads as a jeans break / boot-cut taper.

    Two short frustums per character (one per shin), each merged into the
    pants mesh. Geometry: a single ring of ``RADIAL_PANTS`` segments at
    the ankle level with radius = shin_r * cuff_taper, plus a connector
    ring just above it at shin_r so the bell looks like a flared edge
    rather than a free-floating disc.
    All vertices weighted 100% to the corresponding shin bone, so the
    cuff follows the foot in the same way the existing shin capsule does.
    """
    parts = []
    cloth = max(inflate, 0.006)
    for side in ("L", "R"):
        shin_idx = _find(bones, f"shin_{side}")
        if shin_idx < 0:
            continue
        shin_r = float(bones[shin_idx][4])
        # Bone-local: shin's own frame has +Y along the bone (foot at +Y
        # since the capsule's _profiled_capsule wraps it that way). The
        # ankle is at y = bone_length; pull the cuff slightly above that
        # so it sits over the existing capsule end-cap.
        tip = np.asarray(bones[shin_idx][3], dtype=np.float32)
        bone_len = float(np.linalg.norm(tip))
        # Connector ring just above the natural ankle, cuff bell at the
        # ankle level itself. Cloth pad applied to both so the bell sits
        # outside the leg surface, not inside it.
        y_top = bone_len * 0.88
        y_bot = bone_len * 0.98
        rr_top = shin_r * 1.00 + cloth
        rr_bot = shin_r * cuff_taper + cloth

        verts, norms = [], []
        for ri, (y, rr) in enumerate([(y_top, rr_top), (y_bot, rr_bot)]):
            for j in range(RADIAL_PANTS):
                phi = 2.0 * math.pi * (j / RADIAL_PANTS)
                cp, sp = math.cos(phi), math.sin(phi)
                # Capsules in mesh.py are generated in bone-local frame
                # with the bone axis along +Y. Match that convention.
                verts.append((cp * rr, y, sp * rr))
                # Outward radial normal, with a small +Y tilt on the bell
                # to fake the rim highlight.
                tilt = 0.25 if ri == 1 else 0.0
                norms.append((cp, tilt, sp))

        # Two-ring strip, RADIAL_PANTS segments. Outward winding for a
        # tube whose first ring is the upper ring (smaller y here means
        # closer to the bone head). Capsules use bottom-up winding; the
        # bell ring is below (greater y) the connector, so use the same
        # bottom-up [a,c,b, b,c,d] pattern.
        indices = []
        for j in range(RADIAL_PANTS):
            a = j
            b = (j + 1) % RADIAL_PANTS
            c = RADIAL_PANTS + j
            d = RADIAL_PANTS + (j + 1) % RADIAL_PANTS
            indices.extend([a, c, b, b, c, d])

        n_v = len(verts)
        norms_a = np.asarray(norms, dtype=np.float32)
        norms_a /= np.linalg.norm(norms_a, axis=1, keepdims=True) + 1e-8
        cuff = SkinnedMesh(
            positions=np.asarray(verts, dtype=np.float32),
            normals=norms_a,
            bones=np.full((n_v, 2), shin_idx, dtype=np.int32),
            weights=np.tile(np.array([1.0, 0.0], dtype=np.float32), (n_v, 1)),
            indices=np.asarray(indices, dtype=np.uint32),
        )
        parts.append(cuff)

    if not parts:
        return None
    return _merge_skinned(parts)



def build_torso_loft(bones, shape) -> SkinnedMesh | None:
    """Continuous torso mesh. Returns None if any of pelvis/spine/chest
    is missing (caller falls back to whatever it had)."""
    pelvis_idx = _find(bones, "pelvis")
    spine_idx = _find(bones, "spine")
    chest_idx = _find(bones, "chest")
    if pelvis_idx < 0 or spine_idx < 0 or chest_idx < 0:
        return None

    p_head_y = _world_y(bones, pelvis_idx)
    p_tip_y = _world_y_tip(bones, pelvis_idx)
    s_head_y = _world_y(bones, spine_idx)
    s_tip_y = _world_y_tip(bones, spine_idx)
    c_head_y = _world_y(bones, chest_idx)
    c_tip_y = _world_y_tip(bones, chest_idx)

    pelvis_r = float(bones[pelvis_idx][4])
    spine_r = float(bones[spine_idx][4])
    chest_r = float(bones[chest_idx][4])

    # Shape knobs that drive cross-section.
    shoulder_w = float(getattr(shape, "shoulder_w", 1.0)) if shape else 1.0
    hip_w = float(getattr(shape, "hip_w", 1.0)) if shape else 1.0
    bust = float(getattr(shape, "bust", 0.0)) if shape else 0.0
    bust_proj = float(getattr(shape, "bust_proj", 0.6)) if shape else 0.6
    gender = getattr(shape, "gender", "neutral") if shape else "neutral"

    # Build a list of rings: each ring is (y_world, lateral_radius_x,
    # depth_radius_z, anchor_bone_idx, blend_to_idx, weight_to_anchor,
    # forward_offset). forward_offset is how far the ring center is
    # pushed in +Z (used for bust projection at the chest's front).
    rings = []

    # Pelvis: from y=p_head_y - pelvis_r (below the joint, rounded cap)
    # down to where the thighs branch off. Lateral radius scales with
    # hip_w. We extend the bottom into the thigh root region so the
    # loft + thigh capsules meet without a crotch gap.
    pelvis_bottom = p_head_y - pelvis_r * 1.10  # reach down past the
                                                 # thigh roots (thigh_head
                                                 # sits at p_head_y in the
                                                 # rest pose).
    pelvis_top = s_head_y
    for i in range(RINGS_PELVIS + 1):
        t = i / RINGS_PELVIS
        # Ease the bottom rounding so the lowest ring tapers but doesn't
        # collapse to zero — we want it to still be wide enough to meet
        # the thigh capsules laterally.
        cap_t = min(1.0, t * 2.5 + 0.20)
        cap_factor = min(1.0, math.sqrt(cap_t))
        y = pelvis_bottom + t * (pelvis_top - pelvis_bottom)
        rx = pelvis_r * hip_w * cap_factor
        rz = pelvis_r * ASPECT_Z * cap_factor
        rings.append((y, rx, rz, pelvis_idx, spine_idx, 1.0 - 0.4 * t, 0.0))

    # Spine: from spine_head_y to spine_tip_y. Lateral radius interpolates
    # from pelvis (full) to chest top (waist-pinched). The waist cinch is
    # already encoded in spine_r since `_SHAPE_RULES` set spine.r =
    # bulk*waist; we additionally squeeze X slightly more than Z so the
    # waist reads from the front silhouette.
    waist = float(getattr(shape, "waist", 1.0)) if shape else 1.0
    for i in range(1, RINGS_SPINE + 1):
        t = i / RINGS_SPINE
        y = s_head_y + t * (s_tip_y - s_head_y)
        # Lateral interpolates pelvis → spine waist → chest base.
        if t < 0.5:
            u = t / 0.5
            rx = pelvis_r * hip_w * (1 - u) + spine_r * u
        else:
            u = (t - 0.5) / 0.5
            rx = spine_r * (1 - u) + chest_r * waist * u
        rz = rx * ASPECT_Z
        # Smooth bone weight from pelvis (t=0) to chest (t=1) crossing
        # spine in the middle. We collapse the 3-bone influence into 2:
        # pelvis↔spine in the lower half, spine↔chest in the upper half.
        if t < 0.5:
            anchor, blend = spine_idx, pelvis_idx
            wa = 0.5 + t  # 0.5..1.0 → spine increasingly dominant
        else:
            anchor, blend = spine_idx, chest_idx
            wa = 1.5 - t  # 1.0..0.5 → spine receding
        rings.append((y, rx, rz, anchor, blend, wa, 0.0))

    # Chest: from chest_head_y to chest_tip_y. Top of the chest stretches
    # X by shoulder_w. Add a forward push on the front rings proportional
    # to bust * bust_proj — that bakes the bust ellipsoid into the loft.
    for i in range(1, RINGS_CHEST + 1):
        t = i / RINGS_CHEST
        y = c_head_y + t * (c_tip_y - c_head_y)
        rx = chest_r * (waist + (shoulder_w - waist) * t)
        rz = chest_r * ASPECT_Z * (waist + (1.0 - waist) * t)
        # Bust projection peaks around t = 0.55 (the bust apex), tapers
        # off at the top (collarbones) and bottom (rib cage). Only fires
        # for non-zero bust.
        proj = 0.0
        if bust > 0.01 and gender == "female":
            apex_t = 0.55
            sigma = 0.25
            proj_curve = math.exp(-((t - apex_t) ** 2) / (2 * sigma * sigma))
            proj = bust * bust_proj * chest_r * 0.45 * proj_curve
        # Bone weight: chest dominant; blend toward spine at the bottom.
        wa = 0.5 + 0.5 * t
        rings.append((y, rx, rz, chest_idx, spine_idx, wa, proj))

    # Top cap: shrink the last few rings toward the neck base so the
    # chest doesn't terminate as an open disc. The neck attaches via its
    # own capsule above; we close to a small radius to leave room for
    # the neck base.
    last_y = rings[-1][0]
    neck_radius = chest_r * 0.40
    for j in range(1, 4):
        cap_t = j / 4
        y = last_y + cap_t * chest_r * 0.45
        rx = (chest_r * shoulder_w) * (1 - cap_t) + neck_radius * cap_t
        rz = (chest_r * ASPECT_Z) * (1 - cap_t) + neck_radius * cap_t
        rings.append((y, rx, rz, chest_idx, spine_idx, 1.0, 0.0))

    # Triangulate the side wall.
    n_rings = len(rings)
    n_radial = RADIAL
    verts = np.zeros((n_rings * n_radial, 3), dtype=np.float32)
    norms = np.zeros((n_rings * n_radial, 3), dtype=np.float32)
    bones_a = np.zeros((n_rings * n_radial,), dtype=np.int32)
    bones_b = np.zeros((n_rings * n_radial,), dtype=np.int32)
    weights = np.zeros((n_rings * n_radial, 2), dtype=np.float32)

    # Per-bone world rest head Y values (the chain has zero rest rotations,
    # so the world Y of a bone is just the cumulative `head[1]` from root).
    head_y = {b: _world_y(bones, b) for b in (pelvis_idx, spine_idx, chest_idx)}

    for r_i, (y, rx, rz, anchor, blend, wa, proj) in enumerate(rings):
        # Convert a desired world-space vertex Y into the frame the LBS
        # shader expects: local_y = world_y - (wa*head_a_y + wb*head_b_y).
        # At rest the LBS blend then re-adds the weighted heads and we
        # recover world_y exactly.
        ref_y = wa * head_y[anchor] + (1.0 - wa) * head_y[blend]
        local_y = y - ref_y
        for j in range(n_radial):
            phi = 2.0 * math.pi * (j / n_radial)
            cp, sp = math.cos(phi), math.sin(phi)
            # Front of the body is +Z. Bust projection adds to Z when
            # the vertex is on the front hemisphere (sp > 0).
            front_push = max(0.0, sp) * proj
            x = cp * rx
            z = sp * rz + front_push
            idx = r_i * n_radial + j
            verts[idx] = (x, local_y, z)
            # Cross-section gradient normal, normalised. We build it
            # analytically from the ellipse: gradient of (x/rx)^2 +
            # (z/rz)^2 = 1 is proportional to (x/rx^2, 0, z/rz^2).
            nx = cp / max(rx, 1e-5)
            nz = sp / max(rz, 1e-5)
            n_v = np.array([nx, 0.0, nz], dtype=np.float32)
            n_v /= np.linalg.norm(n_v) + 1e-8
            norms[idx] = n_v
            bones_a[idx] = anchor
            bones_b[idx] = blend
            weights[idx] = (wa, 1.0 - wa)

    indices = []
    for r_i in range(n_rings - 1):
        for j in range(n_radial):
            a = r_i * n_radial + j
            b = r_i * n_radial + (j + 1) % n_radial
            c = (r_i + 1) * n_radial + j
            d = (r_i + 1) * n_radial + (j + 1) % n_radial
            indices.extend([a, c, b, b, c, d])

    # Bottom cap (a triangle fan back to a single point at the lowest
    # ring's center) — closes the mesh as a hemisphere. Express in the
    # pelvis bone's local frame.
    bottom_y_world = rings[0][0] - 0.001
    bottom_y_local = bottom_y_world - head_y[pelvis_idx]
    bottom_idx = verts.shape[0]
    verts = np.vstack([verts, [[0.0, bottom_y_local, 0.0]]])
    norms = np.vstack([norms, [[0.0, -1.0, 0.0]]])
    bones_a = np.append(bones_a, pelvis_idx)
    bones_b = np.append(bones_b, pelvis_idx)
    weights = np.vstack([weights, [[1.0, 0.0]]])
    for j in range(n_radial):
        a = j
        b = (j + 1) % n_radial
        # Wind facing down: bottom_idx, b, a (CCW from below).
        indices.extend([bottom_idx, b, a])

    # Top cap pinches to a center point so the loft is watertight.
    top_y_world = rings[-1][0] + 0.001
    top_y_local = top_y_world - head_y[chest_idx]
    top_idx = verts.shape[0]
    verts = np.vstack([verts, [[0.0, top_y_local, 0.0]]])
    norms = np.vstack([norms, [[0.0, 1.0, 0.0]]])
    bones_a = np.append(bones_a, chest_idx)
    bones_b = np.append(bones_b, chest_idx)
    weights = np.vstack([weights, [[1.0, 0.0]]])
    last_ring_start = (n_rings - 1) * n_radial
    for j in range(n_radial):
        a = last_ring_start + j
        b = last_ring_start + (j + 1) % n_radial
        indices.extend([top_idx, a, b])

    bones_pair = np.stack([bones_a, bones_b], axis=1).astype(np.int32)
    return SkinnedMesh(
        positions=verts.astype(np.float32),
        normals=norms.astype(np.float32),
        bones=bones_pair,
        weights=weights.astype(np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
    )
