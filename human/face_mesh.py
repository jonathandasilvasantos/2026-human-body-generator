"""Dense face mesh — phase F1 of the face-mesh migration.

Replaces `_head_compound`'s primitive ellipsoid stack (skull shell +
face plate + mandible + nose ellipsoids + lip rings) with a single
denser shell that has facial landmarks **carved into the surface**
via per-vertex displacement. The eye stack, eyebrows, lips ring and
ears stay as separate add-ons (driven by ARKit-52 channels and the
existing ports).

Topology: a UV-style head ellipsoid with N_RINGS × N_RADIAL vertices,
identical winding to `_skull_shell` so the same triangulation works.
What's different is the per-vertex displacement: vertices in named
"regions" (brow, eye_orbit, nose, cheek, lip, chin) get pushed
in/out by analytical functions of `Shape`.

Region detection is geometric, computed at build time from each
vertex's (t, phi) coordinates where t in [-1,1] (chin to crown) and
phi in [0, 2pi] (back -> right -> front -> left -> back).

References:
- FLAME parametric head model — https://flame.is.tue.mpg.de/
- Farkas anthropometry landmark fractions
- Documented in notes/face_mesh_migration_plan.md
"""

from __future__ import annotations

import math

import numpy as np

from . import mathx


# Density. The old `_skull_shell` ran at 28 × 32. We bump to 56 × 56
# so the carved landmarks read at face close-up framing without
# pixelated edges. ~3,200 verts is well under our shader budget.
N_RINGS = 56
N_RADIAL = 56


# Landmark fractions along the head bone length (matches the values
# in `human/mesh.py` so head bones / eyes / lips / hair line up).
H_CHIN      = 0.06
H_MOUTH     = 0.31
H_NOSE_BASE = 0.43
H_EYE       = 0.55
H_BROW      = 0.63
H_HAIRLINE  = 0.77
H_TOP       = 1.00


def _gauss(x, mu, sigma):
    """Unit-peak Gaussian, used to weight region-specific displacements."""
    s = max(sigma, 1e-4)
    return math.exp(-((x - mu) ** 2) / (2.0 * s * s))


def _front_factor(phi):
    """1 at the front pole (phi = pi/2), 0 at the back pole (phi = 3pi/2),
    smooth cosine fall-off."""
    return max(0.0, math.sin(phi))


def _lateral_factor(phi):
    """1 at +X (phi=0), 1 at -X (phi=pi), 0 at front and back poles.
    Used to weight cheekbone bands."""
    return abs(math.cos(phi))


def _front_weight(phi, sharpness=2.0):
    s = math.sin(phi)
    if s <= 0.0:
        return 0.0
    return s ** sharpness


def _shape_get(shape, attr, default=1.0):
    if shape is None:
        return default
    return float(getattr(shape, attr, default))


def build_face_mesh(head_idx, parent_idx, tip, gender, shape=None):
    """Build the dense face mesh in head-bone-local coordinates.

    Returns the same 6-tuple shape `_skull_shell` does so it slots
    straight into the head-compound chunk merge in `human/mesh.py`.
    """
    length = float(np.linalg.norm(tip))
    if length < 1e-6:
        return None

    # Aspect roughly follows real adult head proportions
    # height : width : depth ≈ 1.0 : 0.65 : 0.90.
    w_factor = 0.335 * (1.05 if gender == "male" else 0.99)
    d_factor = 0.385

    # Shape knobs that drive per-region morphs.
    eye_sep_k       = _shape_get(shape, "eye_sep", 1.0)
    # Cycle 57: reduce eye scale globally so the head reads less toy-like
    # at body-framing distance. Real adult eyes occupy a much smaller
    # fraction of the skull than the previous default produced.
    eye_size_k      = _shape_get(shape, "eye_size", 1.0) * 0.85
    eye_depth_k     = _shape_get(shape, "eye_depth", 1.0)
    nose_width_k    = _shape_get(shape, "nose_width", 1.0)
    nose_proj_k     = _shape_get(shape, "nose_proj", 1.0)
    nose_bridge_k   = _shape_get(shape, "nose_bridge", 1.0)
    cheekbone_k     = _shape_get(shape, "cheekbone", 1.0)
    jaw_width_k     = _shape_get(shape, "jaw_width", 1.0)
    chin_proj_k     = _shape_get(shape, "chin_proj", 1.0)
    brow_prom_k     = _shape_get(shape, "brow_prominence", 1.0)
    lip_full_k      = _shape_get(shape, "lip_fullness", 1.0)
    asymmetry_k     = _shape_get(shape, "face_asymmetry", 0.0)

    # Centre of the skull (head-local coordinates) and half-height.
    skull_cy = length * (H_TOP + H_CHIN) * 0.5
    skull_half = length * (H_TOP - H_CHIN) * 0.5

    head_w = length * w_factor
    head_d = length * d_factor

    # Pre-compute the skull's oval profile at each ring (matches the
    # existing `_skull_shell` profile shape so the head silhouette
    # doesn't change between phases).
    profiles = []
    for i in range(N_RINGS + 1):
        theta = math.pi * (i / N_RINGS)
        ct = math.cos(theta)
        # `t` = -1 chin, +1 crown; `frac` = 0..1 fraction up the head.
        t = ct
        frac = (t + 1.0) * 0.5
        # X (lateral) profile — narrower at chin and crown, full at
        # mid-cheek band.
        if t < 0:
            rx = head_w * max(0.55 + 0.45 * (1 + t) * 1.5, 0.55)
        else:
            rx = head_w * (1.0 - 0.20 * t * t)
        # Z (depth) profile — front (cheek/jaw) is shallower than back
        # (occiput).
        if t < 0:
            rz_f = head_d * max(0.92 + 0.10 * (1 + t), 0.78)
        else:
            rz_f = head_d * (0.96 - 0.18 * t * t)
        if t < 0:
            rz_b = head_d * max(1.06 + 0.30 * (1 + t), 0.80)
        else:
            rz_b = head_d * (1.06 + 0.07 * t - 0.14 * t * t)
        # Z centre offset shifts the skull back at the upper half so
        # the occiput protrudes; pulls forward at the chin so the jaw
        # line reads.
        if t < -0.5:
            cz = head_d * (-0.04)
        elif t < 0.0:
            cz = head_d * (-0.04 + 0.10 * (1 + t * 2))
        else:
            cz = head_d * (0.06 - 0.10 * t)
        profiles.append((rx, rz_f, rz_b, cz, frac))

    # Generate vertices, then apply landmark morphs in-place.
    verts = np.zeros((N_RINGS + 1, N_RADIAL, 3), dtype=np.float32)
    for i in range(N_RINGS + 1):
        rx, rz_f, rz_b, cz, frac = profiles[i]
        theta = math.pi * (i / N_RINGS)
        st = math.sin(theta)
        ct = math.cos(theta)
        t = ct
        py_base = skull_cy + t * skull_half
        for j in range(N_RADIAL):
            phi = 2.0 * math.pi * (j / N_RADIAL)
            sp, cp = math.sin(phi), math.cos(phi)
            # Smooth blend between front/back radii at the +/-X poles
            # to avoid a visible seam.
            blend = 0.5 * (1.0 + sp)
            blend_s = blend * blend * (3.0 - 2.0 * blend)
            rz = rz_b + (rz_f - rz_b) * blend_s
            px = st * cp * rx
            py = py_base
            pz = st * sp * rz + cz

            # ---- region morphs (front-hemisphere only) ----
            if sp > 0.0:
                fw = _front_weight(phi, 2.0)
                fy = py / max(length, 1e-6)  # head-local Y as 0..1 fraction

                # Eye orbit recession: push back (in -Z) inside two
                # bilateral pockets centred at H_EYE and at +/- eye_sep_k * 0.18 * length.
                orbit_y = length * H_EYE
                orbit_x = head_w * 0.40 * eye_sep_k
                # Distance from each orbit centre in head-local space.
                dx_L = px - (-orbit_x)
                dx_R = px - (+orbit_x)
                dy = py - orbit_y
                # Convert to a roughly isotropic gauss in (x,y) at the orbit scale.
                rx_orbit = head_w * 0.13 * eye_size_k
                ry_orbit = length * 0.06 * eye_size_k
                gL = math.exp(-((dx_L / rx_orbit) ** 2 + (dy / ry_orbit) ** 2))
                gR = math.exp(-((dx_R / rx_orbit) ** 2 + (dy / ry_orbit) ** 2))
                orbit_g = max(gL, gR)
                # Subtle orbit depression — visible as a soft recession
                # around the eye sphere without hiding it. Reduced from
                # 0.06 → 0.035 in cycle 18 so the far eye doesn't get
                # buried in the recess at 3/4 view angles.
                pz -= head_d * 0.035 * eye_depth_k * orbit_g * fw

                # Nose ridge protrusion: a vertical bridge along the
                # central column from H_NOSE_BASE up to about H_EYE,
                # plus a forward-pushed tip at H_NOSE_BASE itself.
                bridge_g = (
                    _gauss(px, 0.0, head_w * 0.06 * nose_width_k)
                    * _gauss(fy, (H_NOSE_BASE + H_EYE) / 2,
                             (H_EYE - H_NOSE_BASE) * 0.55)
                )
                pz += head_d * 0.10 * nose_proj_k * bridge_g * fw

                # Nose tip (just below H_NOSE_BASE, central, highest projection).
                # Reduced from 0.18 -> 0.10 of head_d so the carved tip
                # cohabits with the existing nose-tip primitive without
                # producing a double-bump silhouette in 3/4 view.
                tip_g = (
                    _gauss(px, 0.0, head_w * 0.05 * nose_width_k)
                    * _gauss(fy, H_NOSE_BASE - 0.01,
                             (H_NOSE_BASE - H_MOUTH) * 0.55)
                )
                pz += head_d * 0.10 * nose_proj_k * tip_g * fw

                # Bridge height — push the highest dorsum up and forward
                # for tall noses (controlled by nose_bridge).
                if nose_bridge_k > 1.0:
                    high_bridge_g = (
                        _gauss(px, 0.0, head_w * 0.05)
                        * _gauss(fy, H_EYE - 0.02, 0.04)
                    )
                    pz += head_d * 0.06 * (nose_bridge_k - 1.0) * high_bridge_g * fw

                # Mouth pucker: lips ring puffs slightly forward at H_MOUTH.
                lip_g = (
                    _gauss(px, 0.0, head_w * 0.18)
                    * _gauss(fy, H_MOUTH, 0.04)
                )
                pz += head_d * 0.05 * lip_full_k * lip_g * fw

                # Cheekbone band: lateral X stretch around H_EYE - 0.05
                # at +/- 0.45 * head_w.
                cheek_g = (
                    _gauss(abs(px) - head_w * 0.40, 0.0, head_w * 0.12)
                    * _gauss(fy, H_EYE - 0.07, 0.05)
                )
                lateral_sign = 1.0 if px >= 0 else -1.0
                px += head_w * 0.06 * (cheekbone_k - 1.0 + 0.05) * cheek_g * lateral_sign

                # Chin projection: forward push on the central lower face
                # below H_MOUTH down to H_CHIN.
                chin_g = (
                    _gauss(px, 0.0, head_w * 0.18)
                    * _gauss(fy, (H_CHIN + H_MOUTH) / 2,
                             (H_MOUTH - H_CHIN) * 0.55)
                )
                pz += head_d * 0.06 * chin_proj_k * chin_g * fw

                # Jaw width: lateral scale of the jaw-line band.
                jaw_g = _gauss(fy, (H_CHIN + H_MOUTH) / 2,
                               (H_MOUTH - H_CHIN) * 0.55)
                px *= 1.0 + 0.10 * (jaw_width_k - 1.0) * jaw_g

                # Brow prominence: forward push on the brow strip.
                brow_g = (
                    _gauss(px, 0.0, head_w * 0.36)
                    * _gauss(fy, H_BROW - 0.03,
                             (H_HAIRLINE - H_BROW) * 0.55)
                )
                pz += head_d * 0.04 * (brow_prom_k - 1.0 + 0.5) * brow_g * fw

                # Asymmetry: peaks at the cheekbone band; tiny radial bias.
                if asymmetry_k != 0.0:
                    band = max(0.0, 1.0 - abs(fy - (H_EYE - 0.05)) * 6.0)
                    px *= 1.0 + asymmetry_k * 0.5 * band * (1.0 if px >= 0 else -1.0)

            verts[i, j] = (px, py, pz)

    # Compute per-vertex normals from the triangulated mesh. Normalise
    # in NumPy for speed; the input is small enough that this is a
    # one-time cost per regeneration.
    flat = verts.reshape(-1, 3)
    norms = np.zeros_like(flat, dtype=np.float32)
    indices = []
    for i in range(N_RINGS):
        for j in range(N_RADIAL):
            a = i * N_RADIAL + j
            b = i * N_RADIAL + (j + 1) % N_RADIAL
            c = (i + 1) * N_RADIAL + j
            d = (i + 1) * N_RADIAL + (j + 1) % N_RADIAL
            indices.extend([a, c, b, b, c, d])

    indices_np = np.asarray(indices, dtype=np.uint32)
    # Accumulate face normals onto the three vertices of each tri.
    tri_a = indices_np[0::3]
    tri_b = indices_np[1::3]
    tri_c = indices_np[2::3]
    edge1 = flat[tri_b] - flat[tri_a]
    edge2 = flat[tri_c] - flat[tri_a]
    fn = np.cross(edge1, edge2)
    fn_norm = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    np.add.at(norms, tri_a, fn_norm)
    np.add.at(norms, tri_b, fn_norm)
    np.add.at(norms, tri_c, fn_norm)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8

    # Laplacian smoothing pass over the regular ring × radial grid.
    # The carved morphs create normal discontinuities at the edges of
    # the eye-orbit / nose / cheekbone Gaussians; without smoothing
    # those show up as dark patches under directional lighting.
    # Average each vertex's normal with its 4 grid neighbours
    # (i±1 ring, j±1 segment) for two passes.
    norms_grid = norms.reshape(N_RINGS + 1, N_RADIAL, 3).copy()
    for _ in range(3):
        smoothed = norms_grid.copy()
        # Neighbour rings (clamped at the poles since rings 0 and N
        # are the cap rings — wrapping there would reach across the
        # head). Use simple replication at the boundaries.
        up = np.roll(norms_grid, -1, axis=0)
        up[-1] = norms_grid[-1]
        down = np.roll(norms_grid, 1, axis=0)
        down[0] = norms_grid[0]
        # Neighbour segments wrap around (cylindrical).
        left = np.roll(norms_grid, -1, axis=1)
        right = np.roll(norms_grid, 1, axis=1)
        smoothed = (norms_grid * 2.0 + up + down + left + right) / 6.0
        norms_grid = smoothed
    norms = norms_grid.reshape(-1, 3)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8

    n_verts = (N_RINGS + 1) * N_RADIAL
    bones_a = np.full((n_verts,), head_idx, dtype=np.int32)
    bones_b = np.full((n_verts,), parent_idx if parent_idx >= 0 else head_idx,
                      dtype=np.int32)
    weights = np.zeros((n_verts, 2), dtype=np.float32)
    weights[:, 0] = 1.0

    return (
        flat.astype(np.float32),
        norms.astype(np.float32),
        bones_a,
        bones_b,
        weights,
        indices_np,
    )
