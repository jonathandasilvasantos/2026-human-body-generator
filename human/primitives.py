"""Low-level mesh primitives used by body, head and accessories.

Each primitive emits position/normal arrays plus skinning metadata in a form
directly compatible with :class:`mesh.SkinnedMesh`. Everything is rigidly
skinned (weight 1) to a single bone by default, which is correct for small
head details (eyes, nose, hair, beard) that should track the head bone.
"""

import math
import numpy as np


def ellipsoid(
    center,
    radii,
    bone_index,
    parent_index=-1,
    weight_self=1.0,
    rings=14,
    radial=22,
):
    """Unit-ellipsoid triangulated as latitude/longitude grid."""
    cx, cy, cz = center
    rx, ry, rz = radii
    v, n, ba, bb, w = [], [], [], [], []

    for i in range(rings + 1):
        theta = math.pi * (i / rings)  # 0..pi (south -> north pole)
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(radial):
            phi = 2.0 * math.pi * (j / radial)
            sp, cp = math.sin(phi), math.cos(phi)
            # direction on unit sphere
            dx, dy, dz = st * cp, ct, st * sp
            p = (cx + dx * rx, cy + dy * ry, cz + dz * rz)
            # gradient-based normal for an ellipsoid
            nx, ny, nz = dx / rx, dy / ry, dz / rz
            inv = 1.0 / (math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
            v.append(p)
            n.append((nx * inv, ny * inv, nz * inv))
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

    return (
        np.asarray(v, np.float32),
        np.asarray(n, np.float32),
        np.asarray(ba, np.int32),
        np.asarray(bb, np.int32),
        np.asarray(w, np.float32),
        np.asarray(idx, np.uint32),
    )


def hemisphere_cap(
    center,
    radii,
    bone_index,
    parent_index=-1,
    rings=10,
    radial=22,
    y_cutoff=0.0,
    weight_self=1.0,
):
    """Upper hemisphere of an ellipsoid, cut at ``y_cutoff`` (in units of ry).

    Used for scalp / hair cap. ``y_cutoff`` selects how much of the ellipsoid
    is kept: 0 = exact upper half; -0.5 = keep a lot extending below the
    equator (gives a 'hair falling down' silhouette).
    """
    cx, cy, cz = center
    rx, ry, rz = radii
    v, n, ba, bb, w = [], [], [], [], []

    for i in range(rings + 1):
        # theta: 0 = north pole (top), increases toward equator; extend beyond
        # pi/2 to go below the equator for long hair.
        t = i / rings
        theta = math.acos(max(-1.0, min(1.0, 1.0 - t * (1.0 - y_cutoff))))
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(radial):
            phi = 2.0 * math.pi * (j / radial)
            sp, cp = math.sin(phi), math.cos(phi)
            dx, dy, dz = st * cp, ct, st * sp
            p = (cx + dx * rx, cy + dy * ry, cz + dz * rz)
            nx, ny, nz = dx / rx, dy / ry, dz / rz
            inv = 1.0 / (math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
            v.append(p)
            n.append((nx * inv, ny * inv, nz * inv))
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

    return (
        np.asarray(v, np.float32),
        np.asarray(n, np.float32),
        np.asarray(ba, np.int32),
        np.asarray(bb, np.int32),
        np.asarray(w, np.float32),
        np.asarray(idx, np.uint32),
    )


def flat_patch(
    center,
    size,
    bone_index,
    parent_index=-1,
    normal=(0, 0, 1),
    subdiv=(4, 2),
    thickness=0.003,
    weight_self=1.0,
):
    """Thin oriented box (flattened ellipsoid) used for eyebrows / beard /
    mustache patches. ``size`` is (half_x, half_y) extent along the patch's
    local axes before orienting to ``normal``."""
    # Build an oriented frame with +Z = normal.
    n_ = np.asarray(normal, dtype=np.float32)
    n_ = n_ / (np.linalg.norm(n_) + 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, n_)
    if np.linalg.norm(right) < 1e-4:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= np.linalg.norm(right)
    up = np.cross(n_, right)

    nx_, ny_ = subdiv
    v, n, ba, bb, w = [], [], [], [], []
    c = np.asarray(center, dtype=np.float32)
    hx, hy = size
    # emit two parallel grids (front + back of the thin slab) to give it body
    for side in (+1.0, -1.0):
        base = c + n_ * (side * thickness)
        for i in range(ny_ + 1):
            for j in range(nx_ + 1):
                u = -1.0 + 2.0 * j / nx_
                v_ = -1.0 + 2.0 * i / ny_
                # clip slightly to an oval rather than a rectangle
                if u * u + v_ * v_ > 1.2:
                    # keep inside bounds; squish toward centre
                    k = math.sqrt(1.2 / (u * u + v_ * v_))
                    u *= k; v_ *= k
                p = base + right * (u * hx) + up * (v_ * hy)
                v.append(tuple(p))
                n.append(tuple(n_ * side))
                ba.append(bone_index)
                bb.append(parent_index if parent_index >= 0 else bone_index)
                w.append((weight_self, 1.0 - weight_self))

    idx = []
    stride = nx_ + 1
    layer = (nx_ + 1) * (ny_ + 1)
    for side, base_off in ((0, 0), (1, layer)):
        for i in range(ny_):
            for j in range(nx_):
                a = base_off + i * stride + j
                b = base_off + i * stride + j + 1
                c_ = base_off + (i + 1) * stride + j
                d = base_off + (i + 1) * stride + j + 1
                if side == 0:
                    idx.extend([a, c_, b, b, c_, d])
                else:
                    idx.extend([a, b, c_, b, d, c_])

    return (
        np.asarray(v, np.float32),
        np.asarray(n, np.float32),
        np.asarray(ba, np.int32),
        np.asarray(bb, np.int32),
        np.asarray(w, np.float32),
        np.asarray(idx, np.uint32),
    )


def cone_shell(
    top_center,
    top_radius,
    height,
    bottom_radius,
    bone_index,
    parent_index=-1,
    rings=8,
    radial=26,
    weight_self=1.0,
):
    """Axis-aligned frustum / cone of revolution along -Y from ``top_center``.

    Used for skirts and the lower half of dresses. Rigidly skinned to one
    bone (typically the pelvis) so legs can move inside without pulling the
    garment -- the way a real skirt hangs.
    """
    cx, cy, cz = top_center
    v, n, ba, bb, w = [], [], [], [], []

    for i in range(rings + 1):
        t = i / rings
        y = cy - t * height
        rr = top_radius + t * (bottom_radius - top_radius)
        for j in range(radial):
            phi = 2.0 * math.pi * (j / radial)
            sp, cp = math.sin(phi), math.cos(phi)
            v.append((cx + rr * cp, y, cz + rr * sp))
            # normal points outward, tilted slightly downward if expanding
            slope = (bottom_radius - top_radius) / max(height, 1e-4)
            nx = cp
            ny = slope
            nz = sp
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            n.append((nx * inv, ny * inv, nz * inv))
            ba.append(bone_index)
            bb.append(parent_index if parent_index >= 0 else bone_index)
            w.append((weight_self, 1.0 - weight_self))

    idx = []
    # Rings go top (i=0) -> bottom (i=rings) in -Y. With CCW-front GL, the
    # outward-facing winding for that direction is (a, b, c) / (b, d, c) --
    # i.e. the OPPOSITE order from the hemisphere / capsule helpers which
    # build rings bottom-up.
    for i in range(rings):
        for j in range(radial):
            a = i * radial + j
            b = i * radial + (j + 1) % radial
            c = (i + 1) * radial + j
            d = (i + 1) * radial + (j + 1) % radial
            idx.extend([a, b, c, b, d, c])

    return (
        np.asarray(v, np.float32),
        np.asarray(n, np.float32),
        np.asarray(ba, np.int32),
        np.asarray(bb, np.int32),
        np.asarray(w, np.float32),
        np.asarray(idx, np.uint32),
    )


def merge(chunks):
    """Concatenate a list of ``(v,n,ba,bb,w,idx)`` tuples, fixing up indices."""
    if not chunks:
        z_f = np.zeros((0, 3), np.float32)
        z_i2 = np.zeros((0, 2), np.int32)
        z_w = np.zeros((0, 2), np.float32)
        z_i = np.zeros((0,), np.uint32)
        return z_f, z_f, z_i2[:, 0], z_i2[:, 0], z_w, z_i
    V, N, BA, BB, W, IDX = [], [], [], [], [], []
    off = 0
    for v, n, ba, bb, w, idx in chunks:
        V.append(v); N.append(n); BA.append(ba); BB.append(bb); W.append(w)
        IDX.append(idx + off)
        off += v.shape[0]
    return (
        np.concatenate(V, axis=0),
        np.concatenate(N, axis=0),
        np.concatenate(BA, axis=0),
        np.concatenate(BB, axis=0),
        np.concatenate(W, axis=0),
        np.concatenate(IDX, axis=0),
    )
