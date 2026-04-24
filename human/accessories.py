"""Procedural hair, eyebrows and facial hair.

Every accessory is a separate :class:`mesh.SkinnedMesh` rigidly skinned to
the ``head`` bone, so it follows head rotation during the walk cycle. They
are rendered with ``u_mode = 2`` (hair shader).

Fitted to the anthropometric landmarks defined in :mod:`mesh`
(``H_CHIN`` / ``H_MOUTH`` / ``H_NOSE_BASE`` / ``H_EYE`` / ``H_BROW`` /
``H_HAIRLINE`` / ``H_TOP``).
"""

import math
import numpy as np

from . import mathx
from . import primitives as prim
from .mesh import (
    SkinnedMesh, _empty_mesh, _head_info,
    H_BROW, H_CHIN, H_EYE, H_HAIRLINE, H_MOUTH, H_NOSE_BASE, H_TOP,
)


def _finish(chunks, R):
    """Apply the head's rest rotation ``R`` and merge chunks into a
    SkinnedMesh."""
    rotated = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(rotated)
    if idx.size == 0:
        return _empty_mesh()
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


def _hairline_cards(head_idx, parent, length, head_w, head_d, style):
    """Small hair-card clusters around the frontal hairline and temples.

    The cap/drape meshes provide volume; these cards make the scalp boundary
    read from the front, approximating dense strand clumps without individual
    fibers.
    """
    if style == "buzz":
        y = length * (H_HAIRLINE - 0.010)
        z = head_d * 0.88
        return [prim.flat_patch(
            (0.0, y, z),
            (head_w * 0.68, length * 0.020),
            head_idx, parent, normal=(0, 0.10, 1),
            subdiv=(10, 2), thickness=0.002,
        )]

    if style == "short":
        fringe_y = length * (H_HAIRLINE - 0.055)
        temple_y = length * (H_HAIRLINE - 0.090)
        return [
            prim.flat_patch(
                (0.0, fringe_y, head_d * 0.92),
                (head_w * 0.46, length * 0.055),
                head_idx, parent, normal=(0, 0.18, 1),
                subdiv=(8, 3), thickness=0.003,
            ),
            prim.flat_patch(
                (+head_w * 0.58, temple_y, head_d * 0.72),
                (head_w * 0.18, length * 0.075),
                head_idx, parent, normal=(0.35, 0.08, 1),
                subdiv=(4, 4), thickness=0.003,
            ),
            prim.flat_patch(
                (-head_w * 0.58, temple_y, head_d * 0.72),
                (head_w * 0.18, length * 0.075),
                head_idx, parent, normal=(-0.35, 0.08, 1),
                subdiv=(4, 4), thickness=0.003,
            ),
        ]

    if style in ("medium", "long"):
        fringe_y = length * (H_HAIRLINE - 0.035)
        temple_y = length * (H_EYE + 0.040)
        temple_h = length * (0.13 if style == "medium" else 0.20)
        return [
            prim.flat_patch(
                (0.0, fringe_y, head_d * 0.92),
                (head_w * 0.42, length * 0.030),
                head_idx, parent, normal=(0, 0.15, 1),
                subdiv=(9, 3), thickness=0.003,
            ),
            prim.flat_patch(
                (+head_w * 0.68, temple_y, head_d * 0.58),
                (head_w * 0.16, temple_h),
                head_idx, parent, normal=(0.45, -0.05, 1),
                subdiv=(4, 6), thickness=0.003,
            ),
            prim.flat_patch(
                (-head_w * 0.68, temple_y, head_d * 0.58),
                (head_w * 0.16, temple_h),
                head_idx, parent, normal=(-0.45, -0.05, 1),
                subdiv=(4, 6), thickness=0.003,
            ),
        ]

    return []


# --- hair --------------------------------------------------------------------

def build_hair(bones, style: str) -> SkinnedMesh:
    """Shell over the scalp. Sizes use head length, not bone radius, so
    hair fits regardless of the old ``radius`` scale."""
    if style == "bald":
        return _empty_mesh()
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    # Scalp ellipsoid sized just outside the skull, hugging the crown.
    # Skull center y was (H_TOP + H_CHIN)/2 * length; we match that.
    scalp_cy = length * (H_TOP + H_CHIN) * 0.5 + length * 0.02
    scalp_rx = head_w * 1.05
    scalp_ry = length * (H_TOP - H_CHIN) * 0.5 * 1.02
    scalp_rz = head_d * 1.05

    if style == "buzz":
        # Very short fuzz: keep only the upper dome, ending above the brow.
        y_cut = (length * H_BROW - scalp_cy) / scalp_ry  # map brow y to ellipsoid param
        chunks = [prim.hemisphere_cap(
            (0.0, scalp_cy, 0.0),
            (scalp_rx, scalp_ry, scalp_rz),
            head_idx, parent,
            rings=6, radial=22,
            y_cutoff=max(-0.2, y_cut),
        )]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "short":
        # Cut just below brow for a short layered style with a visible hairline.
        y_cut = (length * (H_BROW - 0.02) - scalp_cy) / scalp_ry
        chunks = [prim.hemisphere_cap(
            (0.0, scalp_cy, -length * 0.01),
            (scalp_rx * 1.03, scalp_ry * 1.04, scalp_rz * 1.04),
            head_idx, parent,
            rings=8, radial=24,
            y_cutoff=max(-0.5, y_cut),
        )]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "medium":
        # Reaches past the ears down toward the jawline.
        y_cut = (length * (H_MOUTH + 0.04) - scalp_cy) / scalp_ry
        chunks = [prim.hemisphere_cap(
            (0.0, scalp_cy, -length * 0.01),
            (scalp_rx * 1.06, scalp_ry * 1.08, scalp_rz * 1.08),
            head_idx, parent,
            rings=10, radial=26,
            y_cutoff=max(-1.3, y_cut),
        )]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "long":
        # Scalp cap + drape flowing down past the shoulders. The drape's
        # own ellipsoid is positioned BELOW the skull so it reads as hair
        # falling naturally, not as a pointy cone standing on top.
        y_cut_top = (length * (H_MOUTH - 0.02) - scalp_cy) / scalp_ry
        drape_cy = length * 0.05
        drape_ry = length * 0.55  # not so tall: its top should sit below the crown
        chunks = [
            prim.hemisphere_cap(
                (0.0, scalp_cy, -length * 0.01),
                (scalp_rx * 1.08, scalp_ry * 1.10, scalp_rz * 1.10),
                head_idx, parent,
                rings=10, radial=28, y_cutoff=max(-1.6, y_cut_top),
            ),
            # Lower drape behind and around the head.
            prim.hemisphere_cap(
                (0.0, drape_cy, -length * 0.04),
                (scalp_rx * 1.12, drape_ry, scalp_rz * 1.14),
                head_idx, parent,
                rings=10, radial=28, y_cutoff=-0.6,
            ),
        ]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    else:
        return _empty_mesh()

    return _finish(chunks, R)


# --- eyebrows ----------------------------------------------------------------

def build_eyebrows(bones) -> SkinnedMesh:
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    # Eyebrows are composed from three small patches per side: an inner
    # head (closer to the nose, slightly lower), a middle body (peak of
    # the arch), and an outer tail (fades out toward the temple, slightly
    # higher). This replaces the single horizontal bar and prevents the
    # fixed "angry" look by arching upward, not inward.
    sep_inner = length * 0.085
    sep_mid   = length * 0.118
    sep_outer = length * 0.148
    y_inner = length * (H_BROW - 0.020)
    y_mid   = length * (H_BROW + 0.005)
    y_outer = length * (H_BROW - 0.002)
    z = head_d * 0.86
    hy = length * 0.013
    inner_size  = (length * 0.024, hy)
    mid_size    = (length * 0.032, hy * 1.05)
    outer_size  = (length * 0.026, hy * 0.85)

    chunks = []
    for side in (+1.0, -1.0):
        chunks += [
            prim.flat_patch((side * sep_inner, y_inner, z), inner_size,
                            head_idx, parent,
                            normal=(side * 0.10, -0.12, 1.0),
                            subdiv=(3, 2), thickness=0.0035),
            prim.flat_patch((side * sep_mid, y_mid, z), mid_size,
                            head_idx, parent,
                            normal=(side * 0.05, 0.05, 1.0),
                            subdiv=(4, 2), thickness=0.0035),
            prim.flat_patch((side * sep_outer, y_outer, z), outer_size,
                            head_idx, parent,
                            normal=(side * 0.25, -0.02, 1.0),
                            subdiv=(3, 2), thickness=0.003),
        ]
    return _finish(chunks, R)


# --- facial hair -------------------------------------------------------------

def build_facial_hair(bones, style: str) -> SkinnedMesh:
    """Beard / mustache / stubble as a set of flat patches distributed over
    the lower face. No hemisphere hacks -- each patch is explicitly placed."""
    if style == "none":
        return _empty_mesh()
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    chunks = []

    # Mustache — a single thin strip just above the upper lip.
    if style in ("mustache", "full"):
        z = head_d * 0.90
        y = length * (H_MOUTH + 0.025)
        chunks.append(prim.flat_patch(
            (0.0, y, z),
            (length * 0.075, length * 0.010),
            head_idx, parent, normal=(0, 0, 1),
            subdiv=(7, 2), thickness=0.002,
        ))

    # Soul patch / goatee — small patch just below the lower lip.
    if style in ("goatee", "full"):
        chunks.append(prim.flat_patch(
            (0.0, length * (H_CHIN + 0.08), head_d * 0.80),
            (length * 0.040, length * 0.035),
            head_idx, parent, normal=(0, 0, 1),
            subdiv=(4, 3), thickness=0.002,
        ))

    # Full beard / stubble — subtle patches on the jawline and chin.
    # Small, thin, positioned well below the mouth so they read as
    # shadow/stubble rather than a second pair of lips.
    if style in ("stubble", "full"):
        jaw_y  = length * (H_CHIN + 0.10)     # between chin and mouth
        chin_y = length * (H_CHIN + 0.03)     # right at the chin line
        chunks += [
            prim.flat_patch(
                (+head_w * 0.55, jaw_y, head_d * 0.62),
                (length * 0.035, length * 0.040),
                head_idx, parent, normal=(0.3, -0.2, 0.95),
                subdiv=(3, 3), thickness=0.0015,
            ),
            prim.flat_patch(
                (-head_w * 0.55, jaw_y, head_d * 0.62),
                (length * 0.035, length * 0.040),
                head_idx, parent, normal=(-0.3, -0.2, 0.95),
                subdiv=(3, 3), thickness=0.0015,
            ),
        ]
        chunks.append(prim.flat_patch(
            (0.0, chin_y, head_d * 0.70),
            (length * 0.055, length * 0.025),
            head_idx, parent, normal=(0, -0.3, 0.95),
            subdiv=(5, 2), thickness=0.0015,
        ))

    if not chunks:
        return _empty_mesh()
    return _finish(chunks, R)
