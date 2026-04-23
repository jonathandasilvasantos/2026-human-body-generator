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


HAIR_STYLES = ("bald", "short", "medium", "long", "buzz")
FACIAL_HAIR_STYLES = ("none", "stubble", "mustache", "goatee", "full")


def _finish(chunks, R):
    """Apply the head's rest rotation ``R`` and merge chunks into a
    SkinnedMesh."""
    rotated = [(v @ R.T, n @ R.T, ba, bb, w, idx) for (v, n, ba, bb, w, idx) in chunks]
    v, n, ba, bb, w, idx = prim.merge(rotated)
    if idx.size == 0:
        return _empty_mesh()
    return SkinnedMesh(v, n, np.stack([ba, bb], axis=1).astype(np.int32), w, idx)


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
    elif style == "long":
        # Scalp cap + drape down to the shoulder area (still skinned to head).
        y_cut_top = (length * (H_MOUTH - 0.02) - scalp_cy) / scalp_ry
        chunks = [
            prim.hemisphere_cap(
                (0.0, scalp_cy, -length * 0.01),
                (scalp_rx * 1.08, scalp_ry * 1.10, scalp_rz * 1.10),
                head_idx, parent,
                rings=10, radial=28, y_cutoff=max(-1.6, y_cut_top),
            ),
            # trailing drape behind the head
            prim.hemisphere_cap(
                (0.0, length * 0.05, -length * 0.06),
                (scalp_rx * 1.10, length * 1.1, scalp_rz * 1.15),
                head_idx, parent,
                rings=10, radial=28, y_cutoff=-0.3,
            ),
        ]
    else:
        return _empty_mesh()

    return _finish(chunks, R)


# --- eyebrows ----------------------------------------------------------------

def build_eyebrows(bones) -> SkinnedMesh:
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    # Eyebrows sit on the brow ridge, just above the eye line. A bit thicker
    # than a strict anatomical brow so they read at typical viewing distance.
    sep = length * 0.115
    y   = length * (H_BROW - 0.01)
    z   = head_d * 0.86
    hx, hy = length * 0.075, length * 0.018

    chunks = [
        prim.flat_patch((+sep, y, z), (hx, hy), head_idx, parent,
                        normal=(0, 0, 1), subdiv=(6, 2), thickness=0.004),
        prim.flat_patch((-sep, y, z), (hx, hy), head_idx, parent,
                        normal=(0, 0, 1), subdiv=(6, 2), thickness=0.004),
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

    # Mustache — two small patches just above the upper lip.
    if style in ("mustache", "full"):
        z = head_d * 0.90
        y = length * (H_MOUTH + 0.03)
        half_sep = length * 0.028
        mw, mh = length * 0.055, length * 0.012
        chunks += [
            prim.flat_patch((+half_sep, y, z), (mw, mh), head_idx, parent,
                            normal=(0, 0, 1), subdiv=(6, 2), thickness=0.004),
            prim.flat_patch((-half_sep, y, z), (mw, mh), head_idx, parent,
                            normal=(0, 0, 1), subdiv=(6, 2), thickness=0.004),
        ]

    # Soul patch / goatee — patch on and below the chin.
    if style in ("goatee", "full"):
        chunks.append(prim.flat_patch(
            (0.0, length * (H_CHIN + 0.06), head_d * 0.80),
            (length * 0.045, length * 0.055),
            head_idx, parent, normal=(0, 0, 1),
            subdiv=(6, 4), thickness=0.004,
        ))

    # Full beard / stubble — jawline + under-chin. Kept low on the face
    # (between H_CHIN and H_MOUTH) so the patches don't read as random dots
    # on the cheeks when viewed from a distance.
    if style in ("stubble", "full"):
        jaw_y_lo = length * (H_CHIN + 0.05)    # near the chin line
        jaw_y_hi = length * (H_CHIN + 0.15)    # just under the mouth corners
        # Side jaw strips, smaller and rotated to follow the jaw front-back.
        chunks += [
            prim.flat_patch(
                (+head_w * 0.70, jaw_y_hi, head_d * 0.45),
                (length * 0.045, length * 0.06),
                head_idx, parent, normal=(0.4, 0.0, 0.9),
                subdiv=(4, 3), thickness=0.003,
            ),
            prim.flat_patch(
                (-head_w * 0.70, jaw_y_hi, head_d * 0.45),
                (length * 0.045, length * 0.06),
                head_idx, parent, normal=(-0.4, 0.0, 0.9),
                subdiv=(4, 3), thickness=0.003,
            ),
        ]
        # Front chin/under-jaw patch.
        chunks.append(prim.flat_patch(
            (0.0, jaw_y_lo, head_d * 0.78),
            (length * 0.09, length * 0.045),
            head_idx, parent, normal=(0, -0.15, 0.98),
            subdiv=(6, 3), thickness=0.003,
        ))

    if not chunks:
        return _empty_mesh()
    return _finish(chunks, R)
