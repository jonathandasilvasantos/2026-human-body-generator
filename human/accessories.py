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
    # Prior cycles used front-facing hair cards for the hairline. On close
    # face captures those read as dark stickers on the forehead/cheeks. The
    # scalp cap and rear drape now carry the silhouette, keeping the face
    # unobstructed and avoiding glued-on hair artifacts.
    return []


def _shape_scalp_cap(chunk, scalp_cy, scalp_rx, scalp_ry, scalp_rz, length,
                     head_w, head_d, style):
    """Displace scalp-cap vertices to add sideburns, temple taper and a
    slight forward fringe.

    The raw hemisphere_cap gives a clean geometric dome which reads as
    plastic. Dropping the hairline at the temples (sideburns) and pushing
    slightly forward over the forehead produces a more anatomical boundary
    without individual strand geometry.
    """
    v, n, ba, bb, w, idx = chunk
    v = v.copy()
    y_min = float(v[:, 1].min())
    y_max = float(v[:, 1].max())
    y_span = max(y_max - y_min, 1e-6)
    for k in range(v.shape[0]):
        x = v[k, 0]
        y = v[k, 1]
        z = v[k, 2]
        dx = (x - 0.0) / max(scalp_rx, 1e-6)
        dz = (z - 0.0) / max(scalp_rz, 1e-6)
        r2 = dx * dx + dz * dz
        if r2 < 1e-6:
            continue
        edge = max(0.0, min(1.0, r2))
        # front-hemisphere (+z): small forward/down fringe push
        front = max(0.0, dz)
        # side factor peaks at |dx| large, |dz| small (temples/ears)
        side = max(0.0, abs(dx) - 0.25) * (1.0 - 0.8 * max(0.0, abs(dz)))
        # Rim weight: 1 at the bottom of the cap, 0 at the crown.
        rim = 1.0 - (y - y_min) / y_span
        rim = min(1.0, max(0.0, rim))
        rim = rim * rim  # bias toward the rim itself
        # Sideburn drop at temples.
        sb_gain = {"buzz": 0.018, "short": 0.028, "medium": 0.032,
                   "long": 0.036}.get(style, 0.0)
        v[k, 1] -= sb_gain * length * side * rim * edge
        # Slight forward lobe over the forehead — brings hair forward so the
        # front edge sits a bit inside the hairline rather than being a
        # perfect arc.
        fringe = {"buzz": 0.004, "short": 0.010, "medium": 0.014,
                  "long": 0.012}.get(style, 0.0)
        v[k, 2] += fringe * length * front * rim
        # Tuck the bottom rim inward toward the skull so the cap's lower
        # edge merges with the scalp instead of reading as a floating shell
        # with a dark interior ridge. Strongest at the rim, zero at the
        # crown.
        # Shrink the rim slightly so the cap's lower edge meets the skull
        # flush instead of hovering a few mm above it (which leaves a dark
        # ring of interior when viewed head-on). The cap starts 2-5% wider
        # than the skull, so a matching shrink lands it on the surface.
        tuck = {"buzz": 0.030, "short": 0.030, "medium": 0.040,
                "long": 0.045}.get(style, 0.0)
        if rim > 0.0:
            v[k, 0] -= tuck * x * rim
            v[k, 2] -= tuck * z * rim
    return (v, n, ba, bb, w, idx)


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
    scalp_cy = length * (H_TOP + H_CHIN) * 0.5 + length * 0.012
    scalp_rx = head_w * 1.025
    scalp_ry = length * (H_TOP - H_CHIN) * 0.5 * 1.005
    scalp_rz = head_d * 1.02

    if style == "buzz":
        # Very short fuzz: keep only the upper dome, ending above the brow.
        y_cut = (length * (H_HAIRLINE - 0.01) - scalp_cy) / scalp_ry
        cap = prim.hemisphere_cap(
            (0.0, scalp_cy, 0.0),
            (scalp_rx, scalp_ry, scalp_rz),
            head_idx, parent,
            rings=8, radial=28,
            y_cutoff=max(-0.2, y_cut),
        )
        chunks = [_shape_scalp_cap(cap, scalp_cy, scalp_rx, scalp_ry, scalp_rz,
                                   length, head_w, head_d, "buzz")]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "short":
        # Cut near the anatomical hairline, not down over the eyebrows.
        y_cut = (length * (H_HAIRLINE - 0.04) - scalp_cy) / scalp_ry
        rx, ry, rz = scalp_rx * 1.02, scalp_ry * 1.02, scalp_rz * 1.02
        cap = prim.hemisphere_cap(
            (0.0, scalp_cy, -length * 0.01),
            (rx, ry, rz),
            head_idx, parent,
            rings=10, radial=30,
            y_cutoff=max(-0.5, y_cut),
        )
        chunks = [_shape_scalp_cap(cap, scalp_cy, rx, ry, rz,
                                   length, head_w, head_d, "short")]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "medium":
        # More volume than short hair, but the cap still stops above the
        # orbits so it doesn't cover the face.
        y_cut = (length * (H_HAIRLINE - 0.08) - scalp_cy) / scalp_ry
        rx, ry, rz = scalp_rx * 1.04, scalp_ry * 1.05, scalp_rz * 1.05
        cap = prim.hemisphere_cap(
            (0.0, scalp_cy, -length * 0.01),
            (rx, ry, rz),
            head_idx, parent,
            rings=12, radial=32,
            y_cutoff=max(-1.3, y_cut),
        )
        chunks = [_shape_scalp_cap(cap, scalp_cy, rx, ry, rz,
                                   length, head_w, head_d, "medium")]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    elif style == "long":
        # Scalp cap + drape flowing down past the shoulders. The drape's
        # own ellipsoid is positioned BELOW the skull so it reads as hair
        # falling naturally, not as a pointy cone standing on top.
        y_cut_top = (length * (H_HAIRLINE - 0.04) - scalp_cy) / scalp_ry
        drape_cy = length * 0.10
        drape_ry = length * 0.55  # not so tall: its top should sit below the crown
        rx_top, ry_top, rz_top = (scalp_rx * 1.05, scalp_ry * 1.07,
                                  scalp_rz * 1.07)
        cap_top = prim.hemisphere_cap(
            (0.0, scalp_cy, -length * 0.01),
            (rx_top, ry_top, rz_top),
            head_idx, parent,
            rings=12, radial=32, y_cutoff=max(-1.6, y_cut_top),
        )
        chunks = [
            _shape_scalp_cap(cap_top, scalp_cy, rx_top, ry_top, rz_top,
                             length, head_w, head_d, "long"),
            # Lower drape behind and around the head.
            prim.hemisphere_cap(
                (0.0, drape_cy, -head_d * 1.55),
                (scalp_rx * 1.08, drape_ry, scalp_rz * 0.72),
                head_idx, parent,
                rings=10, radial=28, y_cutoff=-0.6,
            ),
        ]
        chunks.extend(_hairline_cards(head_idx, parent, length, head_w, head_d, style))
    else:
        return _empty_mesh()

    return _finish(chunks, R)


# --- eyebrows ----------------------------------------------------------------

def build_eyebrows(bones, weights=None) -> SkinnedMesh:
    """Three-patch eyebrow per side, animated by ARKit brow channels.

    Driven channels:
      - ``browInnerUp``         -> both inner heads lift (and pinch
        slightly toward midline -- the surprise/sad inner-brow raise).
      - ``browOuterUp{L,R}``    -> per-side outer tail lifts (the
        outer-brow flash for surprise / curiosity).
      - ``browDown{L,R}``       -> whole brow drops + inner pinch
        toward midline (the AU4 frown).
    """
    from . import face_anim as _fa
    info = _head_info(bones)
    if info is None:
        return _empty_mesh()
    head_idx, parent, R, length, _r, head_w, head_d = info

    sep_inner = length * 0.074
    sep_mid   = length * 0.102
    sep_outer = length * 0.128
    y_inner_b = length * (H_BROW - 0.006)
    y_mid_b   = length * (H_BROW + 0.012)
    y_outer_b = length * (H_BROW + 0.006)
    z = head_d * 0.78
    hy = length * 0.0048
    inner_size  = (length * 0.012, hy)
    mid_size    = (length * 0.017, hy * 1.05)
    outer_size  = (length * 0.014, hy * 0.90)

    inner_up   = _fa.w(weights, "browInnerUp")
    muscles = _fa.muscle_activations(weights)
    LIFT       = length * 0.024     # full-weight vertical lift
    PINCH      = length * 0.010     # full-weight inner-X pull toward midline
    DROP       = length * 0.020     # full-weight brow-down drop

    chunks = []
    for side, suffix in ((+1.0, "Left"), (-1.0, "Right")):
        outer_up = _fa.w(weights, f"browOuterUp{suffix}")
        down     = _fa.w(weights, f"browDown{suffix}")
        corr     = muscles[f"corrugator_{suffix.lower()}"]
        front    = muscles[f"frontalis_{suffix.lower()}"]

        # Inner head: lifts with browInnerUp, drops with browDown,
        # and (for browDown) is pulled medially producing the AU4 pinch.
        inner_dy = LIFT * inner_up - DROP * down - length * 0.004 * corr
        inner_dx = -side * PINCH * (down + 0.30 * inner_up + 0.35 * corr)
        # Mid body: averaged effect.
        mid_dy = 0.5 * (LIFT * inner_up + LIFT * outer_up) - DROP * down
        mid_dx = -side * PINCH * (0.45 * down + 0.20 * corr)
        # Outer tail: only outerUp + brow-down.
        outer_dy = LIFT * outer_up + length * 0.004 * front - 0.6 * DROP * down

        chunks += [
            prim.flat_patch((side * sep_inner + inner_dx, y_inner_b + inner_dy, z),
                            inner_size, head_idx, parent,
                            normal=(side * 0.10, -0.12, 1.0),
                            subdiv=(3, 2), thickness=0.0035),
            prim.flat_patch((side * sep_mid + mid_dx, y_mid_b + mid_dy, z),
                            mid_size, head_idx, parent,
                            normal=(side * 0.05, 0.05, 1.0),
                            subdiv=(4, 2), thickness=0.0035),
            prim.flat_patch((side * sep_outer, y_outer_b + outer_dy, z),
                            outer_size, head_idx, parent,
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
