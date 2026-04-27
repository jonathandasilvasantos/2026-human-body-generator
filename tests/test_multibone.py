"""Multi-bone Human via from_arrays — exercises the path the rigged bake uses."""
from __future__ import annotations

import numpy as np

from bindings import Human


def test_two_bone_chain_along_y():
    # Two segments stacked in Y, each segment is a quad split into 2 tris.
    # Segment 0: y in [0, 1], bone 0
    # Segment 1: y in [1, 2], bone 1
    pos = np.array([
        [-0.5, 0, 0], [0.5, 0, 0], [-0.5, 1, 0], [0.5, 1, 0],   # segment 0
        [-0.5, 1, 0], [0.5, 1, 0], [-0.5, 2, 0], [0.5, 2, 0],   # segment 1
    ], dtype=np.float32)
    idx = np.array([
        0, 1, 2,  1, 3, 2,
        4, 5, 6,  5, 7, 6,
    ], dtype=np.uint32)
    bone_ids = np.zeros((8, 4), dtype=np.uint8)
    bone_ids[4:, 0] = 1
    bone_weights = np.zeros((8, 4), dtype=np.float32)
    bone_weights[:, 0] = 1.0

    parents = np.array([-1, 0], dtype=np.int32)
    # bone 0 at origin (identity), bone 1 at (0, 1, 0) in parent space
    bind = np.tile(np.eye(4, dtype=np.float32).flatten(order="F"), (2, 1))
    bind[1, 12] = 0.0  # column-major: translation lives at indices 12,13,14
    bind[1, 13] = 1.0
    bind[1, 14] = 0.0

    h = Human.from_arrays(
        pos, idx,
        bone_ids=bone_ids, bone_weights=bone_weights,
        bone_parents=parents, bind_locals=bind,
    )
    assert h.bone_count == 2
    v = h.vertices
    # In rest pose, output should match input positions.
    np.testing.assert_allclose(v[:, :3], pos, atol=1e-5)
