"""End-to-end test of human_create_from_arrays via Human.from_arrays.

Builds a tetrahedron from numpy arrays (no normals/uvs/skin), saves and
reloads it, and checks normals were synthesized to unit length.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from bindings import Human


def test_from_arrays_minimal_tetrahedron():
    pos = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    idx = np.array([
        0, 2, 1,
        0, 1, 3,
        0, 3, 2,
        1, 2, 3,
    ], dtype=np.uint32)

    h = Human.from_arrays(pos, idx)
    assert h.vertex_count == 4
    assert h.index_count == 12
    assert h.bone_count == 1

    v = h.vertices
    norms = v[:, 3:6]
    lens = np.linalg.norm(norms, axis=1)
    assert np.all(np.abs(lens - 1.0) < 1e-3)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tet.hmesh"
        h.save(p)
        h2 = Human.load(p)
    np.testing.assert_allclose(h.vertices, h2.vertices, atol=1e-5)
