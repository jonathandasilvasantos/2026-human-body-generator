"""End-to-end test of the Cython binding + C engine via the proto archetype."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from bindings import Human


def test_proto_loads_with_expected_topology():
    h = Human.proto()
    assert h.vertex_count > 0
    assert h.index_count % 3 == 0
    assert h.bone_count == 3
    names = [p[0] for p in h.params]
    assert names == ["height", "weight", "head_size"]


def test_height_param_grows_y_extent():
    h = Human.proto()
    v = h.vertices
    y0 = float(v[:, 1].max())
    h.set_param("height", 1.4)
    h.evaluate()
    y1 = float(v[:, 1].max())
    assert y1 > y0 + 0.05


def test_weight_morph_grows_radius():
    h = Human.proto()
    v = h.vertices
    r0 = float(np.linalg.norm(v[:, [0, 2]], axis=1).max())
    h.set_param("weight", 1.0)
    h.evaluate()
    r1 = float(np.linalg.norm(v[:, [0, 2]], axis=1).max())
    assert r1 > r0 + 0.02


def test_param_clamps_to_range():
    h = Human.proto()
    h.set_param("weight", 999.0)
    assert h.get_param("weight") <= 1.5 + 1e-6
    h.set_param("weight", -10.0)
    assert h.get_param("weight") >= 0.0 - 1e-6


def test_save_load_roundtrip():
    a = Human.proto()
    a.set_param("weight", 0.7)
    a.set_param("height", 1.2)
    a.evaluate()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.hmesh"
        a.save(p)
        b = Human.load(p)
    assert a.vertex_count == b.vertex_count
    assert a.bone_count == b.bone_count
    assert math.isclose(a.get_param("weight"), b.get_param("weight"), abs_tol=1e-6)
    np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-5)


def test_indices_in_range():
    h = Human.proto()
    idx = h.indices
    assert idx.max() < h.vertex_count


def test_normals_unit_length_after_evaluate():
    h = Human.proto()
    h.set_param("weight", 0.8)
    h.evaluate()
    n = h.vertices[:, 3:6]
    lens = np.linalg.norm(n, axis=1)
    assert np.all(np.abs(lens - 1.0) < 1e-3)
