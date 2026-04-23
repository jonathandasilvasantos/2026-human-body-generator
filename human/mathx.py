"""Minimal 4x4 matrix helpers. Row-major numpy arrays; we pass
``transpose=GL_TRUE`` to ``glUniformMatrix4fv`` or transpose manually before
upload."""

import math
import numpy as np


def identity():
    return np.eye(4, dtype=np.float32)


def translate(v):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = v
    return m


def rotate_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def align_y_to(direction):
    """3x3 rotation that maps +Y to ``direction`` (unit-normalized)."""
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    d = d / n
    c = float(np.dot(up, d))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    v = np.cross(up, d)
    s = np.linalg.norm(v)
    v = v / s
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + s * K + (1 - c) * (K @ K)).astype(np.float32)
