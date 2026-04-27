"""Offscreen renderer.

Loads a Human, evaluates the C engine, draws to a moderngl framebuffer with
simple Lambert + headlight shading, returns the framebuffer as a PIL image.
A window-based viewer can be added later; this module is the headless core
the CLI and tests use.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np
from PIL import Image

VERT_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
out vec3 v_norm;
out vec2 v_uv;
void main() {
    v_norm = mat3(model) * in_norm;
    v_uv = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG_SHADER = """
#version 330
uniform vec3 light_dir;
uniform vec3 base_color;
in vec3 v_norm;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    float l = max(dot(n, normalize(light_dir)), 0.0);
    vec3 col = base_color * (0.25 + 0.75 * l);
    frag = vec4(col, 1.0);
}
"""


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def render(human, *, view: str = "front", size: tuple[int, int] = (512, 768)) -> Image.Image:
    """Render `human` into a PIL image. view ∈ {front, side, iso}."""
    human.evaluate()
    ctx = moderngl.create_standalone_context()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)

    verts = np.ascontiguousarray(human.vertices, dtype=np.float32)
    idx = np.ascontiguousarray(human.indices, dtype=np.uint32)

    vbo = ctx.buffer(verts.tobytes())
    ibo = ctx.buffer(idx.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")],
        index_buffer=ibo,
        index_element_size=4,
    )

    w, h = size
    color_tex = ctx.texture((w, h), 4)
    depth_rb = ctx.depth_renderbuffer((w, h))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)
    fbo.use()
    ctx.viewport = (0, 0, w, h)
    ctx.clear(0.07, 0.08, 0.10, 1.0)

    ymin = float(verts[:, 1].min())
    ymax = float(verts[:, 1].max())
    center = np.array([0.0, (ymin + ymax) * 0.5, 0.0], dtype=np.float32)
    span = max(ymax - ymin, 1e-3)
    dist = span * 2.0

    if view == "front":
        direction = np.array([0, 0, 1], dtype=np.float32)
    elif view == "side":
        direction = np.array([1, 0, 0], dtype=np.float32)
    elif view == "iso":
        direction = np.array([1, 0.4, 1], dtype=np.float32)
        direction /= np.linalg.norm(direction)
    else:
        raise ValueError(f"unknown view: {view}")
    eye = center + direction * dist

    up = np.array([0, 1, 0], dtype=np.float32)
    view_m = _look_at(eye, center, up)
    proj = _perspective(35.0, w / h, 0.05, 50.0)
    mvp = proj @ view_m
    model = np.eye(4, dtype=np.float32)

    prog["mvp"].write(mvp.T.astype(np.float32).tobytes())
    prog["model"].write(model.T.astype(np.float32).tobytes())
    prog["light_dir"].value = (0.4, 0.8, 0.6)
    prog["base_color"].value = (0.85, 0.78, 0.70)

    vao.render()

    raw = fbo.read(components=3, alignment=1)
    img = Image.frombytes("RGB", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM)

    fbo.release(); color_tex.release(); depth_rb.release()
    vao.release(); vbo.release(); ibo.release(); prog.release(); ctx.release()
    return img
