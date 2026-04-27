"""Interactive viewer.

Opens a glfw window, renders a Human via the C engine + Cython binding,
provides mouse orbit and keyboard parameter controls. Re-evaluates the C
engine and re-uploads the vertex buffer when a parameter changes.

Controls
--------
  mouse drag (left)   orbit camera around figure
  mouse drag (right)  pan
  scroll              zoom
  TAB                 cycle to next parameter
  shift+TAB           cycle to previous parameter
  + / -               increase / decrease selected parameter
  R                   reset selected parameter to default
  W                   toggle wireframe
  ESC / Q             quit
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image

from app.shaders import FRAG, VERT
from bindings import Human


# ---------- camera ----------

class Orbit:
    def __init__(self, target: np.ndarray, distance: float):
        self.target = target.astype(np.float32)
        self.distance = float(distance)
        self.yaw = 0.0     # radians around Y
        self.pitch = 0.1   # radians from XZ plane
        self.fov = 35.0

    def eye(self) -> np.ndarray:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        d = self.distance
        return self.target + np.array([d*cp*sy, d*sp, d*cp*cy], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        f = self.target - eye; f /= np.linalg.norm(f)
        s = np.cross(f, np.array([0, 1, 0], dtype=np.float32)); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye); m[1, 3] = -np.dot(u, eye); m[2, 3] = np.dot(f, eye)
        return m

    def proj_matrix(self, aspect: float) -> np.ndarray:
        n, fr = 0.05, 100.0
        f = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect; m[1, 1] = f
        m[2, 2] = (fr + n) / (n - fr); m[2, 3] = (2 * fr * n) / (n - fr)
        m[3, 2] = -1.0
        return m


# ---------- viewer ----------

class Viewer:
    def __init__(self, human: Human, *, width: int, height: int, title: str, headless: bool):
        self.h = human
        self.width = width
        self.height = height
        self.headless = headless

        self.params = list(self.h.params)  # snapshot of metadata
        self.selected = 0 if self.params else -1
        self.wireframe = False

        self._mouse_left = False
        self._mouse_right = False
        self._last_mouse = (0.0, 0.0)
        self._dirty = True

        verts = np.ascontiguousarray(self.h.vertices, dtype=np.float32)
        self._vc = verts.shape[0]

        # camera fit
        ymin = float(verts[:, 1].min())
        ymax = float(verts[:, 1].max())
        center = np.array([
            float(verts[:, 0].mean()),
            (ymin + ymax) * 0.5,
            float(verts[:, 2].mean()),
        ], dtype=np.float32)
        span = max(ymax - ymin, 1e-3)
        self.cam = Orbit(center, span * 1.8)
        self._span = span

        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.win = glfw.create_window(width, height, title, None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("glfw window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # On Retina displays the framebuffer is larger than the window.
        self.fb_w, self.fb_h = glfw.get_framebuffer_size(self.win)
        self.ctx.viewport = (0, 0, self.fb_w, self.fb_h)
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        self.vbo = self.ctx.buffer(verts.tobytes())
        idx = np.ascontiguousarray(self.h.indices, dtype=np.uint32)
        self.ibo = self.ctx.buffer(idx.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")],
            index_buffer=self.ibo, index_element_size=4,
        )

        glfw.set_cursor_pos_callback(self.win, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.win, self._on_mouse_button)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_framebuffer_size_callback(self.win, self._on_resize)

        self._print_status()

    # ---------- input ----------

    def _on_mouse_button(self, win, btn, action, mods):
        pressed = action == glfw.PRESS
        if btn == glfw.MOUSE_BUTTON_LEFT:  self._mouse_left = pressed
        if btn == glfw.MOUSE_BUTTON_RIGHT: self._mouse_right = pressed
        self._last_mouse = glfw.get_cursor_pos(win)

    def _on_mouse_move(self, win, x, y):
        lx, ly = self._last_mouse
        dx, dy = x - lx, y - ly
        self._last_mouse = (x, y)
        if self._mouse_left:
            self.cam.yaw   -= dx * 0.005
            self.cam.pitch += dy * 0.005
            self.cam.pitch = max(-1.4, min(1.4, self.cam.pitch))
            self._dirty = True
        elif self._mouse_right:
            scale = self.cam.distance * 0.0015
            self.cam.target[0] -= dx * scale
            self.cam.target[1] += dy * scale
            self._dirty = True

    def _on_scroll(self, win, xoff, yoff):
        self.cam.distance *= (0.92 if yoff > 0 else 1.08)
        self.cam.distance = max(self._span * 0.4, min(self._span * 12.0, self.cam.distance))
        self._dirty = True

    def _on_resize(self, win, w, h):
        # Framebuffer-size callback receives pixel size already (Retina-aware).
        self.fb_w, self.fb_h = max(w, 1), max(h, 1)
        self.ctx.viewport = (0, 0, self.fb_w, self.fb_h)
        self._dirty = True

    def _on_key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_TAB and self.params:
            step = -1 if (mods & glfw.MOD_SHIFT) else 1
            self.selected = (self.selected + step) % len(self.params)
            self._print_status()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) and self.selected >= 0:
            self._adjust(+1)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT) and self.selected >= 0:
            self._adjust(-1)
        elif key == glfw.KEY_R and self.selected >= 0:
            name, _kind, _lo, _hi, df, _v = self.params[self.selected]
            self.h.set_param(name, df)
            self._reeval()
        elif key == glfw.KEY_W:
            self.wireframe = not self.wireframe
            self._dirty = True

    def _adjust(self, sign: int) -> None:
        name, _kind, lo, hi, _df, _v = self.params[self.selected]
        cur = self.h.get_param(name)
        step = (hi - lo) * 0.05
        new = max(lo, min(hi, cur + sign * step))
        self.h.set_param(name, new)
        self._reeval()

    # ---------- rendering ----------

    def _reeval(self) -> None:
        self.h.evaluate()
        verts = np.ascontiguousarray(self.h.vertices, dtype=np.float32)
        self.vbo.write(verts.tobytes())
        self._dirty = True
        self._print_status()

    def _print_status(self) -> None:
        if self.selected < 0:
            print("[no parameters]", flush=True)
            return
        name, kind, lo, hi, df, _v = self.params[self.selected]
        cur = self.h.get_param(name)
        print(f"[{self.selected+1}/{len(self.params)}] {name} ({kind})  "
              f"{cur:.3f}  range [{lo:.3f}, {hi:.3f}]  default {df:.3f}", flush=True)

    def _draw(self) -> None:
        self.ctx.clear(0.07, 0.08, 0.10, 1.0)
        if self.wireframe:
            self.ctx.wireframe = True
        view = self.cam.view_matrix()
        proj = self.cam.proj_matrix(self.fb_w / self.fb_h)
        mvp = proj @ view
        self.prog["mvp"].write(mvp.T.astype(np.float32).tobytes())
        self.prog["model"].write(np.eye(4, dtype=np.float32).T.tobytes())
        self.prog["light_dir"].value = (0.4, 0.8, 0.6)
        self.prog["base_color"].value = (0.85, 0.78, 0.70)
        self.prog["ambient_color"].value = (0.04, 0.04, 0.05)
        self.vao.render()
        if self.wireframe:
            self.ctx.wireframe = False

    def capture(self) -> Image.Image:
        self._draw()
        raw = self.ctx.screen.read(components=3, alignment=1)
        return Image.frombytes("RGB", (self.fb_w, self.fb_h), raw).transpose(Image.FLIP_TOP_BOTTOM)

    def run(self) -> None:
        while not glfw.window_should_close(self.win):
            self._draw()
            glfw.swap_buffers(self.win)
            glfw.poll_events()
        self._cleanup()

    def self_test(self, out: Path) -> None:
        # pump events to settle the framebuffer, then capture
        for _ in range(2):
            glfw.poll_events()
        img = self.capture()
        img.save(out)
        print(f"wrote {out}", flush=True)
        self._cleanup()

    def _cleanup(self) -> None:
        self.vao.release(); self.vbo.release(); self.ibo.release(); self.prog.release()
        self.ctx.release()
        glfw.destroy_window(self.win)
        glfw.terminate()


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="path to .hmesh, or 'proto' for the synthetic body")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--self-test", action="store_true",
                    help="open hidden window, render one frame, save to --out, exit")
    ap.add_argument("--out", type=Path, default=Path("/tmp/viewer_selftest.png"))
    ap.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args(argv)

    h = Human.proto() if args.input == "proto" else Human.load(args.input)
    for kv in args.param:
        k, v = kv.split("=", 1)
        h.set_param(k.strip(), float(v))
    if args.param:
        h.evaluate()

    title = f"human v2 — {Path(args.input).name if args.input != 'proto' else 'proto'}"
    v = Viewer(h, width=args.width, height=args.height, title=title, headless=args.self_test)
    if args.self_test:
        v.self_test(args.out)
    else:
        v.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
