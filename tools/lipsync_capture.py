"""Capture a strip of face renders across a lip-sync track.

Produces a single PNG with the face at N evenly-spaced timestamps
through ``voice.wav`` (or any supplied WAV), each panel labelled with
its timestamp and dominant viseme. Useful for eyeballing mouth shape
diversity, jaw amplitude, and transitions without running the viewer.

Usage:
    env/bin/python -m tools.lipsync_capture \\
        --wav voice.wav --out screenshots/lipsync_v1.png --frames 12
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import replace

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from human import audio_lipsync, face_anim, mathx, renderer, skeleton
from human.capture import _make_context, _make_fbo
from human.character import Appearance, Character


def _face_view(width, height, target):
    proj = mathx.perspective(math.radians(26), width / max(height, 1), 0.05, 20.0)
    eye = np.array([target[0], target[1] + 0.02, target[2] + 0.55], dtype=np.float32)
    view = mathx.look_at(eye, np.asarray(target, dtype=np.float32), [0, 1, 0])
    return proj, view


def _head_world(character: Character):
    mats = character.bone_matrices()
    i = skeleton.bone_index(character.bones, "head")
    tip = np.asarray(character.bones[i][3], dtype=np.float32)
    p = mats[i] @ np.array([tip[0], tip[1], tip[2], 1.0], dtype=np.float32)
    return p[:3]


def _dominant_viseme(track: face_anim.VisemeTrack, t: float) -> str:
    if not track.segments:
        return "sil"
    return min(track.segments, key=lambda s: abs(s.time - t)).name


def render_strip(wav_path: str, out_path: str, frames: int = 12,
                 panel_w: int = 320, panel_h: int = 320, seed: int = 311):
    track = audio_lipsync.from_wav(wav_path)
    dur = track.duration
    # Sample from the first non-silent region to the last, so strips
    # aren't dominated by leading/trailing silence.
    times = np.linspace(0.0, dur, frames + 2)[1:-1]

    win = _make_context(panel_w, panel_h)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glClearColor(0.09, 0.10, 0.13, 1.0)
    skin_prog = renderer.SkinProgram()
    fbo_ms, fbo_res = _make_fbo(panel_w, panel_h)

    random.seed(seed)
    shape = skeleton.random_shape(gender="female")
    app = Appearance(gender="female", expression="neutral",
                     hair_style="medium", top_style="tshirt",
                     top_color=(0.3, 0.35, 0.48), bottom_style="pants",
                     bottom_color=(0.12, 0.13, 0.16))
    character = Character(shape=shape, appearance=app)
    character.build_gpu()

    panels = []
    for t in times:
        w = track.sample(float(t))
        w = {k: v for k, v in w.items() if v > 0.01}
        character.appearance = replace(character.appearance, blendshapes=w)
        character.build_gpu()

        head = _head_world(character)
        target = (float(head[0]), float(head[1]) - 0.05, float(head[2]))
        proj, view = _face_view(panel_w, panel_h, target)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
        glViewport(0, 0, panel_w, panel_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(skin_prog.prog)
        glUniformMatrix4fv(skin_prog.u_proj, 1, GL_TRUE, proj)
        glUniformMatrix4fv(skin_prog.u_view, 1, GL_TRUE, view)
        renderer.upload_bones(skin_prog.u_bones, character.bone_matrices())
        glUniform1f(skin_prog.u_seed, float(character.appearance.seed))
        glUniform1i(skin_prog.u_light_style, 0)
        glUniform1i(skin_prog.u_print_style, 0)
        glUniform1f(skin_prog.u_print_strength, 0.0)
        glUniform1i(skin_prog.u_stamp_style, 0)
        glUniform1f(skin_prog.u_stamp_strength, 0.0)
        for d in character.drawables:
            glUniform3f(skin_prog.u_color, *d.color)
            glUniform1i(skin_prog.u_mode, d.mode)
            glUniform1i(skin_prog.u_material, int(getattr(d, "material", 0)))
            glUniform1f(skin_prog.u_bend_inflate,
                        0.025 if d.mode == 1 else 0.0)
            d.mesh_gpu.draw()
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
        glBlitFramebuffer(0, 0, panel_w, panel_h, 0, 0, panel_w, panel_h,
                          GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
        pixels = glReadPixels(0, 0, panel_w, panel_h, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(panel_h, panel_w, 4)
        img = np.flipud(img).copy()
        label = f"t={t:.2f}s  {_dominant_viseme(track.viseme_track, float(t))}"
        _overlay_label(img, label)
        panels.append(img)
        print(f"  {label}  weights={ {k: round(v,2) for k,v in w.items()} }")

    character.delete()
    glfw.terminate()

    cols = min(len(panels), 6)
    rows = (len(panels) + cols - 1) // cols
    strip = np.zeros((rows * panel_h, cols * panel_w, 4), dtype=np.uint8)
    strip[..., 3] = 255
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        strip[r * panel_h:(r + 1) * panel_h,
              c * panel_w:(c + 1) * panel_w] = p
    Image.fromarray(strip, "RGBA").save(out_path)
    print(f"wrote {out_path}  ({rows}x{cols} panels)")


def _overlay_label(img: np.ndarray, text: str):
    """Ugly-but-readable 1-char-per-5px label in the top-left corner."""
    # Use PIL for text so we don't need a GL text overlay path.
    pil = Image.fromarray(img, "RGBA")
    from PIL import ImageDraw
    d = ImageDraw.Draw(pil)
    d.rectangle([(0, 0), (len(text) * 8 + 8, 20)], fill=(0, 0, 0, 180))
    d.text((4, 3), text, fill=(255, 255, 255, 255))
    img[:] = np.array(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="voice.wav")
    ap.add_argument("--out", default="screenshots/lipsync_strip.png")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--panel", type=int, default=320)
    ap.add_argument("--seed", type=int, default=311)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render_strip(args.wav, args.out, frames=args.frames,
                 panel_w=args.panel, panel_h=args.panel, seed=args.seed)


if __name__ == "__main__":
    main()
