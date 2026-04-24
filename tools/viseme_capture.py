"""Capture renders of spoken visemes for mouth/teeth/lip animation review.

Produces a horizontal strip per character showing the same face holding
each viseme in turn. Complementary to ``face_anim_capture.py`` (which
focuses on emotional expression presets).

Viseme set is the Preston Blair / Disney 12-viseme reduction (Blair 1946,
Disney's animation reference) which most commercial lip-sync stacks
(Annosoft, Rhubarb, Oculus Lipsync, JALI) collapse onto. Each viseme is
mapped to ARKit-52 weights here rather than in ``face_anim`` to keep the
viseme table a single source of truth for this review workflow; if it
proves useful it is promoted into ``face_anim.VISEMES`` in a later cycle.

References for the weight choices:
  - Ezzat, Geiger & Poggio 2002, "Trainable Videorealistic Speech Animation"
  - Edwards, Landreth, Fiume & Singh 2016, "JALI: Jaw and Lip Integration"
  - Cohen & Massaro 1993, "Modeling coarticulation in synthetic visual speech"

Usage:
    env/bin/python -m tools.viseme_capture \
        --out screenshots/viseme_baseline --tag baseline
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from human import face_anim, mathx, renderer, skeleton
from human.capture import _make_context, _make_fbo
from human.character import Appearance, Character


# Preston-Blair 12-viseme set -> ARKit-52. Values tuned for the existing
# parametric mouth so the target reads plainly in a 512px thumbnail.
VISEMES: List[Tuple[str, Dict[str, float]]] = [
    ("sil", {}),                                             # rest / silence
    ("AI",  {"jawOpen": 0.55, "mouthShrugLower": 0.10,
             "mouthLowerDownLeft": 0.25, "mouthLowerDownRight": 0.25}),
    ("E",   {"jawOpen": 0.25,
             "mouthSmileLeft": 0.25, "mouthSmileRight": 0.25,
             "mouthStretchLeft": 0.35, "mouthStretchRight": 0.35}),
    ("O",   {"jawOpen": 0.35, "mouthFunnel": 0.60,
             "mouthPucker": 0.25}),
    ("U",   {"jawOpen": 0.10, "mouthPucker": 0.85,
             "mouthFunnel": 0.30}),
    ("MBP", {"mouthClose": 1.0,
             "mouthPressLeft": 0.55, "mouthPressRight": 0.55,
             "mouthRollLower": 0.35, "mouthRollUpper": 0.35}),
    ("FV",  {"jawOpen": 0.08,
             "mouthRollLower": 0.75, "mouthShrugUpper": 0.10,
             "mouthLowerDownLeft": 0.15, "mouthLowerDownRight": 0.15,
             "mouthUpperUpLeft": 0.20, "mouthUpperUpRight": 0.20}),
    ("L",   {"jawOpen": 0.30, "tongueOut": 0.35,
             "mouthShrugLower": 0.10}),
    ("WQ",  {"jawOpen": 0.20, "mouthPucker": 0.65,
             "mouthFunnel": 0.35}),
    ("etc", {"jawOpen": 0.20,
             "mouthStretchLeft": 0.25, "mouthStretchRight": 0.25,
             "mouthLowerDownLeft": 0.18, "mouthLowerDownRight": 0.18}),
    ("S",   {"jawOpen": 0.08,
             "mouthSmileLeft": 0.15, "mouthSmileRight": 0.15,
             "mouthStretchLeft": 0.20, "mouthStretchRight": 0.20,
             "mouthLowerDownLeft": 0.10, "mouthLowerDownRight": 0.10}),
    ("TH",  {"jawOpen": 0.22, "tongueOut": 0.55,
             "mouthStretchLeft": 0.10, "mouthStretchRight": 0.10}),
]

# Short phrase timelines for the "sequence" matrix: (time_seconds, viseme).
# These are not meant as real lip-sync (no audio) -- they are the kind of
# viseme successions a face rig has to handle smoothly, so we can eyeball
# coarticulation behavior.
SEQUENCES: List[Tuple[str, List[Tuple[float, str]]]] = [
    ("hello",   [(0.00, "sil"), (0.10, "E"),   (0.22, "L"),   (0.35, "O"),   (0.50, "sil")]),
    ("mama",    [(0.00, "sil"), (0.10, "MBP"), (0.22, "AI"),  (0.34, "MBP"), (0.46, "AI"),
                 (0.58, "sil")]),
    ("phone",   [(0.00, "sil"), (0.12, "FV"),  (0.26, "O"),   (0.40, "U"),   (0.52, "sil")]),
    ("thought", [(0.00, "sil"), (0.12, "TH"),  (0.28, "O"),   (0.42, "etc"), (0.54, "sil")]),
]


CHAR_PRESETS = [
    dict(label="adult_female",  gender="female", seed=311,
         skin=(0.88, 0.72, 0.58), eye=(0.30, 0.55, 0.50),
         hair=(0.40, 0.25, 0.15), hair_style="medium"),
    dict(label="adult_male",    gender="male",   seed=412,
         skin=(0.76, 0.58, 0.44), eye=(0.30, 0.18, 0.08),
         hair=(0.10, 0.07, 0.05), hair_style="short"),
]


def _make_character(preset, blendshapes=None) -> Character:
    random.seed(preset["seed"])
    shape = skeleton.random_shape(gender=preset["gender"])
    hair = preset["hair"]
    skin = preset["skin"]
    lip = (min(1.0, skin[0] * 0.84 + 0.14), skin[1] * 0.55, skin[2] * 0.55)
    app = Appearance(
        gender=preset["gender"],
        skin_color=skin,
        eye_color=preset["eye"],
        seed=preset["seed"] / 1000.0,
        hair_style=preset["hair_style"],
        hair_color=hair,
        eyebrow_color=tuple(min(1.0, c * 0.82) for c in hair),
        facial_hair_style=preset.get("facial_hair", "none"),
        facial_hair_color=tuple(skin[i] * 0.46 + hair[i] * 0.26 for i in range(3)),
        lip_color=lip,
        expression="neutral",
        age_group=preset.get("age", "adult"),
        blendshapes=blendshapes,
        top_style="tshirt",
        top_color=(0.26, 0.34, 0.46),
        bottom_style="pants",
        bottom_color=(0.12, 0.13, 0.16),
    )
    ch = Character(shape=shape, appearance=app)
    ch.build_gpu()
    ch.set_pose(skeleton.t_pose(len(ch.bones)))
    return ch


def _head_target(ch):
    head_idx = skeleton.bone_index(ch.bones, "head")
    bone_mats = ch.bone_matrices()
    length = float(np.linalg.norm(np.asarray(ch.bones[head_idx][3], dtype=np.float32)))
    local = np.array([0.0, length * 0.40, 0.055, 1.0], dtype=np.float32)
    p = bone_mats[head_idx] @ local
    return p[:3]


def _view(target, yaw, dist=0.28):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([dist * math.sin(yaw), 0.0, dist * math.cos(yaw)],
                            dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


def _draw(ch, proj, view, width, height, fbo_ms, fbo_res, skin_prog):
    app = ch.appearance
    bone_mats = ch.bone_matrices()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(skin_prog.prog)
    glUniformMatrix4fv(skin_prog.u_proj, 1, GL_TRUE, proj)
    glUniformMatrix4fv(skin_prog.u_view, 1, GL_TRUE, view)
    renderer.upload_bones(skin_prog.u_bones, bone_mats)
    glUniform1f(skin_prog.u_seed, float(app.seed))
    for d in ch.drawables:
        glUniform3f(skin_prog.u_color, *d.color)
        glUniform1i(skin_prog.u_mode, d.mode)
        if hasattr(skin_prog, "u_material"):
            glUniform1i(skin_prog.u_material, int(getattr(d, "material", 0)))
        if hasattr(skin_prog, "u_bend_inflate"):
            glUniform1f(skin_prog.u_bend_inflate,
                        0.025 if d.mode == 1 else 0.0)
        d.mesh_gpu.draw()
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(img)


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    """Paint a small ASCII label in the upper-left of the tile."""
    from PIL import Image as PImage, ImageDraw, ImageFont
    pim = PImage.fromarray(img, "RGBA")
    draw = ImageDraw.Draw(pim)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle((4, 4, 4 + 8 * len(text) + 8, 22), fill=(0, 0, 0, 180))
    draw.text((8, 6), text, fill=(255, 255, 255, 255), font=font)
    return np.array(pim)


def _interp(a: Dict[str, float], b: Dict[str, float], u: float) -> Dict[str, float]:
    u = max(0.0, min(1.0, u))
    u = u * u * (3.0 - 2.0 * u)
    keys = set(a.keys()) | set(b.keys())
    return {k: a.get(k, 0.0) * (1.0 - u) + b.get(k, 0.0) * u for k in keys}


def _sample_sequence(timeline: List[Tuple[float, str]], t: float) -> Dict[str, float]:
    vmap = {name: w for name, w in VISEMES}
    if t <= timeline[0][0]:
        return dict(vmap.get(timeline[0][1], {}))
    if t >= timeline[-1][0]:
        return dict(vmap.get(timeline[-1][1], {}))
    for i in range(len(timeline) - 1):
        (ta, na), (tb, nb) = timeline[i], timeline[i + 1]
        if ta <= t <= tb:
            span = tb - ta
            u = 0.0 if span <= 0 else (t - ta) / span
            return _interp(vmap.get(na, {}), vmap.get(nb, {}), u)
    return {}


def run_visemes(args, fbo_ms, fbo_res, skin_prog):
    proj = mathx.perspective(math.radians(26),
                             args.width / max(args.height, 1), 0.02, 20.0)
    n = 0
    for preset in CHAR_PRESETS:
        row = []
        for viseme, weights in VISEMES:
            ch = _make_character(preset, blendshapes=weights)
            target = _head_target(ch)
            view = _view(target, 0.0, dist=0.22)
            img = _draw(ch, proj, view, args.width, args.height,
                        fbo_ms, fbo_res, skin_prog)
            row.append(_label_tile(img, viseme))
            ch.delete()
            n += 1
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out,
                           f"{args.tag}_{preset['label']}_visemes.png")
        Image.fromarray(strip, "RGBA").save(out)
        print(f"  visemes {preset['label']}")
    # Three-quarter angle pass -- exposes teeth-clip issues the frontal view hides.
    for preset in CHAR_PRESETS:
        row = []
        for viseme, weights in VISEMES:
            ch = _make_character(preset, blendshapes=weights)
            target = _head_target(ch)
            view = _view(target, math.radians(25), dist=0.22)
            img = _draw(ch, proj, view, args.width, args.height,
                        fbo_ms, fbo_res, skin_prog)
            row.append(_label_tile(img, viseme))
            ch.delete()
            n += 1
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out,
                           f"{args.tag}_{preset['label']}_visemes_3q.png")
        Image.fromarray(strip, "RGBA").save(out)
        print(f"  visemes 3q {preset['label']}")
    return n


def run_sequences(args, fbo_ms, fbo_res, skin_prog):
    proj = mathx.perspective(math.radians(26),
                             args.width / max(args.height, 1), 0.02, 20.0)
    frames = 10
    n = 0
    for preset in CHAR_PRESETS[:1]:
        for name, timeline in SEQUENCES:
            row = []
            duration = timeline[-1][0]
            for fi in range(frames):
                t = duration * (fi / (frames - 1))
                weights = _sample_sequence(timeline, t)
                ch = _make_character(preset, blendshapes=weights)
                target = _head_target(ch)
                view = _view(target, 0.0, dist=0.22)
                img = _draw(ch, proj, view, args.width, args.height,
                            fbo_ms, fbo_res, skin_prog)
                row.append(_label_tile(img, f"{fi:02d}"))
                ch.delete()
                n += 1
            strip = np.concatenate(row, axis=1)
            out = os.path.join(args.out,
                               f"{args.tag}_{preset['label']}_seq_{name}.png")
            Image.fromarray(strip, "RGBA").save(out)
            print(f"  seq {preset['label']} {name}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/viseme_baseline")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--width",  type=int, default=512)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--mode", choices=["visemes", "sequences", "all"],
                    default="all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    win = _make_context(args.width, args.height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    fbo_ms, fbo_res = _make_fbo(args.width, args.height)
    skin_prog = renderer.SkinProgram()
    glClearColor(0.10, 0.11, 0.14, 1.0)

    total = 0
    if args.mode in ("visemes", "all"):
        total += run_visemes(args, fbo_ms, fbo_res, skin_prog)
    if args.mode in ("sequences", "all"):
        total += run_sequences(args, fbo_ms, fbo_res, skin_prog)
    glfw.terminate()
    print(f"done: {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
