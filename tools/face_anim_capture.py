"""Capture renders for the ARKit-52 / FACS facial-animation rig.

Modes:

    --matrix=presets   render the named expression set (FACS-coded) on
                       a matched set of preset characters. One PNG per
                       (character x expression x angle).

    --matrix=clip      render a short transition clip per character:
                       neutral -> target -> neutral, sampled at N
                       evenly-spaced frames, stitched into a horizontal
                       strip.

    --matrix=channels  isolation strip: render a single character with
                       *only* one ARKit channel active at w=1, one
                       column per channel. Sanity-check that each
                       channel actually moves geometry.

Usage:
    env/bin/python -m tools.face_anim_capture \\
        --out screenshots/face_anim_cycle1 --matrix=presets
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


# --- presets (face we render) ------------------------------------------------

CHAR_PRESETS = [
    dict(label="adult_female",  gender="female", seed=311,
         skin=(0.88, 0.72, 0.58), eye=(0.30, 0.55, 0.50),
         hair=(0.40, 0.25, 0.15), hair_style="medium"),
    dict(label="adult_male",    gender="male",   seed=412,
         skin=(0.76, 0.58, 0.44), eye=(0.30, 0.18, 0.08),
         hair=(0.10, 0.07, 0.05), hair_style="short", facial_hair="stubble"),
    dict(label="young_female",  gender="female", seed=513,
         skin=(0.94, 0.78, 0.66), eye=(0.25, 0.55, 0.65),
         hair=(0.55, 0.35, 0.20), hair_style="long"),
    dict(label="elder_male",    gender="male",   seed=614,
         skin=(0.86, 0.70, 0.58), eye=(0.40, 0.40, 0.30),
         hair=(0.85, 0.82, 0.78), hair_style="short", facial_hair="full",
         age="elder"),
]

EXPRESSIONS = [
    "neutral", "smile", "smile_duchenne", "frown", "sad",
    "surprise", "fear", "anger", "disgust", "contempt",
]

ANGLES = [
    ("front", 0.0, 0.42),
    ("threeq", math.radians(32), 0.42),
    ("profile", math.radians(72), 0.46),
    ("closeup", math.radians(18), 0.30),
]

LIGHTING_STYLES = {
    "portrait": 0,
    "soft": 1,
    "raking": 2,
    "warmcool": 3,
}


def _make_character(preset, expression=None, blendshapes=None) -> Character:
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
        expression=expression or preset.get("expression", "neutral"),
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
    local = np.array([0.0, length * 0.55, 0.055, 1.0], dtype=np.float32)
    p = bone_mats[head_idx] @ local
    return p[:3]


def _view(target, yaw, dist=0.45):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([dist * math.sin(yaw), 0.0, dist * math.cos(yaw)],
                            dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


def _setup_gl(width, height):
    win = _make_context(width, height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    fbo_ms, fbo_res = _make_fbo(width, height)
    skin_prog = renderer.SkinProgram()
    glClearColor(0.10, 0.11, 0.14, 1.0)
    return win, fbo_ms, fbo_res, skin_prog


def _draw_to_array(ch, proj, view, width, height, fbo_ms, fbo_res, skin_prog,
                   light_style=0):
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
    if hasattr(skin_prog, "u_light_style"):
        glUniform1i(skin_prog.u_light_style, int(light_style))
    if hasattr(skin_prog, "u_print_style"):
        glUniform1i(skin_prog.u_print_style, int(app.print_style))
        glUniform1f(skin_prog.u_print_strength, float(app.print_strength))
        glUniform1i(skin_prog.u_stamp_style, int(app.stamp_style))
        glUniform1f(skin_prog.u_stamp_strength, float(app.stamp_strength))
    for d in ch.drawables:
        glUniform3f(skin_prog.u_color, *d.color)
        glUniform1i(skin_prog.u_mode, d.mode)
        d.mesh_gpu.draw()
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(img)


def _save(img, path):
    Image.fromarray(img, "RGBA").save(path)


# --- matrices ---------------------------------------------------------------

def run_presets(args, fbo_ms, fbo_res, skin_prog):
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    n = 0
    light_items = _lighting_items(args)
    for preset in CHAR_PRESETS:
        for expr in EXPRESSIONS:
            ch = _make_character(preset, expression=expr)
            target = _head_target(ch)
            for light_label, light_id in light_items:
                for angle_label, yaw, dist in ANGLES:
                    view = _view(target, yaw, dist=dist)
                    img = _draw_to_array(ch, proj, view, args.width, args.height,
                                         fbo_ms, fbo_res, skin_prog,
                                         light_style=light_id)
                    light_suffix = f"_{light_label}" if len(light_items) > 1 else ""
                    out = os.path.join(
                        args.out,
                        f"{args.tag}_{preset['label']}_{expr}_{angle_label}{light_suffix}.png",
                    )
                    _save(img, out)
                    n += 1
            ch.delete()
            print(f"  preset {preset['label']} {expr}")
    return n


def run_clips(args, fbo_ms, fbo_res, skin_prog):
    """Stitch a 9-frame strip per (character, transition target).

    Captures neutral -> target -> neutral. Strip layout is horizontal so
    eye/mouth motion reads left-to-right at a glance.
    """
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    transitions = ["smile_duchenne", "surprise", "anger", "blink"]
    frames = 9
    n = 0
    for preset in CHAR_PRESETS[:2]:
        for target_name in transitions:
            clip = face_anim.clip_from_presets([
                (0.0, "neutral"),
                (0.5, target_name),
                (1.0, "neutral"),
            ])
            tiles = []
            for fi in range(frames):
                t = fi / (frames - 1)
                weights = clip.sample(t)
                ch = _make_character(preset, expression="neutral",
                                     blendshapes=weights)
                target = _head_target(ch)
                view = _view(target, 0.0, dist=0.42)
                img = _draw_to_array(ch, proj, view,
                                     args.width, args.height,
                                     fbo_ms, fbo_res, skin_prog,
                                     light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
                tiles.append(img)
                ch.delete()
            strip = np.concatenate(tiles, axis=1)
            out = os.path.join(
                args.out,
                f"{args.tag}_{preset['label']}_{target_name}_clip.png",
            )
            _save(strip, out)
            n += 1
            print(f"  clip {preset['label']} {target_name}")
    return n


def run_channels(args, fbo_ms, fbo_res, skin_prog):
    """One column per channel, one row per character. Each cell shows
    the character with ONLY that ARKit channel set to 1.0.

    Skip channels whose effect is invisible in this primitive set
    (tongue/jaw secondaries/lip rolls) so the strip stays meaningful.
    """
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    visible_channels = [
        "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeWideLeft",
        "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
        "browInnerUp", "browOuterUpLeft", "browDownLeft",
        "cheekSquintLeft", "noseSneerLeft",
        "jawOpen",
        "mouthSmileLeft", "mouthFrownLeft", "mouthDimpleLeft",
        "mouthStretchLeft", "mouthPucker", "mouthFunnel",
        "mouthUpperUpLeft", "mouthLowerDownLeft",
        "mouthPressLeft", "mouthLeft",
    ]
    n = 0
    for preset in CHAR_PRESETS[:1]:
        rows = []
        for ch_name in visible_channels:
            weights = {ch_name: 1.0}
            ch = _make_character(preset, expression="neutral",
                                 blendshapes=weights)
            target = _head_target(ch)
            view = _view(target, 0.0, dist=0.42)
            img = _draw_to_array(ch, proj, view, args.width, args.height,
                                 fbo_ms, fbo_res, skin_prog,
                                 light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
            rows.append(img)
            ch.delete()
            n += 1
            print(f"  channel {ch_name}")
        # 6 columns wide; pad rows to a multiple of 6.
        cols = 6
        while len(rows) % cols != 0:
            rows.append(np.zeros_like(rows[0]))
        grid_rows = []
        for i in range(0, len(rows), cols):
            grid_rows.append(np.concatenate(rows[i:i + cols], axis=1))
        grid = np.concatenate(grid_rows, axis=0)
        out = os.path.join(args.out, f"{args.tag}_{preset['label']}_channels.png")
        _save(grid, out)
    return n


def run_asymmetry(args, fbo_ms, fbo_res, skin_prog):
    """Captures focused on subtle asymmetry: contempt, wink, single-side
    brow flash, asymmetric anger.

    Each row is one character; columns are the asymmetric expressions.
    """
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    cases = [
        ("wink_left",        {"eyeBlinkLeft": 1.0,
                              "mouthSmileLeft": 0.5,
                              "mouthSmileRight": 0.5}),
        ("contempt",         face_anim.preset_weights("contempt")),
        ("brow_flash_L",     {"browOuterUpLeft": 0.85,
                              "browInnerUp": 0.30}),
        ("asym_anger",       {"browDownLeft": 0.95,
                              "browDownRight": 0.55,
                              "noseSneerLeft": 0.55,
                              "mouthPressLeft": 0.55,
                              "mouthPressRight": 0.30,
                              "eyeSquintLeft": 0.50}),
        ("look_left",        {"eyeLookOutLeft": 0.85,
                              "eyeLookInRight": 0.85}),
        ("look_up",          {"eyeLookUpLeft": 0.85,
                              "eyeLookUpRight": 0.85,
                              "browInnerUp": 0.40}),
    ]
    n = 0
    for preset in CHAR_PRESETS:
        row = []
        for label, weights in cases:
            ch = _make_character(preset, expression="neutral",
                                 blendshapes=weights)
            target = _head_target(ch)
            view = _view(target, 0.0, dist=0.40)
            img = _draw_to_array(ch, proj, view, args.width, args.height,
                                 fbo_ms, fbo_res, skin_prog,
                                 light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
            row.append(img)
            ch.delete()
            n += 1
            print(f"  asym {preset['label']} {label}")
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out, f"{args.tag}_{preset['label']}_asym.png")
        _save(strip, out)
    return n


def run_narrative(args, fbo_ms, fbo_res, skin_prog):
    """A 12-frame narrative clip per character: neutral -> surprise ->
    smile_duchenne -> neutral. Tests multi-key transitions, not just a
    single in/out."""
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    n = 0
    clip = face_anim.clip_from_presets([
        (0.0, "neutral"),
        (0.30, "surprise"),
        (0.65, "smile_duchenne"),
        (1.0, "neutral"),
    ])
    frames = 12
    for preset in CHAR_PRESETS[:2]:
        row = []
        for fi in range(frames):
            t = fi / (frames - 1)
            weights = clip.sample(t)
            ch = _make_character(preset, expression="neutral",
                                 blendshapes=weights)
            target = _head_target(ch)
            view = _view(target, 0.0, dist=0.42)
            img = _draw_to_array(ch, proj, view,
                                 args.width, args.height,
                                 fbo_ms, fbo_res, skin_prog,
                                 light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
            row.append(img)
            ch.delete()
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out, f"{args.tag}_{preset['label']}_narrative.png")
        _save(strip, out)
        n += 1
        print(f"  narrative {preset['label']}")
    return n


def run_crossvalidate(args, fbo_ms, fbo_res, skin_prog):
    """Apply the FACS expression set to a sweep of randomly-generated
    characters (genders / ages / skin tones / hair) to confirm the rig
    works without per-character tuning.

    Layout: one image per random character; columns are expressions.
    """
    from human.character import random_appearance
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    expressions = ["neutral", "smile_duchenne", "surprise", "anger",
                   "disgust", "contempt"]
    seeds = list(range(1000, 1006))
    n = 0
    for seed in seeds:
        random.seed(seed)
        gender = random.choice(["male", "female"])
        shape = skeleton.random_shape(gender=gender)
        # Pull a randomized appearance, then *override* expression below.
        random.seed(seed + 1)
        app = random_appearance(gender)
        app.seed = seed / 10000.0
        row = []
        for expr in expressions:
            app.expression = expr
            app.blendshapes = None
            ch = Character(shape=shape, appearance=app)
            ch.build_gpu()
            ch.set_pose(skeleton.t_pose(len(ch.bones)))
            target = _head_target(ch)
            view = _view(target, math.radians(20), dist=0.42)
            img = _draw_to_array(ch, proj, view, args.width, args.height,
                                 fbo_ms, fbo_res, skin_prog,
                                 light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
            row.append(img)
            ch.delete()
            n += 1
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out,
                           f"{args.tag}_seed{seed}_{gender}_xchar.png")
        _save(strip, out)
        print(f"  xchar seed{seed} ({gender})")
    return n


def run_with_pose(args, fbo_ms, fbo_res, skin_prog):
    """Confirm facial weights compose with body animation. Walks the
    character one step while applying surprise + look_left, captures
    head close-ups."""
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    n = 0
    weights = face_anim.preset_weights("surprise")
    weights["eyeLookOutLeft"] = 0.7
    weights["eyeLookInRight"] = 0.7
    for preset in CHAR_PRESETS[:2]:
        ch = _make_character(preset, expression="neutral", blendshapes=weights)
        row = []
        for fi in range(6):
            t = fi * 0.18
            pose, root = skeleton.walk_pose(ch.bones, t=t, speed=2.2)
            ch.set_pose(pose, root_offset=(0.0, root[1], 0.0))
            target = _head_target(ch)
            view = _view(target, 0.0, dist=0.55)
            img = _draw_to_array(ch, proj, view, args.width, args.height,
                                 fbo_ms, fbo_res, skin_prog,
                                 light_style=LIGHTING_STYLES[args.lighting if args.lighting != "all" else "portrait"])
            row.append(img)
        ch.delete()
        strip = np.concatenate(row, axis=1)
        out = os.path.join(args.out, f"{args.tag}_{preset['label']}_walk_face.png")
        _save(strip, out)
        n += 1
        print(f"  walk+face {preset['label']}")
    return n


def _lighting_items(args):
    if args.lighting == "all":
        return list(LIGHTING_STYLES.items())
    return [(args.lighting, LIGHTING_STYLES[args.lighting])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/face_anim")
    ap.add_argument("--tag", default="anim")
    ap.add_argument("--matrix",
                    choices=["presets", "clips", "channels", "asymmetry",
                             "narrative", "crossvalidate", "with_pose",
                             "all"],
                    default="presets")
    ap.add_argument("--width",  type=int, default=512)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--lighting",
                    choices=list(LIGHTING_STYLES.keys()) + ["all"],
                    default="portrait")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _, fbo_ms, fbo_res, skin_prog = _setup_gl(args.width, args.height)

    total = 0
    if args.matrix in ("presets", "all"):
        total += run_presets(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("clips", "all"):
        total += run_clips(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("channels", "all"):
        total += run_channels(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("asymmetry", "all"):
        total += run_asymmetry(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("narrative", "all"):
        total += run_narrative(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("crossvalidate", "all"):
        total += run_crossvalidate(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("with_pose", "all"):
        total += run_with_pose(args, fbo_ms, fbo_res, skin_prog)

    glfw.terminate()
    print(f"done: {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
