"""Unified face-capture tool.

Supports three test matrices:

    --matrix=grid      (default) seeds x genders x angles on T-pose, face
                       framed tight. Mirrors the original face_capture.py.

    --matrix=presets   curated demographic presets (age x gender x
                       expression) swept across static / walk / wave poses
                       and front / three-quarter / profile angles. Mirrors
                       the human-chat realism-cycle capture.

    --matrix=full      both -- grid first, then presets -- in one run.

Usage:
    env/bin/python -m tools.face_capture --out screenshots/face_unified \\
                                         --tag unified --matrix=full
"""

import argparse
import math
import os
import random
import sys

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from human import animation as anim_mod
from human import mathx, renderer, skeleton
from human.capture import _make_context, _make_fbo
from human.character import Appearance, Character, random_appearance


# --- grid matrix (ported from tools/face_capture.py on `main`) --------------

GRID_VIEWS = [
    ("front",   0.0,              0.0),
    ("threeQ",  math.radians(32), 0.0),
    ("profile", math.radians(90), 0.0),
    ("up",      math.radians(20), math.radians(-15)),
]


def _face_view(target, yaw, pitch, dist=0.55):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch),
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


def _head_world_pos(character):
    bi = None
    for i, b in enumerate(character.bones):
        if b[0] == "head":
            bi = i
            break
    if bi is None:
        return np.array([0.0, 1.6, 0.0], dtype=np.float32)
    M = character.bone_matrices()[bi]
    tip = np.asarray(character.bones[bi][3], dtype=np.float32)
    local = np.array([0.0, float(np.linalg.norm(tip)) * 0.55, 0.0, 1.0])
    world = M @ local
    return world[:3].astype(np.float32)


# --- preset matrix (ported from human-chat/tools/face_capture.py) -----------

PRESET_ANGLES = [
    ("front", 0.0),
    ("threeq", math.radians(35.0)),
    ("profile", math.radians(82.0)),
]

PRESET_POSES = [
    ("static", None, 0.0),
    ("walk", "walk", 1.43),
    ("wave", "bandai/dataset-2_wave-both-hands_normal_001.bvh", 1.05),
]

LIGHTING_STYLES = {
    "portrait": 0,
    "soft": 1,
    "raking": 2,
    "warmcool": 3,
}

PRESETS = [
    dict(
        label="young_female_smile", gender="female",
        skin=(0.94, 0.78, 0.66), eyes=(0.25, 0.55, 0.65),
        hair=(0.40, 0.25, 0.15), hair_style="long",
        expression="smile", age_group="young",
        traits=dict(face_asymmetry=0.010, lip_fullness=1.22),
    ),
    dict(
        label="adult_male_neutral", gender="male",
        skin=(0.76, 0.58, 0.44), eyes=(0.30, 0.18, 0.08),
        hair=(0.08, 0.06, 0.05), hair_style="short",
        expression="neutral", age_group="adult", facial_hair="stubble",
        traits=dict(face_asymmetry=0.008, lip_fullness=0.94),
    ),
    dict(
        label="elder_female_squint", gender="female",
        skin=(0.88, 0.72, 0.58), eyes=(0.35, 0.55, 0.30),
        hair=(0.90, 0.85, 0.80), hair_style="medium",
        expression="squint", age_group="elder",
        traits=dict(face_asymmetry=0.013, lip_fullness=1.02),
    ),
    dict(
        label="adult_male_surprised", gender="male",
        skin=(0.42, 0.28, 0.20), eyes=(0.12, 0.08, 0.05),
        hair=(0.08, 0.06, 0.05), hair_style="buzz",
        expression="surprised", age_group="adult",
        traits=dict(face_asymmetry=0.006, lip_fullness=1.02),
    ),
    dict(
        label="adult_female_frown", gender="female",
        skin=(0.28, 0.18, 0.13), eyes=(0.45, 0.30, 0.15),
        hair=(0.60, 0.20, 0.10), hair_style="short",
        expression="frown", age_group="adult",
        traits=dict(face_asymmetry=0.011, lip_fullness=1.16),
    ),
]


def _make_preset_shape(preset, seed):
    random.seed(seed)
    shape = skeleton.random_shape(gender=preset["gender"])
    for key, value in preset["traits"].items():
        setattr(shape, key, value)
    return shape


def _make_preset_appearance(preset, seed):
    hair = preset["hair"]
    eyebrow = tuple(min(1.0, c * 0.82) for c in hair)
    skin = preset["skin"]
    lip = (min(1.0, skin[0] * 0.84 + 0.14), skin[1] * 0.55, skin[2] * 0.55)
    return Appearance(
        gender=preset["gender"],
        skin_color=skin,
        eye_color=preset["eyes"],
        seed=float(seed) / 1000.0,
        hair_style=preset["hair_style"],
        hair_color=hair,
        eyebrow_color=eyebrow,
        facial_hair_style=preset.get("facial_hair", "none"),
        facial_hair_color=tuple(skin[i] * 0.46 + hair[i] * 0.26 for i in range(3)),
        lip_color=lip,
        expression=preset["expression"],
        age_group=preset["age_group"],
        top_style="tshirt",
        top_color=(0.26, 0.34, 0.46),
        bottom_style="pants",
        bottom_color=(0.12, 0.13, 0.16),
    )


def _head_target_pose(character, bone_mats):
    head_idx = skeleton.bone_index(character.bones, "head")
    length = float(np.linalg.norm(np.asarray(character.bones[head_idx][3], dtype=np.float32)))
    local = np.array([0.0, length * 0.55, 0.055, 1.0], dtype=np.float32)
    p = bone_mats[head_idx] @ local
    return p[:3]


# --- shared render helpers --------------------------------------------------

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


def _draw_and_save(character, proj, view, width, height, fbo_ms, fbo_res,
                   skin_prog, out_path, light_style=0):
    app = character.appearance
    bone_mats = character.bone_matrices()
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
    for d in character.drawables:
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
    img = np.flipud(img)
    Image.fromarray(img, "RGBA").save(out_path)


# --- matrix drivers ---------------------------------------------------------

def run_grid(args, fbo_ms, fbo_res, skin_prog):
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)
    genders = [g.strip() for g in args.genders.split(",") if g.strip()]
    light_items = _lighting_items(args)
    wrote = 0
    for gender in genders:
        for s in range(args.seeds):
            seed = 100 + s
            random.seed(seed)
            shape = skeleton.random_shape(gender=gender)
            appearance = random_appearance(gender)
            character = Character(shape=shape, appearance=appearance)
            character.build_gpu()
            character.set_pose(skeleton.t_pose(len(character.bones)))
            head_pos = _head_world_pos(character)
            for light_label, light_id in light_items:
                for view_name, yaw, pitch in GRID_VIEWS:
                    view = _face_view(head_pos, yaw, pitch)
                    light_suffix = f"_{light_label}" if len(light_items) > 1 else ""
                    out_name = f"{args.tag}_{gender}_seed{s}_{view_name}{light_suffix}.png"
                    _draw_and_save(character, proj, view,
                                   args.width, args.height,
                                   fbo_ms, fbo_res, skin_prog,
                                   os.path.join(args.out, out_name),
                                   light_style=light_id)
                    wrote += 1
            character.delete()
            print(f"  grid done {gender} seed{s}")
    return wrote


def run_presets(args, fbo_ms, fbo_res, skin_prog):
    proj = mathx.perspective(math.radians(32),
                             args.width / max(args.height, 1), 0.1, 50.0)
    light_items = _lighting_items(args)
    wrote = 0
    for pi, preset in enumerate(PRESETS):
        seed = 300 + pi * 37
        shape = _make_preset_shape(preset, seed)
        app = _make_preset_appearance(preset, seed)
        character = Character(shape=shape, appearance=app)
        character.build_gpu()

        anim_cache = {}
        for pose_label, pose_kind, pose_time in PRESET_POSES:
            if pose_kind is None:
                character.set_pose(skeleton.t_pose(len(character.bones)))
            elif pose_kind == "walk":
                pose, root = skeleton.walk_pose(character.bones, t=pose_time, speed=2.2)
                character.set_pose(pose, root_offset=(0.0, root[1], 0.0))
            else:
                if pose_kind not in anim_cache:
                    anim_cache[pose_kind] = anim_mod.load(
                        os.path.join(ROOT, "animations", pose_kind),
                        character.bones)
                pose, root = anim_cache[pose_kind].sample(pose_time)
                character.set_pose(pose, root_offset=(0.0, root[1], 0.0))

            bone_mats = character.bone_matrices()
            target = _head_target_pose(character, bone_mats)
            for light_label, light_id in light_items:
                for angle_label, yaw in PRESET_ANGLES:
                    view = _face_view(target, yaw, 0.0, dist=0.50)
                    light_suffix = f"_{light_label}" if len(light_items) > 1 else ""
                    out_name = (f"{args.tag}_{preset['label']}_"
                                f"{pose_label}_{angle_label}{light_suffix}.png")
                    _draw_and_save(character, proj, view,
                                   args.width, args.height,
                                   fbo_ms, fbo_res, skin_prog,
                                   os.path.join(args.out, out_name),
                                   light_style=light_id)
                    wrote += 1
                    print(f"  preset wrote {out_name}")
        character.delete()
    return wrote


def _lighting_items(args):
    if args.lighting == "all":
        return list(LIGHTING_STYLES.items())
    return [(args.lighting, LIGHTING_STYLES[args.lighting])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/face_unified")
    ap.add_argument("--tag", default="unified")
    ap.add_argument("--matrix", choices=["grid", "presets", "full"],
                    default="grid")
    ap.add_argument("--seeds", type=int, default=4,
                    help="grid matrix: seeds per gender")
    ap.add_argument("--genders", default="male,female",
                    help="grid matrix: comma-separated genders")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--lighting",
                    choices=list(LIGHTING_STYLES.keys()) + ["all"],
                    default="portrait",
                    help="lighting preset for surface inspection")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _, fbo_ms, fbo_res, skin_prog = _setup_gl(args.width, args.height)

    total = 0
    if args.matrix in ("grid", "full"):
        total += run_grid(args, fbo_ms, fbo_res, skin_prog)
    if args.matrix in ("presets", "full"):
        total += run_presets(args, fbo_ms, fbo_res, skin_prog)

    glfw.terminate()
    print(f"done: {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
