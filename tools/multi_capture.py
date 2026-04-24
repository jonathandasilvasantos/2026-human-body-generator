"""Capture many frames in a single GL context for shoulder/clipping audit.

Usage:
    env/bin/python -m tools.multi_capture --out screenshots/baseline --tag baseline

Each shot is one (BVH, time, yaw) combination. Forces a longsleeve top
(the worst case for sleeve/shoulder clipping) and a deterministic seed,
so before/after diffs are pixel-comparable.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from human import mathx, renderer, skeleton
from human import animation as anim_mod
from human.character import Character, Appearance
from human.capture import _make_context, _make_fbo, _camera_view


# Pose specs: (bvh_filename_in_animations, time_seconds, label)
SHOTS = [
    ("walk2.bvh", 0.30, "walk_a"),
    ("walk2.bvh", 0.55, "walk_b"),
    ("bandai/dataset-2_raise-up-both-hands_normal_001.bvh",   1.20, "raiseboth_a"),
    ("bandai/dataset-2_raise-up-both-hands_normal_001.bvh",   2.10, "raiseboth_b"),
    ("bandai/dataset-2_raise-up-right-hand_normal_001.bvh",   1.40, "raiseR"),
    ("bandai/dataset-2_raise-up-left-hand_normal_001.bvh",    1.40, "raiseL"),
    ("bandai/dataset-2_wave-both-hands_normal_001.bvh",       0.90, "wave_a"),
    ("bandai/dataset-2_wave-both-hands_normal_001.bvh",       1.60, "wave_b"),
    ("bandai/dataset-1_punch_normal_001.bvh",                 0.80, "punch"),
    ("bandai/dataset-1_bow_normal_001.bvh",                   1.30, "bow"),
]

# Yaws (radians) per shot. Front, three-quarter, side.
YAWS = [0.0, math.radians(35), math.radians(80)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/baseline")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gender", default="male", choices=["male", "female", "neutral"])
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    win = _make_context(args.width, args.height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    skin_prog = renderer.SkinProgram()
    fbo_ms, fbo_res = _make_fbo(args.width, args.height)

    # Force a deterministic male character with a longsleeve in a loud color
    # so sleeve geometry is unambiguous.
    shape = skeleton.random_shape(gender=args.gender)
    appearance = Appearance(
        gender=args.gender,
        skin_color=(0.86, 0.70, 0.58),
        eye_color=(0.20, 0.35, 0.60),
        seed=float(args.seed),
        hair_style="short",
        hair_color=(0.10, 0.07, 0.05),
        eyebrow_color=(0.10, 0.07, 0.05),
        facial_hair_style="none",
        lip_color=(0.55, 0.30, 0.25),
        top_style="longsleeve",
        top_color=(0.90, 0.18, 0.18),  # bright red sleeves
        bottom_style="pants",
        bottom_color=(0.10, 0.10, 0.14),
        shoe_style="sneakers",
        shoe_color=(0.08, 0.08, 0.10),
    )
    character = Character(shape=shape, appearance=appearance)
    character.build_gpu()

    proj = mathx.perspective(math.radians(40), args.width / max(args.height, 1), 0.1, 50.0)

    # cache loaded animations
    anims = {}
    def get_anim(p):
        if p not in anims:
            anims[p] = anim_mod.load(os.path.join(ROOT, "animations", p), character.bones)
        return anims[p]

    bg = (0.10, 0.11, 0.14, 1.0)
    glClearColor(*bg)

    n = 0
    for bvh_rel, t, label in SHOTS:
        try:
            anim = get_anim(bvh_rel)
        except Exception as e:
            print(f"skip {bvh_rel}: {e}")
            continue
        p, root = anim.sample(t)
        character.set_pose(p, root_offset=(0.0, root[1], 0.0))
        bone_mats = character.bone_matrices()

        for yi, yaw in enumerate(YAWS):
            # Frame the upper body — that's where shoulders/sleeves live.
            view = _camera_view(target=(0.0, 0.30, 0.0), dist=2.0, yaw=yaw, pitch=0.04)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
            glViewport(0, 0, args.width, args.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(skin_prog.prog)
            glUniformMatrix4fv(skin_prog.u_proj, 1, GL_TRUE, proj)
            glUniformMatrix4fv(skin_prog.u_view, 1, GL_TRUE, view)
            renderer.upload_bones(skin_prog.u_bones, bone_mats)
            glUniform1f(skin_prog.u_seed, float(character.appearance.seed))
            for d in character.drawables:
                glUniform3f(skin_prog.u_color, *d.color)
                glUniform1i(skin_prog.u_mode, d.mode)
                d.mesh_gpu.draw()

            glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
            glBlitFramebuffer(0, 0, args.width, args.height, 0, 0, args.width, args.height,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
            pixels = glReadPixels(0, 0, args.width, args.height, GL_RGBA, GL_UNSIGNED_BYTE)
            img = np.frombuffer(pixels, dtype=np.uint8).reshape(args.height, args.width, 4)
            img = np.flipud(img)
            out_name = f"{args.tag}_{label}_yaw{yi}.png"
            Image.fromarray(img, "RGBA").save(os.path.join(args.out, out_name))
            n += 1
            print(f"wrote {out_name}")

    character.delete()
    glfw.terminate()
    print(f"done: {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
