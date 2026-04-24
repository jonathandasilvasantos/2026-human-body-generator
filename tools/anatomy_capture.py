"""Capture a body-shape audit grid for the dimorphism cycle.

For each of N seeds per gender, render the character in a few canonical
poses from front / three-quarter / side. Frames the FULL body so torso
proportions, hips, thighs, and waist line are all visible.

Usage:
    env/bin/python -m tools.anatomy_capture --out screenshots/anat_baseline --tag baseline
    env/bin/python -m tools.anatomy_capture --out screenshots/anat_c1 --tag c1
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
from human.character import Character, random_appearance
from human.capture import _make_context, _make_fbo


YAWS = [
    ("front", 0.0),
    ("three_q", math.radians(35)),
    ("side", math.radians(85)),
    ("back", math.radians(180)),
]


def _full_body_view(yaw, dist=3.4, pitch=0.04, target=(0.0, 0.05, 0.0)):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch) + 0.05,
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


# Pose specs for the audit. Mix static + animated so we catch both rest
# silhouette and motion-deformed silhouette.
POSES = [
    ("tpose",  "static", None, 0.0),
    ("walkA",  "static", "walk_pose", 1.43),     # passing pose
    ("walkB",  "static", "walk_pose", 2.86),     # heel strike
    ("raise",  "bvh", "bandai/dataset-2_raise-up-both-hands_normal_001.bvh", 1.20),
    ("bow",    "bvh", "bandai/dataset-1_bow_normal_001.bvh", 1.30),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/anat_baseline")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seeds", type=int, default=4,
                    help="How many seeds per gender")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--genders", default="male,female")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    win = _make_context(args.width, args.height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    fbo_ms, fbo_res = _make_fbo(args.width, args.height)
    skin_prog = renderer.SkinProgram()
    proj = mathx.perspective(math.radians(40),
                             args.width / max(args.height, 1), 0.1, 50.0)

    bg = (0.10, 0.11, 0.14, 1.0)
    glClearColor(*bg)

    # cache loaded animations (per character, since rig adapts to bones)
    anim_cache = {}

    def get_anim(rel, character):
        key = (rel, id(character))
        if key not in anim_cache:
            anim_cache[key] = anim_mod.load(os.path.join(ROOT, "animations", rel),
                                            character.bones)
        return anim_cache[key]

    n = 0
    genders = [g.strip() for g in args.genders.split(",") if g.strip()]
    for gender in genders:
        for s in range(args.seeds):
            seed = 100 + s
            random.seed(seed)
            shape = skeleton.random_shape(gender=gender)
            # Use a fixed bright top so silhouette is unambiguous, but
            # keep procedural bottoms / hair / etc.
            appearance = random_appearance(gender)
            # Force a tight top so the body shape reads, not a baggy dress.
            appearance.top_style = "tank"
            appearance.bottom_style = "shorts"
            character = Character(shape=shape, appearance=appearance)
            character.build_gpu()

            for pname, kind, src, t in POSES:
                if kind == "static" and src is None:
                    p = skeleton.t_pose(len(character.bones))
                    root_y = 0.0
                elif kind == "static" and src == "walk_pose":
                    p, root = skeleton.walk_pose(character.bones, t=t, speed=2.2)
                    root_y = root[1]
                elif kind == "bvh":
                    try:
                        anim = get_anim(src, character)
                    except Exception as e:
                        print(f"skip {src}: {e}")
                        continue
                    p, root = anim.sample(t)
                    root_y = root[1]
                else:
                    continue
                character.set_pose(p, root_offset=(0.0, root_y, 0.0))
                bone_mats = character.bone_matrices()

                for view_name, yaw in YAWS:
                    view = _full_body_view(yaw)
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
                    glBlitFramebuffer(0, 0, args.width, args.height,
                                      0, 0, args.width, args.height,
                                      GL_COLOR_BUFFER_BIT, GL_LINEAR)
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
                    pixels = glReadPixels(0, 0, args.width, args.height,
                                          GL_RGBA, GL_UNSIGNED_BYTE)
                    img = np.frombuffer(pixels, dtype=np.uint8).reshape(
                        args.height, args.width, 4)
                    img = np.flipud(img)
                    out_name = (f"{args.tag}_{gender}_seed{s}_"
                                f"{pname}_{view_name}.png")
                    Image.fromarray(img, "RGBA").save(
                        os.path.join(args.out, out_name))
                    n += 1
            character.delete()
            print(f"  done {gender} seed{s}")

    glfw.terminate()
    print(f"done: {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
