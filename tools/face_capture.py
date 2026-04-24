"""Capture a face-focused audit grid for the realistic-faces cycle.

For each seed per gender, render the head framed tight from front / 3q /
profile. The camera targets the head bone (not the pelvis), so face
features dominate the frame.

Usage:
    env/bin/python -m tools.face_capture --out screenshots/face_baseline --tag baseline
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
from human.character import Character, random_appearance
from human.capture import _make_context, _make_fbo


VIEWS = [
    ("front",   0.0,              0.0),
    ("threeQ",  math.radians(32), 0.0),
    ("profile", math.radians(90), 0.0),
    ("up",      math.radians(20), math.radians(-15)),  # slight from below
]


def _head_world_pos(character):
    """Approximate head centre in world space (rest pose = pose 0)."""
    bi = None
    for i, b in enumerate(character.bones):
        if b[0] == "head":
            bi = i
            break
    if bi is None:
        return np.array([0.0, 1.6, 0.0], dtype=np.float32)
    M = character.bone_matrices()[bi]   # 4x4
    # bone local origin is the joint; shift up a bit into the skull
    tip = np.asarray(character.bones[bi][3], dtype=np.float32)
    local = np.array([0.0, float(np.linalg.norm(tip)) * 0.55, 0.0, 1.0])
    world = M @ local
    return world[:3].astype(np.float32)


def _face_view(target, yaw, pitch, dist=0.55):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch),
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/face_baseline")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=800)
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
    proj = mathx.perspective(math.radians(30),
                             args.width / max(args.height, 1), 0.05, 20.0)

    glClearColor(0.10, 0.11, 0.14, 1.0)

    n = 0
    genders = [g.strip() for g in args.genders.split(",") if g.strip()]
    for gender in genders:
        for s in range(args.seeds):
            seed = 100 + s
            random.seed(seed)
            shape = skeleton.random_shape(gender=gender)
            appearance = random_appearance(gender)
            character = Character(shape=shape, appearance=appearance)
            character.build_gpu()

            # T-pose so the head sits level.
            character.set_pose(skeleton.t_pose(len(character.bones)))
            bone_mats = character.bone_matrices()
            head_pos = _head_world_pos(character)

            for view_name, yaw, pitch in VIEWS:
                view = _face_view(head_pos, yaw, pitch)
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
                out_name = (f"{args.tag}_{gender}_seed{s}_{view_name}.png")
                Image.fromarray(img, "RGBA").save(
                    os.path.join(args.out, out_name))
                n += 1
            character.delete()
            print(f"  done {gender} seed{s}")

    glfw.terminate()
    print(f"done: {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
