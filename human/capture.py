"""Headless capture: render a character into ``screenshot.png`` without
needing an interactive window or user input.

Uses a hidden GLFW window + an offscreen FBO. GLFW is required for context
creation even in headless mode (no EGL on macOS).
"""

import math
import os
import random
import sys

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image

from . import mathx, renderer, skeleton
from .character import Character, random_appearance


def _make_context(width, height):
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.VISIBLE, GL_FALSE)
    glfw.window_hint(glfw.SAMPLES, 4)
    win = glfw.create_window(width, height, "capture", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("window create failed")
    glfw.make_context_current(win)
    return win


def _make_fbo(width, height):
    # Multisample color + depth renderbuffers for MSAA, blit to a resolve FBO.
    fbo_ms = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
    color_ms = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, color_ms)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, color_ms)
    depth_ms = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_ms)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4,
                                     GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, depth_ms)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"MS FBO incomplete: {status}")

    fbo_res = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
    color_res = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_res)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, color_res, 0)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"resolve FBO incomplete: {status}")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo_ms, fbo_res


def _camera_view(target=(0.0, 0.25, 0.0), dist=3.0, yaw=0.0, pitch=0.08):
    target = np.asarray(target, dtype=np.float32)
    eye = target + np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch) + 0.25,
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    return mathx.look_at(eye, target, [0, 1, 0])


def capture(
    out_path="screenshot.png",
    seed=None,
    gender=None,
    pose="walk_passing",     # "t_pose" | "walk_passing" | "walk_heel_strike" | "bvh"
    width=1024,
    height=1024,
    yaw=0.0,
    pitch=0.08,
    dist=3.0,
    target=(0.0, 0.25, 0.0),
    bg=(0.09, 0.10, 0.13, 1.0),
    skin_color=(0.85, 0.68, 0.55),
    bvh_path=None,
    bvh_time=0.5,
):
    if seed is not None:
        random.seed(seed)

    win = _make_context(width, height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glClearColor(*bg)

    skin_prog = renderer.SkinProgram()

    shape = skeleton.random_shape(gender=gender)
    character = Character(shape=shape, appearance=random_appearance(shape.gender))
    character.build_gpu()

    if pose == "t_pose":
        character.set_pose(skeleton.t_pose(len(character.bones)))
    elif pose == "walk_passing":
        p, root = skeleton.walk_pose(character.bones, t=1.43, speed=2.2)  # ~phi=pi/2
        character.set_pose(p, root_offset=(0.0, root[1], 0.0))  # no forward translation
    elif pose == "walk_heel_strike":
        p, root = skeleton.walk_pose(character.bones, t=2.86, speed=2.2)  # ~phi=2*pi
        character.set_pose(p, root_offset=(0.0, root[1], 0.0))
    elif pose == "bvh":
        from . import animation as anim_mod
        if bvh_path is None:
            raise ValueError("pose=bvh requires bvh_path")
        anim = anim_mod.load(bvh_path, character.bones)
        p, root = anim.sample(bvh_time)
        character.set_pose(p, root_offset=(0.0, root[1], 0.0))
    else:
        raise ValueError(f"unknown pose {pose}")

    fbo_ms, fbo_res = _make_fbo(width, height)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    proj = mathx.perspective(math.radians(55), width / max(height, 1), 0.1, 50.0)
    view = _camera_view(target=target, dist=dist, yaw=yaw, pitch=pitch)
    bone_mats = character.bone_matrices()

    glUseProgram(skin_prog.prog)
    glUniformMatrix4fv(skin_prog.u_proj, 1, GL_TRUE, proj)
    glUniformMatrix4fv(skin_prog.u_view, 1, GL_TRUE, view)
    renderer.upload_bones(skin_prog.u_bones, bone_mats)
    glUniform1f(skin_prog.u_seed, float(character.appearance.seed))
    for d in character.drawables:
        glUniform3f(skin_prog.u_color, *d.color)
        glUniform1i(skin_prog.u_mode, d.mode)
        d.mesh_gpu.draw()

    # Blit multisample -> resolve, read back
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    img = np.flipud(img)                      # GL origin is bottom-left
    Image.fromarray(img, "RGBA").save(out_path)

    info = {
        "gender": shape.gender,
        "height": shape.height,
        "bulk": shape.bulk,
        "head_size": shape.head_size,
        "hair": character.appearance.hair_style,
        "facial_hair": character.appearance.facial_hair_style,
        "top": character.appearance.top_style,
        "bottom": character.appearance.bottom_style,
        "shoes": character.appearance.shoe_style,
    }
    print(f"wrote {out_path}  {info}")

    character.delete()
    glfw.terminate()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshot.png")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gender", choices=["male", "female", "neutral"], default=None)
    ap.add_argument("--pose", default="walk_passing",
                    choices=["t_pose", "walk_passing", "walk_heel_strike", "bvh"])
    ap.add_argument("--bvh", dest="bvh_path", default=None,
                    help="Path to a BVH file (use with --pose bvh)")
    ap.add_argument("--bvh-time", dest="bvh_time", type=float, default=0.5,
                    help="Time in seconds into the BVH clip to sample (default 0.5)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.08)
    ap.add_argument("--dist", type=float, default=3.0)
    args = ap.parse_args()
    capture(
        out_path=args.out,
        seed=args.seed,
        gender=args.gender,
        pose=args.pose,
        width=args.width,
        height=args.height,
        yaw=args.yaw,
        pitch=args.pitch,
        dist=args.dist,
        bvh_path=args.bvh_path,
        bvh_time=args.bvh_time,
    )


if __name__ == "__main__":
    main()
