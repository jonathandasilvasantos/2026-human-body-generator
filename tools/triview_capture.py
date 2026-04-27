"""Render the same procedural character from 3 fixed views — front,
back, and 3/4 isometric — for image-to-3D pipelines (Trellis 2,
Hunyuan3D, Tripo, Hitem3D, Meshy).

Outputs four files:
    {out}_front.png   — yaw 0    (camera in front of the character)
    {out}_back.png    — yaw pi   (camera behind)
    {out}_iso.png     — yaw pi/4 + pitch ~0.5 (3/4 view)
    {out}_grid.png    — labelled triptych for human review

Usage:
    env/bin/python -m tools.triview_capture --seed 42 \\
        --out screenshots/triview/cycle1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from human import lighting as _lighting
from human import mathx, renderer, skeleton
from human.capture import _make_context, _make_fbo, _camera_view
from human.character import Character, random_appearance


VIEWS = [
    # name,   yaw,           pitch, dist, target_y, label
    # Hunyuan3D-2mv expects exactly these three view names: front, left, back.
    # The "iso" 3/4 view is also rendered for sanity-checking the silhouette.
    # Cycle 5: iso pitch dropped from 0.30 to 0.10 and dist bumped to 3.10
    # so the head doesn't crop and the figure fills less of the frame at
    # comparable scale to the other three orthographic views.
    ("front", 0.0,            0.05, 2.65, -0.20, "Front"),
    ("left",  -math.pi / 2,   0.05, 2.65, -0.20, "Left"),
    ("back",  math.pi,        0.05, 2.65, -0.20, "Back"),
    ("iso",   math.pi / 4,    0.10, 3.10, -0.20, "3/4 Isometric"),
]

# Cycle 12: face close-ups. The body-framing views render the face at
# sub-100px so any improvement to the head compound goes unseen. These
# four views frame the head tightly so the cycle's compare panels carry
# face quality signal too. Camera target sits at chin / nose level
# (~+0.95 in world Y) and the distance is small enough that the face
# fills the viewport. Light profile 1 (Studio Beauty) on these too.
FACE_VIEWS = [
    # name,         yaw,            pitch,  dist,  target_y, label
    # Head sits near world Y ~0.62 (chin at ~0.51, top of head ~0.73 in
    # rest-pose world coords). Aim slightly above the chin for a head-
    # and-shoulders crop.
    ("face_front",  0.0,            -0.02,  0.64,  0.58,  "Face — Front"),
    ("face_3q",     math.pi / 5,    -0.02,  0.64,  0.58,  "Face — 3/4"),
    ("face_left",   -math.pi / 2,   -0.02,  0.64,  0.58,  "Face — Left"),
    ("face_iso",    math.pi / 4,     0.10,  0.68,  0.58,  "Face — Iso"),
]


def _render_view(skin_prog, character, width, height, yaw, pitch, dist,
                 target, fov_deg, light_profile, fbo_ms, fbo_res,
                 eye_offset_y=0.35):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    proj = mathx.perspective(math.radians(fov_deg),
                             width / max(height, 1), 0.05, 50.0)
    # Inline look-at so we can control the eye-Y offset (the default
    # `_camera_view` adds +0.35 to lift the camera for body framing;
    # face close-ups need eye level with the head).
    target_arr = np.asarray(target, dtype=np.float32)
    eye = target_arr + np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch) + eye_offset_y,
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    view = mathx.look_at(eye, target_arr, [0, 1, 0])
    bone_mats = character.bone_matrices()

    glUseProgram(skin_prog.prog)
    glUniformMatrix4fv(skin_prog.u_proj, 1, GL_TRUE, proj)
    glUniformMatrix4fv(skin_prog.u_view, 1, GL_TRUE, view)
    renderer.upload_bones(skin_prog.u_bones, bone_mats)
    glUniform1f(skin_prog.u_seed, float(character.appearance.seed))
    _lighting.apply(skin_prog, _lighting.by_index(int(light_profile)))
    for d in character.drawables:
        glUniform3f(skin_prog.u_color, *d.color)
        glUniform1i(skin_prog.u_mode, d.mode)
        glUniform1i(skin_prog.u_material, int(getattr(d, "material", 0)))
        glUniform1f(skin_prog.u_bend_inflate, 0.006 if d.mode == 1 else 0.0)
        d.mesh_gpu.draw()

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res)
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    img = np.flipud(img).copy()
    return Image.fromarray(img, "RGBA")


def _vertical_gradient_bg(width: int, height: int,
                          top=(245, 247, 250), bottom=(210, 212, 220)
                          ) -> Image.Image:
    """Soft vertical gradient backdrop. Cycle 4 fix #8: closer to a
    photo studio seamless than a flat clear color, which both helps the
    eye separate figure from background and matches the distribution
    image-to-3D models like Hunyuan3D were trained on."""
    bg = Image.new("RGB", (width, height), top)
    pixels = bg.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(width):
            pixels[x, y] = (r, g, b)
    return bg


def _composite_over_gradient(img: Image.Image) -> Image.Image:
    """Take an RGBA render with the bg cleared to a single color and
    composite it over a vertical gradient. Anything alpha~=255 stays
    opaque; anything close to the original clear color is replaced. We
    use a simple chroma-key against the clear color since our render
    has no transparency anywhere except the cleared background."""
    bg = _vertical_gradient_bg(img.width, img.height)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    bg_r, bg_g, bg_b = 235, 235, 240    # matches `bg=(0.92, 0.92, 0.94)` clear
    diff = (np.abs(arr[..., 0].astype(int) - bg_r)
            + np.abs(arr[..., 1].astype(int) - bg_g)
            + np.abs(arr[..., 2].astype(int) - bg_b))
    mask = diff < 12
    arr[mask, 3] = 0
    figure = Image.fromarray(arr, "RGBA")
    out = bg.convert("RGBA")
    out.alpha_composite(figure)
    return out.convert("RGB")


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", size=28)
    except OSError:
        font = ImageFont.load_default()
    pad = 12
    d.rectangle([(0, 0), (out.width, 44)], fill=(0, 0, 0))
    d.text((pad, 8), text, fill=(255, 255, 255), font=font)
    return out


def capture_triview(out_prefix, seed=None, gender=None, pose="t_pose",
                    width=1024, height=1024, fov_deg=42.0,
                    light_profile=1, bg=(0.92, 0.92, 0.94, 1.0)):
    """Renders 3 fixed views of one character. Pose defaults to t_pose
    (canonical neutral) for cleanest image-to-3D input. Light profile
    1 (Studio Beauty) gives a near-white seamless background that the
    image-to-3D models tend to handle best."""
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
        # Use t_pose_wide so arms are truly horizontal (~83° abduction).
        # The plain t_pose() returns zero rotations -- bind pose -- which
        # leaves the arms hanging by the sides on this rig.
        character.set_pose(skeleton.t_pose_wide(character.bones))
    elif pose == "walk_passing":
        p, root = skeleton.walk_pose(character.bones, t=1.43, speed=2.2)
        character.set_pose(p, root_offset=(0.0, root[1], 0.0))
    else:
        raise ValueError(f"unknown pose {pose}")

    fbo_ms, fbo_res = _make_fbo(width, height)

    images = {}
    all_views = list(VIEWS) + list(FACE_VIEWS)
    for name, yaw, pitch, dist, ty, label in all_views:
        is_face = name.startswith("face_")
        # Face close-ups need a tight FOV and no body-framing eye lift.
        view_fov = 34.0 if is_face else fov_deg
        eye_offset = 0.0 if is_face else 0.35
        img = _render_view(skin_prog, character, width, height,
                           yaw=yaw, pitch=pitch, dist=dist,
                           target=(0.0, ty, 0.0), fov_deg=view_fov,
                           light_profile=light_profile,
                           fbo_ms=fbo_ms, fbo_res=fbo_res,
                           eye_offset_y=eye_offset)
        path = f"{out_prefix}_{name}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # F8: composite the figure over a soft vertical gradient backdrop
        # before saving. Both the photoreal target (FLUX.1 Kontext on bender)
        # and the image-to-3D model (Hunyuan3D-2mv) handle a horizon-line
        # backdrop better than a flat clear color.
        composed = _composite_over_gradient(img)
        composed.save(path)
        images[name] = (composed, label, path)
        print(f"wrote {path}")

    # Build labelled grid for human review (one panel per view).
    # Two rows: body views on top, face close-ups on bottom.
    body_panels = [_label(images[n][0], images[n][1]) for n, _, _, _, _, _ in VIEWS]
    face_panels = [_label(images[n][0], images[n][1]) for n, _, _, _, _, _ in FACE_VIEWS]
    grid = Image.new("RGB", (width * len(VIEWS), height * 2), (255, 255, 255))
    for i, p in enumerate(body_panels):
        grid.paste(p, (i * width, 0))
    for i, p in enumerate(face_panels):
        grid.paste(p, (i * width, height))
    grid_path = f"{out_prefix}_grid.png"
    grid.save(grid_path)
    print(f"wrote {grid_path}")

    info = {
        "seed": seed,
        "gender": shape.gender,
        "hair": character.appearance.hair_style,
        "facial_hair": character.appearance.facial_hair_style,
        "top": character.appearance.top_style,
        "bottom": character.appearance.bottom_style,
        "shoes": character.appearance.shoe_style,
    }
    print(f"character: {info}")

    character.delete()
    glfw.terminate()
    return [images[n][2] for n, *_ in VIEWS] + [grid_path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="screenshots/triview/v1",
                    help="output prefix (suffixes _front/_back/_iso/_grid)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gender", choices=["male", "female", "neutral"],
                    default=None)
    ap.add_argument("--pose", default="t_pose",
                    choices=["t_pose", "walk_passing"])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--light", type=int, default=1,
                    help="lighting profile 0-9 (1=Studio Beauty, recommended)")
    args = ap.parse_args()
    capture_triview(out_prefix=args.out, seed=args.seed, gender=args.gender,
                    pose=args.pose, width=args.width, height=args.height,
                    fov_deg=args.fov, light_profile=args.light)


if __name__ == "__main__":
    main()
