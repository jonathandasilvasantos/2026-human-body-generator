"""Render a .glb mesh from front / left / back / iso views using trimesh +
its built-in pyglet viewer (works headless on macOS via offscreen GL).

Output:
    {out}_{view}.png
    {out}_grid.png   labelled triptych
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


# (name, yaw_deg around +Y, pitch_deg, label)
# yaw=0 is +Z (face the +Z axis); pyrender uses the same convention.
VIEWS = [
    ("front", 0,    -5,  "Front"),
    ("left",  -90,  -5,  "Left"),
    ("back",  180,  -5,  "Back"),
    ("iso",   45,   -20, "3/4 Isometric"),
]


def _camera_pose(target, dist, yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    eye = np.array([
        target[0] + dist * math.cos(pitch) * math.sin(yaw),
        target[1] - dist * math.sin(pitch),
        target[2] + dist * math.cos(pitch) * math.cos(yaw),
    ])
    forward = (target - eye); forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _label(img, text):
    out = img.convert('RGB').copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size=28)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([(0, 0), (out.width, 44)], fill=(0, 0, 0))
    d.text((14, 8), text, fill=(255, 255, 255), font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    args = ap.parse_args()

    scene = trimesh.load(args.glb, force='scene')

    # Reorient: Hunyuan3D-2mv emits in the conditioning-image frame, which
    # is rotated relative to our procedural rig. Apply a fixed rotation
    # so the head is on +Y and the figure faces +Z (front view = yaw 0).
    flip = trimesh.transformations.euler_matrix(math.pi / 2, 0, math.pi)
    scene.apply_transform(flip)

    bounds = scene.bounds
    center = bounds.mean(axis=0)
    size = (bounds[1] - bounds[0]).max()
    dist = size * 1.3

    panels = []
    for name, yaw, pitch, label in VIEWS:
        scene.camera.resolution = (args.width, args.height)
        scene.camera.fov = (42, 42)
        scene.camera_transform = _camera_pose(center, dist, yaw, pitch)
        png_bytes = scene.save_image(resolution=(args.width, args.height),
                                     visible=False, background=(235, 235, 240, 255))
        path = f'{args.out}_{name}.png'
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            f.write(png_bytes)
        print(f'wrote {path}')
        panels.append(_label(Image.open(path), label))

    grid = Image.new('RGB', (args.width * len(VIEWS), args.height), (255, 255, 255))
    for i, p in enumerate(panels):
        grid.paste(p, (i * args.width, 0))
    grid_path = f'{args.out}_grid.png'
    grid.save(grid_path)
    print(f'wrote {grid_path}')


if __name__ == '__main__':
    main()
