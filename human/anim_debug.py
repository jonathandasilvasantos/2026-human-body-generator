"""Automated animation debugger.

Usage:
    env/bin/python -m human.anim_debug animations/walk2.bvh

Produces:
  - ``anim_debug.png``: a horizontal strip of N rendered frames at evenly
    spaced times across the animation, useful for spotting pose drift,
    body rotation, arm stiffness, etc.
  - A diagnostic report printed to stdout, including:
      * duration / frame count / frame time
      * mapped bones count
      * root trajectory (x, y, z) range
      * root Y-rotation range (detects unintended whole-body spin)
      * per-bone rotation range across the clip (detects bones stuck at
        one pose, or oscillating wildly)
      * source-vs-ours world-direction mismatch for each mapped bone
        (distance = how far our bone's forward direction is from the
        source's; ideal = ~0)
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image

from . import animation as anim_mod
from . import bvh as bvh_mod
from . import skeleton
from .capture import capture


# --- world direction verification ------------------------------------------

def _bvh_world_matrix(bvh: bvh_mod.BVHFile, j_idx: int, frame: int) -> np.ndarray:
    """Accumulate world rotation for a BVH joint at a specific frame."""
    joint = bvh.joints[j_idx]
    if joint.parent < 0:
        parent_R = np.eye(3, dtype=np.float32)
    else:
        parent_R = _bvh_world_matrix(bvh, joint.parent, frame)
    if joint.is_end or not joint.channels:
        return parent_R
    row = bvh.motion[frame]
    chans, vals = [], []
    for i, ch in enumerate(joint.channels):
        if ch.endswith("rotation"):
            chans.append(ch)
            vals.append(float(row[joint.channel_offset + i]))
    R_local = bvh_mod.euler_matrix(chans, vals) if chans else np.eye(3, dtype=np.float32)
    return (parent_R @ R_local).astype(np.float32)


def _our_world_matrix(bones, pose, bone_idx: int) -> np.ndarray:
    """Accumulate world rotation for our rig. Uses our rotate_xyz convention
    (Rz @ Ry @ Rx), walking the parent chain."""
    from . import mathx
    if bones[bone_idx][1] < 0:
        parent_R = np.eye(3, dtype=np.float32)
    else:
        parent_R = _our_world_matrix(bones, pose, bones[bone_idx][1])
    rx, ry, rz = pose[bone_idx]
    R_local = mathx.rotate_xyz(rx, ry, rz)[:3, :3]
    return (parent_R @ R_local).astype(np.float32)


def _direction_error(bvh: bvh_mod.BVHFile, bones, anim: anim_mod.Animation,
                      frame: int) -> List[Tuple[str, float]]:
    """For each mapped bone, return the angle (radians) between the source's
    world bone-direction and ours at that frame. 0 = perfect retarget."""
    pose, _ = anim.sample(frame * bvh.frame_time)
    errs = []
    for entry in anim.entries:
        bvh_j = bvh.joints[entry.bvh_joint]
        our_bone_name = bones[entry.our_bone][0]
        # world bone-direction at this frame:
        bvh_R_world = _bvh_world_matrix(bvh, entry.bvh_joint, frame)
        # source rest direction is the local OFFSET of its first non-end child
        from .animation import _bvh_rest_direction, _our_rest_direction
        d_src_rest = _bvh_rest_direction(bvh, entry.bvh_joint)
        if anim.mirror:
            d_src_rest = d_src_rest * np.array([-1, 1, 1], dtype=np.float32)
        d_src_world = bvh_R_world @ d_src_rest

        our_R_world = _our_world_matrix(bones, pose, entry.our_bone)
        d_our_rest = _our_rest_direction(bones, entry.our_bone)
        d_our_world = our_R_world @ d_our_rest

        # Angle between world directions
        c = float(np.clip(np.dot(d_src_world / (np.linalg.norm(d_src_world) + 1e-8),
                                  d_our_world / (np.linalg.norm(d_our_world) + 1e-8)),
                          -1.0, 1.0))
        errs.append((our_bone_name, math.acos(c)))
    return errs


# --- grid of renders --------------------------------------------------------

def render_strip(bvh_path: str, out_path: str = "anim_debug.png",
                  n_frames: int = 8, seed: int = 42, gender: str = "male",
                  width: int = 512, height: int = 512) -> None:
    """Render N equally-spaced frames in a horizontal strip."""
    bvh = bvh_mod.parse(bvh_path)
    duration = bvh.frames * bvh.frame_time
    times = np.linspace(0, duration * (1 - 1.0 / n_frames), n_frames)
    tile_paths = []
    for i, t in enumerate(times):
        p = f"/tmp/anim_dbg_{i:02d}.png"
        capture(out_path=p, seed=seed, gender=gender, pose="bvh",
                bvh_path=bvh_path, bvh_time=float(t),
                width=width, height=height)
        tile_paths.append(p)

    # compose horizontal strip
    tiles = [Image.open(p) for p in tile_paths]
    strip = Image.new("RGBA", (width * n_frames, height), (23, 25, 34, 255))
    for i, img in enumerate(tiles):
        strip.paste(img, (i * width, 0))
    strip.save(out_path)
    for p in tile_paths:
        try: os.unlink(p)
        except OSError: pass
    print(f"\nwrote strip: {out_path} ({width * n_frames}x{height})")


# --- main diagnostic --------------------------------------------------------

def diagnose(bvh_path: str, seed: int = 42, gender: str = "male",
             mirror: bool = False) -> None:
    print(f"\n=== Animation debug: {bvh_path} ===\n")
    bvh = bvh_mod.parse(bvh_path)
    bones = skeleton.apply_shape(skeleton.BONES_BASE, skeleton.Shape())
    anim = anim_mod.Animation(bvh, bones, mirror=mirror)

    duration = bvh.frames * bvh.frame_time
    print(f"frames: {bvh.frames}, frame_time: {bvh.frame_time:.4f}s, duration: {duration:.2f}s")
    print(f"mapped bones: {len(anim.entries)} / {len(bones)}")
    for e in anim.entries:
        print(f"  {bvh.joints[e.bvh_joint].name:30s} -> {bones[e.our_bone][0]}")

    # Root trajectory + yaw
    print("\nroot trajectory (world space, metres):")
    roots_x, roots_y, roots_z = [], [], []
    pelvis_yaws = []
    for f in range(0, bvh.frames, max(1, bvh.frames // 8)):
        pose, root = anim.sample(f * bvh.frame_time)
        roots_x.append(root[0]); roots_y.append(root[1]); roots_z.append(root[2])
        iP = skeleton.bone_index(bones, "pelvis")
        pelvis_yaws.append(math.degrees(pose[iP][1]))
        print(f"  t={f*bvh.frame_time:.2f}s  root=({root[0]:+.3f}, {root[1]:+.3f}, {root[2]:+.3f})  "
              f"pelvis_Y={math.degrees(pose[iP][1]):+6.1f} deg")

    print(f"\nroot X range: {min(roots_x):+.3f} to {max(roots_x):+.3f} (delta {max(roots_x)-min(roots_x):.3f})")
    print(f"root Y range: {min(roots_y):+.3f} to {max(roots_y):+.3f} (delta {max(roots_y)-min(roots_y):.3f})  # bob")
    print(f"root Z range: {min(roots_z):+.3f} to {max(roots_z):+.3f} (delta {max(roots_z)-min(roots_z):.3f})  # forward")
    print(f"pelvis yaw range: {min(pelvis_yaws):+.1f} to {max(pelvis_yaws):+.1f} deg "
          f"(delta {max(pelvis_yaws)-min(pelvis_yaws):.1f})")
    if max(pelvis_yaws) - min(pelvis_yaws) > 40:
        print("  ** WARN: pelvis yaw range > 40 deg, whole body may be spinning")

    # Per-bone rotation range
    print("\nper-bone rotation range across clip (deg):")
    bone_names = [b[0] for b in bones]
    rot_mins = {n: [+999, +999, +999] for n in bone_names}
    rot_maxs = {n: [-999, -999, -999] for n in bone_names}
    for f in range(bvh.frames):
        pose, _ = anim.sample(f * bvh.frame_time)
        for i, (rx, ry, rz) in enumerate(pose):
            name = bone_names[i]
            for ax, v in enumerate((math.degrees(rx), math.degrees(ry), math.degrees(rz))):
                rot_mins[name][ax] = min(rot_mins[name][ax], v)
                rot_maxs[name][ax] = max(rot_maxs[name][ax], v)
    for n in bone_names:
        if rot_mins[n][0] == 999:
            continue
        ranges = [rot_maxs[n][a] - rot_mins[n][a] for a in range(3)]
        if max(ranges) < 1:
            continue  # near-static, skip
        print(f"  {n:12s} X: [{rot_mins[n][0]:+6.1f}, {rot_maxs[n][0]:+6.1f}]  "
              f"Y: [{rot_mins[n][1]:+6.1f}, {rot_maxs[n][1]:+6.1f}]  "
              f"Z: [{rot_mins[n][2]:+6.1f}, {rot_maxs[n][2]:+6.1f}]")

    # World-direction retarget error
    print("\nworld-direction retarget error (mean angle between source and ours, across clip):")
    errs_by_bone: dict = {}
    for f in range(0, bvh.frames, max(1, bvh.frames // 8)):
        for name, ang in _direction_error(bvh, bones, anim, f):
            errs_by_bone.setdefault(name, []).append(ang)
    for name, vals in errs_by_bone.items():
        mean = math.degrees(sum(vals) / len(vals))
        mx = math.degrees(max(vals))
        warn = "  ** DRIFT" if mean > 10 else ""
        print(f"  {name:12s} mean={mean:+6.1f} deg  max={mx:+6.1f} deg{warn}")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh_path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gender", default="male",
                    choices=["male", "female", "neutral"])
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--frames", type=int, default=8, help="number of strip frames")
    ap.add_argument("--out", default="anim_debug.png")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--no-render", action="store_true",
                    help="skip rendering, print diagnostics only")
    args = ap.parse_args()

    diagnose(args.bvh_path, seed=args.seed, gender=args.gender, mirror=args.mirror)
    if not args.no_render:
        render_strip(args.bvh_path, out_path=args.out, n_frames=args.frames,
                     seed=args.seed, gender=args.gender,
                     width=args.width, height=args.height)


if __name__ == "__main__":
    main()
