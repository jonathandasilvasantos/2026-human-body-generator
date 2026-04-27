"""Simplified Mixamo skeleton template.

22 bones, names match Mixamo standard so animations and community rigs
interop. Each entry: (bone_name, parent_name_or_None, head_yfrac,
horizontal_offset, axis). Heads are placed at fractions of the figure's
height (bbox Y range, 0..1). The armature is built procedurally in
cleanup.py, then auto-weighted to the mesh.

Coordinate convention: figure is upright on +Y, faces +Z, T-pose arms along X.
Horizontal offsets are fractions of bbox width.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    head_y: float        # fraction of figure height (0..1)
    head_x: float = 0.0  # fraction of figure half-width
    tail_y: float = 0.0
    tail_x: float = 0.0


# Y fractions roughly match adult human proportions (Vitruvian-ish).
BONES: tuple[Bone, ...] = (
    Bone("Hips",          None,            0.535, 0.000, 0.580, 0.000),
    Bone("Spine",         "Hips",          0.580, 0.000, 0.640, 0.000),
    Bone("Spine1",        "Spine",         0.640, 0.000, 0.715, 0.000),
    Bone("Spine2",        "Spine1",        0.715, 0.000, 0.800, 0.000),
    Bone("Neck",          "Spine2",        0.800, 0.000, 0.865, 0.000),
    Bone("Head",          "Neck",          0.865, 0.000, 0.985, 0.000),

    Bone("LeftShoulder",  "Spine2",        0.800,  0.080, 0.800,  0.180),
    Bone("LeftArm",       "LeftShoulder",  0.800,  0.180, 0.800,  0.450),
    Bone("LeftForeArm",   "LeftArm",       0.800,  0.450, 0.800,  0.700),
    Bone("LeftHand",      "LeftForeArm",   0.800,  0.700, 0.800,  0.820),

    Bone("RightShoulder", "Spine2",        0.800, -0.080, 0.800, -0.180),
    Bone("RightArm",      "RightShoulder", 0.800, -0.180, 0.800, -0.450),
    Bone("RightForeArm",  "RightArm",      0.800, -0.450, 0.800, -0.700),
    Bone("RightHand",     "RightForeArm",  0.800, -0.700, 0.800, -0.820),

    Bone("LeftUpLeg",     "Hips",          0.520,  0.090, 0.290,  0.110),
    Bone("LeftLeg",       "LeftUpLeg",     0.290,  0.110, 0.060,  0.115),
    Bone("LeftFoot",      "LeftLeg",       0.060,  0.115, 0.020,  0.140),
    Bone("LeftToeBase",   "LeftFoot",      0.020,  0.140, 0.000,  0.165),

    Bone("RightUpLeg",    "Hips",          0.520, -0.090, 0.290, -0.110),
    Bone("RightLeg",      "RightUpLeg",    0.290, -0.110, 0.060, -0.115),
    Bone("RightFoot",     "RightLeg",      0.060, -0.115, 0.020, -0.140),
    Bone("RightToeBase",  "RightFoot",     0.020, -0.140, 0.000, -0.165),
)
