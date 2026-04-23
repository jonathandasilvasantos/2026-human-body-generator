"""Biovision Hierarchy (BVH) parser + playback support.

BVH is the standard text-based mocap format:
  - HIERARCHY defines a skeleton (nested ROOT / JOINT blocks, each with
    OFFSET and CHANNELS).
  - MOTION declares Frames + Frame Time and then one row per frame with
    all channel values in hierarchy order.

Reference: https://en.wikipedia.org/wiki/Biovision_Hierarchy

We keep the parser minimal and tolerant: any combination of Xposition,
Yposition, Zposition, Xrotation, Yrotation, Zrotation per joint is
accepted, and channel order is preserved per joint so playback rebuilds
each frame's rotation using the source file's Euler order.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np


@dataclass
class BVHJoint:
    name: str
    parent: int = -1
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: List[str] = field(default_factory=list)
    channel_offset: int = 0            # index into motion row where this joint's channels begin
    children: List[int] = field(default_factory=list)
    is_end: bool = False


@dataclass
class BVHFile:
    joints: List[BVHJoint]
    total_channels: int
    frames: int
    frame_time: float
    motion: np.ndarray                 # shape (frames, total_channels), float32
    unit_scale: float = 0.01           # BVH files are typically in cm; default to metres

    def joint_by_name(self, name: str) -> Optional[int]:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        return None


def parse(path: str) -> BVHFile:
    """Parse a BVH file into a :class:`BVHFile`."""
    with open(path, "r") as f:
        raw = f.read()
    tokens = raw.replace("{", " { ").replace("}", " } ").split()
    it = iter(tokens)

    def peek_next():
        return next(it)

    def expect(tok):
        got = peek_next()
        if got != tok:
            raise ValueError(f"BVH: expected {tok!r}, got {got!r}")

    joints: List[BVHJoint] = []
    stack: List[int] = []
    total_channels = 0

    def read_joint(kind: str):
        nonlocal total_channels
        # kind is "ROOT" or "JOINT" or "End"
        if kind == "End":
            name = peek_next()       # should be "Site"
            joint_name = joints[stack[-1]].name + "_End" if stack else "End"
            expect("{")
            # expect OFFSET ...
            tok = peek_next()
            assert tok == "OFFSET"
            off = (float(peek_next()), float(peek_next()), float(peek_next()))
            joint = BVHJoint(name=joint_name,
                             parent=stack[-1] if stack else -1,
                             offset=off, channels=[],
                             channel_offset=total_channels, is_end=True)
            j_idx = len(joints)
            joints.append(joint)
            if stack:
                joints[stack[-1]].children.append(j_idx)
            expect("}")
            return
        name = peek_next()
        expect("{")
        joint = BVHJoint(name=name,
                         parent=stack[-1] if stack else -1,
                         channel_offset=total_channels)
        j_idx = len(joints)
        joints.append(joint)
        if stack:
            joints[stack[-1]].children.append(j_idx)
        stack.append(j_idx)

        while True:
            t = peek_next()
            if t == "OFFSET":
                joint.offset = (float(peek_next()), float(peek_next()), float(peek_next()))
            elif t == "CHANNELS":
                n = int(peek_next())
                joint.channels = [peek_next() for _ in range(n)]
                joint.channel_offset = total_channels
                total_channels += n
            elif t in ("JOINT", "ROOT"):
                read_joint(t)
            elif t == "End":
                read_joint("End")
            elif t == "}":
                stack.pop()
                return
            else:
                raise ValueError(f"BVH: unexpected token {t!r}")

    expect("HIERARCHY")
    tok = peek_next()
    assert tok == "ROOT", f"expected ROOT, got {tok}"
    read_joint("ROOT")

    # MOTION section
    expect("MOTION")
    tok = peek_next()
    if tok == "Frames:":
        frames = int(peek_next())
    else:
        # sometimes written as "Frames: N" without spaces
        frames = int(tok.split(":")[1])
    tok = peek_next()
    if tok == "Frame":
        tok2 = peek_next()
        assert tok2 == "Time:", f"expected Time:, got {tok2}"
        frame_time = float(peek_next())
    else:
        frame_time = float(tok.split(":")[1])

    values: List[float] = []
    try:
        while True:
            values.append(float(peek_next()))
    except StopIteration:
        pass

    arr = np.asarray(values, dtype=np.float32)
    assert arr.size == frames * total_channels, \
        f"BVH: expected {frames*total_channels} motion values, got {arr.size}"
    motion = arr.reshape(frames, total_channels)

    return BVHFile(joints=joints, total_channels=total_channels,
                   frames=frames, frame_time=frame_time, motion=motion)


# --- rotation helpers --------------------------------------------------------

_RX = lambda a: np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], dtype=np.float32)
_RY = lambda a: np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=np.float32)
_RZ = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=np.float32)


def euler_matrix(channels: List[str], angles_deg: List[float]) -> np.ndarray:
    """Build a 3x3 rotation matrix from BVH channel values (degrees).

    BVH composes rotations intrinsically in the channel order written, which
    corresponds to left-multiplying each successive axis matrix -- i.e. the
    first channel's axis is applied LAST in the resulting matrix product.
    Mixamo's default is ZYX, so the matrix is R_z @ R_y @ R_x.
    """
    R = np.eye(3, dtype=np.float32)
    for ch, ang in zip(channels, angles_deg):
        a = math.radians(ang)
        if ch == "Xrotation":
            R = R @ _RX(a)
        elif ch == "Yrotation":
            R = R @ _RY(a)
        elif ch == "Zrotation":
            R = R @ _RZ(a)
    return R


def matrix_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into (rx, ry, rz) for our rig's
    ``rotate_xyz(rx, ry, rz) = Rz @ Ry @ Rx`` convention.

    Uses the standard XYZ decomposition with the gimbal-lock safe branch
    on cos(ry).
    """
    # Our rig: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    # Extract:
    #   R[2][0] = -sin(ry)
    #   R[2][1] = cos(ry) * sin(rx)
    #   R[2][2] = cos(ry) * cos(rx)
    #   R[1][0] = cos(ry) * sin(rz)
    #   R[0][0] = cos(ry) * cos(rz)
    sy = -R[2, 0]
    cy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if cy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: ry ~ ±pi/2
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(sy, cy)
        rz = 0.0
    return rx, ry, rz
