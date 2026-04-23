"""Animation playback: load a BVH file and retarget it to our rig.

Retargeting is needed because the source skeleton (e.g. Mixamo) and ours
differ in:
  - joint naming   (handled by a name map)
  - joint count    (Mixamo has Spine + Spine1 + Spine2; we have spine + chest)
  - rest pose      (Mixamo T-pose has arms horizontal; ours A-pose has arms
    hanging) -- we compensate by computing, per mapped bone, the rotation
    that takes OUR rest-direction to THEIRS, and conjugating the source
    rotation by it:
        our_rotation = R_rest_ours_to_theirs * bvh_rotation * R_rest_theirs_to_ours
    so that when the BVH says "identity" (T-pose rest) we still output our
    own rest (A-pose), and when the BVH adds a rotation about some axis in
    their frame, it becomes the same rotation about the equivalent axis in
    ours.

The name map covers Mixamo by default. Arbitrary rigs can be supported by
passing a custom ``name_map``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from . import bvh as bvh_mod
from . import mathx


# Default Mixamo -> our bone name map.
MIXAMO_MAP: Dict[str, str] = {
    "Hips":          "pelvis",
    "Spine":         "spine",
    "Spine1":        "chest",      # collapse Mixamo's 3-part spine to our 2
    "Spine2":        "chest",      # both Spine1 and Spine2 drive chest -> last wins
    "Neck":          "neck",
    "Head":          "head",

    "LeftShoulder":  "clav_L",
    "LeftArm":       "uarm_L",
    "LeftForeArm":   "farm_L",
    "LeftHand":      "hand_L",

    "RightShoulder": "clav_R",
    "RightArm":      "uarm_R",
    "RightForeArm":  "farm_R",
    "RightHand":     "hand_R",

    "LeftUpLeg":     "thigh_L",
    "LeftLeg":       "shin_L",
    "LeftFoot":      "foot_L",

    "RightUpLeg":    "thigh_R",
    "RightLeg":      "shin_R",
    "RightFoot":     "foot_R",
}


def _canonical(name: str) -> str:
    """Strip prefixes like ``mixamorig:`` and lowercase for lookup."""
    if ":" in name:
        name = name.split(":", 1)[1]
    return name


def _make_lookup(name_map: Dict[str, str]) -> Dict[str, str]:
    return {k.lower(): v for k, v in name_map.items()}


# --- rest-direction utilities -----------------------------------------------

def _bvh_rest_direction(bvh: bvh_mod.BVHFile, j_idx: int) -> np.ndarray:
    """Unit vector from joint j to its first non-end child, in BVH rest
    pose. Returned in BVH world units (scale-agnostic after normalise)."""
    joint = bvh.joints[j_idx]
    for c in joint.children:
        child = bvh.joints[c]
        off = np.asarray(child.offset, dtype=np.float32)
        n = np.linalg.norm(off)
        if n > 1e-6:
            return off / n
    # No child with positive offset: fall back to +Y
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _our_rest_direction(bones, bone_idx: int) -> np.ndarray:
    tip = np.asarray(bones[bone_idx][3], dtype=np.float32)
    n = np.linalg.norm(tip)
    if n < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return tip / n


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit vector ``a`` to unit vector ``b``."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    c = float(np.dot(a, b))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        # pick any axis perpendicular to a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-4:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis /= np.linalg.norm(axis)
        # 180 deg rotation around axis
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + 2 * K @ K
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    v = v / s
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + s * K + (1 - c) * (K @ K)).astype(np.float32)


# --- animation clip ---------------------------------------------------------

@dataclass
class RetargetEntry:
    bvh_joint: int
    our_bone: int
    # Rotation that maps our rest direction to the BVH rest direction
    R_rest: np.ndarray     # 3x3
    R_rest_inv: np.ndarray # 3x3


class Animation:
    """A loaded, retargeted animation clip.

    ``mirror`` applies a left-right X-axis mirror to the source before
    retargeting. Set this to ``True`` for Mixamo / most DCC exports whose
    +X is the character's left but whose LeftXxx joints are nonetheless
    the ones intended for the character's left side (sign of the X axis
    vs naming convention is the source of left-right swaps).
    """

    def __init__(self, bvh: bvh_mod.BVHFile, bones, name_map: Optional[Dict[str, str]] = None,
                 unit_scale: Optional[float] = None, mirror: bool = True):
        self.bvh = bvh
        self.bones = bones
        self.mirror = mirror
        if unit_scale is not None:
            bvh.unit_scale = unit_scale
        lookup = _make_lookup(name_map or MIXAMO_MAP)

        # Build retarget entries, one per mapped bone in the source. If two
        # BVH joints map to the same rig bone (e.g. Spine1 and Spine2 -> chest)
        # we keep only the LAST one so its rotation drives chest.
        from .skeleton import bone_index
        entries: Dict[int, RetargetEntry] = {}
        for i, j in enumerate(bvh.joints):
            nm = _canonical(j.name).lower()
            if nm not in lookup:
                continue
            our_name = lookup[nm]
            try:
                our_idx = bone_index(bones, our_name)
            except KeyError:
                continue
            bvh_dir = _bvh_rest_direction(bvh, i)
            our_dir = _our_rest_direction(bones, our_idx)
            if self.mirror:
                bvh_dir = bvh_dir * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
            R_rest = _rotation_between(our_dir, bvh_dir)
            R_rest_inv = R_rest.T
            entries[our_idx] = RetargetEntry(bvh_joint=i, our_bone=our_idx,
                                             R_rest=R_rest, R_rest_inv=R_rest_inv)
        self.entries: List[RetargetEntry] = list(entries.values())
        self.root_bvh_idx = 0   # BVH root is always joints[0]

    @property
    def duration(self) -> float:
        return self.bvh.frames * self.bvh.frame_time

    def _frame_rotation(self, frame: int, j_idx: int) -> np.ndarray:
        joint = self.bvh.joints[j_idx]
        if not joint.channels or joint.is_end:
            return np.eye(3, dtype=np.float32)
        row = self.bvh.motion[frame]
        rot_channels = []
        rot_values = []
        for c_idx, ch in enumerate(joint.channels):
            if ch.endswith("rotation"):
                rot_channels.append(ch)
                rot_values.append(float(row[joint.channel_offset + c_idx]))
        if not rot_channels:
            return np.eye(3, dtype=np.float32)
        return bvh_mod.euler_matrix(rot_channels, rot_values)

    def _frame_translation(self, frame: int, j_idx: int) -> np.ndarray:
        joint = self.bvh.joints[j_idx]
        row = self.bvh.motion[frame]
        out = np.zeros(3, dtype=np.float32)
        axis_map = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
        for c_idx, ch in enumerate(joint.channels):
            if ch in axis_map:
                out[axis_map[ch]] = float(row[joint.channel_offset + c_idx])
        return out

    def sample(self, t: float) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
        """Return ``(pose_rot, root_offset)`` at time ``t`` (seconds), looping.

        ``pose_rot`` is a list of per-bone Euler triples in our rig's convention.
        ``root_offset`` is the root (pelvis) translation in meters.
        """
        n = self.bvh.frames
        ft = max(self.bvh.frame_time, 1e-6)
        frac = (t / ft) % n
        f0 = int(math.floor(frac)) % n
        f1 = (f0 + 1) % n
        a = frac - math.floor(frac)

        pose = [(0.0, 0.0, 0.0)] * len(self.bones)

        for entry in self.entries:
            R0 = self._frame_rotation(f0, entry.bvh_joint)
            R1 = self._frame_rotation(f1, entry.bvh_joint)
            # Linear blend of matrices then re-orthonormalise via SVD is
            # expensive; for the sub-frame blend we just interpolate by
            # mixing angles afterwards. Here we use a simple matrix lerp
            # and re-orthonormalise with a cheap SVD.
            R = (1.0 - a) * R0 + a * R1
            U, _, Vt = np.linalg.svd(R)
            R = (U @ Vt).astype(np.float32)
            # Retarget: R_ours = R_rest_inv * R_bvh * R_rest
            if self.mirror:
                # Mirror about X: reflect the BVH rotation so "Left" in the
                # source drives the rig's left side and rotations preserve
                # their visual direction. Reflection matrix M = diag(-1,1,1)
                # is its own inverse, so R_mirrored = M @ R @ M.
                M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
                R = M @ R @ M
            R_ours = entry.R_rest_inv @ R @ entry.R_rest
            pose[entry.our_bone] = bvh_mod.matrix_to_euler_xyz(R_ours)

        # Root translation from BVH root joint (scaled to metres, minus the
        # initial rest-pose root position so playback starts at origin).
        t0 = self._frame_translation(f0, self.root_bvh_idx) * self.bvh.unit_scale
        t1 = self._frame_translation(f1, self.root_bvh_idx) * self.bvh.unit_scale
        root_t = (1.0 - a) * t0 + a * t1
        # Subtract the first frame's root so the motion is origin-relative.
        t_zero = self._frame_translation(0, self.root_bvh_idx) * self.bvh.unit_scale
        root_t = root_t - t_zero
        if self.mirror:
            root_t = root_t * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        root_offset = tuple(root_t.tolist())
        return pose, root_offset


def load(path: str, bones, name_map: Optional[Dict[str, str]] = None,
         unit_scale: float = 0.01) -> Animation:
    """Convenience loader: parse + build retargeted Animation."""
    bvh = bvh_mod.parse(path)
    return Animation(bvh, bones, name_map=name_map, unit_scale=unit_scale)
