"""Skeleton: bone hierarchy, parametric shape morph, forward kinematics,
pose sampling.

A bone is a tuple ``(name, parent_index, head_offset, tip_offset, radius)``:
- ``head_offset`` is the joint position relative to the parent's head, in
  the parent's local frame at rest (T-pose).
- ``tip_offset`` is the bone's tip position in its own local frame. It
  defines both the bone's length and its resting direction (for the capsule
  mesh), and is used to render skeleton lines.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import mathx

TAU = 2.0 * math.pi


Bone = Tuple[str, int, Tuple[float, float, float], Tuple[float, float, float], float]


# T-pose baseline offsets (meters), roughly an adult.
# Baseline bone sizes tuned to anthropometric proportions (classical canon +
# Farkas measurements). One "head unit" == 0.22 m (head bone length). Targets:
#   - total height ~7.5 heads
#   - shoulder breadth ~1.7 heads (slightly less than classic to sit between
#     male and female; per-gender scaling in shape params biases further)
#   - upper arm ~1.4 heads, forearm ~1.2 heads, hand ~0.75 heads
#   - thigh ~2.0 heads, shin ~1.9 heads, foot length ~0.85 heads
BONES_BASE: List[Bone] = [
    ("pelvis",   -1, (0.0, 0.0, 0.0),    (0.0, 0.10, 0.0),   0.14),
    ("spine",     0, (0.0, 0.08, 0.0),   (0.0, 0.18, 0.0),   0.12),
    ("chest",     1, (0.0, 0.18, 0.0),   (0.0, 0.20, 0.0),   0.145),
    ("neck",      2, (0.0, 0.19, 0.0),   (0.0, 0.06, 0.0),   0.042),
    ("head",      3, (0.0, 0.06, 0.0),   (0.0, 0.215, 0.0),  0.102),

    ("clav_L",    2, (0.035, 0.15, 0.0), (0.125, 0.0, 0.0),  0.042),
    ("uarm_L",    5, (0.125, 0.0, 0.0),  (0.0, -0.29, 0.0),  0.050),
    ("farm_L",    6, (0.0, -0.29, 0.0),  (0.0, -0.255, 0.0), 0.043),
    ("hand_L",    7, (0.0, -0.255, 0.0), (0.0, -0.155, 0.0), 0.036),

    ("clav_R",    2, (-0.035, 0.15, 0.0),(-0.125, 0.0, 0.0), 0.042),
    ("uarm_R",    9, (-0.125, 0.0, 0.0), (0.0, -0.29, 0.0),  0.050),
    ("farm_R",   10, (0.0, -0.29, 0.0),  (0.0, -0.255, 0.0), 0.043),
    ("hand_R",   11, (0.0, -0.255, 0.0), (0.0, -0.155, 0.0), 0.036),

    ("thigh_L",   0, (0.085, -0.02, 0.0),(0.0, -0.44, 0.0),  0.088),
    ("shin_L",   13, (0.0, -0.44, 0.0),  (0.0, -0.42, 0.0),  0.064),
    ("foot_L",   14, (0.0, -0.42, 0.0),  (0.0, -0.05, 0.18), 0.047),

    ("thigh_R",   0, (-0.085, -0.02, 0.0),(0.0, -0.44, 0.0), 0.088),
    ("shin_R",   16, (0.0, -0.44, 0.0),  (0.0, -0.42, 0.0),  0.064),
    ("foot_R",   17, (0.0, -0.42, 0.0),  (0.0, -0.05, 0.18), 0.047),
]


@dataclass
class Shape:
    """Parametric shape knobs. 1.0 = baseline for most fields.

    ``gender`` is a categorical that biases every other knob when sampled via
    :func:`random_shape`. SMPL ships separate PCA bases per gender (male /
    female / neutral); here we approximate that with gender-conditioned
    ranges on the same scalar knobs, plus bust geometry for female.
    """
    gender: str = "neutral"     # "male" | "female" | "neutral"
    height: float = 1.0
    bulk: float = 1.0
    limb_len: float = 1.0
    shoulder_w: float = 1.0
    hip_w: float = 1.0
    bust: float = 0.0           # female only; 0 => no bust mesh
    head_size: float = 1.0
    leg_len: float = 1.0
    torso_len: float = 1.0


# Per-bone shape rules: which shape knobs scale the head/tip offsets and
# radius for each named bone. Keeps ``apply_shape`` declarative.
_SHAPE_RULES = {
    "spine":   dict(head=("1", "torso_len", "1"), tip=("1", "torso_len", "1"), r="bulk"),
    "chest":   dict(head=("1", "torso_len", "1"), tip=("1", "torso_len", "1"), r="bulk"),
    "neck":    dict(head=("1", "torso_len", "1"), tip=("1", "1", "1"),         r="bulk*0.9"),
    "head":    dict(head=("1", "1", "1"),         tip=("head_size",) * 3,      r="head_size"),

    "clav_L":  dict(head=("shoulder_w", "1", "1"), tip=("shoulder_w", "1", "1"), r="bulk"),
    "clav_R":  dict(head=("shoulder_w", "1", "1"), tip=("shoulder_w", "1", "1"), r="bulk"),
    "uarm_L":  dict(head=("1", "1", "1"),          tip=("1", "limb_len", "1"),   r="bulk"),
    "uarm_R":  dict(head=("1", "1", "1"),          tip=("1", "limb_len", "1"),   r="bulk"),
    "farm_L":  dict(head=("1", "limb_len", "1"),   tip=("1", "limb_len", "1"),   r="bulk"),
    "farm_R":  dict(head=("1", "limb_len", "1"),   tip=("1", "limb_len", "1"),   r="bulk"),
    "hand_L":  dict(head=("1", "limb_len", "1"),   tip=("1", "1", "1"),          r="bulk"),
    "hand_R":  dict(head=("1", "limb_len", "1"),   tip=("1", "1", "1"),          r="bulk"),

    "thigh_L": dict(head=("hip_w", "1", "1"),  tip=("1", "leg_len", "1"), r="bulk"),
    "thigh_R": dict(head=("hip_w", "1", "1"),  tip=("1", "leg_len", "1"), r="bulk"),
    "pelvis":  dict(head=("1", "1", "1"),      tip=("1", "1", "1"),       r="bulk*((hip_w+1.0)*0.5)"),
    "shin_L":  dict(head=("1", "leg_len", "1"),tip=("1", "leg_len", "1"), r="bulk"),
    "shin_R":  dict(head=("1", "leg_len", "1"),tip=("1", "leg_len", "1"), r="bulk"),
    "foot_L":  dict(head=("1", "leg_len", "1"),tip=("1", "1", "1"),       r="bulk"),
    "foot_R":  dict(head=("1", "leg_len", "1"),tip=("1", "1", "1"),       r="bulk"),
}


def _eval(expr: str, shape: Shape) -> float:
    if expr == "1":
        return 1.0
    # safe eval: only shape fields and simple arithmetic
    return float(eval(expr, {"__builtins__": {}}, vars(shape)))


def apply_shape(bones_base: List[Bone], shape: Shape) -> List[Bone]:
    """Return a shape-morphed copy of the bone list."""
    out = []
    for (name, parent, head, tip, r) in bones_base:
        rule = _SHAPE_RULES.get(name)
        if rule is None:
            head_s = (1.0, 1.0, 1.0)
            tip_s = (1.0, 1.0, 1.0)
            r_s = _eval("bulk", shape)
        else:
            head_s = tuple(_eval(e, shape) for e in rule["head"])
            tip_s = tuple(_eval(e, shape) for e in rule["tip"])
            r_s = _eval(rule["r"], shape)

        # global height scales everything vertically
        head_s = (head_s[0], head_s[1] * shape.height, head_s[2])
        tip_s = (tip_s[0], tip_s[1] * shape.height, tip_s[2])

        out.append((
            name, parent,
            tuple(v * head_s[i] for i, v in enumerate(head)),
            tuple(v * tip_s[i] for i, v in enumerate(tip)),
            r * r_s,
        ))
    return out


def compute_bone_matrices(
    bones: List[Bone], pose_rot, root_offset=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """Forward kinematics. ``pose_rot[i]`` is (rx, ry, rz) in radians.

    ``root_offset`` translates the root bone in world space (useful for a
    walking bob so the whole character rises/falls with the stride)."""
    world = [None] * len(bones)
    for i, (_, parent, head, _, _) in enumerate(bones):
        local = mathx.translate(np.asarray(head, dtype=np.float32)) @ mathx.rotate_xyz(*pose_rot[i])
        if parent < 0:
            world[i] = mathx.translate(np.asarray(root_offset, dtype=np.float32)) @ local
        else:
            world[i] = world[parent] @ local
    return np.stack(world, axis=0).astype(np.float32)


def bone_index(bones: List[Bone], name: str) -> int:
    for i, b in enumerate(bones):
        if b[0] == name:
            return i
    raise KeyError(name)


# --- pose + shape sampling --------------------------------------------------

def t_pose(n: int):
    return [(0.0, 0.0, 0.0)] * n


# Per-bone biomechanical joint limits (radians). Ranges are (X, Y, Z) = (flex/
# extension around the bone's local X, internal/external rotation around Y,
# abduction/adduction around Z). Values are approximations of anatomical ranges
# reported in human biomechanics literature.
#
# References:
# - Akhter & Black, "Pose-Conditioned Joint Angle Limits for 3D Human Pose
#   Reconstruction" (CVPR 2015)
# - Jiang et al., "Data-Driven Approach to Simulating Realistic Human Joint
#   Constraints" (arXiv 1709.08685)
# - Kapandji, "The Physiology of the Joints"
#
# Sign convention in this rig:
#   X > 0  -> bone tip rotates in -Z (backward) about the joint
#   X < 0  -> bone tip rotates in +Z (forward) about the joint
#   Therefore elbow flexion (bicep curl) is negative X on the forearm,
#   knee flexion (foot to buttock) is negative X on the shin.
_JOINT_LIMITS = {
    # bone       X (flex/ext)       Y (rotation)       Z (abduction)
    "pelvis":  ((-0.10, 0.10),    (-0.20, 0.20),    (-0.10, 0.10)),
    "spine":   ((-0.20, 0.30),    (-0.30, 0.30),    (-0.25, 0.25)),
    "chest":   ((-0.20, 0.30),    (-0.30, 0.30),    (-0.25, 0.25)),
    "neck":    ((-0.50, 0.50),    (-0.80, 0.80),    (-0.45, 0.45)),
    "head":    ((-0.30, 0.30),    (-0.50, 0.50),    (-0.30, 0.30)),

    "clav_L":  ((-0.15, 0.15),    (-0.10, 0.10),    (-0.20, 0.20)),
    "clav_R":  ((-0.15, 0.15),    (-0.10, 0.10),    (-0.20, 0.20)),
    # Upper-arm ranges are asymmetric because abduction for L arm goes to -Z
    # in its frame and vice versa for R.
    "uarm_L":  ((-1.80, 0.80),    (-1.20, 1.20),    (-1.50, 0.40)),
    "uarm_R":  ((-1.80, 0.80),    (-1.20, 1.20),    (-0.40, 1.50)),
    # Elbow: flexion only, no hyperextension. ~0 to -150 degrees.
    "farm_L":  ((-2.60,  0.00),   (-0.50, 0.50),    (-0.15, 0.15)),
    "farm_R":  ((-2.60,  0.00),   (-0.50, 0.50),    (-0.15, 0.15)),
    "hand_L":  ((-1.20, 1.20),    (-0.50, 0.50),    (-0.60, 0.60)),
    "hand_R":  ((-1.20, 1.20),    (-0.50, 0.50),    (-0.60, 0.60)),

    # Hip: forward flex ~90, backward ext ~45, lateral abduction moderate.
    "thigh_L": ((-1.30, 0.80),    (-0.40, 0.40),    (-0.60, 0.40)),
    "thigh_R": ((-1.30, 0.80),    (-0.40, 0.40),    (-0.40, 0.60)),
    # Knee: flexion only. In this rig, positive X rotation of the shin sends
    # its tip to -Z (toward the buttocks), which is the flexion direction.
    # Hyperextension (negative X) is disallowed.
    "shin_L":  (( 0.00, 2.40),    (-0.10, 0.10),    (-0.10, 0.10)),
    "shin_R":  (( 0.00, 2.40),    (-0.10, 0.10),    (-0.10, 0.10)),
    "foot_L":  ((-0.70, 0.50),    (-0.30, 0.30),    (-0.30, 0.30)),
    "foot_R":  ((-0.70, 0.50),    (-0.30, 0.30),    (-0.30, 0.30)),
}


def joint_limits(name: str):
    return _JOINT_LIMITS.get(name, ((-0.05, 0.05),) * 3)


def random_pose(bones: List[Bone], intensity: float = 1.0):
    """Sample a pose uniformly inside each bone's biomechanical range,
    scaled by ``intensity`` (1.0 = full range, 0.0 = rest pose)."""
    pose = []
    for (name, *_rest) in bones:
        axes = []
        for (lo, hi) in joint_limits(name):
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * intensity
            axes.append(random.uniform(mid - half, mid + half))
        pose.append(tuple(axes))
    return pose


def clamp_pose(pose, bones: List[Bone]):
    """Project any pose onto the per-bone limit box."""
    out = []
    for i, (name, *_rest) in enumerate(bones):
        limits = joint_limits(name)
        p = pose[i] if i < len(pose) else (0.0, 0.0, 0.0)
        out.append(tuple(max(lo, min(hi, a)) for (lo, hi), a in zip(limits, p)))
    return out


def _solve_leg_ik_2d(dy: float, dz: float, L1: float, L2: float):
    """Analytical two-bone IK in the sagittal plane.

    Given an ankle target ``(dy, dz)`` expressed relative to the hip (with
    ``dy`` negative = below the hip, ``dz`` positive = forward), return the
    thigh and shin X-rotations that place the ankle at that point. Knee
    flexes forward (knee points to +Z, shin X rotation is positive in this
    rig's convention).

    Standard cosine-rule solution. References: any graphics/robotics IK
    text, e.g. Tolani, Goswami & Badler, "Real-Time Inverse Kinematics
    Techniques for Anthropomorphic Limbs" (Graphical Models, 2000).
    """
    d = math.sqrt(dy * dy + dz * dz)
    max_reach = (L1 + L2) * 0.985
    min_reach = abs(L1 - L2) + 1e-3
    if d > max_reach:
        s = max_reach / max(d, 1e-6); dy *= s; dz *= s; d = max_reach
    elif d < min_reach:
        s = min_reach / max(d, 1e-6); dy *= s; dz *= s; d = min_reach

    # Interior knee angle (law of cosines).
    cos_g = (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)
    cos_g = max(-1.0, min(1.0, cos_g))
    gamma = math.acos(cos_g)
    shin_x = math.pi - gamma          # positive: knee flex in our convention

    # Angle at the hip between the ankle direction and the thigh direction.
    cos_a = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
    cos_a = max(-1.0, min(1.0, cos_a))
    alpha = math.acos(cos_a)

    # Angle of ankle direction from -Y (positive = toward +Z = forward).
    theta = math.atan2(dz, -dy)

    # Thigh rotation: negative X tilts the thigh tip toward +Z (forward).
    # The knee needs to sit in front of the hip-to-ankle line so that the
    # shin can bend forward (positive X) back onto the ankle. Hence the
    # thigh must be rotated by -(theta + alpha) rather than -(theta - alpha).
    thigh_x = -(theta + alpha)
    return thigh_x, shin_x


def walk_pose(bones: List[Bone], t: float, speed: float = 2.2, stride: float = 1.0):
    """Sinusoidal biped walk cycle, character facing +Z (forward).

    Sign convention for this rig:
      thigh X > 0  -> leg swings backward (tip toward -Z)
      thigh X < 0  -> leg swings forward  (tip toward +Z)
      shin  X < 0  -> knee flexes, foot tucks toward buttock
      forearm X<0  -> elbow flexes, hand swings forward (bicep curl)

    We therefore use negative thigh X for the forward half of the cycle.
    Knee flexion is modelled as a squared-cosine bump peaking just after
    lift-off, straightening at heel-strike -- same shape commonly used in
    keyframe walk-cycles and Johansen's semi-procedural locomotion model.

    References:
    - "A Learning Tool to Assist in Animation of Bipedal Walk Cycles" (2009)
    - Johansen, "Automated Semi-Procedural Animation for Character
      Locomotion" (MSc thesis, 2009)
    """
    """Physics-aware biped walk cycle driven by foot IK.

    Instead of sinusoidal joint angles, we define physical foot trajectories
    in the pelvis-local frame:
      - stance (50% of cycle): foot fixed at ground height, moving backward
        relative to the body as the pelvis moves forward;
      - swing (50% of cycle): foot arcs from behind to ahead of the hip,
        following a sin-shaped lift profile.

    For each leg we then analytically solve a two-bone IK (hip + knee) so
    the ankle hits its target. The body translates forward at the speed
    implied by ``stride`` and ``speed`` -- so stride length, cadence, and
    body velocity are all mutually consistent, as in real gait.

    References:
      - SIMBICON (Yin, Loken, van de Panne, SIGGRAPH 2007)
      - Inverted-pendulum models of gait (Kuo 2007)
      - Johansen, Automated Semi-Procedural Animation for Locomotion (2009)
      - Tolani, Goswami, Badler -- analytical IK for anthropomorphic limbs
    """
    phi = t * speed
    phase_L = phi % (2.0 * math.pi)
    phase_R = (phi + math.pi) % (2.0 * math.pi)
    idx = {b[0]: i for i, b in enumerate(bones)}
    pose = [(0.0, 0.0, 0.0)] * len(bones)

    # Leg geometry from the bone list (so shape scaling transfers correctly).
    L1 = float(np.linalg.norm(np.asarray(bones[idx["thigh_L"]][3])))
    L2 = float(np.linalg.norm(np.asarray(bones[idx["shin_L"]][3])))
    leg_len = L1 + L2
    # Step length scales with leg length so tall/short characters walk in
    # proportion. 0.42 * leg_len gives a ~36 cm step for a standard leg --
    # slightly shorter than full-reach so the IK doesn't clamp at heel-strike
    # and the leg retains a natural ~25-30 degree mid-stance knee flex.
    step = 0.42 * leg_len * stride

    # Hip joint positions in pelvis-local frame (thigh head offsets).
    hip_L = np.asarray(bones[idx["thigh_L"]][2], dtype=np.float32)
    hip_R = np.asarray(bones[idx["thigh_R"]][2], dtype=np.float32)

    # Vertical bob: body vaults over the stance leg (inverted-pendulum
    # model). Two bobs per stride, synchronised with the mid-stance
    # passing phase.
    bob_amp = 0.03 * stride
    bob = bob_amp * (0.5 - 0.5 * math.cos(phi * 2.0))

    # Ground level in pelvis-local space, compensated for bob so the foot
    # stays pinned at world y=0 during stance.
    # Slightly less than full extension so a natural knee flex remains at
    # mid-stance (~25-30 degrees), matching real gait studies.
    ground_y = -leg_len * 0.97 - bob

    def foot_target(phase: float):
        """Return ankle target in pelvis-local (x=0, y, z)."""
        if phase < math.pi:
            # SWING: foot arcs from behind the hip to ahead of it.
            progress = phase / math.pi
            z = -0.5 * step + step * progress
            lift = 0.10 * leg_len * stride * math.sin(phase)
            y = ground_y + lift
        else:
            # STANCE: foot planted on the ground, pelvis passes over it.
            progress = (phase - math.pi) / math.pi   # 0..1
            z = 0.5 * step - step * progress
            y = ground_y
        return 0.0, y, z

    for (side_name, phase, hip) in (("L", phase_L, hip_L), ("R", phase_R, hip_R)):
        _, fy, fz = foot_target(phase)
        dy = fy - float(hip[1])
        dz = fz - float(hip[2])
        thigh_x, shin_x = _solve_leg_ik_2d(dy, dz, L1, L2)
        pose[idx[f"thigh_{side_name}"]] = (thigh_x, 0.0, 0.0)
        pose[idx[f"shin_{side_name}"]]  = (shin_x,  0.0, 0.0)
        # Foot roll: heel-strike with toes slightly up, toe-off with toes
        # down. During swing we relax toward neutral.
        if phase < math.pi:
            foot_x = 0.25 * math.sin(phase)           # toe up mid-swing
        else:
            t_st = (phase - math.pi) / math.pi
            foot_x = -0.30 + 0.60 * t_st              # heel down -> toe down
        pose[idx[f"foot_{side_name}"]] = (foot_x, 0.0, 0.0)

    # Arm swing: anti-phase with same-side leg. L arm swings BACK while L
    # leg swings FORWARD. Positive uarm X = arm tip to -Z = backward.
    arm_amp = 0.55
    pose[idx["uarm_L"]] = (+arm_amp * math.sin(phi),            0.0, +0.15)
    pose[idx["uarm_R"]] = (+arm_amp * math.sin(phi + math.pi),  0.0, -0.15)
    pose[idx["farm_L"]] = (-0.45 - 0.15 * math.sin(phi),           0.0, 0.0)
    pose[idx["farm_R"]] = (-0.45 - 0.15 * math.sin(phi + math.pi), 0.0, 0.0)

    # Torso counter-rotation (pelvis and shoulders counter-rotate in gait).
    pose[idx["spine"]] = (0.0,  0.05 * math.sin(phi),  0.03 * math.sin(phi))
    pose[idx["chest"]] = (0.0, -0.08 * math.sin(phi), -0.02 * math.sin(phi))
    pose[idx["neck"]]  = (0.0,  0.04 * math.sin(phi),  0.0)
    pose[idx["head"]]  = (0.02 * math.sin(phi * 2), 0.0, 0.015 * math.sin(phi))

    # Forward translation: velocity that matches stride length * cadence.
    # One full cycle (t -> t + 2*pi/speed) advances by 2*step (= one stride).
    v = step * speed / math.pi
    forward = v * t

    return pose, (0.0, bob, forward)


def random_shape(gender: str | None = None) -> Shape:
    """Sample a Shape with gender-conditioned ranges (SMPL-style dimorphism).

    Male:   broader shoulders, narrower hips, higher mean bulk, taller mean.
    Female: narrower shoulders, broader hips, lower mean bulk, optional bust.
    """
    if gender is None:
        gender = random.choice(["male", "female"])

    if gender == "male":
        shoulder_w = random.uniform(0.98, 1.14)
        hip_w      = random.uniform(0.88, 1.02)
        bust       = 0.0
        height     = random.uniform(0.98, 1.12)
        bulk       = random.uniform(0.94, 1.20)
        torso_len  = random.uniform(0.96, 1.08)
        leg_len    = random.uniform(0.96, 1.10)
        head_size  = random.uniform(1.00, 1.10)
    elif gender == "female":
        shoulder_w = random.uniform(0.84, 0.98)
        hip_w      = random.uniform(0.98, 1.16)
        bust       = random.uniform(0.30, 0.95)
        height     = random.uniform(0.90, 1.04)
        bulk       = random.uniform(0.80, 1.02)
        torso_len  = random.uniform(0.94, 1.06)
        leg_len    = random.uniform(0.94, 1.10)
        head_size  = random.uniform(0.98, 1.08)
    else:
        shoulder_w = random.uniform(0.90, 1.08)
        hip_w      = random.uniform(0.92, 1.10)
        bust       = 0.0
        height     = random.uniform(0.94, 1.08)
        bulk       = random.uniform(0.86, 1.10)
        torso_len  = random.uniform(0.94, 1.08)
        leg_len    = random.uniform(0.94, 1.10)
        head_size  = random.uniform(0.98, 1.10)

    return Shape(
        gender=gender,
        height=height,
        bulk=bulk,
        limb_len=random.uniform(0.95, 1.08),
        shoulder_w=shoulder_w,
        hip_w=hip_w,
        bust=bust,
        head_size=head_size,
        leg_len=leg_len,
        torso_len=torso_len,
    )
