"""Character: shape + appearance + animation state, plus the list of GPU
drawables that make up the renderable body (skin, eyes, hair, clothes...).

A single ``Character`` owns a list of :class:`Drawable` entries. Each is a
GPU mesh paired with a solid base color and a shader mode (0=skin, 1=fabric,
2=hair, 3=eye, 4=shoe). The viewer just iterates the list.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import accessories
from . import garments
from . import mesh as mesh_mod
from . import renderer
from . import skeleton


Color = Tuple[float, float, float]


@dataclass
class Drawable:
    mesh_gpu: renderer.MeshGPU
    color: Color
    mode: int  # 0=skin, 1=fabric, 2=hair, 3=eye, 4=shoe


# --- palettes ----------------------------------------------------------------

_SKIN_TONES: List[Color] = [
    (0.98, 0.84, 0.74),  # very fair
    (0.94, 0.78, 0.66),
    (0.88, 0.72, 0.58),
    (0.76, 0.58, 0.44),
    (0.58, 0.42, 0.30),
    (0.42, 0.28, 0.20),
    (0.28, 0.18, 0.13),  # very dark
]

_HAIR_COLORS: List[Color] = [
    (0.08, 0.06, 0.05),  # black
    (0.20, 0.12, 0.08),  # dark brown
    (0.40, 0.25, 0.15),  # brown
    (0.62, 0.45, 0.25),  # light brown
    (0.85, 0.70, 0.40),  # blonde
    (0.90, 0.85, 0.80),  # grey / platinum
    (0.60, 0.20, 0.10),  # auburn / red
]

_EYE_COLORS: List[Color] = [
    (0.20, 0.35, 0.60),  # blue
    (0.25, 0.55, 0.65),  # cyan
    (0.35, 0.55, 0.30),  # green
    (0.45, 0.30, 0.15),  # hazel
    (0.30, 0.18, 0.08),  # brown
    (0.12, 0.08, 0.05),  # dark brown
]

_EYE_WHITE: Color = (0.94, 0.92, 0.88)


# --- appearance --------------------------------------------------------------

@dataclass
class Appearance:
    # identity
    gender: str = "neutral"

    # skin / face
    skin_color: Color = (0.85, 0.68, 0.55)
    eye_color:  Color = (0.30, 0.18, 0.08)
    seed:       float = 0.0

    # hair
    hair_style:    str   = "short"    # bald / buzz / short / medium / long
    hair_color:    Color = (0.2, 0.12, 0.08)
    eyebrow_color: Color = (0.15, 0.10, 0.07)

    # facial hair (usually male)
    facial_hair_style: str   = "none"  # none / stubble / mustache / goatee / full
    facial_hair_color: Color = (0.15, 0.10, 0.07)

    # lips
    lip_color: Color = (0.55, 0.22, 0.20)
    pupil_color: Color = (0.04, 0.03, 0.03)

    # top
    top_style: str   = "tshirt"       # tshirt / longsleeve / tank / dress
    top_color: Color = (0.30, 0.45, 0.75)
    top_inflate: float = 0.022
    top_length_scale: float = 1.0

    # bottom (ignored if top is a dress)
    bottom_style: str   = "pants"     # pants / shorts / skirt / none
    bottom_color: Color = (0.20, 0.22, 0.28)
    bottom_inflate: float = 0.020
    bottom_length_scale: float = 1.0
    skirt_length_frac: float = 0.9
    skirt_flare: float = 1.5

    # shoes
    shoe_style: str   = "sneakers"    # barefoot / sneakers / boots
    shoe_color: Color = (0.15, 0.15, 0.18)

    # dress-only
    dress_length_frac: float = 1.0
    dress_flare: float = 1.3


def random_appearance(gender: str) -> Appearance:
    seed = random.random()
    skin = random.choice(_SKIN_TONES)
    eye = random.choice(_EYE_COLORS)
    hair = random.choice(_HAIR_COLORS)
    eyebrow = tuple(min(1.0, c * 0.85) for c in hair)

    # --- hair ---
    if gender == "female":
        hair_style = random.choices(
            ["medium", "long", "short", "buzz", "bald"],
            weights=[35, 45, 15, 3, 2],
        )[0]
    else:  # male or neutral
        hair_style = random.choices(
            ["short", "buzz", "medium", "bald", "long"],
            weights=[40, 25, 15, 15, 5],
        )[0]

    # --- facial hair ---
    if gender == "female":
        facial_hair = "none"
    else:
        facial_hair = random.choices(
            ["none", "stubble", "mustache", "goatee", "full"],
            weights=[45, 20, 10, 10, 15],
        )[0]
    facial_hair_color = tuple(min(1.0, c * 0.9) for c in hair)

    # --- top ---
    if gender == "female":
        top_style = random.choices(
            ["tshirt", "longsleeve", "tank", "dress"],
            weights=[30, 20, 20, 30],
        )[0]
    else:
        top_style = random.choices(
            ["tshirt", "longsleeve", "tank", "dress"],
            weights=[50, 35, 15, 0],
        )[0]

    # --- bottom ---
    if top_style == "dress":
        bottom_style = "none"
    elif gender == "female":
        bottom_style = random.choices(
            ["pants", "shorts", "skirt"],
            weights=[35, 15, 50],
        )[0]
    else:
        bottom_style = random.choices(
            ["pants", "shorts"],
            weights=[75, 25],
        )[0]

    # --- shoes ---
    shoe_style = random.choices(
        ["sneakers", "boots", "barefoot"],
        weights=[65, 30, 5],
    )[0]

    # --- colors ---
    top_color    = (random.uniform(0.15, 0.85),
                    random.uniform(0.15, 0.85),
                    random.uniform(0.15, 0.85))
    bottom_color = (random.uniform(0.10, 0.55),
                    random.uniform(0.10, 0.55),
                    random.uniform(0.10, 0.55))
    shoe_color   = (random.uniform(0.05, 0.30),
                    random.uniform(0.05, 0.30),
                    random.uniform(0.05, 0.30))

    # lip color biased to warm / red-brown, slightly darker than skin
    lip = (min(1.0, skin[0] * 0.85 + 0.15),
           min(1.0, skin[1] * 0.55),
           min(1.0, skin[2] * 0.55))

    return Appearance(
        gender=gender,
        skin_color=skin,
        eye_color=eye,
        lip_color=lip,
        seed=seed,
        hair_style=hair_style,
        hair_color=hair,
        eyebrow_color=eyebrow,
        facial_hair_style=facial_hair,
        facial_hair_color=facial_hair_color,
        top_style=top_style,
        top_color=top_color,
        top_inflate=random.uniform(0.008, 0.014),
        top_length_scale=random.uniform(1.00, 1.05),
        bottom_style=bottom_style,
        bottom_color=bottom_color,
        bottom_inflate=random.uniform(0.008, 0.013),
        bottom_length_scale=random.uniform(0.98, 1.04),
        skirt_length_frac=random.uniform(0.5, 1.0),
        skirt_flare=random.uniform(1.15, 1.55),
        dress_length_frac=random.uniform(0.8, 1.2),
        dress_flare=random.uniform(1.05, 1.30),
        shoe_style=shoe_style,
        shoe_color=shoe_color,
    )


# --- character ---------------------------------------------------------------

class Character:
    def __init__(
        self,
        shape: Optional[skeleton.Shape] = None,
        pose=None,
        appearance: Optional[Appearance] = None,
    ):
        self.shape = shape or skeleton.Shape()
        self.bones = skeleton.apply_shape(skeleton.BONES_BASE, self.shape)
        self.pose = pose if pose is not None else skeleton.t_pose(len(self.bones))
        self.root_offset = (0.0, 0.0, 0.0)
        self.appearance = appearance or Appearance(gender=self.shape.gender)
        self.drawables: List[Drawable] = []
        self._lines: Optional[renderer.LineBufferGPU] = None

    # ---- GPU lifecycle ----------------------------------------------------

    def build_gpu(self):
        self._free_gpu()
        app = self.appearance
        out: List[Drawable] = []

        def add(mesh, color, mode):
            if mesh is None or mesh.indices.size == 0:
                return
            out.append(Drawable(renderer.MeshGPU(mesh), color, mode))

        # body skin (body + head compound + bust + glutes handled by build())
        add(mesh_mod.build(self.bones, self.shape), app.skin_color, 0)

        # eyes (whites + iris + pupil), anatomical eyelid occlusion, and lips
        add(mesh_mod.build_eyes(self.bones),    _EYE_WHITE,      3)
        add(mesh_mod.build_iris(self.bones),    app.eye_color,   3)
        add(mesh_mod.build_pupils(self.bones),  app.pupil_color, 3)
        add(mesh_mod.build_eyelids(self.bones), app.skin_color,  0)
        add(mesh_mod.build_lips(self.bones),   app.lip_color,   0)

        # hair + eyebrows
        add(accessories.build_hair(self.bones, app.hair_style), app.hair_color, 2)
        add(accessories.build_eyebrows(self.bones),             app.eyebrow_color, 2)
        add(accessories.build_facial_hair(self.bones, app.facial_hair_style),
            app.facial_hair_color, 2)

        # clothing
        if app.top_style == "dress":
            add(garments.build_dress(self.bones,
                                     length_frac=app.dress_length_frac,
                                     flare=app.dress_flare),
                app.top_color, 1)
        else:
            add(garments.build_top(self.bones, app.top_style,
                                   inflate=app.top_inflate,
                                   length_scale=app.top_length_scale),
                app.top_color, 1)

        if app.bottom_style == "skirt":
            add(garments.build_skirt(self.bones,
                                     length_frac=app.skirt_length_frac,
                                     flare=app.skirt_flare),
                app.bottom_color, 1)
        elif app.bottom_style in ("pants", "shorts"):
            add(garments.build_bottom(self.bones, app.bottom_style,
                                      inflate=app.bottom_inflate,
                                      length_scale=app.bottom_length_scale),
                app.bottom_color, 1)

        add(garments.build_shoes(self.bones, app.shoe_style), app.shoe_color, 4)

        self.drawables = out
        self._lines = renderer.LineBufferGPU(max_points=len(self.bones) * 2)

    def _free_gpu(self):
        for d in self.drawables:
            d.mesh_gpu.delete()
        self.drawables = []
        if self._lines is not None:
            self._lines.delete()
            self._lines = None

    def delete(self):
        self._free_gpu()

    # ---- mutations --------------------------------------------------------

    def set_shape(self, shape: skeleton.Shape):
        self.shape = shape
        self.bones = skeleton.apply_shape(skeleton.BONES_BASE, self.shape)
        if len(self.pose) != len(self.bones):
            self.pose = skeleton.t_pose(len(self.bones))
        # keep gender of appearance synced
        self.appearance.gender = shape.gender
        if self.drawables:
            self.build_gpu()

    def set_appearance(self, appearance: Appearance):
        self.appearance = appearance
        if self.drawables:
            self.build_gpu()

    def set_pose(self, pose, root_offset=(0.0, 0.0, 0.0)):
        self.pose = pose
        self.root_offset = root_offset

    # ---- render helpers ---------------------------------------------------

    def bone_matrices(self) -> np.ndarray:
        return skeleton.compute_bone_matrices(self.bones, self.pose, self.root_offset)

    def skeleton_line_points(self, bone_mats: np.ndarray) -> np.ndarray:
        pts = np.empty((len(self.bones) * 2, 3), dtype=np.float32)
        for i, (_, _, _, tip, _) in enumerate(self.bones):
            M = bone_mats[i]
            p0 = M @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            p1 = M @ np.array([tip[0], tip[1], tip[2], 1.0], dtype=np.float32)
            pts[2 * i] = p0[:3]
            pts[2 * i + 1] = p1[:3]
        return pts.reshape(-1)

    @property
    def lines(self):
        assert self._lines is not None
        return self._lines
