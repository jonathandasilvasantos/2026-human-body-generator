"""Character: shape + appearance + animation state, plus the list of GPU
drawables that make up the renderable body (skin, eyes, hair, clothes...).

A single ``Character`` owns a list of :class:`Drawable` entries. Each is a
GPU mesh paired with a solid base color and a shader mode (0=skin, 1=fabric,
2=hair, 3=eye, 4=shoe). The viewer just iterates the list.
"""

import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from . import accessories
from . import face_anim
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

_EYE_WHITE:    Color = (0.93, 0.91, 0.87)  # warm off-white sclera
_EYE_LIMBUS:   Color = (0.06, 0.05, 0.05)  # dark corneoscleral ring
_EYE_CATCH:    Color = (1.00, 0.99, 0.96)  # near-white specular catchlight
_MOUTH_DARK:   Color = (0.08, 0.035, 0.035)


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

    # face expression + age (ported from human-chat realism cycles).
    # expression: "neutral" | "smile" | "frown" | "squint" | "surprised"
    # age_group:  "young"   | "adult" | "elder"   (affects crease rendering)
    expression: str = "neutral"
    age_group: str = "adult"
    crease_color: Color = (0.30, 0.18, 0.13)

    # ARKit-52 blendshape overrides (FACS-aligned). When non-empty, each
    # entry is layered on top of the named ``expression`` preset, so you
    # can do ``expression="smile", blendshapes={"eyeBlinkLeft": 1.0}``
    # to get a smiling left-eye wink. See ``human/face_anim.py``.
    blendshapes: Optional[dict] = None

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

    # clothing prints (all-over pattern) and chest stamps (localised decal).
    # Driven explicitly from Python rather than from a shader hash so that
    # characters visibly cycle through the available styles. Style 0 means
    # "no effect"; strength 0 likewise disables.
    print_style: int = 0          # 0=none 1=stripes 2=dots 3=plaid 4=noise
    print_strength: float = 0.0
    stamp_style: int = 0          # 0=none 1=ring 2=diamond 3=cross 4=star
    stamp_strength: float = 0.0


def reroll_clothes(app: "Appearance") -> "Appearance":
    """Return a copy of ``app`` with only clothing (top/bottom/shoes + prints)
    re-randomized. Skin, hair, face, expression and age are preserved so the
    character remains visually the same person in new outfits."""
    gender = app.gender
    if gender == "female":
        top_style = random.choices(
            ["tshirt", "longsleeve", "tank", "dress"],
            weights=[30, 20, 20, 30])[0]
    else:
        top_style = random.choices(
            ["tshirt", "longsleeve", "tank", "dress"],
            weights=[50, 35, 15, 0])[0]
    if top_style == "dress":
        bottom_style = "none"
    elif gender == "female":
        bottom_style = random.choices(
            ["pants", "shorts", "skirt"], weights=[35, 15, 50])[0]
    else:
        bottom_style = random.choices(
            ["pants", "shorts"], weights=[75, 25])[0]
    shoe_style = random.choices(
        ["sneakers", "boots", "barefoot"], weights=[65, 30, 5])[0]
    top_color    = tuple(random.uniform(0.15, 0.85) for _ in range(3))
    bottom_color = tuple(random.uniform(0.10, 0.55) for _ in range(3))
    shoe_color   = tuple(random.uniform(0.05, 0.30) for _ in range(3))
    if random.random() < 0.55:
        print_style = random.randint(1, 4)
        print_strength = random.uniform(0.40, 0.75)
    else:
        print_style, print_strength = 0, 0.0
    if random.random() < 0.40:
        stamp_style = random.randint(1, 4)
        stamp_strength = random.uniform(0.65, 0.95)
    else:
        stamp_style, stamp_strength = 0, 0.0
    return replace(
        app,
        top_style=top_style, top_color=top_color,
        top_inflate=random.uniform(0.014, 0.020),
        top_length_scale=random.uniform(1.00, 1.05),
        bottom_style=bottom_style, bottom_color=bottom_color,
        bottom_inflate=random.uniform(0.008, 0.013),
        bottom_length_scale=random.uniform(0.98, 1.04),
        skirt_length_frac=random.uniform(0.5, 1.0),
        skirt_flare=random.uniform(1.15, 1.55),
        dress_length_frac=random.uniform(0.8, 1.2),
        dress_flare=random.uniform(1.05, 1.30),
        shoe_style=shoe_style, shoe_color=shoe_color,
        print_style=print_style, print_strength=print_strength,
        stamp_style=stamp_style, stamp_strength=stamp_strength,
    )


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

    # --- prints / stamps ---
    # Probabilities tuned so variety is visible across 4-5 regenerations.
    # 45% plain, 55% patterned; of those, uniform over the 4 styles.
    if random.random() < 0.55:
        print_style = random.randint(1, 4)
        print_strength = random.uniform(0.40, 0.75)
    else:
        print_style = 0
        print_strength = 0.0

    # Chest stamps are more eye-catching, so rarer: 40%.
    if random.random() < 0.40:
        stamp_style = random.randint(1, 4)
        stamp_strength = random.uniform(0.65, 0.95)
    else:
        stamp_style = 0
        stamp_strength = 0.0

    # Expression: weighted toward neutral so crowds read calm by default.
    expression = random.choices(
        ["neutral", "smile", "frown", "squint", "surprised"],
        weights=[55, 25, 6, 8, 6],
    )[0]

    # Age group: mostly adult, some young/elder for visible variety.
    age_group = random.choices(
        ["young", "adult", "elder"],
        weights=[25, 55, 20],
    )[0]

    return Appearance(
        gender=gender,
        skin_color=skin,
        eye_color=eye,
        lip_color=lip,
        seed=seed,
        expression=expression,
        age_group=age_group,
        print_style=print_style,
        print_strength=print_strength,
        stamp_style=stamp_style,
        stamp_strength=stamp_strength,
        hair_style=hair_style,
        hair_color=hair,
        eyebrow_color=eyebrow,
        facial_hair_style=facial_hair,
        facial_hair_color=facial_hair_color,
        top_style=top_style,
        top_color=top_color,
        top_inflate=random.uniform(0.014, 0.020),
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
        # Feed the appearance age into the structural head pass. The value is
        # only a rest-shape hint; expression animation still resolves through
        # ARKit blendshape weights below.
        self.shape.age_group = app.age_group
        add(mesh_mod.build(self.bones, self.shape), app.skin_color, 0)

        # eyes -- back-to-front so each layer can be drawn opaque without
        # losing the one behind it: sclera, limbal ring, iris, pupil,
        # catchlight. Then anatomical eyelid occlusion and lips.
        # Sclera carries a tiny per-character warm/cool jitter so a crowd
        # doesn't share one identical pure-white eye.
        sclera = (
            min(1.0, _EYE_WHITE[0] + (app.seed - 0.5) * 0.04),
            min(1.0, _EYE_WHITE[1] + (app.seed - 0.5) * 0.02),
            min(1.0, _EYE_WHITE[2] - (app.seed - 0.5) * 0.02),
        )
        # Resolve face into ARKit-52 weight vector once per build.
        face_w = face_anim.FaceRig.resolve(app)

        add(mesh_mod.build_eyes(self.bones, self.shape),       sclera,         3)
        add(mesh_mod.build_limbus(self.bones, self.shape),     _EYE_LIMBUS,    3)
        add(mesh_mod.build_iris(self.bones, self.shape, face_w), app.eye_color, 3)
        # Collarette: faintly darker ring inside the iris -- breaks the
        # colored disc into pupillary + ciliary zones the way a real
        # iris does.
        collarette = tuple(c * 0.55 for c in app.eye_color)
        add(mesh_mod.build_collarette(self.bones, self.shape, face_w),
            collarette, 3)
        add(mesh_mod.build_pupils(self.bones, self.shape, face_w),
            app.pupil_color, 3)
        # Lacrimal caruncle (flesh-pink tear-duct bump at medial canthus).
        # Tinted off the character's skin tone so it tracks body color.
        caruncle = (
            min(1.0, app.skin_color[0] * 0.92 + 0.08),
            min(1.0, app.skin_color[1] * 0.62),
            min(1.0, app.skin_color[2] * 0.58),
        )
        add(mesh_mod.build_caruncle(self.bones, self.shape),   caruncle,       0)
        add(mesh_mod.build_catchlights(self.bones, self.shape, face_w),
            _EYE_CATCH, 3)
        add(mesh_mod.build_eyelids(self.bones, self.shape, face_w),
            app.skin_color, 0)
        add(mesh_mod.build_mouth_cavity(self.bones, self.shape, face_w),
            _MOUTH_DARK, 0)
        add(mesh_mod.build_lips(self.bones, self.shape, face_w),
            app.lip_color, 0)

        # Age-dependent creases (forehead / nasolabial / crows'-feet). No-op
        # for young faces. Keep them close to skin tone: under the surface
        # shader they should read as shallow folds, not painted black decals.
        crease_mul = 0.76 if app.age_group == "elder" else 0.88
        crease = tuple(min(1.0, c * crease_mul) for c in app.skin_color)
        dynamic_crease = tuple(min(1.0, c * 0.90) for c in app.skin_color)
        add(mesh_mod.build_expression_folds(self.bones, self.shape, face_w),
            dynamic_crease, 0)
        add(mesh_mod.build_age_detail(self.bones, self.shape, app.age_group),
            crease, 0)

        # hair + eyebrows
        add(accessories.build_hair(self.bones, app.hair_style), app.hair_color, 2)
        add(accessories.build_eyebrows(self.bones, face_w),     app.eyebrow_color, 2)
        add(accessories.build_facial_hair(self.bones, app.facial_hair_style),
            app.facial_hair_color, 2)

        # clothing
        if app.top_style == "dress":
            add(garments.build_dress(self.bones,
                                     length_frac=app.dress_length_frac,
                                     flare=app.dress_flare),
                app.top_color, 1)
            add(garments.build_bust_overlay(self.bones, self.shape,
                                            top_inflate=0.014),
                app.top_color, 1)
        else:
            # Fabric-colored underlayer on the arms, drawn before the sleeve
            # so momentary skin-through-sleeve clipping reveals garment
            # color instead of bare skin. Only applicable to sleeved tops.
            add(garments.build_sleeve_underlayer(self.bones, app.top_style,
                                                 top_inflate=app.top_inflate,
                                                 length_scale=app.top_length_scale,
                                                 gender=self.shape.gender),
                app.top_color, 1)
            add(garments.build_top(self.bones, app.top_style,
                                   inflate=app.top_inflate,
                                   length_scale=app.top_length_scale,
                                   gender=self.shape.gender,
                                   shape=self.shape),
                app.top_color, 1)
            # Fabric bust overlay: nests over the bust skin so the top
            # reads as draped fabric instead of bare skin poking through
            # the cylindrical tank/shirt.
            add(garments.build_bust_overlay(self.bones, self.shape,
                                            top_inflate=app.top_inflate),
                app.top_color, 1)

        if app.bottom_style == "skirt":
            add(garments.build_skirt(self.bones,
                                     length_frac=app.skirt_length_frac,
                                     flare=app.skirt_flare),
                app.bottom_color, 1)
        elif app.bottom_style in ("pants", "shorts"):
            add(garments.build_bottom(self.bones, app.bottom_style,
                                      inflate=app.bottom_inflate,
                                      length_scale=app.bottom_length_scale,
                                      gender=self.shape.gender),
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
        pose = skeleton.derive_cloth_pose(self.bones, self.pose)
        return skeleton.compute_bone_matrices(self.bones, pose, self.root_offset)

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
