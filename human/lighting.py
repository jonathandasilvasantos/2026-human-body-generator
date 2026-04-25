"""Lighting profiles for the procedural human viewer.

Ten named profiles bound to viewer keys 0-9. Each profile is a fully
self-contained lighting setup -- key + fill + back + ambient + grade --
so switching between them gives a clean A/B at runtime instead of layer-
by-layer tweaks on top of a hidden global default.

Design decisions
----------------
* Up to 16 directional lights. A *directional* model (rather than a
  full physically-based area-light integration) is the right complexity
  level for a real-time procedural viewer that does not own a shadow
  map -- and it cleanly supports the ring-light profile by allocating
  N lights radially around the camera axis.

* Per-profile **hemispheric ambient** (sky / ground colours) replaces
  the renderer's previous single-scalar global ambient. That lets a
  golden-hour preset push warm fill while a noir preset crushes the
  shadow side to deep teal.

* Per-profile **filmic grade** (exposure, contrast, tint) baked into
  the ACES-fit tone-map in the shader. Lighting *style* is as much
  about grade as it is about light placement -- noir without crushed
  blacks isn't noir.

* The first light in the array is treated as the canonical "key" by
  the shader so all specular paths pick up its colour and direction;
  the rest of the array contributes only diffuse + ambient occlusion-
  free fill.

References
----------
- Hurter (1909) / Megaw (1939) -- portrait lighting taxonomy
  (Rembrandt, Loop, Split, Butterfly, Broad, Short).
- Burchett (2009), *Lighting Setups* -- studio beauty / cosmetic
  reference with seamless-white background and 1:1 fill.
- Hoffman (2010), *Background: Physics and Math of Shading*
  (SIGGRAPH course) -- wrapped diffuse for soft key lights.
- Driscoll (2002) / Ramamoorthi (2001) -- hemispheric ambient as a
  cheap two-band SH approximation to image-based lighting.
- Karis (2014), *Physically Based Shading at Disney* -- exposure +
  filmic tone-map pipeline.
- Narkowicz (2015), *ACES Filmic Tone Mapping Curve* -- the curve in
  the fragment shader.
- Goldman & Sloan (2019), *Cinematography for Real-Time Rendering* --
  three-point conventions, rim placement.
- Krystek (1985) -- Planckian black-body locus -> sRGB approximation.
- Hudack (2022), *Ring Lights for Portrait Photography* -- N=12-16
  lights on a ~30 cm radius give shadowless beauty illumination.

Each profile is the result of one improvement cycle on top of the
previous baseline; see commits on feature/lighting-profiles for the
A/B captures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from OpenGL.GL import (
    glUniform1i, glUniform1f, glUniform3f, glUniform3fv,
)


MAX_LIGHTS = 16


# --- colour temperature ----------------------------------------------------

def kelvin_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Approximate sRGB triplet for a black-body at ``kelvin`` K.

    Polynomial fit due to Tanner Helland (2012), accurate to ~50 K
    against the Planckian locus across 1000-40000 K. Output is unit-
    intensity (Y normalised to ~1.0) so a profile can scale it
    independently.
    """
    t = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return (max(0.0, min(255.0, r)) / 255.0,
            max(0.0, min(255.0, g)) / 255.0,
            max(0.0, min(255.0, b)) / 255.0)


def _scale(rgb: Tuple[float, float, float], k: float) -> Tuple[float, float, float]:
    return (rgb[0] * k, rgb[1] * k, rgb[2] * k)


def _dir_from_polar(az_deg: float, el_deg: float) -> Tuple[float, float, float]:
    """World-space direction TOWARD the light.

    Convention: character faces +Z, camera sits on +Z looking -Z. So a
    light at azimuth=0, elevation=0 sits directly in front of the
    character (positive Z). Positive azimuth rotates around +Y so the
    light moves to the character's left from its own POV (camera right
    in screen space). Positive elevation lifts the light up.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cx = math.cos(el) * math.sin(az)
    cy = math.sin(el)
    cz = math.cos(el) * math.cos(az)
    n = math.sqrt(cx * cx + cy * cy + cz * cz)
    return (cx / n, cy / n, cz / n)


# --- profile dataclass -----------------------------------------------------

@dataclass
class LightingProfile:
    name: str
    description: str
    lights: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=list)
    """List of ``(direction_xyz, colour_rgb_with_intensity)`` tuples.

    Direction is world-space toward the light (unit vector). Colour is
    linear-RGB premultiplied by intensity; values can exceed 1.0.
    """

    light_wrap: float = 0.0
    """0 = pure Lambert. 1 = full half-sphere wrap (Hoffman 2010)."""

    ambient_top: Tuple[float, float, float] = (0.10, 0.12, 0.16)
    ambient_bot: Tuple[float, float, float] = (0.07, 0.06, 0.05)
    """Hemispheric ambient (sky colour at world-up, ground colour at world-down)."""

    exposure: float = 1.0
    contrast: float = 0.0
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rim_strength: float = 0.0
    rim_color: Tuple[float, float, float] = (1.0, 0.85, 0.7)
    spec_mul: float = 1.0
    face_fill: float = 0.30
    """Skin-readability fill amount. Profiles that want hard half-shadow
    (Split, Noir) drop this to ~0.05; beauty / studio profiles raise it
    to ~0.45 to keep cavities open under broad soft light."""


# --- builders --------------------------------------------------------------

def _three_point(key_az: float, key_el: float, key_color, key_int: float,
                 fill_az: float, fill_el: float, fill_color, fill_int: float,
                 back_az: float, back_el: float, back_color, back_int: float):
    return [
        (_dir_from_polar(key_az, key_el),  _scale(key_color, key_int)),
        (_dir_from_polar(fill_az, fill_el), _scale(fill_color, fill_int)),
        (_dir_from_polar(back_az, back_el), _scale(back_color, back_int)),
    ]


def _ring(n: int, radius_az: float, elevation: float,
          color: Tuple[float, float, float], total_intensity: float):
    """N lights in a ring around the camera forward axis (front-facing).

    Geometry: ring sits in a plane parallel to the image plane at
    elevation 0 (camera level). The ring's "tilt" is set by ``elevation``,
    which lifts every light by that polar angle so the ring is slightly
    above eye-line (typical beauty placement).
    """
    per = total_intensity / max(1, n)
    out = []
    for i in range(n):
        phi = 2.0 * math.pi * (i / n)
        # Map ring position into world-space directions: rotate the
        # camera-forward vector (+Z) around its axis by phi at radius
        # ``radius_az`` (azimuthal half-angle of the cone).
        sin_r = math.sin(math.radians(radius_az))
        cos_r = math.cos(math.radians(radius_az))
        x = sin_r * math.cos(phi)
        y = sin_r * math.sin(phi)
        z = cos_r
        # Lift the entire ring by ``elevation`` so it sits above eye
        # line by tilting around X-axis.
        ce = math.cos(math.radians(elevation))
        se = math.sin(math.radians(elevation))
        y2 = y * ce - z * se
        z2 = y * se + z * ce
        n_ = math.sqrt(x * x + y2 * y2 + z2 * z2)
        out.append(((x / n_, y2 / n_, z2 / n_), _scale(color, per)))
    return out


# --- the ten profiles ------------------------------------------------------

def _build_profiles() -> List[LightingProfile]:
    K_TUNGSTEN = kelvin_rgb(3200.0)   # warm classic studio
    K_DAYLIGHT = kelvin_rgb(5600.0)   # neutral white
    K_OVERCAST = kelvin_rgb(6500.0)   # cool overcast sky
    K_SUNSET   = kelvin_rgb(2900.0)   # golden hour
    K_SHADE    = kelvin_rgb(8500.0)   # cool open-shade fill

    profiles: List[LightingProfile] = []

    # 0 -- Default Portrait. Three-point with a warm key, cool fill,
    # warm rim. Closely matches the previous renderer baseline so old
    # captures stay comparable.
    profiles.append(LightingProfile(
        name="Portrait",
        description="default three-point: warm key, cool fill, warm rim",
        lights=_three_point(
            key_az=28,  key_el=42, key_color=K_DAYLIGHT,  key_int=1.10,
            fill_az=-50, fill_el=18, fill_color=K_SHADE,  fill_int=0.45,
            back_az=180, back_el=35, back_color=K_TUNGSTEN, back_int=0.30,
        ),
        light_wrap=0.20,
        ambient_top=(0.18, 0.20, 0.24),
        ambient_bot=(0.10, 0.09, 0.08),
        exposure=1.05, contrast=0.04,
        tint=(1.00, 0.99, 0.97),
        rim_strength=0.05, rim_color=K_TUNGSTEN,
        spec_mul=1.0, face_fill=0.30,
    ))

    # 1 -- Studio Beauty / Cosmetic. Big soft key from above-front, low
    # fill, white seamless background hemi. Beauty dishes / softboxes
    # have wide angular extent -> very high wrap value.
    profiles.append(LightingProfile(
        name="Studio Beauty",
        description="soft beauty-dish key + 1:1 fill, white seamless hemi",
        lights=_three_point(
            key_az=0,  key_el=55,  key_color=K_DAYLIGHT, key_int=1.30,
            fill_az=0, fill_el=-15, fill_color=K_DAYLIGHT, fill_int=0.55,
            back_az=180, back_el=20, back_color=K_DAYLIGHT, back_int=0.20,
        ),
        light_wrap=0.85,
        ambient_top=(0.40, 0.42, 0.46),
        ambient_bot=(0.32, 0.30, 0.28),
        exposure=1.10, contrast=-0.05,
        tint=(1.02, 1.01, 1.02),
        rim_strength=0.04, rim_color=(1.0, 1.0, 1.0),
        spec_mul=1.20, face_fill=0.45,
    ))

    # 2 -- Rembrandt. 45 degree key, 45 degree elevation; minimal fill.
    # Produces the characteristic illuminated triangle on the shadow
    # cheek -- bright triangle below the eye, point reaching the corner
    # of the mouth. Hurter / Megaw classification.
    profiles.append(LightingProfile(
        name="Rembrandt",
        description="45/45 key, weak fill, deep shadow triangle on far cheek",
        lights=_three_point(
            key_az=45,  key_el=45,  key_color=K_TUNGSTEN, key_int=1.40,
            fill_az=-65, fill_el=10, fill_color=K_SHADE, fill_int=0.18,
            back_az=170, back_el=30, back_color=K_TUNGSTEN, back_int=0.25,
        ),
        light_wrap=0.10,
        ambient_top=(0.10, 0.10, 0.12),
        ambient_bot=(0.05, 0.04, 0.03),
        exposure=1.00, contrast=0.10,
        tint=(1.04, 0.98, 0.92),
        rim_strength=0.06, rim_color=K_TUNGSTEN,
        spec_mul=1.05, face_fill=0.10,
    ))

    # 3 -- Split. Single 90-degree side key. Half the face is in
    # shadow; classic dramatic / interrogation feel.
    profiles.append(LightingProfile(
        name="Split",
        description="single 90deg side key, no fill, half-face shadow",
        lights=[
            (_dir_from_polar(90, 0),  _scale(K_DAYLIGHT, 1.30)),
        ],
        light_wrap=0.05,
        ambient_top=(0.06, 0.07, 0.09),
        ambient_bot=(0.04, 0.04, 0.04),
        exposure=1.00, contrast=0.18,
        tint=(0.98, 0.98, 1.00),
        rim_strength=0.0, rim_color=(1.0, 1.0, 1.0),
        spec_mul=1.10, face_fill=0.05,
    ))

    # 4 -- Loop. Key at ~35deg az / 35deg el. Small "loop" nose shadow
    # falls toward the corner of the mouth without touching it. The
    # most flattering everyday portrait setup (Burchett 2009).
    profiles.append(LightingProfile(
        name="Loop",
        description="35/35 key, soft fill, small nose-loop shadow",
        lights=_three_point(
            key_az=35, key_el=35, key_color=K_DAYLIGHT, key_int=1.20,
            fill_az=-40, fill_el=20, fill_color=K_SHADE, fill_int=0.40,
            back_az=180, back_el=25, back_color=K_TUNGSTEN, back_int=0.22,
        ),
        light_wrap=0.30,
        ambient_top=(0.18, 0.20, 0.22),
        ambient_bot=(0.10, 0.09, 0.08),
        exposure=1.05, contrast=0.05,
        tint=(1.0, 1.0, 1.0),
        rim_strength=0.05, rim_color=K_DAYLIGHT,
        spec_mul=1.05, face_fill=0.28,
    ))

    # 5 -- Butterfly / Paramount. Light directly above-front, tilted
    # down -- symmetrical butterfly-shaped shadow under the nose. Old-
    # Hollywood glamour standard (named for its use on Paramount stars
    # in the 1930s). Pairs naturally with a slight clamshell fill from
    # below, which we simulate with a low front bounce.
    profiles.append(LightingProfile(
        name="Butterfly",
        description="overhead key + low front bounce, butterfly nose shadow",
        lights=[
            (_dir_from_polar(0, 70),  _scale(K_DAYLIGHT, 1.40)),
            (_dir_from_polar(0, -25), _scale(K_DAYLIGHT, 0.35)),  # clamshell
            (_dir_from_polar(180, 30), _scale(K_TUNGSTEN, 0.20)),  # back hint
        ],
        light_wrap=0.25,
        ambient_top=(0.22, 0.22, 0.24),
        ambient_bot=(0.14, 0.12, 0.11),
        exposure=1.10, contrast=0.02,
        tint=(1.01, 1.00, 0.99),
        rim_strength=0.04, rim_color=(1.0, 0.95, 0.88),
        spec_mul=1.15, face_fill=0.32,
    ))

    # 6 -- Cinematic Noir. Single hard key from low side, deep teal-
    # cyan ambient (negative-fill bounce off a painted set wall),
    # crushed blacks, slight desaturation. Single source plus crushed
    # blacks and rim is the canonical noir grammar.
    profiles.append(LightingProfile(
        name="Cinematic Noir",
        description="single low-side hard key + cyan negative fill",
        lights=[
            (_dir_from_polar(75, 25),   _scale(K_TUNGSTEN, 1.55)),
            (_dir_from_polar(200, 40),  _scale((0.45, 0.85, 1.0), 0.18)),
        ],
        light_wrap=0.0,
        ambient_top=(0.05, 0.10, 0.14),
        ambient_bot=(0.03, 0.05, 0.07),
        exposure=0.95, contrast=0.32,
        tint=(0.92, 0.96, 1.05),
        rim_strength=0.16, rim_color=(0.55, 0.85, 1.0),
        spec_mul=1.30, face_fill=0.05,
    ))

    # 7 -- Golden Hour. Warm low-angle "sun" key + cool sky fill.
    # Wide wrap on the sun (large bright ball low on the horizon
    # functions as a soft source from a face's POV).
    profiles.append(LightingProfile(
        name="Golden Hour",
        description="low warm sun + cool sky hemi",
        lights=[
            (_dir_from_polar(50, 12), _scale(K_SUNSET, 1.45)),
            (_dir_from_polar(-30, 60), _scale(K_OVERCAST, 0.35)),
            (_dir_from_polar(190, 10), _scale(K_SUNSET, 0.30)),  # ground-bounce hint
        ],
        light_wrap=0.55,
        ambient_top=(0.30, 0.34, 0.42),
        ambient_bot=(0.28, 0.18, 0.10),
        exposure=1.15, contrast=0.06,
        tint=(1.06, 1.00, 0.92),
        rim_strength=0.20, rim_color=K_SUNSET,
        spec_mul=1.10, face_fill=0.30,
    ))

    # 8 -- HDRI Studio Hemi. No directional key; lighting is dominated
    # by a strong hemisphere ambient with cool sky / warm ground. Cheap
    # SH-band-2 IBL approximation -- the look you get from a softbox
    # tent or an overcast outdoor scene.
    profiles.append(LightingProfile(
        name="HDRI Studio",
        description="hemispheric IBL: cool sky + warm bounce, no harsh key",
        lights=[
            # Very gentle "sun" so specular paths still anchor.
            (_dir_from_polar(15, 60), _scale(K_OVERCAST, 0.55)),
            (_dir_from_polar(-15, 30), _scale(K_OVERCAST, 0.30)),
        ],
        light_wrap=0.95,
        ambient_top=(0.55, 0.60, 0.70),
        ambient_bot=(0.40, 0.34, 0.28),
        exposure=1.00, contrast=-0.02,
        tint=(1.0, 1.0, 1.0),
        rim_strength=0.02, rim_color=(1.0, 1.0, 1.0),
        spec_mul=1.0, face_fill=0.40,
    ))

    # 9 -- Ring Light. 12 small lights in a circle around the camera-
    # forward axis at a 22-degree half-angle, lifted 8 degrees above
    # eye-line. Even shadowless illumination, signature catchlight ring
    # in the eyes (which the multi-layered eye stack already captures).
    profiles.append(LightingProfile(
        name="Ring Light",
        description="12-light circle around camera axis (beauty / vlog)",
        lights=_ring(n=12, radius_az=22.0, elevation=8.0,
                     color=K_DAYLIGHT, total_intensity=1.80),
        light_wrap=0.40,
        ambient_top=(0.20, 0.22, 0.26),
        ambient_bot=(0.16, 0.16, 0.18),
        exposure=1.10, contrast=-0.03,
        tint=(1.00, 1.00, 1.02),
        rim_strength=0.03, rim_color=(1.0, 1.0, 1.0),
        spec_mul=1.20, face_fill=0.40,
    ))

    return profiles


PROFILES: List[LightingProfile] = _build_profiles()
assert len(PROFILES) == 10, "ten profiles must be defined for keys 0-9"


# --- application -----------------------------------------------------------

def apply(prog, profile: LightingProfile) -> None:
    """Push ``profile`` to the bound SkinProgram. Caller has already
    bound ``prog.prog``."""
    n = min(len(profile.lights), MAX_LIGHTS)
    if prog.u_num_lights != -1:
        glUniform1i(prog.u_num_lights, n)
    if n > 0:
        dirs = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        cols = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        for i, (d, c) in enumerate(profile.lights[:n]):
            dirs[i] = d
            cols[i] = c
        if prog.u_light_dir != -1:
            glUniform3fv(prog.u_light_dir, MAX_LIGHTS, dirs)
        if prog.u_light_col != -1:
            glUniform3fv(prog.u_light_col, MAX_LIGHTS, cols)
    if prog.u_light_wrap != -1:
        glUniform1f(prog.u_light_wrap, profile.light_wrap)
    if prog.u_amb_top != -1:
        glUniform3f(prog.u_amb_top, *profile.ambient_top)
    if prog.u_amb_bot != -1:
        glUniform3f(prog.u_amb_bot, *profile.ambient_bot)
    if prog.u_exposure != -1:
        glUniform1f(prog.u_exposure, profile.exposure)
    if prog.u_contrast != -1:
        glUniform1f(prog.u_contrast, profile.contrast)
    if prog.u_tint != -1:
        glUniform3f(prog.u_tint, *profile.tint)
    if prog.u_rim_strength != -1:
        glUniform1f(prog.u_rim_strength, profile.rim_strength)
    if prog.u_rim_color != -1:
        glUniform3f(prog.u_rim_color, *profile.rim_color)
    if prog.u_spec_mul != -1:
        glUniform1f(prog.u_spec_mul, profile.spec_mul)
    if prog.u_face_fill != -1:
        glUniform1f(prog.u_face_fill, profile.face_fill)


def by_index(i: int) -> LightingProfile:
    return PROFILES[i % len(PROFILES)]
