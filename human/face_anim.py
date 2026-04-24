"""ARKit-52 / FACS-aligned facial animation rig.

This module is the *standard surface* the rest of the codebase talks to
when describing a facial expression. It exposes:

- ``ARKIT_BLENDSHAPES`` -- the 52 canonical Apple ARKit blendshape names
  (FACS-coded, also used by Epic MetaHumans, Live Link Face, NVIDIA
  Audio2Face, MediaPipe FaceLandmarker -> ARKit converters, ...).
- ``EXPRESSION_PRESETS`` -- named expressions encoded as ARKit weight
  dicts. Includes the legacy preset names (neutral/smile/frown/squint/
  surprised) plus the FACS-coded Ekman set (anger/disgust/fear/contempt/
  ...) and Duchenne/non-Duchenne smile.
- ``FaceRig.resolve(appearance)`` -- collapses an ``Appearance`` (legacy
  ``expression`` string + optional explicit ``blendshapes`` dict) into a
  final ``{name: float in [0,1]}`` map.
- ``FaceClip`` -- linear-interp keyframe sequence for transitions.

Geometry consumers (``build_eyelids``, ``build_lips``,
``build_eyebrows``, ``build_iris`` family) read individual channel
values via ``w(...)``; channels with no geometric hook in this
parametric face are accepted but no-op (eg. ``tongueOut``,
``mouthRoll*``, ``jawForward``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


# --- 52 ARKit channels -------------------------------------------------------

ARKIT_BLENDSHAPES: Tuple[str, ...] = (
    # Eyes (14)
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # Brows (5)
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight",
    # Nose (2)
    "noseSneerLeft", "noseSneerRight",
    # Jaw (4)
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    # Mouth (23)
    "mouthClose",
    "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    # Tongue (1)
    "tongueOut",
)
assert len(ARKIT_BLENDSHAPES) == 52, "ARKit spec is 52 channels"

ARKIT_INDEX: Dict[str, int] = {n: i for i, n in enumerate(ARKIT_BLENDSHAPES)}


def zeros() -> Dict[str, float]:
    """Empty (neutral) weight set."""
    return {name: 0.0 for name in ARKIT_BLENDSHAPES}


def _sym(prefix: str, value: float) -> Dict[str, float]:
    return {f"{prefix}Left": value, f"{prefix}Right": value}


def _merge(*parts: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in parts:
        out.update(p)
    return out


# --- canonical FACS-coded presets -------------------------------------------
#
# Channel values are the *target weights* for that named expression. They
# follow conventional FACS combinations (Ekman & Friesen 1978, Cohn &
# Kanade 2000). Conservative magnitudes keep transitions smooth and avoid
# pushing the parametric primitives outside their plausible range.

EXPRESSION_PRESETS: Dict[str, Dict[str, float]] = {

    "neutral": {},

    # --- Ekman six -------------------------------------------------------

    # Happy / smile (AU12). Plain non-Duchenne by default.
    "smile": _merge(
        _sym("mouthSmile", 0.70),
        _sym("mouthDimple", 0.20),
    ),
    # Duchenne smile: AU6 (cheek raiser) + AU12. The eye-corner
    # tightening is what distinguishes a real smile from a posed one.
    "smile_duchenne": _merge(
        _sym("mouthSmile", 0.75),
        _sym("mouthDimple", 0.30),
        _sym("cheekSquint", 0.55),
        _sym("eyeSquint", 0.30),
    ),

    # Sad / frown: AU15 (lip corner depressor) + AU4 (brow lowerer,
    # gentle inner pinch).
    "frown": _merge(
        _sym("mouthFrown", 0.65),
        _sym("browDown", 0.40),
        {"browInnerUp": 0.20},
    ),
    "sad": _merge(
        _sym("mouthFrown", 0.55),
        _sym("browDown", 0.20),
        {"browInnerUp": 0.55},
        _sym("mouthLowerDown", 0.15),
    ),

    # Surprise: AU1+AU2 (brows up) + AU5 (upper lid raiser) + AU26
    # (jaw drop).
    "surprise": _merge(
        {"browInnerUp": 0.85},
        _sym("browOuterUp", 0.70),
        _sym("eyeWide", 0.70),
        {"jawOpen": 0.50},
        _sym("mouthStretch", 0.10),
    ),

    # Fear: similar brows to surprise but mouth is stretched horizontal,
    # jaw less open. AU1+AU2+AU4+AU5+AU20+AU26.
    "fear": _merge(
        {"browInnerUp": 0.80},
        _sym("browOuterUp", 0.45),
        _sym("browDown", 0.20),
        _sym("eyeWide", 0.55),
        _sym("mouthStretch", 0.50),
        {"jawOpen": 0.25},
    ),

    # Anger: AU4 (brow lowerer) hard + AU5 (lid raise) + AU7 (lid tight)
    # + AU23 (lip tightener) + AU9 (slight nose wrinkle).
    "anger": _merge(
        _sym("browDown", 0.85),
        _sym("eyeSquint", 0.40),
        _sym("eyeWide", 0.20),
        _sym("noseSneer", 0.40),
        _sym("mouthPress", 0.50),
        _sym("mouthFrown", 0.20),
    ),

    # Disgust: AU9 (nose wrinkler) dominant + AU10 (upper lip raiser)
    # + mild brow lowerer.
    "disgust": _merge(
        _sym("noseSneer", 0.75),
        _sym("mouthUpperUp", 0.55),
        _sym("browDown", 0.35),
        _sym("mouthDimple", 0.15),
    ),

    # Contempt: ASYMMETRIC AU12 + AU14 on one side. The asymmetry IS
    # the expression -- if you symmetrise it, it becomes a smirk.
    "contempt": {
        "mouthSmileLeft": 0.55,
        "mouthDimpleLeft": 0.40,
        "cheekSquintLeft": 0.20,
    },

    # --- legacy aliases & utility states --------------------------------

    "squint": _sym("eyeSquint", 0.85),
    "blink": _sym("eyeBlink", 1.0),
    "wink_left":  {"eyeBlinkLeft": 1.0},
    "wink_right": {"eyeBlinkRight": 1.0},
    "surprised": None,  # filled below via alias
    "kiss": _merge({"mouthPucker": 0.75, "jawOpen": 0.05}),
    "open_mouth": {"jawOpen": 0.55},
    "ee": _merge(_sym("mouthSmile", 0.30), _sym("mouthStretch", 0.45),
                 {"jawOpen": 0.10}),
    "oo": {"mouthPucker": 0.85, "jawOpen": 0.15},
    "aa": {"jawOpen": 0.55, "mouthShrugLower": 0.15},
}

# Aliases.
EXPRESSION_PRESETS["surprised"] = EXPRESSION_PRESETS["surprise"]


def preset_weights(name: str) -> Dict[str, float]:
    """Return a *full* 52-channel weight dict for a named preset.

    Unknown names fall back to neutral. All 52 channels are present so
    callers never need to guard for missing keys.
    """
    out = zeros()
    base = EXPRESSION_PRESETS.get(name)
    if base is None:
        # unknown name -> neutral
        return out
    for k, v in base.items():
        if k in ARKIT_INDEX:
            out[k] = float(v)
    return out


# --- rig --------------------------------------------------------------------

@dataclass
class FaceRig:
    """Resolves an Appearance into a final ARKit-52 weight vector.

    Resolution order (later overrides earlier):
      1. Named preset selected by ``Appearance.expression`` (legacy
         field; defaults to ``neutral``).
      2. Per-channel overrides in ``Appearance.blendshapes`` if set.
      3. Hard clamp to [0, 1].
    """

    @staticmethod
    def resolve(appearance) -> Dict[str, float]:
        weights = preset_weights(getattr(appearance, "expression", "neutral"))
        explicit = getattr(appearance, "blendshapes", None)
        if explicit:
            for k, v in explicit.items():
                if k in ARKIT_INDEX:
                    weights[k] = float(v)
        for k, v in weights.items():
            if v < 0.0:
                weights[k] = 0.0
            elif v > 1.0:
                weights[k] = 1.0
        return weights

    @staticmethod
    def w(weights: Optional[Mapping[str, float]], name: str) -> float:
        """Safe channel read. Missing dict / missing key -> 0."""
        if weights is None:
            return 0.0
        v = weights.get(name, 0.0)
        if v <= 0.0:
            return 0.0
        if v >= 1.0:
            return 1.0
        return float(v)


# Convenient module-level alias so call sites can do ``face_anim.w(...)``.
def w(weights: Optional[Mapping[str, float]], name: str) -> float:
    return FaceRig.w(weights, name)


# --- animation clips --------------------------------------------------------

@dataclass
class FaceKey:
    time: float
    weights: Dict[str, float]


class FaceClip:
    """A keyframed sequence of ARKit weight dicts.

    ``sample(t)`` returns a fully-populated 52-channel dict via linear
    interpolation between the bracketing keys. Times before/after the
    clip clamp to the first/last key (no looping; wrap externally if
    desired).
    """

    def __init__(self, keys: Iterable[FaceKey]):
        self.keys: List[FaceKey] = sorted(
            (FaceKey(k.time, _full(k.weights)) for k in keys),
            key=lambda k: k.time,
        )
        if not self.keys:
            self.keys = [FaceKey(0.0, zeros())]

    @property
    def duration(self) -> float:
        return self.keys[-1].time - self.keys[0].time

    def sample(self, t: float) -> Dict[str, float]:
        if t <= self.keys[0].time:
            return dict(self.keys[0].weights)
        if t >= self.keys[-1].time:
            return dict(self.keys[-1].weights)
        # find bracket
        for i in range(len(self.keys) - 1):
            a, b = self.keys[i], self.keys[i + 1]
            if a.time <= t <= b.time:
                span = b.time - a.time
                u = 0.0 if span <= 0 else (t - a.time) / span
                return {n: a.weights[n] * (1.0 - u) + b.weights[n] * u
                        for n in ARKIT_BLENDSHAPES}
        return dict(self.keys[-1].weights)


def _full(partial: Mapping[str, float]) -> Dict[str, float]:
    out = zeros()
    for k, v in partial.items():
        if k in ARKIT_INDEX:
            out[k] = float(v)
    return out


def clip_from_presets(steps: Iterable[Tuple[float, str]]) -> FaceClip:
    """Build a FaceClip from ``[(time, preset_name), ...]`` pairs.

    Convenience for transition tests like
    ``[(0, "neutral"), (0.5, "smile"), (1.0, "neutral")]``.
    """
    keys = [FaceKey(t, preset_weights(name)) for t, name in steps]
    return FaceClip(keys)
