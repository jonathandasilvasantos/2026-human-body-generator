# Facial animation system — ARKit-52 / FACS-aligned standard

## Why ARKit-52

For body animation we already lean on widely-shared standards (Mixamo
skeleton, BVH playback, Bandai Namco dataset). For faces the equivalent
question is: *is there an industry-standard rig API that downstream tools
can drive?* The answer in 2024-2026 is effectively **yes — Apple's
ARKit 52 face blendshapes**, which is FACS-aligned and consumed by:

- Apple ARKit (`ARFaceAnchor.BlendShapeLocation`) and Live Link Face
- Epic MetaHumans facial rig (channels named `eyeBlinkLeft`, `jawOpen`,
  `mouthSmileLeft`, …)
- NVIDIA Audio2Face / Omniverse output
- Most commercial face-tracking SDKs (Faceware, Dynamixyz, FaceCap)
- The vast majority of recent academic blendshape-personalisation papers
  (e.g. "Automated Blendshape Personalization for Faithful Face
  Animations Using Commodity Smartphones", ACM 2022)

FACS itself (Ekman & Friesen, 1978) is the *muscular* standard — 30+
Action Units with anatomical references — but it is a coding scheme, not
an animation API. ARKit-52 is essentially a curated subset of FACS AUs
exposed as `[name → weight∈[0,1]]`. Picking ARKit-52 gives us a rig that
any animator, motion-capture pipeline or audio-driven facial-animation
model already speaks.

## The 52 channels

Grouped by region (see `human/face_anim.py::ARKIT_BLENDSHAPES`):

- **Eyes (14):** `eyeBlinkLeft/Right`, `eyeLookDownLeft/Right`,
  `eyeLookInLeft/Right`, `eyeLookOutLeft/Right`, `eyeLookUpLeft/Right`,
  `eyeSquintLeft/Right`, `eyeWideLeft/Right`
- **Brows (5):** `browDownLeft/Right`, `browInnerUp`,
  `browOuterUpLeft/Right`
- **Cheeks (3):** `cheekPuff`, `cheekSquintLeft/Right`
- **Nose (2):** `noseSneerLeft/Right`
- **Jaw (4):** `jawForward`, `jawLeft`, `jawRight`, `jawOpen`
- **Mouth (23):** `mouthClose`, `mouthFunnel`, `mouthPucker`,
  `mouthLeft`, `mouthRight`, `mouthSmileLeft/Right`,
  `mouthFrownLeft/Right`, `mouthDimpleLeft/Right`,
  `mouthStretchLeft/Right`, `mouthRollLower`, `mouthRollUpper`,
  `mouthShrugLower`, `mouthShrugUpper`, `mouthPressLeft/Right`,
  `mouthLowerDownLeft/Right`, `mouthUpperUpLeft/Right`
- **Tongue (1):** `tongueOut`

Every channel is a float in `[0, 1]`. Symmetric expressions set the
left/right pair to the same value; asymmetry is just unequal weights.

## Architecture

```
Appearance.expression  ──┐
Appearance.blendshapes ──┤──► FaceRig.resolve()  ──► dict[str,float]
                                  │
                                  ▼
                       build_eyelids / build_lips / build_brows /
                       build_cheeks / build_jaw_offset / ...
```

- `human/face_anim.py` defines `ARKIT_BLENDSHAPES` (the 52 names),
  `EXPRESSION_PRESETS` (FACS-coded named expressions: neutral, smile,
  Duchenne smile, frown, surprise, anger, disgust, fear, contempt,
  squint, blink), and `FaceRig.resolve(...)`.
- The procedural face primitives in `human/mesh.py` consume the resolved
  weights instead of the legacy `expression` string. The legacy string
  is preserved as a convenient preset name; setting
  `Appearance.blendshapes = {...}` overrides it on a per-channel basis.

### Mapping weights → parametric primitives

Because the face is built from ellipsoids and flat patches (not a vertex
morph mesh), each blendshape weight modulates a *parameter* of the
relevant primitive rather than displacing vertices. Examples:

| Channel              | Drives                                            |
|----------------------|---------------------------------------------------|
| `eyeBlinkLeft/Right` | Upper-lid Y position (closes onto pupil at w=1)   |
| `eyeWideLeft/Right`  | Upper-lid Y up + lower-lid Y down                 |
| `eyeSquintLeft/Right`| Aperture × (1 − 0.45·w)                            |
| `eyeLookUp/Down/...` | Iris/pupil offset on the eye sphere               |
| `browInnerUp`        | Inner brow vertex Y up                            |
| `browOuterUp*`       | Outer brow vertex Y up                            |
| `browDown*`          | Brow vertex Y down + slight inner-pinch           |
| `mouthSmile*`        | Corner ellipsoid Y up                             |
| `mouthFrown*`        | Corner ellipsoid Y down                           |
| `jawOpen`            | Lower-lip Y down + chin/jaw band Y down           |
| `mouthPucker`        | Lip ellipsoids X squeezed inward, Z forward       |
| `mouthFunnel`        | Lip ring X out, opening                           |
| `cheekPuff`          | Cheekbone radial inflate                          |
| `cheekSquint*`       | Lower-lid Y up (Duchenne marker)                  |
| `noseSneer*`         | Upper-lip Y up + nose-wing Y up on that side      |

Channels that have no visual hook in the current primitive set
(`tongueOut`, `mouthRoll*`, `jawForward`) are accepted by the API and
clamped, but are no-ops geometrically until the corresponding primitive
exists. They are listed in the API so motion-capture data can still be
fed in without dropping channels.

## Named expression presets (FACS-coded)

`EXPRESSION_PRESETS` encodes the canonical Ekman six + neutral +
contempt + the legacy presets, each as a dict of ARKit weights. Lifted
from the FACS literature (Ekman & Friesen 1978; Cohn & Kanade 2000):

- **neutral** — all zero.
- **smile** — `mouthSmileLeft/Right = 0.7`.
- **smile_duchenne** — adds `cheekSquintLeft/Right = 0.55` and
  `eyeSquintLeft/Right = 0.30`. Distinguishing marker for genuine
  smiles (FACS AU6 + AU12).
- **frown** — `mouthFrownLeft/Right = 0.65`, `browDownLeft/Right = 0.4`.
- **surprise** — `browInnerUp = 0.85`, `browOuterUpLeft/Right = 0.7`,
  `eyeWideLeft/Right = 0.7`, `jawOpen = 0.5`.
- **anger** — `browDownLeft/Right = 0.85`, `noseSneerLeft/Right = 0.4`,
  `mouthPressLeft/Right = 0.5`, lid tightening via
  `eyeSquintLeft/Right = 0.4`.
- **disgust** — `noseSneerLeft/Right = 0.75`,
  `mouthUpperUpLeft/Right = 0.55`, `browDownLeft/Right = 0.35`.
- **fear** — same brows as surprise + `mouthStretchLeft/Right = 0.5`,
  `jawOpen = 0.25`, `eyeWideLeft/Right = 0.55`.
- **contempt** — *asymmetric* `mouthSmileLeft = 0.55`,
  `mouthDimpleLeft = 0.4`. The asymmetry is the point.
- **squint** — `eyeSquintLeft/Right = 0.85`.
- **surprised**, **smile**, **frown** also alias the legacy names so
  existing callers keep working.

## Animation cycles

A full design pass requires moving from *static poses* to *transitions*.
A small `FaceClip` API (`human/face_anim.py::FaceClip`) drives a list of
keyframes (`time → weight dict`) with linear interpolation, evaluated
each frame and pushed into `Appearance.blendshapes` before
`build_gpu()` is called. Capture tooling stitches sequential frames
into image strips (`tools/face_anim_capture.py`).

## What this gives us

- Any facial-tracking source that emits ARKit weights (Live Link Face,
  Audio2Face, MediaPipe FaceLandmarker → ARKit) can drive the rig
  without per-character retargeting.
- Named FACS-coded expressions (anger, fear, disgust, …) are now first
  class instead of just `{neutral, smile, frown, squint, surprised}`.
- Symmetry vs asymmetry is just left/right weight equality — contempt,
  one-eyed wink, single-side brow raise all fall out for free.
- Transitions are explicit (FaceClip), so animation continuity is part
  of the rig instead of an afterthought.

## Cycle log

- **Cycle 1** — rig structure: introduce `face_anim.py`, route weights
  through `build_eyelids` / `build_lips` / `build_brows`, render the
  seven Ekman expressions, compare against reference.
- **Cycle 2** — transitions and asymmetry: add `FaceClip`, render
  neutral→smile→neutral and neutral→surprise→neutral strips, add
  contempt + asymmetric blink, capture short clips.
- **Cycle 3** — cross-character validation: render the full preset
  matrix across multiple seeds/genders/age groups, fix any
  coverage gaps in the brow/cheek/jaw mapping.

## References

- Apple, *ARFaceAnchor.BlendShapeLocation*, developer.apple.com.
- Melinda Ozel, *ARKit-to-FACS cheat sheet*, melindaozel.com.
- Pooya Deperson, *The Ultimate Guide to Creating ARKit's 52 Facial
  Blendshapes* (anatomy references).
- Ekman & Friesen, *Facial Action Coding System*, 1978.
- Lewis et al., *Practice and Theory of Blendshape Facial Models*,
  Eurographics STAR 2014.
- Bouaziz et al., *Online Modeling for Realtime Facial Animation*,
  SIGGRAPH 2013.
- *Audio2Face-3D: Audio-driven Realistic Facial Animation*, NVIDIA,
  arXiv:2508.16401, 2025.
- *A survey on the pipeline evolution of facial capture and tracking
  for digital humans*, Springer Multimedia Systems, 2023.
