# Face system — expressions, age, and asymmetry

Ported from `human-chat` (branch `realistic-human-heads`) on top of the
`_skull_shell` parametric head introduced in commit 9ff91a5 and the
sex-dimorphism pass from commit 4dabad6. Unification landed on branch
`unify/face-system`.

## Expressions

Driven by `Appearance.expression ∈ {neutral, smile, frown, squint, surprised}`.
The expression parameter threads through `build_eyelids(bones, shape, expr)`
and `build_lips(bones, shape, expr)` in `human/mesh.py`.

- **smile**   → mouth corners lifted (+Y ≈ 0.014·head_length).
- **frown**   → mouth corners lowered (−Y ≈ 0.012·head_length).
- **squint**  → eyelid aperture × 0.62 (lids meet closer to the pupil).
- **surprised** → eyelid aperture × 1.22 and a vertical mouth_open that
  slides the upper lip up and the lower lip down by ~0.014·head_length.
- **neutral** → baseline geometry.

The cupid's-bow upper lip (two lobes) and fuller lower-lip pillow from the
`realistic-faces` cycle are preserved; expression only shifts center
positions, it does not reshape the primitives.

## Age groups

Driven by `Appearance.age_group ∈ {young, adult, elder}`. Consumed by
`build_age_detail(bones, shape, age_group)`.

- **young**  → age-detail mesh is empty; skin reads smooth.
- **adult**  → creases rendered at 0.45 × strength.
- **elder**  → creases rendered at 1.0 × strength.

Creases are thin flat patches laid over the skull surface: three horizontal
forehead lines above the brow band, plus mirrored crows'-feet at the outer
eye corners and nasolabial folds from the nose wings toward the mouth.
They are drawn darker than the skin tone (×0.55) so they read as folds
rather than decals.

## Face asymmetry

`Shape.face_asymmetry` ∈ [0, ~0.02] applies a per-vertex side-dependent
radial multiplier in `_skull_shell`. Peaks around the face band (t ~ 0.1,
cheekbone/eye line) and fades toward chin and crown so the silhouette stays
plausible. `random_shape()` samples a small default in [0.004, 0.015]; the
effect is subliminal at front view and visible at three-quarter.

## Lip fullness

`Shape.lip_fullness` scales the upper and lower lip y/z radii. Sampled in
`random_shape()` with a mild gender skew (female [0.95, 1.20], male
[0.80, 1.05]).

## Test matrix

`tools/face_capture.py` exposes three modes:

- `--matrix=grid`      — seeds × genders × 4 angles on T-pose, face framed
                         tight. Matches the original `face_final` audit.
- `--matrix=presets`   — five curated demographic presets (young-female-
                         smile, adult-male-neutral, elder-female-squint,
                         adult-male-surprised, adult-female-frown), each
                         rendered under static / walk / wave × front /
                         three-quarter / profile.
- `--matrix=full`      — grid followed by presets in one run.

## What was NOT ported

- `_anatomical_head_surface` (ring-deformation skull) — replaced by
  `_skull_shell`.
- Flat-decal eyes — replaced by spherical eyes seated in sockets.
- Gender-agnostic `_limb_profile` — kept the sex-dimorphic version.
- The teardrop-nose / stripe-mouth primitives.

## Known gaps

- Age creases are currently flat patches pinned to the skin surface. At
  three-quarter angles the patches can read as stuck-on plates. A future
  cycle should normal-offset them along the skull normal.
- The face remains a primitive-based procedural rig rather than a continuous
  FLAME-style topology, so profile views still show stylized simplifications.
  The current realism pass reduces the most visible caricature cues while
  preserving the ARKit-52/FACS interface.

## Realism / animation compatibility cycle

Branch `feature/face-realism-animation-compat` focused on improving rest
proportions without changing the facial animation standard:

- Added conservative identity traits to `Shape` (`eye_sep`, `eye_size`,
  `eye_depth`, `nose_width`, `nose_proj`, `nose_bridge`, `cheekbone`,
  `jaw_width`, `chin_proj`, `brow_prominence`) so gender, skin tone and face
  structure are not tied to stereotypes.
- Rebalanced landmarks and skull profile to reduce caricature: less pinched
  jaw, subtler chin projection, smaller nose primitives, smaller/deeper eyes,
  finer eyebrows and ears.
- Routed appearance age into the structural head pass, keeping age detail
  compatible with the existing `Appearance.age_group` field.
- Removed front-facing hairline cards that read as stickers in close-up and
  moved long-hair drape behind the skull to avoid eye/cheek intersections.
- Added skin-colored eye socket/canthus cover patches and a dark oral aperture
  behind the lips. These are driven by the same ARKit weights and do not alter
  the channel API.

Validation captures:

- `screenshots/face_realism_baseline/` — pre-change static baseline.
- `screenshots/face_realism_final_static/` — final static grid and demographic
  preset matrix.
- `screenshots/face_realism_final_anim/` — final ARKit/FACS presets, clips,
  channel isolation, asymmetry, cross-character and walk+face validation.
