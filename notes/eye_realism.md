# Eye realism — anatomy notes and design

Reference notes that drive the eye-realism cycles. Numbers come from
standard ophthalmic anatomy; tuning here is in head-length-relative units
because the procedural skull is parametric.

## Anatomy quick-reference

Adult eyeball is ~24 mm in diameter and very nearly spherical. The
**cornea** is a more sharply-curved cap (~7.8 mm radius vs ~12 mm for the
eye proper) that bulges visibly in front of the sclera by roughly 2–3 mm.
This bulge is what catches highlights and is the single biggest cue that
an eye is a 3D ball, not a printed disc.

The **iris** sits behind the cornea, ~12 mm across. Because it is viewed
through the cornea (a positive lens) it appears slightly magnified and
recessed; the colored stroma has visible radial fibers and a darker
**limbal ring** at its outer edge (collagen condensation at the
corneoscleral junction). The pupil is centered in the iris and varies
2–8 mm with light; under typical indoor light it is ~3–4 mm, i.e. ~30%
of iris diameter. Crucially, the pupil and iris boundary are concentric.

The **sclera** is not pure white. Under typical light it reads as a warm
off-white near the limbus (capillary tinting toward the canthi) and
slightly cooler near the apex. Painting it pure white reads as plastic.

## Gaze, convergence, strabismus

Both eyes' optical axes converge on the same focal point. For a distant
target the axes are essentially parallel; for a target at ~50 cm
(camera distance for a portrait) the convergence angle is roughly
6–7°, so each eye is rotated inward by about half that. If we render
the irises pointing strictly along the head's local +Z, the eyes read
as a "thousand-yard stare" — the brain registers it as either
disengagement or, when slight asymmetry creeps in via skull
asymmetry, exotropia (wall-eye).

A **strabismus failure mode** in this codebase is therefore not a
fixed offset between L and R — both eye primitives are placed
symmetrically about X=0 — but rather a *parallel* gaze that the viewer
expects to be slightly convergent. Fix: shift each iris+pupil inward
(toward x=0) by a small amount calibrated for the typical viewer
distance (~50 cm).

## Implementation plan (this branch)

Cycle 1 — geometry + alignment:
- Make the eye white a proper sphere (equal radii).
- Add a corneal bulge: a small spherical cap protruding forward over
  each iris.
- Toe-in the iris and pupil x-coordinates so both eyes converge on a
  focal point ~50 cm in front of the head, eliminating the "wall-eye"
  reading.

Cycle 2 — iris / pupil / sclera:
- Add a **limbal ring**: a thin dark annulus just outside the colored
  iris disc, drawn behind the cornea cap so it reads as the corneal
  edge rather than a painted line.
- Recess the iris slightly (concave bowl) so the pupil sits at the
  base of a visible depression rather than on a flat face.
- Shrink the pupil to ~30% of iris diameter (was ~40%) and keep it
  exactly concentric.
- Introduce a small **sclera tonal variation** between the two eye
  primitives — both within anatomically-plausible range so the
  characters don't turn into a population of mismatched-eye faces.

Cycle 3 — gaze stability across animation:
- Verify across walk and wave that the iris/pupil track the head and
  remain converged. Eyes are rigidly skinned to the head bone so
  alignment should hold; the cycle is a validation pass with
  per-frame screenshots, not a re-rig.

## Capture matrix

`tools/face_capture.py --matrix=full` already covers grid (seed × gender
× angle, T-pose) and presets (age × gender × expression × pose × angle).
For eye work we save under `screenshots/eye_baseline/` then
`screenshots/eye_cycle{1,2,3}/` and `screenshots/eye_final/`, and add
extra walk-cycle frames so we can scrub for drift.
