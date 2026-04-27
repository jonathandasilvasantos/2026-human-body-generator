# Triview enhancement cycle — procedural vs. photoreal vs. image-to-3D

A reusable A/B/C loop that compares our procedural human against an
external "ground-truth" photoreal target *and* a real image-to-3D mesh
reconstruction, then drives prioritized procedural enhancements.

## Pipeline

1. **`tools.triview_capture`** — render one fixed-seed character from
   four locked viewpoints (front, left, back, 3/4 isometric) at the
   same true T-pose (`skeleton.t_pose_wide`), framing, FOV, and
   lighting profile (default Studio Beauty).
   Outputs `_front.png`, `_left.png`, `_back.png`, `_iso.png`,
   `_grid.png`.
2. **`scripts/flux_kontext_remote.py`** — local SSH wrapper that
   ships each procedural PNG to bender and runs FLUX.1 Kontext [dev]
   (`black-forest-labs/FLUX.1-Kontext-dev`) via diffusers
   `FluxKontextPipeline` with bf16 + `enable_model_cpu_offload()`
   in conda env `flux` (separate from `hy3d`). Same "transform into
   a photorealistic person, keep pose/framing/clothing" instruction;
   produces the photoreal target views. Replaces the previous
   Gemini / Nano Banana Pro stage to eliminate per-cycle API spend.
   Inference script: `remote/flux_kontext_infer.py`. One-time
   install: `remote/flux_kontext_setup.sh`.
3. **Hunyuan3D-2mv on bender** — `tencent/Hunyuan3D-2mv` takes
   exactly `{front, left, back}` photoreal images and emits a
   single textured `.glb`. Drives via
   `/home/bender/projects/hunyuan3d/run_image_to_3d.py`. Local
   single-file install at `/home/bender/projects/hunyuan3d/`,
   conda env `hy3d`, RTX 3090 Ti, end-to-end ~1 min after model
   weights are cached.
4. **`tools.glb_render` (run on bender)** — renders the .glb from
   the same four viewpoints using `pyrender` + EGL. Output
   `_hy3d_{front,left,back,iso}.png`.
5. **`tools.triview_compare`** — assembles a 3-row × 4-col grid:
   procedural | photoreal | hunyuan3d-rendered.

## Cycle 46 → 53 pass — fitted clothing and hand/face proportion polish

This pass used the existing `c44` comparison as the baseline, then
iterated fixed-seed female renders through `c53`. Full external
photoreal + Hunyuan3D-2mv checkpoints were run for `c46`, `c52`,
and `c53`; the intermediate `c47`-`c51` steps were local geometry
checks to keep the edit loop tight before spending another remote
GPU/API cycle.

Artifacts:

- `compare/c46_compare.png` — first closed checkpoint for this pass.
- `compare/c52_compare.png` — post torso-shirt migration checkpoint;
  exposed the hip/crotch skin gaps from dropping the pants pelvis
  capsule.
- `compare/c53_compare.png` — final closed checkpoint for this pass.
- `screenshots/triview/c53.glb` and
  `screenshots/triview/c53_hy3d_{front,left,back,iso}.png` — final
  Hunyuan3D-2mv reconstruction and same-view renders.

Shipped changes:

| Cycle | Fix | Where | Result |
|-------|-----|-------|--------|
| 46 | Reduce eye size / push eyes deeper; split hand from fused block into palm, four fingers, thumb, and knuckle pad; reduce render-time fabric bend inflate. | `human/mesh.py`, `tools/triview_capture.py` | Hands read as fingers at body distance instead of one mitten. T-shirt loses the puffer-shell look. |
| 47 | Remove neck from tight top bone sets; shrink shoulder garment yoke. | `human/garments.py`, `human/mesh.py` | Collar no longer builds a high padded neck ring; shoulder cloth sits closer to the body. |
| 48 | Build tight tops from a cropped copy of `body_loft.build_torso_loft` rather than old torso capsules. | `human/garments.py` | Shirt follows the hourglass waist/hip morphology instead of covering it with a cylinder. |
| 49 | Remove pelvis capsule from pants/shorts. | `human/garments.py` | Jeans stop reading as a blue barrel around the waist, but this exposed skin gaps. |
| 50-52 | Retune the shirt crop ring; polish medium-hair cap/fringe; shrink ears. | `human/garments.py`, `human/accessories.py`, `human/mesh.py` | Shirt hem overlaps jeans more cleanly; hair blocks less of the face; ears are less oversized. |
| 53 | Add a lower torso-loft hip shell for pants/shorts and pass `Shape` into `build_bottom`. | `human/garments.py`, `human/character.py` | Closes the hip/crotch skin gaps without restoring the old bulky pelvis capsule. |

Current read of `compare/c53_compare.png`:

- Stronger: waist/hip silhouette, shirt fit, pants waistband coverage,
  hand silhouette, ears/eyes scale, and reduced cloth puff.
- Still weak: primitive jeans crotch and leg merge, capsule arms/legs,
  toy-like procedural face/hair geometry at close crop, and Hunyuan3D
  background-plane fragments. The background fragments are Stage C
  artifacts, not a procedural-mesh defect.

Next highest-payoff geometry work:

1. Replace thigh/shin pants capsules with a continuous leg garment loft
   so the jeans have an inseam and clean crotch transition.
2. Fix `body_loft.build_limb_loft` bind-space math and enable limb
   lofts for arms/legs, then layer clothing on those surfaces.
3. Reframe/repair face close-up capture and port hair/face primitives
   toward a continuous face/hair mesh; current close-up panels are
   useful for detecting scale errors but not for photoreal comparison.

## Cycle 3 artefacts (current)

Seed 42 / female / `t_pose_wide`:

- `screenshots/triview/c3_{front,left,back,iso}.png` — procedural
- `screenshots/triview/c3_{front,left,back}_photoreal.png` — Nano
  Banana Pro
- `screenshots/triview/c3.glb` — Hunyuan3D-2mv reconstruction (14 MB)
- `screenshots/triview/c3_hy3d_{front,left,back,iso}.png` —
  reconstruction rendered from the same views
- `screenshots/triview/c3_compare.png` — 3-row × 4-col side-by-side

## Gap analysis — procedural vs. Hunyuan3D mesh

Reading `c3_compare.png`, comparing our row 1 against row 3:

| Gap | Severity | Where it shows |
|-----|----------|----------------|
| **Arm length truncated**: Hunyuan3D T-pose arms reach a full body-width to either side; ours stop ~70% of the way. The shoulder-to-fingertip span is short. | High | front, back |
| **Hands disappear into blobs**: Hunyuan3D shows palm + finger silhouette with knuckle break at wrist; ours is a single capsule. | High | front, back, iso |
| **Torso is a uniform tube**: Hunyuan3D shows clear waist taper, hip flare, soft bust definition; ours is the same radius from chest to pelvis. | High | front, back |
| **Shoulders are squared off**: Hunyuan3D has a sloped trapezius flowing from neck into arm; ours has a hard 90° shoulder cap with the arm socketed in. | High | front, back, iso |
| **Hair reads as bald** at front yaw=0: Hunyuan3D shows long hair past the shoulders; ours has either no hair geometry or hair coplanar with skull and back-faced from this angle. | High | front |
| **Shoes are cylindrical**: Hunyuan3D has a sneaker silhouette with toe-cap and sole; ours is two black tubes. | Medium | front, back, iso |
| **Pants don't taper to the ankle**: Hunyuan3D pant cuff narrows over the shoe; ours stays full-width and clips into the foot blob. | Medium | front, back |
| **Shirt drapes thick**: Hunyuan3D shirt has neckline + cap-sleeve cuffs; ours has the puffer-jacket inflate (`_bend_inflate=0.025` on a `tshirt`). | Medium | front, back |
| **Head reads small** at body framing — Hunyuan3D head proportion is roughly 1/7.5 of body, ours is ~1/9. | Low | all |
| **Background plane fragments in the Hunyuan3D mesh** (sheet artefacts behind the figure) are a Hunyuan3D-side quirk. Not actionable here, but mask before publishing comparison panels. | n/a | rows 3 panels |

## Cycle 4 — what shipped

Implemented in this branch:

| # | Fix | Where | Effect on `compare/c4_compare.png` |
|---|-----|-------|-----------------------------------|
| 1 | **Tighter `top_inflate` / `bottom_inflate` ranges** — `top` 0.014–0.020 → 0.008–0.013, `bottom` 0.008–0.013 → 0.005–0.010. | `human/character.py` random_appearance + reroll_clothes | Visible: t-shirt no longer reads as a puffer; jeans hug the leg. |
| 2 | **Medium-hair drape** — added a short rear-side ellipsoid past the temples so the hairstyle isn't just a scalp cap. | `human/accessories.py::build_hair` (`elif style == "medium"`) | Visible from back / iso. **Front view still reads bald** — the drape doesn't wrap forward yet (deferred to cycle 5). |
| 3 | **Sneaker toe-cap + sole plate** — added two ellipsoids per foot, skinned to the foot bone. | `human/garments.py::build_shoes` | Clear win at left / iso views: feet now read as shoes, not cylinders. |
| 4 | **Stronger female waist cinch** — random range 0.78–0.88 → 0.70–0.80. | `human/skeleton.py::random_shape` | Spine narrower, but the chest cloth still covers the waist seam, so the *cloth* silhouette change is small. Visible on bare-skin gallery shots, less so in the c4 triptych. |
| 5 | **T-pose flat hands** — N/A in this rig. | — | The hand is a single capsule (no finger bones), so there's nothing to flatten. T-pose-wide already gives the right wrist orientation. Marked done by virtue of being out-of-scope. |
| 6 | **Trapezius slope** — small ellipsoid bridging chest top into the deltoid bilaterally, skinned to chest. | `human/mesh.py::build` (after deltoid loop) | Subtle at thumbnail size; reads better on bare-skin renders. |
| 7 | **Pant-cuff tuck** — `bottom_length_scale` random range 0.98–1.04 → 0.94–0.99 so the cuff narrows over the shoe instead of clipping past it. | `human/character.py` | Clear win at all four views — the trouser leg now meets the shoe shaft cleanly. |
| 8 | **Vertical-gradient backdrop + chroma-keyed composite** — render still clears to flat then is composited over a soft top-light → bottom-grey gradient before save. | `tools/triview_capture.py` (`_vertical_gradient_bg`, `_composite_over_gradient`) | All four panels now sit on the same gradient as the photoreal target — keeps Stage B / Stage C closer to their training distribution and helps the eye separate figure from background. |

Side-effect from environment churn during cycle 4: PyOpenGL was bumped
when `pip install pyrender pyglet` ran. The triview script then
errored at `glGenTextures` with a numpy-2.x / ctypes incompat. Fix was
`pip install --upgrade PyOpenGL` (3.1.0 → latest). Worth noting in
case the same pip combination shows up in CI.

## Cycle 5 — what shipped

| # | Fix | Where | Effect on `compare/c5_compare.png` |
|---|-----|-------|-----------------------------------|
| A | **Iso framing fix** — pitch 0.30 → 0.10, dist 2.85 → 3.10 so the head doesn't crop and the iso panel shows the whole figure at scale comparable to the orthographic views. | `tools/triview_capture.py::VIEWS` | Iso panel is now usable; cycle 4 had the head clipping at the top edge. |
| B | **Shorter top length** — `top_length_scale` random range 1.00–1.05 → 0.92–0.98 so the t-shirt ends at the upper waist instead of covering the whole spine; the spine's narrower (`bulk * waist`) capsule now contributes to the silhouette. | `human/character.py` (random_appearance + reroll_clothes) | The waist seam is visible across all four views; the body silhouette finally reads as hourglass-ish. |
| C | **Medium-hair drape wrapped further forward** — drape ellipsoid centred near z=0 (was pushed back to -head_d * 0.50), wider X (1.18 vs 1.10 of scalp_rx), positive z extent so it surrounds the head instead of falling only behind. | `human/accessories.py::build_hair` (medium branch) | Visible improvement at left/back; **front view still reads as bald** — the drape is wider laterally but doesn't yet reach forward of the ear line. Carry to cycle 6. |

## Cycles 11 → 13 — variety + face attention

- **Cycle 11**: switched seed to 7, gender male — buzz cut, mustache,
  teal t-shirt, grey pants. The lofted torso correctly produced a male
  V-shape silhouette; no female-only morphs (bust) leaked. Confirms
  the loft is gender-clean.
- **Cycle 12**: added face close-up viewpoints
  (`FACE_VIEWS = face_front / face_3q / face_left / face_iso`) to
  `tools/triview_capture.py` and a fifth row to
  `tools/triview_compare.py` so face quality has its own diff signal.
  Diagnosed exact face issues in `compare/c12_compare.png`: clown-ball
  nose, pinhole eyes, no brow geometry, flat skin.
- **Cycle 13**: phase F1 of the **face-mesh migration**
  (`notes/face_mesh_migration_plan.md`). New module
  `human/face_mesh.py` builds a 56×56 ring head shell with **carved-in
  facial landmarks** — eye-orbit recession, nose ridge protrusion,
  mouth puff, cheekbone band, jaw width, brow prominence — driven
  by per-vertex Gaussian fall-offs from the same `Shape` knobs the
  rig already uses. Front-view face close-up reads with proper
  topology (orbits, ridge, mouth ring, brow line). Disabled by
  default (`Shape.use_face_mesh = False`) because the existing eye /
  nose / lip primitives still sit on top with their old positions —
  the 3/4 view shows misaligned features until we port the
  primitives onto the new landmark math (phases F2-F3).

## Cycles 14 → 15 — phase F1 polish (face mesh primitives port)

- **Cycle 14**: reduced face-mesh orbit recession from `head_d * 0.22`
  to `0.06` and the carved-nose tip from `0.18` to `0.10`. Lower
  carving amplitude so the dense mesh cohabits with the existing
  primitive add-ons without stacking double-bumps in 3/4 view.
- **Cycle 15**: disabled the redundant primitives when
  `use_face_mesh = True`:
  - `_head_compound`'s nose primitives (bridge / tip / wings /
    columella) — the dense mesh now carries the nose silhouette.
  - `build_age_detail` and `build_expression_folds` — the flat
    patches were tuned for the old skull surface and read as dark
    spots on the new shell.
  - `build_chin_detail` — same reason.
  Re-enabled `Shape.use_face_mesh = True` as the new default.

What's still on top of the dense mesh: eye stack (sclera/iris/etc),
eyebrow patches, lip primitives, ear compound, mustache/beard,
hair. Phase F2 ports each onto the new mesh's vertex landmarks.

## Cycle 16 — phase F2 (eye + brow primitive port)

- Eye sphere offset: `_eye_metrics_for_shape` now pulls the eye
  sphere back by `head_d * 0.06` when `Shape.use_face_mesh = True`,
  matching the dense mesh's orbit recession depth so iris / sclera
  nest inside the carved orbit instead of bulging out.
- Brow ridge primitives (`_head_compound`'s
  bilateral supraorbital ellipsoids) skipped when `use_face_mesh`
  is on — the dense mesh already carries `brow_prominence` as a
  surface morph.

Carryover for future cycles:
- Skin spots / shading artifacts on the dense face mesh remain.
  These are NOT from disabled overlays (verified by skipping all
  age + chin + expression + brow primitives) — they're shading
  artifacts where the carved morph regions create normal
  discontinuities. Phase F3 needs either smoother normal
  recomputation (Gouraud rather than per-tri accumulate) or
  region-blended morph weighting so the carved transitions are
  C1-continuous.
- Lips / facial hair / ear primitives still positioned by old
  skull math. Acceptable from front view; 3/4 view alignment will
  improve when these are ported in F3.

## Cycle 17 — Laplacian normal smoothing on the face mesh

The "dark spots" diagnosed in cycle 16 were not from any disabled
overlay — they were shading artifacts at the boundaries of the
carved morph regions (eye orbit, nose ridge, cheekbone, brow).
Where the Gaussian morphs overlap or transition the per-vertex
displacements give neighbouring triangles different normals;
under directional lighting that reads as patchy shading.

Cycle 17 adds a 3-pass grid Laplacian over the (ring × radial)
normal grid in `face_mesh.build_face_mesh`. Each vertex's normal
is averaged with its 4 grid neighbours (rings clamped at the
poles, segments wrapped). 3 passes is enough to soften the
transitions without flattening the carved features themselves.

Effect on `compare/c17_compare.png` row 4: face surface is
visibly cleaner; the carved orbits / nose ridge / mouth puff
read as anatomical without the dark patches that made the
character look weathered.

## Cycle 18 — Phase F2 polish (lip + eye depth alignment)

- Orbit recess depth tightened from `0.06 * head_d` to `0.035 * head_d`
  in `face_mesh.py`, with the matching offset in
  `_eye_metrics_for_shape`. The far eye in the 3/4 view is no longer
  buried in its socket.
- `build_lips` now translates forward by `0.05 * head_d` when
  `use_face_mesh=True` so the lip ring sits on top of the carved
  mouth puff instead of behind it.

`compare/c18_compare.png` row 4 shows both orbits with eye stacks
nested cleanly, mustache + lips on the carved bumps, and the same
smooth face surface from cycle 17. With cycles 13 → 18 the face
mesh has a working integration with all of: eye stack, eyebrow
patches, lips ring, ears, mustache, hair drape.

## Cycle 19 — face-mesh on female + Phase Body 2 attempt

- Confirmed the dense face mesh + body loft work for `seed=42 female`:
  the gallery hourglass silhouette is preserved (cycle 10 fixes
  carry forward), and the carved face features show through the
  hair drape from the front view.
- **Phase Body 2 limb loft (`body_loft.build_limb_loft`)**: code
  written but the multi-bone chain produces flat plates instead of
  oriented limbs. Root cause: vertices are generated in the FIRST
  bone's local frame with `y_end = length * sign_y`, but the
  matching `align_y_to(tip)` rotation already encodes the bone
  direction — the sign multiplier double-counts and the geometry
  ends up in a different plane. The limb-loft dispatch is gated
  off in `mesh.py::build` until the bind-pose math gets the same
  treatment the torso loft already has. Limbs continue to use
  `_profiled_capsule` with anatomical muscle-bulge profiles.

## Cycle 9 — phase 1 of dense-mesh migration (torso loft)

`human/body_loft.py` now emits a single continuous lofted torso (24
side rings × 32 segments + bottom/top fan caps, ~1024 verts) replacing
the chest+spine+pelvis capsule trio. `Shape.use_loft = True` is the
default; the capsule path is bypassed for those three bones.

What changed visually in `compare/c9_compare.png` row 1:

- **Hourglass silhouette appears.** Wider chest, narrower waist (X
  pinched by `waist`, Z by aspect 0.65), wider hips (X stretched by
  `hip_w`). The cycle-4 cylinder torso is gone.
- **Bust is baked into the loft.** The bust ellipsoid fires only on
  the front rings of the chest section, peaks around t = 0.55, and
  the projection is proportional to `bust * bust_proj * chest_r`.
  Previously this was a separate ellipsoid floating on top.
- **Trapezius slope reads naturally.** The chest top ring stretches
  X by `shoulder_w` while staying narrow in Z, so the silhouette
  fairs into the deltoid without a hard 90° cap.
- **Cross-section is no longer circular.** `ASPECT_Z = 0.65` makes
  the torso flatter front-to-back than side-to-side — humans seen
  from above are an oval, not a circle.

**Carryover defects** (next cycle):

1. **Garments still inflate from the bone radius**, not from the new
   torso surface, so the shirt sits on top of an *old* cylinder that
   no longer matches the body. The shirt-pant gap shows skin in the
   front view. Phase 5 of the migration plan addresses this:
   `build_top` / `build_bottom` re-anchored to the body mesh's
   surface vertices, with `inflate` becoming a per-vertex outward
   push along the mesh normal.
2. **Pelvis/thigh seam.** The loft's bottom cap closes at the
   pelvis bottom; the thighs are still capsules. There's a visible
   skin gap at the crotch (white triangle) where the cap doesn't
   meet the thigh capsule cleanly. Phase 2 (limb loft + thigh weld)
   stitches them.
3. **No shape morphs yet.** `bulk` and `lower_bulk` aren't in the
   loft's morph set; the chest/waist still uses the bone radius
   directly. Easy to add as a per-ring radial scale.

## Cycles 6 → 8 — what shipped (hair + waist + trapezius polish)

| Cycle | Fix | Where | Effect |
|-------|-----|-------|--------|
| 6 | Wider, full-ellipsoid medium-hair drape (was a back-only shell). | `human/accessories.py::build_hair` (medium) | Hair finally reads from the front, but the first version extended down to the chest. |
| 6 | Drape size first cut → dressed back to shoulder length. | same | Bottom of drape pulled up to mid-shoulder rather than rib cage. |
| 7 | Drape lifted higher; cy raised to length·0.30, ry trimmed to 0.42. | same | Hair frames the face; was still encroaching past the chin line. |
| 8 | Drape cy raised to length·0.42 — bottom now sits at the jawline. | same | Hair no longer reads as a beard. |
| 8 | Trapezius ellipsoid lowered + narrowed (axes 0.55/0.30/0.62 → 0.42/0.20/0.50, cy 0.95 → 0.86, slight z=-0.05 tuck). | `human/mesh.py::build` | Trap bridge now fairs into the existing deltoid instead of perching on top of the shoulder as a visible blob. |
| 8 | `top_length_scale` random range 0.92-0.98 → 0.96-1.00. | `human/character.py` | Closes the skin-gap above the waistband while still letting the spine taper contribute to the silhouette. |

Iso framing fix (cycle 5: pitch 0.30 → 0.10, dist 2.85 → 3.10) carried
forward unchanged.

## Where the loop is now

`compare/c8_compare.png` shows row 1 (procedural) is significantly
closer to row 2 (FLUX Kontext) and row 3 (Hunyuan3D-2mv) at body-
framing distance:

- silhouette no longer reads as bald + puffer + cylinder-feet,
- shoes have shape, pants tuck into them, shirt sits at the hip,
- hair frames the face from the front view,
- trapezius is sloped, not capped,
- background gradient matches the photoreal target.

What's left would be **structural**, not parametric:

1. **Cap rig → continuous mesh.** The torso is currently three stacked
   capsules (chest, spine, pelvis) with single radii. To get the
   cloth to follow a proper hourglass silhouette, we'd either need
   per-bone head/tip radii (so a capsule can taper) or a continuous
   torso mesh in the SMPL/MakeHuman style. That's a body-builder
   rewrite, not a knob change.
2. **Hand topology.** The hand is one capsule. To match the way
   Hunyuan3D resolves the wrist + palm + finger split, we'd need
   actual finger bones + a MANO-style compound hand. The mesh
   skeleton has hooks for that (the `_SHAPE_RULES` table is
   declarative) but the corresponding mesh code in `human/mesh.py`
   would have to grow a hand-builder.
3. **Subdivision-driven cloth drape.** Garment shells inflate from
   the underlying bone radius and don't simulate gravity. Long hair,
   skirts, sleeves all benefit; cycle 4-8 fixes are silhouette-
   approximate. A simple gravity post-pass on garment vertices would
   close most of the remaining "stiff cloth" gap.

Hitting these three would close most of what remains between row 1
and row 3 of the comparison grids; further parameter tweaks past
cycle 8 hit diminishing returns.

Ranked by visual payoff per line of code. The first three are the
same single-character/single-render fixes that close most of the gap.

1. **Disable cloth inflation when garment is `tshirt` / `tank` / `longsleeve`.**
   Drop `_bend_inflate` to 0 for tight tops; keep it for jackets and
   dresses. One-line fix in the draw loops in
   `tools/triview_capture.py`, `human/capture.py`, `human/viewer.py`.
2. **Investigate front-yaw hair culling.**
   Suspect: `glCullFace(GL_BACK)` against a thin 2-sided hair shell, or
   normals flipped on the frontal hair patch. Render seed 42 with
   `--pose t_pose_wide --light 1 --yaw 0` and verify `medium` hair
   reads.
3. **Shoe primitive**: replace foot capsule with a stretched ellipsoid
   (longer in +Z, slightly wider near toe) plus a darker sole-plate
   ellipsoid at y≈ground. ~30 lines in `human/garments.py`.
4. **Torso silhouette taper**: interpolate spine/chest/waist capsule
   radii. Add `Shape.waist` parameter (gender-conditioned default
   ~0.85 of chest); pinch L3-L4 laterally so the front silhouette
   shows a waist line.
5. **Hand pose at T-pose**: rotate finger bones flat with slight
   abduction so palms read as flat plates with fingers spread —
   this is what image-to-3D models prefer too.
6. **Trapezius slope at shoulder**: add a small triangle/capsule from
   neck base to deltoid that rises ~12° above the line shoulders-down.
   Also lower the shoulder-arm joint by ~0.05 head heights so the arm
   reads as inserted *into* the shoulder instead of clipped on.
7. **Pant ankle cuff**: taper the shin capsule's bottom radius by
   ~30% so the trouser hugs the calf on its way to the shoe.
8. **Contact shadow disc + vertical-gradient backdrop** in
   `triview_capture` (offscreen ground polygon + projected shadow).
   Cheap visual win; also mirrors what Hunyuan3D's training data
   looks like, which makes the *next* round of inputs more natural.

## Reproduce a cycle

```bash
# 1. render
env/bin/python -m tools.triview_capture --seed 42 --gender female \
    --out screenshots/triview/cN

# 2. photoreal each *required* view (front/left/back; iso optional)
for v in front left back; do
  env/bin/python scripts/flux_kontext_remote.py \
    --image screenshots/triview/cN_${v}.png \
    --output screenshots/triview/cN_${v}_photoreal.png \
    --prompt "Transform into photorealistic person, same pose/framing, ..."
done

# 3. ship to bender + run Hunyuan3D-2mv
scp screenshots/triview/cN_{front,left,back}_photoreal.png \
    bender@192.168.15.99:/home/bender/projects/hunyuan3d/
ssh bender@192.168.15.99 'source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate hy3d && cd /home/bender/projects/hunyuan3d && \
    PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    python run_image_to_3d.py \
        --front cN_front_photoreal.png \
        --left  cN_left_photoreal.png \
        --back  cN_back_photoreal.png \
        --out-glb cN.glb && \
    PYOPENGL_PLATFORM=egl python glb_render.py --glb cN.glb --out cN_hy3d'

# 4. fetch back
scp 'bender@192.168.15.99:/home/bender/projects/hunyuan3d/cN.glb' \
    'bender@192.168.15.99:/home/bender/projects/hunyuan3d/cN_hy3d_*.png' \
    screenshots/triview/

# 5. side-by-side
env/bin/python -m tools.triview_compare \
    --in screenshots/triview/cN \
    --out screenshots/triview/cN_compare.png
```

## References

- **Hunyuan3D-2mv** model card —
  https://huggingface.co/tencent/Hunyuan3D-2mv (multiview
  finetune of Hunyuan3D-2; expects exactly `{front,left,back}`).
- **Hunyuan3D-2** repo — https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  (`Hunyuan3DDiTFlowMatchingPipeline`, custom_rasterizer +
  differentiable_renderer CUDA extensions).
- **FLUX.1 Kontext [dev]** —
  https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
  (image-edit model, ~12B params). Runs on bender in conda env
  `flux` via diffusers `FluxKontextPipeline` with bf16 +
  `enable_model_cpu_offload()` to fit the 3090 Ti's 24 GB. License
  is FLUX.1 [dev] Non-Commercial.
- **PixelArtistry walkthrough of Trellis 2** —
  https://www.youtube.com/watch?v=FuFm8zBHDWI (we evaluated this
  first; switched to Hunyuan3D-2mv since it natively accepts our
  3-view input and runs on Linux + a single 24 GB card).
