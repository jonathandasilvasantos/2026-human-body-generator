# Migration plan — capsule rig → continuous parametric mesh

The triview-enhancement loop (cycles 3 → 8) hit a ceiling that's no
longer parametric: the torso is three stacked capsules with single
radii, the hand is one capsule, the cloth shells inflate from those
capsules. To move row 1 of `compare/cN_compare.png` materially closer
to row 3 (Hunyuan3D-2mv), the body has to become **a single skinned
mesh whose vertices respond to the same `Shape` knobs we already
have**, while preserving every other axis of variation we ship today
(gender, age, hair, beards, mustaches, garments, ethnic skin tones,
expression, BVH retargeting, lip sync).

This document is a planning + research document, not a refactor. No
code is changed by it.

## What we keep (constraints that are non-negotiable)

- **Procedural at startup.** No external `.obj` / `.fbx` / `.npz`
  asset files. The README's first claim is "no external character
  assets" — the migration must keep that.
- **`Shape` and `Appearance`** as the public API. `Shape` already
  has 18 knobs (height, bulk, hip_w, shoulder_w, bust, waist,
  head_size, leg_len, torso_len, lower_bulk, q_angle, neck_thick,
  bust_proj, jaw_width, cheekbone, etc). Anything we add to drive
  the mesh has to live here, with gender-conditioned ranges in
  `random_shape()`.
- **The 19-bone skeleton** in `human/skeleton.py`, including the
  current parent / head / tip / radius schema and `apply_shape`.
  Anything that changes the bone count breaks BVH retargeting,
  joint limits, walk cycle, IK, ARKit-52 face rig — too much
  surface area.
- **GLFW + GL 3.3 core profile.** No requirement on EGL, GLES,
  WebGL, Vulkan. Up to 32 bone matrices in the skin shader stays.
- **Existing fragment shaders** for skin (mode 0), fabric (1),
  hair (2), eye (3), shoe (4). The new mesh continues to write to
  mode 0 with the same biophysical skin model.
- **Garments and accessories** continue to layer on top of the
  body. They can read the new body topology when useful but must
  not require it (a tank top should still build without the body
  being there).

## What we explicitly are not building

- **Learned blend shapes.** SMPL/SMPL-X get their realism from
  PCA-fit shape blendshapes and pose-corrective blendshapes
  trained on thousands of 3D scans (CAESAR / 4DHumans). We have no
  scan data and no plan to license SMPL's `.npz` files. The
  migration target is "SMPL-shaped" — same topology / skinning /
  per-vertex deformation idea — but with **analytical morphs**
  written in Python that consume `Shape`, not learned ones.
- **A new face stack.** `human/mesh.py::_head_compound`,
  `face_anim.py`, the eye stack, ARKit-52 rig, Cohen-Massaro
  visemes — all stay. The new body mesh terminates at the neck
  base; the head compound continues to be its own thing skinned
  to the head bone.
- **A garment-pattern system.** Cloth shells stay; we are not
  introducing GarmentCode-style sewn patterns in this migration.

## Reference points (research)

- **MakeHuman / Homunculus** — the canonical example of "one
  fixed-topology base mesh, deformed by slider-driven morph
  targets, kept watertight as the slider values change." 14,766
  quads, all-quad, no triangles. Targets are baked vertex-delta
  files. We borrow the *idea* (single template + analytical
  morphs) without the asset library.
- **SMPL** — single mesh (~6,890 verts), 24 joints, LBS, learned
  shape betas (10) and pose correctives. Our `Shape` plays the
  role of betas; pose correctives we skip until phase 4.
- **SMPL-X** — extends SMPL with face / hands / eyes (~10,475
  verts, 54 joints). Useful for the **hand topology** that we'll
  borrow in phase 3.
- **MANO** — hand-only template (15 finger bones, ~778 verts).
  Already cited in our README; the rig already names hand_L /
  hand_R. We add finger bones + a MANO-style compound hand mesh.
- **CAPE** — clothing as displacement over SMPL. We already do
  this conceptually with capsule shells; in phase 5 garments read
  the body mesh's surface positions instead of the bone radius.

Sources:
- https://smpl.is.tue.mpg.de/
- https://github.com/vchoutas/smplx
- https://static.makehumancommunity.org/about/concepts/basemesh.html
- https://github.com/gulvarol/smplpytorch (SMPL forward pass in
  PyTorch — useful as a reference for the LBS math, even though
  we'll keep ours in NumPy + GLSL).

## Topology design

A practical target (shared by SMPL, MakeHuman base, our existing
skin shader budget) is **~6-10k vertices, all triangles, watertight,
edge-flow following anatomical landmarks**. We don't need 14k like
MakeHuman because we don't ship a separate hi-res "Anny Body" mesh.

Construction approach: **stitched capsule strips, generated at
runtime in Python**. We already have ellipsoid + capsule
primitives that emit (verts, normals, bone-indices, weights,
indices). The new builder generates a *connected* strip per body
region instead of independent shells:

| Region            | Approach |
|-------------------|----------|
| Torso (chest, spine, pelvis) | One **lofted tube** with per-ring radius lookups — smooth interpolation between `chest.r`, `spine.r * waist`, `pelvis.r` so the seam isn't at a single y. Cross-section is a non-circular oval (wider X than Z, `0.95 : 0.70`). 24 rings × 32 sides ≈ 768 verts. |
| Neck              | Continuation of the torso loft up to the jaw base. 4 rings × 24 sides ≈ 96 verts. |
| Shoulder + clavicle yoke | Bridge ring stitched to torso top + arm root. ~96 verts. |
| Upper / fore arm  | Lofted tube, 12 rings × 16 sides per limb ≈ 192 × 4 = 768 verts. |
| Hand (phase 3)    | Compound: palm + 5 fingers, MANO-style topology. ~600 verts × 2 hands = 1200 verts. |
| Hip + thigh + shin | Lofted tube with hip flare interpolating `pelvis.r` → `thigh.r` → `shin.r * lower_bulk`. ~600 verts × 2 legs = 1200 verts. |
| Foot (phase 2)    | Stretched ellipsoid (toe-cap + sole, current cycle 4 work) bound into the leg loft so there's no seam at the ankle. 256 verts × 2. |
| Bust / glutes / female-specific volume | Per-region shape morphs applied as vertex deltas onto the torso loft (no separate ellipsoid). |

Total ~4-5k verts for the body, well under the 6k SMPL budget,
trivially within our shader bone limit.

**Why a loft, not a unified-mesh-from-scratch:** the bone
hierarchy is the natural skeleton for the loft. Each bone owns a
strip; ring transitions stitch parent-strip last-ring to
child-strip first-ring. Skinning weights fall out by construction
(per-ring weight = blend of the two adjacent bones, governed by
position along the bone).

## Skinning

Our existing shader supports up to 2 bone influences per vertex.
A continuous loft has 2-bone influence at every joint seam by
design, so we don't need to widen the shader.

Weights:
- **Inside a strip**: 100 % to that bone.
- **At a strip boundary**: smooth blend (cosine ramp) from bone A
  to bone B over a transition zone of ~1-2 rings on either side.
- **At the shoulder yoke**: 3-way blend (chest, clav, uarm) is
  collapsed to 2 by picking the dominant pair per vertex; this
  matches what `build_selected`'s yoke patch already does.

`mathx.lbs(...)` already exists — we re-use it. No GPU change.

## Driving the morphs from `Shape`

The `Shape` dataclass becomes the **only** source of per-vertex
deformation. Each shape parameter controls one or more named morph
deltas applied to the loft *before* skinning. Examples:

```
height       -> uniform Y scale on every vertex
bulk         -> per-ring radial scale (1 + (bulk-1) * profile(y))
                where profile(y) is fuller at the chest and softer
                at the shins, gender-conditioned.
shoulder_w   -> X scale on shoulder/clav rings only
hip_w        -> X scale on pelvis + upper-thigh rings
waist        -> radial scale on the spine rings (carries over the
                effect already in the bone schema, but applied to
                the cloth-relevant surface vertices).
bust         -> additive forward-projected delta on the chest
                front rings (replaces `_bust` ellipsoid).
torso_len    -> Y scale on torso rings only
leg_len      -> Y scale on thigh + shin rings only
limb_len     -> Y scale on uarm + farm rings only
neck_thick   -> radial scale on neck rings
q_angle      -> shear on knee row (keeps current Q-angle physics)
lower_bulk   -> radial scale on hip + thigh rings, on top of bulk
```

These are written by hand in `human/body_morphs.py` (new file).
Each is a NumPy function `def morph_X(verts, rings, shape) -> verts`.
At build time we compose them in a fixed order: scale axes first,
then radial scales, then localized projections (bust, glutes),
then post-shears (q_angle).

Crucially, **none of these need data**. They are explicit anatomical
rules, exactly the way our current capsule rig already encodes them
in the `_SHAPE_RULES` table — just at vertex resolution instead of
bone resolution.

## Phasing

A pragmatic phased rollout that keeps the project shippable at every
step:

### Phase 1 — torso loft (replaces chest+spine+pelvis capsules)

- New module `human/body_loft.py` with one `build_torso_loft(bones,
  shape) -> SkinnedMesh` function.
- Hand it the chest / spine / pelvis bones. Generate a 24-ring x 32-
  segment lofted tube.
- Integrate into `human/mesh.py::build` behind a `shape.use_loft:
  bool = False` flag, then flip the default once the visual is on
  par with the capsule version on a fixed seed.
- Cycle on the triview comparison loop until row 1 reads as
  hourglass + V-taper at the torso.

Expected effect: closes the cycle 4-8 "torso reads as a tube"
finding; cloth still wraps the capsules in this phase since
nothing else has migrated yet.

### Phase 2 — limbs + feet integrated into the loft

- `build_limb_loft(bone_chain, shape)` for each of `uarm + farm`,
  `thigh + shin`, anchored into the torso loft via the shoulder
  yoke and pelvis hip-rings already built in phase 1.
- Sneaker toe-cap + sole plate from cycle 4 fold into the foot
  loft so there's no seam.
- Hand stays as the existing capsule for now (phase 3 lifts it).

Expected effect: silhouette no longer has any visible bone-radius
discontinuity; cloth shells still inflate from the new mesh's
surface (small adapter — read closest mesh vertex along the bone
axis instead of the bone radius).

### Phase 3 — MANO-style compound hand

- Add finger bones to `BONES_BASE` (5 fingers x 3 bones each = 15
  per hand x 2 = 30 new bones; we go from 19 → 49 bones).
- Update `_SHAPE_RULES`, `_JOINT_LIMITS`, T-pose, walk cycle,
  ARKit rig, BVH name map.
- Bone budget jumps from 19 to 49; shader uniform array stays
  fine (limit is 32 *per draw*, not total — split body, head,
  hands into 3 draw calls or raise the uniform array). **Decision
  point:** raise the per-draw bone array from 32 to 64 (cheap
  uniform memory) to keep one draw call per drawable.
- New `build_hand_mano(bones, shape)` that lofts a palm + finger
  segments. ~600 verts per hand.

Expected effect: hand silhouette in the comparison goes from
"capsule blob" to "5 finger T-pose" — matches what Hunyuan3D and
the photoreal target show.

### Phase 4 — pose correctives (optional)

- Hand-written corrective deltas at flex joints (elbow, knee,
  shoulder) that fire when the joint angle exceeds a threshold.
  Mirrors what SMPL's pose blendshapes do, but rule-based.
- Same idea our existing `u_bend_inflate` shader trick uses, but
  applied to the body mesh itself instead of the cloth shell.

Expected effect: cleaner deformation at heavy flexion (BVH walk
cycle high knees, shoulder shrugs from face_anim).

### Phase 5 — garments re-anchored to the body mesh

- `build_top` / `build_bottom` / `build_dress` start from the
  matching body-mesh region (the same ring set they cover today)
  instead of from bone capsules. The "inflate" parameter becomes
  a true per-vertex outward push along the body mesh's vertex
  normal.
- This is the CAPE pattern: clothing as a per-vertex displacement
  field over the body mesh.

Expected effect: cloth follows the actual silhouette (waist
taper, bust projection, hip flare) instead of a cylinder. Closes
the remaining cycle 8 finding "shirt covers the cloth waist
seam".

## Risks and unknowns

1. **Topology consistency under shape extremes.** Lofted tubes
   can self-intersect at extreme shape values (very narrow waist,
   very wide hips). Mitigation: clamp the per-ring radius
   profiles to a minimum thickness, and unit-test the build at
   shape extrema (random_shape × 1000 seeds).
2. **Skinning seam at the yoke.** Three-way bone influence at
   the clavicle/chest/uarm junction is the hardest part of any
   parametric body. SMPL solves it by hand-painting weights;
   we'll need the same. Plan: write a small `tools/weight_paint.py`
   that visualises per-vertex weights as colors so we can iterate
   on the falloff function.
3. **Performance.** 5k verts × 60 fps on Apple Silicon integrated
   GPU is fine for the skinning shader, but the *Python build
   cost* per regeneration matters because pressing `R` in the
   viewer rebuilds everything. Today the build is < 50 ms; the
   loft build will be slower (more vertices). Target < 200 ms
   per regeneration; if we hit that wall, vectorize the loft in
   NumPy or cache by Shape hash.
4. **Bone limit raise from 32 → 64.** Some old GPUs cap uniform
   array length tightly. We're targeting GL 3.3 core which
   guarantees `MAX_VERTEX_UNIFORM_COMPONENTS >= 1024`, i.e. 64
   `mat4`s = 1024 components — exactly fits. No driver risk on
   any Apple Silicon, modern Intel, or RTX board.
5. **BVH retargeting** with finger bones. Mixamo BVHs typically
   include finger bones already, named e.g. `LeftHandIndex1`. The
   name map needs a default for the 30 new bones; characters
   loaded against legacy 19-bone BVHs leave fingers in T-pose.

## Migration order (ranked)

1. Phase 1 (torso loft) — **highest impact**, isolated risk, no
   skeleton changes. ~1 week of focused work.
2. Phase 2 (limb loft + foot weld) — straightforward extension of
   phase 1. ~3 days.
3. Phase 5 (cloth re-anchoring) — needs phases 1-2 done. ~3 days.
4. Phase 3 (MANO hand) — biggest skeletal surface change; do it
   only after the body loft is shipped because it touches BVH /
   ARKit / joint limits. ~2 weeks.
5. Phase 4 (pose correctives) — only if phases 1-3 don't already
   close the silhouette gap.

Total: ~5 weeks of one-person engineering for the full migration.
Phase 1 alone closes about half the remaining gap visible in
`compare/c8_compare.png` and is shippable on its own.

## Decision points that need a human

- **Do we keep the capsule path as a fallback?** Yes for at least
  one cycle, behind `Shape.use_loft`. Once the loft passes the
  triview comparison on every existing seed in our gallery
  (`screenshots/01_*` through `screenshots/12_*`), drop the flag.
- **Hand bone count.** 30 new bones is the SMPL-X choice; if BVH
  / ARKit headaches dominate we can reduce to 15 (one bone per
  finger, no IPJ) and still get a finger silhouette improvement.
- **Per-vertex normals at runtime?** If shape morphs change
  curvature significantly (waist cinch, bust), recomputing
  normals per build keeps shading correct. NumPy can do this for
  5k verts in <10 ms. Plan: yes, recompute on build.
