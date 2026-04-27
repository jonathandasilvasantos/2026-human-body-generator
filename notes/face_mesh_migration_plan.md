# Migration plan — primitive head compound → continuous face mesh

The current head (`human/mesh.py::_head_compound`, `_skull_shell`,
plus the eye/nose/lip/ear primitives) is anatomically reasonable at
body framing distance but reads as "mannequin" the moment we zoom
into the face. The cycle 12 face close-ups in
`compare/c12_compare.png` make this obvious: clown-ball nose,
pinhole eyes, no eyebrow geometry, flat skin.

This document plans the move from "compound of 30+ ellipsoids" to a
**single dense face mesh with per-region morphs** that scales across
gender / age / individual variation while remaining purely procedural
(no external assets).

## What we keep (constraints)

- Procedural at startup, no external `.obj/.npz/.fbx` assets.
- The existing **ARKit-52 facial animation rig** (`human/face_anim.py`,
  Cohen-Massaro VisemeTrack, FACS presets, audio lipsync). The new
  face mesh has to expose the same per-vertex deformation hooks the
  rig drives — eyelid Y, lip-corner Y, jaw open, lip ring squeeze,
  cheek squint, brow lift, etc. We are not rewriting the rig.
- The existing **eye stack semantics** (sclera, limbal ring, iris,
  pupil, collarette, lacrimal caruncle, catchlight, gaze convergence)
  documented in `notes/eye_realism.md`. The new face mesh keeps the
  multi-layer eye as embedded sub-meshes (not part of the skin
  surface).
- The 19-bone skeleton, head + neck bones unchanged.
- `Shape` + `Appearance` continue to be the public knobs. Specifically:
  `head_size`, `face_asymmetry`, `lip_fullness`, `eye_sep`, `eye_size`,
  `eye_depth`, `nose_width`, `nose_proj`, `nose_bridge`, `cheekbone`,
  `jaw_width`, `chin_proj`, `brow_prominence`, `age_group`. These
  drive the new mesh's morphs.
- Skin shader (`u_mode = 0`), biophysical melanin + hemoglobin model,
  fragments unchanged. The new mesh writes to the same shader.

## What we are not doing

- **Not licensing FLAME.** FLAME is the gold standard (5,023 verts,
  PCA-fit identity + expression blendshapes, ~33k 3D scan training
  set) but the asset bundle is academic-license only, derived from
  data we have no rights to. Same reason we didn't license SMPL for
  the body.
- **Not training PCA.** No 3D head scans on hand. The morphs stay
  analytical — written as Python functions of `Shape`, exactly the
  way the body loft does.
- **Not regenerating the skull / cranium silhouette.** The skull
  shell is already anatomically correct from a distance. The face
  mesh covers the *facing* hemisphere of the head from forehead to
  chin; the back of the head stays as the existing skull shell
  (gives us hair attachment for free).

## Reference points (research)

- **FLAME** (https://flame.is.tue.mpg.de/) — single template,
  ~5k verts, 9k triangles, articulated neck/jaw/eyeballs, learned
  identity + expression PCA. We borrow the **topology and
  region structure** without the learned data.
- **MakeHuman base mesh** — fixed topology, slider-driven morph
  targets. Their face is part of the larger Homunculus base; the
  topology around eyes/lips/nose is hand-modelled and ours can
  copy the edge flow.
- **MetaHuman creator** (Epic Games) — proprietary, but the public
  topology charts show the same eye/lip/nose region edge flow.
- **tinyflame** (https://github.com/vaden4d/tinyflame) — minimal
  PyTorch FLAME forward pass. Useful to understand how PCA blends
  + LBS combine at runtime, even though we re-implement in NumPy.
- Shi's "How Does FLAME Work"
  (https://shi-yan.github.io/how_does_flame_work/) — clearest
  explanation of FLAME's V_template + shape blendshape +
  expression blendshape + pose corrective + LBS pipeline.

## Topology design

Target: ~3,000-4,000 vertices on the front hemisphere of the head,
all triangles, watertight where it joins the skull shell. Edge flow
follows the four canonical face regions:

| Region        | Approach |
|---------------|----------|
| **Cranium / forehead** | Concentric rings on the upper skull — co-built with the existing skull shell so there's a continuous seam at the hairline (~`H_HAIRLINE` = 0.77 of head length). Not strictly part of the "face" mesh; uses the existing `_skull_shell` topology with a denser ring count near the hairline. |
| **Brow + glabella** | A horizontal strip across the brow (between `H_NOSE_BASE` and `H_BROW`). 6 rows × 18 cols ≈ 108 verts. Carries `brow_prominence` and per-side expression hooks (browInnerUp, browOuterUpLeft/Right). |
| **Eye orbit** | Two concentric rings around each eye socket — outer orbit + lid edge. The existing eye stack (sclera/iris/etc) sits inside, unchanged. ~80 verts per side × 2 = 160. |
| **Nose** | A lofted tube along the nasal bone, with bridge / dorsum / tip / wing rings. Bridge width scaled by `nose_bridge`, projection by `nose_proj`, wing width by `nose_width`. ~12 rings × 14 segs ≈ 168 verts. Replaces the current ball + wings primitives. |
| **Mid-face / cheeks** | A grid of rings between the orbits and the mouth, fairing in the cheekbones from the side. `cheekbone` shape knob lifts a band of vertices laterally. ~6 rows × 24 cols = 144 verts per side × 2 = 288. |
| **Mouth ring** | The classical "lip ellipse" topology — one ring around the lip vermilion, one around the mouth aperture, with edge loops that support `mouthSmileLeft/Right`, `mouthFrownLeft/Right`, `lipPucker`, `lipFunnel`, `jawOpen`, etc. ~14 rings × 28 segs ≈ 392 verts. The most expressive region; most ARKit blendshapes target it. |
| **Chin / jaw line** | Lower mid-face strip. `chin_proj` and `jaw_width` morphs. ~6 rows × 20 cols = 120 verts. |
| **Ears** | Left as separate primitives (current code). The ear is fiddly enough that adding it to the loft topology is more pain than payoff and we already have it. |

Total face-mesh budget: ~1,400-1,800 face verts + the existing skull
shell, head sub-meshes, eye stack — well under FLAME's 5k.

## Driving the morphs from `Shape` and `Appearance`

Each `Shape` knob maps to a per-vertex delta on the template. The
mapping is hand-written in a new `human/face_morphs.py`:

```
head_size           uniform 3D scale of every face vertex
face_asymmetry      small lateral radial bias along the cheek band
lip_fullness        Y/Z scale on the upper + lower vermilion rings
eye_sep             X translation of the orbit ring centroid (more
                    eye separation = a wider face read).
eye_size            isotropic scale on the eyelid ring radius
eye_depth           Z translation of the orbit ring (deeper-set eyes)
nose_width          X scale on wing + base rings
nose_proj           Z translation on tip + dorsum rings (forward)
nose_bridge         Y position of the highest dorsum ring (taller
                    nose bridge = a more European read).
cheekbone           lateral lift on a band of mid-face verts at the
                    zygomatic level.
jaw_width           X scale on the chin + lower jaw band.
chin_proj           Z translation of the chin strip vertices.
brow_prominence     Z translation of the brow strip — protruding
                    supra-orbital ridge.
age_group           composite morph — adult vs young vs elder bias
                    on lip fullness, brow prominence, mid-face
                    fullness, plus drives the existing
                    `_skin_age_lines` overlay.
```

Each morph is a NumPy function `def morph_X(verts, regions, shape) -> verts`
where `regions` is a dict of vertex-index sets keyed by region name
(brow / eye_orbit / nose / cheek / lip / chin / forehead). The regions
are baked into the topology generator at build time so we don't
re-classify per regen.

Same identity → same vertex layout → cycle-to-cycle diffs are clean.

## ARKit-52 hooks

The face rig today maps each ARKit channel to "modulate primitive X
parameter". We rewrite each mapping as "translate the vertices in
region R by direction D scaled by weight W". Examples:

```
eyeBlinkLeft        eye_orbit_L upper-lid ring  Y down  by  full
                    eyelid travel (~0.014 head-units).
mouthSmileLeft      lip ring left-corner verts  Y up + X out
                    by the mouth-corner travel from FACS.
jawOpen             every vertex below `H_NOSE_BASE`, weighted by
                    proximity to the chin, gets a Y rotation around
                    the jaw pivot — preserves the existing jaw_open
                    physics from face_anim.
mouthPucker         lip ring all verts pulled toward the lip-ring
                    centroid by the FACS pucker amount.
```

One-time porting cost: every line in `human/face_anim.py` that today
sets a primitive parameter becomes a region-vertex translation. The
ARKit-52 channel list and `FaceClip.resolve()` API stay untouched —
the *only* change is what `apply_face_anim()` does internally.

## Phasing

### Phase F1 — face shell topology

- New module `human/face_mesh.py` with one
  `build_face_mesh(bones, shape, appearance) -> SkinnedMesh`.
- Generate the 7 region strips listed above. Stitch them so edges
  share vertices at the seams. Bake region masks (`regions` dict).
- Hide the current `_head_compound` primitives (skull/face/nose/lips
  primitives) behind a `Shape.use_face_mesh = False` kill switch. The
  new mesh layers underneath the existing eye / brow / lip /
  facial-hair primitives (which keep working unchanged for now).
- Validate on a fixed seed against `c12_face_*.png`.

Expected effect: face close-ups stop reading as "mannequin head" and
get edge-flow consistent with the photoreal target.

### Phase F2 — per-region shape morphs

- Implement the morph table above in `human/face_morphs.py`. Each
  morph is a pure NumPy function; the orchestrator composes them in
  a fixed order before LBS.
- Wire `random_shape()` and `Appearance` to drive the morphs.

Expected effect: identity variety (eye sep, nose width, jaw width,
cheekbones) becomes visible at face-close-up framing.

### Phase F3 — ARKit-52 rewrite

- Port every ARKit channel handler in `human/face_anim.py` from
  primitive parameters to region-vertex translations.
- Re-test against `tools/lipsync_capture.py` and
  `tools/face_anim_capture.py` to make sure visemes and FACS
  presets still resolve correctly.
- Decommission the now-unused primitive helpers (skull-shell stays
  for the cranium; eye stack stays; ears stay).

Expected effect: face animation continues to work with no API
changes. Audio lip sync, BVH retargeting, expression presets all
keep their existing call sites.

### Phase F4 — micro-anatomy details

Once the topology is stable, layer in detail that needs higher
vertex resolution to read:
- nasolabial folds as a baked normal-map style ridge in the
  cheek-region verts, age-conditioned.
- philtrum (the central groove from nose to lip) as a small
  Z-recessed strip on the upper lip top edge.
- supra-orbital crease (eyelid fold) on the upper-lid ring.

These are vertex-level deltas, not new topology. ~1 day each.

## Phasing risks

1. **ARKit channel coverage.** The current rig wires 52 channels to
   primitive parameters. If any channel relies on a primitive we
   *delete* in phase F1, audio lipsync and the ARKit-52 capture
   matrix break silently. Plan: keep the primitive helpers
   importable but *unused* until F3 ports each channel; run
   `tools/face_anim_capture.py` before/after every commit.
2. **Eye-orbit alignment.** The eye stack (sclera/iris/etc.) is
   positioned by `_eye_metrics()`. The new face mesh's eye-socket
   ring has to align *exactly* with that helper, otherwise we get
   a visible seam. Plan: have `face_mesh.build_face_mesh` consume
   `_eye_metrics()` directly, not duplicate the math.
3. **Face asymmetry.** Today `Shape.face_asymmetry` is applied as a
   tiny radial bias on the skull shell; on the dense face mesh it
   has to be applied as a per-vertex bias whose strength varies by
   region (peaks at the cheekbone, near zero at the lip). Bigger
   risk of "wrong" asymmetry on a denser mesh.
4. **Build cost.** `_head_compound` rebuilds in <30 ms today. The
   new face mesh adds ~1.5k verts and morph composition; target
   <80 ms total to keep `R` (regenerate) snappy.

## Total estimated cost

| Phase | Work | Cost |
|-------|------|------|
| F1    | Topology generator, region masks, build pipeline hook | ~1.5 weeks |
| F2    | 14 analytical morphs + random_shape wiring | ~3 days |
| F3    | ARKit-52 channel port + regression tests | ~1.5 weeks |
| F4    | Detail layer (folds, philtrum, eyelid crease) | ~3 days |
| **Total** | | **~4 weeks** |

This is the same shape of project as the body migration, with the
ARKit port as the single biggest risk factor (lots of channels, lots
of regression surface area).

## Open decisions for the human

- **Do we ship F1 alone first?** Phase F1 swaps topology without
  changing animation behaviour — low risk, immediate face-close-up
  win, parks the new mesh under the existing primitives until F3.
- **Subdivision multiplier.** 1,400 face verts is a sweet spot for
  edge-flow + perf, but we could 2× to ~2,800 if perf budget allows.
  Recommend starting at 1×.
- **Drop the ear primitives or keep?** Keep — perfect ears at
  procedural distance are a deep rabbit hole and the current ones
  read fine.
