# Muscular facial deformation research notes

This pass keeps the public animation surface as ARKit-52/FACS weights, then
derives lightweight regional muscle activations for real-time deformation.

## Research findings

- **FACS and ARKit remain the right control layer.** FACS is still the common
  muscle-oriented language for face rigs, and ARKit-style blendshape weights are
  broadly supported by capture and audio-driven systems. Apple documents
  blendshape coefficients as named feature motions in `[0, 1]`; NVIDIA
  Audio2Face-3D exports ARKit blendshape streams; Roblox describes FACS controls
  as based on facial muscle placement and transferable between rigs.
- **Blendshapes are practical but need anatomical coupling.** Lewis et al.
  describe blendshapes as the prevalent production method and a simple,
  semantic linear model, while also noting unresolved problems. For this
  procedural primitive face, the implication is to keep semantic sliders but
  avoid isolated one-part movement by deriving coupled cheek, lip, brow and jaw
  responses.
- **Muscle models give the visual target.** Waters' classic muscle model
  identifies directional muscle families: uppers/downers, horizontal muscles,
  oblique lip-to-cheek muscles, and orbital sphincters around mouth and eyes.
  Specific references map well to this rig: zygomaticus major raises and draws
  mouth corners back for happiness; brow lowering plus lip tightening reads as
  anger; nose/upper-lip lift drives disgust.
- **Full physics is too expensive here, but informs approximations.** Sifakis,
  Neverov and Fedkiw used an anatomically accurate volumetric model with
  passive tissue, skeletal structure and anisotropic muscle activations,
  producing spatially and temporally coherent deformation. Their speech work
  reinforces the same point for phonemes: the face should be driven by muscle
  activation signals, not unrelated surface edits.
- **Soft-tissue deformation needs volume and sliding cues.** Surgery-simulation
  literature models expression as contracting muscles acting on soft tissue and
  solves elastic deformation with FEM or precomputed muscle actions. This
  project approximates those effects with small skin-colored tissue volumes,
  nasolabial fold strips and dynamic crease patches.
- **Real-time systems rely on regional masks and temporal smoothing.** NVIDIA's
  runtime configuration exposes upper/lower face strength, region mask
  softness, skin strength and smoothing. Roblox reports temporal losses and
  smoothing to reduce jitter while preserving responsiveness. The implementation
  mirrors this with cheap region activation and smoothstep clip interpolation.

## Implementation guidance

- Keep `Appearance.blendshapes` and named presets backward-compatible.
- Map ARKit weights into muscle groups with `face_anim.muscle_activations`.
- Drive visible deformation by anatomical regions:
  - zygomaticus/risorius/depressor groups affect corners and cheek mass.
  - orbicularis oculi affects lower lids and cheek bunching.
  - orbicularis oris affects pucker/funnel/lip compression.
  - nasalis/levator labii affects sneer, nostril-side mass and upper lip.
  - jaw/mentalis/depressor labii affects lower lip, chin and jaw pad.
- Preserve volume by compensating radii when lips compress, pucker or stretch.
- Add only low-poly primitive overlays so the cost remains predictable in
  headless capture and interactive viewer paths.

## Cycle log

- **Baseline** (`screenshots/muscle_deform_baseline/`): existing ARKit/FACS
  rig rendered across presets, clips, isolated channels, asymmetry,
  cross-character rows and walk+face composition. Issues: static face shell,
  isolated lip/eyelid movement, abrupt linear transition timing, weak
  jaw/lip coupling and limited profile/close-up coverage.
- **Cycle 1** (`screenshots/muscle_deform_cycle1/`): added regional muscle
  activations, smoothstep interpolation, lip volume compensation, jaw/lip
  coupling, brow/lid coupling and soft-tissue overlays. Issue: skin-volume
  primitives were too visible and read as pasted-on facial geometry.
- **Cycle 2** (`screenshots/muscle_deform_cycle2/`): reduced tissue volumes
  and made folds shallower. Issue: close-ups still exposed primitive patch
  boundaries around cheeks and mouth.
- **Cycle 3** (`screenshots/muscle_deform_cycle3/`): removed visible
  skin-volume overlays from the render path, keeping muscle-derived controls
  where they remain artifact-free: lips, jaw interaction, eyelid/brow coupling,
  dynamic expression folds and transition smoothing. This is the validated
  capture set for the branch.

## References

- Keith Waters, "A Muscle Model for Animating Three-Dimensional Facial
  Expressions", Computer Graphics, 1987.
  https://people.ece.cornell.edu/land/OldStudentProjects/cs490-95to96/HJKIM/waters87.html
- Eftychios Sifakis, Igor Neverov, Ronald Fedkiw, "Automatic Determination of
  Facial Muscle Activations from Sparse Motion Capture Marker Data", SIGGRAPH
  2005. https://graphics.cs.wisc.edu/Papers/2005/SNF05/
- Eftychios Sifakis, Andrew Selle, Avram Robinson-Mosher, Ronald Fedkiw,
  "Simulating Speech with a Physics-Based Facial Muscle Model", SCA 2006.
  https://graphics.cs.wisc.edu/Papers/2006/SSRF06/
- J. P. Lewis et al., "Practice and Theory of Blendshape Facial Models",
  Eurographics STAR 2014. https://doi.org/10.2312/egst.20141042
- E. Gladilin et al., "Anatomy- and physics-based facial animation for
  craniofacial surgery simulations", Med Biol Eng Comput, 2004.
  https://pubmed.ncbi.nlm.nih.gov/15125145/
- Apple ARKit `ARFaceAnchor.blendShapes` documentation.
  https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapes
- NVIDIA Audio2Face-3D documentation.
  https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/text/architecture/audio2face-ms.html
- Roblox, "Real Time Facial Animation for Avatars", 2022.
  https://about.roblox.com/newsroom/2022/03/real-time-facial-animation-avatars
