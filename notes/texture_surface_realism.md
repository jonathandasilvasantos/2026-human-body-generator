# Texture and surface realism

Branch: `texture-surface-realism`

This pass improves facial realism through stable procedural texture
coordinates, layered skin albedo, and bump/normal-style surface detail.
The project still has no authored texture files or UV assets, so the
implementation treats the skinned local vertex position as a procedural
UV domain and keeps all texture sampling attached to the deforming mesh.

## Research summary

- Weyrich et al., *Analysis of Human Faces Using a Measurement-Based Skin
  Reflectance Model* (SIGGRAPH 2006): measured 149 subjects across age,
  gender and race, separating skin appearance into spatially varying BRDF,
  diffuse albedo, and subsurface scattering. Practical guideline: keep
  base color, specular/roughness, and scattering cues separable; vary them
  spatially instead of using one flat skin color.
- Graham et al., *Measurement-Based Synthesis of Facial Microgeometry*
  (2013): scan-derived mesostructure can be extended with synthesized
  high-resolution displacement to recover pores, wrinkles, and specular
  breakup. Practical guideline: pores should mostly affect normals and
  specular response; albedo speckle must stay subtle.
- Epic Digital Humans documentation: high-end real-time skin uses diffuse,
  roughness, specular, scatter, and normal maps, with separate meso normals
  for expression wrinkles and micro normals for pores. Practical guideline:
  layer broad albedo, meso folds, and micro pores independently; mask micro
  detail around fragile regions like eyelids and tear ducts.
- Epic shading model documentation: skin benefits from subsurface profile
  style behavior and soft diffuse transitions. Practical guideline: avoid
  hard Lambertian cutoffs and plastic-tight specular highlights.
- UV parameterization literature, including LSCM/ABF++/ARAP families and
  recent boundary-aware seam work, consistently optimizes for low angular
  stretch, controlled area distortion, and semantically hidden seams.
  Practical guideline for this procedural renderer: use continuous local
  coordinates on the face, avoid world-space sampling that swims during
  animation, and keep high-frequency detail isotropic enough that it does
  not expose chart seams or primitive boundaries.
- MacDorman et al., *Too Real for Comfort?* and Schindler et al.,
  *Differential effects of face-realism and emotion...*: uncanny responses
  rise when realism cues are inconsistent. Practical guideline: reduce
  over-symmetry and flatness, but avoid pushing one cue (pores, shine,
  expression) far beyond the fidelity of neighboring cues.

## Implementation guidelines

- Texture coordinates: sample skin from `v_local`, not world `v_pos`, so
  facial animation and body motion do not make pores or mottling slide.
- Albedo: combine low-frequency melanin/undertone variation with regional
  hemoglobin masks on cheeks, nose, eyelids, and mouth. Keep freckles and
  pores low contrast.
- Bump/normal detail: use shader-side height derivatives for pores and
  shallow forehead/eye-corner folds. Pores are sub-millimeter visual cues,
  not visible craters.
- Surface response: tie roughness variation to the same local skin domain
  and use wrapped diffuse for skin to avoid wax/plastic contrast.
- Critical regions: inspect eyes, eyelids, nose, lips, and mouth corners at
  front, three-quarter, and profile views. These regions expose stretching,
  stuck-on crease patches, and inconsistent detail first.
- Uncanny valley checks: look for over-symmetry, blank skin, excessive
  pore noise, shiny plastic, dead eyes, visible primitive seams, and detail
  that does not deform with expressions.

## Iteration checklist

For each cycle:

- Capture static face grids with `tools.face_capture` across diverse seeds,
  genders, ages, expressions, front/three-quarter/profile views, and all
  lighting styles.
- Capture animated facial presets and clips with `tools.face_anim_capture`
  so pores, folds, and albedo variation can be checked under deformation.
- Compare against the guidelines above: low stretch, no swimming detail,
  subtle albedo breakup, pores visible only in close-up, and soft skin
  light response.
- Refine shader masks and amplitudes conservatively. When a detail reads
  artificial, reduce contrast before adding complexity.

## Validation outputs

This branch uses:

- `screenshots/texture_surface_cycle1_static/`
- `screenshots/texture_surface_cycle1_anim/`
- `screenshots/texture_surface_final_static/`
- `screenshots/texture_surface_final_anim/`

## Source links

- Weyrich et al. 2006:
  https://dash.harvard.edu/entities/publication/73120378-8d6d-6bd4-e053-0100007fdf3b
- Graham et al. 2013:
  https://ict.usc.edu/pubs/A%20Measurement-based%20Synthesis%20of%20Facial%20Microgeometry.pdf
- Digital Ira SIGGRAPH 2013:
  https://history.siggraph.org/wp-content/uploads/2023/01/2013-Poster-36-Alexander_Digital-Ira-Creating-a-Real-Time-Photoreal-Digital-Actor.pdf
- Epic Digital Humans:
  https://dev.epicgames.com/documentation/pl-pl/unreal-engine/digital-humans?application_version=4.27
- Epic Shading Models:
  https://dev.epicgames.com/documentation/en-us/unreal-engine/shading-models-in-unreal-engine?application_version=5.6
- MacDorman et al. 2009:
  https://pubmed.ncbi.nlm.nih.gov/25506126/
- Schindler et al. 2017:
  https://www.nature.com/articles/srep45003
