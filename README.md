# 2026 Human Body Generator

A fully procedural, parametric 3D human character generator written in pure
Python + OpenGL. Generates a random humanoid on every run — gender-appropriate
body proportions, face, hair, facial hair, and layered clothing — and plays it
back with either a built-in physics-coherent walk cycle or any BVH motion-capture
file you point it at.

No external character assets, no pre-modelled geometry, no texture files.
Everything — skeleton, skinned mesh, face features, hair, clothes, skin/fabric
shading — is generated procedurally at startup.

---

## Features

### Parametric body
- 19-bone skeleton laid out on classical anthropometric proportions
  (~7.5 heads tall, shoulder width ~1.7 heads, etc.)
- Per-character `Shape` with SMPL-style gender-conditioned sampling: `height`,
  `bulk`, `limb_len`, `shoulder_w`, `hip_w`, `bust`, `head_size`, `leg_len`,
  `torso_len`
- Female variant adds bust + glutes geometry

### Anthropometric head
- Compound head built from the classical rule-of-thirds landmarks (chin,
  mouth, nose base, eye line, brow, hairline)
- Eyes (whites + iris + pupil), lips, nose ridge, cheekbones, ears, chin;
  jaw emphasis for male
- MANO-inspired compound hands with anatomical thumb placement

### Hair + facial hair
- Styles: bald / buzz / short / medium / long (drape past the shoulders)
- Procedural eyebrows
- Facial hair: none / stubble / mustache / goatee / full

### Clothing
- Tops: tshirt / longsleeve / tank / dress
- Bottoms: pants / shorts / skirt (flared cone)
- Shoes: barefoot / sneakers / boots
- Tight garments implemented as CAPE-style inflated capsule shells over
  the relevant bones; loose garments (skirt, dress) as cone shells rigidly
  skinned to the pelvis so legs swing freely inside.

### Rendering
- OpenGL 3.3 core profile via GLFW + PyOpenGL
- LBS (linear blend skinning) in the vertex shader, up to 32 bone matrices
- Fragment shader does biophysical-inspired skin (melanin + hemoglobin +
  pores), fBm fabric dye + weave, anisotropic fBm hair streaks, plus
  Blinn-Phong specular on skin
- MSAA fullscreen viewer + headless capture mode (offscreen FBO → PNG)

### Animation
- **Built-in physics-coherent walk cycle** driven by analytical two-bone
  IK. Foot targets are fixed on the ground during stance and arc in swing,
  so feet are truly grounded rather than sliding. Body translates forward
  at the speed implied by stride × cadence (inverted-pendulum model).
- **BVH animation retargeting**: load any Mixamo-style BVH; the retargeter
  walks both hierarchies in parallel and solves per-bone local rotations
  that reproduce the source's world-space rotation on our rig. Default
  name map covers Mixamo; custom rigs supported via a `name_map` dict.
- **Biomechanical joint limits**: per-bone `(X, Y, Z)` anatomical ranges
  (elbow/knee flexion only, shoulder ad/abduction asymmetric, etc.).
  Both `random_pose()` and BVH playback respect them.

---

## Research foundation

The implementation draws from:

- **SMPL** (Loper et al., SIGGRAPH Asia 2015) — parametric body + blend skinning
- **SMPL-X** — expressive extension
- **MakeHuman** — procedural humanoid pipeline
- **SKEL** (Keller et al., 2023) — skin-to-skeleton biomechanical rig of SMPL
- **FLAME** — parametric face; we borrow the classical landmark layout
- **MANO** (Romero et al., SIGGRAPH Asia 2017) — hand topology, simplified here
- **CAPE** (Ma et al., CVPR 2020) — clothing as displacement over SMPL
- **GarmentCode** (Korosteleva et al., 2023) — programmatic sewing-pattern
  garments
- **SIMBICON** (Yin, Loken, van de Panne, SIGGRAPH 2007) — biped locomotion
- **Inverted-pendulum gait models** (Kuo 2007) — walk bob + stance dynamics
- **Johansen** — semi-procedural locomotion (MSc thesis 2009)
- **Tolani, Goswami & Badler** — analytical IK for anthropomorphic limbs
  (Graphical Models 2000)
- **Akhter & Black** — Pose-Conditioned Joint Angle Limits (CVPR 2015)
- **Alotaibi & Smith** — Biophysical 3D Morphable Model of Face Appearance
  (ICCV 2017); Smith et al. — Morphable Face Albedo Model (CVPR 2020)
- **NVIDIA GPU Gems** — improved Perlin noise / fBm in GLSL

---

## Quick start

### Install

```bash
python3 -m venv env
env/bin/pip install -r requirements.txt
```

### Run the interactive viewer

```bash
./run.sh                       # walks the character using walk2.bvh
env/bin/python human_generator.py                   # idle, procedural
env/bin/python human_generator.py --bvh path/to/any.bvh
```

Controls:

| Key | Action |
|-----|--------|
| `SPACE` / `R` | regenerate everything (random gender, outfit, pose) |
| `M` / `F`     | regenerate forcing male / female |
| `S`           | re-roll shape only |
| `C`           | re-roll appearance (hair, clothes, colors) |
| `W`           | toggle procedural walk animation |
| `A`           | toggle BVH playback (if a BVH is loaded) |
| `P`           | random static pose |
| `T`           | T-pose |
| `B`           | toggle skeleton overlay |
| `[` / `]`     | slower / faster walk |
| `mouse drag`  | orbit camera |
| `scroll`      | zoom |
| `Esc` / `Q`   | quit |

### Headless capture to PNG

```bash
# Render a random still of the default character
env/bin/python -m human.capture --seed 42

# A specific pose
env/bin/python -m human.capture --seed 7 --gender female --pose t_pose

# A specific frame of a BVH
env/bin/python -m human.capture --pose bvh --bvh animations/walk2.bvh --bvh-time 0.5
```

Flags: `--out`, `--seed`, `--gender male|female|neutral`,
`--pose t_pose|walk_passing|walk_heel_strike|bvh`, `--bvh PATH`, `--bvh-time S`,
`--width`, `--height`, `--yaw`, `--pitch`, `--dist`.

### Animation diagnostics

```bash
env/bin/python -m human.anim_debug animations/walk2.bvh
```

Prints frame/duration, mapped bones, root trajectory, pelvis yaw range,
per-bone rotation range, and world-direction retarget error per bone, plus
renders a horizontal strip of N frames to `anim_debug.png` for visual review.
Flags: `--mirror`, `--frames`, `--seed`, `--gender`, `--width`, `--height`,
`--no-render`.

---

## Project layout

```
animations/walk2.bvh           sample Mixamo BVH (69 frames @ 30fps)
human_generator.py             entry point: interactive viewer
run.sh                         launcher that plays walk2.bvh on startup
human/
  skeleton.py                  bone hierarchy, shape morph, FK, IK, walk cycle,
                               joint limits, pose sampling
  mesh.py                      body + compound head + hands + glutes + bust
  accessories.py               hair, eyebrows, facial hair
  garments.py                  tops, bottoms, dresses, skirts, shoes
  primitives.py                ellipsoid / capsule / cone_shell / flat_patch
  mathx.py                     4x4 matrix helpers
  character.py                 Shape + Appearance + GPU drawable bundle
  renderer.py                  shader programs, VBO helpers
  viewer.py                    GLFW fullscreen viewer + input
  capture.py                   offscreen FBO -> PNG
  bvh.py                       BVH parser
  animation.py                 BVH retargeting (world-rotation FK)
  anim_debug.py                diagnostic + strip render
```

---

## License

MIT.
