# v2 rewrite plan — C core + Cython + Python app

Decided 2026-04-27. Replaces the v1 Python-only implementation (preserved at tag `v1-python-final`).

## Goals
- Portable runtime in pure C99, no external deps beyond libc/libm.
- Python is renderer + UI only; never touches mesh math at runtime.
- Six base meshes generated offline by 2D→3D AI (Hy3D 2.1 on bender), nude + bald + T-pose, rigged to a shared Mixamo skeleton.
- Parameters added one at a time, each fully wired through C → Cython → UI before moving on.

## Archetypes (IDs frozen in `core/human.h`)
0. child F  · 1. child M  · 2. adult F  · 3. adult M  · 4. old F  · 5. old M

## Skeleton
Mixamo standard. Lets us reuse Mixamo animations during validation and import community rigs/clothing later.

## .hmesh format (little-endian)
```
header   : magic "HMSH", version, archetype_id, vert_count, idx_count, bone_count, morph_count
verts    : pos(3f) + normal(3f) + uv(2f) + bone_ids(4u8) + bone_weights(4f)
indices  : u32[]
bones    : parent_idx(i32), bind_pose_mat4, name[32]
morphs   : name[32], delta_count, [vert_idx(u32), dpos(3f), dnormal(3f)]*
```

## C public API (frozen surface — keep small for portability)
See `core/human.h`. ~20 functions covering load, evaluate, vertex/index buffer access, and parameter set/get.

## Phase order
1. **Scaffold** ✅ (this commit) — directory tree, stub C lib, Cython binding, build files.
2. **Math + I/O** — vec3/mat4/quat in C, `.hmesh` reader/writer, unit tests in C.
3. **First render** — Cython exposes vertex buffer via NumPy buffer protocol; moderngl draws a hardcoded cube.
4. **Offline pipeline (one archetype)** — Nano Banana Pro → Hy3D → Blender cleanup → `.hmesh` for adult M.
5. **Skeleton + LBS** — bind pose render, then linear blend skinning in C.
6. **Remaining 5 archetypes** through the same pipeline.
7. **Parameter system** — registry in C, first parameter `height` (bone-scale, no morph). End-to-end UI slider.
8. **Morphs** — bake belly/limb girth, expose `weight` parameter.
9. Continue per the parameter roadmap.

## Parameter roadmap (priority)
| # | Param | Type | Notes |
|---|---|---|---|
| 1 | height | bone-scale | uniform spine + IK feet |
| 2 | weight (BMI) | morph | per-archetype |
| 3 | muscle | morph | per-archetype |
| 4 | breast_size | morph | F archetypes only |
| 5 | shoulder_width | bone-scale | |
| 6 | head_size | bone-scale | |
| 7+ | facial morphs | morph | requires face submesh subdivision |

Hair, facial hair, and clothing are layered meshes attached to the skeleton — out of scope for v2 phase 1.

## What we kept from v1
- `screenshots/` — validated Hy3D / Trellis2 / triview outputs (saves regen cost).
- `notes/` — historical decisions and enhancement-cycle records.
- `LICENSE`, `.gitignore`.

Everything under `human/`, `tools/`, `scripts/`, `compare/`, `remote/`, `animations/`, `human_generator.py`, `run.sh`, `voice.wav`, `requirements.txt`, `enhance-pipeline.txt`, and the v1 README is gone on this branch but recoverable via `git checkout v1-python-final -- <path>`.
