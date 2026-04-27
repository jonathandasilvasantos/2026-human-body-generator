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
1. **Scaffold** ✅ — directory tree, stub C lib, Cython binding, build files.
2. **Math + I/O** ✅ — `hmath.{c,h}` (vec3/mat4 + general inverse), `.hmesh` reader/writer in `io.c`, C unit tests in `core/tests/test_engine.c`.
3. **First render** ✅ — Cython exposes vertex buffer via NumPy zero-copy (NPY_FLOAT32 view). `app/render.py` does offscreen moderngl Lambert shading; `human triview` CLI produces a 3-up grid.
4. **Offline pipeline (one archetype)** 🟡 partial:
   - Stage 1 (text→reference image): scaffolded as `gen_reference.py`. **Open question:** which backend? v1 memory says we moved AI image gen off paid APIs (Nano Banana Pro replaced by FLUX.1 Kontext on bender) for the triview Stage B, but Kontext is image *editing* not text-to-image. Stage 1 needs a text-to-image model — candidates: FLUX.1 dev / SDXL on bender (free, slower), or Nano Banana Pro one-time (12 images, then never again).
   - Stage 2 (image→raw GLB): scaffolded as `gen_mesh.py`, SSHes to bender for Hy3D 2.1.
   - Stage 3 (cleanup + Mixamo bind): documented stub, requires Blender headless work — lands when first raw GLB exists.
   - Stage 4 (bake) ✅ **implemented** — `bake_hmesh.py` ingests a GLB via pygltflib and calls the new `human_create_from_arrays` C entry point. Naive mode (single root bone) works on any GLB; rigged mode (skin + JOINTS_0/WEIGHTS_0 + IBM→bind_local reconstruction) is stubbed for the cleanup output. Validated against `screenshots/triview/c10.glb` (584k-vert Hy3D output) → renders correctly through the offscreen pipeline.
5. **Skeleton + LBS** ✅ — `skeleton.c` resolves bind world from local + parent chain, computes skin palette per evaluate. `skin.c` does 4-influence LBS with normal renormalization. Validated by the proto archetype (3 bones, smooth weights along Y).
6. **Remaining 5 archetypes** ⏳ blocked on phase 4.
7. **Parameter system** ✅ — `param_t` registry in `internal.h`, dispatch in `human_evaluate`. `HUMAN_PARAM_BONE_SCALE_Y` (drives the proto's `height`) and `HUMAN_PARAM_MORPH` (drives `weight`/`head_size`). UI consumes `human_param_range` for slider bounds.
8. **Morphs** ✅ — sparse `(vert_idx, dpos, dnorm)` deltas per morph in `morph.c`, weighted accumulation before skinning. Proto bakes belly + head-cap morphs procedurally to validate the path.
9. **Continue per the parameter roadmap below** ⏳ — gated on phase 4 producing real archetypes.

## Validated end-to-end on the proto archetype
- C tests: math inverse, proto build, param clamping, save/load roundtrip → `OK`
- Python tests (pytest): 7 cases including height growth, weight radial expansion, `.hmesh` save/load equivalence, normals unit-length post-evaluate.
- Render: `human triview proto -p height=1.3 -p weight=1.2 -p head_size=0.5 --out posed.png` produces a visibly taller, fatter, larger-headed figure than baseline.

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
