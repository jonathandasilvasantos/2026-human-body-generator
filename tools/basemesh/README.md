# Base mesh generation pipeline

Offline pipeline that produces the six `.hmesh` archetypes consumed by the
runtime. Runs once per archetype; not part of the runtime.

Stages:
1. `gen_reference.py` — Nano Banana Pro renders the front+back T-pose nude bald reference (12 images).
2. `gen_mesh.py` — Hy3D 2.1 on bender turns reference images into raw GLB.
3. `cleanup.py` — Blender headless: remesh, symmetrize, auto-rig to Mixamo skeleton, UV unwrap.
4. `bake_hmesh.py` — Writes the portable `.hmesh` binary into `assets/basemesh/`.

Archetype IDs match `human_archetype_t` in `core/human.h`.
