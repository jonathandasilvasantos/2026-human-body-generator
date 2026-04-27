# Base mesh generation pipeline

Offline pipeline that produces the six `.hmesh` archetypes consumed by the
runtime. Runs once per archetype on a workstation; not part of the runtime.

## Stages

| Step | Script | Where it runs | Input | Output |
|------|--------|---------------|-------|--------|
| 1 | `gen_reference.py` | local (Claude/`nano-banana-pro` skill) | prompts | `assets/basemesh/reference/{slug}_{front,back}.png` |
| 2 | `gen_mesh.py` | bender (Hy3D 2.1) via SSH | front PNG | `assets/basemesh/raw_glb/{slug}_raw.glb` |
| 3 | `cleanup.py` | local (Blender headless) | raw GLB | `assets/basemesh/cleaned/{slug}.glb` |
| 4 | `bake_hmesh.py` | local (Python) | cleaned GLB | `assets/basemesh/{slug}.hmesh` |

Archetypes (slug → `human_archetype_t`):

| Slug | ID |
|------|----|
| `child_f` | 0 |
| `child_m` | 1 |
| `adult_f` | 2 |
| `adult_m` | 3 |
| `old_f` | 4 |
| `old_m` | 5 |

## Dependencies (one-time setup)

- **Local**: `pip install pygltflib pillow`
- **Blender**: 4.0+, headless invocations only
- **Bender host**: Hy3D 2.1 in `~/human/hy3d-env`, reachable via `ssh bender`

## Status

Stages 3–4 are documented stubs. They will be implemented when the first
raw GLB lands from stage 2. Stage 1 prints prompts; the operator runs the
`nano-banana-pro` skill manually to keep API spend visible.

## First archetype rollout

Order of work: `adult_m` first (most reference material, easiest to validate),
then `adult_f`, then the four child/old variants. Each archetype goes
through the full 4 stages before moving to the next — discovers pipeline
bugs early.
