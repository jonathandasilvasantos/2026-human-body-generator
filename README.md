# human v2

Parametric human body generator. Six AI-generated base meshes (child / adult / old × F / M), rigged to a shared Mixamo skeleton, deformed by a portable C core, rendered by a Python app.

The previous Python-only implementation is preserved at the `v1-python-final` tag.

## Architecture

```
core/                pure C99, no deps beyond libc + libm — the portable runtime
bindings/            Cython wrapper exposing the C API to Python
app/                 moderngl + glfw + imgui renderer / parameter UI
tools/basemesh/      offline pipeline: Nano Banana Pro → Hy3D → Blender → .hmesh
assets/basemesh/     baked .hmesh archetypes (6 files)
```

The C core is intentionally minimal so it can be ported to Rust / Zig / WASM by reimplementing ~20 functions declared in `core/human.h`.

## Build

```
pip install -e .
human
```

Builds `libhuman.a` and the Cython extension in one step.

## Status

Phase 1 scaffold. See `notes/v2_rewrite_plan.md` for the staged roadmap.
