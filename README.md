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

Phases 1-3, 5, 7, 8 complete. Engine produces real geometry from a synthetic
proto archetype. Phases 4 / 6 (offline AI pipeline) are scaffolded — they
need bender SSH access and Nano Banana Pro quota to execute.

```
core/         libhuman.a  (C99, no deps beyond libc/libm)
              math + skeleton + LBS + morph + param dispatch + .hmesh I/O
              + synthetic proto generator + C unit tests
bindings/     Cython wrapper, NumPy zero-copy buffer protocol
app/          offscreen moderngl renderer, CLI: proto / render / triview
tools/basemesh/   stage 1 prompts ready, stages 2-4 stubbed with clear plan
tests/        pytest end-to-end (proto, height, weight, roundtrip, normals)
```

Quick smoke:
```
human proto /tmp/p.hmesh
human triview proto -p height=1.3 -p weight=1.2 --out /tmp/posed.png
```

See `notes/v2_rewrite_plan.md` for the full roadmap.
