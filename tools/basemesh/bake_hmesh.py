"""Stage 4 — convert a GLB into a portable .hmesh.

Two modes:
  - **naive** (default): single root bone, all vertices fully bound to bone 0.
    Used for raw Hy3D outputs that lack skinning data. The resulting .hmesh
    renders correctly but cannot deform; height/weight params will not apply
    until the GLB is re-baked through the cleanup stage with a Mixamo rig.
  - **rigged**: parses the GLB's skin (bones + bind matrices + per-vertex
    JOINTS_0/WEIGHTS_0). Required output of the cleanup stage.

The GLB primitive is required to expose POSITION + indices. Other channels
(NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0) are used when present, otherwise
sensible defaults are filled in by the C builder (computed normals, zero UVs,
bone-0 binding).

Usage:
  python -m tools.basemesh.bake_hmesh --in path/to/raw.glb --out path/to/x.hmesh
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, BufferFormat

from bindings import Human


_DTYPES = {
    5120: ("i1", 1),  # BYTE
    5121: ("u1", 1),  # UBYTE
    5122: ("i2", 2),  # SHORT
    5123: ("u2", 2),  # USHORT
    5125: ("u4", 4),  # UINT
    5126: ("f4", 4),  # FLOAT
}
_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_accessor(g: GLTF2, idx: int | None) -> np.ndarray | None:
    if idx is None:
        return None
    acc = g.accessors[idx]
    if acc.bufferView is None:
        return None
    bv = g.bufferViews[acc.bufferView]
    buf = g.buffers[bv.buffer]
    raw = g.binary_blob() if buf.uri is None else g.get_data_from_buffer_uri(buf.uri)
    dt_code, comp_size = _DTYPES[acc.componentType]
    comps = _TYPE_COUNTS[acc.type]
    stride = bv.byteStride or comp_size * comps
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    out = np.empty((acc.count, comps), dtype=np.dtype(dt_code))
    elem_size = comp_size * comps
    if stride == elem_size:
        out[:] = np.frombuffer(raw, dtype=np.dtype(dt_code), count=acc.count * comps,
                               offset=offset).reshape(acc.count, comps)
    else:
        for i in range(acc.count):
            out[i] = np.frombuffer(raw, dtype=np.dtype(dt_code), count=comps,
                                    offset=offset + i * stride)
    return out


def bake(in_path: Path, out_path: Path, *, rigged: bool, mesh_idx: int = 0, prim_idx: int = 0,
         archetype: int = 0xFF) -> Human:
    g = GLTF2().load(str(in_path))
    g.convert_buffers(BufferFormat.BINARYBLOB)
    prim = g.meshes[mesh_idx].primitives[prim_idx]
    a = prim.attributes

    pos = _read_accessor(g, a.POSITION)
    if pos is None:
        raise ValueError("GLB missing POSITION")
    idx = _read_accessor(g, prim.indices)
    if idx is None:
        raise ValueError("GLB missing indices")
    idx = idx.reshape(-1).astype(np.uint32)

    nrm = _read_accessor(g, a.NORMAL)
    uvs = _read_accessor(g, a.TEXCOORD_0)
    nrm = nrm.astype(np.float32) if nrm is not None else None
    uvs = uvs.astype(np.float32) if uvs is not None else None
    pos = pos.astype(np.float32)

    bone_ids = bone_weights = None
    bone_parents = bind_locals = None

    if rigged:
        joints = _read_accessor(g, a.JOINTS_0)
        weights = _read_accessor(g, a.WEIGHTS_0)
        if joints is None or weights is None or not g.skins:
            raise ValueError("--rigged requires JOINTS_0/WEIGHTS_0 + skin in the GLB")
        bone_ids = joints.astype(np.uint8)
        bone_weights = weights.astype(np.float32)
        skin = g.skins[0]
        ibm = _read_accessor(g, skin.inverseBindMatrices)
        # Mixamo rigs store inverse-bind-matrices; we want bind_locals.
        # Reconstruction from IBM + node hierarchy is non-trivial; left for
        # the proper rigged-bake path that lands with cleanup.py.
        raise NotImplementedError(
            "rigged GLB bake: hierarchy/IBM->bind_local reconstruction lands with cleanup.py"
        )

    h = Human.from_arrays(
        pos, idx,
        normals=nrm, uvs=uvs,
        bone_ids=bone_ids, bone_weights=bone_weights,
        bone_parents=bone_parents, bind_locals=bind_locals,
        archetype=archetype,
    )
    h.save(out_path)
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--rigged", action="store_true",
                    help="parse skin + JOINTS_0/WEIGHTS_0 (requires Mixamo-rigged GLB)")
    ap.add_argument("--archetype", type=lambda x: int(x, 0), default=0xFF)
    args = ap.parse_args()
    h = bake(args.in_path, args.out_path, rigged=args.rigged, archetype=args.archetype)
    print(f"wrote {args.out_path}: {h.vertex_count} verts, {h.bone_count} bones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
