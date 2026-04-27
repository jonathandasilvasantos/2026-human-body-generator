#ifndef HUMAN_H
#define HUMAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HUMAN_VERSION_MAJOR 0
#define HUMAN_VERSION_MINOR 1
#define HUMAN_HMESH_MAGIC   0x48534D48u  /* "HMSH" little-endian */
#define HUMAN_HMESH_VERSION 1u

#define HUMAN_MAX_NAME      32
#define HUMAN_MAX_BONE_INFL 4

typedef enum {
    HUMAN_OK = 0,
    HUMAN_ERR_IO = -1,
    HUMAN_ERR_FORMAT = -2,
    HUMAN_ERR_OOM = -3,
    HUMAN_ERR_RANGE = -4,
} human_status_t;

typedef enum {
    HUMAN_ARCH_CHILD_F = 0,
    HUMAN_ARCH_CHILD_M = 1,
    HUMAN_ARCH_ADULT_F = 2,
    HUMAN_ARCH_ADULT_M = 3,
    HUMAN_ARCH_OLD_F   = 4,
    HUMAN_ARCH_OLD_M   = 5,
    HUMAN_ARCH_COUNT   = 6,
    HUMAN_ARCH_PROTO   = 0xFF,  /* synthetic capsule for testing */
} human_archetype_t;

typedef enum {
    HUMAN_PARAM_MORPH        = 0,  /* drives morph weight directly */
    HUMAN_PARAM_BONE_SCALE_Y = 1,  /* scales target bone along its local Y axis */
} human_param_kind_t;

typedef struct human_s human_t;

/* ---------- lifecycle ---------- */
human_status_t human_load(const char* path, human_t** out);
human_status_t human_save(const human_t* h, const char* path);
human_status_t human_create_proto(uint32_t archetype_hint, human_t** out);

/*
 * Build a human_t from flat arrays. Used by the offline bake pipeline.
 * Optional pointers may be NULL with the following defaults:
 *   normals       — recomputed from face winding (per-vertex average)
 *   uvs           — zero
 *   bone_ids      — all zero (single-bone skin)
 *   bone_weights  — (1,0,0,0) per vertex
 *   bind_locals   — identity per bone
 *   parents       — root for bone 0, -1; required if bone_count > 1
 * If bone_count == 0, a single identity root bone is created automatically.
 * The created human has zero morphs and zero parameters; add them later.
 */
human_status_t human_create_from_arrays(
    uint32_t archetype,
    uint32_t vertex_count,
    const float*    positions,     /* 3*vertex_count, required */
    const float*    normals,       /* 3*vertex_count, optional */
    const float*    uvs,           /* 2*vertex_count, optional */
    const uint8_t*  bone_ids,      /* 4*vertex_count, optional */
    const float*    bone_weights,  /* 4*vertex_count, optional */
    uint32_t        index_count,
    const uint32_t* indices,       /* index_count, required */
    uint32_t        bone_count,
    const int32_t*  parents,       /* bone_count, optional iff bone_count<=1 */
    const float*    bind_locals,   /* 16*bone_count column-major, optional */
    human_t**       out);

void           human_free(human_t* h);

/* ---------- mesh (output buffers, valid after human_evaluate) ---------- */
uint32_t        human_vertex_count(const human_t* h);
uint32_t        human_index_count(const human_t* h);
/* 8 floats per vertex, interleaved: pos.xyz, norm.xyz, uv.xy */
const float*    human_vertex_buffer(const human_t* h);
const uint32_t* human_index_buffer(const human_t* h);

/* ---------- skeleton (read-only) ---------- */
uint32_t        human_bone_count(const human_t* h);
/* 16 floats per bone, column-major model-space transform after evaluate */
const float*    human_bone_world_matrices(const human_t* h);

/* ---------- parameters ---------- */
uint32_t            human_param_count(const human_t* h);
const char*         human_param_name(const human_t* h, uint32_t id);
human_param_kind_t  human_param_kind(const human_t* h, uint32_t id);
human_status_t      human_param_range(const human_t* h, uint32_t id,
                                      float* out_min, float* out_max, float* out_default);
human_status_t      human_set_param(human_t* h, uint32_t id, float value);
float               human_get_param(const human_t* h, uint32_t id);

/* ---------- evaluation ---------- */
human_status_t      human_evaluate(human_t* h);

#ifdef __cplusplus
}
#endif

#endif
