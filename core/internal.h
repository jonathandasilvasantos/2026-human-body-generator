#ifndef HUMAN_INTERNAL_H
#define HUMAN_INTERNAL_H

#include "human.h"
#include "hmath.h"

typedef struct {
    /* rest pose, never mutated after load */
    float    pos[3];
    float    norm[3];
    float    uv[2];
    uint8_t  bone_ids[HUMAN_MAX_BONE_INFL];
    float    bone_weights[HUMAN_MAX_BONE_INFL];
} vert_rest_t;

typedef struct {
    int32_t  parent;
    char     name[HUMAN_MAX_NAME];
    hmat4_t  bind_local;     /* rest local transform (parent space) */
    hmat4_t  bind_world;     /* rest world transform, computed at load */
    hmat4_t  bind_world_inv; /* cached inverse, used by LBS */
} bone_t;

typedef struct {
    uint32_t vert_idx;
    float    dpos[3];
    float    dnorm[3];
} morph_delta_t;

typedef struct {
    char            name[HUMAN_MAX_NAME];
    uint32_t        delta_count;
    morph_delta_t*  deltas;
} morph_t;

typedef struct {
    char                name[HUMAN_MAX_NAME];
    human_param_kind_t  kind;
    int32_t             target_idx;   /* morph idx or bone idx depending on kind */
    float               min_val;
    float               max_val;
    float               default_val;
    float               value;
} param_t;

struct human_s {
    uint32_t      archetype;

    uint32_t      vertex_count;
    uint32_t      index_count;
    vert_rest_t*  rest_verts;
    uint32_t*     indices;

    uint32_t      bone_count;
    bone_t*       bones;

    uint32_t      morph_count;
    morph_t*      morphs;

    uint32_t      param_count;
    param_t*      params;

    /* output buffers, sized vertex_count */
    float*   out_vertices;       /* 8f interleaved: pos3 norm3 uv2 */
    float*   bone_world;         /* 16f per bone, column-major */
    hmat4_t* skin_palette;       /* per-bone world * bind_world_inv */

    /* scratch for morph accumulation */
    float*   morphed_pos;        /* 3f per vertex */
    float*   morphed_norm;       /* 3f per vertex */
};

#endif
