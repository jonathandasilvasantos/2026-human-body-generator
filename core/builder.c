#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

human_status_t human__finalize_load(human_t* h);

static void compute_face_normals(uint32_t vc, const float* positions,
                                 uint32_t ic, const uint32_t* indices,
                                 float* out_norm) {
    for (uint32_t i = 0; i < vc * 3; ++i) out_norm[i] = 0.0f;
    for (uint32_t t = 0; t + 2 < ic; t += 3) {
        uint32_t a = indices[t+0], b = indices[t+1], c = indices[t+2];
        if (a >= vc || b >= vc || c >= vc) continue;
        const float* pa = &positions[a*3];
        const float* pb = &positions[b*3];
        const float* pc = &positions[c*3];
        float e1[3] = { pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2] };
        float e2[3] = { pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2] };
        float n[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        };
        for (int k = 0; k < 3; ++k) {
            out_norm[a*3+k] += n[k];
            out_norm[b*3+k] += n[k];
            out_norm[c*3+k] += n[k];
        }
    }
    for (uint32_t i = 0; i < vc; ++i) {
        float* n = &out_norm[i*3];
        float l = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (l > 1e-12f) { n[0]/=l; n[1]/=l; n[2]/=l; }
        else { n[0]=0; n[1]=1; n[2]=0; }
    }
}

human_status_t human_create_from_arrays(
    uint32_t archetype,
    uint32_t vertex_count,
    const float*    positions,
    const float*    normals,
    const float*    uvs,
    const uint8_t*  bone_ids,
    const float*    bone_weights,
    uint32_t        index_count,
    const uint32_t* indices,
    uint32_t        bone_count,
    const int32_t*  parents,
    const float*    bind_locals,
    human_t**       out)
{
    if (!out || !positions || !indices || vertex_count == 0 || index_count == 0)
        return HUMAN_ERR_RANGE;
    *out = NULL;

    if (bone_count == 0) bone_count = 1;
    if (bone_count > 1 && !parents) return HUMAN_ERR_RANGE;

    human_t* h = (human_t*)calloc(1, sizeof(human_t));
    if (!h) return HUMAN_ERR_OOM;
    h->archetype    = archetype;
    h->vertex_count = vertex_count;
    h->index_count  = index_count;
    h->bone_count   = bone_count;
    h->morph_count  = 0;
    h->param_count  = 0;

    h->rest_verts = (vert_rest_t*)calloc(vertex_count, sizeof(vert_rest_t));
    h->indices    = (uint32_t*)   calloc(index_count, sizeof(uint32_t));
    h->bones      = (bone_t*)     calloc(bone_count, sizeof(bone_t));
    if (!h->rest_verts || !h->indices || !h->bones) { human_free(h); return HUMAN_ERR_OOM; }

    /* normals: provided or computed */
    float* tmp_norm = NULL;
    const float* eff_norm = normals;
    if (!eff_norm) {
        tmp_norm = (float*)calloc(vertex_count * 3, sizeof(float));
        if (!tmp_norm) { human_free(h); return HUMAN_ERR_OOM; }
        compute_face_normals(vertex_count, positions, index_count, indices, tmp_norm);
        eff_norm = tmp_norm;
    }

    for (uint32_t i = 0; i < vertex_count; ++i) {
        vert_rest_t* v = &h->rest_verts[i];
        v->pos[0] = positions[i*3+0];
        v->pos[1] = positions[i*3+1];
        v->pos[2] = positions[i*3+2];
        v->norm[0] = eff_norm[i*3+0];
        v->norm[1] = eff_norm[i*3+1];
        v->norm[2] = eff_norm[i*3+2];
        if (uvs) { v->uv[0] = uvs[i*2+0]; v->uv[1] = uvs[i*2+1]; }
        else     { v->uv[0] = 0.0f;       v->uv[1] = 0.0f; }
        if (bone_ids) {
            for (int k = 0; k < HUMAN_MAX_BONE_INFL; ++k) v->bone_ids[k] = bone_ids[i*4+k];
        }
        if (bone_weights) {
            for (int k = 0; k < HUMAN_MAX_BONE_INFL; ++k) v->bone_weights[k] = bone_weights[i*4+k];
        } else {
            v->bone_weights[0] = 1.0f;
            v->bone_weights[1] = v->bone_weights[2] = v->bone_weights[3] = 0.0f;
        }
    }
    free(tmp_norm);

    memcpy(h->indices, indices, index_count * sizeof(uint32_t));

    for (uint32_t i = 0; i < bone_count; ++i) {
        bone_t* b = &h->bones[i];
        b->parent = (i == 0 || !parents) ? -1 : parents[i];
        if (i == 0 && parents) b->parent = parents[0];
        snprintf(b->name, HUMAN_MAX_NAME, "bone_%u", i);
        if (bind_locals) {
            for (int k = 0; k < 16; ++k) b->bind_local.m[k] = bind_locals[i*16 + k];
        } else {
            b->bind_local = hmat4_identity();
        }
    }

    human_status_t st = human__finalize_load(h);
    if (st != HUMAN_OK) { human_free(h); return st; }
    *out = h;
    return HUMAN_OK;
}
