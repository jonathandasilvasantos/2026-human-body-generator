#include "internal.h"
#include <stdlib.h>
#include <string.h>

/* declared in proto.c, used by human.h headers' clients indirectly */
float smoothstepf(float a, float b, float x);

/* declared in skeleton.c */
human_status_t human__skeleton_resolve_bind(human_t* h);
void           human__skeleton_compute_world(human_t* h, const hmat4_t* current_local);

/* declared in morph.c, skin.c */
void human__morph_apply(human_t* h);
void human__skin_apply(human_t* h);

/*
 * Common post-load wiring: resolve bind pose, allocate output buffers,
 * run an initial evaluate so vertex/bone buffers are populated.
 */
human_status_t human__finalize_load(human_t* h) {
    human_status_t st = human__skeleton_resolve_bind(h);
    if (st != HUMAN_OK) return st;
    h->out_vertices  = (float*)   calloc(h->vertex_count, 8 * sizeof(float));
    h->bone_world    = (float*)   calloc(h->bone_count,   16 * sizeof(float));
    h->skin_palette  = (hmat4_t*) calloc(h->bone_count,   sizeof(hmat4_t));
    h->morphed_pos   = (float*)   calloc(h->vertex_count, 3 * sizeof(float));
    h->morphed_norm  = (float*)   calloc(h->vertex_count, 3 * sizeof(float));
    if (!h->out_vertices || !h->bone_world || !h->skin_palette ||
        !h->morphed_pos || !h->morphed_norm) return HUMAN_ERR_OOM;
    return human_evaluate(h);
}

void human_free(human_t* h) {
    if (!h) return;
    free(h->rest_verts);
    free(h->indices);
    free(h->bones);
    if (h->morphs) {
        for (uint32_t i = 0; i < h->morph_count; ++i) free(h->morphs[i].deltas);
        free(h->morphs);
    }
    free(h->params);
    free(h->out_vertices);
    free(h->bone_world);
    free(h->skin_palette);
    free(h->morphed_pos);
    free(h->morphed_norm);
    free(h);
}

uint32_t human_vertex_count(const human_t* h) { return h ? h->vertex_count : 0; }
uint32_t human_index_count(const human_t* h)  { return h ? h->index_count : 0; }
const float*    human_vertex_buffer(const human_t* h) { return h ? h->out_vertices : NULL; }
const uint32_t* human_index_buffer(const human_t* h)  { return h ? h->indices : NULL; }

uint32_t human_bone_count(const human_t* h) { return h ? h->bone_count : 0; }
const float* human_bone_world_matrices(const human_t* h) { return h ? h->bone_world : NULL; }

uint32_t human_param_count(const human_t* h) { return h ? h->param_count : 0; }

const char* human_param_name(const human_t* h, uint32_t id) {
    if (!h || id >= h->param_count) return NULL;
    return h->params[id].name;
}

human_param_kind_t human_param_kind(const human_t* h, uint32_t id) {
    if (!h || id >= h->param_count) return HUMAN_PARAM_MORPH;
    return h->params[id].kind;
}

human_status_t human_param_range(const human_t* h, uint32_t id,
                                 float* lo, float* hi, float* def) {
    if (!h || id >= h->param_count) return HUMAN_ERR_RANGE;
    const param_t* p = &h->params[id];
    if (lo)  *lo  = p->min_val;
    if (hi)  *hi  = p->max_val;
    if (def) *def = p->default_val;
    return HUMAN_OK;
}

human_status_t human_set_param(human_t* h, uint32_t id, float value) {
    if (!h || id >= h->param_count) return HUMAN_ERR_RANGE;
    param_t* p = &h->params[id];
    if (value < p->min_val) value = p->min_val;
    if (value > p->max_val) value = p->max_val;
    p->value = value;
    return HUMAN_OK;
}

float human_get_param(const human_t* h, uint32_t id) {
    if (!h || id >= h->param_count) return 0.0f;
    return h->params[id].value;
}

human_status_t human_evaluate(human_t* h) {
    if (!h) return HUMAN_ERR_RANGE;

    /* build current local transforms = bind_local with bone-scale params applied */
    hmat4_t* cur = (hmat4_t*)calloc(h->bone_count, sizeof(hmat4_t));
    if (!cur && h->bone_count) return HUMAN_ERR_OOM;
    for (uint32_t i = 0; i < h->bone_count; ++i) cur[i] = h->bones[i].bind_local;
    for (uint32_t p = 0; p < h->param_count; ++p) {
        const param_t* pa = &h->params[p];
        if (pa->kind != HUMAN_PARAM_BONE_SCALE_Y) continue;
        if (pa->target_idx < 0 || (uint32_t)pa->target_idx >= h->bone_count) continue;
        hmat4_t s = hmat4_scale(hvec3(1.0f, pa->value, 1.0f));
        cur[pa->target_idx] = hmat4_mul(cur[pa->target_idx], s);
    }
    human__skeleton_compute_world(h, cur);
    free(cur);

    human__morph_apply(h);
    human__skin_apply(h);
    return HUMAN_OK;
}
