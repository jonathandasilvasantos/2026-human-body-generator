#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float smoothstepf(float a, float b, float x);

/*
 * Synthetic capsule humanoid for testing the engine end-to-end without the
 * offline AI pipeline. A 16-longitude x 12-latitude capsule (~2m tall),
 * rigged to a 3-bone Y-axis chain (pelvis -> spine -> head), with two
 * morphs ("weight" radial expansion, "head_size" cap expansion) and three
 * parameters ("height", "weight", "head_size").
 */

#define PROTO_LON 16
#define PROTO_LAT 12

static void set_name(char dst[HUMAN_MAX_NAME], const char* src) {
    size_t n = strlen(src);
    if (n >= HUMAN_MAX_NAME) n = HUMAN_MAX_NAME - 1;
    memcpy(dst, src, n);
    dst[n] = 0;
}

human_status_t human__finalize_load(human_t* h);  /* in human.c */

human_status_t human_create_proto(uint32_t archetype_hint, human_t** out) {
    if (!out) return HUMAN_ERR_RANGE;
    *out = NULL;

    human_t* h = (human_t*)calloc(1, sizeof(human_t));
    if (!h) return HUMAN_ERR_OOM;
    h->archetype = archetype_hint;

    /* ---------- bones: 3-link chain along +Y ---------- */
    h->bone_count = 3;
    h->bones = (bone_t*)calloc(3, sizeof(bone_t));
    if (!h->bones) { human_free(h); return HUMAN_ERR_OOM; }
    set_name(h->bones[0].name, "pelvis");
    set_name(h->bones[1].name, "spine");
    set_name(h->bones[2].name, "head");
    h->bones[0].parent = -1;
    h->bones[1].parent = 0;
    h->bones[2].parent = 1;
    /* pelvis at world origin Y=0; spine offset +0.6 in pelvis space; head +0.7 in spine space */
    h->bones[0].bind_local = hmat4_translate(hvec3(0, 0.0f, 0));
    h->bones[1].bind_local = hmat4_translate(hvec3(0, 0.6f, 0));
    h->bones[2].bind_local = hmat4_translate(hvec3(0, 0.7f, 0));

    /* ---------- mesh: capsule from y=-0.9 to y=+1.1 ---------- */
    const int cols = PROTO_LON;
    const int rows = PROTO_LAT;
    h->vertex_count = (uint32_t)(cols * (rows + 1));
    h->index_count  = (uint32_t)(cols * rows * 6);
    h->rest_verts = (vert_rest_t*)calloc(h->vertex_count, sizeof(vert_rest_t));
    h->indices    = (uint32_t*)   calloc(h->index_count,  sizeof(uint32_t));
    if (!h->rest_verts || !h->indices) { human_free(h); return HUMAN_ERR_OOM; }

    const float y_min = -0.9f, y_max = 1.6f;
    const float radius = 0.25f;
    for (int r = 0; r <= rows; ++r) {
        float t = (float)r / (float)rows;
        float y = y_min + (y_max - y_min) * t;
        /* capsule: pinch the ends so we approximate a body silhouette */
        float pinch = sinf(t * 3.14159265f);
        float ring_r = radius * (0.6f + 0.4f * pinch);
        for (int c = 0; c < cols; ++c) {
            float a = (float)c / (float)cols * 6.2831853f;
            float cx = cosf(a), cz = sinf(a);
            uint32_t i = (uint32_t)(r * cols + c);
            vert_rest_t* v = &h->rest_verts[i];
            v->pos[0]  = ring_r * cx;
            v->pos[1]  = y;
            v->pos[2]  = ring_r * cz;
            v->norm[0] = cx; v->norm[1] = 0.0f; v->norm[2] = cz;
            v->uv[0] = (float)c / (float)cols;
            v->uv[1] = t;
            /* skinning: pelvis (y<0.6), spine (0.6..1.3), head (>1.3), with smooth blends */
            float w_pelvis = 1.0f - smoothstepf(0.3f, 0.7f, y);
            float w_head   = smoothstepf(1.2f, 1.5f, y);
            float w_spine  = 1.0f - w_pelvis - w_head;
            if (w_spine < 0) w_spine = 0;
            float sum = w_pelvis + w_spine + w_head;
            if (sum > 0) { w_pelvis /= sum; w_spine /= sum; w_head /= sum; }
            v->bone_ids[0]     = 0; v->bone_weights[0] = w_pelvis;
            v->bone_ids[1]     = 1; v->bone_weights[1] = w_spine;
            v->bone_ids[2]     = 2; v->bone_weights[2] = w_head;
            v->bone_ids[3]     = 0; v->bone_weights[3] = 0.0f;
        }
    }
    /* triangulate */
    uint32_t* idx = h->indices;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int c1 = (c + 1) % cols;
            uint32_t a = (uint32_t)(r * cols + c);
            uint32_t b = (uint32_t)(r * cols + c1);
            uint32_t cc= (uint32_t)((r+1) * cols + c);
            uint32_t d = (uint32_t)((r+1) * cols + c1);
            *idx++ = a; *idx++ = cc; *idx++ = b;
            *idx++ = b; *idx++ = cc; *idx++ = d;
        }
    }

    /* ---------- morphs ---------- */
    h->morph_count = 2;
    h->morphs = (morph_t*)calloc(2, sizeof(morph_t));
    if (!h->morphs) { human_free(h); return HUMAN_ERR_OOM; }
    set_name(h->morphs[0].name, "weight");
    set_name(h->morphs[1].name, "head_size");
    h->morphs[0].delta_count = h->vertex_count;
    h->morphs[1].delta_count = h->vertex_count;
    h->morphs[0].deltas = (morph_delta_t*)calloc(h->vertex_count, sizeof(morph_delta_t));
    h->morphs[1].deltas = (morph_delta_t*)calloc(h->vertex_count, sizeof(morph_delta_t));
    if (!h->morphs[0].deltas || !h->morphs[1].deltas) { human_free(h); return HUMAN_ERR_OOM; }
    for (uint32_t i = 0; i < h->vertex_count; ++i) {
        const vert_rest_t* v = &h->rest_verts[i];
        /* "weight": radial expansion strongest at mid-torso */
        float belly = 1.0f - fabsf(v->pos[1] - 0.4f);
        if (belly < 0) belly = 0;
        h->morphs[0].deltas[i].vert_idx = i;
        h->morphs[0].deltas[i].dpos[0]  = v->norm[0] * 0.18f * belly;
        h->morphs[0].deltas[i].dpos[1]  = 0.0f;
        h->morphs[0].deltas[i].dpos[2]  = v->norm[2] * 0.18f * belly;
        /* "head_size": radial expansion above 1.3 */
        float head = smoothstepf(1.2f, 1.5f, v->pos[1]);
        h->morphs[1].deltas[i].vert_idx = i;
        h->morphs[1].deltas[i].dpos[0]  = v->norm[0] * 0.20f * head;
        h->morphs[1].deltas[i].dpos[1]  = head * 0.10f;
        h->morphs[1].deltas[i].dpos[2]  = v->norm[2] * 0.20f * head;
    }

    /* ---------- params ---------- */
    h->param_count = 3;
    h->params = (param_t*)calloc(3, sizeof(param_t));
    if (!h->params) { human_free(h); return HUMAN_ERR_OOM; }
    set_name(h->params[0].name, "height");
    h->params[0].kind = HUMAN_PARAM_BONE_SCALE_Y;
    h->params[0].target_idx = 1; /* spine */
    h->params[0].min_val = 0.6f; h->params[0].max_val = 1.4f;
    h->params[0].default_val = 1.0f; h->params[0].value = 1.0f;
    set_name(h->params[1].name, "weight");
    h->params[1].kind = HUMAN_PARAM_MORPH;
    h->params[1].target_idx = 0;
    h->params[1].min_val = 0.0f; h->params[1].max_val = 1.5f;
    h->params[1].default_val = 0.0f; h->params[1].value = 0.0f;
    set_name(h->params[2].name, "head_size");
    h->params[2].kind = HUMAN_PARAM_MORPH;
    h->params[2].target_idx = 1;
    h->params[2].min_val = 0.0f; h->params[2].max_val = 1.5f;
    h->params[2].default_val = 0.0f; h->params[2].value = 0.0f;

    human_status_t st = human__finalize_load(h);
    if (st != HUMAN_OK) { human_free(h); return st; }
    *out = h;
    return HUMAN_OK;
}

float smoothstepf(float a, float b, float x) {
    if (b == a) return x < a ? 0.0f : 1.0f;
    float t = (x - a) / (b - a);
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return t * t * (3.0f - 2.0f * t);
}
