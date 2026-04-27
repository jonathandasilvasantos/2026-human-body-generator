#include "human.h"
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * .hmesh layout, all little-endian:
 *
 * uint32 magic (HUMAN_HMESH_MAGIC)
 * uint32 version
 * uint32 archetype
 * uint32 vertex_count
 * uint32 index_count
 * uint32 bone_count
 * uint32 morph_count
 * uint32 param_count
 *
 * vert_rest_t[vertex_count]   (packed: 3f pos, 3f norm, 2f uv, 4u8 ids, 4f weights)
 * uint32[index_count]
 * for each bone: int32 parent, char[32] name, 16f bind_local
 * for each morph: char[32] name, uint32 delta_count, morph_delta_t[delta_count]
 * for each param: char[32] name, uint32 kind, int32 target_idx, 3f (min,max,default), 1f value
 */

static int read_all(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}
static int write_all(FILE* f, const void* buf, size_t n) {
    return fwrite(buf, 1, n, f) == n ? 0 : -1;
}

human_status_t human_save(const human_t* h, const char* path) {
    if (!h || !path) return HUMAN_ERR_RANGE;
    FILE* f = fopen(path, "wb");
    if (!f) return HUMAN_ERR_IO;
    uint32_t hdr[8] = {
        HUMAN_HMESH_MAGIC, HUMAN_HMESH_VERSION, h->archetype,
        h->vertex_count, h->index_count, h->bone_count,
        h->morph_count, h->param_count,
    };
    if (write_all(f, hdr, sizeof hdr)) goto err;
    /* per-vertex packed record matches vert_rest_t */
    for (uint32_t i = 0; i < h->vertex_count; ++i) {
        const vert_rest_t* v = &h->rest_verts[i];
        if (write_all(f, v->pos,  sizeof v->pos))  goto err;
        if (write_all(f, v->norm, sizeof v->norm)) goto err;
        if (write_all(f, v->uv,   sizeof v->uv))   goto err;
        if (write_all(f, v->bone_ids,     sizeof v->bone_ids))     goto err;
        if (write_all(f, v->bone_weights, sizeof v->bone_weights)) goto err;
    }
    if (h->index_count && write_all(f, h->indices, h->index_count * sizeof(uint32_t))) goto err;
    for (uint32_t i = 0; i < h->bone_count; ++i) {
        const bone_t* b = &h->bones[i];
        if (write_all(f, &b->parent,    sizeof b->parent))    goto err;
        if (write_all(f, b->name,       HUMAN_MAX_NAME))      goto err;
        if (write_all(f, b->bind_local.m, sizeof b->bind_local.m)) goto err;
    }
    for (uint32_t i = 0; i < h->morph_count; ++i) {
        const morph_t* m = &h->morphs[i];
        if (write_all(f, m->name, HUMAN_MAX_NAME))      goto err;
        if (write_all(f, &m->delta_count, sizeof m->delta_count)) goto err;
        if (m->delta_count && write_all(f, m->deltas, m->delta_count * sizeof(morph_delta_t))) goto err;
    }
    for (uint32_t i = 0; i < h->param_count; ++i) {
        const param_t* p = &h->params[i];
        uint32_t kind = (uint32_t)p->kind;
        if (write_all(f, p->name, HUMAN_MAX_NAME))     goto err;
        if (write_all(f, &kind, sizeof kind))          goto err;
        if (write_all(f, &p->target_idx, sizeof p->target_idx)) goto err;
        if (write_all(f, &p->min_val,    sizeof p->min_val))    goto err;
        if (write_all(f, &p->max_val,    sizeof p->max_val))    goto err;
        if (write_all(f, &p->default_val, sizeof p->default_val)) goto err;
        if (write_all(f, &p->value,      sizeof p->value))      goto err;
    }
    fclose(f);
    return HUMAN_OK;
err:
    fclose(f);
    return HUMAN_ERR_IO;
}

/* forward decl from human.c */
human_status_t human__finalize_load(human_t* h);

human_status_t human_load(const char* path, human_t** out) {
    if (!path || !out) return HUMAN_ERR_RANGE;
    *out = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return HUMAN_ERR_IO;

    uint32_t hdr[8];
    if (read_all(f, hdr, sizeof hdr)) { fclose(f); return HUMAN_ERR_IO; }
    if (hdr[0] != HUMAN_HMESH_MAGIC || hdr[1] != HUMAN_HMESH_VERSION) {
        fclose(f); return HUMAN_ERR_FORMAT;
    }

    human_t* h = (human_t*)calloc(1, sizeof(human_t));
    if (!h) { fclose(f); return HUMAN_ERR_OOM; }

    h->archetype    = hdr[2];
    h->vertex_count = hdr[3];
    h->index_count  = hdr[4];
    h->bone_count   = hdr[5];
    h->morph_count  = hdr[6];
    h->param_count  = hdr[7];

    h->rest_verts = (vert_rest_t*)calloc(h->vertex_count, sizeof(vert_rest_t));
    h->indices    = h->index_count ? (uint32_t*)calloc(h->index_count, sizeof(uint32_t)) : NULL;
    h->bones      = h->bone_count  ? (bone_t*)calloc(h->bone_count,    sizeof(bone_t))   : NULL;
    h->morphs     = h->morph_count ? (morph_t*)calloc(h->morph_count,  sizeof(morph_t))  : NULL;
    h->params     = h->param_count ? (param_t*)calloc(h->param_count,  sizeof(param_t))  : NULL;
    if ((!h->rest_verts && h->vertex_count) ||
        (!h->indices && h->index_count) ||
        (!h->bones && h->bone_count) ||
        (!h->morphs && h->morph_count) ||
        (!h->params && h->param_count)) {
        fclose(f); human_free(h); return HUMAN_ERR_OOM;
    }

    for (uint32_t i = 0; i < h->vertex_count; ++i) {
        vert_rest_t* v = &h->rest_verts[i];
        if (read_all(f, v->pos, sizeof v->pos) ||
            read_all(f, v->norm, sizeof v->norm) ||
            read_all(f, v->uv,   sizeof v->uv) ||
            read_all(f, v->bone_ids,     sizeof v->bone_ids) ||
            read_all(f, v->bone_weights, sizeof v->bone_weights))
        { fclose(f); human_free(h); return HUMAN_ERR_IO; }
    }
    if (h->index_count && read_all(f, h->indices, h->index_count * sizeof(uint32_t))) {
        fclose(f); human_free(h); return HUMAN_ERR_IO;
    }
    for (uint32_t i = 0; i < h->bone_count; ++i) {
        bone_t* b = &h->bones[i];
        if (read_all(f, &b->parent, sizeof b->parent) ||
            read_all(f, b->name, HUMAN_MAX_NAME) ||
            read_all(f, b->bind_local.m, sizeof b->bind_local.m))
        { fclose(f); human_free(h); return HUMAN_ERR_IO; }
    }
    for (uint32_t i = 0; i < h->morph_count; ++i) {
        morph_t* m = &h->morphs[i];
        if (read_all(f, m->name, HUMAN_MAX_NAME) ||
            read_all(f, &m->delta_count, sizeof m->delta_count))
        { fclose(f); human_free(h); return HUMAN_ERR_IO; }
        if (m->delta_count) {
            m->deltas = (morph_delta_t*)calloc(m->delta_count, sizeof(morph_delta_t));
            if (!m->deltas) { fclose(f); human_free(h); return HUMAN_ERR_OOM; }
            if (read_all(f, m->deltas, m->delta_count * sizeof(morph_delta_t))) {
                fclose(f); human_free(h); return HUMAN_ERR_IO;
            }
        }
    }
    for (uint32_t i = 0; i < h->param_count; ++i) {
        param_t* p = &h->params[i];
        uint32_t kind;
        if (read_all(f, p->name, HUMAN_MAX_NAME) ||
            read_all(f, &kind, sizeof kind) ||
            read_all(f, &p->target_idx, sizeof p->target_idx) ||
            read_all(f, &p->min_val,    sizeof p->min_val) ||
            read_all(f, &p->max_val,    sizeof p->max_val) ||
            read_all(f, &p->default_val, sizeof p->default_val) ||
            read_all(f, &p->value,      sizeof p->value))
        { fclose(f); human_free(h); return HUMAN_ERR_IO; }
        p->kind = (human_param_kind_t)kind;
    }
    fclose(f);

    human_status_t st = human__finalize_load(h);
    if (st != HUMAN_OK) { human_free(h); return st; }
    *out = h;
    return HUMAN_OK;
}
