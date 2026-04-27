#include "internal.h"
#include <string.h>

/*
 * Reset morphed buffers to rest pose, then accumulate weighted morph deltas.
 * Morph weights come from any HUMAN_PARAM_MORPH parameters that target a morph.
 */
void human__morph_apply(human_t* h) {
    /* reset to rest */
    for (uint32_t i = 0; i < h->vertex_count; ++i) {
        const vert_rest_t* v = &h->rest_verts[i];
        h->morphed_pos[i*3+0]  = v->pos[0];
        h->morphed_pos[i*3+1]  = v->pos[1];
        h->morphed_pos[i*3+2]  = v->pos[2];
        h->morphed_norm[i*3+0] = v->norm[0];
        h->morphed_norm[i*3+1] = v->norm[1];
        h->morphed_norm[i*3+2] = v->norm[2];
    }
    /* gather weights per morph from params */
    for (uint32_t p = 0; p < h->param_count; ++p) {
        const param_t* pa = &h->params[p];
        if (pa->kind != HUMAN_PARAM_MORPH) continue;
        if (pa->target_idx < 0 || (uint32_t)pa->target_idx >= h->morph_count) continue;
        float w = pa->value;
        if (w == 0.0f) continue;
        const morph_t* m = &h->morphs[pa->target_idx];
        for (uint32_t d = 0; d < m->delta_count; ++d) {
            const morph_delta_t* dd = &m->deltas[d];
            if (dd->vert_idx >= h->vertex_count) continue;
            h->morphed_pos[dd->vert_idx*3+0]  += w * dd->dpos[0];
            h->morphed_pos[dd->vert_idx*3+1]  += w * dd->dpos[1];
            h->morphed_pos[dd->vert_idx*3+2]  += w * dd->dpos[2];
            h->morphed_norm[dd->vert_idx*3+0] += w * dd->dnorm[0];
            h->morphed_norm[dd->vert_idx*3+1] += w * dd->dnorm[1];
            h->morphed_norm[dd->vert_idx*3+2] += w * dd->dnorm[2];
        }
    }
}
