#include "internal.h"

/*
 * Linear blend skinning. For each vertex: blend the per-bone palette matrices
 * using the vertex's bone weights, transform morphed pos/norm by the result,
 * write 8f interleaved (pos3 norm3 uv2) into h->out_vertices.
 */
void human__skin_apply(human_t* h) {
    for (uint32_t i = 0; i < h->vertex_count; ++i) {
        const vert_rest_t* v = &h->rest_verts[i];
        hvec3_t mp = hvec3(h->morphed_pos[i*3+0],  h->morphed_pos[i*3+1],  h->morphed_pos[i*3+2]);
        hvec3_t mn = hvec3(h->morphed_norm[i*3+0], h->morphed_norm[i*3+1], h->morphed_norm[i*3+2]);

        hvec3_t out_p = hvec3(0,0,0);
        hvec3_t out_n = hvec3(0,0,0);
        float total_w = 0.0f;

        for (int k = 0; k < HUMAN_MAX_BONE_INFL; ++k) {
            float w = v->bone_weights[k];
            if (w <= 0.0f) continue;
            uint8_t bid = v->bone_ids[k];
            if (bid >= h->bone_count) continue;
            hmat4_t M = h->skin_palette[bid];
            hvec3_t p = hmat4_mul_point(M, mp);
            hvec3_t n = hmat4_mul_dir(M, mn);
            out_p = hvec3_add(out_p, hvec3_scale(p, w));
            out_n = hvec3_add(out_n, hvec3_scale(n, w));
            total_w += w;
        }
        if (total_w == 0.0f) { out_p = mp; out_n = mn; }
        else if (total_w != 1.0f) {
            float s = 1.0f / total_w;
            out_p = hvec3_scale(out_p, s);
            out_n = hvec3_scale(out_n, s);
        }
        out_n = hvec3_norm(out_n);

        float* o = &h->out_vertices[i * 8];
        o[0] = out_p.x; o[1] = out_p.y; o[2] = out_p.z;
        o[3] = out_n.x; o[4] = out_n.y; o[5] = out_n.z;
        o[6] = v->uv[0]; o[7] = v->uv[1];
    }
}
