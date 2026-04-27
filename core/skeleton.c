#include "internal.h"

/*
 * Compute bind_world for every bone from bind_local + parent chain.
 * Bones must be topologically ordered (parent index < child index).
 * Also computes bind_world_inv used by the LBS skinning palette.
 */
human_status_t human__skeleton_resolve_bind(human_t* h) {
    for (uint32_t i = 0; i < h->bone_count; ++i) {
        bone_t* b = &h->bones[i];
        if (b->parent < 0) {
            b->bind_world = b->bind_local;
        } else if ((uint32_t)b->parent >= i) {
            return HUMAN_ERR_FORMAT;  /* not topologically sorted */
        } else {
            b->bind_world = hmat4_mul(h->bones[b->parent].bind_world, b->bind_local);
        }
        b->bind_world_inv = hmat4_inverse(b->bind_world);
    }
    return HUMAN_OK;
}

/*
 * Recompute world matrices given current local transforms (which may have been
 * adjusted by BONE_SCALE_Y parameters). Stores results in h->bone_world (16f
 * column-major per bone) and h->skin_palette (world * bind_world_inv).
 */
void human__skeleton_compute_world(human_t* h, const hmat4_t* current_local) {
    for (uint32_t i = 0; i < h->bone_count; ++i) {
        const bone_t* b = &h->bones[i];
        hmat4_t world;
        if (b->parent < 0) {
            world = current_local[i];
        } else {
            hmat4_t parent_world;
            for (int k = 0; k < 16; ++k) parent_world.m[k] = h->bone_world[b->parent * 16 + k];
            world = hmat4_mul(parent_world, current_local[i]);
        }
        for (int k = 0; k < 16; ++k) h->bone_world[i * 16 + k] = world.m[k];
        h->skin_palette[i] = hmat4_mul(world, b->bind_world_inv);
    }
}
