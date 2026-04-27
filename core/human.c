#include "human.h"
#include <stdlib.h>

struct human_s {
    uint32_t vertex_count;
    uint32_t index_count;
    float*   vertices;
    uint32_t* indices;
    uint32_t param_count;
    float*   params;
};

human_status_t human_load(const char* path, human_t** out) {
    (void)path;
    if (!out) return HUMAN_ERR_RANGE;
    *out = NULL;
    return HUMAN_ERR_IO;
}

void human_free(human_t* h) {
    if (!h) return;
    free(h->vertices);
    free(h->indices);
    free(h->params);
    free(h);
}

uint32_t human_vertex_count(const human_t* h) { return h ? h->vertex_count : 0; }
uint32_t human_index_count(const human_t* h)  { return h ? h->index_count : 0; }
const float*    human_vertex_buffer(const human_t* h) { return h ? h->vertices : NULL; }
const uint32_t* human_index_buffer(const human_t* h)  { return h ? h->indices : NULL; }

uint32_t human_param_count(const human_t* h) { return h ? h->param_count : 0; }

human_status_t human_set_param(human_t* h, uint32_t id, float value) {
    if (!h || id >= h->param_count) return HUMAN_ERR_RANGE;
    h->params[id] = value;
    return HUMAN_OK;
}

float human_get_param(const human_t* h, uint32_t id) {
    if (!h || id >= h->param_count) return 0.0f;
    return h->params[id];
}

human_status_t human_evaluate(human_t* h) {
    if (!h) return HUMAN_ERR_RANGE;
    return HUMAN_OK;
}
