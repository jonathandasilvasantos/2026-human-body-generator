#ifndef HUMAN_H
#define HUMAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HUMAN_VERSION_MAJOR 0
#define HUMAN_VERSION_MINOR 1

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
} human_archetype_t;

typedef struct human_s human_t;

human_status_t human_load(const char* path, human_t** out);
void           human_free(human_t* h);

uint32_t       human_vertex_count(const human_t* h);
uint32_t       human_index_count(const human_t* h);
const float*   human_vertex_buffer(const human_t* h);
const uint32_t* human_index_buffer(const human_t* h);

uint32_t       human_param_count(const human_t* h);
human_status_t human_set_param(human_t* h, uint32_t id, float value);
float          human_get_param(const human_t* h, uint32_t id);

human_status_t human_evaluate(human_t* h);

#ifdef __cplusplus
}
#endif

#endif
