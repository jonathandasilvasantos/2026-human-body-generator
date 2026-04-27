from libc.stdint cimport uint32_t

cdef extern from "human.h":
    ctypedef struct human_t

    ctypedef enum human_status_t:
        HUMAN_OK
        HUMAN_ERR_IO
        HUMAN_ERR_FORMAT
        HUMAN_ERR_OOM
        HUMAN_ERR_RANGE

    human_status_t human_load(const char* path, human_t** out)
    void           human_free(human_t* h)

    uint32_t       human_vertex_count(const human_t* h)
    uint32_t       human_index_count(const human_t* h)
    const float*   human_vertex_buffer(const human_t* h)
    const uint32_t* human_index_buffer(const human_t* h)

    uint32_t       human_param_count(const human_t* h)
    human_status_t human_set_param(human_t* h, uint32_t id, float value)
    float          human_get_param(const human_t* h, uint32_t id)

    human_status_t human_evaluate(human_t* h)
