from libc.stdint cimport uint32_t, int32_t

cdef extern from "human.h":
    ctypedef struct human_t

    ctypedef enum human_status_t:
        HUMAN_OK
        HUMAN_ERR_IO
        HUMAN_ERR_FORMAT
        HUMAN_ERR_OOM
        HUMAN_ERR_RANGE

    ctypedef enum human_param_kind_t:
        HUMAN_PARAM_MORPH
        HUMAN_PARAM_BONE_SCALE_Y

    int HUMAN_ARCH_PROTO

    human_status_t human_load(const char* path, human_t** out)
    human_status_t human_save(const human_t* h, const char* path)
    human_status_t human_create_proto(uint32_t archetype_hint, human_t** out)
    human_status_t human_create_from_arrays(
        uint32_t archetype,
        uint32_t vertex_count,
        const float*    positions,
        const float*    normals,
        const float*    uvs,
        const unsigned char*  bone_ids,
        const float*    bone_weights,
        uint32_t        index_count,
        const uint32_t* indices,
        uint32_t        bone_count,
        const int32_t*  parents,
        const float*    bind_locals,
        human_t**       out)
    void           human_free(human_t* h)

    uint32_t        human_vertex_count(const human_t* h)
    uint32_t        human_index_count(const human_t* h)
    const float*    human_vertex_buffer(const human_t* h)
    const uint32_t* human_index_buffer(const human_t* h)

    uint32_t        human_bone_count(const human_t* h)
    const float*    human_bone_world_matrices(const human_t* h)

    uint32_t        human_param_count(const human_t* h)
    const char*     human_param_name(const human_t* h, uint32_t id)
    human_param_kind_t human_param_kind(const human_t* h, uint32_t id)
    human_status_t  human_param_range(const human_t* h, uint32_t id,
                                      float* lo, float* hi, float* default_val)
    human_status_t  human_set_param(human_t* h, uint32_t id, float value)
    float           human_get_param(const human_t* h, uint32_t id)

    human_status_t  human_evaluate(human_t* h)
