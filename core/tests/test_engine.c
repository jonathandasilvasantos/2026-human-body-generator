#include "human.h"
#include "hmath.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #expr); exit(1); } \
} while (0)

static void test_math(void) {
    hmat4_t I = hmat4_identity();
    hmat4_t T = hmat4_translate(hvec3(1, 2, 3));
    hmat4_t Ti = hmat4_inverse(T);
    hmat4_t R = hmat4_mul(T, Ti);
    for (int i = 0; i < 16; ++i) {
        float exp = I.m[i];
        CHECK(fabsf(R.m[i] - exp) < 1e-5f);
    }
    hvec3_t p = hmat4_mul_point(T, hvec3(0, 0, 0));
    CHECK(fabsf(p.x - 1.0f) < 1e-6f && fabsf(p.y - 2.0f) < 1e-6f && fabsf(p.z - 3.0f) < 1e-6f);
}

static float bbox_y_max(const float* verts, uint32_t n) {
    float ymax = -1e30f;
    for (uint32_t i = 0; i < n; ++i) if (verts[i*8+1] > ymax) ymax = verts[i*8+1];
    return ymax;
}
static float bbox_radius_xz(const float* verts, uint32_t n) {
    float r = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float x = verts[i*8+0], z = verts[i*8+2];
        float d = sqrtf(x*x + z*z);
        if (d > r) r = d;
    }
    return r;
}

static void test_proto_and_params(void) {
    human_t* h = NULL;
    CHECK(human_create_proto(HUMAN_ARCH_PROTO, &h) == HUMAN_OK);
    CHECK(h != NULL);
    CHECK(human_vertex_count(h) > 0);
    CHECK(human_index_count(h) % 3 == 0);
    CHECK(human_param_count(h) == 3);
    CHECK(strcmp(human_param_name(h, 0), "height") == 0);

    /* baseline */
    const float* verts = human_vertex_buffer(h);
    float y0 = bbox_y_max(verts, human_vertex_count(h));
    float r0 = bbox_radius_xz(verts, human_vertex_count(h));

    /* increase height: bone_scale_y on spine -> top should move up */
    CHECK(human_set_param(h, 0, 1.4f) == HUMAN_OK);
    CHECK(human_evaluate(h) == HUMAN_OK);
    float y1 = bbox_y_max(verts, human_vertex_count(h));
    CHECK(y1 > y0 + 0.05f);

    /* reset height, increase weight morph -> radius should grow */
    CHECK(human_set_param(h, 0, 1.0f) == HUMAN_OK);
    CHECK(human_set_param(h, 1, 1.0f) == HUMAN_OK);
    CHECK(human_evaluate(h) == HUMAN_OK);
    float r1 = bbox_radius_xz(verts, human_vertex_count(h));
    CHECK(r1 > r0 + 0.02f);

    /* clamping */
    CHECK(human_set_param(h, 1, 999.0f) == HUMAN_OK);
    CHECK(human_get_param(h, 1) <= 1.5f + 1e-6f);

    human_free(h);
}

static void test_io_roundtrip(void) {
    human_t* a = NULL;
    CHECK(human_create_proto(HUMAN_ARCH_PROTO, &a) == HUMAN_OK);
    CHECK(human_set_param(a, 1, 0.7f) == HUMAN_OK);
    const char* path = "/tmp/human_test.hmesh";
    CHECK(human_save(a, path) == HUMAN_OK);

    human_t* b = NULL;
    CHECK(human_load(path, &b) == HUMAN_OK);
    CHECK(human_vertex_count(a) == human_vertex_count(b));
    CHECK(human_index_count(a) == human_index_count(b));
    CHECK(human_bone_count(a) == human_bone_count(b));
    CHECK(human_param_count(a) == human_param_count(b));
    CHECK(fabsf(human_get_param(a, 1) - human_get_param(b, 1)) < 1e-6f);

    /* same param values -> same evaluated geometry */
    CHECK(human_evaluate(a) == HUMAN_OK);
    CHECK(human_evaluate(b) == HUMAN_OK);
    const float* va = human_vertex_buffer(a);
    const float* vb = human_vertex_buffer(b);
    for (uint32_t i = 0; i < human_vertex_count(a) * 8; ++i) {
        CHECK(fabsf(va[i] - vb[i]) < 1e-5f);
    }
    human_free(a); human_free(b);
}

int main(void) {
    test_math();
    test_proto_and_params();
    test_io_roundtrip();
    printf("OK\n");
    return 0;
}
