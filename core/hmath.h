#ifndef HUMAN_MATH_H
#define HUMAN_MATH_H

#include <math.h>

typedef struct { float x, y, z; }    hvec3_t;
typedef struct { float x, y, z, w; } hquat_t;
typedef struct { float m[16]; }      hmat4_t; /* column-major */

static inline hvec3_t hvec3(float x, float y, float z) {
    hvec3_t v = { x, y, z }; return v;
}
static inline hvec3_t hvec3_add(hvec3_t a, hvec3_t b) { return hvec3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline hvec3_t hvec3_sub(hvec3_t a, hvec3_t b) { return hvec3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline hvec3_t hvec3_scale(hvec3_t a, float s)  { return hvec3(a.x*s, a.y*s, a.z*s); }
static inline float   hvec3_dot(hvec3_t a, hvec3_t b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline hvec3_t hvec3_cross(hvec3_t a, hvec3_t b) {
    return hvec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float   hvec3_len(hvec3_t a) { return sqrtf(hvec3_dot(a, a)); }
static inline hvec3_t hvec3_norm(hvec3_t a) {
    float l = hvec3_len(a);
    return l > 1e-12f ? hvec3_scale(a, 1.0f / l) : hvec3(0, 0, 0);
}

hmat4_t hmat4_identity(void);
hmat4_t hmat4_translate(hvec3_t t);
hmat4_t hmat4_scale(hvec3_t s);
hmat4_t hmat4_mul(hmat4_t a, hmat4_t b);
hmat4_t hmat4_inverse(hmat4_t m);
hvec3_t hmat4_mul_point(hmat4_t m, hvec3_t p);
hvec3_t hmat4_mul_dir(hmat4_t m, hvec3_t d);

#endif
