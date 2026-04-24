"""GL resources: shader programs, skinned-mesh GPU buffers, skeleton lines."""

import numpy as np
from OpenGL.GL import *


MAX_BONES = 32


SKIN_VERT = """
#version 330 core
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_nrm;
layout(location=2) in ivec2 a_bones;
layout(location=3) in vec2  a_weights;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_bones[""" + str(MAX_BONES) + """];
// Outward inflate along the vertex normal, scaled by the per-vertex
// bend scalar. Used to push fabric off the body at elbow/knee folds so
// a tight sleeve or pant leg doesn't let skin poke through when the
// joint flexes. Set to 0 for skin and non-garment meshes.
uniform float u_bend_inflate;

out vec3 v_nrm;
out vec3 v_pos;
out vec3 v_local;
// Pose-space bend scalar: positive where the two skinning bones' rotations
// diverge (elbow/knee/waist flexion seam on a dual-skinned garment). Zero
// at rest pose and for single-bone-skinned vertices. Fed into the fabric
// fragment shader as a wrinkle-intensity mask so cloth crumples at joints.
out float v_bend;

void main() {
    mat4 Ma = u_bones[a_bones.x];
    mat4 Mb = u_bones[a_bones.y];
    mat4 M = Ma * a_weights.x + Mb * a_weights.y;

    vec3 da = mat3(Ma) * a_pos;
    vec3 db = mat3(Mb) * a_pos;
    float blend = a_weights.x * a_weights.y;
    float bend_raw = clamp(length(da - db) * blend * 4.0, 0.0, 1.2);
    v_bend = bend_raw;

    // Push the vertex outward along its (skinned) world normal by a
    // small amount proportional to bend. A tight sleeve / pant leg
    // puffs off the skin at the elbow / knee so the body mesh doesn't
    // clip through. Amount is modest because heavy inflation looks
    // balloon-like; the improvement to clip-through is worth the
    // silhouette softening.
    vec3 world_nrm = normalize(mat3(M) * a_nrm);
    vec3 p_world = (M * vec4(a_pos, 1.0)).xyz
                    + world_nrm * u_bend_inflate * bend_raw;
    gl_Position = u_proj * u_view * vec4(p_world, 1.0);
    v_pos = p_world;
    v_nrm = world_nrm;
    v_local = a_pos;
}
"""

SKIN_FRAG = """
#version 330 core
in vec3 v_nrm;
in vec3 v_pos;
in vec3 v_local;
in float v_bend;
out vec4 frag;

uniform vec3  u_color;
uniform int   u_mode;   // 0 = skin, 1 = fabric, 2 = hair, 3 = eye, 4 = shoe
uniform float u_seed;   // per-character random seed in [0,1]
uniform int   u_light_style; // 0=portrait 1=soft 2=raking 3=warm/cool
// Clothing pattern controls (per-character; Python-driven for guaranteed
// variety instead of hash-gated). 0 means "no effect".
uniform int   u_print_style;    // 0=none 1=stripes 2=dots 3=plaid 4=noise
uniform float u_print_strength; // 0..1
uniform int   u_stamp_style;    // 0=none 1=ring 2=diamond-ring 3=cross 4=star
uniform float u_stamp_strength; // 0..1
uniform int   u_material;       // fabric type when u_mode==1: 0 cotton, 1 denim, 2 silk, 3 knit

// cheap 3D hash -> [0,1]
float hash3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// value noise (trilinear hash interpolation)
float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(i + vec3(0,0,0));
    float n100 = hash3(i + vec3(1,0,0));
    float n010 = hash3(i + vec3(0,1,0));
    float n110 = hash3(i + vec3(1,1,0));
    float n001 = hash3(i + vec3(0,0,1));
    float n101 = hash3(i + vec3(1,0,1));
    float n011 = hash3(i + vec3(0,1,1));
    float n111 = hash3(i + vec3(1,1,1));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z);
}

// fBm (fractional Brownian motion): sum of noise octaves with falling
// amplitude -- the workhorse of procedural textures (NVIDIA GPU Gems).
float fbm(vec3 p) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += amp * vnoise(p);
        norm += amp;
        p *= 2.1;
        amp *= 0.5;
    }
    return sum / norm;
}

float ellipse_mask(vec2 p, vec2 center, vec2 radius) {
    vec2 q = (p - center) / radius;
    return 1.0 - smoothstep(0.55, 1.0, dot(q, q));
}

float skin_height(vec3 p, float front_gate) {
    vec3 q = p + vec3(u_seed * 19.0, u_seed * 31.0, u_seed * 47.0);
    float pores_a = fbm(q * 155.0);
    float pores_b = fbm(q * 315.0 + vec3(7.0, 3.0, 11.0));
    float pore_pits = smoothstep(0.56, 0.86, pores_b);
    float fine = (pores_a - 0.5) * 0.0011 - pore_pits * 0.0015;

    float horizontal = 0.5 + 0.5 * cos((p.y * 125.0)
        + fbm(q * 18.0) * 2.8);
    float forehead = ellipse_mask(p.xy, vec2(0.0, 0.155), vec2(0.095, 0.040));
    float crow_l = ellipse_mask(p.xy, vec2(-0.060, 0.130), vec2(0.032, 0.030));
    float crow_r = ellipse_mask(p.xy, vec2( 0.060, 0.130), vec2(0.032, 0.030));
    float meso = smoothstep(0.74, 0.96, horizontal)
        * max(forehead, max(crow_l, crow_r)) * front_gate * -0.0022;
    return fine + meso;
}

vec3 perturb_skin_normal(vec3 n, vec3 p, float front_gate) {
    float h = skin_height(p, front_gate);
    float hx = dFdx(h);
    float hy = dFdy(h);
    vec3 sx = dFdx(v_pos);
    vec3 sy = dFdy(v_pos);
    vec3 tx = normalize(sx - n * dot(sx, n));
    vec3 ty = normalize(sy - n * dot(sy, n));
    vec3 bumped = normalize(n - 55.0 * (hx * tx + hy * ty));
    return normalize(mix(n, bumped, 0.70 + 0.20 * front_gate));
}

float circle_tile(vec2 uv, float r) {
    vec2 p = fract(uv) - 0.5;
    return 1.0 - smoothstep(r, r + 0.035, length(p));
}

float diamond(vec2 uv, float r) {
    return 1.0 - smoothstep(r, r + 0.035, abs(uv.x) + abs(uv.y));
}

void main() {
    vec3 n_base = normalize(v_nrm);
    vec3 n = n_base;
    vec3 sample_p = v_local + vec3(u_seed * 37.0, u_seed * 13.0, u_seed * 91.0);

    vec3 albedo = u_color;
    float skin_front = smoothstep(0.005, 0.055, v_local.z)
        * (1.0 - smoothstep(0.145, 0.210, abs(v_local.x)))
        * smoothstep(0.000, 0.040, v_local.y)
        * (1.0 - smoothstep(0.245, 0.340, v_local.y));
    float skin_micro = 0.0;

    if (u_mode == 0) {
        // BIOPHYSICAL-INSPIRED SKIN:
        // 1) stable local-space pseudo-UVs so texture does not swim during
        //    facial/body animation;
        // 2) broad melanin/undertone variation plus regional hemoglobin;
        // 3) pores and shallow meso folds through normal perturbation;
        // 4) roughness/spec response coupled to the microstructure.
        n = perturb_skin_normal(n_base, v_local, skin_front);

        float melanin = fbm(sample_p * 4.0) - 0.5;
        float undertone = fbm(sample_p * vec3(2.0, 5.0, 3.0) + vec3(5.0)) - 0.5;
        albedo *= exp2(melanin * 0.30);
        albedo = mix(albedo, albedo * vec3(1.035, 0.985, 0.945),
                     clamp(undertone * 0.5 + 0.5, 0.0, 1.0) * 0.20);

        vec2 face_xy = v_local.xy;
        float cheek = max(
            ellipse_mask(face_xy, vec2(-0.047, 0.105), vec2(0.045, 0.046)),
            ellipse_mask(face_xy, vec2( 0.047, 0.105), vec2(0.045, 0.046))
        );
        float nose = ellipse_mask(face_xy, vec2(0.000, 0.112), vec2(0.030, 0.055));
        float eyelid = max(
            ellipse_mask(face_xy, vec2(-0.046, 0.134), vec2(0.035, 0.020)),
            ellipse_mask(face_xy, vec2( 0.046, 0.134), vec2(0.035, 0.020))
        );
        float mouth = ellipse_mask(face_xy, vec2(0.000, 0.070), vec2(0.060, 0.020));
        float hemo_patch = clamp(cheek * 0.42 + nose * 0.30
            + eyelid * 0.22 + mouth * 0.18, 0.0, 1.0) * skin_front;
        float hemo_noise = smoothstep(0.25, 0.88,
            fbm(sample_p * 24.0 + vec3(1.0, 4.0, 9.0)));
        albedo = mix(albedo, albedo * vec3(1.13, 0.86, 0.82),
                     hemo_patch * (0.55 + 0.45 * hemo_noise));

        float freckle_n = fbm(sample_p * 74.0 + vec3(11.0, 2.0, 5.0));
        float freckles = smoothstep(0.82, 0.96, freckle_n) * skin_front;
        albedo = mix(albedo, albedo * vec3(0.70, 0.48, 0.36), freckles * 0.28);

        float pores = fbm(sample_p * 180.0);
        skin_micro = pores;
        albedo *= 0.985 + 0.035 * pores;
        float cavity = clamp(-n_base.y * 0.5 + 0.5, 0.0, 1.0);
        albedo = mix(albedo, albedo * vec3(1.04, 0.91, 0.88), 0.10 * cavity);
    } else if (u_mode == 1) {
        // FABRIC: interlaced yarns, dye variation, and broad compression
        // folds. This approximates pattern/texture-flow approaches used by
        // procedural garment systems without requiring authored UVs.
        // u_material picks a sub-type (cotton/denim/silk/knit) that
        // retunes weave frequency, warp/weft balance, dye variance and
        // specular response. Authored by eye to sit somewhere close to
        // the plainclothes looks of each fabric family.
        float dye_scale = 3.0;
        float dye_amp = 0.13;
        float weave_fx = 86.0;
        float weave_fy = 92.0;
        float warp_share = 0.55;
        float weave_amp = 0.12;
        if (u_material == 1) {              // denim: coarse weave, strong warp
            dye_scale = 1.8; dye_amp = 0.18;
            weave_fx = 130.0; weave_fy = 58.0;
            warp_share = 0.72; weave_amp = 0.19;
        } else if (u_material == 2) {       // silk: very fine, low contrast
            dye_scale = 4.2; dye_amp = 0.06;
            weave_fx = 210.0; weave_fy = 230.0;
            warp_share = 0.50; weave_amp = 0.05;
        } else if (u_material == 3) {       // knit: vertical ribbed wales
            dye_scale = 2.4; dye_amp = 0.11;
            weave_fx = 48.0;  weave_fy = 160.0;
            warp_share = 0.30; weave_amp = 0.22;
        }

        float dye = fbm(sample_p * dye_scale);
        albedo *= 1.0 - 0.5 * dye_amp + dye_amp * dye;

        float warp = 0.5 + 0.5 * cos(v_pos.x * weave_fx + fbm(sample_p * 7.0) * 2.5);
        float weft = 0.5 + 0.5 * cos(v_pos.y * weave_fy + fbm(sample_p * 6.0 + vec3(3.0)) * 2.0);
        float weave = warp * warp_share + weft * (1.0 - warp_share);
        albedo *= 1.0 - weave_amp * 0.5 + weave_amp * weave;

        // Denim sits cooler and indigo-biased; silk gets a slight sheen
        // tint; knit tones down a hair. All relative to the base u_color
        // so any sampled garment color still reads recognisably.
        if (u_material == 1) {
            albedo = mix(albedo, albedo * vec3(0.82, 0.88, 1.05), 0.35);
        } else if (u_material == 2) {
            albedo = mix(albedo, albedo * vec3(1.05, 1.03, 1.01), 0.15);
        } else if (u_material == 3) {
            albedo *= 0.94;
        }

        float folds = 0.5 + 0.5 * cos(v_pos.y * 18.0 + fbm(sample_p * 2.0) * 4.0);
        float fold_mask = smoothstep(0.55, 1.0, folds) * (0.65 + 0.35 * abs(n.z));
        albedo *= 1.0 - 0.06 * fold_mask;

        // Pose-driven compression wrinkles: at elbows / knees / waist the
        // vertex shader reports a non-zero v_bend, peaking along the 50/50
        // two-bone blend band. Add a high-frequency ripple pattern that
        // darkens with bend intensity and also lifts/dips the perceived
        // normal so the lighting picks up the fold. Direction uses
        // v_local.y so wrinkles run across the limb, not along it.
        if (v_bend > 0.02) {
            float bendc = clamp(v_bend, 0.0, 1.0);
            float ripple = 0.5 + 0.5 * sin(v_local.y * 140.0
                                            + fbm(sample_p * 3.0) * 4.0);
            float ripple_mask = smoothstep(0.35, 0.95, ripple);
            // Dark "valley" lines between ridges.
            albedo *= 1.0 - 0.18 * bendc * ripple_mask;
            // A second, lower-frequency fold that reads as a single deep
            // crease at the joint apex (strong bends only).
            float crease = smoothstep(0.55, 1.0,
                0.5 + 0.5 * cos(v_local.y * 32.0
                                 + fbm(sample_p * 1.8) * 3.0));
            albedo *= 1.0 - 0.12 * bendc * bendc * crease;
            // Tint toward a slightly desaturated shadow so the fold reads
            // as depth rather than dirt.
            albedo = mix(albedo, albedo * vec3(0.88, 0.90, 0.94),
                         0.18 * bendc * ripple_mask);
        }

        float lint = fbm(sample_p * 24.0);
        albedo *= 0.96 + 0.07 * lint;

        // PROCEDURAL PRINTS (explicit style, Python-driven for variety).
        // UV-free tile coordinates: cylindrical angle for seamless torso wrap,
        // plus a small world-space component so sleeves/pants stay aligned.
        float lum = dot(u_color, vec3(0.299, 0.587, 0.114));
        vec3 light_print = mix(u_color, vec3(1.0), 0.52);
        vec3 dark_print  = u_color * 0.38;
        vec3 print_color = (lum < 0.48) ? light_print : dark_print;

        if (u_print_style > 0 && u_print_strength > 0.001) {
            float theta = atan(v_pos.z, v_pos.x) / 6.2831853;
            vec2 tile_uv = vec2(theta * 5.0 + v_pos.x * 1.7, v_pos.y * 5.4);
            float print_mask = 0.0;
            if (u_print_style == 1) {
                print_mask = smoothstep(0.58, 0.66, 0.5 + 0.5 * cos(tile_uv.y * 6.2831853));
            } else if (u_print_style == 2) {
                vec2 dots_uv = tile_uv * vec2(1.25, 1.0);
                dots_uv.x += 0.5 * step(0.5, fract(dots_uv.y * 0.5));
                print_mask = circle_tile(dots_uv, 0.22);
            } else if (u_print_style == 3) {
                float sx = smoothstep(0.72, 0.80, 0.5 + 0.5 * cos(tile_uv.x * 6.2831853));
                float sy = smoothstep(0.70, 0.78, 0.5 + 0.5 * cos(tile_uv.y * 6.2831853));
                print_mask = max(sx * 0.75, sy);
            } else {
                // camo-ish clumpy noise
                print_mask = smoothstep(0.52, 0.78, fbm(sample_p * 6.0));
            }
            albedo = mix(albedo, print_color, print_mask * u_print_strength);
        }

        // CHEST STAMP: front-facing torso decal with soft spatial gating so
        // it can't bleed onto sleeves or the lower garment. Centered on a
        // normalised torso-height band that works across character heights.
        if (u_stamp_style > 0 && u_stamp_strength > 0.001) {
            vec2 decal_uv = vec2(v_pos.x / 0.14, (v_pos.y - 0.36) / 0.17);
            float front_gate = smoothstep(0.30, 0.60, n.z);
            float x_gate = 1.0 - smoothstep(0.75, 1.05, abs(decal_uv.x));
            float y_gate = 1.0 - smoothstep(0.85, 1.15, abs(decal_uv.y));
            float decal_gate = front_gate * x_gate * y_gate;
            float decal_mask = 0.0;
            if (u_stamp_style == 1) {
                // Ring (disc minus inner disc)
                float r = length(decal_uv);
                decal_mask = (1.0 - smoothstep(0.80, 0.95, r))
                           * smoothstep(0.32, 0.48, r);
            } else if (u_stamp_style == 2) {
                // Diamond ring
                decal_mask = diamond(decal_uv, 0.80);
                decal_mask *= 1.0 - diamond(decal_uv, 0.42);
            } else if (u_stamp_style == 3) {
                // Plus / cross
                float bar_w = 0.18;
                float horiz = step(abs(decal_uv.y), bar_w) * step(abs(decal_uv.x), 0.80);
                float vert  = step(abs(decal_uv.x), bar_w) * step(abs(decal_uv.y), 0.80);
                decal_mask = max(horiz, vert);
            } else {
                // 5-pointed star (polar-angle petals)
                float a = atan(decal_uv.y, decal_uv.x);
                float r = length(decal_uv);
                float petals = 0.60 + 0.30 * cos(5.0 * a);
                decal_mask = 1.0 - smoothstep(petals - 0.05, petals + 0.05, r);
            }
            decal_mask *= decal_gate;
            vec3 stamp_color = (lum < 0.48)
                ? mix(u_color, vec3(1.0), 0.75)
                : u_color * 0.22;
            albedo = mix(albedo, stamp_color, decal_mask * u_stamp_strength);
        }
    } else if (u_mode == 2) {
        // HAIR: strongly anisotropic fBm -- long wavelength along the head's
        // vertical axis, short across it -> reads as vertical strands.
        float strand = fbm(sample_p * vec3(55.0, 8.0, 55.0));
        albedo *= 0.75 + 0.30 * strand;
        float fine = fbm(sample_p * vec3(180.0, 30.0, 180.0));
        albedo *= 0.92 + 0.14 * fine;
        // Strand lines, warped by a low-frequency fbm so they don't read as
        // a regular corduroy pattern.
        float warp = fbm(sample_p * 6.0);
        float line = 0.5 + 0.5 * cos((v_pos.x + v_pos.z * 0.35) * 170.0
                                      + warp * 7.0);
        albedo *= 0.90 + 0.16 * line;
        // Per-clump color variation: low-frequency multiplier breaks up the
        // uniform block color so a head of hair reads as many strands of
        // similar but not identical shade.
        float clump = fbm(sample_p * vec3(6.0, 3.0, 6.0));
        albedo *= 0.88 + 0.22 * clump;
        // Root-to-tip gradient: real hair is slightly darker at the scalp
        // (more shadow between strands) and brighter at exposed tips. Use
        // world-space Y around the head as a proxy.
        // Long hair drape tips sit well below the crown (negative local Y
        // for drape verts). Treat both ends as slightly lighter: crown
        // catches sky light, tips are dusty/exposed.
        float tip_lo = 1.0 - smoothstep(-0.25, 0.05, v_local.y);
        float tip_hi = smoothstep(0.28, 0.38, v_local.y);
        albedo *= 1.0 + 0.08 * max(tip_lo, tip_hi);
    } else if (u_mode == 4) {
        // SHOES: matte leather/rubber. Keep them out of clothing print logic.
        float grain = fbm(sample_p * 18.0);
        albedo *= 0.92 + 0.10 * grain;
    } else {
        // EYES: keep sclera/iris/pupil out of the skin pigmentation path.
        // The eye surface needs clean wet specular response, not pores or
        // melanin/hemoglobin mottling.
        float limbal = smoothstep(0.35, 0.95, fbm(sample_p * 18.0));
        albedo *= 0.97 + 0.04 * limbal;
    }

    vec3 L1 = normalize(vec3(0.4, 0.8, 0.6));
    vec3 L2 = normalize(vec3(-0.5, 0.3, -0.4));
    vec3 C1 = vec3(1.0, 0.96, 0.90);
    vec3 C2 = vec3(0.70, 0.78, 1.0);
    float ambient = 0.46;
    if (u_light_style == 1) {
        L1 = normalize(vec3(0.0, 0.65, 0.76));
        L2 = normalize(vec3(-0.25, 0.55, 0.30));
        C1 = vec3(0.98, 0.98, 1.0);
        C2 = vec3(0.75, 0.82, 0.95);
        ambient = 0.54;
    } else if (u_light_style == 2) {
        L1 = normalize(vec3(0.95, 0.28, 0.16));
        L2 = normalize(vec3(-0.25, 0.45, -0.55));
        C1 = vec3(1.0, 0.92, 0.84);
        C2 = vec3(0.55, 0.62, 0.82);
        ambient = 0.34;
    } else if (u_light_style == 3) {
        L1 = normalize(vec3(-0.40, 0.70, 0.55));
        L2 = normalize(vec3(0.55, 0.35, -0.48));
        C1 = vec3(1.0, 0.78, 0.58);
        C2 = vec3(0.48, 0.62, 1.0);
        ambient = 0.42;
    }

    float ndl1 = max(dot(n, L1), 0.0);
    float ndl2 = max(dot(n, L2), 0.0);
    if (u_mode == 0) {
        ndl1 = clamp((dot(n, L1) + 0.32) / 1.32, 0.0, 1.0);
        ndl2 = clamp((dot(n, L2) + 0.22) / 1.22, 0.0, 1.0);
    }
    vec3 light = C1 * ndl1 * 0.88 + C2 * ndl2 * 0.32 + vec3(ambient);
    if (u_mode == 0) {
        // Face readability: keep a soft skin-only fill so darker skin tones
        // and eye/nose/mouth cavities do not collapse under portrait light.
        float skin_luma = dot(u_color, vec3(0.299, 0.587, 0.114));
        light += vec3(0.10 + 0.28 * (1.0 - skin_luma));
    }
    vec3 c = albedo * light;

    // Blinn-Phong specular: skin has a soft oil highlight; eyes use a tighter
    // wet highlight. Fabric stays matte.
    if (u_mode == 0 || u_mode == 3) {
        vec3 V = normalize(-v_pos);
        vec3 H = normalize(L1 + V);
        float rough_var = fbm(sample_p * 28.0 + vec3(2.0, 6.0, 3.0));
        float roughness = clamp(0.46 + 0.20 * rough_var
            - 0.08 * skin_front + 0.06 * skin_micro, 0.32, 0.78);
        float gloss = (u_mode == 3) ? 96.0 : mix(72.0, 22.0, roughness);
        float strength = (u_mode == 3) ? 0.20 : mix(0.105, 0.045, roughness);
        float spec = pow(max(dot(n, H), 0.0), gloss);
        c += strength * spec * vec3(1.0, 0.97, 0.93);
        if (u_mode == 0) {
            float scatter = pow(max(dot(-L1, n_base), 0.0), 2.0) * skin_front;
            c += scatter * 0.035 * albedo * vec3(1.18, 0.56, 0.44);
        }
    } else if (u_mode == 1 && u_material == 2) {
        // SILK / SATIN: smooth surface with a cross-grain anisotropic
        // highlight. We approximate the fiber direction as horizontal in
        // the garment's local frame (weft) and run a Kajiya-style sheen
        // off it. Softer than hair -- the highlight is a broad band, not
        // a narrow strand specular.
        vec3 V = normalize(-v_pos);
        vec3 T = normalize(vec3(1.0, 0.0, 0.0)
                           - n * dot(vec3(1.0, 0.0, 0.0), n));
        vec3 H = normalize(L1 + V);
        float sinTH = sqrt(max(0.0, 1.0 - dot(T, H) * dot(T, H)));
        float sheen = pow(sinTH, 18.0);
        c += 0.14 * sheen * albedo * vec3(1.05, 1.02, 0.98);
    } else if (u_mode == 1 && u_material == 1) {
        // DENIM: coarse weave catches a low broad specular at grazing
        // angles. Blinn-Phong with a low exponent and very small amp so
        // it reads as dry cotton twill, not plastic.
        vec3 V = normalize(-v_pos);
        vec3 H = normalize(L1 + V);
        float spec = pow(max(dot(n, H), 0.0), 14.0);
        c += 0.035 * spec * vec3(0.92, 0.95, 1.0);
    } else if (u_mode == 2) {
        // Kajiya-Kay style strand highlight. Approximate strand flow in the
        // surface tangent plane: mostly downward with a small procedural sway.
        vec3 V = normalize(-v_pos);
        // Strand flow approximates gravity-pulled fibers: local -Y projected
        // onto the surface tangent plane, with a small procedural sway so
        // the highlight doesn't read as a perfect horizontal band.
        vec3 down = vec3(
            0.12 * sin((v_local.x + u_seed) * 14.0),
            -1.0,
            0.08 * cos((v_local.z + u_seed) * 12.0)
        );
        vec3 T = normalize(down - n * dot(down, n));
        vec3 H = normalize(L1 + V);
        float sinTH = sqrt(max(0.0, 1.0 - dot(T, H) * dot(T, H)));
        // Primary specular: tight white highlight. Secondary: broader tinted
        // highlight shifted slightly, marketing-photo style anisotropy.
        float primary = pow(sinTH, 56.0);
        float secondary = pow(sinTH, 9.0) * 0.45;
        // Break up the highlight band with a clump-scale noise so strand
        // clusters catch light asymmetrically.
        float hl_noise = 0.65 + 0.55 * fbm(sample_p * vec3(22.0, 8.0, 22.0));
        c += (0.11 * primary + 0.048 * secondary) * hl_noise
             * vec3(1.0, 0.92, 0.72);
    }

    float rim = pow(1.0 - max(dot(n, normalize(-v_pos)), 0.0), 3.0);
    float rim_k = (u_mode == 0) ? 0.055 : ((u_mode == 2) ? 0.10 : ((u_mode == 3) ? 0.06 : 0.05));
    c += rim_k * rim * vec3(1.0, 0.85, 0.7);
    frag = vec4(c, 1.0);
}
"""

LINE_VERT = """
#version 330 core
layout(location=0) in vec3 a_pos;
uniform mat4 u_proj;
uniform mat4 u_view;
void main() { gl_Position = u_proj * u_view * vec4(a_pos, 1.0); }
"""

LINE_FRAG = """
#version 330 core
out vec4 frag;
uniform vec3 u_color;
void main() { frag = vec4(u_color, 1.0); }
"""


def _compile(src, stage):
    s = glCreateShader(stage)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s


def make_program(vsrc, fsrc):
    vs = _compile(vsrc, GL_VERTEX_SHADER)
    fs = _compile(fsrc, GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, vs); glAttachShader(p, fs)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(vs); glDeleteShader(fs)
    return p


class SkinProgram:
    def __init__(self):
        self.prog = make_program(SKIN_VERT, SKIN_FRAG)
        self.u_proj = glGetUniformLocation(self.prog, "u_proj")
        self.u_view = glGetUniformLocation(self.prog, "u_view")
        self.u_bones = glGetUniformLocation(self.prog, "u_bones")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_mode  = glGetUniformLocation(self.prog, "u_mode")
        self.u_seed  = glGetUniformLocation(self.prog, "u_seed")
        self.u_light_style = glGetUniformLocation(self.prog, "u_light_style")
        self.u_print_style    = glGetUniformLocation(self.prog, "u_print_style")
        self.u_print_strength = glGetUniformLocation(self.prog, "u_print_strength")
        self.u_stamp_style    = glGetUniformLocation(self.prog, "u_stamp_style")
        self.u_stamp_strength = glGetUniformLocation(self.prog, "u_stamp_strength")
        self.u_material       = glGetUniformLocation(self.prog, "u_material")
        self.u_bend_inflate   = glGetUniformLocation(self.prog, "u_bend_inflate")


class LineProgram:
    def __init__(self):
        self.prog = make_program(LINE_VERT, LINE_FRAG)
        self.u_proj = glGetUniformLocation(self.prog, "u_proj")
        self.u_view = glGetUniformLocation(self.prog, "u_view")
        self.u_color = glGetUniformLocation(self.prog, "u_color")


class MeshGPU:
    """Owns the VAO + VBOs for one skinned mesh."""

    def __init__(self, mesh):
        self.num_indices = mesh.indices.size
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_pos = self._vbo(mesh.positions, 0, 3, GL_FLOAT)
        self.vbo_nrm = self._vbo(mesh.normals,   1, 3, GL_FLOAT)
        self.vbo_bon = self._vbo(mesh.bones,     2, 2, GL_INT, integer=True)
        self.vbo_wts = self._vbo(mesh.weights,   3, 2, GL_FLOAT)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    @staticmethod
    def _vbo(data, loc, size, gltype, integer=False):
        b = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        if integer:
            glVertexAttribIPointer(loc, size, gltype, 0, None)
        else:
            glVertexAttribPointer(loc, size, gltype, GL_FALSE, 0, None)
        glEnableVertexAttribArray(loc)
        return b

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)

    def delete(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(4, [self.vbo_pos, self.vbo_nrm, self.vbo_bon, self.vbo_wts])
        glDeleteBuffers(1, [self.ebo])


class LineBufferGPU:
    """Dynamic VBO for skeleton overlay lines."""

    def __init__(self, max_points):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, max_points * 3 * 4, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def upload(self, flat_points: np.ndarray):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, flat_points.nbytes, flat_points)

    def draw(self, num_points):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, num_points)

    def delete(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])


def upload_bones(uniform_loc, bone_mats: np.ndarray):
    """Upload a (N,4,4) row-major numpy stack to a ``mat4[]`` uniform."""
    n = bone_mats.shape[0]
    assert n <= MAX_BONES, f"need MAX_BONES>= {n}"
    # Transpose each matrix: numpy is row-major, GL expects column-major unless
    # transpose=GL_TRUE; we pre-transpose to send in one flat call.
    flat = np.transpose(bone_mats, (0, 2, 1)).astype(np.float32).reshape(-1)
    glUniformMatrix4fv(uniform_loc, n, GL_FALSE, flat)
