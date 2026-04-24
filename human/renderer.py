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

out vec3 v_nrm;
out vec3 v_pos;

void main() {
    mat4 M = u_bones[a_bones.x] * a_weights.x + u_bones[a_bones.y] * a_weights.y;
    vec4 p = M * vec4(a_pos, 1.0);
    gl_Position = u_proj * u_view * p;
    v_pos = p.xyz;
    v_nrm = mat3(M) * a_nrm;
}
"""

SKIN_FRAG = """
#version 330 core
in vec3 v_nrm;
in vec3 v_pos;
out vec4 frag;

uniform vec3  u_color;
uniform int   u_mode;   // 0 = skin, 1 = fabric, 2 = hair, 3 = eye
uniform float u_seed;   // per-character random seed in [0,1]

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

void main() {
    vec3 n = normalize(v_nrm);
    vec3 sample_p = v_pos + vec3(u_seed * 37.0, u_seed * 13.0, u_seed * 91.0);

    vec3 albedo = u_color;

    if (u_mode == 0) {
        // BIOPHYSICAL-INSPIRED SKIN (melanin + hemoglobin + pores):
        //   1) Low-freq melanin modulation darkens/lightens broad regions.
        //   2) Mid-freq hemoglobin patches add warm red-ish tint.
        //   3) High-freq fBm simulates pores without harsh speckle.
        //   4) Cavity tint warms downward-facing concave regions (thin skin).
        // Ref: Alotaibi & Smith, "A Biophysical 3D Morphable Model of Face
        //      Appearance" (ICCV 2017); Smith et al., Morphable Face Albedo
        //      Model (CVPR 2020); NVIDIA GPU Gems 3 subsurface chapter.
        float melanin = fbm(sample_p * 1.1) - 0.5;
        albedo *= exp2(melanin * 0.35);
        float hemo = smoothstep(0.35, 0.75, fbm(sample_p * 2.2 + vec3(5.0)));
        albedo = mix(albedo, albedo * vec3(1.12, 0.86, 0.84), hemo * 0.35);
        float pores = fbm(sample_p * 38.0);
        albedo *= 0.96 + 0.06 * pores;
        float cavity = clamp(-n.y * 0.5 + 0.5, 0.0, 1.0);
        albedo = mix(albedo, albedo * vec3(1.05, 0.90, 0.88), 0.12 * cavity);
    } else if (u_mode == 1) {
        // FABRIC: interlaced yarns, dye variation, and broad compression
        // folds. This approximates pattern/texture-flow approaches used by
        // procedural garment systems without requiring authored UVs.
        float dye = fbm(sample_p * 3.0);
        albedo *= 0.91 + 0.13 * dye;

        float warp = 0.5 + 0.5 * cos(v_pos.x * 86.0 + fbm(sample_p * 7.0) * 2.5);
        float weft = 0.5 + 0.5 * cos(v_pos.y * 92.0 + fbm(sample_p * 6.0 + vec3(3.0)) * 2.0);
        float weave = warp * 0.55 + weft * 0.45;
        albedo *= 0.91 + 0.12 * weave;

        float folds = 0.5 + 0.5 * cos(v_pos.y * 18.0 + fbm(sample_p * 2.0) * 4.0);
        float fold_mask = smoothstep(0.55, 1.0, folds) * (0.65 + 0.35 * abs(n.z));
        albedo *= 1.0 - 0.06 * fold_mask;

        float lint = fbm(sample_p * 24.0);
        albedo *= 0.96 + 0.07 * lint;
    } else if (u_mode == 2) {
        // HAIR: strongly anisotropic fBm -- long wavelength along the head's
        // vertical axis, short across it -> reads as vertical strands.
        float strand = fbm(sample_p * vec3(55.0, 8.0, 55.0));
        albedo *= 0.75 + 0.30 * strand;
        float fine = fbm(sample_p * vec3(180.0, 30.0, 180.0));
        albedo *= 0.92 + 0.14 * fine;
        float line = 0.5 + 0.5 * cos((v_pos.x + v_pos.z * 0.35) * 170.0
                                      + fbm(sample_p * 9.0) * 5.0);
        albedo *= 0.90 + 0.16 * line;
    } else {
        // EYES: keep sclera/iris/pupil out of the skin pigmentation path.
        // The eye surface needs clean wet specular response, not pores or
        // melanin/hemoglobin mottling.
        float limbal = smoothstep(0.35, 0.95, fbm(sample_p * 18.0));
        albedo *= 0.97 + 0.04 * limbal;
    }

    vec3 L1 = normalize(vec3(0.4, 0.8, 0.6));
    vec3 L2 = normalize(vec3(-0.5, 0.3, -0.4));
    float d = max(dot(n, L1), 0.0) * 0.9
            + max(dot(n, L2), 0.0) * 0.35
            + 0.20;
    vec3 c = albedo * d;

    // Blinn-Phong specular: skin has a soft oil highlight; eyes use a tighter
    // wet highlight. Fabric stays matte.
    if (u_mode == 0 || u_mode == 3) {
        vec3 V = normalize(-v_pos);
        vec3 H = normalize(L1 + V);
        float gloss = (u_mode == 3) ? 96.0 : 40.0;
        float strength = (u_mode == 3) ? 0.20 : 0.09;
        float spec = pow(max(dot(n, H), 0.0), gloss);
        c += strength * spec * vec3(1.0, 0.97, 0.93);
    } else if (u_mode == 2) {
        // Kajiya-Kay style strand highlight. Approximate strand flow in the
        // surface tangent plane: mostly downward with a small procedural sway.
        vec3 V = normalize(-v_pos);
        vec3 flow = normalize(vec3(
            0.18 * sin((v_pos.x + u_seed) * 16.0),
            1.0,
            0.10 * cos((v_pos.z + u_seed) * 14.0)
        ));
        vec3 T = normalize(flow - n * dot(flow, n));
        vec3 H = normalize(L1 + V);
        float sinTH = sqrt(max(0.0, 1.0 - dot(T, H) * dot(T, H)));
        float primary = pow(sinTH, 42.0);
        float secondary = pow(sinTH, 9.0) * 0.45;
        c += (0.10 * primary + 0.045 * secondary) * vec3(1.0, 0.92, 0.72);
    }

    float rim = pow(1.0 - max(dot(n, normalize(-v_pos)), 0.0), 3.0);
    float rim_k = (u_mode == 0) ? 0.15 : ((u_mode == 2) ? 0.10 : ((u_mode == 3) ? 0.06 : 0.05));
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
