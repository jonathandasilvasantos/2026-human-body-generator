"""Shared GLSL shader sources used by both the offscreen renderer and viewer."""

VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
out vec3 v_norm;
out vec2 v_uv;
void main() {
    v_norm = mat3(model) * in_norm;
    v_uv   = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
uniform vec3 light_dir;
uniform vec3 base_color;
uniform vec3 ambient_color;
in vec3 v_norm;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    float l = max(dot(n, normalize(light_dir)), 0.0);
    vec3 col = base_color * (0.25 + 0.75 * l) + ambient_color;
    frag = vec4(col, 1.0);
}
"""
