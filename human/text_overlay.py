"""Simple PIL-rasterised text overlay drawn as a screen-space textured quad.

The main viewer is fullscreen, so the GLFW title bar is hidden; we need an
in-window HUD to show the currently active animation. Pillow handles the
glyph rasterisation (already a project dep for texture generation); GL
uploads it once per text change.
"""

import ctypes

import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont

from . import renderer


_VERT = """
#version 330 core
layout(location=0) in vec2 a_pos;   // pixel coordinates, origin top-left
layout(location=1) in vec2 a_uv;
out vec2 v_uv;
uniform vec2 u_viewport;
void main() {
    vec2 ndc = (a_pos / u_viewport) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"""

_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex;
void main() { frag = texture(u_tex, v_uv); }
"""


_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


class TextOverlay:
    """A single-line label rendered at a screen-space position."""

    def __init__(self, size: int = 22):
        self.prog = renderer.make_program(_VERT, _FRAG)
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport")
        self.u_tex = glGetUniformLocation(self.prog, "u_tex")
        self.tex = glGenTextures(1)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self._font = _load_font(size)
        self._text = None
        self._w = 0
        self._h = 0

    def set_text(self, text: str):
        if text == self._text:
            return
        self._text = text
        pad = 10
        tmp = Image.new("RGBA", (4, 4))
        bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=self._font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])
        w = tw + pad * 2
        h = th + pad * 2
        img = Image.new("RGBA", (w, h), (20, 22, 28, 210))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w - 1, h - 1], outline=(140, 150, 170, 230))
        draw.text((pad - bbox[0], pad - bbox[1]), text,
                  fill=(245, 230, 195, 255), font=self._font)
        data = np.array(img, dtype=np.uint8)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self._w, self._h = w, h

    def draw(self, viewport_w: int, viewport_h: int, x: int, y: int):
        if self._text is None:
            return
        w, h = self._w, self._h
        verts = np.array([
            x,     y,     0.0, 0.0,
            x + w, y,     1.0, 0.0,
            x + w, y + h, 1.0, 1.0,
            x,     y,     0.0, 0.0,
            x + w, y + h, 1.0, 1.0,
            x,     y + h, 0.0, 1.0,
        ], dtype=np.float32)

        glUseProgram(self.prog)
        glUniform2f(self.u_viewport, float(viewport_w), float(viewport_h))
        glUniform1i(self.u_tex, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glBindVertexArray(0)

    def delete(self):
        try:
            glDeleteTextures(1, [self.tex])
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
            glDeleteProgram(self.prog)
        except Exception:
            pass
