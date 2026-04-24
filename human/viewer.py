"""Fullscreen viewer + input handling + walk-cycle driver."""

import math
import os
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import glfw
from OpenGL.GL import *

from . import animation as anim_mod
from . import audio_lipsync, mathx, renderer, skeleton
from .character import Character, random_appearance, reroll_clothes


EXPRESSION_CYCLE = [
    "neutral", "smile", "smile_duchenne", "frown", "sad",
    "surprise", "fear", "anger", "disgust", "contempt",
    "squint", "blink", "wink_left", "wink_right",
    "kiss", "open_mouth", "ee", "oo", "aa",
]
from .text_overlay import TextOverlay


ANIMATIONS_DIR = Path(__file__).resolve().parent.parent / "animations"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_WAV = PROJECT_ROOT / "voice.wav"


def _scan_animations(startup_path: str | None):
    """Return a list of .bvh paths under ./animations, plus the index of
    the preferred startup file (the one passed via --bvh, if any)."""
    paths = []
    if ANIMATIONS_DIR.is_dir():
        paths = sorted(
            str(p) for p in ANIMATIONS_DIR.rglob("*.bvh")
        )
    start = 0
    if startup_path:
        abs_startup = os.path.abspath(startup_path)
        # Ensure the startup file is in the list; prepend if not.
        if abs_startup not in (os.path.abspath(p) for p in paths):
            paths.insert(0, startup_path)
            start = 0
        else:
            for i, p in enumerate(paths):
                if os.path.abspath(p) == abs_startup:
                    start = i
                    break
    return paths, start


class OrbitCamera:
    def __init__(self, target=(0.0, 0.18, 0.0), dist=2.7):
        self.target = np.asarray(target, dtype=np.float32)
        self.dist = dist
        # Character faces +Z; camera sits on +Z looking toward -Z so the
        # face is visible from the first frame. Pitch slightly up so the
        # eye sits roughly level with the character's chest.
        self.yaw = 0.0
        self.pitch = 0.08

    def view(self, follow=(0.0, 0.0, 0.0)):
        target = self.target + np.asarray(follow, dtype=np.float32)
        eye = target + np.array([
            self.dist * math.cos(self.pitch) * math.sin(self.yaw),
            self.dist * math.sin(self.pitch) + 0.25,
            self.dist * math.cos(self.pitch) * math.cos(self.yaw),
        ], dtype=np.float32)
        return mathx.look_at(eye, target, [0, 1, 0])


class Viewer:
    BG = (0.09, 0.10, 0.13, 1.0)
    BONE_COLOR = (1.0, 0.35, 0.2)

    def __init__(self, bvh_path: str | None = None):
        self._init_glfw()
        self._init_gl()

        self.camera = OrbitCamera()
        self.show_bones = False
        # Start still so the user sees the face cleanly on the first frame;
        # press W to start the walk animation.
        self.walking = False
        self.walk_speed = 2.4
        self._drag_origin = None
        self._t_walk = 0.0

        self.bvh_animation: anim_mod.Animation | None = None
        self._t_anim = 0.0
        self._expr_index = 0

        # Audio-driven lip sync (K key). Track is analysed lazily on first
        # activation so startup cost is not paid when the feature is unused.
        self._lipsync_track: audio_lipsync.LipsyncTrack | None = None
        self._lipsync_t0: float | None = None
        self._lipsync_proc: subprocess.Popen | None = None
        self._lipsync_saved_expression: str | None = None
        self._lipsync_last_weights: dict[str, float] = {}
        self._camera_default = (0.0, 0.18, 0.0, 2.7, 0.0, 0.08)

        # Animation library: every .bvh under ./animations is cyclable.
        self.anim_paths, self.anim_index = _scan_animations(bvh_path)
        self.bvh_path = (
            self.anim_paths[self.anim_index] if self.anim_paths else bvh_path
        )

        self.text_overlay = TextOverlay(size=22)

        random.seed()
        self.character = Character()
        self._regenerate_all()
        if self.bvh_path:
            self._load_bvh(self.bvh_path)
        self._update_label()

    # ---- setup ------------------------------------------------------------

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.window_hint(glfw.RED_BITS, mode.bits.red)
        glfw.window_hint(glfw.GREEN_BITS, mode.bits.green)
        glfw.window_hint(glfw.BLUE_BITS, mode.bits.blue)
        glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)

        self.win = glfw.create_window(
            mode.size.width, mode.size.height,
            "Procedural Human Generator", monitor, None,
        )
        if not self.win:
            glfw.terminate()
            raise RuntimeError("window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        glfw.set_mouse_button_callback(self.win, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_key_callback(self.win, self._on_key)

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(*self.BG)
        self.skin_prog = renderer.SkinProgram()
        self.line_prog = renderer.LineProgram()

    # ---- regeneration -----------------------------------------------------

    def _load_bvh(self, path: str):
        try:
            self.bvh_animation = anim_mod.load(path, self.character.bones)
            self._t_anim = 0.0
            self.walking = False
            print(f"[bvh] loaded {path}: {self.bvh_animation.bvh.frames} frames, "
                  f"{self.bvh_animation.duration:.2f}s, mapped {len(self.bvh_animation.entries)} bones")
        except Exception as e:
            print(f"[bvh] failed to load {path}: {e}")
            self.bvh_animation = None

    def _cycle_animation(self, step: int):
        if not self.anim_paths:
            return
        self.anim_index = (self.anim_index + step) % len(self.anim_paths)
        path = self.anim_paths[self.anim_index]
        self.bvh_path = path
        self._load_bvh(path)
        self._update_label()

    def _current_anim_label(self) -> str:
        if not self.anim_paths:
            return "anim: (none)"
        path = self.anim_paths[self.anim_index]
        name = Path(path).stem
        return f"anim [{self.anim_index + 1}/{len(self.anim_paths)}]  {name}"

    def _update_label(self):
        self.text_overlay.set_text(self._current_anim_label())

    def _regenerate_all(self, gender: str | None = None):
        shape = skeleton.random_shape(gender=gender)
        self.character.set_shape(shape)
        self.character.set_appearance(random_appearance(shape.gender))
        self.character.build_gpu()
        self._t_walk = 0.0
        print(f"[regen] gender={shape.gender} top={self.character.appearance.top_style} "
              f"bottom={self.character.appearance.bottom_style} "
              f"shoes={self.character.appearance.shoe_style} "
              f"hair={self.character.appearance.hair_style} "
              f"facial_hair={self.character.appearance.facial_hair_style}")

    # ---- input ------------------------------------------------------------

    def _on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._drag_origin = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                self._drag_origin = None

    def _on_cursor(self, win, x, y):
        if self._drag_origin is None:
            return
        lx, ly = self._drag_origin
        self.camera.yaw += (x - lx) * 0.01
        self.camera.pitch = max(-1.3, min(1.3, self.camera.pitch + (y - ly) * 0.01))
        self._drag_origin = (x, y)

    def _on_scroll(self, win, xoff, yoff):
        self.camera.dist = max(1.0, min(10.0, self.camera.dist * (0.9 ** yoff)))

    def _on_key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key in (glfw.KEY_SPACE, glfw.KEY_R):
            self._regenerate_all()
        elif key == glfw.KEY_M:
            self._regenerate_all(gender="male")
        elif key == glfw.KEY_G:
            self._regenerate_all(gender="female")
        elif key == glfw.KEY_S:
            shape = skeleton.random_shape(gender=self.character.shape.gender)
            self.character.set_shape(shape)
        elif key == glfw.KEY_C:
            # C: change only the outfit, keep the same person underneath.
            new_app = reroll_clothes(self.character.appearance)
            self.character.set_appearance(new_app)
            self.character.build_gpu()
        elif key == glfw.KEY_P:
            self.walking = False
            self.character.set_pose(
                skeleton.random_pose(self.character.bones, 0.9),
                root_offset=(0.0, 0.0, 0.0),
            )
        elif key == glfw.KEY_T:
            # T: true T-pose (arms extended horizontally).
            self.walking = False
            self.bvh_animation = None
            self.character.set_pose(
                skeleton.t_pose_wide(self.character.bones),
                root_offset=(0.0, 0.0, 0.0),
            )
        elif key == glfw.KEY_A:
            # A: A-pose (arms hanging at natural rest).
            self.walking = False
            self.bvh_animation = None
            self.character.set_pose(
                skeleton.a_pose(self.character.bones),
                root_offset=(0.0, 0.0, 0.0),
            )
        elif key == glfw.KEY_F:
            # F: frame the camera on the character's face.
            self._center_camera_on_face()
        elif key == glfw.KEY_B:
            # B: restart initial defaults (fresh character, default camera).
            self._reset_defaults()
        elif key == glfw.KEY_LEFT:
            self._cycle_expression(-1)
        elif key == glfw.KEY_RIGHT:
            self._cycle_expression(+1)
        elif key == glfw.KEY_W:
            self.walking = not self.walking
        elif key == glfw.KEY_K:
            # K: play ./voice.wav with audio-driven lip sync.
            self._toggle_lipsync()
        elif key == glfw.KEY_J:
            # J: toggle BVH playback (moved off K, which now drives lipsync).
            if self.bvh_animation is not None:
                self.bvh_animation = None
                print("[bvh] disabled; use procedural walk (W)")
            elif self.bvh_path:
                self._load_bvh(self.bvh_path)
        elif key == glfw.KEY_N:
            # N: toggle skeleton overlay (was B).
            self.show_bones = not self.show_bones
        elif key == glfw.KEY_LEFT_BRACKET:
            self.walk_speed = max(0.2, self.walk_speed - 0.3)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.walk_speed = min(8.0, self.walk_speed + 0.3)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            # '+' (same key as '=') cycles to the next animation.
            self._cycle_animation(+1)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._cycle_animation(-1)

    # ---- commands ---------------------------------------------------------

    def _cycle_expression(self, step: int):
        self._expr_index = (self._expr_index + step) % len(EXPRESSION_CYCLE)
        name = EXPRESSION_CYCLE[self._expr_index]
        from dataclasses import replace
        self.character.set_appearance(
            replace(self.character.appearance, expression=name))
        self.character.build_gpu()
        print(f"[expr] {name}")

    def _center_camera_on_face(self):
        bone_mats = self.character.bone_matrices()
        head_pos = self._head_world_pos(bone_mats)
        self.camera.target = np.array(
            [head_pos[0], head_pos[1] - 0.05, head_pos[2]], dtype=np.float32)
        self.camera.dist = 0.55
        self.camera.yaw = 0.0
        self.camera.pitch = 0.0

    def _reset_defaults(self):
        tx, ty, tz, dist, yaw, pitch = self._camera_default
        self.camera.target = np.array([tx, ty, tz], dtype=np.float32)
        self.camera.dist = dist
        self.camera.yaw = yaw
        self.camera.pitch = pitch
        self.walking = False
        self.walk_speed = 2.4
        self.bvh_animation = None
        self.show_bones = False
        self._expr_index = 0
        self._regenerate_all()

    # ---- audio-driven lip sync -------------------------------------------

    def _toggle_lipsync(self):
        if self._lipsync_t0 is not None:
            self._stop_lipsync()
            return
        if not VOICE_WAV.is_file():
            print(f"[lipsync] missing {VOICE_WAV}")
            return
        if self._lipsync_track is None:
            print(f"[lipsync] analysing {VOICE_WAV.name} ...")
            try:
                self._lipsync_track = audio_lipsync.from_wav(str(VOICE_WAV))
            except Exception as e:
                print(f"[lipsync] analysis failed: {e}")
                return
            print(f"[lipsync] {self._lipsync_track.duration:.2f}s, "
                  f"{len(self._lipsync_track.viseme_track.segments)} segments")
        # Spawn afplay (macOS) or aplay (linux) as a non-blocking subprocess.
        cmd = self._audio_cmd()
        if cmd is None:
            print("[lipsync] no audio player found; animating silently")
            self._lipsync_proc = None
        else:
            try:
                self._lipsync_proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[lipsync] audio player failed: {e}")
                self._lipsync_proc = None
        self._lipsync_saved_expression = self.character.appearance.expression
        # Neutral base so emotion presets don't fight the viseme weights.
        self.character.set_appearance(
            replace(self.character.appearance,
                    expression="neutral", blendshapes={}))
        self.character.build_gpu()
        self._lipsync_t0 = time.perf_counter()
        print("[lipsync] playing")

    @staticmethod
    def _audio_cmd():
        for exe in ("/usr/bin/afplay", "/usr/bin/aplay", "/usr/bin/paplay"):
            if os.path.isfile(exe):
                return [exe, str(VOICE_WAV)]
        return None

    def _stop_lipsync(self):
        if self._lipsync_proc is not None:
            try:
                self._lipsync_proc.terminate()
            except Exception:
                pass
            self._lipsync_proc = None
        if self._lipsync_saved_expression is not None:
            self.character.set_appearance(
                replace(self.character.appearance,
                        expression=self._lipsync_saved_expression,
                        blendshapes={}))
            self.character.build_gpu()
            self._lipsync_saved_expression = None
        self._lipsync_t0 = None
        self._lipsync_last_weights = {}
        print("[lipsync] stopped")

    def _tick_lipsync(self) -> bool:
        """Update face blendshapes from the audio track.

        Returns True when the clip has ended.
        """
        if self._lipsync_t0 is None or self._lipsync_track is None:
            return False
        t = time.perf_counter() - self._lipsync_t0
        if t >= self._lipsync_track.duration + 0.1:
            return True
        weights = self._lipsync_track.sample(max(0.0, t))
        # Snap very-small weights to zero so we don't rebuild geometry for
        # sub-visible motion; small threshold preserves tail decay.
        weights = {k: v for k, v in weights.items() if v > 0.01}
        # Skip the full face rebuild when no channel has changed by more
        # than ~2%. build_gpu() rebuilds every drawable on the character
        # (measured ~35 ms on this machine); guarding it frees budget for
        # audio-to-display sync when the mouth is coasting between
        # visemes. Audio clock drives sampling, so this never desyncs.
        if self._weights_similar(self._lipsync_last_weights, weights):
            return False
        self._lipsync_last_weights = weights
        app = self.character.appearance
        self.character.appearance = replace(app, blendshapes=weights)
        self.character.build_gpu()
        return False

    @staticmethod
    def _weights_similar(a: dict, b: dict, eps: float = 0.02) -> bool:
        keys = set(a) | set(b)
        for k in keys:
            if abs(a.get(k, 0.0) - b.get(k, 0.0)) > eps:
                return False
        return True

    # ---- per frame --------------------------------------------------------

    def _tick_animation(self, dt):
        # BVH playback takes precedence when loaded.
        if self.bvh_animation is not None:
            self._t_anim += dt
            pose, root = self.bvh_animation.sample(self._t_anim)
            self.character.set_pose(pose, root_offset=root)
            return
        if not self.walking:
            return
        self._t_walk += dt
        pose, root = skeleton.walk_pose(
            self.character.bones, self._t_walk, speed=self.walk_speed,
        )
        self.character.set_pose(pose, root_offset=root)

    def _head_world_pos(self, bone_mats):
        """World-space position of the top of the skull."""
        try:
            i = skeleton.bone_index(self.character.bones, "head")
        except KeyError:
            return np.array([0.0, 1.6, 0.0], dtype=np.float32) + self.character.root_offset
        tip_local = np.asarray(self.character.bones[i][3], dtype=np.float32)
        p = bone_mats[i] @ np.array([tip_local[0], tip_local[1], tip_local[2], 1.0], dtype=np.float32)
        return p[:3]

    def _draw_scene(self, proj, view, bone_mats):
        glUseProgram(self.skin_prog.prog)
        glUniformMatrix4fv(self.skin_prog.u_proj, 1, GL_TRUE, proj)
        glUniformMatrix4fv(self.skin_prog.u_view, 1, GL_TRUE, view)
        renderer.upload_bones(self.skin_prog.u_bones, bone_mats)
        app = self.character.appearance
        glUniform1f(self.skin_prog.u_seed, float(app.seed))
        glUniform1i(self.skin_prog.u_light_style, 0)
        glUniform1i(self.skin_prog.u_print_style, int(app.print_style))
        glUniform1f(self.skin_prog.u_print_strength, float(app.print_strength))
        glUniform1i(self.skin_prog.u_stamp_style, int(app.stamp_style))
        glUniform1f(self.skin_prog.u_stamp_strength, float(app.stamp_strength))

        for d in self.character.drawables:
            glUniform3f(self.skin_prog.u_color, *d.color)
            glUniform1i(self.skin_prog.u_mode, d.mode)
            glUniform1i(self.skin_prog.u_material, int(getattr(d, "material", 0)))
            glUniform1f(self.skin_prog.u_bend_inflate,
                        0.025 if d.mode == 1 else 0.0)
            d.mesh_gpu.draw()

        if self.show_bones:
            pts = self.character.skeleton_line_points(bone_mats)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self.line_prog.prog)
            glUniformMatrix4fv(self.line_prog.u_proj, 1, GL_TRUE, proj)
            glUniformMatrix4fv(self.line_prog.u_view, 1, GL_TRUE, view)
            glUniform3f(self.line_prog.u_color, *self.BONE_COLOR)
            self.character.lines.upload(pts)
            self.character.lines.draw(pts.size // 3)
            glEnable(GL_DEPTH_TEST)

    def _pip_view(self, target, eye_offset):
        target = np.asarray(target, dtype=np.float32)
        eye = target + np.asarray(eye_offset, dtype=np.float32)
        return mathx.look_at(eye, target, [0, 1, 0])

    def _draw_pip(self, x, y, vw, vh, proj, view, bone_mats):
        # Border frame: clear a 1px-larger region to a light color, then the
        # interior to the scene background, leaving a thin outline.
        glEnable(GL_SCISSOR_TEST)
        border = 2
        glScissor(x - border, y - border, vw + 2 * border, vh + 2 * border)
        glClearColor(0.55, 0.6, 0.7, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glScissor(x, y, vw, vh)
        glClearColor(*self.BG)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(x, y, vw, vh)
        self._draw_scene(proj, view, bone_mats)
        glDisable(GL_SCISSOR_TEST)

    def _draw(self, w, h):
        glViewport(0, 0, w, h)
        glClearColor(*self.BG)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # A 55 deg vertical FOV with the default camera frames feet-to-head
        # without clipping while keeping the face large enough to read.
        proj = mathx.perspective(math.radians(55), w / max(h, 1), 0.1, 50.0)
        # Camera follows the character's forward motion so it stays in frame
        # as it walks along +Z. Only track the horizontal component; vertical
        # bob is absorbed by the framing.
        follow = (0.0, 0.0, float(self.character.root_offset[2]))
        view = self.camera.view(follow=follow)
        bone_mats = self.character.bone_matrices()

        self._draw_scene(proj, view, bone_mats)

        # ---- picture-in-picture cameras (right edge) ----------------------
        rx = self.character.root_offset[0]
        rz = self.character.root_offset[2]
        head_pos = self._head_world_pos(bone_mats)

        pip_w = max(180, w // 5)
        pip_h = max(220, h // 3)
        margin = 16
        right_x = w - pip_w - margin

        # Face cam: tight on the head, viewed from slightly above eye level.
        face_target = head_pos + np.array([0.0, -0.04, 0.0], dtype=np.float32)
        face_view = self._pip_view(face_target, (0.0, 0.05, 0.55))
        face_proj = mathx.perspective(math.radians(28), pip_w / pip_h, 0.05, 20.0)
        face_y = h - pip_h - margin
        self._draw_pip(right_x, face_y, pip_w, pip_h, face_proj, face_view, bone_mats)

        # Body cam: full figure framed head-to-toe, slight 3/4 angle.
        body_target = np.array([rx, 0.95, rz], dtype=np.float32)
        body_view = self._pip_view(body_target, (1.1, 0.25, 2.6))
        body_proj = mathx.perspective(math.radians(38), pip_w / pip_h, 0.1, 30.0)
        body_y = face_y - pip_h - margin
        self._draw_pip(right_x, body_y, pip_w, pip_h, body_proj, body_view, bone_mats)

        # Restore main viewport for any subsequent draws (e.g. swap).
        glViewport(0, 0, w, h)

        # HUD: current animation label, top-left.
        self.text_overlay.draw(w, h, x=18, y=18)

    # ---- main loop --------------------------------------------------------

    def run(self):
        try:
            prev = time.perf_counter()
            while not glfw.window_should_close(self.win):
                glfw.poll_events()
                now = time.perf_counter()
                dt = now - prev
                prev = now

                self._tick_animation(dt)
                if self._tick_lipsync():
                    self._stop_lipsync()

                w, h = glfw.get_framebuffer_size(self.win)
                if h > 0:
                    self._draw(w, h)
                glfw.swap_buffers(self.win)
        finally:
            self.text_overlay.delete()
            self.character.delete()
            glfw.terminate()
