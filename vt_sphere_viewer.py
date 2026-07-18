#!/usr/bin/env python3
"""Sphere-band viewer: the mosaic wrapped onto a band of a sphere.

The image maps equirectangularly onto a band centered on the equator:
latitude spans -60..+60, and the longitude span is whatever the image's
aspect ratio dictates for a distortion-free equator (a 16:9 mosaic gets
120 * 16/9 = 213 degrees; capped at 360).

Two modes:
  * default: the band is on the OUTSIDE of the sphere (globe-style); the
    camera orbits it, scroll changes height above the surface.
  * --inside: the camera sits at the sphere's center looking out
    (panorama-style); scroll changes FOV.

Clicking glides the camera (ease-in/ease-out) to center the clicked
*image* — found via the mosaic's layout.json manifest, falling back to
the clicked point — by moving the camera's azimuth/elevation, while
zooming +50%. The clicked file's name is shown in the window title.

Streaming, page table, atlas, and shaders are shared with vt_viewer.py —
virtual texturing doesn't care what geometry the UVs live on.

Controls: drag = orbit / look around | scroll = zoom
          click = center clicked image and zoom +50%
          F = freeze/unfreeze streaming (explore the frozen LOD state)
          L = LOD debug overlay | R = reset view | ESC = quit
"""
import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *  # noqa: F403
from PIL import Image

from vt_viewer import (VERT, FRAG_MAIN, FRAG_FEEDBACK, FrustumLines,
                       VirtualTexture, compile_program, perspective, look_at)

LAT_MAX = math.radians(60.0)
RADIUS = 10.0
OUT_FOV = 55.0            # fixed perspective FOV in outside mode


def band_mesh(lon_half, z_sign, n_lon=192, n_lat=64):
    """Band vertices; z_sign picks the mapping that reads unmirrored
    from outside (+1) or from the center (-1)."""
    verts = []
    for j in range(n_lat + 1):
        v = j / n_lat
        lat = (0.5 - v) * 2 * LAT_MAX
        for i in range(n_lon + 1):
            u = i / n_lon
            lon = (u - 0.5) * 2 * lon_half
            verts.append((RADIUS * math.cos(lat) * math.sin(lon),
                          RADIUS * math.sin(lat),
                          z_sign * RADIUS * math.cos(lat) * math.cos(lon),
                          u, v))
    idx = []
    for j in range(n_lat):
        for i in range(n_lon):
            a = j * (n_lon + 1) + i
            b, c = a + 1, a + n_lon + 1
            idx += [a, b, c + 1, a, c + 1, c]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SphereViewer:

    def __init__(self, args):
        self.args = args
        self.inside = args.inside
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.win = glfw.create_window(1280, 800, "vt sphere viewer",
                                      None, None)
        if not self.win:
            raise RuntimeError("window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if args.frames else 1)

        self.prog_main = compile_program(VERT, FRAG_MAIN)
        self.prog_fb = compile_program(VERT, FRAG_FEEDBACK)
        self.frustum = FrustumLines()
        self.vt = VirtualTexture(args.pyramid)
        self.manifest = self.load_manifest()

        # longitude span from the image aspect: distortion-free equator
        aspect = self.vt.virt_w / self.vt.virt_h
        self.lon_half = min(math.pi, LAT_MAX * aspect)
        print(f"band: lon ±{math.degrees(self.lon_half):.1f}°, "
              f"lat ±{math.degrees(LAT_MAX):.0f}°, "
              f"{'inside' if self.inside else 'outside'} view")

        verts, idx = band_mesh(self.lon_half, -1 if self.inside else 1)
        self.n_idx = len(idx)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        vbo, ebo = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

        fbw, fbh = glfw.get_framebuffer_size(self.win)
        self.fb_w, self.fb_h = max(fbw // 10, 64), max(fbh // 10, 64)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.fb_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fb_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.fb_w, self.fb_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.fb_tex, 0)
        rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              self.fb_w, self.fb_h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, rb)
        assert (glCheckFramebufferStatus(GL_FRAMEBUFFER)
                == GL_FRAMEBUFFER_COMPLETE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # az/el = camera azimuth/elevation; zoom = FOV (inside) or
        # height above the surface (outside)
        self.az, self.el = 0.0, 0.0
        self.fov = 75.0
        self.height = 12.0
        self.debug = 0
        self.freeze = 0
        self._drag = None
        self._press = None
        self._anim = None
        self._last_click = ""
        self._vel = [0.0, 0.0]        # az/el rad/s for trackball inertia
        self._move_t = None           # time of last drag movement
        self._last_input = time.time()
        self._idle_dir = 1.0
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_key_callback(self.win, self.on_key)

    def load_manifest(self):
        """layout.json rects for click-to-center; derived from the pyramid
        name (<x>_pyramid -> <x>_layout.json) unless given explicitly."""
        path = self.args.manifest
        if path is None:
            name = Path(self.args.pyramid).name
            if name.endswith("_pyramid"):
                path = Path(self.args.pyramid).parent / \
                    f"{name[:-len('_pyramid')]}_layout.json"
        if path and Path(path).exists():
            rects = json.loads(Path(path).read_text())
            print(f"manifest: {path} ({len(rects)} images)")
            return rects
        return None

    # ---- camera --------------------------------------------------------
    def radial(self):
        """Unit vector at (az, el); +z at az=0 so lon==az faces camera."""
        return np.array([math.cos(self.el) * math.sin(self.az),
                         math.sin(self.el),
                         math.cos(self.el) * math.cos(self.az)], np.float32)

    def mvp(self, w, h):
        up = np.array([0, 1, 0], np.float32)
        if self.inside:
            d = self.radial()
            view = look_at(np.zeros(3, np.float32),
                           d * np.array([1, 1, -1], np.float32), up)
            proj = perspective(self.fov, w / h, 0.05, 100.0)
        else:
            eye = (RADIUS + self.height) * self.radial()
            view = look_at(eye, np.zeros(3, np.float32), up)
            proj = perspective(OUT_FOV, w / h, 0.01, 200.0)
        return proj @ view

    def cursor_uv(self, cx, cy):
        """Cursor -> ray -> point on band -> mosaic uv (or None)."""
        w, h = glfw.get_window_size(self.win)
        if w == 0 or h == 0:
            return None
        inv = np.linalg.inv(self.mvp(w, h).astype(np.float64))
        ndc = (2 * cx / w - 1, 1 - 2 * cy / h)
        a = inv @ np.array([ndc[0], ndc[1], -1, 1.0])
        b = inv @ np.array([ndc[0], ndc[1], 1, 1.0])
        p0 = a[:3] / a[3]
        d = b[:3] / b[3] - p0
        d /= np.linalg.norm(d)
        if self.inside:
            p = d * RADIUS
        else:                                   # near ray-sphere hit
            bb = np.dot(p0, d)
            disc = bb * bb - (np.dot(p0, p0) - RADIUS ** 2)
            if disc < 0:
                return None
            t = -bb - math.sqrt(disc)
            if t <= 0:
                return None
            p = p0 + t * d
        lat = math.asin(min(max(p[1] / RADIUS, -1.0), 1.0))
        lon = math.atan2(p[0], -p[2] if self.inside else p[2])
        if abs(lat) > LAT_MAX or abs(lon) > self.lon_half:
            return None
        return (0.5 + lon / (2 * self.lon_half),
                0.5 - lat / (2 * LAT_MAX))

    # ---- input ---------------------------------------------------------
    def on_mouse_button(self, win, button, action, mods):
        pos = glfw.get_cursor_pos(win)
        self._last_input = time.time()
        if action == glfw.PRESS:
            self._anim = None
            self._vel = [0.0, 0.0]    # grabbing stops the spin
            self._move_t = None
            self._drag = pos
            self._press = pos
        else:
            if (button == glfw.MOUSE_BUTTON_LEFT and self._press is not None
                    and abs(pos[0] - self._press[0]) < 4
                    and abs(pos[1] - self._press[1]) < 4):
                self._vel = [0.0, 0.0]
                self.click_center_zoom(*pos)
            elif (self._move_t is None
                    or time.time() - self._move_t > 0.12):
                self._vel = [0.0, 0.0]  # held still before release: no fling
            self._drag = None
            self._press = None

    def drag_rate(self):
        if self.inside:
            return 0.0025 * self.fov / 70.0     # slower when zoomed in
        return 0.005 * min(max(self.height / RADIUS, 0.03), 1.5)

    def on_cursor(self, win, x, y):
        if not self._drag:
            return
        now = time.time()
        self._last_input = now
        px, py = self._drag
        self._drag = (x, y)
        rate = self.drag_rate()
        daz, del_ = -(x - px) * rate, (y - py) * rate
        self.az = wrap_pi(self.az + daz)
        self.el = min(max(self.el + del_, -LAT_MAX - 0.25), LAT_MAX + 0.25)
        # velocity estimate for release momentum (smoothed, capped)
        dt = now - self._move_t if self._move_t else 0
        self._move_t = now
        if 1e-4 < dt < 0.2:
            s = min(dt / 0.05, 1.0)
            self._vel[0] += (daz / dt - self._vel[0]) * s
            self._vel[1] += (del_ / dt - self._vel[1]) * s
            cap = 4.0
            self._vel[0] = min(max(self._vel[0], -cap), cap)
            self._vel[1] = min(max(self._vel[1], -cap), cap)

    def on_scroll(self, win, dx, dy):
        self._anim = None
        self._last_input = time.time()
        if self.inside:
            self.fov = min(max(self.fov * (0.92 ** dy), 3.0), 110.0)
        else:
            self.height = min(max(self.height * (0.92 ** dy), 0.02), 60.0)

    def on_key(self, win, key, sc, action, mods):
        self._last_input = time.time()
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_L:
            self.debug ^= 1
        elif key == glfw.KEY_F:
            self.toggle_freeze()
        elif key == glfw.KEY_R:
            self._anim = None
            self.az, self.el = 0.0, 0.0
            self.fov, self.height = 75.0, 12.0

    def toggle_freeze(self):
        self.freeze ^= 1
        if self.freeze:                    # pin the frustum that chose
            w, h = glfw.get_framebuffer_size(self.win)   # the working set
            if self.inside:
                eye = np.zeros(3, np.float32)
                near, far = 0.05, 2.5 * RADIUS
            else:
                eye = (RADIUS + self.height) * self.radial()
                near, far = 0.01, self.height + 2 * RADIUS
            self.frustum.capture(self.mvp(w, h), eye, near, far)
            self.vt.evict_unused()         # keep only this view's pages

    # ---- click: center the image under the cursor, zoom +50% ----------
    def click_center_zoom(self, cx, cy, factor=1.5, duration=0.5):
        uv = self.cursor_uv(cx, cy)
        if uv is None:
            return
        cu, cv = uv
        if self.manifest:
            px, py = cu * self.vt.virt_w, cv * self.vt.virt_h
            for r in self.manifest:
                if (r["x"] <= px < r["x"] + r["w"]
                        and r["y"] <= py < r["y"] + r["h"]):
                    cu = (r["x"] + r["w"] / 2) / self.vt.virt_w
                    cv = (r["y"] + r["h"] / 2) / self.vt.virt_h
                    self._last_click = Path(r["path"]).name
                    break
        a = self._anim
        base_zoom = a["zoom1"] if a else (self.fov if self.inside
                                          else self.height)
        az1 = (cu - 0.5) * 2 * self.lon_half
        el1 = (0.5 - cv) * 2 * LAT_MAX
        self._anim = {
            "t": time.time(), "dur": duration,
            "az0": self.az, "az1": self.az + wrap_pi(az1 - self.az),
            "el0": self.el, "el1": el1,
            "zoom0": self.fov if self.inside else self.height,
            "zoom1": max(base_zoom / factor, 3.0 if self.inside else 0.02),
        }

    def update_free_motion(self, dt):
        """Trackball inertia after a fling, and slow idle auto-rotate."""
        if self._drag or self._anim is not None:
            return
        full_wrap = self.lon_half >= math.pi - 1e-6
        vaz, vel_ = self._vel
        if abs(vaz) > 1e-3 or abs(vel_) > 1e-3:
            self.az += vaz * dt
            self.el += vel_ * dt
            decay = math.exp(-3.5 * dt)
            self._vel[0] *= decay
            self._vel[1] *= decay
            if abs(self.el) > LAT_MAX + 0.25:       # bump the pole clamp
                self.el = math.copysign(LAT_MAX + 0.25, self.el)
                self._vel[1] = 0.0
            if full_wrap:
                self.az = wrap_pi(self.az)
            elif abs(self.az) > self.lon_half:      # bump the band edge
                self.az = math.copysign(self.lon_half, self.az)
                self._vel[0] = 0.0
        elif (self.args.idle_delay > 0
              and time.time() - self._last_input > self.args.idle_delay):
            step = math.radians(self.args.idle_speed) * dt
            if full_wrap:
                self.az = wrap_pi(self.az + step)
            else:                                   # ping-pong the band
                lim = 0.85 * self.lon_half
                if abs(self.az) > lim:
                    self._idle_dir = -math.copysign(1.0, self.az)
                self.az += step * self._idle_dir

    def update_anim(self):
        a = self._anim
        if a is None:
            return
        u = min((time.time() - a["t"]) / a["dur"], 1.0)
        e = u * u * (3 - 2 * u)              # smoothstep ease-in/out
        self.az = a["az0"] + (a["az1"] - a["az0"]) * e
        self.el = a["el0"] + (a["el1"] - a["el0"]) * e
        zoom = a["zoom0"] + (a["zoom1"] - a["zoom0"]) * e
        if self.inside:
            self.fov = zoom
        else:
            self.height = zoom
        if u >= 1.0:
            self.az = wrap_pi(self.az)
            self._anim = None

    # ---- frame loop ----------------------------------------------------
    def draw(self, prog, mvp):
        self.vt.bind(prog)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_TRUE, mvp)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.n_idx, GL_UNSIGNED_INT, None)

    def feedback_pass(self, mvp, main_w):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.fb_w, self.fb_h)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.prog_fb)
        glUniform1f(glGetUniformLocation(self.prog_fb, "uLodBias"),
                    math.log2(self.fb_w / main_w))
        self.draw(self.prog_fb, mvp)
        buf = glReadPixels(0, 0, self.fb_w, self.fb_h,
                           GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        px = np.frombuffer(buf, np.uint8).reshape(-1, 4)
        px = px[px[:, 3] >= 240].astype(np.uint32)
        page_x = px[:, 0] | ((px[:, 2] & 15) << 8)
        page_y = px[:, 1] | ((px[:, 2] >> 4) << 8)
        level = px[:, 3] - 240
        codes = np.unique(page_x | (page_y << 12) | (level << 24))
        self.vt.request([(int(c >> 24), int(c & 4095), int((c >> 12) & 4095))
                         for c in codes])

    def run(self):
        print(__doc__.split("Controls:")[1].strip()
              if self.args.frames is None else "self-test run")
        glEnable(GL_DEPTH_TEST)
        frame, t0, fps_t, fps_n = 0, time.time(), time.time(), 0
        prev_t = time.time()
        while not glfw.window_should_close(self.win):
            now = time.time()
            dt, prev_t = min(now - prev_t, 0.1), now
            if self.args.frames is None:
                self.update_free_motion(dt)
            if self.args.frames is not None:      # scripted sweep
                if frame < self.args.frames:
                    k = frame / max(self.args.frames - 1, 1)
                    self.az = 0.7 * self.lon_half * math.sin(2 * math.pi * k)
                    self.el = 0.5 * LAT_MAX * math.sin(4 * math.pi * k)
                    if self.inside:
                        self.fov = 75.0 * (self.args.end_fov / 75.0) ** k
                    else:
                        self.height = 12.0 * (self.args.end_height / 12.0) ** k
                else:
                    extra = frame - self.args.frames
                    if self.args.click_test and extra in (20, 60, 100):
                        w, h = glfw.get_window_size(self.win)
                        self.click_center_zoom(w * 0.32, h * 0.38)
                        if self._anim:
                            print(f"click -> {self._last_click or '(point)'}"
                                  f"  az {math.degrees(self._anim['az1']):.1f}"
                                  f"  el {math.degrees(self._anim['el1']):.1f}"
                                  f"  zoom {self._anim['zoom1']:.2f}")
                    min_extra = 120 if self.args.click_test else 0
                    settled = (extra > min_extra and self._anim is None
                               and not self.vt.pending
                               and self.vt.ready_q.empty())
                    if settled or extra > 600:
                        break
            self.update_anim()
            w, h = glfw.get_framebuffer_size(self.win)
            mvp = self.mvp(w, h)

            if not self.freeze:
                self.feedback_pass(mvp, w)
                self.vt.pump_uploads()

            glViewport(0, 0, w, h)
            glClearColor(0.05, 0.06, 0.08, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.prog_main)
            glUniform1i(glGetUniformLocation(self.prog_main, "uDebug"),
                        self.debug)
            self.draw(self.prog_main, mvp)
            if self.freeze:
                self.frustum.draw(mvp)

            glfw.swap_buffers(self.win)
            glfw.poll_events()
            frame += 1
            fps_n += 1
            if time.time() - fps_t > 0.5:
                fps = fps_n / (time.time() - fps_t)
                click = f" | {self._last_click}" if self._last_click else ""
                glfw.set_window_title(
                    self.win,
                    f"vt sphere viewer | {fps:5.1f} fps | resident "
                    f"{len(self.vt.residency)}/{self.vt.n_slots} | pending "
                    f"{len(self.vt.pending)}"
                    f"{' | FROZEN' if self.freeze else ''}{click}")
                fps_t, fps_n = time.time(), 0

        if self.args.screenshot:
            w, h = glfw.get_framebuffer_size(self.win)
            buf = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[::-1]
            Image.fromarray(img).save(self.args.screenshot)
            print(f"screenshot -> {self.args.screenshot}")
        print(f"frames: {frame}  time: {time.time() - t0:.1f}s  "
              f"tiles loaded: {self.vt.loads_done}  "
              f"resident: {len(self.vt.residency)}/{self.vt.n_slots}")
        glfw.terminate()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pyramid", default="test_image_16k_pyramid")
    ap.add_argument("--inside", action="store_true",
                    help="view from the sphere's center (panorama mode)")
    ap.add_argument("--manifest", default=None,
                    help="layout.json for click-to-center "
                         "(default: derived from the pyramid name)")
    ap.add_argument("--frames", type=int, default=None,
                    help="run a scripted N-frame sweep and exit")
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--end-fov", type=float, default=28.0,
                    help="final FOV for the scripted sweep (inside mode)")
    ap.add_argument("--end-height", type=float, default=1.0,
                    help="final surface height for the sweep (outside mode)")
    ap.add_argument("--click-test", action="store_true",
                    help="fire 3 click-center-zooms during settle")
    ap.add_argument("--idle-delay", type=float, default=5.0,
                    help="seconds of no input before the slow auto-rotate "
                         "starts (<= 0 disables)")
    ap.add_argument("--idle-speed", type=float, default=3.0,
                    help="auto-rotate speed in degrees/second")
    SphereViewer(ap.parse_args()).run()


if __name__ == "__main__":
    main()
