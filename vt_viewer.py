#!/usr/bin/env python3
"""Step 3: Orbit viewer for a tiled pyramid using virtual texturing.

A software virtual-texturing system in the spirit of Mayer's LibVT /
id Software's MegaTexture:

  * The full image is never resident. A 4096^2 physical texture atlas
    holds 260^2 tile slots (256 payload + 2px baked border).
  * A page-table texture *array* (one layer per pyramid level) maps each
    virtual page to its atlas slot. Pages that are not resident inherit
    the entry of their finest resident ancestor, so the shader always has
    something to sample (progressive refinement, no holes).
  * Visibility feedback pass: the scene is re-rendered each frame into a
    small offscreen buffer with a shader that outputs (pageX, pageY,
    level) per pixel. It is read back on the CPU to build the working set.
  * Tiles are decoded on a background thread, uploaded (budgeted per
    frame) into the atlas, and evicted LRU when the atlas is full.
  * The sampling shader picks the LOD per fragment from UV derivatives
    and blends two page-table levels (manual trilinear) to hide seams
    and popping.

Works with any rectangular pyramid from build_pyramid.py (square 16k test
image, 16:9 mosaic, ...); partial edge tiles are supported.

Controls: drag = orbit | right-drag/shift-drag = pan | scroll = zoom
          hover = outline the mosaic image under the pointer
          click = center the view on that image (or on the point)
          H = cycle highlight style | F = freeze/unfreeze streaming
          L = LOD debug overlay | R = reset camera | ESC = quit
"""
import argparse
import ctypes
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *  # noqa: F403
from PIL import Image

# ---------------------------------------------------------------- shaders

VERT = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uMVP;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = uMVP * vec4(aPos, 1.0); }
"""

VT_COMMON = """#version 330 core
uniform sampler2DArray uPageTable;
uniform sampler2D uAtlas;
uniform vec2 uVirtDim;   // full-res image size in texels
uniform vec2 uPagesF;    // uVirtDim / tile size (may be fractional)
uniform float uMaxLod, uAtlasSize, uSlot, uTile, uBorder;
in vec2 vUV;
out vec4 frag;

float vtLod(vec2 uv, float bias) {
    vec2 dx = dFdx(uv) * uVirtDim;
    vec2 dy = dFdy(uv) * uVirtDim;
    float rho = max(length(dx), length(dy));
    return clamp(log2(max(rho, 1e-6)) + bias, 0.0, uMaxLod);
}

ivec2 vtPageCoord(vec2 uv, int level) {
    vec2 pages = uPagesF / exp2(float(level));
    ivec2 n = ivec2(ceil(pages - 1e-5));       // page count at this level
    return clamp(ivec2(uv * pages), ivec2(0), n - 1);
}

vec4 vtSampleLevel(vec2 uv, int level) {
    ivec2 pc = vtPageCoord(uv, level);
    vec3 e = floor(texelFetch(uPageTable, ivec3(pc, level), 0).rgb
                   * 255.0 + 0.5);
    vec2 inTile = fract(uv * uPagesF / exp2(e.z));  // e.z: resident level
    vec2 auv = (e.xy * uSlot + uBorder + inTile * uTile) / uAtlasSize;
    return texture(uAtlas, auv);
}
"""

FRAG_MAIN = VT_COMMON + """
uniform vec4 uHoverRect;   // (u0,v0,u1,v1) in image UV; off when z <= x
uniform int uHiStyle;      // 1 flat  2 halo+core  3 adaptive  4 scrim  5 ants
uniform float uTime;
uniform float uHiScale;    // framebuffer px per logical point (HiDPI)

// Screen-space outline of the hovered image. The rect is a signed distance
// field in UV; dividing by the SDF's screen-space gradient converts it to
// pixels, so the outline keeps a constant pixel width at any zoom, on flat
// or curved geometry alike.
vec3 vtHighlight(vec3 col, vec2 uv) {
    if (uHiStyle == 0 || uHoverRect.z <= uHoverRect.x) return col;
    vec2 ctr = (uHoverRect.xy + uHoverRect.zw) * 0.5;
    vec2 hlf = (uHoverRect.zw - uHoverRect.xy) * 0.5;
    vec2 d = abs(uv - ctr) - hlf;
    float sd = max(d.x, d.y);                        // < 0 inside the image
    float spx = length(vec2(dFdx(sd), dFdy(sd)));    // UV units per pixel
    if (spx <= 0.0) return col;
    float a = abs(sd / spx);                         // |distance| in pixels

    if (uHiStyle == 4 && sd > 0.0) col *= 0.42;      // scrim: dim the rest

    float k = max(uHiScale, 1.0);                    // widths in points
    float core = 1.0 - smoothstep(0.6 * k, 1.6 * k, a);   // ~3pt core line
    float halo = 1.0 - smoothstep(1.6 * k, 3.4 * k, a);   // flanking halo

    if (uHiStyle == 1)                               // plain yellow 2px
        return mix(col, vec3(1.0, 0.85, 0.0), core);

    if (uHiStyle == 5) {                             // marching ants
        float run = (abs(d.x) > abs(d.y)) ? gl_FragCoord.y : gl_FragCoord.x;
        vec3 ant = fract((run - uTime * 40.0 * k) / (14.0 * k)) < 0.5
                 ? vec3(1.0) : vec3(0.05);
        col = mix(col, vec3(0.0), halo * 0.5);
        return mix(col, ant, core);
    }

    vec3 hi = vec3(1.0, 0.85, 0.0);                  // 2: yellow core
    if (uHiStyle == 3 || uHiStyle == 4)              // 3/4: pick for contrast
        hi = dot(col, vec3(0.299, 0.587, 0.114)) > 0.5 ? vec3(0.0) : vec3(1.0);
    col = mix(col, vec3(0.0), halo * 0.55);          // dark halo under it
    return mix(col, hi, core);
}

uniform int uDebug;
void main() {
    vec2 uv = clamp(vUV, 0.0, 0.9999999);
    float lod = vtLod(uv, 0.0);
    int l0 = int(floor(lod));
    int l1 = min(l0 + 1, int(uMaxLod));
    vec4 c = mix(vtSampleLevel(uv, l0), vtSampleLevel(uv, l1), fract(lod));
    if (uDebug == 1) {
        vec3 lc = 0.5 + 0.5 * cos(6.2832 * (lod / 7.0 + vec3(0.0, 0.33, 0.67)));
        c.rgb = mix(c.rgb, lc, 0.45);
    }
    frag = vec4(vtHighlight(c.rgb, uv), 1.0);
}
"""

FRAG_FEEDBACK = VT_COMMON + """
uniform float uLodBias;
// pack (pageX:12, pageY:12, level:4) into RGBA8; alpha's high nibble
// 0xF marks a valid sample (clear color alpha 0 = no geometry)
void main() {
    vec2 uv = clamp(vUV, 0.0, 0.9999999);
    int l0 = int(floor(vtLod(uv, uLodBias)));
    ivec2 pc = vtPageCoord(uv, l0);
    frag = vec4(float(pc.x & 255), float(pc.y & 255),
                float((pc.x >> 8) | ((pc.y >> 8) << 4)),
                float(240 + l0)) / 255.0;
}
"""


LINE_VERT = """#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

LINE_FRAG = """#version 330 core
uniform vec4 uColor;
out vec4 frag;
void main() { frag = uColor; }
"""


HI_STYLES = ["off", "flat yellow 2px", "yellow core + dark halo",
             "contrast-adaptive core", "adaptive + scrim dim",
             "marching ants"]


def check_pyramid(pyramid_dir):
    """Fail with build instructions instead of a bare traceback when the
    pyramid hasn't been generated yet (the usual first-run mistake)."""
    d = Path(pyramid_dir)
    if (d / "meta.json").exists():
        return
    why = (f"'{d}' has no meta.json" if d.is_dir()
           else f"no such directory: '{d}'")
    msg = [f"error: {why}", "", "Build a pyramid first:", "",
           "    ./gen_test_image_16k.py   # synthetic 16384\u00b2 test image",
           "    ./build_pyramid.py        # -> pics/test_image_16k_pyramid/",
           "", "...or from your own photos:", "",
           "    ./layout_mosaic.py pics/some_tree",
           "    ./build_pyramid.py pics/some_tree_mosaic.npy",
           f"    ./{Path(sys.argv[0]).name} pics/some_tree_mosaic_pyramid"]
    here = sorted(str(p) for d in (".", "pics")
                  for p in Path(d).glob("*_pyramid")
                  if (p / "meta.json").exists())
    if here:
        msg += ["", "Pyramids found in this directory:"]
        msg += [f"    {n}" for n in here]
    raise SystemExit("\n".join(msg))


def load_manifest(pyramid, explicit=None):
    """layout.json rects for hover/click, derived from the pyramid name
    (<x>_pyramid -> <x>_layout.json) unless given explicitly."""
    path = explicit
    if path is None:
        name = Path(pyramid).name
        if name.endswith("_pyramid"):
            path = Path(pyramid).parent / \
                f"{name[:-len('_pyramid')]}_layout.json"
    if path and Path(path).exists():
        rects = json.loads(Path(path).read_text())
        print(f"manifest: {path} ({len(rects)} images)")
        return rects
    return None


def rect_at_uv(manifest, u, v, virt_w, virt_h):
    """(u0, v0, u1, v1, name) of the mosaic image under (u, v), or None."""
    if not manifest:
        return None
    px, py = u * virt_w, v * virt_h
    for r in manifest:
        if (r["x"] <= px < r["x"] + r["w"]
                and r["y"] <= py < r["y"] + r["h"]):
            return (r["x"] / virt_w, r["y"] / virt_h,
                    (r["x"] + r["w"]) / virt_w, (r["y"] + r["h"]) / virt_h,
                    Path(r["path"]).name)
    return None


def compile_program(vs_src, fs_src):
    prog = glCreateProgram()
    for kind, src in ((GL_VERTEX_SHADER, vs_src), (GL_FRAGMENT_SHADER, fs_src)):
        sh = glCreateShader(kind)
        glShaderSource(sh, src)
        glCompileShader(sh)
        if not glGetShaderiv(sh, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(sh).decode())
        glAttachShader(prog, sh)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    return prog


class FrustumLines:
    """Captures a camera frustum as a wireframe and draws it later (the
    frozen-LOD visualization: the yellow frustum shows which view chose
    the currently resident working set)."""

    EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # near rectangle
             (4, 5), (5, 6), (6, 7), (7, 4),      # far rectangle
             (0, 4), (1, 5), (2, 6), (3, 7)]      # connecting edges

    def __init__(self):
        self.prog = compile_program(LINE_VERT, LINE_FRAG)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(self.EDGES) * 2 * 12, None,
                     GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12,
                              ctypes.c_void_p(0))
        self.valid = False

    def capture(self, mvp, eye, near, far_dist):
        """Near corners come from the inverse MVP; far corners extend the
        per-corner rays from the eye out to far_dist (a display depth,
        not the projection's far plane, so the wireframe stays legible)."""
        inv = np.linalg.inv(mvp.astype(np.float64))
        pts = []
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = inv @ np.array([x, y, -1.0, 1.0])
            pts.append(p[:3] / p[3])
        k = far_dist / near
        pts += [eye + (p - eye) * k for p in pts[:4]]
        verts = np.array([pts[i] for e in self.EDGES for i in e],
                         np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        self.valid = True

    def draw(self, mvp):
        if not self.valid:
            return
        glUseProgram(self.prog)
        glUniformMatrix4fv(glGetUniformLocation(self.prog, "uMVP"),
                           1, GL_TRUE, mvp)
        glUniform4f(glGetUniformLocation(self.prog, "uColor"),
                    1.0, 0.9, 0.0, 1.0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, len(self.EDGES) * 2)


# ---------------------------------------------------------------- matrices

def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    u = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, u, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------- VT system

class VirtualTexture:
    """Page table + physical atlas + async tile streaming."""

    ATLAS_SIZE = 4096
    MAX_UPLOADS_PER_FRAME = 24

    def __init__(self, pyramid_dir):
        self.dir = Path(pyramid_dir)
        check_pyramid(self.dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.virt_w = meta.get("image_width", meta.get("image_size"))
        self.virt_h = meta.get("image_height", meta.get("image_size"))
        self.tile = meta["tile_size"]
        self.border = meta["border"]
        self.levels = meta["levels"]
        self.fmt = meta["format"]
        self.slot_px = self.tile + 2 * self.border
        self.max_lod = self.levels - 1
        self.slots_per_row = self.ATLAS_SIZE // self.slot_px
        self.n_slots = self.slots_per_row ** 2

        # per-level page counts (partial edge tiles round up)
        self.nx, self.ny = [], []
        w, h = self.virt_w, self.virt_h
        for l in range(self.levels):
            self.nx.append(ceil_div(w, self.tile))
            self.ny.append(ceil_div(h, self.tile))
            w, h = ceil_div(w, 2), ceil_div(h, 2)

        # feedback encoding limits: 12 bits per page axis, 4-bit level
        assert self.nx[0] <= 4096 and self.ny[0] <= 4096, \
            "image exceeds 4096 tiles per axis"
        assert self.levels <= 16, "too many pyramid levels"
        assert glGetIntegerv(GL_MAX_TEXTURE_SIZE) >= self.ATLAS_SIZE

        # residency state
        self.own = [np.full((self.ny[l], self.nx[l]), -1, np.int32)
                    for l in range(self.levels)]
        self.residency = {}          # (l, tx, ty) -> slot
        self.slot_page = [None] * self.n_slots
        self.free_slots = list(range(self.n_slots - 1, -1, -1))
        self.lru = {}                # page -> last frame touched
        self.frame = 0
        self.dirty = True
        self.loads_done = 0

        # physical atlas
        self.atlas_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.ATLAS_SIZE,
                     self.ATLAS_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # page table: a texture array with one layer per pyramid level
        # (a mip chain would force power-of-two halving of the page grid,
        # which breaks for non-square images with partial tiles)
        self.table_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.table_tex)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
                     self.nx[0], self.ny[0], self.levels, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # streaming
        self.pending = set()
        self.req_q = queue.PriorityQueue()
        self.ready_q = queue.Queue()
        self.seq = 0
        t = threading.Thread(target=self._loader, daemon=True)
        t.start()

        # the coarsest tile must always be resident: load it synchronously
        root = (self.max_lod, 0, 0)
        self._upload_tile(root, self._decode(root))
        self.update_page_table()

    # ---- tile IO -------------------------------------------------------
    def _tile_path(self, page):
        l, tx, ty = page
        return self.dir / f"L{l}" / f"{ty}_{tx}.{self.fmt}"

    def _decode(self, page):
        img = Image.open(self._tile_path(page)).convert("RGB")
        return np.ascontiguousarray(np.asarray(img, np.uint8))

    def _loader(self):
        while True:
            _, _, page = self.req_q.get()
            try:
                self.ready_q.put((page, self._decode(page)))
            except Exception as e:  # missing/corrupt tile: drop the request
                print(f"tile load failed {page}: {e}")
                self.pending.discard(page)

    # ---- residency -----------------------------------------------------
    def _upload_tile(self, page, data):
        if page in self.residency:
            return True
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            slot = self._evict()
            if slot is None:
                return False        # nothing evictable this frame
        l, tx, ty = page
        sx = (slot % self.slots_per_row) * self.slot_px
        sy = (slot // self.slots_per_row) * self.slot_px
        glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, sx, sy, self.slot_px, self.slot_px,
                        GL_RGB, GL_UNSIGNED_BYTE, data)
        self.residency[page] = slot
        self.slot_page[slot] = page
        self.own[l][ty, tx] = slot
        self.lru[page] = self.frame
        self.dirty = True
        self.loads_done += 1
        return True

    def _evict(self):
        best, best_frame = None, None
        for slot, page in enumerate(self.slot_page):
            if page is None or page[0] == self.max_lod:
                continue            # never evict the root tile
            f = self.lru.get(page, -1)
            if f >= self.frame:     # in this frame's working set
                continue
            if best_frame is None or f < best_frame:
                best, best_frame = slot, f
        if best is None:
            return None
        page = self.slot_page[best]
        l, tx, ty = page
        del self.residency[page]
        self.lru.pop(page, None)
        self.own[l][ty, tx] = -1
        self.slot_page[best] = None
        self.dirty = True
        return best

    def evict_unused(self):
        """Evict every resident page the latest feedback pass did not
        touch (except the pinned root). Used by the freeze demo so the
        frozen state holds exactly the frozen view's working set."""
        for slot, page in enumerate(self.slot_page):
            if page is None or page[0] == self.max_lod:
                continue
            if self.lru.get(page, -1) >= self.frame:
                continue
            l, tx, ty = page
            del self.residency[page]
            self.lru.pop(page, None)
            self.own[l][ty, tx] = -1
            self.slot_page[slot] = None
            self.free_slots.append(slot)
            self.dirty = True
        if self.dirty:
            self.update_page_table()
            self.dirty = False

    def request(self, pages):
        """Feed this frame's working set (from the feedback pass)."""
        self.frame += 1
        for page in pages:
            l, tx, ty = page
            while l <= self.max_lod:          # include ancestors as fallback
                p = (l, tx, ty)
                if p in self.residency:
                    self.lru[p] = self.frame
                elif p not in self.pending:
                    self.pending.add(p)
                    self.seq += 1
                    self.req_q.put((-l, self.seq, p))  # coarse levels first
                l, tx, ty = l + 1, tx >> 1, ty >> 1

    def pump_uploads(self):
        for _ in range(self.MAX_UPLOADS_PER_FRAME):
            try:
                page, data = self.ready_q.get_nowait()
            except queue.Empty:
                break
            if self._upload_tile(page, data):
                self.pending.discard(page)
            else:                    # atlas full of in-use pages; retry later
                self.ready_q.put((page, data))
                break
        if self.dirty:
            self.update_page_table()
            self.dirty = False

    def update_page_table(self):
        """Rebuild all page-table layers: non-resident pages inherit their
        finest resident ancestor's entry (quadtree fallback)."""
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.table_tex)
        parent = None
        spr = self.slots_per_row
        for l in range(self.max_lod, -1, -1):
            nx, ny = self.nx[l], self.ny[l]
            t = np.zeros((ny, nx, 4), np.uint8)
            t[..., 3] = 255
            if parent is not None:
                t[:] = np.repeat(np.repeat(parent, 2, axis=0),
                                 2, axis=1)[:ny, :nx]
            o = self.own[l]
            m = o >= 0
            t[m, 0] = (o[m] % spr).astype(np.uint8)
            t[m, 1] = (o[m] // spr).astype(np.uint8)
            t[m, 2] = l
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, l, nx, ny, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, t)
            parent = t

    def bind(self, prog):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.table_tex)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
        glUseProgram(prog)
        glUniform1i(glGetUniformLocation(prog, "uPageTable"), 0)
        glUniform1i(glGetUniformLocation(prog, "uAtlas"), 1)
        glUniform2f(glGetUniformLocation(prog, "uVirtDim"),
                    self.virt_w, self.virt_h)
        glUniform2f(glGetUniformLocation(prog, "uPagesF"),
                    self.virt_w / self.tile, self.virt_h / self.tile)
        glUniform1f(glGetUniformLocation(prog, "uMaxLod"), self.max_lod)
        glUniform1f(glGetUniformLocation(prog, "uAtlasSize"), self.ATLAS_SIZE)
        glUniform1f(glGetUniformLocation(prog, "uSlot"), self.slot_px)
        glUniform1f(glGetUniformLocation(prog, "uTile"), self.tile)
        glUniform1f(glGetUniformLocation(prog, "uBorder"), self.border)


# ---------------------------------------------------------------- viewer

class Viewer:
    PLANE_WIDTH = 16.0

    def __init__(self, args):
        self.args = args
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.win = glfw.create_window(1280, 800, "virtual texture viewer",
                                      None, None)
        if not self.win:
            raise RuntimeError("window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if args.frames else 1)

        self.prog_main = compile_program(VERT, FRAG_MAIN)
        self.prog_fb = compile_program(VERT, FRAG_FEEDBACK)
        self.frustum = FrustumLines()
        self.vt = VirtualTexture(args.pyramid)

        # ground-plane quad with the image's aspect ratio
        hx = self.PLANE_WIDTH / 2
        hz = hx * self.vt.virt_h / self.vt.virt_w
        self.half = np.array([hx, hz], np.float32)
        verts = np.array([
            # x, y, z, u, v
            [-hx, 0, -hz, 0, 0], [hx, 0, -hz, 1, 0],
            [hx, 0, hz, 1, 1], [-hx, 0, hz, 0, 1]], np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
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

        # feedback FBO (small: page IDs only need ~1 sample per tile on screen)
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

        # camera + input
        self.yaw, self.pitch, self.dist = 0.6, 0.9, 14.0
        self.target = np.array([0.0, 0.0, 0.0], np.float32)
        self.debug = 1 if args.debug_lod else 0
        self.freeze = 0
        self.manifest = load_manifest(args.pyramid, args.manifest)
        self.hover = None                  # (u0, v0, u1, v1, name)
        self.hi_style = args.hi_style
        self._t0 = time.time()
        self._drag = None
        self._press = None
        self._zoom_anim = None
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_key_callback(self.win, self.on_key)

    # ---- input ---------------------------------------------------------
    def on_mouse_button(self, win, button, action, mods):
        pos = glfw.get_cursor_pos(win)
        if action == glfw.PRESS:
            self._zoom_anim = None    # manual input interrupts the glide
            pan = (button == glfw.MOUSE_BUTTON_RIGHT
                   or mods & glfw.MOD_SHIFT)
            self._drag = ("pan" if pan else "orbit", pos)
            self._press = pos
        else:
            # a left release that barely moved is a click: zoom toward it
            if (button == glfw.MOUSE_BUTTON_LEFT and self._press is not None
                    and abs(pos[0] - self._press[0]) < 4
                    and abs(pos[1] - self._press[1]) < 4
                    and not mods & glfw.MOD_SHIFT):
                self.update_hover(*pos)
                if self.hover:
                    self.center_on(self.hover)
                else:
                    self.center_on_point(*pos)
            self._drag = None
            self._press = None

    def refresh_hover(self):
        """Recompute the hovered image from the last cursor position: the
        camera moves on its own (click-to-centre glides), so what sits
        under a stationary pointer changes without any mouse event."""
        if not self.manifest or not glfw.get_window_attrib(self.win,
                                                           glfw.HOVERED):
            return
        self.update_hover(*glfw.get_cursor_pos(self.win))

    def update_hover(self, cx, cy):
        hit = self.cursor_hit(cx, cy)
        if hit is None:
            self.hover = None
            return
        u = (hit[0] + self.half[0]) / (2 * self.half[0])
        v = (hit[2] + self.half[1]) / (2 * self.half[1])
        self.hover = rect_at_uv(self.manifest, u, v,
                                self.vt.virt_w, self.vt.virt_h)

    def upload_highlight(self, prog):
        r = self.hover
        glUniform4f(glGetUniformLocation(prog, "uHoverRect"),
                    *(r[:4] if r else (0.0, 0.0, -1.0, -1.0)))
        glUniform1i(glGetUniformLocation(prog, "uHiStyle"), self.hi_style)
        glUniform1f(glGetUniformLocation(prog, "uTime"),
                    time.time() - self._t0)
        fw, _ = glfw.get_framebuffer_size(self.win)
        ww, _ = glfw.get_window_size(self.win)
        glUniform1f(glGetUniformLocation(prog, "uHiScale"),
                    fw / ww if ww else 1.0)

    def center_on(self, rect, duration=0.45):
        """Glide so the hovered image is centered; zoom is unchanged."""
        cu, cv = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        self.glide_to((( cu - 0.5) * 2 * self.half[0], 0.0,
                       ( cv - 0.5) * 2 * self.half[1]), duration)

    def on_cursor(self, win, x, y):
        if not self._drag:
            self.update_hover(x, y)
            return
        mode, (px, py) = self._drag
        dx, dy = x - px, y - py
        self._drag = (mode, (x, y))
        if mode == "orbit":
            self.yaw -= dx * 0.005
            self.pitch = min(max(self.pitch + dy * 0.005, 0.05), 1.52)
        else:
            s = self.dist * 0.0012
            right = np.array([math.cos(self.yaw), 0, -math.sin(self.yaw)])
            fwd = np.array([math.sin(self.yaw), 0, math.cos(self.yaw)])
            self.target += (-dx * right - dy * fwd) * s  # image follows cursor
            self.target[0] = min(max(self.target[0], -self.half[0]),
                                 self.half[0])
            self.target[2] = min(max(self.target[2], -self.half[1]),
                                 self.half[1])

    def cursor_hit(self, cx, cy):
        """Unproject window coords onto the ground plane (y=0)."""
        w, h = glfw.get_window_size(self.win)
        if w == 0 or h == 0:
            return None
        inv = np.linalg.inv(self.mvp(w, h).astype(np.float64))
        ndc = (2 * cx / w - 1, 1 - 2 * cy / h)
        p0 = inv @ np.array([ndc[0], ndc[1], -1, 1.0])
        p1 = inv @ np.array([ndc[0], ndc[1], 1, 1.0])
        p0, p1 = p0[:3] / p0[3], p1[:3] / p1[3]
        d = p1 - p0
        if abs(d[1]) < 1e-9:
            return None
        t = -p0[1] / d[1]
        if t <= 0:
            return None
        hit = p0 + t * d
        if abs(hit[0]) > self.half[0] or abs(hit[2]) > self.half[1]:
            return None
        return hit.astype(np.float32)

    def center_on_point(self, cx, cy, duration=0.4):
        """Glide the view to center the clicked point; zoom is unchanged."""
        hit = self.cursor_hit(cx, cy)
        if hit is not None:
            self.glide_to(hit, duration)

    def glide_to(self, end_target, duration):
        """Eased glide of the orbit target, holding the camera distance."""
        end = np.array(end_target, np.float32).copy()
        end[1] = 0.0
        end[0] = min(max(end[0], -self.half[0]), self.half[0])
        end[2] = min(max(end[2], -self.half[1]), self.half[1])
        self._zoom_anim = {
            "t": time.time(), "dur": duration,
            "target0": self.target.copy(), "target1": end,
            "dist0": self.dist, "dist1": self.dist,
        }

    def update_zoom_anim(self):
        a = self._zoom_anim
        if a is None:
            return
        u = min((time.time() - a["t"]) / a["dur"], 1.0)
        e = u * u * (3 - 2 * u)              # smoothstep ease-in/out
        self.target = a["target0"] + (a["target1"] - a["target0"]) * e
        self.dist = a["dist0"] + (a["dist1"] - a["dist0"]) * e
        if u >= 1.0:
            self._zoom_anim = None

    def on_scroll(self, win, dx, dy):
        self._zoom_anim = None
        self.dist = min(max(self.dist * (0.92 ** dy), 0.05), 60.0)

    def on_key(self, win, key, sc, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_L:
            self.debug ^= 1
        elif key == glfw.KEY_F:
            self.toggle_freeze()
        elif key == glfw.KEY_H:
            self.hi_style = (self.hi_style + 1) % 6
            print(f"highlight style {self.hi_style}: {HI_STYLES[self.hi_style]}")
        elif key == glfw.KEY_R:
            self._zoom_anim = None
            self.yaw, self.pitch, self.dist = 0.6, 0.9, 14.0
            self.target[:] = 0

    # ---- frame ---------------------------------------------------------
    def eye(self):
        return self.target + self.dist * np.array([
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
            math.cos(self.pitch) * math.cos(self.yaw)], np.float32)

    def mvp(self, w, h):
        view = look_at(self.eye(), self.target,
                       np.array([0, 1, 0], np.float32))
        proj = perspective(55.0, w / h, 0.02, 400.0)
        return proj @ view

    def toggle_freeze(self):
        self.freeze ^= 1
        if self.freeze:                    # pin the frustum that chose
            w, h = glfw.get_framebuffer_size(self.win)   # the working set
            self.frustum.capture(self.mvp(w, h), self.eye(), 0.02,
                                 max(2.5 * self.dist, 1.0))
            self.vt.evict_unused()         # keep only this view's pages

    def draw(self, prog, mvp):
        self.vt.bind(prog)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_TRUE, mvp)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

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
        while not glfw.window_should_close(self.win):
            if self.args.frames is not None:      # scripted orbit + zoom-in
                if frame < self.args.frames:
                    k = frame / max(self.args.frames - 1, 1)
                    self.yaw = 0.5 + 2.0 * k
                    self.pitch = 0.25 + 0.9 * k
                    self.dist = 18.0 * (self.args.end_dist / 18.0) ** k
                    u, v = (float(s) for s in self.args.end_uv.split(","))
                    self.target[:] = ((u - 0.5) * 2 * self.half[0] * k, 0,
                                      (v - 0.5) * 2 * self.half[1] * k)
                elif self.args.freeze_test:
                    # freeze at the end pose, then pull the camera back
                    # so the screenshot shows the pinned frustum + LOD
                    extra = frame - self.args.frames
                    if extra == 60:        # let streaming settle first
                        self.toggle_freeze()
                    elif extra > 60:
                        self.dist = min(self.dist * 1.015, 60.0)
                        self.yaw += 0.008
                        self.pitch = min(self.pitch + 0.002, 1.5)
                    if extra > 200:
                        break
                else:
                    # hold the pose until streaming settles so the
                    # screenshot shows the converged result
                    extra = frame - self.args.frames
                    if self.args.click_test and extra in (20, 45, 70):
                        w, h = glfw.get_window_size(self.win)
                        self.center_on_point(w / 2, h / 2)
                        if self._zoom_anim:
                            t = self._zoom_anim["target1"]
                            print(f"click -> center ({t[0]:.2f}, {t[2]:.2f})")
                    min_extra = 90 if self.args.click_test else 0
                    settled = (extra > min_extra
                               and self._zoom_anim is None
                               and not self.vt.pending
                               and self.vt.ready_q.empty())
                    if settled or extra > 600:
                        break
            self.update_zoom_anim()
            if self.args.frames is None:
                self.refresh_hover()
            w, h = glfw.get_framebuffer_size(self.win)
            mvp = self.mvp(w, h)

            if not self.freeze:
                self.feedback_pass(mvp, w)
                self.vt.pump_uploads()

            glViewport(0, 0, w, h)
            glClearColor(0.08, 0.09, 0.11, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.prog_main)
            glUniform1i(glGetUniformLocation(self.prog_main, "uDebug"),
                        self.debug)
            self.upload_highlight(self.prog_main)
            self.draw(self.prog_main, mvp)
            if self.freeze:
                self.frustum.draw(mvp)

            glfw.swap_buffers(self.win)
            glfw.poll_events()
            frame += 1
            fps_n += 1
            if time.time() - fps_t > 0.5:
                fps = fps_n / (time.time() - fps_t)
                glfw.set_window_title(
                    self.win,
                    f"virtual texture viewer | {fps:5.1f} fps | resident "
                    f"{len(self.vt.residency)}/{self.vt.n_slots} | pending "
                    f"{len(self.vt.pending)} | loaded {self.vt.loads_done}"
                    f"{' | FROZEN' if self.freeze else ''}")
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
    ap.add_argument("pyramid_pos", nargs="?", metavar="PYRAMID",
                    help="pyramid directory (same as --pyramid)")
    ap.add_argument("--pyramid", default="pics/test_image_16k_pyramid")
    ap.add_argument("--frames", type=int, default=None,
                    help="run a scripted N-frame orbit and exit (self-test)")
    ap.add_argument("--screenshot", default=None,
                    help="save the final frame to this PNG")
    ap.add_argument("--end-dist", type=float, default=0.12,
                    help="final camera distance for the scripted orbit")
    ap.add_argument("--end-uv", default="0.625,0.5625",
                    help="final scripted-orbit target as u,v in [0,1]")
    ap.add_argument("--manifest", default=None,
                    help="layout.json for hover/click "
                         "(default: derived from the pyramid name)")
    ap.add_argument("--hi-style", type=int, default=2,
                    help="highlight style 0-5 (see H key)")
    ap.add_argument("--debug-lod", action="store_true",
                    help="start with the LOD debug overlay enabled")
    ap.add_argument("--click-test", action="store_true",
                    help="in scripted mode, fire 3 click-zooms at the "
                         "window center during the settle phase")
    ap.add_argument("--freeze-test", action="store_true",
                    help="in scripted mode, freeze at the end pose then "
                         "pull back to show the pinned frustum")
    args = ap.parse_args()
    if args.pyramid_pos:                   # allow a bare positional path
        args.pyramid = args.pyramid_pos
    check_pyramid(args.pyramid)            # before opening a window
    Viewer(args).run()


if __name__ == "__main__":
    main()
