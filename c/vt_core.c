/* vt_core.c - shared virtual-texturing core (see vt_core.h) */
#include "vt_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ---------------------------------------------------------------- misc */

const char *vt_argv0 = "./vt_viewer";   /* set from main() for error hints */
const char *vt_tile_url = NULL;          /* web: set from main() */

double now_seconds(void)
{
    return (double)SDL_GetPerformanceCounter()
         / (double)SDL_GetPerformanceFrequency();
}

float smoothstep01(float u)
{
    if (u < 0) u = 0;
    if (u > 1) u = 1;
    return u * u * (3.0f - 2.0f * u);
}

/* ------------------------------------------------------------- shaders */

static const char *VS_SRC =
    "attribute vec3 aPos;\n"
    "attribute vec2 aUV;\n"
    "uniform mat4 uMVP;\n"
    "varying vec2 vUV;\n"
    "void main(){ vUV = aUV; gl_Position = uMVP*vec4(aPos,1.0); }\n";

#define VT_GLSL_COMMON \
    "#extension GL_OES_standard_derivatives : enable\n" \
    "precision highp float;\n" \
    "varying vec2 vUV;\n" \
    "uniform sampler2D uPageTable;\n" \
    "uniform sampler2D uAtlas;\n" \
    "uniform vec2 uVirtDim;\n" \
    "uniform vec2 uPagesF;\n" \
    "uniform vec2 uTableDim;\n" \
    "uniform float uMaxLod;\n" \
    "uniform float uAtlasSize;\n" \
    "uniform float uSlot;\n" \
    "uniform float uTile;\n" \
    "uniform float uBorder;\n" \
    "uniform vec4 uLevelRect[16];\n" \
    "float vtLod(vec2 uv, float bias){\n" \
    "  vec2 dx = dFdx(uv)*uVirtDim;\n" \
    "  vec2 dy = dFdy(uv)*uVirtDim;\n" \
    "  float rho = max(length(dx), length(dy));\n" \
    "  return clamp(log2(max(rho,1e-6))+bias, 0.0, uMaxLod);\n" \
    "}\n" \
    "vec2 vtPageCoord(vec2 uv, float level){\n" \
    "  vec2 pages = uPagesF/exp2(level);\n" \
    "  vec2 n = ceil(pages-1e-5);\n" \
    "  return clamp(floor(uv*pages), vec2(0.0), n-vec2(1.0));\n" \
    "}\n" \
    "vec4 vtEntry(vec2 pc, float level){\n" \
    "  vec4 rect = vec4(0.0);\n" \
    "  for (int i=0;i<16;i++){ if (abs(float(i)-level)<0.5) rect = uLevelRect[i]; }\n" \
    "  vec2 tuv = (rect.xy + pc + vec2(0.5))/uTableDim;\n" \
    "  return texture2D(uPageTable, tuv);\n" \
    "}\n" \
    "vec4 vtSampleLevel(vec2 uv, float level){\n" \
    "  vec3 e = floor(vtEntry(vtPageCoord(uv,level), level).rgb*255.0+0.5);\n" \
    "  vec2 inTile = fract(uv*uPagesF/exp2(e.z));\n" \
    "  vec2 auv = (e.xy*uSlot + vec2(uBorder) + inTile*uTile)/uAtlasSize;\n" \
    "  return texture2D(uAtlas, auv);\n" \
    "}\n"


#define VT_GLSL_HIGHLIGHT \
    "uniform vec4 uHoverRect;\n"      /* (u0,v0,u1,v1); off when z <= x */ \
    "uniform int uHiStyle;\n" \
    "uniform float uTime;\n" \
    "uniform float uHiScale;\n"       /* framebuffer px per logical point */ \
    /* Screen-space outline of the hovered image: the rect is an SDF in UV, \
       divided by its screen-space gradient to get pixels, so the line keeps \
       a constant width at any zoom, on flat or curved geometry alike. */ \
    "vec3 vtHighlight(vec3 col, vec2 uv) {\n" \
    "  if (uHiStyle == 0 || uHoverRect.z <= uHoverRect.x) return col;\n" \
    "  vec2 ctr = (uHoverRect.xy + uHoverRect.zw) * 0.5;\n" \
    "  vec2 hlf = (uHoverRect.zw - uHoverRect.xy) * 0.5;\n" \
    "  vec2 d = abs(uv - ctr) - hlf;\n" \
    "  float sd = max(d.x, d.y);\n" \
    "  float spx = length(vec2(dFdx(sd), dFdy(sd)));\n" \
    "  if (spx <= 0.0) return col;\n" \
    "  float a = abs(sd / spx);\n" \
    "  if (uHiStyle == 4 && sd > 0.0) col *= 0.42;\n" \
    "  float k = max(uHiScale, 1.0);\n" \
    "  float core = 1.0 - smoothstep(0.6*k, 1.6*k, a);\n" \
    "  float halo = 1.0 - smoothstep(1.6*k, 3.4*k, a);\n" \
    "  if (uHiStyle == 1) return mix(col, vec3(1.0,0.85,0.0), core);\n" \
    "  if (uHiStyle == 5) {\n" \
    "    float run = (abs(d.x) > abs(d.y)) ? gl_FragCoord.y : gl_FragCoord.x;\n" \
    "    vec3 ant = fract((run - uTime*40.0*k)/(14.0*k)) < 0.5\n" \
    "             ? vec3(1.0) : vec3(0.05);\n" \
    "    col = mix(col, vec3(0.0), halo * 0.5);\n" \
    "    return mix(col, ant, core);\n" \
    "  }\n" \
    "  vec3 hi = vec3(1.0, 0.85, 0.0);\n" \
    "  if (uHiStyle == 3 || uHiStyle == 4)\n" \
    "    hi = dot(col, vec3(0.299,0.587,0.114)) > 0.5 ? vec3(0.0) : vec3(1.0);\n" \
    "  col = mix(col, vec3(0.0), halo * 0.55);\n" \
    "  return mix(col, hi, core);\n" \
    "}\n"

static const char *FS_MAIN_SRC = VT_GLSL_COMMON VT_GLSL_HIGHLIGHT
    "uniform int uDebug;\n"
    "void main(){\n"
    "  vec2 uv = clamp(vUV, 0.0, 0.9999999);\n"
    "  float lod = vtLod(uv, 0.0);\n"
    "  float l0 = floor(lod);\n"
    "  float l1 = min(l0+1.0, uMaxLod);\n"
    "  vec4 c = mix(vtSampleLevel(uv,l0), vtSampleLevel(uv,l1), fract(lod));\n"
    "  if (uDebug==1){\n"
    "    vec3 lc = 0.5+0.5*cos(6.2832*(lod/7.0+vec3(0.0,0.33,0.67)));\n"
    "    c.rgb = mix(c.rgb, lc, 0.45);\n"
    "  }\n"
    "  gl_FragColor = vec4(vtHighlight(c.rgb, uv), 1.0);\n"
    "}\n";

/* feedback: pack (pageX:12, pageY:12, level:4) with float arithmetic;
   alpha >= 240 marks a valid sample (clear alpha is 0) */
static const char *FS_FB_SRC = VT_GLSL_COMMON
    "uniform float uLodBias;\n"
    "void main(){\n"
    "  vec2 uv = clamp(vUV, 0.0, 0.9999999);\n"
    "  float l0 = floor(vtLod(uv, uLodBias));\n"
    "  vec2 pc = vtPageCoord(uv, l0);\n"
    "  gl_FragColor = vec4(mod(pc.x,256.0), mod(pc.y,256.0),\n"
    "                      floor(pc.x/256.0)+16.0*floor(pc.y/256.0),\n"
    "                      240.0+l0)/255.0;\n"
    "}\n";

const char *vt_fs_main_src = NULL;        /* set in vt_init */
const char *vt_fs_feedback_src = NULL;

static GLuint compile_shader(GLenum kind, const char *src)
{
    GLuint sh = glCreateShader(kind);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return sh;
}

GLuint vt_compile_program(const char *vs, const char *fs)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, compile_shader(GL_VERTEX_SHADER, vs));
    glAttachShader(prog, compile_shader(GL_FRAGMENT_SHADER, fs));
    glBindAttribLocation(prog, 0, "aPos");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof log, NULL, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
        exit(1);
    }
    return prog;
}

/* ------------------------------------------------- frustum wireframe */

static const char *LINE_VS_SRC =
    "attribute vec3 aPos;\n"
    "uniform mat4 uMVP;\n"
    "void main(){ gl_Position = uMVP*vec4(aPos,1.0); }\n";

static const char *LINE_FS_SRC =
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
    "void main(){ gl_FragColor = uColor; }\n";

void vt_lines_init(VtLines *fl)
{
    fl->prog = vt_compile_program(LINE_VS_SRC, LINE_FS_SRC);
    glGenBuffers(1, &fl->vbo);
    fl->valid = false;
}

void vt_lines_capture(VtLines *fl, const float *mvp, const float eye[3],
                      float near_dist, float far_dist)
{
    /* near rect from the inverse MVP; far rect extends the corner rays
       to far_dist (a display depth, so the wireframe stays legible) */
    float inv[16];
    if (!mat4_invert(inv, mvp))
        return;
    static const float ndc[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    float pts[8][3];
    float s = far_dist / near_dist;
    for (int i = 0; i < 4; i++) {
        float v[4] = {ndc[i][0], ndc[i][1], -1, 1}, o[4];
        for (int r = 0; r < 4; r++)
            o[r] = inv[r*4+0]*v[0] + inv[r*4+1]*v[1]
                 + inv[r*4+2]*v[2] + inv[r*4+3]*v[3];
        for (int k = 0; k < 3; k++) {
            pts[i][k] = o[k] / o[3];
            pts[i+4][k] = eye[k] + (pts[i][k] - eye[k]) * s;
        }
    }
    static const int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},
                                     {4,5},{5,6},{6,7},{7,4},
                                     {0,4},{1,5},{2,6},{3,7}};
    float verts[24][3];
    for (int e = 0; e < 12; e++) {
        memcpy(verts[e*2],     pts[edges[e][0]], 3 * sizeof(float));
        memcpy(verts[e*2 + 1], pts[edges[e][1]], 3 * sizeof(float));
    }
    glBindBuffer(GL_ARRAY_BUFFER, fl->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_DYNAMIC_DRAW);
    fl->valid = true;
}

void vt_lines_draw(VtLines *fl, const float *mvp_now)
{
    if (!fl->valid)
        return;
    glUseProgram(fl->prog);
    mat4_upload(glGetUniformLocation(fl->prog, "uMVP"), mvp_now);
    glUniform4f(glGetUniformLocation(fl->prog, "uColor"),
                1.0f, 0.9f, 0.0f, 1.0f);
    glBindBuffer(GL_ARRAY_BUFFER, fl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void *)0);
    glDisableVertexAttribArray(1);
    glDrawArrays(GL_LINES, 0, 24);
}

/* --------------------------------------------------------- meta parsing */

static char *read_file(const char *path, long *size_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    fread(buf, 1, n, f);
    buf[n] = 0;
    fclose(f);
    if (size_out) *size_out = n;
    return buf;
}

static int json_int(const char *s, const char *key, int def)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(s, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    return p ? atoi(p + 1) : def;
}

static void json_str(const char *s, const char *key, char *out, int cap)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(s, pat);
    out[0] = 0;
    if (!p) return;
    p = strchr(p + strlen(pat), ':');
    if (!p) return;
    p = strchr(p, '"');
    if (!p) return;
    p++;
    const char *e = strchr(p, '"');
    if (!e || e - p >= cap) return;
    memcpy(out, p, e - p);
    out[e - p] = 0;
}

/* ------------------------------------------------------------ manifest */

bool vt_manifest_load(VtManifest *m, const char *pyramid_dir,
                      const char *explicit_path)
{
    memset(m, 0, sizeof *m);
    char path[1200];
    if (explicit_path) {
        snprintf(path, sizeof path, "%s", explicit_path);
    } else {
        const char *suffix = "_pyramid";
        size_t n = strlen(pyramid_dir), sn = strlen(suffix);
        if (n <= sn || strcmp(pyramid_dir + n - sn, suffix) != 0)
            return false;
        snprintf(path, sizeof path, "%.*s_layout.json",
                 (int)(n - sn), pyramid_dir);
    }
    char *buf = read_file(path, NULL);
    if (!buf)
        return false;

    int cap = 256;
    m->rects = malloc(cap * sizeof(VtRect));
    const char *p = buf;
    while ((p = strstr(p, "\"path\"")) != NULL) {
        const char *q = strchr(p + 6, ':');
        if (!q) break;
        q = strchr(q, '"');
        if (!q) break;
        q++;
        const char *e = strchr(q, '"');
        if (!e) break;
        if (m->n == cap) {
            cap *= 2;
            m->rects = realloc(m->rects, cap * sizeof(VtRect));
        }
        VtRect *r = &m->rects[m->n];
        const char *base = q;                 /* keep only the basename */
        for (const char *c = q; c < e; c++)
            if (*c == '/') base = c + 1;
        int len = (int)(e - base);
        if (len >= (int)sizeof r->name) len = sizeof r->name - 1;
        memcpy(r->name, base, len);
        r->name[len] = 0;
        const char *kx = strstr(e, "\"x\""), *ky = strstr(e, "\"y\"");
        const char *kw = strstr(e, "\"w\""), *kh = strstr(e, "\"h\"");
        if (!kx || !ky || !kw || !kh) break;
        r->x = atoi(strchr(kx, ':') + 1);
        r->y = atoi(strchr(ky, ':') + 1);
        r->w = atoi(strchr(kw, ':') + 1);
        r->h = atoi(strchr(kh, ':') + 1);
        m->n++;
        p = kh;
    }
    free(buf);
    printf("manifest: %s (%d images)\n", path, m->n);
    return m->n > 0;
}

int vt_manifest_rect_at(const VtManifest *m, float u, float v,
                        int virt_w, int virt_h, float out_uv[4])
{
    if (!m || !m->rects)
        return -1;
    float px = u * virt_w, py = v * virt_h;
    for (int i = 0; i < m->n; i++) {
        const VtRect *r = &m->rects[i];
        if (px >= r->x && px < r->x + r->w
                && py >= r->y && py < r->y + r->h) {
            out_uv[0] = (float)r->x / virt_w;
            out_uv[1] = (float)r->y / virt_h;
            out_uv[2] = (float)(r->x + r->w) / virt_w;
            out_uv[3] = (float)(r->y + r->h) / virt_h;
            return i;
        }
    }
    return -1;
}

void vt_highlight_uniforms(GLuint prog, const float rect_uv[4], int style,
                           float t, float dpi_scale)
{
    static const float none[4] = {0.0f, 0.0f, -1.0f, -1.0f};
    const float *r = rect_uv ? rect_uv : none;
    glUniform4f(glGetUniformLocation(prog, "uHoverRect"),
                r[0], r[1], r[2], r[3]);
    glUniform1i(glGetUniformLocation(prog, "uHiStyle"), style);
    glUniform1f(glGetUniformLocation(prog, "uTime"), t);
    glUniform1f(glGetUniformLocation(prog, "uHiScale"), dpi_scale);
}

/* --------------------------------------------------------------- loader */

static void tile_path(VtSystem *vt, VtPage p, char *out, int cap)
{
    snprintf(out, cap, "%s/L%d/%d_%d.%s", vt_tile_url ? vt_tile_url : vt->dir,
             p.level, p.ty, p.tx, vt->format);
}

static void push_ready(VtSystem *vt, VtPage p, unsigned char *pix, int bitmap)
{
    SDL_LockMutex(vt->ready_mtx);
    if (vt->ready_count == vt->ready_cap) {
        vt->ready_cap *= 2;
        vt->ready_q = realloc(vt->ready_q,
                              vt->ready_cap * sizeof *vt->ready_q);
    }
    vt->ready_q[vt->ready_count].page = p;
    vt->ready_q[vt->ready_count].pixels = pix;
    vt->ready_q[vt->ready_count].bitmap = bitmap;
    vt->ready_count++;
    SDL_UnlockMutex(vt->ready_mtx);
}

#ifdef __EMSCRIPTEN__
/* The browser fetches and decodes -- createImageBitmap runs off the main
   thread -- and the bitmap waits on the JS side under a small integer
   handle until upload_tile hands it to texSubImage2D, so tile pixels never
   pass through wasm memory at all. */
EM_JS(void, js_fetch_tile, (const char *url, int key, int size), {
    const u = UTF8ToString(url);
    const bm = Module.vtBitmaps
            || (Module.vtBitmaps = { next: 1, map: new Map() });
    fetch(u)
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status);
                     return r.blob(); })
        .then(b => createImageBitmap(b, { premultiplyAlpha: 'none',
                                          colorSpaceConversion: 'none' }))
        .then(img => {
            if (img.width != size || img.height != size) {
                img.close();
                throw new Error('bad size');
            }
            const h = bm.next++;
            bm.map.set(h, img);
            _vt_web_tile_done(key, h);
        })
        .catch(e => { console.warn('tile load failed: ' + u, e);
                      _vt_web_tile_done(key, 0); });
});

EM_JS(void, js_upload_bitmap, (int h, GLuint tex, int x, int y), {
    const img = Module.vtBitmaps.map.get(h);
    Module.vtBitmaps.map.delete(h);
    GLctx.bindTexture(GLctx.TEXTURE_2D, GL.textures[tex]);
    GLctx.texSubImage2D(GLctx.TEXTURE_2D, 0, x, y, GLctx.RGB,
                        GLctx.UNSIGNED_BYTE, img);
    img.close();
});

EM_JS(void, js_drop_bitmap, (int h), {
    Module.vtBitmaps.map.get(h).close();
    Module.vtBitmaps.map.delete(h);
});

static VtSystem *web_vt;     /* the callback's way back in */

EMSCRIPTEN_KEEPALIVE void vt_web_tile_done(int key, int bitmap)
{
    VtPage p = {key >> 24, key & 4095, (key >> 12) & 4095};
    web_vt->inflight--;
    push_ready(web_vt, p, NULL, bitmap);
}

/* Keep up to VT_MAX_INFLIGHT fetches going, coarsest first. Network
   latency is far longer than a disk read, so a quick fling can queue
   hundreds of pages that are gone from view before their turn comes;
   those are dropped (and simply re-requested if they come back). */
static void web_issue_fetches(VtSystem *vt)
{
    while (vt->inflight < VT_MAX_INFLIGHT && vt->req_count > 0) {
        int best = -1;
        for (int i = 0; i < vt->req_count; i++) {
            VtPage p = vt->req_q[i];
            int idx = p.ty * vt->nx[p.level] + p.tx;
            if (p.level != vt->levels - 1
                    && vt->frame - vt->wanted[p.level][idx] > VT_STALE_FRAMES) {
                vt->pending[p.level][idx] = 0;
                vt->n_pending--;
                vt->req_q[i--] = vt->req_q[--vt->req_count];
                continue;
            }
            if (best < 0 || p.level > vt->req_q[best].level)
                best = i;
        }
        if (best < 0)
            break;
        VtPage p = vt->req_q[best];
        vt->req_q[best] = vt->req_q[--vt->req_count];
        char url[1200];
        tile_path(vt, p, url, sizeof url);
        js_fetch_tile(url, (p.level << 24) | (p.ty << 12) | p.tx,
                      vt->slot_px);
        vt->inflight++;
    }
}
#else
static int loader_main(void *arg)
{
    VtSystem *vt = arg;
    for (;;) {
        SDL_LockMutex(vt->req_mtx);
        while (vt->req_count == 0 && !vt->quit)
            SDL_CondWait(vt->req_cond, vt->req_mtx);
        if (vt->quit) {
            SDL_UnlockMutex(vt->req_mtx);
            return 0;
        }
        int best = 0;                       /* coarsest level first */
        for (int i = 1; i < vt->req_count; i++)
            if (vt->req_q[i].level > vt->req_q[best].level)
                best = i;
        VtPage p = vt->req_q[best];
        vt->req_q[best] = vt->req_q[--vt->req_count];
        SDL_UnlockMutex(vt->req_mtx);

        char path[1200];
        tile_path(vt, p, path, sizeof path);
        int w, h, n;
        unsigned char *pix = stbi_load(path, &w, &h, &n, 3);
        if (pix && (w != vt->slot_px || h != vt->slot_px)) {
            stbi_image_free(pix);
            pix = NULL;
        }
        if (!pix)
            fprintf(stderr, "tile load failed: %s\n", path);
        push_ready(vt, p, pix, 0);
    }
}
#endif

/* ------------------------------------------------------------ residency */

static void enqueue_request(VtSystem *vt, VtPage p)
{
    SDL_LockMutex(vt->req_mtx);
    if (vt->req_count == vt->req_cap) {
        vt->req_cap *= 2;
        vt->req_q = realloc(vt->req_q, vt->req_cap * sizeof *vt->req_q);
    }
    vt->req_q[vt->req_count++] = p;
    SDL_CondSignal(vt->req_cond);
    SDL_UnlockMutex(vt->req_mtx);
}

static int evict_slot(VtSystem *vt)
{
    int best = -1;
    unsigned best_use = 0;
    for (int s = 0; s < vt->n_slots; s++) {
        VtPage p = vt->slot_page[s];
        if (p.level < 0 || p.level == vt->levels - 1)
            continue;                       /* free, or the pinned root */
        if (vt->slot_lastuse[s] >= vt->frame)
            continue;                       /* in this frame's working set */
        if (best < 0 || vt->slot_lastuse[s] < best_use) {
            best = s;
            best_use = vt->slot_lastuse[s];
        }
    }
    if (best >= 0) {
        VtPage p = vt->slot_page[best];
        vt->own[p.level][p.ty * vt->nx[p.level] + p.tx] = -1;
        vt->slot_page[best].level = -1;
        vt->dirty = true;
    }
    return best;
}

/* On the web, a true return means the bitmap has been consumed. */
static bool upload_tile(VtSystem *vt, VtPage p, const unsigned char *pix,
                        int bitmap)
{
    if (vt->own[p.level][p.ty * vt->nx[p.level] + p.tx] >= 0) {
#ifdef __EMSCRIPTEN__
        js_drop_bitmap(bitmap);
#endif
        return true;                        /* already resident */
    }
    int slot = -1;
    for (int s = 0; s < vt->n_slots; s++)
        if (vt->slot_page[s].level < 0) { slot = s; break; }
    if (slot < 0)
        slot = evict_slot(vt);
    if (slot < 0)
        return false;                       /* everything in use right now */
    int sx = (slot % vt->slots_per_row) * vt->slot_px;
    int sy = (slot / vt->slots_per_row) * vt->slot_px;
#ifdef __EMSCRIPTEN__
    (void)pix;
    js_upload_bitmap(bitmap, vt->atlas_tex, sx, sy);
#else
    (void)bitmap;
    glBindTexture(GL_TEXTURE_2D, vt->atlas_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, sx, sy, vt->slot_px, vt->slot_px,
                    GL_RGB, GL_UNSIGNED_BYTE, pix);
#endif
    vt->own[p.level][p.ty * vt->nx[p.level] + p.tx] = slot;
    vt->slot_page[slot] = p;
    vt->slot_lastuse[slot] = vt->frame;
    vt->dirty = true;
    vt->loads_done++;
    return true;
}

static void rebuild_page_table(VtSystem *vt)
{
    unsigned char *t = vt->table_staging;
    int tw = vt->table_w;
    for (int l = vt->levels - 1; l >= 0; l--) {
        int nx = vt->nx[l], ny = vt->ny[l];
        int base = vt->table_off_y[l];
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                unsigned char *e = t + ((base + y) * tw + x) * 4;
                int slot = vt->own[l][y * nx + x];
                if (slot >= 0) {
                    e[0] = slot % vt->slots_per_row;
                    e[1] = slot / vt->slots_per_row;
                    e[2] = l;
                } else if (l == vt->levels - 1) {  /* web: root not in yet */
                    e[0] = e[1] = e[2] = 0;
                } else {                    /* inherit finest ancestor */
                    int pb = vt->table_off_y[l + 1];
                    unsigned char *pe =
                        t + ((pb + y / 2) * tw + x / 2) * 4;
                    e[0] = pe[0];
                    e[1] = pe[1];
                    e[2] = pe[2];
                }
                e[3] = 255;
            }
        }
    }
    glBindTexture(GL_TEXTURE_2D, vt->table_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vt->table_w, vt->table_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, t);
}

void vt_pump_uploads(VtSystem *vt)
{
#ifdef __EMSCRIPTEN__
    web_issue_fetches(vt);
#endif
    for (int i = 0; i < VT_MAX_UPLOADS_PER_FRAME; i++) {
        SDL_LockMutex(vt->ready_mtx);
        if (vt->ready_count == 0) {
            SDL_UnlockMutex(vt->ready_mtx);
            break;
        }
        struct VtReady r = vt->ready_q[0];
        memmove(vt->ready_q, vt->ready_q + 1,
                (vt->ready_count - 1) * sizeof *vt->ready_q);
        vt->ready_count--;
        SDL_UnlockMutex(vt->ready_mtx);

        int idx = r.page.ty * vt->nx[r.page.level] + r.page.tx;
        if (!r.pixels && !r.bitmap) {       /* load failed: drop */
            vt->pending[r.page.level][idx] = 0;
            vt->n_pending--;
            continue;
        }
        if (upload_tile(vt, r.page, r.pixels, r.bitmap)) {
            vt->pending[r.page.level][idx] = 0;
            vt->n_pending--;
            stbi_image_free(r.pixels);
        } else {                            /* atlas full: retry later */
            SDL_LockMutex(vt->ready_mtx);
            if (vt->ready_count == vt->ready_cap) {
                vt->ready_cap *= 2;
                vt->ready_q = realloc(vt->ready_q,
                                      vt->ready_cap * sizeof *vt->ready_q);
            }
            memmove(vt->ready_q + 1, vt->ready_q,
                    vt->ready_count * sizeof *vt->ready_q);
            vt->ready_q[0] = r;
            vt->ready_count++;
            SDL_UnlockMutex(vt->ready_mtx);
            break;
        }
    }
    if (vt->dirty) {
        rebuild_page_table(vt);
        vt->dirty = false;
    }
}

static int cmp_u32(const void *a, const void *b)
{
    unsigned x = *(const unsigned *)a, y = *(const unsigned *)b;
    return x < y ? -1 : x > y;
}

void vt_request_from_feedback(VtSystem *vt, const unsigned char *rgba, int n)
{
    vt->frame++;
    static unsigned *codes = NULL;
    static int cap = 0;
    if (cap < n) {
        cap = n;
        codes = realloc(codes, cap * sizeof *codes);
    }
    int m = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char *p = rgba + i * 4;
        if (p[3] < 240)
            continue;
        unsigned x = p[0] | ((unsigned)(p[2] & 15) << 8);
        unsigned y = p[1] | ((unsigned)(p[2] >> 4) << 8);
        unsigned l = p[3] - 240;
        codes[m++] = x | (y << 12) | (l << 24);
    }
    qsort(codes, m, sizeof *codes, cmp_u32);
    for (int i = 0; i < m; i++) {
        if (i > 0 && codes[i] == codes[i - 1])
            continue;
        int l = codes[i] >> 24;
        int tx = codes[i] & 4095;
        int ty = (codes[i] >> 12) & 4095;
        while (l < vt->levels) {            /* include ancestors */
            int idx = ty * vt->nx[l] + tx;
            int slot = vt->own[l][idx];
            vt->wanted[l][idx] = vt->frame;
            if (slot >= 0) {
                vt->slot_lastuse[slot] = vt->frame;
            } else if (!vt->pending[l][idx]) {
                vt->pending[l][idx] = 1;
                vt->n_pending++;
                enqueue_request(vt, (VtPage){l, tx, ty});
            }
            l++;
            tx >>= 1;
            ty >>= 1;
        }
    }
}

/* ----------------------------------------------------------- init / bind */

bool vt_check_pyramid(const char *pyramid_dir)
{
    char meta_path[1100];
    snprintf(meta_path, sizeof meta_path, "%s/meta.json", pyramid_dir);
    FILE *f = fopen(meta_path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    fprintf(stderr,
        "error: no tile pyramid at '%s' (no meta.json)\n\n"
        "Build one first:\n\n"
        "    ./gen_test_image_16k.py   # synthetic test image\n"
        "    ./build_pyramid.py        # -> pics/test_image_16k_pyramid/\n\n"
        "...or from your own photos:\n\n"
        "    ./layout_mosaic.py pics/some_tree\n"
        "    ./build_pyramid.py pics/some_tree_mosaic.npy\n\n"
        "then point the viewer at the pyramid directory:\n\n"
        "    %s <pyramid_dir>\n", pyramid_dir, vt_argv0);
    return false;
}

bool vt_init(VtSystem *vt, const char *pyramid_dir)
{
    memset(vt, 0, sizeof *vt);
    snprintf(vt->dir, sizeof vt->dir, "%s", pyramid_dir);

    char meta_path[1100];
    snprintf(meta_path, sizeof meta_path, "%s/meta.json", pyramid_dir);
    char *meta = read_file(meta_path, NULL);
    if (!meta) {
        vt_check_pyramid(pyramid_dir);
        return false;
    }
    int size = json_int(meta, "image_size", 0);
    vt->virt_w = json_int(meta, "image_width", size);
    vt->virt_h = json_int(meta, "image_height", size);
    vt->tile = json_int(meta, "tile_size", 256);
    vt->border = json_int(meta, "border", 2);
    vt->levels = json_int(meta, "levels", 1);
    json_str(meta, "format", vt->format, sizeof vt->format);
    free(meta);
    if (!vt->virt_w || !vt->virt_h || vt->levels > VT_MAX_LEVELS) {
        fprintf(stderr, "bad meta.json\n");
        return false;
    }
    vt->slot_px = vt->tile + 2 * vt->border;
    vt->slots_per_row = VT_ATLAS_SIZE / vt->slot_px;
    vt->n_slots = vt->slots_per_row * vt->slots_per_row;

    int w = vt->virt_w, h = vt->virt_h, off = 0;
    for (int l = 0; l < vt->levels; l++) {
        vt->nx[l] = (w + vt->tile - 1) / vt->tile;
        vt->ny[l] = (h + vt->tile - 1) / vt->tile;
        vt->table_off_y[l] = off;
        off += vt->ny[l];
        vt->own[l] = malloc(vt->nx[l] * vt->ny[l] * sizeof(int));
        vt->pending[l] = calloc(vt->nx[l], vt->ny[l]);
        vt->wanted[l] = calloc(vt->nx[l] * vt->ny[l], sizeof(unsigned));
        for (int i = 0; i < vt->nx[l] * vt->ny[l]; i++)
            vt->own[l][i] = -1;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    vt->table_w = vt->nx[0];
    vt->table_h = off;
    vt->table_staging = calloc(vt->table_w * vt->table_h, 4);

    vt->slot_page = malloc(vt->n_slots * sizeof(VtPage));
    vt->slot_lastuse = calloc(vt->n_slots, sizeof(unsigned));
    for (int s = 0; s < vt->n_slots; s++)
        vt->slot_page[s].level = -1;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenTextures(1, &vt->atlas_tex);
    glBindTexture(GL_TEXTURE_2D, vt->atlas_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VT_ATLAS_SIZE, VT_ATLAS_SIZE,
                 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glGenTextures(1, &vt->table_tex);
    glBindTexture(GL_TEXTURE_2D, vt->table_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vt->table_w, vt->table_h,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    vt_fs_main_src = FS_MAIN_SRC;
    vt_fs_feedback_src = FS_FB_SRC;
    vt->prog_main = vt_compile_program(VS_SRC, FS_MAIN_SRC);
    vt->prog_fb = vt_compile_program(VS_SRC, FS_FB_SRC);

    /* streaming */
    vt->req_cap = 256;
    vt->req_q = malloc(vt->req_cap * sizeof(VtPage));
    vt->ready_cap = 256;
    vt->ready_q = malloc(vt->ready_cap * sizeof *vt->ready_q);
    VtPage root = {vt->levels - 1, 0, 0};
#ifdef __EMSCRIPTEN__
    /* no thread (SDL's locks are no-ops here), and nothing may block, so
       the root is simply the first request; vt_root_resident() says when
       there is something to draw */
    web_vt = vt;
    vt->pending[root.level][0] = 1;
    vt->n_pending++;
    enqueue_request(vt, root);
    web_issue_fetches(vt);                  /* start now, not next frame */
    rebuild_page_table(vt);
#else
    vt->req_mtx = SDL_CreateMutex();
    vt->req_cond = SDL_CreateCond();
    vt->ready_mtx = SDL_CreateMutex();
    vt->loader = SDL_CreateThread(loader_main, "vt_loader", vt);

    /* the root tile must always be resident: load it synchronously */
    char path[1200];
    tile_path(vt, root, path, sizeof path);
    int tw, th, tn;
    unsigned char *pix = stbi_load(path, &tw, &th, &tn, 3);
    if (!pix) {
        fprintf(stderr, "cannot load root tile %s\n", path);
        return false;
    }
    upload_tile(vt, root, pix, 0);
    stbi_image_free(pix);
    rebuild_page_table(vt);
#endif
    vt->dirty = false;

    printf("vt: %dx%d, %d levels, %d slots, format %s\n",
           vt->virt_w, vt->virt_h, vt->levels, vt->n_slots, vt->format);
    return true;
}

void vt_evict_unused(VtSystem *vt)
{
    for (int s = 0; s < vt->n_slots; s++) {
        VtPage p = vt->slot_page[s];
        if (p.level < 0 || p.level == vt->levels - 1)
            continue;                       /* free, or the pinned root */
        if (vt->slot_lastuse[s] >= vt->frame)
            continue;                       /* in the current working set */
        vt->own[p.level][p.ty * vt->nx[p.level] + p.tx] = -1;
        vt->slot_page[s].level = -1;
        vt->dirty = true;
    }
    if (vt->dirty) {
        rebuild_page_table(vt);
        vt->dirty = false;
    }
}

bool vt_root_resident(VtSystem *vt)
{
    return vt->own[vt->levels - 1][0] >= 0;
}

bool vt_idle(VtSystem *vt)
{
    SDL_LockMutex(vt->ready_mtx);
    int ready = vt->ready_count;
    SDL_UnlockMutex(vt->ready_mtx);
    return vt->n_pending == 0 && ready == 0;
}

void vt_destroy(VtSystem *vt)
{
    if (!vt->loader)
        return;                             /* web: no thread to stop */
    SDL_LockMutex(vt->req_mtx);
    vt->quit = true;
    SDL_CondSignal(vt->req_cond);
    SDL_UnlockMutex(vt->req_mtx);
    SDL_WaitThread(vt->loader, NULL);
}

void vt_bind(VtSystem *vt, GLuint prog)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vt->table_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, vt->atlas_tex);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uPageTable"), 0);
    glUniform1i(glGetUniformLocation(prog, "uAtlas"), 1);
    glUniform2f(glGetUniformLocation(prog, "uVirtDim"),
                (float)vt->virt_w, (float)vt->virt_h);
    glUniform2f(glGetUniformLocation(prog, "uPagesF"),
                (float)vt->virt_w / vt->tile, (float)vt->virt_h / vt->tile);
    glUniform2f(glGetUniformLocation(prog, "uTableDim"),
                (float)vt->table_w, (float)vt->table_h);
    glUniform1f(glGetUniformLocation(prog, "uMaxLod"),
                (float)(vt->levels - 1));
    glUniform1f(glGetUniformLocation(prog, "uAtlasSize"), VT_ATLAS_SIZE);
    glUniform1f(glGetUniformLocation(prog, "uSlot"), (float)vt->slot_px);
    glUniform1f(glGetUniformLocation(prog, "uTile"), (float)vt->tile);
    glUniform1f(glGetUniformLocation(prog, "uBorder"), (float)vt->border);
    float rects[VT_MAX_LEVELS * 4] = {0};
    for (int l = 0; l < vt->levels; l++) {
        rects[l * 4 + 0] = 0.0f;
        rects[l * 4 + 1] = (float)vt->table_off_y[l];
        rects[l * 4 + 2] = (float)vt->nx[l];
        rects[l * 4 + 3] = (float)vt->ny[l];
    }
    glUniform4fv(glGetUniformLocation(prog, "uLevelRect"),
                 VT_MAX_LEVELS, rects);
}

/* ------------------------------------------------------------- matrices */

void mat4_perspective(float *m, float fovy_deg, float aspect,
                      float near, float far)
{
    memset(m, 0, 16 * sizeof(float));
    float f = 1.0f / tanf(fovy_deg * (float)M_PI / 360.0f);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = 2 * far * near / (near - far);
    m[14] = -1.0f;
}

static void v3_sub(float *o, const float *a, const float *b)
{ o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2]; }
static void v3_cross(float *o, const float *a, const float *b)
{
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}
static void v3_norm(float *v)
{
    float n = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 0) { v[0]/=n; v[1]/=n; v[2]/=n; }
}
static float v3_dot(const float *a, const float *b)
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

void mat4_look_at(float *m, const float eye[3], const float target[3],
                  const float up[3])
{
    float fwd[3], right[3], u[3];
    v3_sub(fwd, target, eye);
    v3_norm(fwd);
    v3_cross(right, fwd, up);
    v3_norm(right);
    v3_cross(u, right, fwd);
    memset(m, 0, 16 * sizeof(float));
    m[0]=right[0]; m[1]=right[1]; m[2]=right[2];  m[3]  = -v3_dot(right, eye);
    m[4]=u[0];     m[5]=u[1];     m[6]=u[2];      m[7]  = -v3_dot(u, eye);
    m[8]=-fwd[0];  m[9]=-fwd[1];  m[10]=-fwd[2];  m[11] =  v3_dot(fwd, eye);
    m[15] = 1.0f;
}

void mat4_mul(float *out, const float *a, const float *b)
{
    float t[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            t[r*4+c] = 0;
            for (int k = 0; k < 4; k++)
                t[r*4+c] += a[r*4+k] * b[k*4+c];
        }
    memcpy(out, t, sizeof t);
}

bool mat4_invert(float *out, const float *m)
{
    /* standard adjugate inverse (MESA gluInvertMatrix), row-major */
    float inv[16];
    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
           + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
           - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
           + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
            - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
           - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
           + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
           - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
            + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
           + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
           - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
            + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
            - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
           - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
           + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
            - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
            + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0)
        return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; i++)
        out[i] = inv[i] * det;
    return true;
}

void mat4_upload(GLint loc, const float *m)
{
    /* GLES2 forbids transpose=GL_TRUE: transpose row-major -> column */
    float t[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            t[c * 4 + r] = m[r * 4 + c];
    glUniformMatrix4fv(loc, 1, GL_FALSE, t);
}

/* ------------------------------------------------------ window / feedback */

bool vtw_create(VtWindow *w, const char *title)
{
    memset(w, 0, sizeof *w);

#ifdef VT_ANGLE_LIB_DIR
    /* point SDL at the ANGLE dylibs by absolute path, so no
       DYLD_FALLBACK_LIBRARY_PATH is needed at runtime */
    if (!getenv("SDL_VIDEO_GL_DRIVER")) {
        static char gles[1200], egl[1200];
        snprintf(gles, sizeof gles, "%s/libGLESv2.dylib", VT_ANGLE_LIB_DIR);
        snprintf(egl, sizeof egl, "%s/libEGL.dylib", VT_ANGLE_LIB_DIR);
        setenv("SDL_VIDEO_GL_DRIVER", gles, 0);
        setenv("SDL_VIDEO_EGL_DRIVER", egl, 0);
    }
#endif
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

    /* GLES2 context via ANGLE (see opengl-for-mac sdl_gles_minimal.c) */
    SDL_SetHint(SDL_HINT_OPENGL_ES_DRIVER, "1");
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_EGL, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    w->win = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED, 1280, 800,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_ALLOW_HIGHDPI
                              | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!w->win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return false;
    }
    w->ctx = SDL_GL_CreateContext(w->win);
    if (!w->ctx) {
        fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError());
        return false;
    }
    printf("GL version: %s\n", glGetString(GL_VERSION));
    vtw_update_sizes(w);

    /* small feedback FBO: page IDs need ~1 sample per tile on screen */
    w->fb_w = w->draw_w / 10 < 64 ? 64 : w->draw_w / 10;
    w->fb_h = w->draw_h / 10 < 64 ? 64 : w->draw_h / 10;
    glGenFramebuffers(1, &w->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, w->fbo);
    glGenTextures(1, &w->fb_tex);
    glBindTexture(GL_TEXTURE_2D, w->fb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w->fb_w, w->fb_h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, w->fb_tex, 0);
    GLuint rb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,
                          w->fb_w, w->fb_h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rb);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "feedback FBO incomplete\n");
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    w->fb_pixels = malloc(w->fb_w * w->fb_h * 4);
    return true;
}

void vtw_update_sizes(VtWindow *w)
{
    int ww, wh;
    SDL_GL_GetDrawableSize(w->win, &w->draw_w, &w->draw_h);
    SDL_GetWindowSize(w->win, &ww, &wh);
    w->dpi_scale = ww > 0 ? (float)w->draw_w / ww : 1.0f;
}

void vtw_run_feedback(VtWindow *w, VtSystem *vt, GLuint prog_fb,
                      const float *mvp,
                      void (*draw_geom)(void *), void *geom_arg)
{
    (void)mvp;
    glBindFramebuffer(GL_FRAMEBUFFER, w->fbo);
    glViewport(0, 0, w->fb_w, w->fb_h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(prog_fb);
    glUniform1f(glGetUniformLocation(prog_fb, "uLodBias"),
                log2f((float)w->fb_w / w->draw_w));
    draw_geom(geom_arg);
    glReadPixels(0, 0, w->fb_w, w->fb_h, GL_RGBA, GL_UNSIGNED_BYTE,
                 w->fb_pixels);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    vt_request_from_feedback(vt, w->fb_pixels, w->fb_w * w->fb_h);
}

void vtw_screenshot(VtWindow *w, const char *path)
{
    unsigned char *buf = malloc(w->draw_w * w->draw_h * 4);
    glReadPixels(0, 0, w->draw_w, w->draw_h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    stbi_flip_vertically_on_write(1);
    stbi_write_png(path, w->draw_w, w->draw_h, 4, buf, w->draw_w * 4);
    free(buf);
    printf("screenshot -> %s\n", path);
}
