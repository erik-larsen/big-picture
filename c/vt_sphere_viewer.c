/*
    vt_sphere_viewer.c - the mosaic wrapped onto a band of a sphere.
    C / SDL2 / GLES2 port of vt_sphere_viewer.py.

    Latitude spans +-60 deg; longitude span comes from the image aspect
    (distortion-free equator, capped at 360). Default view orbits the
    OUTSIDE of the sphere (globe); --inside views from the center.
    Hovering outlines the image under the pointer; clicking centers it,
    double-clicking zooms until it fills the window (flattening it)
    (via <pyramid>_layout.json) with an eased glide. Trackball fling inertia and slow
    idle auto-rotate included.

    Usage: vt_sphere_viewer [--pyramid DIR] [--inside] [--manifest F]
                            [--idle-delay S] [--idle-speed DEG/S]
                            [--frames N] [--screenshot OUT.png]
                            [--end-fov F] [--end-height H] [--click-test]
                            [--tiles URL]   (web build only)

    Built with Emscripten (build_web.sh) it runs in the browser: the page
    drives frame() once per display refresh, and adds pinch to zoom.
*/
#include "vt_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/* The band is generated in the vertex shader so it can morph: as you zoom
   into a picture the surface eases from sphere to the plane tangent at
   that picture, which also undoes the equirectangular squeeze away from
   the equator, so the photo ends up flat and in its true aspect ratio. */
static const char *SPHERE_VS_SRC =
    "attribute vec3 aPos;\n"
    "attribute vec2 aUV;\n"
    "uniform mat4 uMVP;\n"
    "uniform float uRadius, uLonHalf, uLatMax, uZSign, uFlat;\n"
    "uniform vec2 uFocus;\n"          /* lon, lat of the anchor */
    "uniform vec2 uFlatDim;\n"        /* full mosaic size once flat */
    "varying vec2 vUV;\n"
    "void main(){\n"
    "  vUV = aUV;\n"
    "  float lat = (0.5 - aUV.y) * 2.0 * uLatMax;\n"
    "  float lon = (aUV.x - 0.5) * 2.0 * uLonHalf;\n"
    "  vec3 p = uRadius * vec3(cos(lat)*sin(lon), sin(lat),\n"
    "                          uZSign*cos(lat)*cos(lon));\n"
    "  if (uFlat > 0.0) {\n"
    "    float lc = uFocus.x, tc = uFocus.y;\n"
    "    vec3 n     = vec3(cos(tc)*sin(lc), sin(tc), uZSign*cos(tc)*cos(lc));\n"
    "    vec3 east  = vec3(cos(lc), 0.0, -uZSign*sin(lc));\n"
    "    vec3 north = vec3(-sin(tc)*sin(lc), cos(tc), -uZSign*sin(tc)*cos(lc));\n"
    "    float uc = 0.5 + lc/(2.0*uLonHalf);\n"
    "    float vc = 0.5 - tc/(2.0*uLatMax);\n"
    "    vec3 fp = uRadius*n + east*((aUV.x-uc)*uFlatDim.x)\n"
    "                        + north*((vc-aUV.y)*uFlatDim.y);\n"
    "    p = mix(p, fp, uFlat);\n"
    "  }\n"
    "  gl_Position = uMVP * vec4(p, 1.0);\n"
    "}\n";

#define LAT_MAX (60.0f * (float)M_PI / 180.0f)
#define RADIUS 10.0f
#define OUT_FOV 55.0f
#define N_LON 192
#define N_LAT 64

typedef struct {
    VtWindow w;
    VtSystem vt;
    GLuint vbo, ebo;
    int n_idx;
    float lon_half;

    VtManifest manifest;
    char last_click[256];
    float hover_uv[4];
    bool has_hover;
    int hi_style;
    double t0;

    bool inside;
    float az, el, fov, height;
    int debug, freeze;
    VtLines lines;

    bool dragging;
    float drag_x, drag_y, press_x, press_y;
    bool has_press;
    float vel_az, vel_el;
    double move_t, last_input;
    float idle_dir;

    struct {
        bool active;
        double t0, dur;
        float az0, az1, el0, el1, zoom0, zoom1;
    } anim;

    float mvp[16];

    /* args */
    const char *pyramid, *manifest_path, *screenshot;
    int frames;
    float end_fov, end_height, idle_delay, idle_speed;
    bool click_test, hover_test;
    bool user_moved;            /* auto-orbit stops on first interaction */
    float flat;                 /* 0 = sphere, 1 = flat rectangle */
    float ref_rect[4];          /* fixed size driving the flattening ramp */
    float flat_w, flat_h;
    double last_click_t;

    /* touch: two fingers pinch-zoom (one finger arrives as the mouse) */
    SDL_FingerID finger_id[2];
    float finger_x[2], finger_y[2];
    int n_fingers;

    /* main loop state, kept here so the browser can drive frame() */
    bool running, reported_ready;
    int frame, fps_n;
    double loop_t0, fps_t, prev_t;
    char shown[256];            /* image name last reported to the page */
} App;

static float clampf(float v, float lo, float hi)
{ return v < lo ? lo : v > hi ? hi : v; }

static float wrap_pi(float a)
{
    while (a > (float)M_PI) a -= 2 * (float)M_PI;
    while (a < -(float)M_PI) a += 2 * (float)M_PI;
    return a;
}

/* ------------------------------------------------------------ web page */

/* The page around the canvas shows loading progress and credits whoever
   took the photo under the pointer; natively these are no-ops. */
static void page_notify_photo(App *a, const char *name)
{
    if (!strcmp(a->shown, name))
        return;
    snprintf(a->shown, sizeof a->shown, "%s", name);
#ifdef __EMSCRIPTEN__
    EM_ASM({ if (Module.vtPhoto) Module.vtPhoto(UTF8ToString($0)); }, name);
#endif
}

#ifdef __EMSCRIPTEN__
static void page_notify_status(App *a, float fps, int resident)
{
    EM_ASM({ if (Module.vtStatus) Module.vtStatus($0, $1, $2, $3, $4); },
           fps, resident, a->vt.n_slots, a->vt.n_pending, a->vt.loads_done);
}
#endif

/* ------------------------------------------------------------- manifest */

static void load_manifest(App *a)
{
    vt_manifest_load(&a->manifest, a->pyramid, a->manifest_path);
}

static bool cursor_uv(App *a, float cx, float cy, float *cu, float *cv);

/* cursor -> band -> mosaic uv -> hovered image rect */
static void update_hover(App *a, float cx, float cy)
{
    float cu, cv;
    a->has_hover = false;
    if (!a->manifest.rects || !cursor_uv(a, cx, cy, &cu, &cv))
        return;
    int hit = vt_manifest_rect_at(&a->manifest, cu, cv, a->vt.virt_w,
                                  a->vt.virt_h, a->hover_uv);
    a->has_hover = hit >= 0;
    if (hit >= 0)
        page_notify_photo(a, a->manifest.rects[hit].name);
}

/* --------------------------------------------------------------- camera */

static void radial(App *a, float out[3])
{
    out[0] = cosf(a->el) * sinf(a->az);
    out[1] = sinf(a->el);
    out[2] = cosf(a->el) * cosf(a->az);
}

static void compute_mvp(App *a)
{
    float up[3] = {0, 1, 0}, view[16], proj[16], dir[3];
    radial(a, dir);
    if (a->inside) {
        float eye[3] = {0, 0, 0};
        float tgt[3] = {dir[0], dir[1], -dir[2]};
        mat4_look_at(view, eye, tgt, up);
        mat4_perspective(proj, a->fov,
                         (float)a->w.draw_w / a->w.draw_h, 0.05f, 100.0f);
    } else {
        float eye[3] = {(RADIUS + a->height) * dir[0],
                        (RADIUS + a->height) * dir[1],
                        (RADIUS + a->height) * dir[2]};
        float tgt[3] = {0, 0, 0};
        mat4_look_at(view, eye, tgt, up);
        mat4_perspective(proj, OUT_FOV,
                         (float)a->w.draw_w / a->w.draw_h, 0.01f, 200.0f);
    }
    mat4_mul(a->mvp, proj, view);
}

static void mat4_xform(const float *m, const float *v, float *out)
{
    for (int r = 0; r < 4; r++)
        out[r] = m[r*4+0]*v[0] + m[r*4+1]*v[1]
               + m[r*4+2]*v[2] + m[r*4+3]*v[3];
}

/* cursor -> ray -> point on band -> mosaic uv; false on miss */
/* camera height at which `rect` exactly fills the window */
static float fit_height(App *a, const float rect[4])
{
    float pw = (rect[2] - rect[0]) * a->flat_w;
    float ph = (rect[3] - rect[1]) * a->flat_h;
    int ww, wh;
    SDL_GetWindowSize(a->w.win, &ww, &wh);
    float tan_v = tanf(OUT_FOV * (float)M_PI / 360.0f);
    float tan_h = tan_v * (float)ww / (wh > 0 ? wh : 1);
    return fmaxf(ph / 2 / tan_v, pw / 2 / tan_h);
}

/* Opening camera height: the globe fills the window's height, or its
   width when the window is portrait (a phone), so it is always whole. */
static float home_height(App *a)
{
    int ww, wh;
    SDL_GetWindowSize(a->w.win, &ww, &wh);
    float tan_v = tanf(OUT_FOV * (float)M_PI / 360.0f);
    float tan_h = tan_v * (float)ww / (wh > 0 ? wh : 1);
    float half = atanf(fminf(tan_v, tan_h));
    return fmaxf(12.0f, RADIUS / sinf(half) - RADIUS);
}

/* Tangent point = where the camera looks. Continuous by construction,
   so the flattening never snaps sideways. */
static void focus_lonlat(App *a, float *lon, float *lat)
{
    *lon = a->az;
    *lat = a->el;
}

/* Ease the surface flat purely as a function of how close the camera is
   to the surface. Deliberately not keyed to whichever picture is centred:
   anchoring on a specific photo made both the anchor and the ramp jump
   every time one scrolled past the middle of the view. Smooth everywhere
   beats exact somewhere — and when the view *is* centred on a photo
   (after a double click) the tangent point lands on it anyway. */
static void update_flat(App *a)
{
    if (a->inside) {
        a->flat = 0.0f;
        return;
    }
    float lo = fit_height(a, a->ref_rect);     /* fully flat by here */
    float ratio = a->height / fmaxf(lo, 1e-6f);
    const float RAMP = 8.0f;                   /* start 8x further out */
    float t = clampf(1.0f - log2f(fmaxf(ratio, 1e-6f)) / log2f(RAMP),
                     0.0f, 1.0f);
    a->flat = t * t * (3.0f - 2.0f * t);
}

/* hit the sphere itself */
static bool uv_sphere(App *a, const float p0[3], const float d[3],
                      float *cu, float *cv)
{
    float px, py, pz;
    if (a->inside) {
        px = d[0]; py = d[1]; pz = d[2];
    } else {                                    /* near ray-sphere hit */
        float b = p0[0]*d[0] + p0[1]*d[1] + p0[2]*d[2];
        float c = p0[0]*p0[0] + p0[1]*p0[1] + p0[2]*p0[2] - RADIUS*RADIUS;
        float disc = b*b - c;
        if (disc < 0)
            return false;
        float t = -b - sqrtf(disc);
        if (t <= 0)
            return false;
        px = (p0[0] + t*d[0]) / RADIUS;
        py = (p0[1] + t*d[1]) / RADIUS;
        pz = (p0[2] + t*d[2]) / RADIUS;
    }
    float lat = asinf(clampf(py, -1, 1));
    float lon = atan2f(px, a->inside ? -pz : pz);
    if (fabsf(lat) > LAT_MAX || fabsf(lon) > a->lon_half)
        return false;
    *cu = 0.5f + lon / (2 * a->lon_half);
    *cv = 0.5f - lat / (2 * LAT_MAX);
    return true;
}

/* hit the tangent plane the surface is flattening onto */
static bool uv_plane(App *a, const float p0[3], const float d[3],
                     float *cu, float *cv)
{
    float lc, tc;
    focus_lonlat(a, &lc, &tc);
    float z = a->inside ? -1.0f : 1.0f;
    float n[3]     = {cosf(tc)*sinf(lc), sinf(tc), z*cosf(tc)*cosf(lc)};
    float east[3]  = {cosf(lc), 0.0f, -z*sinf(lc)};
    float north[3] = {-sinf(tc)*sinf(lc), cosf(tc), -z*sinf(tc)*cosf(lc)};
    float dn = d[0]*n[0] + d[1]*n[1] + d[2]*n[2];
    if (fabsf(dn) < 1e-9f)
        return false;
    float num = 0.0f;
    for (int i = 0; i < 3; i++)
        num += (RADIUS*n[i] - p0[i]) * n[i];
    float t = num / dn;
    if (t <= 0)
        return false;
    float q[3];
    for (int i = 0; i < 3; i++)
        q[i] = p0[i] + t*d[i] - RADIUS*n[i];
    *cu = 0.5f + lc / (2 * a->lon_half)
        + (q[0]*east[0] + q[1]*east[1] + q[2]*east[2]) / a->flat_w;
    *cv = 0.5f - tc / (2 * LAT_MAX)
        - (q[0]*north[0] + q[1]*north[1] + q[2]*north[2]) / a->flat_h;
    return true;
}

/* The surface morphs between sphere and tangent plane, so the pick
   blends the two by the same factor: exact at either end, and the two
   agree near the focus in between. */
static bool cursor_uv(App *a, float cx, float cy, float *cu, float *cv)
{
    int ww, wh;
    SDL_GetWindowSize(a->w.win, &ww, &wh);
    if (!ww || !wh)
        return false;
    compute_mvp(a);
    float inv[16];
    if (!mat4_invert(inv, a->mvp))
        return false;
    float nx = 2.0f * cx / ww - 1.0f, ny = 1.0f - 2.0f * cy / wh;
    float pn[4] = {nx, ny, -1, 1}, pf[4] = {nx, ny, 1, 1};
    float e0[4], e1[4];
    mat4_xform(inv, pn, e0);
    mat4_xform(inv, pf, e1);
    float p0[3], d[3];
    for (int i = 0; i < 3; i++) {
        p0[i] = e0[i] / e0[3];
        d[i] = e1[i] / e1[3] - p0[i];
    }
    float dl = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    for (int i = 0; i < 3; i++)
        d[i] /= dl;

    float su, sv, pu, pv;
    bool hs = uv_sphere(a, p0, d, &su, &sv);
    if (a->flat <= 0.0f) {
        if (!hs)
            return false;
        *cu = su; *cv = sv;
        return true;
    }
    bool hp = uv_plane(a, p0, d, &pu, &pv);
    if (!hs && !hp)
        return false;
    if (!hs) { su = pu; sv = pv; }
    if (!hp) { pu = su; pv = sv; }
    *cu = su + (pu - su) * a->flat;
    *cv = sv + (pv - sv) * a->flat;
    return *cu >= 0.0f && *cu <= 1.0f && *cv >= 0.0f && *cv <= 1.0f;
}

/* ---------------------------------------------- click: center + zoom 50% */

static void click_center_zoom(App *a, float cx, float cy, bool fit)
{
    float cu, cv;
    if (!cursor_uv(a, cx, cy, &cu, &cv))
        return;
    float rect[4];
    int hit = vt_manifest_rect_at(&a->manifest, cu, cv, a->vt.virt_w,
                                  a->vt.virt_h, rect);
    bool have_rect = hit >= 0;
    if (hit >= 0) {                      /* center the image, not the point */
        cu = (rect[0] + rect[2]) * 0.5f;
        cv = (rect[1] + rect[3]) * 0.5f;
        snprintf(a->last_click, sizeof a->last_click, "%s",
                 a->manifest.rects[hit].name);
        page_notify_photo(a, a->last_click);
    }
    float zoom = a->inside ? a->fov : a->height;
    float end_zoom = zoom;              /* single click: centre only */
    if (fit && have_rect && !a->inside)
        end_zoom = fit_height(a, rect);
    float az1 = (cu - 0.5f) * 2 * a->lon_half;
    float el1 = (0.5f - cv) * 2 * LAT_MAX;
    a->anim.az0 = a->az;
    a->anim.az1 = a->az + wrap_pi(az1 - a->az);
    a->anim.el0 = a->el;
    a->anim.el1 = el1;
    a->anim.zoom0 = zoom;
    a->anim.zoom1 = end_zoom;
    a->anim.t0 = now_seconds();
    a->anim.dur = 0.5;
    a->anim.active = true;
}

static void update_anim(App *a)
{
    if (!a->anim.active)
        return;
    float u = (float)((now_seconds() - a->anim.t0) / a->anim.dur);
    float e = smoothstep01(u);
    a->az = a->anim.az0 + (a->anim.az1 - a->anim.az0) * e;
    a->el = a->anim.el0 + (a->anim.el1 - a->anim.el0) * e;
    float zoom = a->anim.zoom0 + (a->anim.zoom1 - a->anim.zoom0) * e;
    if (a->inside)
        a->fov = zoom;
    else
        a->height = zoom;
    if (u >= 1.0f) {
        a->az = wrap_pi(a->az);
        a->anim.active = false;
    }
}

static void toggle_freeze(App *a)
{
    a->freeze ^= 1;
    if (a->freeze) {                /* pin the frustum that chose the
                                       currently resident working set */
        compute_mvp(a);
        float eye[3] = {0, 0, 0}, near, far, dir[3];
        if (a->inside) {
            near = 0.05f;
            far = 2.5f * RADIUS;
        } else {
            radial(a, dir);
            eye[0] = (RADIUS + a->height) * dir[0];
            eye[1] = (RADIUS + a->height) * dir[1];
            eye[2] = (RADIUS + a->height) * dir[2];
            near = 0.01f;
            far = a->height + 2.0f * RADIUS;
        }
        vt_lines_capture(&a->lines, a->mvp, eye, near, far);
        vt_evict_unused(&a->vt);    /* keep only this view's pages */
    }
}

/* the camera moves on its own (glides, inertia, attract spin), so what
   sits under a stationary pointer changes without any mouse event */
static void refresh_hover(App *a)
{
    if (!a->manifest.rects || SDL_GetMouseFocus() != a->w.win)
        return;
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    update_hover(a, (float)mx, (float)my);
}

/* ------------------------------------------- inertia + idle auto-rotate */

static void update_free_motion(App *a, float dt)
{
    if (a->dragging || a->anim.active)
        return;
    bool full_wrap = a->lon_half >= (float)M_PI - 1e-6f;
    if (fabsf(a->vel_az) > 1e-3f || fabsf(a->vel_el) > 1e-3f) {
        a->az += a->vel_az * dt;
        a->el += a->vel_el * dt;
        float decay = expf(-3.5f * dt);
        a->vel_az *= decay;
        a->vel_el *= decay;
        if (fabsf(a->el) > LAT_MAX + 0.25f) {
            a->el = copysignf(LAT_MAX + 0.25f, a->el);
            a->vel_el = 0;
        }
        if (full_wrap) {
            a->az = wrap_pi(a->az);
        } else if (fabsf(a->az) > a->lon_half) {
            a->az = copysignf(a->lon_half, a->az);
            a->vel_az = 0;
        }
    } else if (a->idle_delay > 0 && !a->user_moved
               && now_seconds() - a->last_input > a->idle_delay) {
        float step = a->idle_speed * (float)M_PI / 180.0f * dt;
        if (full_wrap) {
            a->az = wrap_pi(a->az + step);
        } else {
            float lim = 0.85f * a->lon_half;
            if (fabsf(a->az) > lim)
                a->idle_dir = -copysignf(1.0f, a->az);
            a->az += step * a->idle_dir;
        }
    }
}

/* ----------------------------------------------------------------- draw */

typedef struct {
    App *a;
    GLuint prog;
} DrawCtx;

static void draw_band(void *arg)
{
    DrawCtx *c = arg;
    App *a = c->a;
    vt_bind(&a->vt, c->prog);
    mat4_upload(glGetUniformLocation(c->prog, "uMVP"), a->mvp);
    float lon_c, lat_c;
    focus_lonlat(a, &lon_c, &lat_c);
    glUniform1f(glGetUniformLocation(c->prog, "uRadius"), RADIUS);
    glUniform1f(glGetUniformLocation(c->prog, "uLonHalf"), a->lon_half);
    glUniform1f(glGetUniformLocation(c->prog, "uLatMax"), LAT_MAX);
    glUniform1f(glGetUniformLocation(c->prog, "uZSign"),
                a->inside ? -1.0f : 1.0f);
    glUniform1f(glGetUniformLocation(c->prog, "uFlat"), a->flat);
    glUniform2f(glGetUniformLocation(c->prog, "uFocus"), lon_c, lat_c);
    glUniform2f(glGetUniformLocation(c->prog, "uFlatDim"),
                a->flat_w, a->flat_h);
    glBindBuffer(GL_ARRAY_BUFFER, a->vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, a->ebo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, (void *)12);
    glDrawElements(GL_TRIANGLES, a->n_idx, GL_UNSIGNED_SHORT, 0);
}

static void build_band_mesh(App *a)
{
    float z_sign = a->inside ? -1.0f : 1.0f;
    int nv = (N_LON + 1) * (N_LAT + 1);
    float *verts = malloc(nv * 5 * sizeof(float));
    for (int j = 0; j <= N_LAT; j++) {
        float v = (float)j / N_LAT;
        float lat = (0.5f - v) * 2 * LAT_MAX;
        for (int i = 0; i <= N_LON; i++) {
            float u = (float)i / N_LON;
            float lon = (u - 0.5f) * 2 * a->lon_half;
            float *p = verts + (j * (N_LON + 1) + i) * 5;
            p[0] = RADIUS * cosf(lat) * sinf(lon);
            p[1] = RADIUS * sinf(lat);
            p[2] = z_sign * RADIUS * cosf(lat) * cosf(lon);
            p[3] = u;
            p[4] = v;
        }
    }
    a->n_idx = N_LON * N_LAT * 6;
    unsigned short *idx = malloc(a->n_idx * sizeof(unsigned short));
    int k = 0;
    for (int j = 0; j < N_LAT; j++)
        for (int i = 0; i < N_LON; i++) {
            int p0 = j * (N_LON + 1) + i;
            int p1 = p0 + 1, p2 = p0 + N_LON + 1, p3 = p2 + 1;
            idx[k++] = p0; idx[k++] = p1; idx[k++] = p3;
            idx[k++] = p0; idx[k++] = p3; idx[k++] = p2;
        }
    glGenBuffers(1, &a->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, a->vbo);
    glBufferData(GL_ARRAY_BUFFER, nv * 5 * sizeof(float), verts,
                 GL_STATIC_DRAW);
    glGenBuffers(1, &a->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, a->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 a->n_idx * sizeof(unsigned short), idx, GL_STATIC_DRAW);
    free(verts);
    free(idx);
}

/* ---------------------------------------------------------------- input */

static float drag_rate(App *a)
{
    if (a->inside)
        return 0.0025f * a->fov / 70.0f;
    return 0.005f * clampf(a->height / RADIUS, 0.03f, 1.5f);
}

static float finger_spread(App *a)
{
    int ww, wh;
    SDL_GetWindowSize(a->w.win, &ww, &wh);
    float dx = (a->finger_x[0] - a->finger_x[1]) * ww;
    float dy = (a->finger_y[0] - a->finger_y[1]) * wh;
    return sqrtf(dx * dx + dy * dy);
}

static void zoom_by(App *a, float factor)
{
    a->last_input = now_seconds();
    a->user_moved = true;
    a->anim.active = false;
    if (a->inside)
        a->fov = clampf(a->fov * factor, 3.0f, 110.0f);
    else
        a->height = clampf(a->height * factor, 0.02f, 60.0f);
}

static void handle_finger(App *a, const SDL_Event *ev)
{
    const SDL_TouchFingerEvent *f = &ev->tfinger;
    int k = -1;
    for (int i = 0; i < a->n_fingers; i++)
        if (a->finger_id[i] == f->fingerId)
            k = i;
    if (ev->type == SDL_FINGERDOWN) {
        if (k < 0 && a->n_fingers < 2) {
            k = a->n_fingers++;
            a->finger_id[k] = f->fingerId;
        }
        if (k >= 0) {
            a->finger_x[k] = f->x;
            a->finger_y[k] = f->y;
        }
        if (a->n_fingers == 2) {     /* a pinch, not a drag or a tap */
            a->dragging = a->has_press = false;
            a->vel_az = a->vel_el = 0;
        }
    } else if (ev->type == SDL_FINGERMOTION && k >= 0) {
        float before = a->n_fingers == 2 ? finger_spread(a) : 0;
        a->finger_x[k] = f->x;
        a->finger_y[k] = f->y;
        if (a->n_fingers == 2 && before > 1) {
            float after = finger_spread(a);
            if (after > 1)
                zoom_by(a, before / after);
        }
    } else if (ev->type == SDL_FINGERUP && k >= 0) {
        a->finger_id[k] = a->finger_id[--a->n_fingers];
        a->finger_x[k] = a->finger_x[a->n_fingers];
        a->finger_y[k] = a->finger_y[a->n_fingers];
    }
}

static void handle_events(App *a, bool *running)
{
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
        case SDL_QUIT:
            *running = false;
            break;
        case SDL_KEYDOWN:
            a->last_input = now_seconds();
            if (ev.key.keysym.sym == SDLK_ESCAPE) {
#ifndef __EMSCRIPTEN__                /* a web page has nothing to quit */
                *running = false;
#endif
            } else if (ev.key.keysym.sym == SDLK_l)
                a->debug ^= 1;
            else if (ev.key.keysym.sym == SDLK_f)
                toggle_freeze(a);
            else if (ev.key.keysym.sym == SDLK_h) {
                a->hi_style = (a->hi_style + 1) % 6;
                printf("highlight style %d\n", a->hi_style);
            }
            else if (ev.key.keysym.sym == SDLK_r) {
                a->anim.active = false;
                a->vel_az = a->vel_el = 0;
                a->az = a->el = 0;
                a->fov = 75.0f;
                a->height = home_height(a);
                a->vel_az = a->vel_el = 0;
                a->user_moved = false;   /* back to the opening spin */
            }
            break;
        case SDL_FINGERDOWN:
        case SDL_FINGERMOTION:
        case SDL_FINGERUP:
            handle_finger(a, &ev);
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (a->n_fingers >= 2)
                break;                  /* second finger of a pinch */
            a->last_input = now_seconds();
            a->user_moved = true;
            a->anim.active = false;
            a->vel_az = a->vel_el = 0;
            a->move_t = 0;
            a->dragging = true;
            a->drag_x = a->press_x = ev.button.x;
            a->drag_y = a->press_y = ev.button.y;
            a->has_press = true;
            break;
        case SDL_MOUSEBUTTONUP: {
            a->last_input = now_seconds();
            /* a fingertip wobbles more than a mouse between down and up */
            float slop = ev.button.which == SDL_TOUCH_MOUSEID ? 12 : 4;
            if (ev.button.button == SDL_BUTTON_LEFT && a->has_press
                    && fabsf(ev.button.x - a->press_x) < slop
                    && fabsf(ev.button.y - a->press_y) < slop) {
                a->vel_az = a->vel_el = 0;
                {
                    double now = now_seconds();
                    bool dbl = now - a->last_click_t < 0.35;
                    a->last_click_t = now;
                    click_center_zoom(a, ev.button.x, ev.button.y, dbl);
                }
            } else if (a->move_t == 0
                       || now_seconds() - a->move_t > 0.12) {
                a->vel_az = a->vel_el = 0;   /* held still: no fling */
            }
            a->dragging = false;
            a->has_press = false;
            break;
        }
        case SDL_MOUSEMOTION:
            if (!a->dragging)
                update_hover(a, ev.motion.x, ev.motion.y);
            if (a->dragging) {
                double now = now_seconds();
                a->last_input = now;
                a->user_moved = true;
                float rate = drag_rate(a);
                float daz = -(ev.motion.x - a->drag_x) * rate;
                float del = (ev.motion.y - a->drag_y) * rate;
                a->drag_x = ev.motion.x;
                a->drag_y = ev.motion.y;
                a->az = wrap_pi(a->az + daz);
                a->el = clampf(a->el + del,
                               -LAT_MAX - 0.25f, LAT_MAX + 0.25f);
                float dt = a->move_t ? (float)(now - a->move_t) : 0;
                a->move_t = now;
                if (dt > 1e-4f && dt < 0.2f) {
                    float s = dt / 0.05f;
                    if (s > 1) s = 1;
                    a->vel_az += (daz / dt - a->vel_az) * s;
                    a->vel_el += (del / dt - a->vel_el) * s;
                    a->vel_az = clampf(a->vel_az, -4, 4);
                    a->vel_el = clampf(a->vel_el, -4, 4);
                }
            }
            break;
        case SDL_MOUSEWHEEL:
            zoom_by(a, powf(0.92f, ev.wheel.preciseY));
            break;
        case SDL_WINDOWEVENT:
            if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                vtw_update_sizes(&a->w);
            break;
        }
    }
}

/* ----------------------------------------------------------------- main */

/* One iteration of the main loop: natively called from a while loop,
   on the web by the browser once per display refresh. */
static void frame(void *arg)
{
    App *a = arg;
    handle_events(a, &a->running);
    double now = now_seconds();
    float dt = (float)(now - a->prev_t);
    if (dt > 0.1f) dt = 0.1f;
    a->prev_t = now;

    if (a->frames >= 0) {                       /* scripted sweep */
        if (a->frame < a->frames) {
            float k = (float)a->frame / (a->frames > 1 ? a->frames - 1 : 1);
            a->az = 0.7f * a->lon_half * sinf(2 * (float)M_PI * k);
            a->el = 0.5f * LAT_MAX * sinf(4 * (float)M_PI * k);
            if (a->inside)
                a->fov = 75.0f * powf(a->end_fov / 75.0f, k);
            else
                a->height = 12.0f * powf(a->end_height / 12.0f, k);
        } else {
            int extra = a->frame - a->frames;
            if (a->hover_test) {             /* hover the window centre */
                int ww, wh;
                SDL_GetWindowSize(a->w.win, &ww, &wh);
                update_hover(a, ww / 2.0f, wh / 2.0f);
            }
            if (a->click_test
                    && (extra == 20 || extra == 60 || extra == 100)) {
                int ww, wh;
                SDL_GetWindowSize(a->w.win, &ww, &wh);
                click_center_zoom(a, ww * 0.32f, wh * 0.38f, false);
                if (a->anim.active)
                    printf("click -> %s  az %.1f  el %.1f  zoom %.2f\n",
                           a->last_click[0] ? a->last_click : "(point)",
                           a->anim.az1 * 180.0 / M_PI,
                           a->anim.el1 * 180.0 / M_PI, a->anim.zoom1);
            }
            int min_extra = a->click_test ? 120 : 0;
            if ((extra > min_extra && !a->anim.active
                    && vt_idle(&a->vt)) || extra > 600)
                a->running = false;
        }
    } else {
        update_free_motion(a, dt);
    }
    update_anim(a);
    update_flat(a);
    if (a->frames < 0)
        refresh_hover(a);
    compute_mvp(a);

    if (!a->freeze) {
        DrawCtx fb_ctx = {a, a->vt.prog_fb};
        vtw_run_feedback(&a->w, &a->vt, a->vt.prog_fb, a->mvp,
                         draw_band, &fb_ctx);
        vt_pump_uploads(&a->vt);
    }
    if (!a->reported_ready && vt_root_resident(&a->vt)) {
        a->reported_ready = true;
#ifdef __EMSCRIPTEN__
        EM_ASM({ if (Module.vtReady) Module.vtReady(); });
#endif
    }

    glViewport(0, 0, a->w.draw_w, a->w.draw_h);
    glClearColor(0.05f, 0.06f, 0.08f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    DrawCtx main_ctx = {a, a->vt.prog_main};
    vt_bind(&a->vt, a->vt.prog_main);
    glUniform1i(glGetUniformLocation(a->vt.prog_main, "uDebug"), a->debug);
    vt_highlight_uniforms(a->vt.prog_main, a->has_hover ? a->hover_uv : NULL,
                          a->hi_style, (float)(now_seconds() - a->t0),
                          a->w.dpi_scale);
    draw_band(&main_ctx);
    if (a->freeze)
        vt_lines_draw(&a->lines, a->mvp);

    if (!a->running && a->screenshot)
        vtw_screenshot(&a->w, a->screenshot);
    SDL_GL_SwapWindow(a->w.win);
    a->frame++;
    a->fps_n++;
    if (now - a->fps_t > 0.5) {
        int resident = 0;
        for (int s = 0; s < a->vt.n_slots; s++)
            if (a->vt.slot_page[s].level >= 0)
                resident++;
        float fps = (float)(a->fps_n / (now - a->fps_t));
#ifdef __EMSCRIPTEN__             /* the page's status line, not the tab */
        page_notify_status(a, fps, resident);
#else
        char title[400];
        snprintf(title, sizeof title,
                 "vt sphere viewer (C/GLES2) | %5.1f fps | resident "
                 "%d/%d | pending %d%s%s%s",
                 fps, resident, a->vt.n_slots,
                 a->vt.n_pending, a->freeze ? " | FROZEN" : "",
                 a->last_click[0] ? " | " : "", a->last_click);
        SDL_SetWindowTitle(a->w.win, title);
#endif
        a->fps_t = now;
        a->fps_n = 0;
    }
#ifdef __EMSCRIPTEN__
    if (!a->running)
        emscripten_cancel_main_loop();
#endif
}

int main(int argc, char **argv)
{
    static App app;       /* static: on the web it outlives main() */
    App *a = &app;
    vt_argv0 = argv[0];
    a->pyramid = "../pics/test_image_16k_pyramid";
    a->frames = -1;
    a->end_fov = 28.0f;
    a->end_height = 1.0f;
    a->idle_delay = 5.0f;
    a->idle_speed = 3.0f;
    a->idle_dir = 1.0f;
    a->hi_style = 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pyramid") && i + 1 < argc)
            a->pyramid = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc)
            a->manifest_path = argv[++i];
        else if (!strcmp(argv[i], "--inside"))
            a->inside = true;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            a->frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc)
            a->screenshot = argv[++i];
        else if (!strcmp(argv[i], "--end-fov") && i + 1 < argc)
            a->end_fov = atof(argv[++i]);
        else if (!strcmp(argv[i], "--end-height") && i + 1 < argc)
            a->end_height = atof(argv[++i]);
        else if (!strcmp(argv[i], "--idle-delay") && i + 1 < argc)
            a->idle_delay = atof(argv[++i]);
        else if (!strcmp(argv[i], "--idle-speed") && i + 1 < argc)
            a->idle_speed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--click-test"))
            a->click_test = true;
        else if (!strcmp(argv[i], "--hover-test"))
            a->hover_test = true;
        else if (!strcmp(argv[i], "--hi-style") && i + 1 < argc)
            a->hi_style = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tiles") && i + 1 < argc)
            vt_tile_url = argv[++i];
        else if (argv[i][0] != '-')
            a->pyramid = argv[i];
    }

    if (!vt_check_pyramid(a->pyramid))    /* before opening a window */
        return 1;
    if (!vtw_create(&a->w, "vt sphere viewer (C/GLES2)"))
        return 1;
    SDL_GL_SetSwapInterval(a->frames >= 0 ? 0 : 1);
    if (!vt_init(&a->vt, a->pyramid))
        return 1;
    load_manifest(a);
    vt_lines_init(&a->lines);
    a->t0 = now_seconds();

    float aspect = (float)a->vt.virt_w / a->vt.virt_h;
    a->lon_half = fminf((float)M_PI, LAT_MAX * aspect);
    printf("band: lon +-%.1f deg, lat +-60 deg, %s view\n",
           a->lon_half * 180.0 / M_PI, a->inside ? "inside" : "outside");
    /* pair the shared fragment shaders with our morphing vertex shader */
    a->vt.prog_main = vt_compile_program(SPHERE_VS_SRC, vt_fs_main_src);
    a->vt.prog_fb = vt_compile_program(SPHERE_VS_SRC, vt_fs_feedback_src);
    a->flat_w = 2 * a->lon_half * RADIUS;
    a->flat_h = 2 * LAT_MAX * RADIUS;
    /* largest rect: every photo is fully flat by the time it fills the
       window, and the ramp never depends on which one is centred */
    a->ref_rect[0] = a->ref_rect[1] = 0.0f;
    a->ref_rect[2] = a->ref_rect[3] = 1.0f / 12;  /* no manifest fallback */
    if (a->manifest.rects) {
        int mw = 0, mh = 0;
        for (int i = 0; i < a->manifest.n; i++) {
            if (a->manifest.rects[i].w > mw) mw = a->manifest.rects[i].w;
            if (a->manifest.rects[i].h > mh) mh = a->manifest.rects[i].h;
        }
        a->ref_rect[2] = (float)mw / a->vt.virt_w;
        a->ref_rect[3] = (float)mh / a->vt.virt_h;
    }
    build_band_mesh(a);

    a->fov = 75.0f;
    a->height = home_height(a);
    a->last_input = now_seconds();

    glEnable(GL_DEPTH_TEST);

    a->running = true;
    a->loop_t0 = a->fps_t = a->prev_t = now_seconds();
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(frame, a, 0, true);   /* does not return */
#else
    while (a->running)
        frame(a);

    printf("frames: %d  time: %.1fs  tiles loaded: %d\n",
           a->frame, now_seconds() - a->loop_t0, a->vt.loads_done);
    vt_destroy(&a->vt);
    SDL_GL_DeleteContext(a->w.ctx);
    SDL_DestroyWindow(a->w.win);
    SDL_Quit();
#endif
    return 0;
}
