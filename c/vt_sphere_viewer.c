/*
    vt_sphere_viewer.c - the mosaic wrapped onto a band of a sphere.
    C / SDL2 / GLES2 port of vt_sphere_viewer.py.

    Latitude spans +-60 deg; longitude span comes from the image aspect
    (distortion-free equator, capped at 360). Default view orbits the
    OUTSIDE of the sphere (globe); --inside views from the center.
    Hovering outlines the image under the pointer; clicking centers it
    (via <pyramid>_layout.json) with an eased glide. Trackball fling inertia and slow
    idle auto-rotate included.

    Usage: vt_sphere_viewer [--pyramid DIR] [--inside] [--manifest F]
                            [--idle-delay S] [--idle-speed DEG/S]
                            [--frames N] [--screenshot OUT.png]
                            [--end-fov F] [--end-height H] [--click-test]
*/
#include "vt_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} App;

static float clampf(float v, float lo, float hi)
{ return v < lo ? lo : v > hi ? hi : v; }

static float wrap_pi(float a)
{
    while (a > (float)M_PI) a -= 2 * (float)M_PI;
    while (a < -(float)M_PI) a += 2 * (float)M_PI;
    return a;
}

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
    a->has_hover = vt_manifest_rect_at(&a->manifest, cu, cv, a->vt.virt_w,
                                       a->vt.virt_h, a->hover_uv) >= 0;
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
    float p0[4], p1[4];
    mat4_xform(inv, pn, p0);
    mat4_xform(inv, pf, p1);
    for (int i = 0; i < 3; i++) {
        p0[i] /= p0[3];
        p1[i] /= p1[3];
    }
    float d[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    float dn = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    d[0] /= dn; d[1] /= dn; d[2] /= dn;

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

/* ---------------------------------------------- click: center + zoom 50% */

static void click_center_zoom(App *a, float cx, float cy)
{
    float cu, cv;
    if (!cursor_uv(a, cx, cy, &cu, &cv))
        return;
    float rect[4];
    int hit = vt_manifest_rect_at(&a->manifest, cu, cv, a->vt.virt_w,
                                  a->vt.virt_h, rect);
    if (hit >= 0) {                      /* center the image, not the point */
        cu = (rect[0] + rect[2]) * 0.5f;
        cv = (rect[1] + rect[3]) * 0.5f;
        snprintf(a->last_click, sizeof a->last_click, "%s",
                 a->manifest.rects[hit].name);
    }
    float zoom = a->inside ? a->fov : a->height;
    float az1 = (cu - 0.5f) * 2 * a->lon_half;
    float el1 = (0.5f - cv) * 2 * LAT_MAX;
    a->anim.az0 = a->az;
    a->anim.az1 = a->az + wrap_pi(az1 - a->az);
    a->anim.el0 = a->el;
    a->anim.el1 = el1;
    a->anim.zoom0 = zoom;
    a->anim.zoom1 = zoom;                 /* centering only, no zoom */
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
    } else if (a->idle_delay > 0
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
            if (ev.key.keysym.sym == SDLK_ESCAPE)
                *running = false;
            else if (ev.key.keysym.sym == SDLK_l)
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
                a->height = 12.0f;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            a->last_input = now_seconds();
            a->anim.active = false;
            a->vel_az = a->vel_el = 0;
            a->move_t = 0;
            a->dragging = true;
            a->drag_x = a->press_x = ev.button.x;
            a->drag_y = a->press_y = ev.button.y;
            a->has_press = true;
            break;
        case SDL_MOUSEBUTTONUP:
            a->last_input = now_seconds();
            if (ev.button.button == SDL_BUTTON_LEFT && a->has_press
                    && fabsf(ev.button.x - a->press_x) < 4
                    && fabsf(ev.button.y - a->press_y) < 4) {
                a->vel_az = a->vel_el = 0;
                click_center_zoom(a, ev.button.x, ev.button.y);
            } else if (a->move_t == 0
                       || now_seconds() - a->move_t > 0.12) {
                a->vel_az = a->vel_el = 0;   /* held still: no fling */
            }
            a->dragging = false;
            a->has_press = false;
            break;
        case SDL_MOUSEMOTION:
            if (!a->dragging)
                update_hover(a, ev.motion.x, ev.motion.y);
            if (a->dragging) {
                double now = now_seconds();
                a->last_input = now;
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
            a->last_input = now_seconds();
            a->anim.active = false;
            if (a->inside)
                a->fov = clampf(a->fov * powf(0.92f, ev.wheel.preciseY),
                                3.0f, 110.0f);
            else
                a->height = clampf(a->height
                                   * powf(0.92f, ev.wheel.preciseY),
                                   0.02f, 60.0f);
            break;
        case SDL_WINDOWEVENT:
            if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                vtw_update_sizes(&a->w);
            break;
        }
    }
}

/* ----------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    App a;
    memset(&a, 0, sizeof a);
    vt_argv0 = argv[0];
    a.pyramid = "test_image_16k_pyramid";
    a.frames = -1;
    a.end_fov = 28.0f;
    a.end_height = 1.0f;
    a.idle_delay = 5.0f;
    a.idle_speed = 3.0f;
    a.idle_dir = 1.0f;
    a.hi_style = 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pyramid") && i + 1 < argc)
            a.pyramid = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc)
            a.manifest_path = argv[++i];
        else if (!strcmp(argv[i], "--inside"))
            a.inside = true;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            a.frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc)
            a.screenshot = argv[++i];
        else if (!strcmp(argv[i], "--end-fov") && i + 1 < argc)
            a.end_fov = atof(argv[++i]);
        else if (!strcmp(argv[i], "--end-height") && i + 1 < argc)
            a.end_height = atof(argv[++i]);
        else if (!strcmp(argv[i], "--idle-delay") && i + 1 < argc)
            a.idle_delay = atof(argv[++i]);
        else if (!strcmp(argv[i], "--idle-speed") && i + 1 < argc)
            a.idle_speed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--click-test"))
            a.click_test = true;
        else if (!strcmp(argv[i], "--hover-test"))
            a.hover_test = true;
        else if (!strcmp(argv[i], "--hi-style") && i + 1 < argc)
            a.hi_style = atoi(argv[++i]);
        else if (argv[i][0] != '-')
            a.pyramid = argv[i];
    }

    if (!vt_check_pyramid(a.pyramid))     /* before opening a window */
        return 1;
    if (!vtw_create(&a.w, "vt sphere viewer (C/GLES2)"))
        return 1;
    SDL_GL_SetSwapInterval(a.frames >= 0 ? 0 : 1);
    if (!vt_init(&a.vt, a.pyramid))
        return 1;
    load_manifest(&a);
    vt_lines_init(&a.lines);
    a.t0 = now_seconds();

    float aspect = (float)a.vt.virt_w / a.vt.virt_h;
    a.lon_half = fminf((float)M_PI, LAT_MAX * aspect);
    printf("band: lon +-%.1f deg, lat +-60 deg, %s view\n",
           a.lon_half * 180.0 / M_PI, a.inside ? "inside" : "outside");
    build_band_mesh(&a);

    a.fov = 75.0f;
    a.height = 12.0f;
    a.last_input = now_seconds();

    glEnable(GL_DEPTH_TEST);

    bool running = true;
    int frame = 0, fps_n = 0;
    double t0 = now_seconds(), fps_t = t0, prev_t = t0;
    while (running) {
        handle_events(&a, &running);
        double now = now_seconds();
        float dt = (float)(now - prev_t);
        if (dt > 0.1f) dt = 0.1f;
        prev_t = now;

        if (a.frames >= 0) {                    /* scripted sweep */
            if (frame < a.frames) {
                float k = (float)frame / (a.frames > 1 ? a.frames - 1 : 1);
                a.az = 0.7f * a.lon_half * sinf(2 * (float)M_PI * k);
                a.el = 0.5f * LAT_MAX * sinf(4 * (float)M_PI * k);
                if (a.inside)
                    a.fov = 75.0f * powf(a.end_fov / 75.0f, k);
                else
                    a.height = 12.0f * powf(a.end_height / 12.0f, k);
            } else {
                int extra = frame - a.frames;
                if (a.hover_test) {          /* hover the window centre */
                    int ww, wh;
                    SDL_GetWindowSize(a.w.win, &ww, &wh);
                    update_hover(&a, ww / 2.0f, wh / 2.0f);
                }
                if (a.click_test
                        && (extra == 20 || extra == 60 || extra == 100)) {
                    int ww, wh;
                    SDL_GetWindowSize(a.w.win, &ww, &wh);
                    click_center_zoom(&a, ww * 0.32f, wh * 0.38f);
                    if (a.anim.active)
                        printf("click -> %s  az %.1f  el %.1f  zoom %.2f\n",
                               a.last_click[0] ? a.last_click : "(point)",
                               a.anim.az1 * 180.0 / M_PI,
                               a.anim.el1 * 180.0 / M_PI, a.anim.zoom1);
                }
                int min_extra = a.click_test ? 120 : 0;
                if ((extra > min_extra && !a.anim.active
                        && vt_idle(&a.vt)) || extra > 600)
                    running = false;
            }
        } else {
            update_free_motion(&a, dt);
        }
        update_anim(&a);
        compute_mvp(&a);

        if (!a.freeze) {
            DrawCtx fb_ctx = {&a, a.vt.prog_fb};
            vtw_run_feedback(&a.w, &a.vt, a.vt.prog_fb, a.mvp,
                             draw_band, &fb_ctx);
            vt_pump_uploads(&a.vt);
        }

        glViewport(0, 0, a.w.draw_w, a.w.draw_h);
        glClearColor(0.05f, 0.06f, 0.08f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        DrawCtx main_ctx = {&a, a.vt.prog_main};
        vt_bind(&a.vt, a.vt.prog_main);
        glUniform1i(glGetUniformLocation(a.vt.prog_main, "uDebug"), a.debug);
        vt_highlight_uniforms(a.vt.prog_main, a.has_hover ? a.hover_uv : NULL,
                              a.hi_style, (float)(now_seconds() - a.t0),
                              a.w.dpi_scale);
        draw_band(&main_ctx);
        if (a.freeze)
            vt_lines_draw(&a.lines, a.mvp);

        if (!running && a.screenshot)
            vtw_screenshot(&a.w, a.screenshot);
        SDL_GL_SwapWindow(a.w.win);
        frame++;
        fps_n++;
        if (now - fps_t > 0.5) {
            char title[400];
            int resident = 0;
            for (int s = 0; s < a.vt.n_slots; s++)
                if (a.vt.slot_page[s].level >= 0)
                    resident++;
            snprintf(title, sizeof title,
                     "vt sphere viewer (C/GLES2) | %5.1f fps | resident "
                     "%d/%d | pending %d%s%s%s",
                     fps_n / (now - fps_t), resident, a.vt.n_slots,
                     a.vt.n_pending, a.freeze ? " | FROZEN" : "",
                     a.last_click[0] ? " | " : "", a.last_click);
            SDL_SetWindowTitle(a.w.win, title);
            fps_t = now;
            fps_n = 0;
        }
    }

    printf("frames: %d  time: %.1fs  tiles loaded: %d\n",
           frame, now_seconds() - t0, a.vt.loads_done);
    vt_destroy(&a.vt);
    SDL_GL_DeleteContext(a.w.ctx);
    SDL_DestroyWindow(a.w.win);
    SDL_Quit();
    return 0;
}
