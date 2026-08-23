/*
    vt_viewer.c - orbit a giant tiled image pyramid with virtual texturing.
    C / SDL2 / GLES2 port of vt_viewer.py.

    Usage: vt_viewer [--pyramid DIR] [--debug-lod]
                     [--frames N] [--screenshot OUT.png]
                     [--end-dist D] [--end-uv U,V] [--click-test]

    Controls: drag = orbit | right-drag/shift-drag = pan | scroll = zoom
              hover = outline the mosaic image under the pointer
              click = center the view on that image (or on the point)
              H = cycle highlight style
              F = freeze/unfreeze streaming (explore the frozen LOD state)
              L = LOD debug overlay | R = reset camera | ESC = quit
*/
#include "vt_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    VtWindow w;
    VtSystem vt;
    GLuint vbo, ebo;
    float half_x, half_z;

    float yaw, pitch, dist;
    float target[3];
    int debug, freeze;
    VtLines lines;

    VtManifest manifest;
    float hover_uv[4];
    bool has_hover;
    int hi_style;
    double t0;

    bool dragging, drag_pan;
    float drag_x, drag_y, press_x, press_y;
    bool has_press;

    struct {
        bool active;
        double t0, dur;
        float target0[3], target1[3], dist0, dist1;
    } anim;

    float mvp[16];

    /* args */
    const char *pyramid, *screenshot;
    int frames;
    float end_dist, end_u, end_v;
    bool click_test, debug_lod, freeze_test, hover_test;
    const char *manifest_path;
} App;

static void compute_mvp(App *a)
{
    float eye[3] = {
        a->target[0] + a->dist * cosf(a->pitch) * sinf(a->yaw),
        a->target[1] + a->dist * sinf(a->pitch),
        a->target[2] + a->dist * cosf(a->pitch) * cosf(a->yaw),
    };
    float up[3] = {0, 1, 0};
    float view[16], proj[16];
    mat4_look_at(view, eye, a->target, up);
    mat4_perspective(proj, 55.0f, (float)a->w.draw_w / a->w.draw_h,
                     0.02f, 400.0f);
    mat4_mul(a->mvp, proj, view);
}

static void mat4_xform(const float *m, const float *v, float *out)
{
    for (int r = 0; r < 4; r++)
        out[r] = m[r*4+0]*v[0] + m[r*4+1]*v[1]
               + m[r*4+2]*v[2] + m[r*4+3]*v[3];
}

/* unproject window coords onto the ground plane (y=0) */
static bool cursor_hit(App *a, float cx, float cy, float out[3])
{
    int ww, wh;
    SDL_GetWindowSize(a->w.win, &ww, &wh);
    if (!ww || !wh)
        return false;
    float inv[16];
    compute_mvp(a);
    if (!mat4_invert(inv, a->mvp))
        return false;
    float ndc_x = 2.0f * cx / ww - 1.0f, ndc_y = 1.0f - 2.0f * cy / wh;
    float pn[4] = {ndc_x, ndc_y, -1, 1}, pf[4] = {ndc_x, ndc_y, 1, 1};
    float p0[4], p1[4];
    mat4_xform(inv, pn, p0);
    mat4_xform(inv, pf, p1);
    for (int i = 0; i < 3; i++) {
        p0[i] /= p0[3];
        p1[i] /= p1[3];
    }
    float d[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    if (fabsf(d[1]) < 1e-9f)
        return false;
    float t = -p0[1] / d[1];
    if (t <= 0)
        return false;
    out[0] = p0[0] + t * d[0];
    out[1] = 0;
    out[2] = p0[2] + t * d[2];
    return fabsf(out[0]) <= a->half_x && fabsf(out[2]) <= a->half_z;
}

static float clampf(float v, float lo, float hi)
{ return v < lo ? lo : v > hi ? hi : v; }

/* eased glide of the orbit target; the camera distance is held */
static void glide_to(App *a, float x, float z, float duration)
{
    memcpy(a->anim.target0, a->target, sizeof a->anim.target0);
    a->anim.target1[0] = clampf(x, -a->half_x, a->half_x);
    a->anim.target1[1] = 0.0f;
    a->anim.target1[2] = clampf(z, -a->half_z, a->half_z);
    a->anim.dist0 = a->dist;
    a->anim.dist1 = a->dist;              /* centering only, no zoom */
    a->anim.t0 = now_seconds();
    a->anim.dur = duration;
    a->anim.active = true;
}

/* centre the view on the clicked point */
static void center_on_point(App *a, float cx, float cy)
{
    float hit[3];
    if (cursor_hit(a, cx, cy, hit))
        glide_to(a, hit[0], hit[2], 0.4f);
}

/* cursor -> plane -> mosaic uv -> hovered image rect */
static void update_hover(App *a, float cx, float cy)
{
    float hit[3];
    a->has_hover = false;
    if (!a->manifest.rects || !cursor_hit(a, cx, cy, hit))
        return;
    float u = (hit[0] + a->half_x) / (2 * a->half_x);
    float v = (hit[2] + a->half_z) / (2 * a->half_z);
    a->has_hover = vt_manifest_rect_at(&a->manifest, u, v, a->vt.virt_w,
                                       a->vt.virt_h, a->hover_uv) >= 0;
}

/* the camera moves on its own (click-to-centre glides), so what sits
   under a stationary pointer changes without any mouse event */
static void refresh_hover(App *a)
{
    if (!a->manifest.rects || SDL_GetMouseFocus() != a->w.win)
        return;
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    update_hover(a, (float)mx, (float)my);
}

/* centre the view on the hovered image */
static void center_on_hover(App *a, float duration)
{
    float cu = (a->hover_uv[0] + a->hover_uv[2]) * 0.5f;
    float cv = (a->hover_uv[1] + a->hover_uv[3]) * 0.5f;
    glide_to(a, (cu - 0.5f) * 2 * a->half_x,
                (cv - 0.5f) * 2 * a->half_z, duration);
}

static void update_anim(App *a)
{
    if (!a->anim.active)
        return;
    float u = (float)((now_seconds() - a->anim.t0) / a->anim.dur);
    float e = smoothstep01(u);
    for (int i = 0; i < 3; i++)
        a->target[i] = a->anim.target0[i]
                     + (a->anim.target1[i] - a->anim.target0[i]) * e;
    a->dist = a->anim.dist0 + (a->anim.dist1 - a->anim.dist0) * e;
    if (u >= 1.0f)
        a->anim.active = false;
}

static void toggle_freeze(App *a)
{
    a->freeze ^= 1;
    if (a->freeze) {                /* pin the frustum that chose the
                                       currently resident working set */
        compute_mvp(a);
        float eye[3] = {
            a->target[0] + a->dist * cosf(a->pitch) * sinf(a->yaw),
            a->target[1] + a->dist * sinf(a->pitch),
            a->target[2] + a->dist * cosf(a->pitch) * cosf(a->yaw),
        };
        vt_lines_capture(&a->lines, a->mvp, eye, 0.02f,
                         fmaxf(2.5f * a->dist, 1.0f));
        vt_evict_unused(&a->vt);    /* keep only this view's pages */
    }
}

typedef struct {
    App *a;
    GLuint prog;
} DrawCtx;

static void draw_plane(void *arg)
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
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
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
                a->yaw = 0.6f; a->pitch = 0.9f; a->dist = 14.0f;
                memset(a->target, 0, sizeof a->target);
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            a->anim.active = false;
            a->dragging = true;
            a->drag_pan = ev.button.button == SDL_BUTTON_RIGHT
                       || (SDL_GetModState() & KMOD_SHIFT);
            a->drag_x = a->press_x = ev.button.x;
            a->drag_y = a->press_y = ev.button.y;
            a->has_press = true;
            break;
        case SDL_MOUSEBUTTONUP:
            if (ev.button.button == SDL_BUTTON_LEFT && a->has_press
                    && fabsf(ev.button.x - a->press_x) < 4
                    && fabsf(ev.button.y - a->press_y) < 4
                    && !(SDL_GetModState() & KMOD_SHIFT)) {
                update_hover(a, ev.button.x, ev.button.y);
                if (a->has_hover)
                    center_on_hover(a, 0.45f);
                else
                    center_on_point(a, ev.button.x, ev.button.y);
            }
            a->dragging = false;
            a->has_press = false;
            break;
        case SDL_MOUSEMOTION:
            if (!a->dragging)
                update_hover(a, ev.motion.x, ev.motion.y);
            if (a->dragging) {
                float dx = ev.motion.x - a->drag_x;
                float dy = ev.motion.y - a->drag_y;
                a->drag_x = ev.motion.x;
                a->drag_y = ev.motion.y;
                if (a->drag_pan) {          /* image follows the cursor */
                    float s = a->dist * 0.0012f;
                    a->target[0] += (-dx * cosf(a->yaw)
                                    - dy * sinf(a->yaw)) * s;
                    a->target[2] += (dx * sinf(a->yaw)
                                    - dy * cosf(a->yaw)) * s;
                    a->target[0] = clampf(a->target[0],
                                          -a->half_x, a->half_x);
                    a->target[2] = clampf(a->target[2],
                                          -a->half_z, a->half_z);
                } else {
                    a->yaw -= dx * 0.005f;
                    a->pitch = clampf(a->pitch + dy * 0.005f,
                                      0.05f, 1.52f);
                }
            }
            break;
        case SDL_MOUSEWHEEL:
            a->anim.active = false;
            a->dist = clampf(a->dist * powf(0.92f, ev.wheel.preciseY),
                             0.05f, 60.0f);
            break;
        case SDL_WINDOWEVENT:
            if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                vtw_update_sizes(&a->w);
            break;
        }
    }
}

int main(int argc, char **argv)
{
    App a;
    memset(&a, 0, sizeof a);
    vt_argv0 = argv[0];
    a.pyramid = "../pics/test_image_16k_pyramid";
    a.frames = -1;
    a.hi_style = 2;
    a.end_dist = 0.12f;
    a.end_u = 0.625f;
    a.end_v = 0.5625f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pyramid") && i + 1 < argc)
            a.pyramid = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            a.frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc)
            a.screenshot = argv[++i];
        else if (!strcmp(argv[i], "--end-dist") && i + 1 < argc)
            a.end_dist = atof(argv[++i]);
        else if (!strcmp(argv[i], "--end-uv") && i + 1 < argc)
            sscanf(argv[++i], "%f,%f", &a.end_u, &a.end_v);
        else if (!strcmp(argv[i], "--click-test"))
            a.click_test = true;
        else if (!strcmp(argv[i], "--freeze-test"))
            a.freeze_test = true;
        else if (!strcmp(argv[i], "--hover-test"))
            a.hover_test = true;
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc)
            a.manifest_path = argv[++i];
        else if (!strcmp(argv[i], "--hi-style") && i + 1 < argc)
            a.hi_style = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--debug-lod"))
            a.debug_lod = true;
        else if (argv[i][0] != '-')
            a.pyramid = argv[i];
    }

    if (!vt_check_pyramid(a.pyramid))     /* before opening a window */
        return 1;
    if (!vtw_create(&a.w, "vt viewer (C/GLES2)"))
        return 1;
    SDL_GL_SetSwapInterval(a.frames >= 0 ? 0 : 1);
    if (!vt_init(&a.vt, a.pyramid))
        return 1;
    vt_lines_init(&a.lines);
    vt_manifest_load(&a.manifest, a.pyramid, a.manifest_path);
    a.t0 = now_seconds();

    a.half_x = 8.0f;
    a.half_z = 8.0f * a.vt.virt_h / a.vt.virt_w;
    a.yaw = 0.6f; a.pitch = 0.9f; a.dist = 14.0f;
    a.debug = a.debug_lod ? 1 : 0;

    float hx = a.half_x, hz = a.half_z;
    float verts[] = {
        -hx, 0, -hz, 0, 0,   hx, 0, -hz, 1, 0,
         hx, 0,  hz, 1, 1,  -hx, 0,  hz, 0, 1,
    };
    unsigned short idx[] = {0, 1, 2, 0, 2, 3};
    glGenBuffers(1, &a.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, a.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glGenBuffers(1, &a.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, a.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glEnable(GL_DEPTH_TEST);

    bool running = true;
    int frame = 0, fps_n = 0;
    double t0 = now_seconds(), fps_t = t0;
    while (running) {
        handle_events(&a, &running);

        if (a.frames >= 0) {                    /* scripted orbit */
            if (frame < a.frames) {
                float k = (float)frame / (a.frames > 1 ? a.frames - 1 : 1);
                a.yaw = 0.5f + 2.0f * k;
                a.pitch = 0.25f + 0.9f * k;
                a.dist = 18.0f * powf(a.end_dist / 18.0f, k);
                a.target[0] = (a.end_u - 0.5f) * 2 * a.half_x * k;
                a.target[2] = (a.end_v - 0.5f) * 2 * a.half_z * k;
            } else if (a.freeze_test) {
                /* freeze at the end pose, then pull back to show the
                   pinned frustum + frozen LOD state */
                int extra = frame - a.frames;
                if (extra == 60)
                    toggle_freeze(&a);
                else if (extra > 60) {
                    a.dist = fminf(a.dist * 1.015f, 60.0f);
                    a.yaw += 0.008f;
                    a.pitch = fminf(a.pitch + 0.002f, 1.5f);
                }
                if (extra > 200)
                    running = false;
            } else {
                int extra = frame - a.frames;
                if (a.hover_test) {          /* hover the window centre */
                    int ww, wh;
                    SDL_GetWindowSize(a.w.win, &ww, &wh);
                    update_hover(&a, ww / 2.0f, wh / 2.0f);
                }
                if (a.click_test
                        && (extra == 20 || extra == 45 || extra == 70)) {
                    int ww, wh;
                    SDL_GetWindowSize(a.w.win, &ww, &wh);
                    center_on_point(&a, ww / 2.0f, wh / 2.0f);
                    if (a.anim.active)
                        printf("click -> center (%.2f, %.2f)\n",
                               a.anim.target1[0], a.anim.target1[2]);
                }
                int min_extra = a.click_test ? 90 : 0;
                if ((extra > min_extra && !a.anim.active
                        && vt_idle(&a.vt)) || extra > 600)
                    running = false;
            }
        }
        update_anim(&a);
        if (a.frames < 0)
            refresh_hover(&a);
        compute_mvp(&a);

        if (!a.freeze) {
            DrawCtx fb_ctx = {&a, a.vt.prog_fb};
            vtw_run_feedback(&a.w, &a.vt, a.vt.prog_fb, a.mvp,
                             draw_plane, &fb_ctx);
            vt_pump_uploads(&a.vt);
        }

        glViewport(0, 0, a.w.draw_w, a.w.draw_h);
        glClearColor(0.08f, 0.09f, 0.11f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        DrawCtx main_ctx = {&a, a.vt.prog_main};
        vt_bind(&a.vt, a.vt.prog_main);
        glUniform1i(glGetUniformLocation(a.vt.prog_main, "uDebug"), a.debug);
        vt_highlight_uniforms(a.vt.prog_main, a.has_hover ? a.hover_uv : NULL,
                              a.hi_style, (float)(now_seconds() - a.t0),
                              a.w.dpi_scale);
        draw_plane(&main_ctx);
        if (a.freeze)
            vt_lines_draw(&a.lines, a.mvp);

        if (!running && a.screenshot)
            vtw_screenshot(&a.w, a.screenshot);
        SDL_GL_SwapWindow(a.w.win);
        frame++;
        fps_n++;
        double now = now_seconds();
        if (now - fps_t > 0.5) {
            char title[256];
            int resident = 0;
            for (int s = 0; s < a.vt.n_slots; s++)
                if (a.vt.slot_page[s].level >= 0)
                    resident++;
            snprintf(title, sizeof title,
                     "vt viewer (C/GLES2) | %5.1f fps | resident %d/%d | "
                     "pending %d | loaded %d%s",
                     fps_n / (now - fps_t), resident, a.vt.n_slots,
                     a.vt.n_pending, a.vt.loads_done,
                     a.freeze ? " | FROZEN" : "");
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
