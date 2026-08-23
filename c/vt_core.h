/*
    vt_core.h - shared virtual-texturing core for the C viewers.

    Port of the Python vt_viewer.py VT system to C / SDL2 / OpenGL ES 2.0
    (via ANGLE on macOS, see github.com/erik-larsen/opengl-for-mac).

    GLES2 differences from the GL3.3 Python version:
      * no texture arrays and no texelFetch: the per-level page tables are
        packed side by side into one 2D texture, sampled NEAREST with
        half-texel-centered normalized coords; the per-level rect comes
        from a uniform array indexed with a constant-index loop.
      * no integer/bit ops in GLSL ES 1.00: the feedback pass encodes
        (pageX:12, pageY:12, level:4) with mod()/floor() arithmetic.
      * derivatives come from GL_OES_standard_derivatives.
*/
#ifndef VT_CORE_H
#define VT_CORE_H

#include <stdbool.h>
#include <SDL.h>
#include <SDL_opengles2.h>

#define VT_ATLAS_SIZE 4096
#define VT_MAX_LEVELS 16
#define VT_MAX_UPLOADS_PER_FRAME 24

typedef struct {
    int level, tx, ty;
} VtPage;

typedef struct {
    /* pyramid metadata */
    char dir[1024];
    char format[8];          /* "png" or "jpg" */
    int virt_w, virt_h;
    int tile, border, slot_px;
    int levels;
    int nx[VT_MAX_LEVELS], ny[VT_MAX_LEVELS];

    /* page table packing: level l occupies rect (0, off_y[l], nx, ny)
       inside a table_w x table_h texture */
    int table_off_y[VT_MAX_LEVELS];
    int table_w, table_h;
    GLuint table_tex, atlas_tex;
    unsigned char *table_staging;      /* table_w * table_h * 4 */

    /* residency */
    int slots_per_row, n_slots;
    int *own[VT_MAX_LEVELS];           /* per level: ny*nx, slot or -1 */
    VtPage *slot_page;                 /* n_slots; level<0 = free */
    unsigned *slot_lastuse;            /* frame number */
    unsigned frame;
    bool dirty;
    int loads_done;
    int n_pending;           /* requested but not yet uploaded/failed */

    /* streaming (single loader thread) */
    unsigned char *pending[VT_MAX_LEVELS];   /* per level: ny*nx flags */
    SDL_mutex *req_mtx;
    SDL_cond *req_cond;
    VtPage *req_q;                     /* unsorted; loader picks coarsest */
    int req_count, req_cap;
    SDL_mutex *ready_mtx;
    struct VtReady { VtPage page; unsigned char *pixels; } *ready_q;
    int ready_count, ready_cap;
    SDL_Thread *loader;
    bool quit;

    /* GL programs (built here so both viewers share the shaders) */
    GLuint prog_main, prog_fb;
} VtSystem;

extern const char *vt_argv0;   /* program name, used in error hints */

/* True if <dir>/meta.json exists; otherwise prints build instructions.
   Call before opening a window so a missing pyramid doesn't flash one. */
bool vt_check_pyramid(const char *pyramid_dir);

/* lifecycle */
bool vt_init(VtSystem *vt, const char *pyramid_dir);
void vt_destroy(VtSystem *vt);

/* per frame: feed the working set decoded from the feedback buffer
   (RGBA bytes, n pixels), then pump uploads */
void vt_request_from_feedback(VtSystem *vt, const unsigned char *rgba, int n);
void vt_pump_uploads(VtSystem *vt);
bool vt_idle(VtSystem *vt);   /* nothing pending and nothing ready */
void vt_evict_unused(VtSystem *vt);  /* drop pages the latest feedback
                                        pass did not touch (freeze demo) */

/* bind textures + set the VT uniforms on a program */
void vt_bind(VtSystem *vt, GLuint prog);

/* shader helpers */
GLuint vt_compile_program(const char *vs, const char *fs);

/* the fragment shaders, exposed so a viewer can pair them with its own
   vertex shader (the sphere viewer generates and morphs its geometry) */
extern const char *vt_fs_main_src;
extern const char *vt_fs_feedback_src;

/* yellow frustum wireframe (freeze-LOD visualization) */
typedef struct {
    GLuint prog, vbo;
    bool valid;
} VtLines;
void vt_lines_init(VtLines *fl);
void vt_lines_capture(VtLines *fl, const float *mvp, const float eye[3],
                      float near_dist, float far_dist);
void vt_lines_draw(VtLines *fl, const float *mvp_now);

/* ---- mosaic manifest (layout.json): hover + click-to-center ---- */
typedef struct { char name[256]; int x, y, w, h; } VtRect;
typedef struct { VtRect *rects; int n; } VtManifest;

/* Loads <pyramid>_layout.json (or explicit_path). Returns false if absent. */
bool vt_manifest_load(VtManifest *m, const char *pyramid_dir,
                      const char *explicit_path);
/* Index of the image covering (u,v), or -1. Fills out_uv = u0,v0,u1,v1. */
int vt_manifest_rect_at(const VtManifest *m, float u, float v,
                        int virt_w, int virt_h, float out_uv[4]);
/* Upload the highlight uniforms; pass rect_uv = NULL for "no hover". */
void vt_highlight_uniforms(GLuint prog, const float rect_uv[4], int style,
                           float t, float dpi_scale);

/* ---- small matrix / vector helpers (row-major float[16]) ---- */
void mat4_perspective(float *m, float fovy_deg, float aspect,
                      float near, float far);
void mat4_look_at(float *m, const float eye[3], const float target[3],
                  const float up[3]);
void mat4_mul(float *out, const float *a, const float *b);
bool mat4_invert(float *out, const float *m);
void mat4_upload(GLint loc, const float *m);   /* transposes for GLES2 */

/* ---- shared viewer scaffolding ---- */
typedef struct {
    SDL_Window *win;
    SDL_GLContext ctx;
    int draw_w, draw_h;      /* drawable (pixel) size */
    float dpi_scale;         /* drawable / window-point ratio */
    GLuint fbo, fb_tex;
    int fb_w, fb_h;          /* feedback buffer size */
    unsigned char *fb_pixels;
} VtWindow;

bool vtw_create(VtWindow *w, const char *title);
void vtw_update_sizes(VtWindow *w);           /* after resize events */
void vtw_run_feedback(VtWindow *w, VtSystem *vt, GLuint prog_fb,
                      const float *mvp,
                      void (*draw_geom)(void *), void *geom_arg);
void vtw_screenshot(VtWindow *w, const char *path);

double now_seconds(void);
float smoothstep01(float u);

#endif
