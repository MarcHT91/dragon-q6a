// LD_PRELOAD shim: add explicit DRM_FORMAT_MOD_LINEAR to EGL dma-buf imports
// that carry no modifier. Kodi drops the modifier when it is LINEAR, and Mesa
// (freedreno) refuses implicit-modifier imports of iris (qcom_iris) dmabufs.
// Build: gcc -O2 -shared -fPIC -o libegl_linear_shim.so shim/egl_linear_shim.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tests/egl_min.h"
typedef intptr_t EGLAttrib; typedef void *EGLImage;
typedef void (*__eglMustCastToProperFunctionPointerType)(void);

static PFNEGLCREATEIMAGEKHRPROC real_khr;
static EGLImage (*real_core)(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLAttrib *);
static __eglMustCastToProperFunctionPointerType (*real_gpa)(const char *);
static int dbg = -1;

static const EGLint mod_lo[3] = {EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                                 EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT};
static const EGLint mod_hi[3] = {EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                                 EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT};
static const EGLint plane_fd[3] = {EGL_DMA_BUF_PLANE0_FD_EXT, EGL_DMA_BUF_PLANE1_FD_EXT,
                                   EGL_DMA_BUF_PLANE2_FD_EXT};

static void init_dbg(void) { if (dbg < 0) dbg = getenv("EGL_SHIM_DEBUG") != NULL; }

// EGL_SHIM_LIMITED_OUT=1: the board's HDMI (DP -> RA620) sends RGB 0-255 without telling the
// TV, which then reads it as 16-235. Kodi's "limited colour range" setting fixes the GUI but
// its DRM PRIME EGL renderer ignores it for video. Telling Mesa the (narrow) YUV is FULL range
// makes it pass luma through untouched, so video black/white land on 16/235 like the GUI.
// Chroma comes out ~2% more saturated than an exact limited conversion (224/255 vs 219/255).
#define EGL_SAMPLE_RANGE_HINT_EXT 0x327C
#define EGL_YUV_FULL_RANGE_EXT 0x3282
#define EGL_YUV_NARROW_RANGE_EXT 0x3283
static int limited_out = -1;
static int want_limited(void) {
  if (limited_out < 0) { const char *e = getenv("EGL_SHIM_LIMITED_OUT"); limited_out = e && *e == '1'; }
  return limited_out;
}

// Generic over attrib type (EGLint or EGLAttrib). Returns malloc'd list or NULL if untouched.
#define PATCH(T, attrs, out)                                                        \
  do {                                                                              \
    int n = 0, has_mod = 0, narrow = 0, planes[3] = {0, 0, 0};                      \
    for (; attrs[n] != EGL_NONE; n += 2) {                                          \
      for (int p = 0; p < 3; p++) {                                                 \
        if (attrs[n] == mod_lo[p] || attrs[n] == mod_hi[p]) has_mod = 1;            \
        if (attrs[n] == plane_fd[p]) planes[p] = 1;                                 \
      }                                                                             \
      if (attrs[n] == EGL_SAMPLE_RANGE_HINT_EXT && attrs[n + 1] == EGL_YUV_NARROW_RANGE_EXT) narrow = 1; \
    }                                                                               \
    int relabel = narrow && want_limited();                                         \
    if (has_mod && !relabel) break;                                                 \
    T *nl_ = malloc(sizeof(T) * (n + 13));                                            \
    memcpy(nl_, attrs, sizeof(T) * n);                                                \
    int k = n;                                                                      \
    if (relabel)                                                                    \
      for (int i = 0; i < n; i += 2)                                                \
        if (nl_[i] == EGL_SAMPLE_RANGE_HINT_EXT) nl_[i + 1] = EGL_YUV_FULL_RANGE_EXT; \
    if (!has_mod)                                                                   \
      for (int p = 0; p < 3; p++)                                                   \
        if (planes[p]) { nl_[k++] = mod_lo[p]; nl_[k++] = 0; nl_[k++] = mod_hi[p]; nl_[k++] = 0; } \
    nl_[k] = EGL_NONE;                                                              \
    out = nl_;                                                                      \
  } while (0)

static EGLImageKHR shim_khr(EGLDisplay d, EGLContext c, EGLenum t, EGLClientBuffer b, const EGLint *a) {
  EGLint *l = NULL;
  if (t == EGL_LINUX_DMA_BUF_EXT && a) PATCH(EGLint, a, l);
  EGLImageKHR img = real_khr(d, c, t, b, l ? l : a);
  init_dbg();
  if (dbg) fprintf(stderr, "egl_shim: CreateImageKHR patched=%d -> %p err=0x%x\n", !!l, img,
                   img ? 0x3000 : (EGLint)((EGLint (*)(void))dlsym(RTLD_NEXT, "eglGetError"))());
  free(l);
  return img;
}

static EGLImage shim_core(EGLDisplay d, EGLContext c, EGLenum t, EGLClientBuffer b, const EGLAttrib *a) {
  EGLAttrib *l = NULL;
  if (t == EGL_LINUX_DMA_BUF_EXT && a) PATCH(EGLAttrib, a, l);
  EGLImage img = real_core(d, c, t, b, l ? l : a);
  free(l);
  return img;
}

__eglMustCastToProperFunctionPointerType eglGetProcAddress_shim(const char *name) __asm__("eglGetProcAddress");
__eglMustCastToProperFunctionPointerType eglGetProcAddress_shim(const char *name) {
  if (!real_gpa) real_gpa = dlsym(RTLD_NEXT, "eglGetProcAddress");
  __eglMustCastToProperFunctionPointerType f = real_gpa(name);
  if (!f || !name) return f;
  if (!strcmp(name, "eglCreateImageKHR")) { real_khr = (PFNEGLCREATEIMAGEKHRPROC)f; return (void (*)(void))shim_khr; }
  if (!strcmp(name, "eglCreateImage")) { real_core = (void *)f; return (void (*)(void))shim_core; }
  return f;
}

EGLImage eglCreateImage(EGLDisplay d, EGLContext c, EGLenum t, EGLClientBuffer b, const EGLAttrib *a) {
  if (!real_core) real_core = dlsym(RTLD_NEXT, "eglCreateImage");
  return shim_core(d, c, t, b, a);
}

// ---- GBM/EGL swap tracing: ring of last 64 events, dumped on swap failure ----
#include <time.h>
struct gbm_surface; struct gbm_bo;
static int locked_n;
static int pending_front;            // a swap succeeded and nobody locked its front buffer yet
static struct gbm_surface *last_surf; // surface seen in lock/release
static struct gbm_bo *(*real_lock)(struct gbm_surface *);
static void (*real_release)(struct gbm_surface *, struct gbm_bo *);
static unsigned long frame_n;
static char ring[64][96]; static unsigned ring_i;
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#define EV(...) snprintf(ring[ring_i++ % 64], 96, __VA_ARGS__)
static void dump(void) {
  unsigned start = ring_i > 64 ? ring_i - 64 : 0;
  for (unsigned i = start; i < ring_i; i++) fprintf(stderr, "  ev: %s\n", ring[i % 64]);
}

struct gbm_bo *gbm_surface_lock_front_buffer(struct gbm_surface *s) {
  if (!real_lock) real_lock = dlsym(RTLD_NEXT, "gbm_surface_lock_front_buffer");
  struct gbm_bo *bo = real_lock(s);
  last_surf = s; pending_front = 0;
  if (bo) locked_n++;
  EV("[%.1f] lock s=%p -> %p locked=%d", now_ms(), (void *)s, (void *)bo, locked_n);
  return bo;
}

void gbm_surface_release_buffer(struct gbm_surface *s, struct gbm_bo *bo) {
  if (!real_release) real_release = dlsym(RTLD_NEXT, "gbm_surface_release_buffer");
  real_release(s, bo);
  last_surf = s;
  locked_n--;
  EV("[%.1f] release s=%p %p locked=%d", now_ms(), (void *)s, (void *)bo, locked_n);
}

int gbm_surface_has_free_buffers(struct gbm_surface *s) {
  static int (*real)(struct gbm_surface *);
  if (!real) real = dlsym(RTLD_NEXT, "gbm_surface_has_free_buffers");
  int r = real(s);
  EV("[%.1f] has_free=%d", now_ms(), r);
  return r;
}

unsigned eglSwapBuffers(void *dpy, void *surf) {
  static unsigned (*real)(void *, void *);
  static EGLint (*geterr)(void);
  if (!real) { real = dlsym(RTLD_NEXT, "eglSwapBuffers"); geterr = dlsym(RTLD_NEXT, "eglGetError"); }
  // Kodi (GBM + atomic DRM) swaps once in SetFullScreen without locking the result.
  // Mesa then keeps that buffer as 'current'; drop it here so the next swap has a free buffer.
  if (pending_front && last_surf && real_lock && real_release && !getenv("EGL_SHIM_NOFIX")) {
    struct gbm_bo *bo = real_lock(last_surf);
    if (bo) real_release(last_surf, bo);
    EV("[%.1f] FIX: dropped unconsumed front %p", now_ms(), (void *)bo);
    fprintf(stderr, "[%.1f] egl_shim: dropped unconsumed front buffer %p (frame %lu)\n", now_ms(), (void *)bo, frame_n + 1);
  }
  unsigned ok = real(dpy, surf);
  frame_n++;
  if (ok) pending_front = 1;
  { // EGL_SHIM_STATS=1: write swaps/s every 5 s
    static int stats = -1; static double t0; static unsigned long f0;
    if (stats < 0) stats = getenv("EGL_SHIM_STATS") != NULL;
    if (stats) { double t = now_ms(); if (!t0) { t0 = t; f0 = frame_n; }
      else if (t - t0 >= 5000) { fprintf(stderr, "[%.1f] egl_shim: %.1f swaps/s\n", t, (frame_n - f0) * 1000.0 / (t - t0)); t0 = t; f0 = frame_n; } }
  }
  EV("[%.1f] swap surf=%p ok=%u frame=%lu locked=%d", now_ms(), surf, ok, frame_n, locked_n);
  if (!ok) {
    EGLint e = geterr();
    fprintf(stderr, "[%.1f] egl_shim: eglSwapBuffers FAILED err=0x%x frame=%lu locked=%d\n", now_ms(), e, frame_n, locked_n);
    dump();
    // put the error back so Kodi's own log shows the real code
    ok = real(dpy, surf);
    fprintf(stderr, "  retry ok=%u\n", ok);
  }
  return ok;
}
