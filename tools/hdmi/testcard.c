// testcard: show a range test card on the active CRTC for N seconds (needs DRM master: stop Kodi first).
//   top third:    16 grey steps, 0..255 (step 17)
//   middle third: 16 near-black steps, 0..60 (step 4)
//   bottom third: colour bars (white, yellow, cyan, green, magenta, red, blue, black)
// Reading it: if the TV shows the 5 leftmost near-black steps (0..16) as one black block, it is
// treating the signal as limited range (16-235) while the board sends full range (0-255).
// Build: gcc -O2 -o testcard tools/hdmi/testcard.c -I/usr/include/libdrm -ldrm
// Usage: testcard [seconds=60] [/dev/dri/card1]
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static uint32_t prop_id(int fd, uint32_t obj, uint32_t type, const char *name, uint64_t *val) {
  drmModeObjectProperties *p = drmModeObjectGetProperties(fd, obj, type);
  uint32_t id = 0;
  for (uint32_t i = 0; p && i < p->count_props && !id; i++) {
    drmModePropertyRes *r = drmModeGetProperty(fd, p->props[i]);
    if (r && !strcmp(r->name, name)) { id = r->prop_id; if (val) *val = p->prop_values[i]; }
    drmModeFreeProperty(r);
  }
  drmModeFreeObjectProperties(p);
  return id;
}

int main(int argc, char **argv) {
  int secs = argc > 1 ? atoi(argv[1]) : 60;
  const char *dev = argc > 2 ? argv[2] : "/dev/dri/card1";
  int fd = open(dev, O_RDWR);
  if (fd < 0) { perror(dev); return 1; }
  drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  if (drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1)) { perror("atomic"); return 1; }

  drmModeRes *res = drmModeGetResources(fd);
  drmModeCrtc *crtc = NULL;
  for (int i = 0; res && i < res->count_crtcs && !crtc; i++) {
    drmModeCrtc *c = drmModeGetCrtc(fd, res->crtcs[i]);
    if (c && c->mode_valid) crtc = c; else drmModeFreeCrtc(c);
  }
  if (!crtc) { fprintf(stderr, "no active CRTC\n"); return 1; }
  int W = crtc->mode.hdisplay, H = crtc->mode.vdisplay;

  struct drm_mode_create_dumb cd = {.width = W, .height = H, .bpp = 32};
  if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd)) { perror("create dumb"); return 1; }
  uint32_t fb, handles[4] = {cd.handle}, pitches[4] = {cd.pitch}, offsets[4] = {0};
  if (drmModeAddFB2(fd, W, H, DRM_FORMAT_XRGB8888, handles, pitches, offsets, &fb, 0)) { perror("addfb"); return 1; }
  struct drm_mode_map_dumb md = {.handle = cd.handle};
  drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md);
  uint8_t *map = mmap(0, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, md.offset);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }

  static const uint32_t bars[8] = {0xffffff, 0xffff00, 0x00ffff, 0x00ff00, 0xff00ff, 0xff0000, 0x0000ff, 0};
  for (int y = 0; y < H; y++) {
    uint32_t *row = (uint32_t *)(map + (size_t)y * cd.pitch);
    for (int x = 0; x < W; x++) {
      int step = x * 16 / W;
      if (y < H / 3) row[x] = (uint32_t)(step * 17) * 0x010101u;
      else if (y < 2 * H / 3) row[x] = (uint32_t)(step * 4) * 0x010101u;
      else row[x] = bars[x * 8 / W];
    }
  }

  // primary plane of this CRTC
  drmModePlaneRes *pr = drmModeGetPlaneResources(fd);
  uint32_t plane = 0, crtc_idx = 0;
  for (int i = 0; i < res->count_crtcs; i++) if (res->crtcs[i] == crtc->crtc_id) crtc_idx = i;
  for (uint32_t i = 0; pr && i < pr->count_planes && !plane; i++) {
    drmModePlane *p = drmModeGetPlane(fd, pr->planes[i]);
    uint64_t type = 0;
    prop_id(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "type", &type);
    if (type == DRM_PLANE_TYPE_PRIMARY && (p->possible_crtcs & (1u << crtc_idx))) plane = p->plane_id;
    drmModeFreePlane(p);
  }
  if (!plane) { fprintf(stderr, "no primary plane\n"); return 1; }

  const char *names[] = {"FB_ID", "CRTC_ID", "SRC_X", "SRC_Y", "SRC_W", "SRC_H", "CRTC_X", "CRTC_Y", "CRTC_W", "CRTC_H"};
  uint64_t vals[] = {fb, crtc->crtc_id, 0, 0, (uint64_t)W << 16, (uint64_t)H << 16, 0, 0, W, H};
  drmModeAtomicReq *req = drmModeAtomicAlloc();
  for (int k = 0; k < 10; k++)
    drmModeAtomicAddProperty(req, plane, prop_id(fd, plane, DRM_MODE_OBJECT_PLANE, names[k], NULL), vals[k]);
  if (drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL)) {
    fprintf(stderr, "commit: %s%s\n", strerror(errno), errno == EACCES ? " (not DRM master: stop Kodi first)" : "");
    return 1;
  }
  printf("test card on %s, %dx%d, for %d s\n", crtc->mode.name, W, H, secs);
  fflush(stdout);
  sleep(secs);
  return 0;
}
