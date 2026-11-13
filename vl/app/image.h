// Image loading (stb_image) and the LFM3-VL preprocessing: bicubic resize exactly like Pillow's Image.resize
// (BICUBIC, a = -0.5, antialiased support when downscaling, 8-bit fixed point, horizontal pass then vertical),
// then (x/255 - 0.5)/0.5, planar [3, H, W] float.
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "third_party/stb_image.h"

struct RGB8 { int w = 0, h = 0; std::vector<uint8_t> px; };   // interleaved RGB

inline RGB8 load_rgb(const std::string& path) {
    int w, h, n;
    uint8_t* d = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!d) throw std::runtime_error("cannot read image " + path + ": " + stbi_failure_reason());
    RGB8 im; im.w = w; im.h = h; im.px.assign(d, d + (size_t)w * h * 3);
    stbi_image_free(d);
    return im;
}

namespace pil {
constexpr int PRECISION_BITS = 32 - 8 - 2;

inline double bicubic(double x) {
    const double a = -0.5;
    if (x < 0) x = -x;
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

// Pillow's precompute_coeffs + normalize_coeffs_8bpc
inline int coeffs(int in, int out, std::vector<int>& bounds, std::vector<int>& kk) {
    double scale = (double)in / out, filterscale = scale < 1 ? 1 : scale;
    double support = 2.0 * filterscale;
    int ksize = (int)std::ceil(support) * 2 + 1;
    bounds.assign(out * 2, 0); kk.assign((size_t)out * ksize, 0);
    std::vector<double> k(ksize);
    for (int xx = 0; xx < out; xx++) {
        double center = (xx + 0.5) * scale, ww = 0, ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
        xmax -= xmin;
        for (int x = 0; x < xmax; x++) { double w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (int x = 0; x < xmax; x++) {
            double w = ww != 0 ? k[x] / ww : k[x];
            kk[(size_t)xx * ksize + x] = w < 0 ? (int)(-0.5 + w * (1 << PRECISION_BITS)) : (int)(0.5 + w * (1 << PRECISION_BITS));
        }
        bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
    }
    return ksize;
}

inline uint8_t clip8(int v) { v >>= PRECISION_BITS; return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v); }

inline RGB8 resize(const RGB8& src, int ow, int oh) {
    RGB8 cur = src;
    if (ow != cur.w) {                       // horizontal pass
        std::vector<int> b, kk; int ks = coeffs(cur.w, ow, b, kk);
        RGB8 t; t.w = ow; t.h = cur.h; t.px.resize((size_t)ow * cur.h * 3);
        for (int y = 0; y < cur.h; y++)
            for (int xx = 0; xx < ow; xx++) {
                int xmin = b[xx * 2], xmax = b[xx * 2 + 1]; const int* k = &kk[(size_t)xx * ks];
                for (int c = 0; c < 3; c++) {
                    int ss = 1 << (PRECISION_BITS - 1);
                    for (int x = 0; x < xmax; x++) ss += cur.px[((size_t)y * cur.w + x + xmin) * 3 + c] * k[x];
                    t.px[((size_t)y * ow + xx) * 3 + c] = clip8(ss);
                }
            }
        cur = std::move(t);
    }
    if (oh != cur.h) {                       // vertical pass
        std::vector<int> b, kk; int ks = coeffs(cur.h, oh, b, kk);
        RGB8 t; t.w = cur.w; t.h = oh; t.px.resize((size_t)cur.w * oh * 3);
        for (int yy = 0; yy < oh; yy++) {
            int ymin = b[yy * 2], ymax = b[yy * 2 + 1]; const int* k = &kk[(size_t)yy * ks];
            for (int x = 0; x < cur.w; x++)
                for (int c = 0; c < 3; c++) {
                    int ss = 1 << (PRECISION_BITS - 1);
                    for (int y = 0; y < ymax; y++) ss += cur.px[((size_t)(y + ymin) * cur.w + x) * 3 + c] * k[y];
                    t.px[((size_t)yy * cur.w + x) * 3 + c] = clip8(ss);
                }
        }
        cur = std::move(t);
    }
    return cur;
}
}  // namespace pil

// Resized to w x h and normalized: planar float [3][h][w]
inline std::vector<float> preprocess(const RGB8& im, int w, int h) {
    RGB8 r = pil::resize(im, w, h);
    std::vector<float> out((size_t)3 * w * h);
    for (int c = 0; c < 3; c++)
        for (size_t i = 0; i < (size_t)w * h; i++) out[c * (size_t)w * h + i] = (r.px[i * 3 + c] / 255.0f - 0.5f) / 0.5f;
    return out;
}
