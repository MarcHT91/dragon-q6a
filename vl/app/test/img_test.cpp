// img_test IMAGE W H OUT.f32 [OUT_RAW.u8]: preprocessed planar float [3][H][W], optionally the decoded RGB8
#include <cstdio>
#define STB_IMAGE_IMPLEMENTATION
#include "../image.h"
int main(int argc, char** argv) {
    RGB8 im = load_rgb(argv[1]);
    auto x = preprocess(im, atoi(argv[2]), atoi(argv[3]));
    FILE* f = fopen(argv[4], "wb"); fwrite(x.data(), 4, x.size(), f); fclose(f);
    if (argc > 5) { f = fopen(argv[5], "wb"); fwrite(im.px.data(), 1, im.px.size(), f); fclose(f); }
    printf("%dx%d\n", im.w, im.h);
}
