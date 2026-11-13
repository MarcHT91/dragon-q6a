// enc_bench ENC_DIR SIZE IMAGE [runs]: encoder alone on the GPU, per-run time
#include <chrono>
#include <cstdio>
#include "net.h"
#include "gpu.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../image.h"
int main(int argc, char** argv) {
    int S = atoi(argv[2]), runs = argc > 4 ? atoi(argv[4]) : 5;
    ncnn::create_gpu_instance();
    {
        ncnn::Net net; int dev = 0;
        for (int i = 0; i < ncnn::get_gpu_count(); i++) if (ncnn::get_gpu_info(i).type() != 3) { dev = i; break; }
        net.set_vulkan_device(dev);
        net.opt.use_vulkan_compute = true; net.opt.use_subgroup_ops = false; net.opt.lightmode = false;
        std::string b = std::string(argv[1]) + "/vlenc_" + argv[2] + "x" + argv[2] + ".ncnn";
        net.load_param((b + ".param").c_str()); net.load_model((b + ".bin").c_str());
        auto x = preprocess(load_rgb(argv[3]), S, S);
        for (int r = 0; r < runs; r++) {
            auto t = std::chrono::steady_clock::now();
            ncnn::Mat in(S, S, 3, x.data()); ncnn::Extractor ex = net.create_extractor(); ex.input("in0", in);
            ncnn::Mat out; ex.extract("out0", out);
            printf("run %d: %.0f ms (out dims %d w %d h %d c %d)\n", r, std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count() * 1e3, out.dims, out.w, out.h, out.c);
        }
    }
    ncnn::destroy_gpu_instance();
}
