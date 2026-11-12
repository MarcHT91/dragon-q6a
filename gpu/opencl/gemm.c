// OpenCL fp16 GEMM microbenchmark for the Adreno 643: C[M,N] = A[M,K] * B[K,N].
// Variants: "buf" (A,B buffers) and "img" (B as RGBA-half image2d, read via the texture cache).
// Each work-item computes R rows x 4 cols.
// Build: git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers; gcc -O2 gemm.c -IOpenCL-Headers -l:libOpenCL.so.1 -lm -o gemm
// Run:   ./gemm img|buf R lws_x lws_y acc32 [M N K]
//        Qualcomm driver: source ../kgsl/adreno-env.sh first. Open driver (rusticl): RUSTICL_ENABLE=freedreno ./gemm ...
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static const char* SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"#ifdef ACC32\n#define ACC float4\n#define CVT(x) convert_float4(x)\n#define CVTS(x) convert_float(x)\n#else\n#define ACC half4\n#define CVT(x) (x)\n#define CVTS(x) (x)\n#endif\n"
"__kernel void gemm_img(const int M, const int N, const int K, __global const half* A, __read_only image2d_t B, __global half* C) {\n"
"  const int gx = get_global_id(0), gy = get_global_id(1);\n"
"  ACC acc[R]; for (int r = 0; r < R; r++) acc[r] = (ACC)(0);\n"
"  for (int k = 0; k < K; k += 4) {\n"
"    ACC b0 = CVT(read_imageh(B, smp, (int2)(gx, k)));\n"
"    ACC b1 = CVT(read_imageh(B, smp, (int2)(gx, k+1)));\n"
"    ACC b2 = CVT(read_imageh(B, smp, (int2)(gx, k+2)));\n"
"    ACC b3 = CVT(read_imageh(B, smp, (int2)(gx, k+3)));\n"
"    for (int r = 0; r < R; r++) {\n"
"      ACC a = CVT(vload4(0, A + (gy*R + r)*K + k));\n"
"      acc[r] = mad(a.x, b0, mad(a.y, b1, mad(a.z, b2, mad(a.w, b3, acc[r]))));\n"
"    }\n"
"  }\n"
"  for (int r = 0; r < R; r++) vstore4(convert_half4(acc[r]), 0, C + (gy*R + r)*N + gx*4);\n"
"}\n"
"__kernel void gemm_buf(const int M, const int N, const int K, __global const half* A, __global const half* B, __global half* C) {\n"
"  const int gx = get_global_id(0), gy = get_global_id(1);\n"
"  ACC acc[R]; for (int r = 0; r < R; r++) acc[r] = (ACC)(0);\n"
"  for (int k = 0; k < K; k += 4) {\n"
"    ACC b0 = CVT(vload4(0, B + (k+0)*N + gx*4));\n"
"    ACC b1 = CVT(vload4(0, B + (k+1)*N + gx*4));\n"
"    ACC b2 = CVT(vload4(0, B + (k+2)*N + gx*4));\n"
"    ACC b3 = CVT(vload4(0, B + (k+3)*N + gx*4));\n"
"    for (int r = 0; r < R; r++) {\n"
"      ACC a = CVT(vload4(0, A + (gy*R + r)*K + k));\n"
"      acc[r] = mad(a.x, b0, mad(a.y, b1, mad(a.z, b2, mad(a.w, b3, acc[r]))));\n"
"    }\n"
"  }\n"
"  for (int r = 0; r < R; r++) vstore4(convert_half4(acc[r]), 0, C + (gy*R + r)*N + gx*4);\n"
"}\n";

static unsigned short f2h(float f) {  // float -> IEEE half (round to nearest, no denormals needed here)
    unsigned int x; memcpy(&x, &f, 4);
    unsigned int s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xff) - 127 + 15; unsigned int m = x & 0x7fffff;
    if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | ((m + 0x1000) >> 13);
}
static float h2f(unsigned short h) {
    unsigned int s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    if (e == 0) x = s; else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
#define CK(x) do { cl_int _e = (x); if (_e) { fprintf(stderr, "%s -> %d (line %d)\n", #x, _e, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    const char* var = argc > 1 ? argv[1] : "img";
    int R = argc > 2 ? atoi(argv[2]) : 8;
    int lx = argc > 3 ? atoi(argv[3]) : 16, ly = argc > 4 ? atoi(argv[4]) : 8;
    int acc32 = argc > 5 ? atoi(argv[5]) : 0;
    if (argc > 6) { M = atoi(argv[6]); N = atoi(argv[7]); K = atoi(argv[8]); }

    cl_platform_id pl; cl_device_id dev; cl_int err;
    CK(clGetPlatformIDs(1, &pl, NULL)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CK(err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err); CK(err);
    char opts[128]; snprintf(opts, sizeof opts, "-DR=%d %s -cl-fast-relaxed-math", R, acc32 ? "-DACC32" : "");
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL)) {
        char log[8192]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); fprintf(stderr, "%s\n", log); return 1;
    }
    cl_kernel kr = clCreateKernel(prog, strcmp(var, "img") ? "gemm_buf" : "gemm_img", &err); CK(err);

    unsigned short *hA = malloc(2*M*K), *hB = malloc(2*K*N), *hC = malloc(2*M*N);
    float *fA = malloc(4*M*K), *fB = malloc(4*K*N);
    srand(1);
    for (int i = 0; i < M*K; i++) { fA[i] = (rand() / (float)RAND_MAX - 0.5f); hA[i] = f2h(fA[i]); fA[i] = h2f(hA[i]); }
    for (int i = 0; i < K*N; i++) { fB[i] = (rand() / (float)RAND_MAX - 0.5f); hB[i] = f2h(fB[i]); fB[i] = h2f(hB[i]); }
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*M*K, hA, &err); CK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 2*M*N, NULL, &err); CK(err);
    cl_mem dB;
    if (!strcmp(var, "img")) {
        cl_image_format f = {CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc d = {0}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = N / 4; d.image_height = K;
        dB = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &f, &d, hB, &err); CK(err);
    } else {
        dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*K*N, hB, &err); CK(err);
    }
    CK(clSetKernelArg(kr, 0, 4, &M)); CK(clSetKernelArg(kr, 1, 4, &N)); CK(clSetKernelArg(kr, 2, 4, &K));
    CK(clSetKernelArg(kr, 3, sizeof dA, &dA)); CK(clSetKernelArg(kr, 4, sizeof dB, &dB)); CK(clSetKernelArg(kr, 5, sizeof dC, &dC));
    size_t gws[2] = {N / 4, M / R}, lws[2] = {lx, ly};
    double best = 1e9;
    for (int it = 0; it < 6; it++) {
        cl_event ev;
        CK(clEnqueueNDRangeKernel(q, kr, 2, NULL, gws, lws, 0, NULL, &ev)); CK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t0, NULL); clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
        double ms = (t1 - t0) / 1e6; if (it && ms < best) best = ms; clReleaseEvent(ev);
    }
    CK(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, 2*M*N, hC, 0, NULL, NULL));
    double num = 0, den = 0;
    for (int i = 0; i < M; i += 37) for (int j = 0; j < N; j += 13) {
        double s = 0; for (int k = 0; k < K; k++) s += fA[i*K + k] * fB[k*N + j];
        double d = h2f(hC[i*N + j]) - s; num += d*d; den += s*s;
    }
    printf("%s %s R=%d lws=%dx%d acc%s M=%d N=%d K=%d: %.2f ms, %.0f GFLOPS, relerr %.4f\n", name, var, R, lx, ly, acc32 ? "32" : "16",
           M, N, K, best, 2.0*M*N*K / best / 1e6, sqrt(num / den));
    return 0;
}
