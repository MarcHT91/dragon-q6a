// CLML (Adreno OpenCL ML) microbenchmark: fp16 GEMM and 1x1 convolution, with clTuneMLOpQCOM tuning.
// Build: g++ -O2 clml_bench.cpp -I<Adreno OpenCL ML SDK dir, has CL/cl_qcom_ml_ops.h> -l:libOpenCL.so.1 -o clml_bench
//        (the SDK is a free download from Qualcomm with an account; its header is not redistributed here)
// Run:   source ../kgsl/adreno-env.sh; ./clml_bench [gemm|conv] M N K [acc32]
// Result on this board (qcom-adreno 1.877.5 from the Ubuntu PPA): every clQueryMLInterfaceVersionsQCOM /
// clGetMLInterfaceQCOM call returns -30. The driver advertises cl_qcom_ml_ops 1.0.1, but its libCB.so does not
// export the mlb* backend the shim looks up, so CLML is unusable with this build. The version probe below
// prints what each call returns.
#include <CL/cl.h>
#include <CL/cl_qcom_ml_ops.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define CK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { fprintf(stderr, "%s -> %d (line %d)\n", #x, _e, __LINE__); exit(1); } } while (0)

static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | ((m + 0x1000) >> 13);
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    x = e == 0 ? s : (s | ((e - 15 + 127) << 23) | (m << 13));
    float f; memcpy(&f, &x, 4); return f;
}

typedef cl_int (*pfnQuery)(cl_int*, cl_int*, cl_uint, cl_uint*);
typedef cl_int (*pfnGet)(void*, cl_int, cl_int);

static CLMLInterfaceV1QCOM* ml;
static cl_context ctx;
static cl_command_queue q;

struct T { cl_ml_tensor_qcom t; cl_mem mem; };

static T make_tensor(cl_uint n, cl_uint c, cl_uint h, cl_uint w, const std::vector<uint16_t>* data) {
    cl_ml_tensor_desc_qcom d = {};
    d.data_type = CL_HALF_FLOAT; d.layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM;
    d.n = n; d.c = c; d.h = h; d.w = w; d.d = 1; d.num_dimensions = CL_TENSOR_DIMENSIONS_4D_QCOM;
    T r;
    CK(ml->clCreateMLTensorQCOM(ctx, NULL, &d, &r.t));
    cl_uint sz = 0; CK(ml->clGetMLTensorMemorySizeQCOM(ctx, r.t, &sz));
    cl_int err; r.mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz, NULL, &err); CK(err);
    if (data) {
        CK(ml->clEnqueueWriteMLTensorDataQCOM(q, (void*)data->data(), CL_TENSOR_LAYOUT_NCHW_QCOM, r.t, r.mem, 0, NULL, NULL));
        CK(clFinish(q));
    }
    return r;
}

int main(int argc, char** argv) {
    std::string op = argc > 1 ? argv[1] : "gemm";
    int M = argc > 2 ? atoi(argv[2]) : 1024, N = argc > 3 ? atoi(argv[3]) : 1024, K = argc > 4 ? atoi(argv[4]) : 1024;
    bool acc32 = argc > 5 && atoi(argv[5]);
    cl_arithmetic_mode_qcom am = acc32 ? CL_ARITHMETIC_MODE_FP16_ACC32_QCOM : CL_ARITHMETIC_MODE_FP16_QCOM;

    cl_platform_id pl; cl_device_id dev; cl_int err;
    CK(clGetPlatformIDs(1, &pl, NULL)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CK(err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CK(err);

    pfnQuery query = (pfnQuery)clGetExtensionFunctionAddressForPlatform(pl, "clQueryMLInterfaceVersionsQCOM");
    pfnGet get = (pfnGet)clGetExtensionFunctionAddressForPlatform(pl, "clGetMLInterfaceQCOM");
    if (!query || !get) { fprintf(stderr, "CLML entry points not found\n"); return 1; }
    for (cl_uint n = 1; n <= 8; n++) { std::vector<cl_int> mj(n), mn(n); cl_uint r = 0; cl_int e = query(mj.data(), mn.data(), n, &r);
        printf("query n=%u -> %d (ret %u)", n, e, r); if (!e) for (cl_uint i = 0; i < n && i < r; i++) printf(" %d.%d", mj[i], mn[i]); printf("\n"); if (!e) break; }
    int v1min = -1;
    for (int mi = 0; mi < 8; mi++) { void* p = NULL; cl_int e = get(&p, 1, mi); printf("get 1.%d -> %d %p\n", mi, e, p); if (!e && p) v1min = mi; }
    for (int mj = 2; mj <= 5; mj++) for (int mi = 0; mi < 4; mi++) { void* p = NULL; cl_int e = get(&p, mj, mi); if (!e && p) printf("get %d.%d OK\n", mj, mi); }
    if (v1min < 0) { fprintf(stderr, "no V1 interface\n"); return 1; }
    CK(get((void*)&ml, 1, v1min));

    srand(1);
    double flops; std::vector<T> ts; cl_ml_op_qcom opr;
    std::vector<float> fa, fb; std::vector<uint16_t> ha, hb;
    auto fill = [](std::vector<float>& f, std::vector<uint16_t>& h, size_t n) {
        f.resize(n); h.resize(n);
        for (size_t i = 0; i < n; i++) { h[i] = f2h(rand() / (float)RAND_MAX - 0.5f); f[i] = h2f(h[i]); }
    };
    cl_ml_tuningcache_qcom tc; CK(ml->clCreateMLTuningCacheQCOM(&tc));
    if (op == "gemm") {
        // C[M,N] = A[M,K] * B[K,N]; tensors as (n=1, c=1, h=rows, w=cols)
        fill(fa, ha, (size_t)M * K); fill(fb, hb, (size_t)K * N);
        T A = make_tensor(1, 1, M, K, &ha), B = make_tensor(1, 1, K, N, &hb), C = make_tensor(1, 1, M, N, NULL);
        cl_ml_value_qcom alpha = {}, beta = {}; alpha.u.fp32 = 1.f; alpha.type = CL_FLOAT; beta.u.fp32 = 0.f; beta.type = CL_FLOAT;
        cl_ml_op_gemm_desc_qcom gd = {(cl_uint)M, (cl_uint)N, (cl_uint)K, CL_GEMM_TRANSFORM_NONE_QCOM, CL_GEMM_TRANSFORM_NONE_QCOM, alpha, beta, am};
        CK(ml->clCreateMLOpGemmQCOM(ctx, NULL, &gd, A.t, B.t, C.t, &opr, tc));
        ts = {A, B, C};
        flops = 2.0 * M * N * K;
    } else {
        // 1x1 conv == Linear: input [1, K, H, W] with H*W = M positions, weight [N, K, 1, 1], output [1, N, H, W]
        int H = (int)std::sqrt((double)M), W = M / H;
        fill(fa, ha, (size_t)K * M); fill(fb, hb, (size_t)N * K);
        T X = make_tensor(1, K, H, W, &ha), Wt = make_tensor(N, K, 1, 1, &hb), Y = make_tensor(1, N, H, W, NULL);
        cl_ml_op_convolution_desc_qcom cd = {};
        cd.mode = CL_CONVOLUTION_MODE_CONVOLUTION_QCOM; cd.group_count = 1; cd.num_dimensions = 2;
        cd.filter_stride[0] = cd.filter_stride[1] = 1; cd.dilation[0] = cd.dilation[1] = 1; cd.arithmetic_mode = am;
        CK(ml->clCreateMLOpConvolutionForwardQCOM(ctx, NULL, &cd, X.t, Wt.t, NULL, Y.t, &opr, tc));
        ts = {X, Wt, Y};
        flops = 2.0 * M * N * K;
    }
    cl_ml_tensor_mem_desc_set_qcom ds; CK(ml->clCreateMLTensorMemoryDescriptorSetQCOM(&ds));
    std::vector<cl_ml_tensor_memory_desc_qcom> md;
    for (auto& t : ts) md.push_back({t.t, t.mem});
    CK(ml->clUpdateMLTensorMemoryDescriptorSetQCOM(ds, md.size(), md.data()));

    auto run = [&]() {
        cl_event ev; CK(ml->clEnqueueMLOpQCOM(q, opr, ds, 0, NULL, &ev)); CK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t0, NULL); clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
        clReleaseEvent(ev); return (t1 - t0) / 1e6;
    };
    double before = 1e9; for (int i = 0; i < 4; i++) { double t = run(); if (i && t < before) before = t; }
    cl_ulong tuned = 0; CK(ml->clTuneMLOpQCOM(q, opr, ds, tc, &tuned));
    double after = 1e9; for (int i = 0; i < 6; i++) { double t = run(); if (i && t < after) after = t; }

    // check a sample of outputs: read back via a plain NCHW read is not in V1, so copy the raw buffer only if layout is plain;
    // instead verify with clEnqueueReadMLTensorDataQCOM when available (V2+). Here we just report timing.
    printf("CLML %s M=%d N=%d K=%d %s: untuned %.2f ms (%.0f GFLOPS), tuned %.2f ms (%.0f GFLOPS)\n", op.c_str(), M, N, K,
           acc32 ? "fp16-acc32" : "fp16", before, flops / before / 1e6, after, flops / after / 1e6);
    return 0;
}
