// Native QNN HTP runner for the pre-compiled context binaries (*_ctx_qnn.bin) written by the ONNX Runtime QNN EP.
// Every graph input/output lives in its own rpcmem buffer registered once with QnnMem_register, so graphExecute
// does no per-call copies; the host reads/writes the buffers in place (numpy views from npu/qnnrun.py).
// lf_* functions run a whole LFM3 decode step (host work included) in C++. Build: native/build.sh
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

#include "QnnInterface.h"
#include "QnnMem.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

#define LOGE(...) do { fprintf(stderr, "[qnnrun] " __VA_ARGS__); fputc('\n', stderr); } while (0)

typedef void* (*rpcmem_alloc_t)(int, uint32_t, int);
typedef void (*rpcmem_free_t)(void*);
typedef int (*rpcmem_to_fd_t)(void*);

static QNN_INTERFACE_VER_TYPE qnn;
static QNN_SYSTEM_INTERFACE_VER_TYPE qsys;
static Qnn_BackendHandle_t backend = nullptr;
static Qnn_DeviceHandle_t device = nullptr;
static rpcmem_alloc_t p_rpcmem_alloc; static rpcmem_free_t p_rpcmem_free; static rpcmem_to_fd_t p_rpcmem_to_fd;

struct Buf { void* ptr = nullptr; size_t bytes = 0; Qnn_MemHandle_t mh = nullptr; bool owned = true; };
struct Graph {
    std::string name;
    Qnn_ContextHandle_t ctx = nullptr;
    Qnn_GraphHandle_t g = nullptr;
    QnnSystemContext_Handle_t sysctx = nullptr;
    std::vector<Qnn_Tensor_t> in, out;
    std::vector<Buf> inb, outb;
};
static std::vector<Graph> graphs;

static size_t dtype_size(Qnn_DataType_t t) {
    switch (t) {
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_INT_8: case QNN_DATATYPE_BOOL_8: return 1;
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_FLOAT_16: return 2;
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_INT_32: return 4;
    default: return 0;
    }
}
// Qnn_TensorV2_t starts with the same fields as Qnn_TensorV1_t, so v1 accessors work for both.
static Qnn_TensorV1_t& T(Qnn_Tensor_t& t) { return t.v1; }

template <class F> static F sym(void* h, const char* n) {
    F f = (F)dlsym(h, n);
    if (!f) LOGE("missing symbol %s", n);
    return f;
}

static int set_burst() {
    QnnDevice_Infrastructure_t inf = nullptr;
    if (qnn.deviceGetInfrastructure(&inf) != QNN_SUCCESS || !inf) { LOGE("deviceGetInfrastructure failed"); return -1; }
    auto* hinf = (QnnHtpDevice_Infrastructure_t*)inf;
    auto& perf = hinf->perfInfra;
    uint32_t id = 0;
    if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) { LOGE("createPowerConfigId failed"); return -1; }
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs; memset(&dcvs, 0, sizeof dcvs);
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& d = dcvs.dcvsV3Config;
    d.contextId = id; d.setDcvsEnable = 1; d.dcvsEnable = 0;
    d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    d.setSleepLatency = 1; d.sleepLatency = 40; d.setSleepDisable = 1; d.sleepDisable = 1;
    d.setBusParams = 1; d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    d.setCoreParams = 1; d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    QnnHtpPerfInfrastructure_PowerConfig_t lat; memset(&lat, 0, sizeof lat);
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY; lat.rpcControlLatencyConfig = 100;
    QnnHtpPerfInfrastructure_PowerConfig_t pol; memset(&pol, 0, sizeof pol);
    pol.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME; pol.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t* cfg[] = {&dcvs, &lat, &pol, nullptr};
    if (perf.setPowerConfig(id, cfg) != QNN_SUCCESS) { LOGE("setPowerConfig failed"); return -1; }
    return 0;
}

extern "C" {

int qr_init(const char* backend_so, const char* system_so, int burst) {
    void* hb = dlopen(backend_so, RTLD_NOW | RTLD_LOCAL);
    if (!hb) { LOGE("dlopen %s: %s", backend_so, dlerror()); return -1; }
    void* hs = dlopen(system_so, RTLD_NOW | RTLD_LOCAL);
    if (!hs) { LOGE("dlopen %s: %s", system_so, dlerror()); return -1; }
    void* hr = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!hr) { LOGE("dlopen libcdsprpc.so: %s", dlerror()); return -1; }
    p_rpcmem_alloc = sym<rpcmem_alloc_t>(hr, "rpcmem_alloc"); p_rpcmem_free = sym<rpcmem_free_t>(hr, "rpcmem_free");
    p_rpcmem_to_fd = sym<rpcmem_to_fd_t>(hr, "rpcmem_to_fd");
    if (!p_rpcmem_alloc || !p_rpcmem_free || !p_rpcmem_to_fd) return -1;

    typedef Qnn_ErrorHandle_t (*GetProv)(const QnnInterface_t***, uint32_t*);
    const QnnInterface_t** prov = nullptr; uint32_t n = 0;
    if (sym<GetProv>(hb, "QnnInterface_getProviders")(&prov, &n) != QNN_SUCCESS || n == 0) { LOGE("getProviders"); return -1; }
    qnn = prov[0]->QNN_INTERFACE_VER_NAME;
    typedef Qnn_ErrorHandle_t (*GetSysProv)(const QnnSystemInterface_t***, uint32_t*);
    const QnnSystemInterface_t** sprov = nullptr; uint32_t sn = 0;
    if (sym<GetSysProv>(hs, "QnnSystemInterface_getProviders")(&sprov, &sn) != QNN_SUCCESS || sn == 0) { LOGE("sys getProviders"); return -1; }
    qsys = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    if (qnn.backendCreate(nullptr, nullptr, &backend) != QNN_SUCCESS) { LOGE("backendCreate"); return -1; }
    if (qnn.deviceCreate && qnn.deviceCreate(nullptr, nullptr, &device) != QNN_SUCCESS) { LOGE("deviceCreate"); return -1; }
    if (burst && set_burst() != 0) LOGE("burst mode not set (continuing)");
    return 0;
}

static int alloc_reg(Graph& G, Qnn_Tensor_t& t, Buf& b) {
    auto& v = T(t);
    size_t n = dtype_size(v.dataType);
    for (uint32_t i = 0; i < v.rank; i++) n *= v.dimensions[i];
    if (!n) { LOGE("tensor %s: unknown size/dtype %d", v.name, (int)v.dataType); return -1; }
    b.bytes = n;
    b.ptr = p_rpcmem_alloc(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1 /*RPCMEM_DEFAULT_FLAGS*/, (uint32_t)n);
    if (!b.ptr) { LOGE("rpcmem_alloc %zu failed", n); return -1; }
    memset(b.ptr, 0, n);
    Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT;
    md.memShape.numDim = v.rank; md.memShape.dimSize = v.dimensions; md.memShape.shapeConfig = nullptr;
    md.dataType = v.dataType; md.memType = QNN_MEM_TYPE_ION; md.ionInfo.fd = p_rpcmem_to_fd(b.ptr);
    if (qnn.memRegister(G.ctx, &md, 1, &b.mh) != QNN_SUCCESS) { LOGE("memRegister %s failed", v.name); return -1; }
    v.memType = QNN_TENSORMEMTYPE_MEMHANDLE; v.memHandle = b.mh;
    return 0;
}

// Load a context binary (one graph). Returns graph index or <0.
int qr_load(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { LOGE("open %s", path); return -1; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char>* blob = new std::vector<char>(sz);   // kept alive (system ctx info points into it)
    f.read(blob->data(), sz);
    Graph G;
    if (qsys.systemContextCreate(&G.sysctx) != QNN_SUCCESS) { LOGE("systemContextCreate"); return -2; }
    const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bisz = 0;
    if (qsys.systemContextGetBinaryInfo(G.sysctx, blob->data(), sz, &bi, &bisz) != QNN_SUCCESS || !bi) { LOGE("getBinaryInfo"); return -3; }
    uint32_t ng = 0; QnnSystemContext_GraphInfo_t* gi = nullptr;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { ng = bi->contextBinaryInfoV1.numGraphs; gi = bi->contextBinaryInfoV1.graphs; }
    else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { ng = bi->contextBinaryInfoV2.numGraphs; gi = bi->contextBinaryInfoV2.graphs; }
    else { ng = bi->contextBinaryInfoV3.numGraphs; gi = bi->contextBinaryInfoV3.graphs; }
    if (ng != 1) { LOGE("expected 1 graph, got %u", ng); return -4; }
    const char* gname; uint32_t nin, nout; Qnn_Tensor_t *tin, *tout;
    if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { auto& g = gi->graphInfoV1; gname = g.graphName; nin = g.numGraphInputs; tin = g.graphInputs; nout = g.numGraphOutputs; tout = g.graphOutputs; }
    else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { auto& g = gi->graphInfoV2; gname = g.graphName; nin = g.numGraphInputs; tin = g.graphInputs; nout = g.numGraphOutputs; tout = g.graphOutputs; }
    else { auto& g = gi->graphInfoV3; gname = g.graphName; nin = g.numGraphInputs; tin = g.graphInputs; nout = g.numGraphOutputs; tout = g.graphOutputs; }
    G.name = gname;
    if (qnn.contextCreateFromBinary(backend, device, nullptr, blob->data(), sz, &G.ctx, nullptr) != QNN_SUCCESS) { LOGE("contextCreateFromBinary %s", path); return -5; }
    if (qnn.graphRetrieve(G.ctx, gname, &G.g) != QNN_SUCCESS) { LOGE("graphRetrieve %s", gname); return -6; }
    G.in.assign(tin, tin + nin); G.out.assign(tout, tout + nout);
    G.inb.resize(nin); G.outb.resize(nout);
    for (uint32_t i = 0; i < nin; i++) if (alloc_reg(G, G.in[i], G.inb[i])) return -7;
    for (uint32_t i = 0; i < nout; i++) if (alloc_reg(G, G.out[i], G.outb[i])) return -8;
    graphs.push_back(G);
    return (int)graphs.size() - 1;
}

int qr_num(int g, int is_out) { return (int)(is_out ? graphs[g].out.size() : graphs[g].in.size()); }

// Tensor metadata + buffer pointer. dims must hold 8 entries.
int qr_info(int g, int is_out, int i, char* name, int name_len, uint32_t* dims, uint32_t* rank, int* dtype,
            float* scale, int* offset, void** ptr, size_t* bytes) {
    Graph& G = graphs[g];
    Qnn_Tensor_t& t = is_out ? G.out[i] : G.in[i];
    Buf& b = is_out ? G.outb[i] : G.inb[i];
    auto& v = T(t);
    snprintf(name, name_len, "%s", v.name);
    *rank = v.rank; for (uint32_t k = 0; k < v.rank && k < 8; k++) dims[k] = v.dimensions[k];
    *dtype = (int)v.dataType;
    *scale = 0; *offset = 0;
    if (v.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        *scale = v.quantizeParams.scaleOffsetEncoding.scale; *offset = v.quantizeParams.scaleOffsetEncoding.offset;
    }
    *ptr = b.ptr; *bytes = b.bytes;
    return 0;
}

// Make input `in_idx` of graph `gdst` use the same rpcmem buffer as tensor (gsrc, src_is_out, src_idx):
// the buffer is registered a second time in gdst's context, and gdst's own buffer is freed.
int qr_share_input(int gdst, int in_idx, int gsrc, int src_is_out, int src_idx) {
    Graph& D = graphs[gdst]; Graph& S = graphs[gsrc];
    Buf& sb = src_is_out ? S.outb[src_idx] : S.inb[src_idx];
    Buf& db = D.inb[in_idx];
    auto& v = T(D.in[in_idx]);
    if (sb.bytes != db.bytes) { LOGE("share %s: size %zu != %zu", v.name, sb.bytes, db.bytes); return -1; }
    Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT;
    md.memShape.numDim = v.rank; md.memShape.dimSize = v.dimensions; md.memShape.shapeConfig = nullptr;
    md.dataType = v.dataType; md.memType = QNN_MEM_TYPE_ION; md.ionInfo.fd = p_rpcmem_to_fd(sb.ptr);
    Qnn_MemHandle_t mh = nullptr;
    if (qnn.memRegister(D.ctx, &md, 1, &mh) != QNN_SUCCESS) { LOGE("share memRegister %s failed", v.name); return -2; }
    if (db.mh) qnn.memDeRegister(&db.mh, 1);
    if (db.ptr && db.owned) p_rpcmem_free(db.ptr);
    db.ptr = sb.ptr; db.mh = mh; db.owned = false;
    v.memHandle = mh;
    return 0;
}

int qr_execute(int g) {
    Graph& G = graphs[g];
    Qnn_ErrorHandle_t e = qnn.graphExecute(G.g, G.in.data(), G.in.size(), G.out.data(), G.out.size(), nullptr, nullptr);
    if (e != QNN_SUCCESS) { LOGE("graphExecute %s -> %lu", G.name.c_str(), (unsigned long)e); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// LFM3 decode step entirely in C++ (host work that used to be numpy): embedding quantization, RoPE rows
// (from tables precomputed in Python), masks, KV slot update (hi/lo split), conv state copy, graph chaining.
struct LayerPlan { int type; int in_state, out_state, in_kh, in_kl, in_vh, in_vl, out_k, out_v; };  // type 0 conv, 1 full, 2 sliding
struct GraphPlan {
    int g; int x, cos_f, sin_f, mask_f, cos_s, sin_s, mask_s, x_out;
    uint16_t mask_zero_f, mask_neg_f, mask_zero_s, mask_neg_s;
    std::vector<LayerPlan> layers;
};
static std::vector<GraphPlan> plans, pre_plans;      // decode graphs, prefill graphs (C tokens per call)
static int lf_C = 0, lf_rep = 1, lf_win = 0;
static const __fp16* lf_embed = nullptr;
static int lf_hid = 0, lf_T = 0, lf_TS = 0, lf_KV = 0, lf_D = 0, lf_head = -1, lf_head_x = -1, lf_head_logits = -1;
static const uint16_t *rope_cf, *rope_sf, *rope_cs, *rope_ss;   // [T][D] q16 tables

static inline uint16_t q16f(float x, float s) {
    float v = nearbyintf(x / s) + 32768.0f;
    return (uint16_t)(v < 0.f ? 0.f : (v > 65535.f ? 65535.f : v));
}

int lf_set_model(const void* embed, int hid, int T, int TS, int KV, int D,
                 const uint16_t* cf, const uint16_t* sf, const uint16_t* cs, const uint16_t* ss,
                 int head_g, int head_x, int head_logits) {
    lf_embed = (const __fp16*)embed; lf_hid = hid; lf_T = T; lf_TS = TS; lf_KV = KV; lf_D = D;
    rope_cf = cf; rope_sf = sf; rope_cs = cs; rope_ss = ss;
    lf_head = head_g; lf_head_x = head_x; lf_head_logits = head_logits;
    plans.clear();
    return 0;
}

// io: x, cos_f, sin_f, mask_f, cos_s, sin_s, mask_s, x_out, mask_zero_f, mask_neg_f, mask_zero_s, mask_neg_s (-1 = absent)
// layers: nlayers x 9 ints (type, in_state, out_state, in_kh, in_kl, in_vh, in_vl, out_k, out_v)
static GraphPlan make_plan(int g, const int* io, int nlayers, const int* layers) {
    GraphPlan P;
    P.g = g; P.x = io[0]; P.cos_f = io[1]; P.sin_f = io[2]; P.mask_f = io[3]; P.cos_s = io[4]; P.sin_s = io[5]; P.mask_s = io[6]; P.x_out = io[7];
    P.mask_zero_f = (uint16_t)io[8]; P.mask_neg_f = (uint16_t)io[9]; P.mask_zero_s = (uint16_t)io[10]; P.mask_neg_s = (uint16_t)io[11];
    for (int i = 0; i < nlayers; i++) {
        const int* L = layers + 9 * i;
        P.layers.push_back({L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8]});
    }
    return P;
}

int lf_add_graph(int g, const int* io, int nlayers, const int* layers) {
    plans.push_back(make_plan(g, io, nlayers, layers));
    return 0;
}

// Prefill graphs: C tokens per call, rep = attention heads per KV head (mask rows are tiled rep times),
// win = sliding window. io/layers as in lf_add_graph; their state inputs must share the decode buffers.
int lf_set_prefill(int C, int rep, int win) {
    lf_C = C; lf_rep = rep; lf_win = win; pre_plans.clear();
    return 0;
}

int lf_add_prefill_graph(int g, const int* io, int nlayers, const int* layers) {
    pre_plans.push_back(make_plan(g, io, nlayers, layers));
    return 0;
}

static inline void split16(uint16_t q, uint8_t& hi, uint8_t& lo) {
    int v = (int)q - 32768; if (v > 32639) v = 32639;
    int l = ((v + 128) & 255) - 128;
    int h = (v - l) >> 8;
    hi = (uint8_t)(h + 128); lo = (uint8_t)(l + 128);
}

static float tensor_scale(Qnn_Tensor_t& t) {
    auto& v = T(t);
    return v.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ? v.quantizeParams.scaleOffsetEncoding.scale : 1.f;
}

static void store_kv(Graph& G, const LayerPlan& L, int pos, int n);

// One decode step at position pos. Returns a pointer to the head's uint16 logits buffer (valid until the next call).
const uint16_t* lf_step(int token, int pos) {
    if (pos >= lf_T) { LOGE("context full"); return nullptr; }
    const int D = lf_D, hid = lf_hid;
    const uint16_t* prev = nullptr;
    for (size_t gi = 0; gi < plans.size(); gi++) {
        GraphPlan& P = plans[gi];
        Graph& G = graphs[P.g];
        uint16_t* x = (uint16_t*)G.inb[P.x].ptr;
        if (gi == 0) {
            float s = tensor_scale(G.in[P.x]);
            const __fp16* e = lf_embed + (size_t)token * hid;
            for (int i = 0; i < hid; i++) x[i] = q16f((float)e[i], s);
        } else {
            memcpy(x, prev, hid * sizeof(uint16_t));
        }
        if (P.cos_f >= 0) { memcpy(G.inb[P.cos_f].ptr, rope_cf + (size_t)pos * D, D * 2); memcpy(G.inb[P.sin_f].ptr, rope_sf + (size_t)pos * D, D * 2); }
        if (P.cos_s >= 0) { memcpy(G.inb[P.cos_s].ptr, rope_cs + (size_t)pos * D, D * 2); memcpy(G.inb[P.sin_s].ptr, rope_ss + (size_t)pos * D, D * 2); }
        if (P.mask_f >= 0) {
            uint16_t* m = (uint16_t*)G.inb[P.mask_f].ptr;
            for (int i = 0; i < pos; i++) m[i] = P.mask_zero_f;
            for (int i = pos; i < lf_T; i++) m[i] = P.mask_neg_f;
            m[lf_T] = P.mask_zero_f;
        }
        if (P.mask_s >= 0) {
            uint16_t* m = (uint16_t*)G.inb[P.mask_s].ptr;
            int k = pos < lf_TS ? pos : lf_TS;
            for (int i = 0; i < k; i++) m[i] = P.mask_zero_s;
            for (int i = k; i < lf_TS; i++) m[i] = P.mask_neg_s;
            m[lf_TS] = P.mask_zero_s;
        }
        if (qr_execute(P.g)) return nullptr;
        for (auto& L : P.layers) {
            if (L.type == 0) {
                memcpy(G.inb[L.in_state].ptr, G.outb[L.out_state].ptr, G.inb[L.in_state].bytes);
            } else {
                store_kv(G, L, pos, 1);
            }
        }
        prev = (const uint16_t*)G.outb[P.x_out].ptr;
    }
    Graph& H = graphs[lf_head];
    memcpy(H.inb[lf_head_x].ptr, prev, hid * sizeof(uint16_t));
    if (qr_execute(lf_head)) return nullptr;
    return (const uint16_t*)H.outb[lf_head_logits].ptr;
}

// Store the new K/V rows (positions pos..pos+n-1, uint16 [KV,n,D]) as hi/lo bytes in their cache slots.
static void store_kv(Graph& G, const LayerPlan& L, int pos, int n) {
    const int D = lf_D, KV = lf_KV;
    const int N = L.type == 1 ? lf_T : lf_TS;
    const uint16_t* kn = (const uint16_t*)G.outb[L.out_k].ptr;
    const uint16_t* vn = (const uint16_t*)G.outb[L.out_v].ptr;
    uint8_t* kh = (uint8_t*)G.inb[L.in_kh].ptr; uint8_t* kl = (uint8_t*)G.inb[L.in_kl].ptr;   // [KV,D,N]
    uint8_t* vh = (uint8_t*)G.inb[L.in_vh].ptr; uint8_t* vl = (uint8_t*)G.inb[L.in_vl].ptr;   // [KV,N,D]
    for (int t = 0; t < n; t++) {
        const int slot = L.type == 1 ? pos + t : (pos + t) % lf_TS;
        for (int h = 0; h < KV; h++)
            for (int d = 0; d < D; d++) {
                uint8_t a, b;
                const size_t io = ((size_t)h * n + t) * D + d;
                split16(kn[io], a, b);
                size_t ik = ((size_t)h * D + d) * N + slot; kh[ik] = a; kl[ik] = b;
                split16(vn[io], a, b);
                size_t iv = ((size_t)h * N + slot) * D + d; vh[iv] = a; vl[iv] = b;
            }
    }
}

// One prefill chunk: tokens[0..C) at positions pos..pos+C-1 (no logits). Same host work as the numpy version.
int lf_prefill(const int* tokens, int pos) {
    const int C = lf_C, D = lf_D, hid = lf_hid, T = lf_T, TS = lf_TS;
    if (!C || pre_plans.empty()) { LOGE("prefill not configured"); return -1; }
    if (pos + C > T) { LOGE("context full"); return -1; }
    const uint16_t* prev = nullptr;
    std::vector<uint16_t> row;
    for (size_t gi = 0; gi < pre_plans.size(); gi++) {
        GraphPlan& P = pre_plans[gi];
        Graph& G = graphs[P.g];
        uint16_t* x = (uint16_t*)G.inb[P.x].ptr;                 // [C, hid]
        if (gi == 0) {
            float s = tensor_scale(G.in[P.x]);
            for (int t = 0; t < C; t++) {
                const __fp16* e = lf_embed + (size_t)tokens[t] * hid;
                for (int i = 0; i < hid; i++) x[(size_t)t * hid + i] = q16f((float)e[i], s);
            }
        } else {
            memcpy(x, prev, (size_t)C * hid * sizeof(uint16_t));
        }
        if (P.cos_f >= 0) { memcpy(G.inb[P.cos_f].ptr, rope_cf + (size_t)pos * D, (size_t)C * D * 2); memcpy(G.inb[P.sin_f].ptr, rope_sf + (size_t)pos * D, (size_t)C * D * 2); }
        if (P.cos_s >= 0) { memcpy(G.inb[P.cos_s].ptr, rope_cs + (size_t)pos * D, (size_t)C * D * 2); memcpy(G.inb[P.sin_s].ptr, rope_ss + (size_t)pos * D, (size_t)C * D * 2); }
        if (P.mask_f >= 0) {                                       // [rep*C, T+C]: row r*C+t is the row of token t
            uint16_t* m = (uint16_t*)G.inb[P.mask_f].ptr;
            const int W = T + C;
            row.resize(W);
            for (int t = 0; t < C; t++) {
                for (int i = 0; i < W; i++) row[i] = P.mask_neg_f;
                for (int i = 0; i < pos; i++) row[i] = P.mask_zero_f;
                for (int i = T; i <= T + t; i++) row[i] = P.mask_zero_f;
                for (int r = 0; r < lf_rep; r++) memcpy(m + ((size_t)r * C + t) * W, row.data(), W * 2);
            }
        }
        if (P.mask_s >= 0) {                                       // [rep*C, TS+C]
            uint16_t* m = (uint16_t*)G.inb[P.mask_s].ptr;
            const int W = TS + C;
            row.resize(W);
            for (int t = 0; t < C; t++) {
                for (int i = 0; i < W; i++) row[i] = P.mask_neg_s;
                for (int sl = 0; sl < TS && sl < pos; sl++) {
                    const long q = sl + (long)TS * ((pos - 1 - sl) / TS);   // last position < pos stored in slot sl
                    if (q > (long)pos + t - lf_win) row[sl] = P.mask_zero_s;
                }
                for (int i = TS; i <= TS + t; i++) row[i] = P.mask_zero_s;
                for (int r = 0; r < lf_rep; r++) memcpy(m + ((size_t)r * C + t) * W, row.data(), W * 2);
            }
        }
        if (qr_execute(P.g)) return -1;
        for (auto& L : P.layers) {
            if (L.type == 0) memcpy(G.inb[L.in_state].ptr, G.outb[L.out_state].ptr, G.inb[L.in_state].bytes);
            else store_kv(G, L, pos, C);
        }
        prev = (const uint16_t*)G.outb[P.x_out].ptr;
    }
    return 0;
}

void qr_shutdown() {
    for (auto& G : graphs) {
        for (auto* bs : {&G.inb, &G.outb})
            for (auto& b : *bs) { if (b.mh) qnn.memDeRegister(&b.mh, 1); if (b.ptr && b.owned) p_rpcmem_free(b.ptr); }
        if (G.ctx) qnn.contextFree(G.ctx, nullptr);
        if (G.sysctx) qsys.systemContextFree(G.sysctx);
    }
    graphs.clear();
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend) qnn.backendFree(backend);
    device = nullptr; backend = nullptr;
}

}  // extern "C"
