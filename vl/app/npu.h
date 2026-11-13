// LFM3 / LFM3-VL language model on the Hexagon NPU: C++ port of npu/npu/lfm3_native.py on top of the
// native QNN runner (npu/native/qnnrun.cpp, compiled into the same binary). Loads a build dir (manifest.json,
// *_ctx_qnn.bin, embed_f16.npy), sets up the per-token / per-chunk host plans and exposes feed / step.
#pragma once
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "third_party/json.hpp"

extern "C" {
int qr_init(const char* backend_so, const char* system_so, int burst);
int qr_load(const char* path);
int qr_num(int g, int is_out);
int qr_info(int g, int is_out, int i, char* name, int name_len, uint32_t* dims, uint32_t* rank, int* dtype,
            float* scale, int* offset, void** ptr, size_t* bytes);
int qr_share_input(int gdst, int in_idx, int gsrc, int src_is_out, int src_idx);
int lf_set_model(const void* embed, int hid, int T, int TS, int KV, int D, const uint16_t* cf, const uint16_t* sf,
                 const uint16_t* cs, const uint16_t* ss, int head_g, int head_x, int head_logits);
int lf_set_extra(const void* rows, int nvocab, int n);
int lf_add_graph(int g, const int* io, int nlayers, const int* layers);
int lf_set_prefill(int C, int rep, int win);
int lf_add_prefill_graph(int g, const int* io, int nlayers, const int* layers);
const uint16_t* lf_step(int token, int pos);
int lf_prefill(const int* tokens, int pos);
void qr_shutdown();
}

class NpuLM {
public:
    struct Tensor { int idx; float scale; void* ptr; size_t bytes; int dtype; };
    struct Graph { int g; std::map<std::string, Tensor> in, out; };

    int T = 0, TS = 0, C = 0, hid = 0, KV = 0, D = 0, nembed = 0, nvocab = 0, pos = 0;
    float logit_scale = 1.f; int logit_zp = 32768;
    std::vector<int> eos;
    double load_s = 0;

    NpuLM(const std::string& dir, const std::string& qnn_dir) {
        auto t0 = std::chrono::steady_clock::now();
        std::ifstream mf(dir + "/manifest.json");
        if (!mf) throw std::runtime_error("no manifest.json in " + dir);
        auto man = nlohmann::json::parse(mf);
        auto& c = man["cfg"];
        T = man["T"]; TS = man["TS"]; KV = c["num_key_value_heads"]; D = c["head_dim"]; hid = c["hidden_size"];
        for (auto& e : man["eos"]) eos.push_back(e);
        auto& lt = c["layer_types"];
        for (auto& t : lt) types_.push_back(t);

        setenv("ADSP_LIBRARY_PATH", qnn_dir.c_str(), 0);
        if (!dlopen((qnn_dir + "/libQnnHtpV68Stub.so").c_str(), RTLD_NOW | RTLD_GLOBAL))
            throw std::runtime_error(std::string("dlopen stub: ") + dlerror());
        if (qr_init((qnn_dir + "/libQnnHtp.so").c_str(), (qnn_dir + "/libQnnSystem.so").c_str(), 1))
            throw std::runtime_error("qr_init failed");
        inited_ = true;

        for (auto& g : man["graphs"]) { dec_.push_back(load(dir, g["ctx"])); layers_.push_back(g["layers"].get<std::vector<int>>()); }
        if (man.contains("prefill")) {
            C = man["prefill"]["C"];
            size_t k = 0;
            for (auto& gp : man["prefill"]["graphs"]) {
                Graph P = load(dir, gp["ctx"]);
                for (int i : layers_[k])
                    for (auto& nm : state_names(i)) {
                        if (qr_share_input(P.g, P.in.at(nm).idx, dec_[k].g, 0, dec_[k].in.at(nm).idx))
                            throw std::runtime_error("share " + nm + " failed");
                        P.in[nm].ptr = dec_[k].in.at(nm).ptr;
                    }
                pre_.push_back(P); k++;
            }
        }
        head_ = load(dir, man["head"]["ctx"]);
        logit_scale = man["head"]["qparams"]["logits"][0]; logit_zp = man["head"]["qparams"]["logits"][1];
        load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        map_embed(dir + "/embed_f16.npy");
        setup(c);
        reset();
    }

    ~NpuLM() {
        if (inited_) qr_shutdown();
        if (emap_) munmap(emap_, elen_);
    }

    // Extra embedding rows (image features, fp16 [n, hid]) selected by ids nembed..nembed+n-1.
    void set_extra(const std::vector<uint16_t>* rows) {
        lf_set_extra(rows ? rows->data() : nullptr, nembed, rows ? (int)(rows->size() / hid) : 0);
    }

    void reset() {
        for (size_t k = 0; k < dec_.size(); k++)
            for (int i : layers_[k])
                for (auto& nm : state_names(i)) {
                    auto& t = dec_[k].in.at(nm);
                    if (t.dtype == 0x0416 || t.dtype == 0x0116) { auto* p = (uint16_t*)t.ptr; for (size_t j = 0; j < t.bytes / 2; j++) p[j] = 32768; }
                    else memset(t.ptr, 128, t.bytes);
                }
        pos = 0;
    }

    // Feeds tokens and returns the logits after the last one. Prefill chunks of C tokens + single steps for the
    // rest; steps_first puts the (n-1) % C single steps before the chunks instead of after (same results, but
    // the last token is then the only step after the chunks).
    const uint16_t* feed(const std::vector<int>& ids, bool steps_first = false) {
        int i = 0, n = (int)ids.size();
        const uint16_t* lq = nullptr;
        if (C && steps_first)
            for (int r = (n - 1) % C; r > 0; r--) lq = step(ids[i++]);
        if (C)
            while (n - 1 - i >= C && pos + C <= T) {
                if (lf_prefill(ids.data() + i, pos)) throw std::runtime_error("lf_prefill failed");
                pos += C; i += C;
            }
        for (; i < n; i++) lq = step(ids[i]);
        return lq;
    }

    const uint16_t* step(int token) {
        if (pos >= T) throw std::runtime_error("context full");
        const uint16_t* p = lf_step(token, pos);
        if (!p) throw std::runtime_error("lf_step failed");
        pos++;
        return p;
    }

    float logit(const uint16_t* lq, int i) const { return ((int)lq[i] - logit_zp) * logit_scale; }

private:
    std::vector<Graph> dec_, pre_;
    std::vector<std::vector<int>> layers_;
    Graph head_;
    std::vector<std::string> types_;
    std::vector<uint16_t> tabs_[4];
    bool inited_ = false;
    void* emap_ = nullptr; size_t elen_ = 0;

    std::vector<std::string> state_names(int i) const {
        if (types_[i] == "conv") return {"state" + std::to_string(i)};
        std::string s = std::to_string(i);
        return {"past_kh" + s, "past_kl" + s, "past_vh" + s, "past_vl" + s};
    }

    static Graph load(const std::string& dir, const std::string& ctx_onnx) {
        std::string bin = dir + "/" + ctx_onnx.substr(0, ctx_onnx.size() - 5) + "_qnn.bin";
        Graph G; G.g = qr_load(bin.c_str());
        if (G.g < 0) throw std::runtime_error("qr_load " + bin + " -> " + std::to_string(G.g));
        for (int is_out = 0; is_out < 2; is_out++)
            for (int i = 0; i < qr_num(G.g, is_out); i++) {
                char name[256]; uint32_t dims[8], rank; int dt, off; float sc; void* ptr; size_t nb;
                qr_info(G.g, is_out, i, name, 256, dims, &rank, &dt, &sc, &off, &ptr, &nb);
                (is_out ? G.out : G.in)[name] = {i, sc, ptr, nb, dt};
            }
        return G;
    }

    void map_embed(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st; fstat(fd, &st); elen_ = st.st_size;
        emap_ = mmap(nullptr, elen_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (emap_ == MAP_FAILED) throw std::runtime_error("mmap " + path);
        const char* h = (const char*)emap_;
        if (memcmp(h, "\x93NUMPY", 6)) throw std::runtime_error("not a .npy file: " + path);
        size_t hl = h[6] == 1 ? (uint8_t)h[8] | ((uint8_t)h[9] << 8) : *(uint32_t*)(h + 8);
        size_t off = (h[6] == 1 ? 10 : 12) + hl;
        std::string hdr(h + (h[6] == 1 ? 10 : 12), hl);
        if (hdr.find("<f2") == std::string::npos) throw std::runtime_error("embed must be float16");
        nembed = (int)((elen_ - off) / 2 / hid);
        embed_ = h + off;
    }
    const void* embed_ = nullptr;

    static uint16_t q16(float x, float s) {
        float v = nearbyintf(x / s) + 32768.0f;
        return (uint16_t)(v < 0.f ? 0.f : (v > 65535.f ? 65535.f : v));
    }

    float scale_of(const std::string& name) const {
        float s = -1;
        for (auto* v : {&dec_, &pre_})
            for (auto& G : *v) {
                auto it = G.in.find(name);
                if (it == G.in.end()) continue;
                if (s >= 0 && s != it->second.scale) throw std::runtime_error("rope scale mismatch " + name);
                s = it->second.scale;
            }
        return s;
    }

    void setup(const nlohmann::json& c) {
        // q16 rope tables [T][D] (cos, sin) for full and sliding attention, as npu_common.rope_inputs + partial rotary
        const char* names[2][3] = {{"full_attention", "cos_f", "sin_f"}, {"sliding_attention", "cos_s", "sin_s"}};
        for (int a = 0; a < 2; a++) {
            auto& rp = c["rope_parameters"][names[a][0]];
            double theta = rp["rope_theta"].get<double>();
            int ang = (int)(rp.value("partial_rotary_factor", 1.0) * D) / 2, h = D / 2;
            float sc = scale_of(names[a][1]), ss = scale_of(names[a][2]);
            auto& cq = tabs_[2 * a]; auto& sq = tabs_[2 * a + 1];
            cq.resize((size_t)T * D); sq.resize((size_t)T * D);
            for (int p = 0; p < T; p++)
                for (int j = 0; j < D; j++) {
                    int jj = j % h;
                    double inv = 1.0 / std::pow(theta, (2.0 * jj) / D), f = p * inv;
                    float cv = (float)std::cos(f), sv = (float)(std::sin(f) * (j < h ? -1.0 : 1.0));
                    if (jj >= ang) { cv = 1.f; sv = 0.f; }
                    cq[(size_t)p * D + j] = q16(cv, sc); sq[(size_t)p * D + j] = q16(sv, ss);
                }
        }
        nvocab = (int)(head_.out.at("logits").bytes / 2);
        lf_set_model(embed_, hid, T, TS, KV, D, tabs_[0].data(), tabs_[1].data(), tabs_[2].data(), tabs_[3].data(),
                     head_.g, head_.in.at("x").idx, head_.out.at("logits").idx);
        for (size_t k = 0; k < dec_.size(); k++) plan(dec_[k], layers_[k], false);
        if (C) {
            lf_set_prefill(C, c["num_attention_heads"].get<int>() / KV, c["sliding_window"].get<int>());
            for (size_t k = 0; k < pre_.size(); k++) plan(pre_[k], layers_[k], true);
        }
    }

    void plan(const Graph& G, const std::vector<int>& layers, bool prefill) {
        auto idx = [](const std::map<std::string, Tensor>& d, const std::string& n) { auto it = d.find(n); return it == d.end() ? -1 : it->second.idx; };
        auto mq = [&](const std::string& n, float v) { auto it = G.in.find(n); return it == G.in.end() ? 0 : (int)q16(v, it->second.scale); };
        std::vector<int> io = {idx(G.in, "x"), idx(G.in, "cos_f"), idx(G.in, "sin_f"), idx(G.in, "mask_f"), idx(G.in, "cos_s"),
                               idx(G.in, "sin_s"), idx(G.in, "mask_s"), idx(G.out, "x_out"),
                               mq("mask_f", 0.f), mq("mask_f", -100.f), mq("mask_s", 0.f), mq("mask_s", -100.f)};
        std::vector<int> lay;
        for (int i : layers) {
            std::string s = std::to_string(i);
            if (types_[i] == "conv") lay.insert(lay.end(), {0, G.in.at("state" + s).idx, G.out.at("state_new" + s).idx, -1, -1, -1, -1, -1, -1});
            else lay.insert(lay.end(), {types_[i] == "full_attention" ? 1 : 2, -1, -1, G.in.at("past_kh" + s).idx, G.in.at("past_kl" + s).idx,
                                         G.in.at("past_vh" + s).idx, G.in.at("past_vl" + s).idx, G.out.at("k_new" + s).idx, G.out.at("v_new" + s).idx});
        }
        (prefill ? lf_add_prefill_graph : lf_add_graph)(G.g, io.data(), (int)layers.size(), lay.data());
    }
};
