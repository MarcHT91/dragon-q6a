// d1: LiquidAI d1-2B (decision model) on the Radxa Dragon Q6A, one C++ process.
//   pictures: vision encoder on the Adreno GPU (ncnn/Vulkan), the exported shape closest to the picture's aspect ratio
//   language model: QNN HTP stage graphs on the Hexagon NPU (d1/build_d1.py), `set` attention stage by stage
//   readout: final norm + the readout tokens' logits on the CPU, d1's option readout and calibration
// C++ port of d1/d1_prompt.py + d1/d1_npu.py (same prompt, packing, masks, quantization and answers).
//
// usage: d1 --build NPU_BUILD_DIR [--enc ENCODER_DIR] [--qnn QNN_LIB_DIR] [--serve | REQUEST.json]
//   request: {"state": text or JSON or null, "image": "photo.jpg" (or "feat": "features.npy"),
//             "questions": {name: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ..., "multiple": bool}}}
//   answer:  {"answers": {name: ...}, "usage": {...}, "time": {...}}   (as model.system_one)
//   --serve: one request per stdin line, one answer per stdout line (the model stays loaded: one NPU process).
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "net.h"
#include "gpu.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "image.h"
#include "tokenizer.h"

using json = nlohmann::ordered_json;     // keeps the questions' and options' order, as Python dicts do
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

#ifndef QNN_DIR_DEFAULT
#define QNN_DIR_DEFAULT ""
#endif

extern "C" {
int qr_init(const char* backend_so, const char* system_so, int burst);
int qr_load(const char* path);
int qr_num(int g, int is_out);
int qr_info(int g, int is_out, int i, char* name, int name_len, uint32_t* dims, uint32_t* rank, int* dtype,
            float* scale, int* offset, void** ptr, size_t* bytes);
int qr_execute(int g);
void qr_shutdown();
}

static const int IMG = 125030;
static const float MASKV = 100.0f;

static inline uint16_t q16(float x, float s) {
    float v = nearbyintf(x / s) + 32768.0f;
    return (uint16_t)(v < 0.f ? 0.f : (v > 65535.f ? 65535.f : v));
}
static inline void split16(uint16_t q, uint8_t& hi, uint8_t& lo) {
    int v = (int)q - 32768; if (v > 32639) v = 32639;
    int l = ((v + 128) & 255) - 128;
    hi = (uint8_t)(((v - l) >> 8) + 128); lo = (uint8_t)(l + 128);
}

// ------------------------------------------------------------------ .npy (float16 embedding, float32 features)
struct Npy { void* map = nullptr; size_t len = 0; const char* data = nullptr; std::vector<size_t> shape; std::string descr; };
static Npy npy_map(const std::string& path) {
    Npy a; int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd, &st); a.len = st.st_size;
    a.map = mmap(nullptr, a.len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0); close(fd);
    if (a.map == MAP_FAILED) throw std::runtime_error("mmap " + path);
    const char* h = (const char*)a.map;
    if (memcmp(h, "\x93NUMPY", 6)) throw std::runtime_error("not a .npy file: " + path);
    size_t hl = h[6] == 1 ? (uint8_t)h[8] | ((uint8_t)h[9] << 8) : *(uint32_t*)(h + 8);
    size_t off = (h[6] == 1 ? 10 : 12);
    std::string hdr(h + off, hl);
    a.data = h + off + hl;
    auto d = hdr.find("'descr': '"); a.descr = hdr.substr(d + 10, 3);
    auto sp = hdr.find("'shape': ("), se = hdr.find(")", sp);
    std::string sh = hdr.substr(sp + 10, se - sp - 10);
    for (size_t i = 0; i < sh.size();) {
        while (i < sh.size() && !isdigit(sh[i])) i++;
        size_t j = i; while (j < sh.size() && isdigit(sh[j])) j++;
        if (j > i) a.shape.push_back(std::stoul(sh.substr(i, j - i)));
        i = j;
    }
    return a;
}

// ------------------------------------------------------------------ prompt and readout (d1_prompt.py)
class Prompt {
public:
    Tokenizer tok; json cfg, cal; int set_size = 16, scheme = 1;
    explicit Prompt(const std::string& dir) : tok(dir + "/tokenizer.json") {
        cfg = json::parse(std::ifstream(dir + "/config.json"));
        std::ifstream cf(dir + "/calibration.json");
        cal = cf ? json::parse(cf) : json{{"default", 1.0}, {"temperatures", json::object()}, {"tables", json::object()}};
        set_size = cfg.value("d1_set_size", 4); scheme = cfg.value("d1_codes", 1);
        std::vector<std::string> L; for (char c = 'A'; c <= 'Z'; c++) L.push_back(std::string(1, c));
        fallback_ = L;
        for (int i = 0; i < 100; i++) { char b[8]; snprintf(b, 8, "%02d", i); fallback_.push_back(b); }
        for (char c = 'a'; c <= 'z'; c++) fallback_.push_back(std::string(1, c));
        for (int i = 0; i < 200; i++) fallback_.push_back("#" + std::to_string(i));
        for (auto& a : L) for (auto& b : L) fallback_.push_back(a + b);
    }
    std::vector<int> enc(const std::string& s) const { return tok.encode(s); }
    int single(const std::string& s) const { auto e = enc(s); return e.size() == 1 ? e[0] : -1; }

    std::vector<std::pair<std::string, int>> codes(const std::vector<std::string>& labels) const {
        std::vector<std::string> want; bool letters = !labels.empty();
        for (auto& l : labels) { auto t = strip(l); if (!(t.size() == 1 && isalpha((unsigned char)t[0]))) letters = false; }
        if (letters) for (auto& l : labels) want.push_back(strip(l));
        else if (labels.size() <= 26) for (size_t i = 0; i < labels.size(); i++) want.push_back(std::string(1, 'A' + i));
        else if (scheme == 2) for (size_t i = 0; i < labels.size(); i++) want.push_back(i < 26 ? std::string(1, 'A' + i) : std::string(1, 'A' + (i - 26) / 26) + std::string(1, 'A' + (i - 26) % 26));
        else for (size_t i = 0; i < labels.size(); i++) { char b[8]; snprintf(b, 8, "%02d", (int)i); want.push_back(b); }
        std::vector<std::pair<std::string, int>> out; std::set<int> used;
        for (auto& c : want) {
            bool ok = false; std::vector<std::string> tries{c}; tries.insert(tries.end(), fallback_.begin(), fallback_.end());
            for (auto& raw : tries) { int t = single(raw); if (t >= 0 && !used.count(t)) { out.push_back({raw, t}); used.insert(t); ok = true; break; } }
            if (!ok) throw std::runtime_error("no single-token code left");
        }
        return out;
    }
    std::vector<int> forms(const std::vector<std::string>& texts) const {
        std::vector<int> out;
        for (auto& t : texts) { int i = single(t); if (i >= 0 && std::find(out.begin(), out.end(), i) == out.end()) out.push_back(i); }
        return out;
    }
    static std::string state_block(const json& s) {
        return s.is_string() ? s.get<std::string>() + "\n\n" : s.dump(2, ' ', false, json::error_handler_t::replace) + "\n\n";
    }
    std::string prefix(const json& state, const std::string& media) const {
        std::string body = state.is_null() ? "" : state_block(state) + "\nQUESTION:\n";
        return "<|startoftext|><|im_start|>user\n" + media + body;
    }
    static std::string str(const json& v) { return v.is_string() ? v.get<std::string>() : v.is_null() ? "None" : v.dump(); }
    std::vector<std::string> names(const json& crit) const {
        std::vector<std::string> n;
        if (crit.is_object()) for (auto& [k, v] : crit.items()) n.push_back(k);
        else for (auto& v : crit) n.push_back(str(v));
        return n;
    }
    std::string block(const json& q) const {
        std::string t = q.at("type"), said = str(q.at("instructions"));
        bool multiple = q.value("multiple", false);
        if (t == "choice" && multiple) {
            std::string lines;
            for (auto& [k, v] : q.at("criteria").items()) {
                if (!lines.empty()) lines += "\n";
                lines += json(k).dump() + ": " + (v.is_null() || (v.is_string() && v.get<std::string>().empty()) ? k : str(v));
            }
            return said + "\n\nOptions:\n" + lines + "\n\nReply with yes or no for each option.";
        }
        if (t == "choice") {
            auto nm = names(q.at("criteria")); auto cd = codes(nm); std::string lines;
            for (size_t i = 0; i < nm.size(); i++) {
                const json& v = q.at("criteria").at(nm[i]);
                std::string text = (v.is_null() || (v.is_string() && v.get<std::string>().empty())) ? replace_all(nm[i], "_", " ") : str(v);
                lines += (i ? "\n" : "") + cd[i].first + " " + text;
            }
            return said + "\n\nOptions:\n" + lines + "\n\nReply with the option code only.";
        }
        if (t == "noul") {
            std::string extra;
            if (q.contains("criteria") && q["criteria"].is_object() && !q["criteria"].empty())
                extra = "\nYes: " + str(q["criteria"].value("true", json())) + "\nNo: " + str(q["criteria"].value("false", json()));
            return said + extra + "\n\nReply with yes or no only.";
        }
        if (t == "score") {
            auto& c = q.at("criteria"); std::string legend;
            for (size_t i = 0; i < c.size(); i++) legend += (i ? "\n" : "") + std::to_string(i) + " " + str(c[i]);
            return said + "\n\n" + legend + "\n\nReply with a single digit 0-" + std::to_string(c.size() - 1) + " only.";
        }
        throw std::runtime_error("question type '" + t + "' is not supported (noul, choice, score)");
    }
    // (suffix text, slot offsets from its last token)
    std::pair<std::string, std::vector<int>> suffix(const json& q) const {
        std::string text = block(q) + "<|im_end|>\n<|im_start|>assistant\n<think></think>";
        if (!(q.at("type") == "choice" && q.value("multiple", false))) return {text, {0}};
        std::vector<size_t> ends; int i = 0;
        for (auto& [k, v] : q.at("criteria").items()) { text += (i++ ? "\n" : "") + json(k).dump() + ":"; ends.push_back(text.size()); }
        auto ids = enc(text);
        std::vector<size_t> start(ids.size()), stop(ids.size()); size_t pos = 0;      // byte span of each token
        for (size_t t = 0; t < ids.size(); t++) { start[t] = pos; pos += tok.token_bytes(ids[t], true).size(); stop[t] = pos; }
        std::vector<int> slots;
        for (size_t ch : ends) {
            int found = -1;
            for (size_t t = 0; t < ids.size(); t++) if (start[t] <= ch - 1 && ch - 1 < stop[t]) found = (int)t;
            if (found < 0) throw std::runtime_error("slot marker without an exact colon token");
            slots.push_back(found - (int)ids.size() + 1);
        }
        return {text, slots};
    }
    struct Packed { std::vector<int> ids, g; int s; std::vector<std::vector<int>> ends; };
    Packed pack(const std::string& prefix, const std::vector<json>& qs) const {
        Packed P; std::vector<std::pair<std::string, std::vector<int>>> sf;
        for (auto& q : qs) sf.push_back(suffix(q));
        P.ids = enc(prefix + sf[0].first);
        P.s = (int)P.ids.size() - (int)enc(sf[0].first).size();
        P.g.assign(P.s, -1); P.g.resize(P.ids.size(), 1);
        std::vector<int> e; for (int o : sf[0].second) e.push_back((int)P.ids.size() - 1 + o); P.ends.push_back(e);
        bool hdr = prefix.size() >= 10 && prefix.compare(prefix.size() - 10, 10, "QUESTION:\n") == 0;
        for (size_t k = 1; k < sf.size(); k++) {
            auto own = enc((hdr ? "QUESTION:\n" : "") + sf[k].first);
            P.ids.insert(P.ids.end(), own.begin(), own.end()); P.g.resize(P.ids.size(), (int)k + 1);
            std::vector<int> e2; for (int o : sf[k].second) e2.push_back((int)P.ids.size() - 1 + o); P.ends.push_back(e2);
        }
        return P;
    }
    double temperature(const json& q) const {
        std::string t = q.at("type"), b;
        size_t k = q.contains("criteria") ? q["criteria"].size() : 2;
        if (t == "noul") b = "noul";
        else if (t == "score") b = "score:" + std::to_string(k);
        else b = k <= 3 ? "choice:small" : k <= 8 ? "choice:mid" : k <= 26 ? "choice:large" : "choice:inventory";
        auto& T = cal["temperatures"];
        return T.contains(b) ? T[b].get<double>() : cal.value("default", 1.0);
    }
    std::vector<std::vector<int>> groups(const json& q) const {
        if (q.at("type") == "noul" || q.value("multiple", false)) return {forms({"yes", "Yes", "YES"}), forms({"no", "No", "NO"})};
        if (q.at("type") == "score") { std::vector<std::vector<int>> g; for (size_t i = 0; i < q["criteria"].size(); i++) g.push_back(forms({std::to_string(i)})); return g; }
        std::vector<std::vector<int>> g;
        for (auto& [c, t] : codes(names(q.at("criteria")))) {
            std::vector<int> f{t}; for (int i : forms({" " + c})) if (i != t) f.push_back(i); g.push_back(f);
        }
        return g;
    }
    double through(double v, const json& knots) const {
        if (!knots.is_array() || knots.empty()) return v;
        size_t j = 0; while (j < knots.size() && knots[j][0].get<double>() < v) j++;
        if (j == 0) return knots[0][1];
        if (j == knots.size()) return knots.back()[1];
        double a = knots[j - 1][0], pa = knots[j - 1][1], b = knots[j][0], pb = knots[j][1];
        return pa + (v - a) * (pb - pa) / (b - a);
    }
    static std::vector<double> softmax(const std::vector<double>& v) {
        double m = *std::max_element(v.begin(), v.end()), s = 0; std::vector<double> e;
        for (double x : v) { e.push_back(std::exp(x - m)); s += e.back(); }
        for (auto& x : e) x /= s;
        return e;
    }
private:
    std::vector<std::string> fallback_;
    static std::string strip(const std::string& s) { size_t a = s.find_first_not_of(" \t\n\r"), b = s.find_last_not_of(" \t\n\r"); return a == std::string::npos ? "" : s.substr(a, b - a + 1); }
    static std::string replace_all(std::string s, const std::string& a, const std::string& b) { size_t p = 0; while ((p = s.find(a, p)) != std::string::npos) { s.replace(p, a.size(), b); p += b.size(); } return s; }
};

// ------------------------------------------------------------------ NPU stages (d1_npu.py)
class Npu {
public:
    struct Tensor { int idx = -1; float scale = 1; void* ptr = nullptr; size_t bytes = 0; int dtype = 0; std::vector<uint32_t> dims; };
    struct Stage { int g; std::vector<int> layers; bool kv_last; std::map<std::string, Tensor> in, out; };
    int T, TS, C, hid, KV, D, H, rep, win, nembed; float eps;
    std::vector<std::string> types; std::vector<Stage> st; Npy embed; std::vector<float> fnorm; double load_s = 0;
    struct Rope { double theta; int ang; }; std::map<std::string, Rope> rope;

    Npu(const std::string& dir, const std::string& qnn) {
        auto t0 = clk::now();
        json man = json::parse(std::ifstream(dir + "/manifest.json"));
        auto& c = man["cfg"];
        T = man["T"]; TS = man["TS"]; C = man["C"]; hid = c["hidden_size"]; KV = c["num_key_value_heads"]; D = c["head_dim"];
        H = c["num_attention_heads"]; rep = H / KV; win = c["sliding_window"]; eps = c["norm_eps"];
        for (auto& t : c["layer_types"]) types.push_back(t);
        for (auto& [typ, rp] : c["rope_parameters"].items())
            rope[typ] = {rp["rope_theta"].get<double>(), (int)(rp.value("partial_rotary_factor", 1.0) * D) / 2};
        setenv("ADSP_LIBRARY_PATH", qnn.c_str(), 0);
        if (!dlopen((qnn + "/libQnnHtpV68Stub.so").c_str(), RTLD_NOW | RTLD_GLOBAL)) throw std::runtime_error(std::string("dlopen stub: ") + dlerror());
        if (qr_init((qnn + "/libQnnHtp.so").c_str(), (qnn + "/libQnnSystem.so").c_str(), 1)) throw std::runtime_error("qr_init failed");
        inited_ = true;
        for (auto& s : man["stages"]) {
            Stage S; std::string ctx = s["ctx"]; ctx = dir + "/" + ctx.substr(0, ctx.size() - 5) + "_qnn.bin";
            S.g = qr_load(ctx.c_str()); if (S.g < 0) throw std::runtime_error("qr_load " + ctx);
            S.layers = s["layers"].get<std::vector<int>>(); S.kv_last = s["kv_only_last"];
            for (int o = 0; o < 2; o++)
                for (int i = 0; i < qr_num(S.g, o); i++) {
                    char nm[256]; uint32_t dims[8], rank; Tensor t; int off;
                    qr_info(S.g, o, i, nm, 256, dims, &rank, &t.dtype, &t.scale, &off, &t.ptr, &t.bytes);
                    t.idx = i; t.dims.assign(dims, dims + rank); (o ? S.out : S.in)[nm] = t;
                }
            st.push_back(S);
        }
        load_s = since(t0);
        embed = npy_map(dir + "/embed_f16.npy"); nembed = (int)embed.shape[0];
        Npy fn = npy_map(dir + "/fnorm.npy"); fnorm.assign((const float*)fn.data, (const float*)fn.data + hid); munmap(fn.map, fn.len);
    }
    ~Npu() { if (inited_) qr_shutdown(); if (embed.map) munmap(embed.map, embed.len); }

    double exec_s = 0;           // time inside graph execution (the rest of hidden() is host work)
    float embed_val(int tok, int j) const { return (float)((const __fp16*)embed.data)[(size_t)tok * hid + j]; }

    // final hidden states [n][hid] under `set` attention; feat: [N_img][hid] rows for the <|image|> tokens
    std::vector<float> hidden(const std::vector<int>& ids, int s, const std::vector<int>& g, const std::vector<float>* feat, double& npu_s) {
        auto t0 = clk::now();
        int n = (int)ids.size(), N = (n + C - 1) / C * C;
        if (N > T) throw std::runtime_error(std::to_string(n) + " tokens: the build's context is " + std::to_string(T));
        std::vector<float> x((size_t)N * hid, 0.f);
        int k_img = 0;
        for (int t = 0; t < N; t++) {
            int id = t < n ? ids[t] : 124893;                                 // <|pad|>
            float* r = &x[(size_t)t * hid];
            if (id == IMG && feat && k_img * hid < (int)feat->size()) { memcpy(r, &(*feat)[(size_t)k_img * hid], hid * 4); k_img++; }
            else for (int j = 0; j < hid; j++) r[j] = embed_val(id, j);
        }
        reset();
        // masks per chunk (the same for every stage): allowed 32768, masked 1 (scale MASK/32767, zero point 32768)
        std::map<int, std::vector<uint16_t>> mf, ms;
        for (int p = 0; p < N; p += C) {
            std::vector<uint16_t> m((size_t)C * (T + C), 1);
            for (int t = 0; t < C; t++) {
                int q = p + t; uint16_t* row = &m[(size_t)t * (T + C)];
                for (int k = 0; k < n; k++) row[k] = (q >= n || q < s || k < s || g[q] == g[k]) ? 32768 : 1;
            }
            auto& M = mf[p]; M.resize((size_t)rep * C * (T + C));
            for (int r = 0; r < rep; r++) memcpy(&M[(size_t)r * C * (T + C)], m.data(), m.size() * 2);
            std::vector<uint16_t> w((size_t)C * (TS + C), 1);
            for (int t = 0; t < C; t++) {
                uint16_t* row = &w[(size_t)t * (TS + C)];
                for (int sl = 0; sl < TS; sl++) {
                    long last = sl < p ? sl + (long)((p - 1 - sl) / TS) * TS : -1000000000L;
                    if (last >= 0 && last > p + t - win) row[sl] = 32768;
                }
                for (int u = 0; u <= t; u++) row[TS + u] = 32768;
            }
            auto& W = ms[p]; W.resize((size_t)rep * C * (TS + C));
            for (int r = 0; r < rep; r++) memcpy(&W[(size_t)r * C * (TS + C)], w.data(), w.size() * 2);
        }
        std::vector<float> out((size_t)N * hid);
        std::vector<uint16_t> xq((size_t)N * hid);
        for (size_t k = 0; k < st.size(); k++) {
            Stage& S = st[k]; Stage* nxt = k + 1 < st.size() ? &st[k + 1] : nullptr;
            float xs = S.in.at("x").scale;
            for (size_t i = 0; i < xq.size(); i++) xq[i] = q16(x[i], xs);
            for (int p = 0; p < N; p += C) {
                memcpy(S.in.at("x").ptr, &xq[(size_t)p * hid], (size_t)C * hid * 2);
                for (auto [typ, sfx] : {std::pair<const char*, const char*>{"full_attention", "f"}, {"sliding_attention", "s"}}) {
                    auto it = S.in.find(std::string("cos_") + sfx);
                    if (it == S.in.end()) continue;
                    auto& cs = rope_q(typ, it->second.scale, N, true); auto& sn = rope_q(typ, S.in.at(std::string("sin_") + sfx).scale, N, false);
                    memcpy(it->second.ptr, &cs[(size_t)p * D], (size_t)C * D * 2);
                    memcpy(S.in.at(std::string("sin_") + sfx).ptr, &sn[(size_t)p * D], (size_t)C * D * 2);
                }
                if (S.in.count("mask_f")) memcpy(S.in["mask_f"].ptr, mf[p].data(), mf[p].size() * 2);
                if (S.in.count("mask_s")) memcpy(S.in["mask_s"].ptr, ms[p].data(), ms[p].size() * 2);
                auto te = clk::now();
                if (qr_execute(S.g)) throw std::runtime_error("graph execute failed");
                exec_s += since(te);
                const uint16_t* xo = (const uint16_t*)S.out.at("x_out").ptr; float os = S.out.at("x_out").scale;
                for (size_t i = 0; i < (size_t)C * hid; i++) out[(size_t)p * hid + i] = ((int)xo[i] - 32768) * os;
                for (int i : S.layers) {
                    std::string si = std::to_string(i);
                    if (types[i] == "conv") { memcpy(S.in.at("state" + si).ptr, S.out.at("state_new" + si).ptr, S.in.at("state" + si).bytes); continue; }
                    bool kvl = S.kv_last && i == S.layers.back();
                    if (!kvl && types[i] != "sliding_attention") continue;
                    Stage& Dst = kvl ? *nxt : S;
                    int nslot = kvl ? T : TS;
                    const uint16_t* kn = (const uint16_t*)S.out.at("k_new" + si).ptr; const uint16_t* vn = (const uint16_t*)S.out.at("v_new" + si).ptr;
                    uint8_t* kh = (uint8_t*)Dst.in.at("past_kh" + si).ptr; uint8_t* kl = (uint8_t*)Dst.in.at("past_kl" + si).ptr;
                    uint8_t* vh = (uint8_t*)Dst.in.at("past_vh" + si).ptr; uint8_t* vl = (uint8_t*)Dst.in.at("past_vl" + si).ptr;
                    for (int h = 0; h < KV; h++)
                        for (int t = 0; t < C; t++) {
                            int slot = kvl ? p + t : (p + t) % TS;
                            for (int d = 0; d < D; d++) {
                                size_t src = ((size_t)h * C + t) * D + d;
                                uint8_t a, b;
                                split16(kn[src], a, b); kh[((size_t)h * D + d) * nslot + slot] = a; kl[((size_t)h * D + d) * nslot + slot] = b;
                                split16(vn[src], a, b); vh[((size_t)h * nslot + slot) * D + d] = a; vl[((size_t)h * nslot + slot) * D + d] = b;
                            }
                        }
                }
            }
            x.swap(out);
        }
        x.resize((size_t)n * hid);
        npu_s = since(t0);
        return x;
    }

private:
    bool inited_ = false;
    std::map<std::tuple<std::string, float, bool>, std::vector<uint16_t>> rope_cache_;
    void reset() {
        for (auto& S : st)
            for (auto& [nm, t] : S.in) {
                if (nm.rfind("state", 0) == 0) { auto* p = (uint16_t*)t.ptr; for (size_t i = 0; i < t.bytes / 2; i++) p[i] = 32768; }
                else if (nm.rfind("past_", 0) == 0) memset(t.ptr, 128, t.bytes);
            }
    }
    const std::vector<uint16_t>& rope_q(const std::string& typ, float scale, int N, bool cos_) {
        auto key = std::make_tuple(typ, scale, cos_);
        auto& v = rope_cache_[key];
        if ((int)v.size() >= N * D) return v;
        const Rope& R = rope.at(typ); int h = D / 2;
        v.resize((size_t)T * D);
        for (int p = 0; p < T; p++)
            for (int j = 0; j < D; j++) {
                int jj = j % h;
                double f = p * (1.0 / std::pow(R.theta, (2.0 * jj) / D));
                float cv = (float)std::cos(f), sv = (float)(std::sin(f) * (j < h ? -1.0 : 1.0));
                if (jj >= R.ang) { cv = 1.f; sv = 0.f; }
                v[(size_t)p * D + j] = q16(cos_ ? cv : sv, scale);
            }
        return v;
    }
};

// ------------------------------------------------------------------ vision encoder (GPU)
class Encoder {
public:
    explicit Encoder(const std::string& dir) : dir_(dir) {
        DIR* d = opendir(dir.c_str());
        if (!d) throw std::runtime_error("no encoder dir " + dir);
        while (auto* e = readdir(d)) {
            int H, W; char tail[64];
            if (sscanf(e->d_name, "vlenc_%dx%d.ncnn.%63s", &H, &W, tail) == 3 && std::string(tail) == "bin") shapes_.push_back({H, W});
        }
        closedir(d);
        if (shapes_.empty()) throw std::runtime_error("no vlenc_HxW.ncnn.bin in " + dir);
        for (int i = 0; i < ncnn::get_gpu_count(); i++) if (ncnn::get_gpu_info(i).type() != 3) { dev_ = i; break; }
        if (dev_ < 0) throw std::runtime_error("no Vulkan GPU");
    }
    std::pair<int, int> pick(int w, int h) const {          // the exported (H, W) with the closest aspect ratio
        double a = std::log((double)w / h); auto best = shapes_[0];
        for (auto& s : shapes_) if (std::fabs(std::log((double)s.second / s.first) - a) < std::fabs(std::log((double)best.second / best.first) - a)) best = s;
        return best;
    }
    // load + one run (Vulkan pipeline creation): the first real picture of that shape then runs at full speed
    bool warm(int H, int W) {
        if (std::find(shapes_.begin(), shapes_.end(), std::make_pair(H, W)) == shapes_.end()) return false;
        RGB8 z; z.w = W; z.h = H; z.px.assign((size_t)W * H * 3, 128); int n; std::string sh;
        run(z, n, sh); run(z, n, sh); return true;
    }
    std::vector<float> run(const RGB8& im, int& n_tok, std::string& shape) {
        auto [H, W] = pick(im.w, im.h);
        auto x = preprocess(im, W, H);
        ncnn::Net& n = net(H, W);
        ncnn::Mat in(W, H, 3, x.data());
        ncnn::Extractor ex = n.create_extractor(); ex.input("in0", in);
        ncnn::Mat out; if (ex.extract("out0", out)) throw std::runtime_error("encoder failed");
        int F = out.dims == 3 ? out.c : out.h, N = out.dims == 3 ? out.w * out.h : out.w;
        std::vector<float> r((size_t)N * F);
        for (int f = 0; f < F; f++) { const float* s = out.dims == 3 ? (const float*)out.channel(f) : out.row(f); for (int t = 0; t < N; t++) r[(size_t)t * F + f] = s[t]; }
        n_tok = N; shape = std::to_string(W) + "x" + std::to_string(H);
        return r;
    }
private:
    std::string dir_; std::vector<std::pair<int, int>> shapes_; int dev_ = -1;
    std::map<std::pair<int, int>, std::unique_ptr<ncnn::Net>> nets_;
    ncnn::Net& net(int H, int W) {
        auto& p = nets_[{H, W}];
        if (!p) {
            p.reset(new ncnn::Net); p->set_vulkan_device(dev_);
            p->opt.use_vulkan_compute = true; p->opt.use_subgroup_ops = false; p->opt.lightmode = false;
            std::string b = dir_ + "/vlenc_" + std::to_string(H) + "x" + std::to_string(W) + ".ncnn";
            if (p->load_param((b + ".param").c_str()) || p->load_model((b + ".bin").c_str())) throw std::runtime_error("cannot load " + b);
        }
        return *p;
    }
};

// ------------------------------------------------------------------ one request
static json answer(Prompt& P, Npu& M, Encoder* E, const json& req) {
    auto t0 = clk::now(); double exec0 = M.exec_s;
    json state = req.contains("state") ? req["state"] : json();
    std::vector<float> feat; bool has_img = false; std::string shape; double enc_s = 0;
    if (req.contains("feat")) {
        Npy a = npy_map(req["feat"]);
        if (a.descr != "<f4") throw std::runtime_error("feat must be float32");
        feat.assign((const float*)a.data, (const float*)a.data + a.shape[0] * a.shape[1]); munmap(a.map, a.len); has_img = true;
    } else if (req.contains("image") && !req["image"].is_null()) {
        if (!E) throw std::runtime_error("an image needs --enc");
        auto te = clk::now(); RGB8 im = load_rgb(req["image"]); int nt;
        feat = E->run(im, nt, shape); enc_s = since(te); has_img = true;
    }
    int n_img = has_img ? (int)(feat.size() / M.hid) : 0;
    std::string media;
    if (has_img) { media = "<|vision_start|>"; for (int i = 0; i < n_img; i++) media += "<|image|>"; media += "<|vision_end|>"; }
    std::string prefix = P.prefix(state, media);
    std::vector<std::string> names; std::vector<json> qs;
    for (auto& [k, v] : req.at("questions").items()) { names.push_back(k); qs.push_back(v); }
    json out = json::object(); int tokens = 0; double npu_s = 0;
    for (size_t k0 = 0; k0 < qs.size(); k0 += P.set_size) {
        std::vector<json> part(qs.begin() + k0, qs.begin() + std::min(qs.size(), k0 + P.set_size));
        auto pk = P.pack(prefix, part);
        double ns; auto x = M.hidden(pk.ids, pk.s, pk.g, has_img ? &feat : nullptr, ns); npu_s += ns; tokens += (int)pk.ids.size();
        for (size_t qi = 0; qi < part.size(); qi++) {
            const json& q = part[qi]; auto grp = P.groups(q);
            std::vector<std::vector<double>> rows;                       // per slot: the groups' best raw logits
            for (int slot : pk.ends[qi]) {
                const float* r = &x[(size_t)slot * M.hid]; double ss = 0;
                for (int j = 0; j < M.hid; j++) ss += (double)r[j] * r[j];
                double inv = 1.0 / std::sqrt(ss / M.hid + M.eps);
                std::vector<double> best;
                for (auto& gr : grp) {
                    double b = -1e300;
                    for (int t : gr) { double z = 0; for (int j = 0; j < M.hid; j++) z += (double)((float)(r[j] * inv) * M.fnorm[j]) * M.embed_val(t, j); b = std::max(b, z); }
                    best.push_back(b);
                }
                rows.push_back(best);
            }
            std::string nm = names[k0 + qi]; auto cr = q.contains("criteria") ? P.names(q["criteria"]) : std::vector<std::string>{};
            if (q.value("multiple", false)) {
                json probs = json::object(), ch = json::array(); size_t i = 0;
                for (auto& [kname, v] : q["criteria"].items()) {
                    double py = P.through(Prompt::softmax(rows[i++])[0], P.cal["tables"].value("multiple", json()));
                    probs[kname] = py; if (py >= 0.5) ch.push_back(kname);
                }
                out[nm] = {{"type", "choice"}, {"choices", ch}, {"probabilities", probs}};
                continue;
            }
            double T = P.temperature(q); std::vector<double> sc; for (double v : rows[0]) sc.push_back(v / T);
            auto p = Prompt::softmax(sc);
            if (q["type"] == "noul") out[nm] = {{"type", "noul"}, {"noul", p[0]}};
            else if (q["type"] == "choice") {
                size_t b = std::max_element(p.begin(), p.end()) - p.begin(); json probs = json::object();
                for (size_t i = 0; i < cr.size(); i++) probs[cr[i]] = p[i];
                out[nm] = {{"type", "choice"}, {"choice", cr[b]}, {"confidence", p[b]}, {"probabilities", probs}};
            } else {
                double acc = 0; int lvl = (int)p.size() - 1;
                for (size_t i = 0; i < p.size(); i++) { acc += p[i]; if (acc >= 0.5) { lvl = (int)i; break; } }
                json legend = json::object(); for (size_t i = 0; i < q["criteria"].size(); i++) legend[std::to_string(i)] = q["criteria"][i];
                out[nm] = {{"type", "score"}, {"score", (double)lvl}, {"probabilities", p}, {"legend", legend}};
            }
        }
    }
    json r = {{"answers", out}, {"usage", {{"input_tokens", tokens}, {"output_tokens", 0}, {"image_tokens", n_img}}},
              {"time", {{"total_s", since(t0)}, {"npu_s", npu_s}, {"npu_exec_s", M.exec_s - exec0}, {"gpu_s", enc_s}}}};
    if (!shape.empty()) r["usage"]["image_shape"] = shape;
    return r;
}

int main(int argc, char** argv) {
    std::string build = getenv("D1_BUILD") ? getenv("D1_BUILD") : "", enc = getenv("D1_ENC") ? getenv("D1_ENC") : "";
    std::string qnn = getenv("QNN_DIR") ? getenv("QNN_DIR") : QNN_DIR_DEFAULT, reqf; bool serve = false;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&]() { if (i + 1 >= argc) throw std::runtime_error(k + " needs a value"); return std::string(argv[++i]); };
        if (k == "--build") build = val(); else if (k == "--enc") enc = val(); else if (k == "--qnn") qnn = val();
        else if (k == "--serve") serve = true; else reqf = k;
    }
    if (build.empty() || (!serve && reqf.empty())) {
        fprintf(stderr, "usage: d1 --build NPU_BUILD_DIR [--enc ENCODER_DIR] [--qnn DIR] (--serve | REQUEST.json)\n"); return 2;
    }
    try {
        auto t0 = clk::now();
        Prompt P(build);
        std::unique_ptr<Encoder> E;
        if (!enc.empty()) {
            ncnn::create_gpu_instance(); E.reset(new Encoder(enc));
            if (serve) for (auto [h, w] : {std::pair<int, int>{384, 512}, {512, 384}}) E->warm(h, w);   // the common 4:3 / 3:4
        }
        // the QNN / fastrpc libraries print on stdout at init and shutdown: keep stdout for the answers
        fflush(stdout); int saved = dup(1), dn = open("/dev/null", O_WRONLY); dup2(dn, 1); close(dn);
        auto M = std::make_unique<Npu>(build, qnn);
        fflush(stdout); dup2(saved, 1);
        fprintf(stderr, "[d1: NPU stages loaded in %.1f s, ready in %.1f s, T=%d C=%d]\n", M->load_s, since(t0), M->T, M->C);
        if (serve) {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
                json r;
                try { r = answer(P, *M, E.get(), json::parse(line)); } catch (std::exception& e) { r = {{"error", e.what()}}; }
                std::cout << r.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
            }
        } else {
            std::cout << answer(P, *M, E.get(), json::parse(std::ifstream(reqf))).dump(1, ' ', false, json::error_handler_t::replace) << std::endl;
        }
        fflush(stdout); saved = dup(1); dn = open("/dev/null", O_WRONLY); dup2(dn, 1); close(dn);
        M.reset(); E.reset();
        if (!enc.empty()) ncnn::destroy_gpu_instance();
        fflush(stdout); dup2(saved, 1);
    } catch (std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}
