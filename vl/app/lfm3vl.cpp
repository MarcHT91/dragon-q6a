// lfm3vl: LFM3-VL-2B chat on the Radxa Dragon Q6A, all in one C++ process.
//   vision encoder + projector: ncnn on the Adreno GPU (Vulkan, Mesa Turnip)  -> image features [N, 2048]
//   language model:             QNN HTP on the Hexagon NPU (npu/native/qnnrun.cpp), the features injected as extra
//                               embedding rows at the <|image|> positions
// While the GPU encodes the image, the NPU already consumes the prompt tokens that come before it.
//
// usage: lfm3vl --build NPU_BUILD_DIR --enc ENCODER_DIR [--size 384] [--image FILE] [-p PROMPT] [-n 512]
//               [--temp 0] [--top-p 0.9] [--think] [--system TEXT] [--qnn QNN_LIB_DIR] [--seed N]
// Without -p it is an interactive chat; commands: /image FILE (attach to the next message), /reset, /quit.
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "net.h"
#include "gpu.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "image.h"
#include "npu.h"
#include "tokenizer.h"

#ifndef QNN_DIR_DEFAULT
#define QNN_DIR_DEFAULT ""
#endif

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

// The QNN / fastrpc libraries print debug lines on stdout at init and shutdown; keep stdout for the answers.
struct MuteStdout {
    int saved;
    MuteStdout() { fflush(stdout); saved = dup(1); int n = open("/dev/null", O_WRONLY); dup2(n, 1); close(n); }
    ~MuteStdout() { fflush(stdout); dup2(saved, 1); close(saved); }
};

// ---------------------------------------------------------------- vision encoder (GPU)
class Encoder {
public:
    int size;
    std::string device;
    Encoder(const std::string& dir, int size) : size(size) {
        int dev = -1;                           // the Adreno, not a CPU Vulkan device such as llvmpipe
        for (int i = 0; i < ncnn::get_gpu_count(); i++)
            if (ncnn::get_gpu_info(i).type() != 3) { dev = i; break; }
        if (dev < 0) throw std::runtime_error("no Vulkan GPU found");
        net_.set_vulkan_device(dev);
        device = ncnn::get_gpu_info(dev).device_name();
        net_.opt.use_vulkan_compute = true;
        net_.opt.use_subgroup_ops = false;      // subgroup Gemm hangs Turnip on a6xx
        net_.opt.lightmode = false;             // blob recycling + fp16 storage corrupts results on Vulkan
        std::string base = dir + "/vlenc_" + std::to_string(size) + "x" + std::to_string(size) + ".ncnn";
        if (net_.load_param((base + ".param").c_str()) || net_.load_model((base + ".bin").c_str()))
            throw std::runtime_error("cannot load encoder " + base + ".{param,bin}");
    }
    // planar float [3][size][size] -> fp16 features [N][2048]
    std::vector<uint16_t> run(std::vector<float>& img, int& n_tok) {
        ncnn::Mat in(size, size, 3, img.data());
        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        if (ex.extract("out0", out)) throw std::runtime_error("encoder failed");
        int F, N;
        if (out.dims == 3) { F = out.c; N = out.w * out.h; } else { F = out.h; N = out.w; }
        std::vector<uint16_t> r((size_t)N * F);
        __fp16* h = (__fp16*)r.data();
        for (int f = 0; f < F; f++) {
            const float* src = out.dims == 3 ? (const float*)out.channel(f) : out.row(f);
            for (int t = 0; t < N; t++) h[(size_t)t * F + f] = (__fp16)src[t];
        }
        n_tok = N;
        return r;
    }
private:
    ncnn::Net net_;
};

// ---------------------------------------------------------------- sampling
static int sample(const NpuLM& lm, const uint16_t* lq, float temp, float top_p, std::mt19937& rng) {
    int n = lm.nvocab, best = 0;
    for (int i = 1; i < n; i++) if (lq[i] > lq[best]) best = i;
    if (temp <= 0) return best;
    std::vector<std::pair<float, int>> c;
    float mx = lm.logit(lq, best);
    for (int i = 0; i < n; i++) {
        float p = std::exp((lm.logit(lq, i) - mx) / temp);
        if (p > 1e-6f) c.push_back({p, i});
    }
    std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    float tot = 0; for (auto& x : c) tot += x.first;
    float acc = 0; size_t k = 0;
    for (; k < c.size(); k++) { acc += c[k].first / tot; if (acc >= top_p) { k++; break; } }
    c.resize(std::max<size_t>(k, 1));
    float t2 = 0; for (auto& x : c) t2 += x.first;
    float r = std::uniform_real_distribution<float>(0, t2)(rng);
    for (auto& x : c) { r -= x.first; if (r <= 0) return x.second; }
    return c.back().second;
}

// ---------------------------------------------------------------- chat
struct Args {
    std::string build = getenv("LFM3VL_BUILD") ? getenv("LFM3VL_BUILD") : "";
    std::string enc = getenv("LFM3VL_ENC") ? getenv("LFM3VL_ENC") : "";
    std::string qnn = getenv("QNN_DIR") ? getenv("QNN_DIR") : QNN_DIR_DEFAULT;
    std::string image, prompt, system;
    int size = 384, max_new = 512; unsigned seed = 0;
    float temp = 0.f, top_p = 0.9f;
    bool think = false, quiet = false;
};

class Chat {
public:
    Chat(const Args& a, NpuLM& lm, Tokenizer& tok, std::shared_future<Encoder*> enc) : a_(a), lm_(lm), tok_(tok), enc_(enc), rng_(a.seed) {
        img_id_ = tok.id("<|image|>");
        stop_ = {tok.id("<|im_end|>")};
        for (int e : lm.eos) stop_.push_back(e);
        reset();
    }
    void reset() {
        lm_.reset(); started_ = false; pending_ = "<|startoftext|>";
        if (!a_.system.empty()) pending_ += "<|im_start|>system\n" + a_.system + "<|im_end|>\n";
    }

    // One user turn (optionally with an image): streams the answer to stdout.
    void turn(const std::string& text, const std::string& image) {
        auto t0 = clk::now();
        std::string pre = pending_ + "<|im_start|>user\n";
        std::string post = text + "<|im_end|>\n<|im_start|>assistant\n" + (a_.think ? "<think>" : "<think></think>");
        std::future<std::vector<uint16_t>> feats;
        int n_img = 0; double enc_s = 0;
        if (!image.empty()) {
            RGB8 im = load_rgb(image);                 // read errors surface here, before touching the NPU state
            pre += "<|vision_start|>";
            post = "<|vision_end|>" + post;
            feats = std::async(std::launch::async, [this, im, &n_img, &enc_s]() {
                Encoder* e = enc_.get();
                auto t = clk::now();
                auto x = preprocess(im, e->size, e->size);
                double tpp = since(t);
                auto f = e->run(x, n_img);
                enc_s = since(t);
                if (getenv("LFM3VL_DEBUG")) fprintf(stderr, "[preprocess %.0f ms, encoder %.0f ms]\n", tpp * 1e3, (enc_s - tpp) * 1e3);
                return f;
            });
        }
        auto ids_pre = tok_.encode(pre), ids_post = tok_.encode(post);
        int n_exp = image.empty() ? 0 : (a_.size / 32) * (a_.size / 32);     // 16 px patches, 2x2 merge
        int L = (int)(ids_pre.size() + n_exp + ids_post.size());
        if (lm_.pos > 0 && lm_.pos + L + 256 > lm_.T) {
            fprintf(stderr, "[context nearly full (%d/%d): resetting the conversation]\n", lm_.pos, lm_.T);
            reset(); return turn(text, image);
        }
        // The prefill graphs take exactly C tokens and the last prompt token must be a single step (it gives the
        // logits), so (L-1) % C tokens go as single steps. Do those first, on the tokens before the image, while
        // the GPU is still encoding; then everything else is whole chunks plus one final step.
        auto tp = clk::now();
        int r0 = lm_.C ? (L - 1) % lm_.C : 0;
        size_t head = image.empty() ? 0 : std::min<size_t>(r0, ids_pre.size());
        for (size_t k = 0; k < head; k++) lm_.step(ids_pre[k]);
        std::vector<int> rest(ids_pre.begin() + head, ids_pre.end());
        if (!image.empty()) {
            img_rows_ = feats.get();
            if (n_img != n_exp) throw std::runtime_error("encoder gave " + std::to_string(n_img) + " tokens");
            lm_.set_extra(&img_rows_);
            for (int k = 0; k < n_img; k++) rest.push_back(lm_.nembed + k);
        }
        rest.insert(rest.end(), ids_post.begin(), ids_post.end());
        const uint16_t* lq = lm_.feed(rest, true);
        lm_.set_extra(nullptr);
        double pre_s = since(tp);
        int n_prompt = L;

        auto td = clk::now();
        int n_out = 0; bool stopped = false;
        for (; n_out < a_.max_new; n_out++) {
            int t = sample(lm_, lq, a_.temp, a_.top_p, rng_);
            if (std::find(stop_.begin(), stop_.end(), t) != stop_.end()) { stopped = true; break; }
            std::string s = tok_.token_bytes(t);
            fwrite(s.data(), 1, s.size(), stdout); fflush(stdout);
            if (lm_.pos >= lm_.T - 2) break;
            lq = lm_.step(t);
        }
        double dec_s = since(td);
        printf("\n"); fflush(stdout);
        (void)stopped;
        pending_ = "<|im_end|>\n";               // the end token is not fed yet (stopped or cut at -n)
        if (!a_.quiet)
            fprintf(stderr, "[%s%d prompt tok in %.2f s | %d tok at %.1f tok/s | total %.2f s | ctx %d/%d]\n",
                    image.empty() ? "" : ("image " + std::to_string(n_img) + " tok, GPU " + fmt(enc_s) + " s | ").c_str(),
                    n_prompt, pre_s, n_out, n_out / std::max(dec_s, 1e-9), since(t0), lm_.pos, lm_.T);
    }

private:
    const Args& a_; NpuLM& lm_; Tokenizer& tok_; std::shared_future<Encoder*> enc_;
    std::mt19937 rng_;
    int img_id_; std::vector<int> stop_;
    std::string pending_; bool started_ = false;
    std::vector<uint16_t> img_rows_;
    static std::string fmt(double v) { char b[32]; snprintf(b, sizeof b, "%.2f", v); return b; }
};

static void usage() {
    fprintf(stderr,
            "usage: lfm3vl --build NPU_BUILD_DIR --enc ENCODER_DIR [--size 256|384|512] [--image FILE] [-p PROMPT]\n"
            "              [-n MAX_NEW] [--temp T] [--top-p P] [--seed N] [--think] [--system TEXT] [--qnn DIR] [-q]\n"
            "  (defaults: $LFM3VL_BUILD, $LFM3VL_ENC, $QNN_DIR) Without -p: interactive chat, /image FILE, /reset, /quit\n");
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (k == "--build") a.build = val();
        else if (k == "--enc") a.enc = val();
        else if (k == "--qnn") a.qnn = val();
        else if (k == "--size") a.size = std::stoi(val());
        else if (k == "--image") a.image = val();
        else if (k == "-p") a.prompt = val();
        else if (k == "-n") a.max_new = std::stoi(val());
        else if (k == "--temp") a.temp = std::stof(val());
        else if (k == "--top-p") a.top_p = std::stof(val());
        else if (k == "--seed") a.seed = std::stoul(val());
        else if (k == "--system") a.system = val();
        else if (k == "--think") a.think = true;
        else if (k == "-q") a.quiet = true;
        else { usage(); return 2; }
    }
    if (a.build.empty() || a.enc.empty() || a.qnn.empty()) { usage(); return 2; }

    try {
        auto t0 = clk::now();
        // GPU: Vulkan init, encoder load and a warm-up run (shader compilation) in the background
        std::promise<Encoder*> pe;
        std::shared_future<Encoder*> enc = pe.get_future().share();
        std::thread gpu([&]() {
            try {
                ncnn::create_gpu_instance();
                auto* e = new Encoder(a.enc, a.size);
                std::vector<float> z((size_t)3 * a.size * a.size, 0.f); int n;
                auto t = clk::now(); e->run(z, n); e->run(z, n);   // 2 runs: the first real one is then at full speed
                if (!a.quiet) fprintf(stderr, "[GPU encoder %dx%d on %s ready, warm-up %.1f s]\n", a.size, a.size, e->device.c_str(), since(t));
                pe.set_value(e);
            } catch (...) { pe.set_exception(std::current_exception()); }
        });
        Tokenizer tok(a.build + "/tokenizer.json");
        std::unique_ptr<MuteStdout> mute(new MuteStdout);
        auto lmp = std::make_unique<NpuLM>(a.build, a.qnn);
        NpuLM& lm = *lmp;
        mute.reset();
        if (!a.quiet) fprintf(stderr, "[NPU model loaded in %.1f s, ready in %.1f s]\n", lm.load_s, since(t0));
        Chat chat(a, lm, tok, enc);

        if (!a.prompt.empty()) chat.turn(a.prompt, a.image);
        else {
            std::string pending_img = a.image, line;
            fprintf(stderr, "Commands: /image FILE (attach to the next message), /reset, /quit\n");
            while (true) {
                fprintf(stderr, pending_img.empty() ? "> " : "[image] > ");
                if (!std::getline(std::cin, line) || line == "/quit") break;
                if (line.empty()) continue;
                if (line.rfind("/image ", 0) == 0) { pending_img = line.substr(7); fprintf(stderr, "[image %s attached]\n", pending_img.c_str()); continue; }
                if (line == "/reset") { chat.reset(); pending_img.clear(); fprintf(stderr, "[conversation reset]\n"); continue; }
                try { chat.turn(line, pending_img); } catch (std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); }
                pending_img.clear();
            }
        }
        gpu.join();
        mute.reset(new MuteStdout);
        lmp.reset();
        try { delete enc.get(); } catch (...) {}
        ncnn::destroy_gpu_instance();
    } catch (std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
