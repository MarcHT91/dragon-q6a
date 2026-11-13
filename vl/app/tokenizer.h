// Byte-level BPE tokenizer reading a HuggingFace tokenizer.json (the LFM3 one: Split regex pre-tokenizer +
// ByteLevel, BPE with ignore_merges, added/special tokens matched before pre-tokenization).
// encode() matches `tokenizers` (add_special_tokens=False); decode() skips special tokens unless asked.
#pragma once
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <algorithm>
#include <climits>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "third_party/json.hpp"

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        nlohmann::json j = nlohmann::json::parse(f);
        auto& m = j["model"];
        if (m["type"] != "BPE") throw std::runtime_error("only BPE tokenizers are supported");
        ignore_merges_ = m.value("ignore_merges", false);
        for (auto& [k, v] : m["vocab"].items()) vocab_[k] = v.get<int>();
        int rank = 0;
        for (auto& mg : m["merges"]) {
            std::string a, b;
            if (mg.is_array()) { a = mg[0]; b = mg[1]; }
            else { std::string s = mg; auto p = s.find(' '); a = s.substr(0, p); b = s.substr(p + 1); }
            auto it = vocab_.find(a + b);
            if (it != vocab_.end()) merges_[a + '\x01' + b] = {rank, it->second};
            rank++;
        }
        for (auto& t : j["added_tokens"]) {
            added_.push_back({t["content"].get<std::string>(), t["id"].get<int>(), t.value("special", false)});
            vocab_[added_.back().text] = added_.back().id;
        }
        id2tok_.resize(0);
        for (auto& [k, v] : vocab_) { if ((size_t)v >= id2tok_.size()) id2tok_.resize(v + 1); id2tok_[v] = k; }
        special_.assign(id2tok_.size(), false);
        for (auto& a : added_) special_[a.id] = a.special;
        // pre-tokenizer regex (the first Split in the Sequence)
        std::string pat = find_split_regex(j["pre_tokenizer"]);
        int err; PCRE2_SIZE off;
        re_ = pcre2_compile((PCRE2_SPTR)pat.c_str(), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &err, &off, nullptr);
        if (!re_) throw std::runtime_error("bad pre-tokenizer regex");
        pcre2_jit_compile(re_, PCRE2_JIT_COMPLETE);
        md_ = pcre2_match_data_create_from_pattern(re_, nullptr);
        // GPT-2 bytes_to_unicode
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; b++) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
        std::vector<int> cs = bs; int n = 0;
        for (int b = 0; b < 256; b++)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); i++) { byte2u_[bs[i]] = utf8(cs[i]); u2byte_[cs[i]] = (uint8_t)bs[i]; }
    }
    ~Tokenizer() { if (md_) pcre2_match_data_free(md_); if (re_) pcre2_code_free(re_); }

    int id(const std::string& tok) const { auto it = vocab_.find(tok); return it == vocab_.end() ? -1 : it->second; }
    size_t size() const { return id2tok_.size(); }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        size_t i = 0, seg = 0;
        while (i < text.size()) {            // split on added tokens (leftmost, longest)
            const Added* best = nullptr;
            for (auto& a : added_)
                if (a.text.size() && text.compare(i, a.text.size(), a.text) == 0 && (!best || a.text.size() > best->text.size())) best = &a;
            if (best) {
                encode_plain(text.substr(seg, i - seg), out);
                out.push_back(best->id);
                i += best->text.size(); seg = i;
            } else i++;
        }
        encode_plain(text.substr(seg), out);
        return out;
    }

    // Bytes of one token (byte-level decoded). Special tokens give "" unless with_special.
    std::string token_bytes(int t, bool with_special = false) const {
        if (t < 0 || (size_t)t >= id2tok_.size()) return "";
        if (special_[t]) return with_special ? id2tok_[t] : "";
        for (auto& a : added_) if (a.id == t) return a.text;
        std::string s; const std::string& u = id2tok_[t];
        for (size_t k = 0; k < u.size();) {
            int cp, len = decode_utf8(u, k, cp);
            auto it = u2byte_.find(cp);
            if (it != u2byte_.end()) s += (char)it->second;
            k += len;
        }
        return s;
    }

    std::string decode(const std::vector<int>& ids, bool with_special = false) const {
        std::string s;
        for (int t : ids) s += token_bytes(t, with_special);
        return s;
    }

private:
    struct Added { std::string text; int id; bool special; };
    struct Merge { int rank, id; };
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, Merge> merges_;
    std::vector<std::string> id2tok_;
    std::vector<bool> special_;
    std::vector<Added> added_;
    std::string byte2u_[256];
    std::unordered_map<int, uint8_t> u2byte_;
    bool ignore_merges_ = false;
    pcre2_code* re_ = nullptr;
    pcre2_match_data* md_ = nullptr;

    static std::string find_split_regex(const nlohmann::json& p) {
        if (p.is_null()) return "";
        if (p["type"] == "Split") return p["pattern"]["Regex"];
        if (p["type"] == "Sequence")
            for (auto& q : p["pretokenizers"]) { auto r = find_split_regex(q); if (!r.empty()) return r; }
        return "";
    }
    static std::string utf8(int cp) {
        std::string s;
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
        else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
        return s;
    }
    static int decode_utf8(const std::string& s, size_t i, int& cp) {
        unsigned char c = s[i];
        if (c < 0x80) { cp = c; return 1; }
        if ((c >> 5) == 6 && i + 1 < s.size()) { cp = ((c & 31) << 6) | (s[i + 1] & 63); return 2; }
        if ((c >> 4) == 14 && i + 2 < s.size()) { cp = ((c & 15) << 12) | ((s[i + 1] & 63) << 6) | (s[i + 2] & 63); return 3; }
        if (i + 3 < s.size()) { cp = ((c & 7) << 18) | ((s[i + 1] & 63) << 12) | ((s[i + 2] & 63) << 6) | (s[i + 3] & 63); return 4; }
        cp = c; return 1;
    }

    void encode_plain(const std::string& text, std::vector<int>& out) const {
        if (text.empty()) return;
        // Split with behavior Isolated: matches and the gaps between them are both pieces
        size_t start = 0, last = 0;
        while (start <= text.size()) {
            int rc = pcre2_match(re_, (PCRE2_SPTR)text.data(), text.size(), start, 0, md_, nullptr);
            if (rc < 0) break;
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md_);
            if (ov[0] > last) bpe(text.substr(last, ov[0] - last), out);
            if (ov[1] > ov[0]) bpe(text.substr(ov[0], ov[1] - ov[0]), out);
            last = ov[1];
            start = ov[1] > ov[0] ? ov[1] : ov[1] + 1;
        }
        if (last < text.size()) bpe(text.substr(last), out);
    }

    void bpe(const std::string& piece, std::vector<int>& out) const {
        std::string word;                          // byte-level: every byte -> one unicode char
        std::vector<std::string> sym;
        for (unsigned char c : piece) { word += byte2u_[c]; sym.push_back(byte2u_[c]); }
        if (ignore_merges_) { auto it = vocab_.find(word); if (it != vocab_.end()) { out.push_back(it->second); return; } }
        while (sym.size() > 1) {
            int best = -1, brank = INT32_MAX;
            for (size_t k = 0; k + 1 < sym.size(); k++) {
                auto it = merges_.find(sym[k] + '\x01' + sym[k + 1]);
                if (it != merges_.end() && it->second.rank < brank) { brank = it->second.rank; best = (int)k; }
            }
            if (best < 0) break;
            sym[best] += sym[best + 1];
            sym.erase(sym.begin() + best + 1);
        }
        for (auto& s : sym) { auto it = vocab_.find(s); if (it != vocab_.end()) out.push_back(it->second); }
    }
};
