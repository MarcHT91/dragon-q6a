"""Memory-light fp32 reference of LFM3 (torch, CPU or CUDA), lazy per-layer weights.

Layer = sandwich-normed operator (gated short conv | gated GQA attention) + SwiGLU MLP:
  x = x + post_op_norm(op(op_norm(x)));  x = x + post_ffn_norm(mlp(ffn_norm(x)))
State for incremental decoding: attention -> (K [KV,P,D], V [KV,P,D]); conv -> last (L-1) columns of B*x [hid, L-1].
"""
import json, math
from pathlib import Path
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


class Weights:
    def __init__(self, d):
        self.dir = Path(d)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.cfg = self.cfg.get("text_config", self.cfg)       # a vision-language checkpoint: its language model
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.where = json.loads(idx.read_text())["weight_map"]
        else:
            f = safe_open(str(self.dir / "model.safetensors"), framework="pt")
            self.where = {k: "model.safetensors" for k in f.keys()}
        self.files = {}
        # VL checkpoints (LFM3-VL, d1) keep the language model under model.language_model.
        self.pre = "model.language_model." if "model.language_model.embed_tokens.weight" in self.where else "model."

    def _name(self, name):
        return self.pre + name[len("model."):] if name.startswith("model.") else name

    def get(self, name):
        name = self._name(name)
        f = self.where[name]
        if f not in self.files:
            self.files[f] = safe_open(str(self.dir / f), framework="pt")
        return self.files[f].get_tensor(name).float()

    def has(self, name):
        return self._name(name) in self.where

    def layer(self, i):
        p = self._name(f"model.layers.{i}.")
        return {k[len(p):]: self.get("model." + k[len(self.pre):]) for k in self.where if k.startswith(p)}


class LFM3Ref:
    def __init__(self, d, n_layers=None, device="cpu"):
        self.W = Weights(d)
        self._st = None
        self.full_allow = None    # optional [S, P] bool mask for the full-attention layers (default causal)
        # Per-token rescaled projections (build_lfm3 PT_NORM): {(layer, "op" | "ffn"): d}. With it set, layer_forward
        # also records the ranges of out/r ("o_n", "ffn_n") and sqrt(N*eps)/r ("cn_op", "cn_ffn"), r = ||[u, d]||,
        # u the projection's input; and "un_op"/"un_ffn" = per-token ||u|| samples in self.pt_norms (to choose d).
        self.pt, self.pt_norms = None, None
        c = self.c = self.W.cfg
        self.dev = torch.device(device)
        self.L = n_layers or c["num_hidden_layers"]
        self.types = c["layer_types"]
        self.H, self.KV, self.D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.win, self.K = c["norm_eps"], c["sliding_window"], c["conv_L_cache"]
        self.embed = self.W.get("model.embed_tokens.weight").to(self.dev)
        self.fnorm = self.W.get("model.embedding_norm.weight").to(self.dev)
        self.inv = {}
        for t, rp in c["rope_parameters"].items():
            ang = int(rp.get("partial_rotary_factor", 1.0) * self.D) // 2
            inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, 2 * ang, 2, dtype=torch.float64) / self.D))
            self.inv[t] = torch.cat([inv, torch.zeros(self.D // 2 - ang, dtype=torch.float64)])

    def cos_sin(self, t, pos):
        f = torch.outer(torch.as_tensor(pos, dtype=torch.float64), self.inv[t])
        e = torch.cat([f, f], -1)
        return e.cos().float().to(self.dev), e.sin().float().to(self.dev)

    def rec(self, i, nm, t):
        if self._st is not None:
            v = float(t.abs().max())
            k = (i, nm)
            self._st[k] = max(self._st.get(k, 0.0), v)

    def attn(self, i, w, h, pos, past, cap):
        S = h.shape[0]; H, KV, D = self.H, self.KV, self.D
        a = "self_attn."
        qr, kr = h @ w[a + "q_proj.weight"].T, h @ w[a + "k_proj.weight"].T
        q = rms(qr.view(S, H, D), w[a + "q_norm.weight"], self.eps)
        k = rms(kr.view(S, KV, D), w[a + "k_norm.weight"], self.eps)
        v = (h @ w[a + "v_proj.weight"].T).view(S, KV, D)
        self.rec(i, "q_raw", qr); self.rec(i, "k_raw", kr); self.rec(i, "v", v)
        cos, sin = self.cos_sin(self.types[i], pos)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        self.rec(i, "q", q); self.rec(i, "k_all", k)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if past is not None:
            k, v = torch.cat([past[0], k], 1), torch.cat([past[1], v], 1)
        P = k.shape[1]; rep = H // KV
        s = (q @ k.repeat_interleave(rep, 0).transpose(1, 2)) / math.sqrt(D)
        qp = torch.as_tensor(pos, device=self.dev)[:, None]; kp = torch.arange(P, device=self.dev)[None]
        bad = kp > qp
        if self.types[i] == "full_attention" and self.full_allow is not None:
            bad = ~self.full_allow.to(self.dev)        # [S, P] which keys each query sees (d1's `set` attention)
        if self.types[i] == "sliding_attention":
            bad = bad | (kp <= qp - self.win)
        self.rec(i, "scores", s.masked_fill(bad, 0.0))
        self.rec(i, "k_all", k); self.rec(i, "v_all", v)
        s = torch.softmax(s.masked_fill(bad, float("-inf")), -1)
        o = (s @ v.repeat_interleave(rep, 0)).transpose(0, 1)               # [S,H,D]
        gl = h @ w[a + "gate_proj.weight"].T
        gate = torch.sigmoid(gl)[..., None]                                  # [S,H,1]
        self.rec(i, "ctx", o); self.rec(i, "glogit", gl)
        o = (o * gate).reshape(S, H * D)
        self.rec(i, "ctxg", o)
        if cap is not None:
            cap["qkvg_in"], cap["o_in"] = h, o
        return o @ w[a + "o_proj.weight"].T, (k, v)

    def conv(self, i, w, h, past, cap):
        S, hid = h.shape
        c = "conv."
        BCx = h @ w[c + "in_proj.weight"].T                                  # [S, 3*hid]
        B, C, x = BCx[:, :hid], BCx[:, hid:2 * hid], BCx[:, 2 * hid:]
        bx = (B * x).T                                                       # [hid, S]
        prev = past if past is not None else torch.zeros(hid, self.K - 1, device=self.dev)
        full = torch.cat([prev, bx], 1)                                      # [hid, K-1+S]
        kw = w[c + "conv.weight"][:, 0, :]                                   # [hid, K]
        win = full.unfold(1, self.K, 1)                                      # [hid, S, K]
        y = (win * kw[:, None, :]).sum(-1).T                                 # [S, hid]
        for nm, t in [("B", B), ("C", C), ("xx", x), ("bx", full), ("convo", y)]:
            self.rec(i, nm, t)
        y = C * y
        self.rec(i, "y", y)
        if cap is not None:
            cap["inproj_in"], cap["out_in"] = h, y
        return y @ w[c + "out_proj.weight"].T, full[:, -(self.K - 1):]

    def cache_weights(self):
        """keep every layer's weights resident on self.dev (needed for fast token-by-token generation)."""
        self._wc = {i: {k: v.to(self.dev) for k, v in self.W.layer(i).items()} for i in range(self.L)}

    def layer_forward(self, i, x, pos, past, w=None, cap=None, stats=None):
        wc = getattr(self, "_wc", None)
        w = w or (wc[i] if wc else {k: v.to(self.dev) for k, v in self.W.layer(i).items()})
        h = rms(x, w["operator_norm.weight"], self.eps)
        if self.types[i] == "conv":
            o, st = self.conv(i, w, h, past, cap)
        else:
            o, st = self.attn(i, w, h, pos, past, cap)
        if self.pt is not None or self.pt_norms is not None:
            u = cap.get("out_in" if self.types[i] == "conv" else "o_in") if cap is not None else None
            if u is None:
                c2 = {}
                if self.types[i] == "conv":
                    self.conv(i, w, h, past, c2)
                else:
                    self.attn(i, w, h, pos, past, c2)
                u = c2["out_in" if self.types[i] == "conv" else "o_in"]
            self._pt_rec(i, "op", u, o)
        on = rms(o, w["post_operator_norm.weight"], self.eps)
        x1 = x + on
        h2 = rms(x1, w["ffn_norm.weight"], self.eps)
        g, u = h2 @ w["feed_forward.w1.weight"].T, h2 @ w["feed_forward.w3.weight"].T
        sl = torch.nn.functional.silu(g)
        m = sl * u
        f = m @ w["feed_forward.w2.weight"].T
        if self.pt is not None or self.pt_norms is not None:
            self._pt_rec(i, "ffn", m, f)
        fn = rms(f, w["post_ffn_norm.weight"], self.eps)
        x2 = x1 + fn
        for nm, t in [("x_in", x), ("ln1", h), ("o", o), ("on", on), ("x_mid", x1), ("ln2", h2), ("gate", g),
                      ("up", u), ("silu", sl), ("act", m), ("ffn", f), ("fn", fn), ("x_out", x2)]:
            self.rec(i, nm, t)
        if cap is not None:
            cap["gu_in"], cap["down_in"] = h2, m
        if stats is not None:
            for nm, t in [("x_in", x), ("op", o), ("x_mid", x1), ("ln2", h2), ("gate", g), ("up", u), ("act", m), ("ffn", f), ("x_out", x2)]:
                stats.setdefault((i, nm), []).append(t.abs().amax(-1).cpu())
        return x2, st

    def _pt_rec(self, i, kind, u, out):
        nu = u.norm(dim=-1)
        if self.pt_norms is not None:
            self.pt_norms.setdefault((i, kind), []).append(nu.float().cpu())
        if self.pt is not None and (i, kind) in self.pt:
            d = self.pt[(i, kind)]
            r = torch.sqrt(nu ** 2 + d * d)[:, None]
            self.rec(i, "o_n" if kind == "op" else "ffn_n", out / r)
            self.rec(i, f"cn_{kind}", math.sqrt(out.shape[-1] * self.eps) / r)

    def forward(self, ids, start=0, past=None, stats=None):
        ids = torch.as_tensor(ids, device=self.dev)
        pos = list(range(start, start + len(ids)))
        x = self.embed[ids]
        new = []
        for i in range(self.L):
            x, st = self.layer_forward(i, x, pos, past[i] if past else None, stats=stats)
            new.append(st)
        hf = rms(x, self.fnorm, self.eps)
        lg = hf @ self.embed.T
        self.rec("head", "fnorm", hf); self.rec("head", "logits", lg)
        return lg, new

    def generate(self, ids, n, eos=(7,)):
        logits, past = self.forward(ids)
        out, pos = [], len(ids)
        for _ in range(n):
            t = int(logits[-1].argmax())
            if t in eos:
                break
            out.append(t)
            logits, past = self.forward([t], start=pos, past=past)
            pos += 1
        return out
