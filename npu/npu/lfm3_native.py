"""LFM3 on the QCS6490 NPU through the native QNN runner (native/libqnnrun.so, npu/qnnrun.py).

Same interface as LFM3NPU (feed / step / reset / st / logits_f32 / generate) and the same compiled contexts
(build_dir/*_ctx_qnn.bin), but:
- all graph inputs/outputs live in persistent rpcmem buffers, so the ~53 MB of KV cache and conv state is never
  copied per token (ORT copies it into QNN's buffers on every run);
- decode and prefill graphs share the same state buffers (registered in both QNN contexts);
- the whole decode step (embedding quantization, RoPE rows, masks, KV slot update, conv state, graph chaining,
  head) runs in C++ (lf_step), and so does a prefill chunk (lf_prefill). Set LFM3_PY_STEP=1 to use the
  equivalent numpy code instead.
Outputs are bit-identical to LFM3NPU (test_native.py)."""
import json, time, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qnnrun
from npu_common import q16, dq16, rope_inputs, split16
from build_graphs import MASK


def _bin(build_dir, ctx_onnx):
    return Path(build_dir) / ctx_onnx.replace(".onnx", "_qnn.bin")


class LFM3Native:
    def __init__(self, build_dir, no_prefill=False):
        self.dir = Path(build_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        c = self.c = self.man["cfg"]
        self.T, self.TS = self.man["T"], self.man["TS"]
        self.KV, self.D, self.K, self.hid = c["num_key_value_heads"], c["head_dim"], c["conv_L_cache"], c["hidden_size"]
        self.types = c["layer_types"]
        t = time.time()
        self.graphs = [(g, qnnrun.Graph(_bin(self.dir, g["ctx"]))) for g in self.man["graphs"]]
        self.pre, self.C = [], 0
        if "prefill" in self.man and not no_prefill:
            self.C = self.man["prefill"]["C"]
            for (g, G), gp in zip(self.graphs, self.man["prefill"]["graphs"]):
                assert g["layers"] == gp["layers"]
                P = qnnrun.Graph(_bin(self.dir, gp["ctx"]))
                for i in g["layers"]:          # share the persistent state buffers with the decode graph
                    names = [f"state{i}"] if self.types[i] == "conv" else [f"past_{k}{i}" for k in ("kh", "kl", "vh", "vl")]
                    for nm in names:
                        P.share_input(nm, G, nm)
                self.pre.append((gp, P))
        self.head = qnnrun.Graph(_bin(self.dir, self.man["head"]["ctx"]))
        self.head_qp = self.man["head"]["qparams"]
        self.load_s = time.time() - t
        self.embed = np.load(self.dir / "embed_f16.npy", mmap_mode="r")
        self.nembed = self.embed.shape[0]
        self._extra = None
        self.rope = {}
        for typ, rp in c["rope_parameters"].items():
            self.rope[typ] = (rp["rope_theta"], int(rp.get("partial_rotary_factor", 1.0) * self.D) // 2)
        # q16 rope tables per graph input scale, indexed by position (cached lazily)
        self._rope_cache = {}
        self.fast = os.environ.get("LFM3_PY_STEP") != "1"
        if self.fast:
            self._setup_fast()
        self.reset()

    def _setup_fast(self):
        """Hand the per-token (lf_step) and per-chunk (lf_prefill) host work to C++."""
        import ctypes as C
        lib = qnnrun.lib()
        def scale_of(name):
            sc = {G.inputs[name].scale for _, G in self.graphs + self.pre if name in G.inputs}   # decode + prefill share the tables
            assert len(sc) == 1, (name, sc)
            return sc.pop()
        tabs = []
        for typ, cn, sn in (("full_attention", "cos_f", "sin_f"), ("sliding_attention", "cos_s", "sin_s")):
            cq = np.zeros((self.T, self.D), np.uint16); sq = np.zeros((self.T, self.D), np.uint16)
            s_c, s_s = scale_of(cn), scale_of(sn)
            for p in range(self.T):
                c, s = self._rope(typ, p)
                cq[p] = q16(c, s_c)[0]; sq[p] = q16(s, s_s)[0]
            tabs += [cq, sq]
        self._tabs = tabs
        self._embed_c = np.ascontiguousarray(self.embed)
        ptr = lambda a: a.ctypes.data_as(C.c_void_p)
        H = self.head
        lib.lf_set_model(ptr(self._embed_c), self.hid, self.T, self.TS, self.KV, self.D,
                         ptr(tabs[0]), ptr(tabs[1]), ptr(tabs[2]), ptr(tabs[3]),
                         H.g, H.inputs["x"].idx, H.outputs["logits"].idx)
        def plan(g, G):
            I, O = G.inputs, G.outputs
            idx = lambda d, n: d[n].idx if n in d else -1
            mz = lambda n: int(q16(0.0, I[n].scale)) if n in I else 0
            mn = lambda n: int(q16(-MASK, I[n].scale)) if n in I else 0
            io = [idx(I, "x"), idx(I, "cos_f"), idx(I, "sin_f"), idx(I, "mask_f"), idx(I, "cos_s"), idx(I, "sin_s"), idx(I, "mask_s"),
                  idx(O, "x_out"), mz("mask_f"), mn("mask_f"), mz("mask_s"), mn("mask_s")]
            lay = []
            for i in g["layers"]:
                if self.types[i] == "conv":
                    lay += [0, I[f"state{i}"].idx, O[f"state_new{i}"].idx, -1, -1, -1, -1, -1, -1]
                else:
                    lay += [1 if self.types[i] == "full_attention" else 2, -1, -1,
                            I[f"past_kh{i}"].idx, I[f"past_kl{i}"].idx, I[f"past_vh{i}"].idx, I[f"past_vl{i}"].idx,
                            O[f"k_new{i}"].idx, O[f"v_new{i}"].idx]
            return G.g, (C.c_int * len(io))(*io), len(g["layers"]), (C.c_int * len(lay))(*lay)
        for g, G in self.graphs:
            lib.lf_add_graph(*plan(g, G))
        if self.pre:
            lib.lf_set_prefill(self.C, self.c["num_attention_heads"] // self.KV, self.c["sliding_window"])
            for g, P in self.pre:
                lib.lf_add_prefill_graph(*plan(g, P))
        self._nvocab = H.outputs["logits"].a.size

    # --- extra embedding rows (LFM3-VL image features) ---
    def set_extra(self, rows):
        """Rows [n, hidden] (e.g. projected image features) that token ids nembed..nembed+n-1 select; None clears."""
        self._extra = None if rows is None else np.ascontiguousarray(np.asarray(rows, np.float32).astype(np.float16))
        if self.fast:
            import ctypes as C
            r = self._extra
            qnnrun.lib().lf_set_extra(r.ctypes.data_as(C.c_void_p) if r is not None else None, self.nembed,
                                      0 if r is None else len(r))

    def _emb(self, tokens):
        t = np.atleast_1d(np.asarray(tokens))
        if self._extra is None or (t < self.nembed).all():
            return np.asarray(self.embed[t], np.float32)
        return np.stack([self._extra[i - self.nembed] if i >= self.nembed else self.embed[i] for i in t]).astype(np.float32)

    def image_ids(self, ids, image_token):
        """ids with the k-th image_token replaced by the id of extra row k (call set_extra first)."""
        out, k = [], 0
        for t in ids:
            if t == image_token:
                out.append(self.nembed + k); k += 1
            else:
                out.append(t)
        return out

    # --- persistent state: dict layer -> view(s) into the rpcmem buffers (same layout as LFM3NPU.st) ---
    @property
    def st(self):
        d = {}
        for g, G in self.graphs:
            for i in g["layers"]:
                if self.types[i] == "conv":
                    d[i] = G.inputs[f"state{i}"].a
                else:
                    d[i] = [G.inputs[f"past_{k}{i}"].a for k in ("kh", "kl", "vh", "vl")]
        return d

    @st.setter
    def st(self, new):
        cur = self.st
        for i, v in new.items():
            if isinstance(v, np.ndarray):
                np.copyto(cur[i], v)
            else:
                for a, b in zip(cur[i], v):
                    np.copyto(a, b)

    def reset(self):
        for i, v in self.st.items():
            if isinstance(v, np.ndarray):
                v.fill(32768)
            else:
                for a in v:
                    a.fill(128)
        self.pos = 0

    def _rope(self, typ, p):
        theta, ang = self.rope[typ]
        cos, sin = rope_inputs(p, self.D, theta)
        h = self.D // 2
        cos[0, ang:h] = 1; cos[0, h + ang:] = 1; sin[0, ang:h] = 0; sin[0, h + ang:] = 0
        return cos, sin

    def _rope_q(self, typ, p, scale):
        key = (typ, p, scale)
        r = self._rope_cache.get(key)
        if r is None:
            cos, sin = self._rope(typ, p)
            r = (q16(cos, scale), q16(sin, scale))
            if len(self._rope_cache) > 50000:
                self._rope_cache.clear()
            self._rope_cache[key] = r
        return r

    def step(self, token):
        if self.fast:
            import ctypes as C
            ptr = qnnrun.lib().lf_step(int(token), self.pos)
            if not ptr:
                raise RuntimeError("lf_step failed")
            self.pos += 1
            return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_uint16)), shape=(self._nvocab,)).copy()
        p = self.pos
        if p >= self.T:
            raise RuntimeError("context full")
        first = True
        prev = None
        for g, G in self.graphs:
            I = G.inputs
            if first:
                I["x"].a[...] = q16(self._emb(token), I["x"].scale)
                first = False
            else:
                np.copyto(I["x"].a, prev)
            if "cos_f" in I:
                cq, sq = self._rope_q("full_attention", p, I["cos_f"].scale)
                I["cos_f"].a[...] = cq; I["sin_f"].a[...] = sq
            if "cos_s" in I:
                cq, sq = self._rope_q("sliding_attention", p, I["cos_s"].scale)
                I["cos_s"].a[...] = cq; I["sin_s"].a[...] = sq
            if "mask_f" in I:
                m, s = I["mask_f"].a.reshape(-1), I["mask_f"].scale
                m[:p] = 32768; m[p:self.T] = q16(-MASK, s); m[self.T] = 32768
            if "mask_s" in I:
                m, s = I["mask_s"].a.reshape(-1), I["mask_s"].scale
                k = min(p, self.TS)
                m[:k] = 32768; m[k:self.TS] = q16(-MASK, s); m[self.TS] = 32768
            G.run()
            O = G.outputs
            for i in g["layers"]:
                if self.types[i] == "conv":
                    np.copyto(I[f"state{i}"].a, O[f"state_new{i}"].a)
                else:
                    slot = p if self.types[i] == "full_attention" else p % self.TS
                    kh, kl = split16(O[f"k_new{i}"].a[:, 0, :]); vh, vl = split16(O[f"v_new{i}"].a[:, 0, :])
                    I[f"past_kh{i}"].a[:, :, slot] = kh; I[f"past_kl{i}"].a[:, :, slot] = kl
                    I[f"past_vh{i}"].a[:, slot, :] = vh; I[f"past_vl{i}"].a[:, slot, :] = vl
            prev = O["x_out"].a
        np.copyto(self.head.inputs["x"].a, prev)
        self.head.run()
        self.pos += 1
        return self.head.outputs["logits"].a.reshape(-1).copy()

    def prefill_chunk(self, tokens):
        """Feed exactly C tokens at positions pos..pos+C-1 through the prefill graphs (no logits)."""
        C, p = self.C, self.pos
        assert len(tokens) == C and p + C <= self.T
        if self.fast:
            import ctypes
            tk = np.ascontiguousarray(tokens, np.int32)
            if qnnrun.lib().lf_prefill(tk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), p):
                raise RuntimeError("lf_prefill failed")
            self.pos += C
            return
        rep = self.c["num_attention_heads"] // self.KV
        win = self.c["sliding_window"]
        def rope_rows(typ):
            cs = np.zeros((C, self.D), np.float32); sn = np.zeros((C, self.D), np.float32)
            for t in range(C):
                a, b = self._rope(typ, p + t); cs[t], sn[t] = a[0], b[0]
            return cs, sn
        cf, sf = rope_rows("full_attention"); cs, ss = rope_rows("sliding_attention")
        mf = np.full((C, self.T + C), -MASK, np.float32); mf[:, :p] = 0
        ms = np.full((C, self.TS + C), -MASK, np.float32)
        slot_pos = np.array([max(range(s, p, self.TS), default=-10**9) if s < p else -10**9 for s in range(self.TS)]) \
            if p > 0 else np.full(self.TS, -10**9)
        for t in range(C):
            mf[t, self.T:self.T + t + 1] = 0
            ok = (slot_pos >= 0) & (slot_pos > p + t - win)
            ms[t, :self.TS][ok] = 0; ms[t, self.TS:self.TS + t + 1] = 0
        mf = np.tile(mf, (rep, 1))[None]; ms = np.tile(ms, (rep, 1))[None]
        prev = None
        for (gd, Gd), (g, P) in zip(self.graphs, self.pre):
            I = P.inputs
            if prev is None:
                I["x"].a[...] = q16(self._emb(tokens), I["x"].scale).reshape(I["x"].a.shape)
            else:
                np.copyto(I["x"].a, prev)
            for nm, arr in [("cos_f", cf), ("sin_f", sf), ("mask_f", mf), ("cos_s", cs), ("sin_s", ss), ("mask_s", ms)]:
                if nm in I:
                    I[nm].a[...] = q16(arr, I[nm].scale).reshape(I[nm].a.shape)
            P.run()
            O = P.outputs
            for i in g["layers"]:
                if self.types[i] == "conv":
                    np.copyto(I[f"state{i}"].a, O[f"state_new{i}"].a)
                else:
                    kh, kl = split16(O[f"k_new{i}"].a); vh, vl = split16(O[f"v_new{i}"].a)   # [KV,C,D]
                    if self.types[i] == "full_attention":
                        I[f"past_kh{i}"].a[:, :, p:p + C] = kh.transpose(0, 2, 1); I[f"past_kl{i}"].a[:, :, p:p + C] = kl.transpose(0, 2, 1)
                        I[f"past_vh{i}"].a[:, p:p + C, :] = vh; I[f"past_vl{i}"].a[:, p:p + C, :] = vl
                    else:
                        for t in range(C):
                            slot = (p + t) % self.TS
                            I[f"past_kh{i}"].a[:, :, slot] = kh[:, t]; I[f"past_kl{i}"].a[:, :, slot] = kl[:, t]
                            I[f"past_vh{i}"].a[:, slot, :] = vh[:, t]; I[f"past_vl{i}"].a[:, slot, :] = vl[:, t]
            prev = O["x_out"].a
        self.pos += C

    def feed(self, ids):
        i = 0
        if self.C:
            while len(ids) - 1 - i >= self.C and self.pos + self.C <= self.T:
                self.prefill_chunk(ids[i:i + self.C]); i += self.C
        lq = None
        for t in ids[i:]:
            lq = self.step(t)
        return lq

    def logits_f32(self, lq):
        s, zp = self.head_qp["logits"]
        return dq16(lq, s, zp)

    def generate(self, ids, n, eos=None, on_token=None):
        eos = set(eos or self.man["eos"])
        self.reset()
        t0 = time.time()
        lq = self.feed(ids)
        t1 = time.time()
        out = []
        for _ in range(n):
            nxt = int(np.argmax(lq))
            if nxt in eos:
                break
            out.append(nxt)
            if on_token:
                on_token(nxt)
            lq = self.step(nxt)
        t2 = time.time()
        return out, {"prompt_tokens": len(ids), "prefill_s": t1 - t0, "decode_tokens": len(out),
                     "decode_tok_s": len(out) / (t2 - t1) if out else 0.0}
