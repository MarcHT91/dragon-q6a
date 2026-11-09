"""LFM3 decode on the QCS6490 NPU: host loop (conv states, full KV cache, sliding-window ring)."""
import json, time, sys
from pathlib import Path
import numpy as np

from npu_common import npu_session, q16, dq16, rope_inputs, split16
from build_graphs import MASK


class LFM3NPU:
    def __init__(self, build_dir, no_prefill=False):
        self.dir = Path(build_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        c = self.c = self.man["cfg"]
        self.T, self.TS = self.man["T"], self.man["TS"]
        self.KV, self.D, self.K, self.hid = c["num_key_value_heads"], c["head_dim"], c["conv_L_cache"], c["hidden_size"]
        self.types = c["layer_types"]
        t = time.time()
        self.graphs = []
        for g in self.man["graphs"]:
            s, _ = npu_session(self.dir / g["onnx"], self.dir / g["ctx"])
            self.graphs.append((g, s, [o.name for o in s.get_outputs()], {i.name for i in s.get_inputs()}))
        self.pre, self.C = [], 0
        if "prefill" in self.man and not no_prefill:
            self.C = self.man["prefill"]["C"]
            for g in self.man["prefill"]["graphs"]:
                s, _ = npu_session(self.dir / g["onnx"], self.dir / g["ctx"])
                self.pre.append((g, s, [o.name for o in s.get_outputs()], {i.name for i in s.get_inputs()}))
        self.head, _ = npu_session(self.dir / self.man["head"]["onnx"], self.dir / self.man["head"]["ctx"])
        self.head_qp = self.man["head"]["qparams"]
        self.load_s = time.time() - t
        self.embed = np.load(self.dir / "embed_f16.npy", mmap_mode="r")
        self.rope = {}
        for typ, rp in c["rope_parameters"].items():
            self.rope[typ] = (rp["rope_theta"], int(rp.get("partial_rotary_factor", 1.0) * self.D) // 2)
        self.reset()

    def reset(self):
        KV, D = self.KV, self.D
        self.st = {}
        for g, *_ in self.graphs:
            for i in g["layers"]:
                if self.types[i] == "conv":
                    self.st[i] = np.full((self.K - 1, self.hid), 32768, np.uint16)
                else:
                    n = self.T if self.types[i] == "full_attention" else self.TS
                    self.st[i] = [np.full((KV, D, n), 128, np.uint8), np.full((KV, D, n), 128, np.uint8),
                                  np.full((KV, n, D), 128, np.uint8), np.full((KV, n, D), 128, np.uint8)]
        self.pos = 0

    def _rope(self, typ, p):
        theta, ang = self.rope[typ]
        cos, sin = rope_inputs(p, self.D, theta)
        h = self.D // 2
        cos[0, ang:h] = 1; cos[0, h + ang:] = 1; sin[0, ang:h] = 0; sin[0, h + ang:] = 0
        return cos, sin

    def step(self, token):
        p = self.pos
        if p >= self.T:
            raise RuntimeError("context full")
        mf = np.full((1, 1, self.T + 1), -MASK, np.float32); mf[..., :p] = 0; mf[..., self.T] = 0
        ms = np.full((1, 1, self.TS + 1), -MASK, np.float32); ms[..., :min(p, self.TS)] = 0; ms[..., self.TS] = 0
        cf, sf = self._rope("full_attention", p); cs, ss = self._rope("sliding_attention", p)
        x = None
        for g, s, outs, ins in self.graphs:
            qp = g["qparams"]
            if x is None:
                x = q16(self.embed[token].astype(np.float32)[None], qp["x"][0])
            feeds = {"x": x}
            for nm, arr in [("cos_f", cf), ("sin_f", sf), ("mask_f", mf), ("cos_s", cs), ("sin_s", ss), ("mask_s", ms)]:
                if nm in ins:
                    feeds[nm] = q16(arr, qp[nm][0])
            for i in g["layers"]:
                if self.types[i] == "conv":
                    feeds[f"state{i}"] = self.st[i]
                else:
                    feeds[f"past_kh{i}"], feeds[f"past_kl{i}"], feeds[f"past_vh{i}"], feeds[f"past_vl{i}"] = self.st[i]
            r = dict(zip(outs, s.run(None, feeds)))
            for i in g["layers"]:
                if self.types[i] == "conv":
                    self.st[i] = r[f"state_new{i}"]
                else:
                    slot = p if self.types[i] == "full_attention" else p % self.TS
                    c = self.st[i]
                    c[0][:, :, slot], c[1][:, :, slot] = split16(r[f"k_new{i}"][:, 0, :])
                    c[2][:, slot, :], c[3][:, slot, :] = split16(r[f"v_new{i}"][:, 0, :])
            x = r["x_out"]
        self.pos += 1
        return self.head.run(None, {"x": x})[0][0]

    def prefill_chunk(self, tokens):
        """Feed exactly C tokens at positions pos..pos+C-1 through the prefill graphs (no logits)."""
        C, p = self.C, self.pos
        assert len(tokens) == C and p + C <= self.T
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
        x = None
        for g, s, outs, ins in self.pre:
            qp = g["qparams"]
            if x is None:
                x = q16(np.asarray(self.embed[tokens], np.float32), qp["x"][0])
            feeds = {"x": x}
            for nm, arr in [("cos_f", cf), ("sin_f", sf), ("mask_f", mf), ("cos_s", cs), ("sin_s", ss), ("mask_s", ms)]:
                if nm in ins:
                    feeds[nm] = q16(arr, qp[nm][0])
            for i in g["layers"]:
                if self.types[i] == "conv":
                    feeds[f"state{i}"] = self.st[i]
                else:
                    feeds[f"past_kh{i}"], feeds[f"past_kl{i}"], feeds[f"past_vh{i}"], feeds[f"past_vl{i}"] = self.st[i]
            r = dict(zip(outs, s.run(None, feeds)))
            for i in g["layers"]:
                if self.types[i] == "conv":
                    self.st[i] = r[f"state_new{i}"]
                else:
                    c = self.st[i]
                    kh, kl = split16(r[f"k_new{i}"]); vh, vl = split16(r[f"v_new{i}"])   # [KV,C,D]
                    for t in range(C):
                        slot = p + t if self.types[i] == "full_attention" else (p + t) % self.TS
                        c[0][:, :, slot], c[1][:, :, slot] = kh[:, t], kl[:, t]
                        c[2][:, slot, :], c[3][:, slot, :] = vh[:, t], vl[:, t]
            x = r["x_out"]
        self.pos += C

    def feed(self, ids):
        """Feed a prompt: full C-chunks through prefill, the rest (incl. the last token) through decode.
        Returns the uint16 logits after the last token."""
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
