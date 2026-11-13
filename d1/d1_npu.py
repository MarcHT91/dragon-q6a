"""d1-2B on the QCS6490 NPU: the stage graphs of build_d1.py run by the native QNN runner (npu/native, through
npu/npu/qnnrun.py), host work in numpy. Same prompt and readout as the fp32 reference (d1_prompt.py).

    m = D1NPU(BUILD_DIR)
    m.ask(state, {name: question}, feat=image_features)      # like model.system_one; feat [N, 2048] or None

Per packed sequence (n tokens, padded to whole chunks of C):
  for each stage: for each chunk: x (embeddings, or the previous stage's output) -> the stage graph; conv states and
  sliding-window caches carried chunk to chunk (causal); the next full-attention layer's k/v written at the chunk's
  positions into the following stage's cache. A full-attention layer then runs with every token's keys present and
  the `set` mask over the whole cache (its own in-chunk part masked out).
  The final hidden rows at the answer slots go through the final norm and the LM head on the CPU, for the readout
  tokens only.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "npu" / "npu", HERE):
    sys.path.insert(0, str(_p))
import qnnrun  # noqa: E402
from npu_common import q16, dq16, rope_inputs, split16  # noqa: E402
from build_graphs import MASK  # noqa: E402
from d1_prompt import D1Prompt, IMG, set_mask  # noqa: E402


class D1NPU(D1Prompt):
    def __init__(self, build_dir):
        super().__init__(build_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        c = self.c = self.man["cfg"]
        self.T, self.TS, self.C = self.man["T"], self.man["TS"], self.man["C"]
        self.KV, self.D, self.H = c["num_key_value_heads"], c["head_dim"], c["num_attention_heads"]
        self.rep, self.win, self.types = self.H // self.KV, c["sliding_window"], c["layer_types"]
        t = time.time()
        self.stages = [(st, qnnrun.Graph(self.dir / st["ctx"].replace(".onnx", "_qnn.bin"))) for st in self.man["stages"]]
        # The head runs on the CPU, only for the readout tokens: final norm + a dot product per token. The answers
        # are softmaxes over the options' log-probabilities, where the log-sum-exp over the vocabulary cancels, so
        # raw logits give the same answers; and the 263 MB LM head stays off the DSP.
        self.fnorm = np.load(self.dir / "fnorm.npy").astype(np.float32)
        self.eps = c["norm_eps"]
        self.load_s = time.time() - t
        self.embed = np.load(self.dir / "embed_f16.npy", mmap_mode="r")
        self.pad = self.tok.token_to_id("<|pad|>") or 0
        self.rope = {}
        for typ, rp in c["rope_parameters"].items():
            self.rope[typ] = (rp["rope_theta"], int(rp.get("partial_rotary_factor", 1.0) * self.D) // 2)
        self.stats = {}

    def _rope(self, typ, pos):
        theta, ang = self.rope[typ]; h = self.D // 2
        cs = np.zeros((len(pos), self.D), np.float32); sn = np.zeros_like(cs)
        for t, p in enumerate(pos):
            a, b = rope_inputs(p, self.D, theta)
            a[0, ang:h] = 1; a[0, h + ang:] = 1; b[0, ang:h] = 0; b[0, h + ang:] = 0
            cs[t], sn[t] = a[0], b[0]
        return cs, sn

    def _reset(self):
        for _, G in self.stages:
            for nm, tz in G.inputs.items():
                if nm.startswith("state"):
                    tz.a.fill(32768)
                elif nm.startswith("past_"):
                    tz.a.fill(128)

    def hidden(self, ids, s, g, feat=None):
        """final-stage hidden states [n, hid] (before the final norm) under `set` attention"""
        n, C, T, TS = len(ids), self.C, self.T, self.TS
        N = -(-n // C) * C
        if N > T:
            raise ValueError(f"{n} tokens: the build's context is {T}")
        t0 = time.time()
        ids_p = np.array(list(ids) + [self.pad] * (N - n))
        x = np.asarray(self.embed[ids_p], np.float32)
        if feat is not None:
            mk = ids_p == IMG
            x[mk] = np.asarray(feat, np.float32)[: int(mk.sum())]
        allow = set_mask(n, s, g)                                  # [n, n]
        self._reset()
        rope = {typ: self._rope(typ, range(N)) for typ in self.rope}
        # Per request, not per stage: the masks and RoPE rows are the same for every stage (the mask inputs all have
        # scale MASK/32767, zero point 32768: allowed = 32768, masked = 1), so they are built once per chunk.
        ALLOW, BLOCK = np.uint16(32768), np.uint16(1)
        mask_f, mask_s, ropeq = {}, {}, {}
        ring = np.arange(TS)
        for p in range(0, N, C):
            m = np.full((C, T + C), BLOCK, np.uint16)
            q_real = min(C, max(0, n - p))
            m[:q_real, :n] = np.where(allow[p:p + q_real], ALLOW, BLOCK)
            m[q_real:, :n] = ALLOW                                   # padding queries: any real key (ignored)
            mask_f[p] = np.tile(m, (self.rep, 1))[None]
            # sliding window: ring slot sl holds the latest position < p with position % TS == sl
            last = np.where(ring < p, ring + ((p - 1 - ring) // TS) * TS, -10 ** 9)
            t = np.arange(C)[:, None]
            ms = np.where((last[None] >= 0) & (last[None] > p + t - self.win), ALLOW, BLOCK).astype(np.uint16)
            inch = np.where(np.arange(C)[None] <= t, ALLOW, BLOCK).astype(np.uint16)
            mask_s[p] = np.tile(np.concatenate([ms, inch], 1), (self.rep, 1))[None]

        def rq(typ, which, scale):
            key = (typ, which, scale)
            if key not in ropeq:
                ropeq[key] = q16(rope[typ][0 if which == "cos" else 1], scale)
            return ropeq[key]
        for k, (st, G) in enumerate(self.stages):
            I, O = G.inputs, G.outputs
            nxt = self.stages[k + 1][1] if k + 1 < len(self.stages) else None
            layers = st["layers"]
            kv_last = layers[-1] if st["kv_only_last"] else None
            slide = [i for i in layers if self.types[i] == "sliding_attention"]
            conv = [i for i in layers if self.types[i] == "conv"]
            xq = q16(x, I["x"].scale)
            out = np.zeros_like(x)
            for p in range(0, N, C):
                I["x"].a[...] = xq[p:p + C].reshape(I["x"].a.shape)
                for typ, sfx in (("full_attention", "f"), ("sliding_attention", "s")):
                    if f"cos_{sfx}" in I:
                        I[f"cos_{sfx}"].a[...] = rq(typ, "cos", I[f"cos_{sfx}"].scale)[p:p + C]
                        I[f"sin_{sfx}"].a[...] = rq(typ, "sin", I[f"sin_{sfx}"].scale)[p:p + C]
                if "mask_f" in I:
                    I["mask_f"].a[...] = mask_f[p]
                if "mask_s" in I:
                    I["mask_s"].a[...] = mask_s[p]
                G.run()
                out[p:p + C] = dq16(O["x_out"].a.reshape(C, -1), O["x_out"].scale)
                for i in conv:
                    np.copyto(I[f"state{i}"].a, O[f"state_new{i}"].a)
                slots = (p + np.arange(C)) % TS
                for i in slide:                                      # ring of the sliding-window cache
                    kh, kl = split16(O[f"k_new{i}"].a); vh, vl = split16(O[f"v_new{i}"].a)       # [KV,C,D]
                    I[f"past_kh{i}"].a[:, :, slots] = kh.transpose(0, 2, 1); I[f"past_kl{i}"].a[:, :, slots] = kl.transpose(0, 2, 1)
                    I[f"past_vh{i}"].a[:, slots, :] = vh; I[f"past_vl{i}"].a[:, slots, :] = vl
                if kv_last is not None:                              # full-attention keys for the next stage
                    i = kv_last; J = nxt.inputs
                    kh, kl = split16(O[f"k_new{i}"].a); vh, vl = split16(O[f"v_new{i}"].a)
                    J[f"past_kh{i}"].a[:, :, p:p + C] = kh.transpose(0, 2, 1); J[f"past_kl{i}"].a[:, :, p:p + C] = kl.transpose(0, 2, 1)
                    J[f"past_vh{i}"].a[:, p:p + C, :] = vh; J[f"past_vl{i}"].a[:, p:p + C, :] = vl
            x = out
        self.stats = {"tokens": n, "padded": N, "npu_s": time.time() - t0}
        return x[:n]

    def slot_logprobs(self, ids, s, g, feat, slots):
        """per slot, a row of logits indexed by token id, computed on demand (the readout reads a few dozen)"""
        x = self.hidden(ids, s, g, feat)[slots]
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * self.fnorm
        return [_Row(r, self.embed) for r in h]


class _Row:
    """the logits of one slot, z[token] = h . embed[token], computed and cached on first use"""
    def __init__(self, h, embed):
        self.h, self.embed, self.cache = h, embed, {}

    def __getitem__(self, i):
        i = int(i)
        if i not in self.cache:
            self.cache[i] = float(self.h @ np.asarray(self.embed[i], np.float32))
        return self.cache[i]


if __name__ == "__main__":
    from d1_prompt import QUESTIONS
    m = D1NPU(sys.argv[1])
    print(f"loaded in {m.load_s:.1f}s", flush=True)
    for _ in range(2):
        t = time.time()
        a = m.ask("I was charged twice and my parcel never arrived. Please refund one charge.", QUESTIONS)
        print(f"[{time.time() - t:.2f}s, NPU {m.stats['npu_s']:.2f}s, {m.stats['tokens']} tok]", round(a["refund"]["noul"], 2),
              a["team"]["choice"], a["topics"]["choices"], a["urgency"]["legend"][str(int(a["urgency"]["score"]))],
              "   expected: 1.0 billing ['billing', 'delivery'] medium", flush=True)
    for tk, exp in [("Where is my parcel? It was due Monday.", "shipping"), ("The app crashes when I open settings.", "technical")]:
        print(m.ask(tk, {"team": QUESTIONS["team"]})["team"]["choice"], "  expected:", exp)
