"""QDQ ONNX decode graphs for LFM3 on HTP v68 (w4a16, uint16 I/O), built on the primitives in build_graphs.py.

Per-graph inputs (only those its layers need):
  x [1,hid] u16; cos_f/sin_f, cos_s/sin_s [1,D] u16 (RoPE of full / sliding layers, sign folded into sin);
  mask_f [1,1,T+1], mask_s [1,1,TS+1] u16 (additive 0/-MASK, last column = self)
  conv layer i:  state{i} [K-1,hid] u16 (previous B*x rows, oldest first)  -> out bx_new{i} [1,hid]
  attn layer i:  past_kh/kl{i} [KV,D,n] u8, past_vh/vl{i} [KV,n,D] u8 (n = T or TS ring) -> out k_new{i}, v_new{i} [KV,1,D]
outputs: x_out [1,hid] u16 (+ the per-layer state outputs)
"""
import sys, math
from pathlib import Path
import numpy as np
from onnx import helper, TensorProto as TP

from build_graphs import G, sym_u16, lp_rmsnorm, quant_w_int4_pc, MASK

U8 = TP.UINT8
import os
INT8_LAYERS = {int(v) for v in os.environ.get("INT8_LAYERS", "").split(",") if v}
# INT8_MATS=file.json: [[layer, "self_attn.v_proj.weight"], ...] -> those matrices RTN int8 per-channel instead of GPTQ int4
import json as _json
INT8_MATS = {(int(a), b) for a, b in _json.load(open(os.environ["INT8_MATS"]))} if os.environ.get("INT8_MATS") else set()


def dequant_layer(w, npz):
    """replace GPTQ'd weights by their dequantized values (for calibration of the quantized model)."""
    w = dict(w)
    for k in npz.files:
        if k.endswith("_q"):
            nm = k[:-2]
            w[nm] = (np.asarray(npz[k], np.float32) * npz[nm + "_s"][None, :]).T.copy()
            import torch
            w[nm] = torch.from_numpy(w[nm])
    return w


def rmsnorm_eps(g, x, gamma, in_qp, out_qp, name, eps, lead):
    """Exact RMSNorm with eps on HTP: LpNorm([x, c]) with c = sqrt(N*eps) appended, then drop c.
    ||[x,c]||^2 = ||x||^2 + N*eps  ->  x/||.|| * sqrt(N) == x / sqrt(mean(x^2) + eps)."""
    n = len(gamma)
    c = math.sqrt(n * eps)
    s, zp = in_qp
    cq = g.const(g.uid(name + "_eps"), np.full(lead + [1], min(65535, int(round(c / s)) + zp), np.uint16), TP.UINT16)
    xc = g.op("Concat", [x, g.dq(cq, s, zp)], in_qp, name + "_cat", axis=-1)
    u = g.op("LpNormalization", [xc], (1.0 / 32767.0, 32768), name + "_lp", axis=-1, p=2)
    a, b = g.uid(name + "_keep"), g.uid(name + "_drop")
    g.nodes.append(helper.make_node("Split", [u, g.const(g.uid("sp"), np.array([n, 1], np.int64), TP.INT64)], [a, b], axis=-1))
    uq = g.dq(g.q(a, 1.0 / 32767.0, 32768), 1.0 / 32767.0, 32768)
    g.dq(g.q(b, 1.0 / 32767.0, 32768), 1.0 / 32767.0, 32768)
    gd = g.weight_u16(name + "_g", np.asarray(gamma, np.float32) * math.sqrt(n))
    return g.op("Mul", [uq, gd], out_qp, name)


def build_group(ref_w, cfg, cal, layers, T, TS, gptq_dir=None, bits=4):
    bits0 = bits
    types = cfg["layer_types"]
    H, KV, D, hid = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["hidden_size"]
    K = cfg["conv_L_cache"]
    rep = H // KV
    g = G(f"lfm3_{layers[0]}_{layers[-1]}")
    xs = lambda i: sym_u16(cal["layers"][i]["x_in"] if i == layers[0] else cal["layers"][i - 1]["x_out"])
    x = g.inp("x", [1, hid], xs(layers[0]))
    rq = (1.0 / 32767.0, 32768)
    need = {types[i] for i in layers}
    rope = {}
    if "full_attention" in need:
        rope["full_attention"] = (g.inp("cos_f", [1, D], rq), g.inp("sin_f", [1, D], rq), g.inp("mask_f", [1, 1, T + 1], (MASK / 32767.0, 32768)), T)
    if "sliding_attention" in need:
        rope["sliding_attention"] = (g.inp("cos_s", [1, D], rq), g.inp("sin_s", [1, D], rq), g.inp("mask_s", [1, 1, TS + 1], (MASK / 32767.0, 32768)), TS)

    for i in layers:
        c = cal["layers"][i]
        Q = lambda nm: sym_u16(c[nm])
        w = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in ref_w.layer(i).items()}
        bits = 8 if i in INT8_LAYERS else bits0          # INT8_LAYERS: RTN int8 per-channel instead of GPTQ int4
        gq = dict(np.load(f"{gptq_dir}/layer{i}.npz")) if gptq_dir and i not in INT8_LAYERS else None

        def W(name, cols=None):
            """(q [in,out] int8, s [out]) for weight `name` (optionally a column slice of the output)."""
            if gq is not None and name + "_q" in gq:
                q, s = gq[name + "_q"], gq[name + "_s"]
            else:
                q, s = quant_w_int4_pc(w[name], bits)
            if cols is not None:
                q, s = q[:, cols], s[cols]
            return q, s

        def linear(xin, name, qp, tag, cols=None, b=None):
            b = b or (8 if (i, name) in INT8_MATS else bits)
            if b == bits:
                pre = W(name, cols)
            else:
                q, s = quant_w_int4_pc(w[name], b)
                pre = (q[:, cols], s[cols]) if cols is not None else (q, s)
            return g.op("MatMul", [xin, g.weight_int4(f"l{i}_{tag}", None, b, pre)], qp, f"l{i}_{tag}")

        eps = cfg["norm_eps"]
        norm = lambda t, wname, in_qp, qp, tag, lead=[1]: rmsnorm_eps(g, t, w[wname], in_qp, qp, f"l{i}_{tag}", eps, lead)
        h = norm(x, "operator_norm.weight", xs(i), Q("ln1"), "ln1")

        if types[i] == "conv":
            qp_bx = sym_u16(c["bx"])
            st = g.inp(f"state{i}", [K - 1, hid], qp_bx)
            Bv = linear(h, "conv.in_proj.weight", Q("B"), "B", slice(0, hid))
            Cv = linear(h, "conv.in_proj.weight", Q("C"), "C", slice(hid, 2 * hid))
            Xv = linear(h, "conv.in_proj.weight", Q("xx"), "X", slice(2 * hid, 3 * hid))
            bx = g.op("Mul", [Bv, Xv], qp_bx, f"l{i}_bx")
            # output the whole shifted state (Concat): a Mul output exported directly came back
            # with exact zeros turned into 65535 on v68
            sa, sb = g.uid(f"l{i}_st0"), g.uid(f"l{i}_st1")
            g.nodes.append(helper.make_node("Split", [st, g.const(g.uid("sp"), np.array([1, K - 2], np.int64), TP.INT64)], [sa, sb], axis=0))
            g.dq(g.q(sa, *qp_bx), *qp_bx)
            sbq = g.dq(g.q(sb, *qp_bx), *qp_bx)
            g.op("Concat", [sbq, bx], qp_bx, f"l{i}_stn", out_q=f"state_new{i}", axis=0)
            g.out(f"state_new{i}", [K - 1, hid])
            kw = w["conv.conv.weight"][:, 0, :]                                 # [hid, K]
            wp = g.weight_u16(f"l{i}_cw", kw[:, :K - 1].T)                      # [K-1, hid]
            wl = g.weight_u16(f"l{i}_cwl", kw[:, K - 1][None])                 # [1, hid]
            co = Q("convo")
            pm = g.op("Mul", [st, wp], sym_u16(c["bx"] * float(np.abs(kw).max())), f"l{i}_cpm")
            ps = g.op("ReduceSum", [pm, g.const(g.uid("ax"), np.array([0], np.int64), TP.INT64)], co, f"l{i}_cps", keepdims=1)
            pl = g.op("Mul", [bx, wl], co, f"l{i}_cpl")
            cv = g.op("Add", [ps, pl], co, f"l{i}_conv")
            y = g.op("Mul", [Cv, cv], Q("y"), f"l{i}_y")
            o = linear(y, "conv.out_proj.weight", Q("o"), "out")
        else:
            cos_dq, sin_dq, mask_dq, n = rope[types[i]]
            a = "self_attn."
            qp_k = sym_u16(c["k_all"]); qp_v = sym_u16(c["v_all"])
            s_k, s_v = qp_k[0], qp_v[0]

            def inp8(name, shape, scale):
                g.inputs.append(helper.make_tensor_value_info(name, U8, shape))
                return g.dq(name, scale, 128, dt=U8)
            pkh = inp8(f"past_kh{i}", [KV, D, n], s_k * 256); pkl = inp8(f"past_kl{i}", [KV, D, n], s_k)
            pvh = inp8(f"past_vh{i}", [KV, n, D], s_v * 256); pvl = inp8(f"past_vl{i}", [KV, n, D], s_v)

            def rope_apply(t, qp, tag):
                aa, bb = g.uid(f"l{i}_{tag}_a"), g.uid(f"l{i}_{tag}_b")
                g.nodes.append(helper.make_node("Split", [t, g.const(g.uid("sp"), np.array([D // 2, D // 2], np.int64), TP.INT64)], [aa, bb], axis=-1))
                aq, bq = g.dq(g.q(aa, *qp), *qp), g.dq(g.q(bb, *qp), *qp)
                sw = g.op("Concat", [bq, aq], qp, f"l{i}_{tag}_sw", axis=-1)
                return g.op("Add", [g.op("Mul", [t, cos_dq], qp, f"l{i}_{tag}_c"), g.op("Mul", [sw, sin_dq], qp, f"l{i}_{tag}_s")], qp, f"l{i}_{tag}_r")

            q = linear(h, a + "q_proj.weight", Q("q_raw"), "q")
            k = linear(h, a + "k_proj.weight", Q("k_raw"), "k")
            v = linear(h, a + "v_proj.weight", Q("v"), "v")
            gl = linear(h, a + "gate_proj.weight", Q("glogit"), "g", b=8)
            q = g.op("Reshape", [q, g.shape_const([1, H, D])], Q("q_raw"), f"l{i}_qr")
            k = g.op("Reshape", [k, g.shape_const([1, KV, D])], Q("k_raw"), f"l{i}_kr")
            q_max = c["q"] / math.sqrt(D); qs = sym_u16(q_max)
            q = rmsnorm_eps(g, q, w[a + "q_norm.weight"] / math.sqrt(D), Q("q_raw"), qs, f"l{i}_qn", eps, [1, H])
            k = rmsnorm_eps(g, k, w[a + "k_norm.weight"], Q("k_raw"), qp_k, f"l{i}_kn", eps, [1, KV])
            q = rope_apply(q, qs, "qrope"); k = rope_apply(k, qp_k, "krope")
            v = g.dq(g.op("Reshape", [v, g.shape_const([KV, 1, D])], qp_v, f"l{i}_vr", out_q=f"v_new{i}"), *qp_v)
            kt = g.dq(g.op("Reshape", [k, g.shape_const([KV, 1, D])], qp_k, f"l{i}_kt", out_q=f"k_new{i}"), *qp_k)
            g.out(f"k_new{i}", [KV, 1, D]); g.out(f"v_new{i}", [KV, 1, D])
            qg = g.op("Reshape", [q, g.shape_const([KV, rep, D])], qs, f"l{i}_qg")
            sc_max = c["scores"]; qp_sc = sym_u16(sc_max)
            sh = g.op("MatMul", [qg, pkh], qp_sc, f"l{i}_sch")
            sl = g.op("MatMul", [qg, pkl], sym_u16(q_max * math.sqrt(D) * 128 * s_k * 3), f"l{i}_scl")
            sp = g.op("Add", [sh, sl], qp_sc, f"l{i}_scp")
            qk = g.op("Mul", [qg, kt], sym_u16(q_max * c["k_all"]), f"l{i}_qk")
            ss = g.op("ReduceSum", [qk, g.const(g.uid("ax"), np.array([-1], np.int64), TP.INT64)], qp_sc, f"l{i}_scs", keepdims=1)
            sc = g.op("Concat", [sp, ss], qp_sc, f"l{i}_sc", axis=-1)
            lo, hi = -(sc_max * 1.25 + MASK), sc_max * 1.25
            s_m = (hi - lo) / 65535.0
            scm = g.op("Add", [sc, mask_dq], (s_m, int(round(-lo / s_m))), f"l{i}_scm")
            pq = (1.0 / 65535.0, 0)
            pr = g.op("Softmax", [scm], pq, f"l{i}_pr", axis=-1)
            pa, pb = g.uid(f"l{i}_pp"), g.uid(f"l{i}_ps")
            g.nodes.append(helper.make_node("Split", [pr, g.const(g.uid("sp"), np.array([n, 1], np.int64), TP.INT64)], [pa, pb], axis=-1))
            pp, ps = g.dq(g.q(pa, *pq), *pq), g.dq(g.q(pb, *pq), *pq)
            ch = g.op("MatMul", [pp, pvh], Q("ctx"), f"l{i}_ctxh")
            cl = g.op("MatMul", [pp, pvl], sym_u16(128 * s_v), f"l{i}_ctxl")
            cs = g.op("Mul", [ps, v], Q("ctx"), f"l{i}_ctxs")
            ctx = g.op("Add", [g.op("Add", [ch, cl], Q("ctx"), f"l{i}_ctxp"), cs], Q("ctx"), f"l{i}_ctx")   # [KV,rep,D]
            ctx = g.op("Reshape", [ctx, g.shape_const([H, D])], Q("ctx"), f"l{i}_ctxr")
            gs = g.op("Sigmoid", [gl], (1.0 / 65535.0, 0), f"l{i}_gsig")
            gs = g.op("Reshape", [gs, g.shape_const([H, 1])], (1.0 / 65535.0, 0), f"l{i}_gsr")
            cg = g.op("Mul", [ctx, gs], Q("ctxg"), f"l{i}_ctxg")
            cg = g.op("Reshape", [cg, g.shape_const([1, H * D])], Q("ctxg"), f"l{i}_ctxgr")
            o = linear(cg, a + "o_proj.weight", Q("o"), "o")

        on = norm(o, "post_operator_norm.weight", Q("o"), Q("on"), "pon")
        x1 = g.op("Add", [x, on], Q("x_mid"), f"l{i}_x1")
        h2 = norm(x1, "ffn_norm.weight", Q("x_mid"), Q("ln2"), "ln2")
        gt = linear(h2, "feed_forward.w1.weight", Q("gate"), "w1")
        up = linear(h2, "feed_forward.w3.weight", Q("up"), "w3")
        sg = g.op("Sigmoid", [gt], (1.0 / 65535.0, 0), f"l{i}_sig")
        si = g.op("Mul", [gt, sg], Q("silu"), f"l{i}_silu")
        act = g.op("Mul", [si, up], Q("act"), f"l{i}_act")
        f = linear(act, "feed_forward.w2.weight", Q("ffn"), "w2")
        fn = norm(f, "post_ffn_norm.weight", Q("ffn"), Q("fn"), "pfn")
        last = i == layers[-1]
        x = g.op("Add", [x1, fn], sym_u16(c["x_out"]), f"l{i}_x2", out_q="x_out" if last else None)
    g.out("x_out", [1, hid])
    global LAST_TAPS
    LAST_TAPS = g.taps
    return g.model()


def build_head(ref_w, cfg, cal, bits=8):
    g = G("lfm3_head")
    hid, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    x = g.inp("x", [1, hid], sym_u16(cal["layers"][L - 1]["x_out"]))
    h = rmsnorm_eps(g, x, ref_w.get("model.embedding_norm.weight").numpy(), sym_u16(cal["layers"][L - 1]["x_out"]),
                    sym_u16(cal["head"]["fnorm"]), "fnorm", cfg["norm_eps"], [1])
    E = ref_w.get("model.embed_tokens.weight").numpy()
    from build_graphs import quant_rows_lowmem
    wq = g.weight_int4("lm_head", None, bits, quant_rows_lowmem(E, bits))
    g.op("MatMul", [h, wq], sym_u16(cal["head"]["logits"], 1.5), "logits", out_q="logits")
    g.out("logits", [1, E.shape[0]])
    return g.model()


def build_prefill_group(ref_w, cfg, cal, layers, T, TS, C=16, gptq_dir=None, bits=4):
    bits0 = bits
    """Chunked prefill: C new tokens at positions p..p+C-1 (all real; remainders go through decode).
    Inputs:  x [C,hid]; cos_*/sin_* [C,D]; mask_f [1,rep*C,T+C], mask_s [1,rep*C,TS+C] (rows = (r,t), host-built,
             includes the causal in-chunk part); per layer the same cache/state inputs as decode.
    Outputs: x_out [C,hid]; conv: state_new{i} [K-1,hid]; attn: k_new{i}, v_new{i} [KV,C,D]."""
    types = cfg["layer_types"]
    H, KV, D, hid = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["hidden_size"]
    K, eps = cfg["conv_L_cache"], cfg["norm_eps"]
    rep = H // KV
    g = G(f"lfm3p_{layers[0]}_{layers[-1]}")
    xs = lambda i: sym_u16(cal["layers"][i]["x_in"] if i == layers[0] else cal["layers"][i - 1]["x_out"])
    x = g.inp("x", [C, hid], xs(layers[0]))
    rq = (1.0 / 32767.0, 32768)
    need = {types[i] for i in layers}
    rope = {}
    for typ, sfx, n in [("full_attention", "f", T), ("sliding_attention", "s", TS)]:
        if typ in need:
            cs = g.inp(f"cos_{sfx}", [C, D], rq); sn = g.inp(f"sin_{sfx}", [C, D], rq)
            cs = g.op("Reshape", [cs, g.shape_const([C, 1, D])], rq, f"cos{sfx}r")
            sn = g.op("Reshape", [sn, g.shape_const([C, 1, D])], rq, f"sin{sfx}r")
            rope[typ] = (cs, sn, g.inp(f"mask_{sfx}", [1, rep * C, n + C], (MASK / 32767.0, 32768)), n)
    ax = lambda a: g.const(g.uid("ax"), np.array(a, np.int64), TP.INT64)

    def split(t, sizes, axis, qp, tag):
        outs = [g.uid(f"{tag}{j}") for j in range(len(sizes))]
        g.nodes.append(helper.make_node("Split", [t, g.const(g.uid("sp"), np.array(sizes, np.int64), TP.INT64)], outs, axis=axis))
        return [g.dq(g.q(o, *qp), *qp) for o in outs]

    for i in layers:
        c = cal["layers"][i]
        Q = lambda nm: sym_u16(c[nm])
        w = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in ref_w.layer(i).items()}
        bits = 8 if i in INT8_LAYERS else bits0          # INT8_LAYERS: RTN int8 per-channel instead of GPTQ int4
        gq = dict(np.load(f"{gptq_dir}/layer{i}.npz")) if gptq_dir and i not in INT8_LAYERS else None

        def linear(xin, name, qp, tag, cols=None, b=None):
            b = b or (8 if (i, name) in INT8_MATS else bits)
            if b == bits and gq is not None and name + "_q" in gq:
                q, s = gq[name + "_q"], gq[name + "_s"]
            else:
                q, s = quant_w_int4_pc(w[name], b)
            if cols is not None:
                q, s = q[:, cols], s[cols]
            return g.op("MatMul", [xin, g.weight_int4(f"l{i}_{tag}", None, b, (q, s))], qp, f"l{i}_{tag}")

        norm = lambda t, wname, in_qp, qp, tag, lead=[C]: rmsnorm_eps(g, t, w[wname], in_qp, qp, f"l{i}_{tag}", eps, lead)
        h = norm(x, "operator_norm.weight", xs(i), Q("ln1"), "ln1")
        if types[i] == "conv":
            qp_bx = sym_u16(c["bx"])
            st = g.inp(f"state{i}", [K - 1, hid], qp_bx)
            Bv = linear(h, "conv.in_proj.weight", Q("B"), "B", slice(0, hid))
            Cv = linear(h, "conv.in_proj.weight", Q("C"), "C", slice(hid, 2 * hid))
            Xv = linear(h, "conv.in_proj.weight", Q("xx"), "X", slice(2 * hid, 3 * hid))
            bx = g.op("Mul", [Bv, Xv], qp_bx, f"l{i}_bx")                        # [C,hid]
            full = g.op("Concat", [st, bx], qp_bx, f"l{i}_full", axis=0)          # [K-1+C,hid]
            parts = split(full, [C, K - 1], 0, qp_bx, f"l{i}_fs")
            g.op("Reshape", [parts[1], g.shape_const([K - 1, hid])], qp_bx, f"l{i}_stn", out_q=f"state_new{i}")
            g.out(f"state_new{i}", [K - 1, hid])
            kw = w["conv.conv.weight"][:, 0, :]                                  # [hid,K]
            co = Q("convo"); acc = None
            for k in range(K):
                win = split(full, [k, C, K - 1 - k] if 0 < k < K - 1 else ([C, K - 1] if k == 0 else [K - 1, C]), 0, qp_bx, f"l{i}_w{k}_")
                win = win[1] if 0 < k < K - 1 else (win[0] if k == 0 else win[1])
                term = g.op("Mul", [win, g.weight_u16(f"l{i}_cw{k}", kw[:, k][None])], co, f"l{i}_ct{k}")
                acc = term if acc is None else g.op("Add", [acc, term], co, f"l{i}_ca{k}")
            y = g.op("Mul", [Cv, acc], Q("y"), f"l{i}_y")
            o = linear(y, "conv.out_proj.weight", Q("o"), "out")
        else:
            cos_dq, sin_dq, mask_dq, n = rope[types[i]]
            a = "self_attn."
            qp_k = sym_u16(c["k_all"]); qp_v = sym_u16(c["v_all"]); s_k, s_v = qp_k[0], qp_v[0]

            def inp8(name, shape, scale):
                g.inputs.append(helper.make_tensor_value_info(name, U8, shape))
                return g.dq(name, scale, 128, dt=U8)
            pkh = inp8(f"past_kh{i}", [KV, D, n], s_k * 256); pkl = inp8(f"past_kl{i}", [KV, D, n], s_k)
            pvh = inp8(f"past_vh{i}", [KV, n, D], s_v * 256); pvl = inp8(f"past_vl{i}", [KV, n, D], s_v)

            def rope_apply(t, qp, tag):                                            # t [C,h,D]
                aa, bb = split(t, [D // 2, D // 2], -1, qp, f"l{i}_{tag}_h")
                sw = g.op("Concat", [bb, aa], qp, f"l{i}_{tag}_sw", axis=-1)
                return g.op("Add", [g.op("Mul", [t, cos_dq], qp, f"l{i}_{tag}_c"), g.op("Mul", [sw, sin_dq], qp, f"l{i}_{tag}_s")], qp, f"l{i}_{tag}_r")

            q = linear(h, a + "q_proj.weight", Q("q_raw"), "q")
            k = linear(h, a + "k_proj.weight", Q("k_raw"), "k")
            v = linear(h, a + "v_proj.weight", Q("v"), "v")
            gl = linear(h, a + "gate_proj.weight", Q("glogit"), "g", b=8)          # [C,H]
            q = g.op("Reshape", [q, g.shape_const([C, H, D])], Q("q_raw"), f"l{i}_qr")
            k = g.op("Reshape", [k, g.shape_const([C, KV, D])], Q("k_raw"), f"l{i}_kr")
            q_max = c["q"] / math.sqrt(D); qs = sym_u16(q_max)
            q = rmsnorm_eps(g, q, w[a + "q_norm.weight"] / math.sqrt(D), Q("q_raw"), qs, f"l{i}_qn", eps, [C, H])
            k = rmsnorm_eps(g, k, w[a + "k_norm.weight"], Q("k_raw"), qp_k, f"l{i}_kn", eps, [C, KV])
            q = rope_apply(q, qs, "qrope"); k = rope_apply(k, qp_k, "krope")
            kT = g.op("Transpose", [k, ], qp_k, f"l{i}_kT", perm=[1, 0, 2])          # [KV,C,D]
            g.op("Reshape", [kT, g.shape_const([KV, C, D])], qp_k, f"l{i}_ko", out_q=f"k_new{i}")
            v = g.op("Reshape", [v, g.shape_const([C, KV, D])], qp_v, f"l{i}_vr")
            vT = g.op("Transpose", [v], qp_v, f"l{i}_vT", perm=[1, 0, 2])             # [KV,C,D]
            g.op("Reshape", [vT, g.shape_const([KV, C, D])], qp_v, f"l{i}_vo", out_q=f"v_new{i}")
            g.out(f"k_new{i}", [KV, C, D]); g.out(f"v_new{i}", [KV, C, D])
            qT = g.op("Transpose", [q], qs, f"l{i}_qT", perm=[1, 0, 2])                # [H,C,D]
            qg = g.op("Reshape", [qT, g.shape_const([KV, rep * C, D])], qs, f"l{i}_qg")  # rows (r,t)
            sc_max = c["scores"]; qp_sc = sym_u16(sc_max)
            sh = g.op("MatMul", [qg, pkh], qp_sc, f"l{i}_sch")
            sl = g.op("MatMul", [qg, pkl], sym_u16(q_max * math.sqrt(D) * 128 * s_k * 3), f"l{i}_scl")
            sp = g.op("Add", [sh, sl], qp_sc, f"l{i}_scp")                             # [KV,rC,n]
            q4 = g.op("Reshape", [qg, g.shape_const([KV, rep * C, 1, D])], qs, f"l{i}_q4")
            k4 = g.op("Reshape", [kT, g.shape_const([KV, 1, C, D])], qp_k, f"l{i}_k4")
            qk = g.op("Mul", [q4, k4], sym_u16(q_max * c["k_all"]), f"l{i}_qk")       # [KV,rC,C,D]
            ss = g.op("ReduceSum", [qk, ax([-1])], qp_sc, f"l{i}_scs", keepdims=0)      # [KV,rC,C]
            sc = g.op("Concat", [sp, ss], qp_sc, f"l{i}_sc", axis=-1)
            lo, hi = -(sc_max * 1.25 + MASK), sc_max * 1.25
            s_m = (hi - lo) / 65535.0
            scm = g.op("Add", [sc, mask_dq], (s_m, int(round(-lo / s_m))), f"l{i}_scm")
            pq = (1.0 / 65535.0, 0)
            pr = g.op("Softmax", [scm], pq, f"l{i}_pr", axis=-1)
            pp, ps = split(pr, [n, C], -1, pq, f"l{i}_pps")
            ch = g.op("MatMul", [pp, pvh], Q("ctx"), f"l{i}_ctxh")
            cl = g.op("MatMul", [pp, pvl], sym_u16(128 * s_v), f"l{i}_ctxl")
            p4 = g.op("Reshape", [ps, g.shape_const([KV, rep * C, C, 1])], pq, f"l{i}_p4")
            v4 = g.op("Reshape", [vT, g.shape_const([KV, 1, C, D])], qp_v, f"l{i}_v4")
            pv = g.op("Mul", [p4, v4], qp_v, f"l{i}_pv")                                # [KV,rC,C,D]
            cs = g.op("ReduceSum", [pv, ax([2])], Q("ctx"), f"l{i}_ctxs", keepdims=0)  # [KV,rC,D]
            ctx = g.op("Add", [g.op("Add", [ch, cl], Q("ctx"), f"l{i}_ctxp"), cs], Q("ctx"), f"l{i}_ctx")
            ctx = g.op("Reshape", [ctx, g.shape_const([H, C, D])], Q("ctx"), f"l{i}_ctxr")
            ctx = g.op("Transpose", [ctx], Q("ctx"), f"l{i}_ctxT", perm=[1, 0, 2])      # [C,H,D]
            gs = g.op("Sigmoid", [gl], pq, f"l{i}_gsig")
            gs = g.op("Reshape", [gs, g.shape_const([C, H, 1])], pq, f"l{i}_gsr")
            cg = g.op("Mul", [ctx, gs], Q("ctxg"), f"l{i}_ctxg")
            cg = g.op("Reshape", [cg, g.shape_const([C, H * D])], Q("ctxg"), f"l{i}_ctxgr")
            o = linear(cg, a + "o_proj.weight", Q("o"), "o")

        on = norm(o, "post_operator_norm.weight", Q("o"), Q("on"), "pon")
        x1 = g.op("Add", [x, on], Q("x_mid"), f"l{i}_x1")
        h2 = norm(x1, "ffn_norm.weight", Q("x_mid"), Q("ln2"), "ln2")
        gt = linear(h2, "feed_forward.w1.weight", Q("gate"), "w1")
        up = linear(h2, "feed_forward.w3.weight", Q("up"), "w3")
        sg = g.op("Sigmoid", [gt], (1.0 / 65535.0, 0), f"l{i}_sig")
        si = g.op("Mul", [gt, sg], Q("silu"), f"l{i}_silu")
        act = g.op("Mul", [si, up], Q("act"), f"l{i}_act")
        f = linear(act, "feed_forward.w2.weight", Q("ffn"), "w2")
        fn = norm(f, "post_ffn_norm.weight", Q("ffn"), Q("fn"), "pfn")
        x = g.op("Add", [x1, fn], sym_u16(c["x_out"]), f"l{i}_x2")
    g.op("Reshape", [x, g.shape_const([C, hid])], sym_u16(cal["layers"][layers[-1]]["x_out"]), "xo", out_q="x_out")
    g.out("x_out", [C, hid])
    return g.model()
