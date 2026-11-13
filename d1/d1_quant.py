"""Which weight quantization keeps d1-2B's answers: simulated (fake-quant weights, fp32 activations) on the fp32
reference with `set` attention, compared with fp32 on held-out requests. CUDA.

usage: python d1_quant.py MODEL_DIR COCO_PARQUET WIKI_PARQUET INT8_MATS_JSON OUT_DIR [methods]
methods (comma list, default all): rtn4, gptq4, gptq4_i8 (layers 0-5 + the INT8_MATS matrices int8, as the LFM3
NPU builds), awq_gptq4, int8.
Calibration and eval requests are d1 prompts (pictures, Wikipedia passages, home-assistant commands) packed with
their questions under `set` attention; the eval set uses other pictures, passages and commands.
Writes OUT_DIR/<method>/layer{i}.npz (gptq_lfm3.py format) for the GPTQ methods and OUT_DIR/results.json.
Metrics per question: agreement of the answer (argmax option, or each option of a multiple choice), mean |dp|
(largest probability change among its options) and KL(fp32 || quant) of the option distribution.
"""
import io, json, math, random, sys, time
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d1_ref import D1Ref  # noqa: E402

torch.set_grad_enabled(False)
M, COCO, WIKI, MATS, OUT = sys.argv[1:6]
OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
METHODS = sys.argv[6].split(",") if len(sys.argv) > 6 else ["rtn4", "gptq4", "gptq4_i8", "awq_gptq4", "int8"]
DEV = "cuda"

from d1_data import requests, CMDS  # noqa: E402
m = D1Ref(M, DEV); m.lm.cache_weights()
t0 = time.time()
CAL, EVAL = requests(m, COCO, WIKI, OUT / "img")
CAL = [r[:3] for r in CAL]; EVAL = [r[:3] for r in EVAL]
print(f"requests: {len(CAL)} calibration, {len(EVAL)} eval ({time.time() - t0:.0f}s for the pictures)", flush=True)


def sequences(reqs):
    """(embedded input [T, hid], set mask [T, T]) for every packed sequence of the requests"""
    out = []
    for state, qs, feat in reqs:
        media = "" if feat is None else "<|vision_start|>" + "<|image|>" * len(feat) + "<|vision_end|>"
        prefix = m.prefix(state, media)
        ql = list(qs.values())
        for k in range(0, len(ql), m.set_size):
            ids, s, g, _ = m.pack(prefix, ql[k:k + m.set_size])
            t = torch.as_tensor(ids, device=DEV)
            x = m.lm.embed[t].clone()
            if feat is not None:
                mk = t == 125030
                x[mk] = torch.as_tensor(feat, dtype=x.dtype, device=DEV)[: int(mk.sum())]
            n = len(ids); qi = torch.arange(n)[:, None]; ki = torch.arange(n)[None]; gg = torch.as_tensor(g)
            out.append((x, ((qi < s) | (ki < s) | (gg[:, None] == gg[None, :])).to(DEV)))
    return out


CAL_SEQ = sequences(CAL)
NTOK = sum(len(x) for x, _ in CAL_SEQ)
print(f"calibration: {len(CAL_SEQ)} sequences, {NTOK} tokens", flush=True)


def probs_of(a, q):
    """option distributions of one answer: [[p...]] (a multiple choice: one [yes, no] per option)"""
    if a["type"] == "noul":
        return [[a["noul"], 1 - a["noul"]]]
    if "choices" in a:
        return [[p, 1 - p] for p in a["probabilities"].values()]
    p = a["probabilities"]
    return [list(p.values()) if isinstance(p, dict) else list(p)]


def evaluate():
    res = []
    for state, qs, feat in EVAL:
        a = m.ask(state, qs, feat=feat)
        res.append({n: probs_of(a[n], qs[n]) for n in qs})
    return res


t0 = time.time(); REF = evaluate(); print(f"fp32 eval {time.time() - t0:.0f}s", flush=True)
# one model copy on the GPU at a time (fp32 2B = 6.4 GB): the originals go to host RAM
ORIG = {i: {k: v.cpu() for k, v in m.lm._wc[i].items()} for i in range(m.lm.L)}
del m.lm._wc; torch.cuda.empty_cache()


def compare(res):
    agree = n = 0; dps, kls = [], []
    by = {"picture": [], "text": [], "command": []}
    for (state, qs, feat), ra, rb in zip(EVAL, REF, res):
        kind = "picture" if feat is not None else "command" if state in CMDS else "text"
        for name in qs:
            for pa, pb in zip(ra[name], rb[name]):
                pa, pb = np.clip(np.array(pa), 1e-9, 1), np.clip(np.array(pb), 1e-9, 1)
                ok = int(pa.argmax() == pb.argmax()); agree += ok; n += 1
                dp = float(np.abs(pa - pb).max()); dps.append(dp); kls.append(float((pa * np.log(pa / pb)).sum()))
                by[kind].append((ok, dp))
    out = {"agree": agree / n, "mean_dp": float(np.mean(dps)), "p95_dp": float(np.percentile(dps, 95)),
           "max_dp": float(np.max(dps)), "kl": float(np.mean(kls)), "n": n}
    for k, v in by.items():
        if v:
            out[f"agree_{k}"] = float(np.mean([o for o, _ in v])); out[f"dp_{k}"] = float(np.mean([d for _, d in v]))
    return out


# ------------------------------------------------------------------ quantizers
QMAX4 = 7
GROUPS = {
    "conv": {"inproj_in": ["conv.in_proj.weight"], "out_in": ["conv.out_proj.weight"]},
    "attn": {"qkvg_in": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], "o_in": ["self_attn.o_proj.weight"]},
    "mlp": {"gu_in": ["feed_forward.w1.weight", "feed_forward.w3.weight"], "down_in": ["feed_forward.w2.weight"]},
}
INT8_MATS = {(int(i), n) for i, n in json.load(open(MATS))}


def scales_mse(W, qmax):
    amax = W.abs().amax(1).clamp_min(1e-12)
    best_s, best_e = amax / qmax, torch.full_like(amax, float("inf"))
    for r in torch.linspace(0.6, 1.0, 17):
        s = amax * r / qmax
        e = ((torch.clamp(torch.round(W / s[:, None]), -qmax - 1, qmax) * s[:, None] - W) ** 2).sum(1)
        b = e < best_e
        best_e[b], best_s[b] = e[b], s[b]
    return best_s


def rtn(W, bits):
    q = 2 ** (bits - 1) - 1; s = scales_mse(W, q)
    return torch.clamp(torch.round(W / s[:, None]), -q - 1, q) * s[:, None]


def gptq(W, H, s, block=128, damp=0.01):
    W = W.clone(); n = W.shape[1]
    dead = torch.diag(H) == 0
    H = H.clone(); H[dead, dead] = 1; W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True)
    W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.diag(H).mean() * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for j in range(i2 - i1):
            w, d = W1[:, j], Hi[j, j]
            q = torch.clamp(torch.round(w / s), -QMAX4 - 1, QMAX4)
            Q1[:, j] = q
            e = (w - q * s) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)]


def layer_pass(i, X, w, cap_keys=None):
    """forward of layer i over all calibration sequences (with their set masks); optionally capture inputs"""
    outs, caps = [], []
    for x, mask in X:
        m.lm.full_allow = mask
        cap = {} if cap_keys else None
        y, _ = m.lm.layer_forward(i, x, list(range(len(x))), None, w=w, cap=cap)
        outs.append((y, mask)); caps.append(cap)
    m.lm.full_allow = None
    return outs, caps


H_, KV_, D_ = m.lm.H, m.lm.KV, m.lm.D
HID = m.lm.c["hidden_size"]


def awq_fold(i, X, w, groups):
    """AWQ per-input-channel scales, folded exactly (as npu/experiments/awq_gptq_lfm3.py)"""
    gen = torch.Generator(device="cpu").manual_seed(i)
    _, caps = layer_pass(i, X, w, True)
    samp, amean = {}, {}
    for (x, _), cap in zip(X, caps):
        idx = torch.randperm(len(x), generator=gen)[:48].to(DEV)
        for k, a in cap.items():
            if k in groups:
                samp.setdefault(k, []).append(a[idx]); amean[k] = amean.get(k, 0) + a.abs().sum(0)
    info = []
    for k, names in groups.items():
        Xs = torch.cat(samp[k]); am = (amean[k] / NTOK).clamp_min(1e-5)
        shape = (lambda a: a.view(KV_, H_ // KV_, D_).mean(1, keepdim=True).expand(KV_, H_ // KV_, D_).reshape(-1)) if k == "o_in" else (lambda a: a)
        Ws = [w[n] for n in names]; ref_out = [Xs @ W.T for W in Ws]
        best = (sum(float(((Xs @ rtn(W, 4).T - o) ** 2).sum()) for W, o in zip(Ws, ref_out)), 0.0, None)
        for alpha in [j / 20 for j in range(1, 21)]:
            s = shape(am.pow(alpha)); s = s / (s.max() * s.min()).sqrt()
            err = sum(float((((Xs / s) @ rtn(W * s[None], 4).T - o) ** 2).sum()) for W, o in zip(Ws, ref_out))
            if err < best[0]:
                best = (err, alpha, s)
        info.append(f"{k}:{best[1]:.2f}")
        s = best[2]
        if s is None:
            continue
        for n in names + (["self_attn.gate_proj.weight"] if k == "qkvg_in" else []):
            w[n] = w[n] * s[None]
        if k in ("qkvg_in", "inproj_in"):
            w["operator_norm.weight"] = w["operator_norm.weight"] / s
        elif k == "gu_in":
            w["ffn_norm.weight"] = w["ffn_norm.weight"] / s
        elif k == "o_in":
            sv = s.view(KV_, H_ // KV_, D_)[:, 0].reshape(-1)
            w["self_attn.v_proj.weight"] = w["self_attn.v_proj.weight"] / sv[:, None]
        elif k == "out_in":
            Wc = w["conv.in_proj.weight"].clone(); Wc[HID:2 * HID] /= s[:, None]; w["conv.in_proj.weight"] = Wc
        elif k == "down_in":
            w["feed_forward.w3.weight"] = w["feed_forward.w3.weight"] / s[:, None]
    return w, " ".join(info)


def quantize(method):
    X = list(CAL_SEQ)
    wcs = {}
    save = OUT / method
    if method.startswith("gptq") or method.startswith("awq"):
        save.mkdir(exist_ok=True)
    t0 = time.time()
    for i in range(m.lm.L):
        w = {k: v.to(DEV) for k, v in ORIG[i].items()}
        groups = {**GROUPS["conv" if m.lm.types[i] == "conv" else "attn"], **GROUPS["mlp"]}
        info = ""
        if method == "awq_gptq4":
            w, info = awq_fold(i, X, w, groups)
        kind = {}
        for k, names in groups.items():
            for nm in names:
                if method == "int8" or (method == "gptq4_i8" and (i < 6 or (i, nm) in INT8_MATS)):
                    kind[nm] = "rtn8"
                elif method == "rtn4":
                    kind[nm] = "rtn4"
                else:
                    kind[nm] = "gptq4"
        wq = dict(w); out = {}
        if "gptq4" in kind.values():
            _, caps = layer_pass(i, X, w, True)
            Hs = {}
            for cap in caps:
                for k, a in cap.items():
                    if k in groups:
                        Hs[k] = Hs.get(k, 0) + a.T @ a
            del caps
        for k, names in groups.items():
            for nm in names:
                if kind[nm] == "gptq4":
                    s = scales_mse(w[nm], QMAX4)
                    Q = gptq(w[nm], Hs[k] * (2.0 / NTOK), s)
                    wq[nm] = Q * s[:, None]
                    out[nm + "_q"] = Q.T.contiguous().cpu().numpy().astype(np.int8); out[nm + "_s"] = s.cpu().numpy().astype(np.float32)
                else:
                    wq[nm] = rtn(w[nm], 8 if kind[nm] == "rtn8" else 4)
        if out:
            np.savez(save / f"layer{i}.npz", **out)
            if method == "awq_gptq4":       # the folded non-quantized tensors the NPU build would need
                np.savez(save / f"layer{i}_folded.npz", **{k: v.cpu().numpy() for k, v in w.items() if k not in kind})
        X, _ = layer_pass(i, X, wq)
        wcs[i] = wq
        print(f"  {method} layer {i:2d} {time.time() - t0:5.0f}s {info}", flush=True)
    return wcs


results = {}
rp = OUT / "results.json"
if rp.exists():
    results = json.loads(rp.read_text())
for meth in METHODS:
    t0 = time.time()
    wcs = quantize(meth)
    m.lm._wc = wcs
    results[meth] = compare(evaluate())
    del m.lm._wc, wcs; torch.cuda.empty_cache()
    print(f"== {meth} ({time.time() - t0:.0f}s): " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in results[meth].items()), flush=True)
    rp.write_text(json.dumps(results, indent=1))
print("done")
