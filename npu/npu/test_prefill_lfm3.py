"""One-layer chunked-prefill graph on the NPU vs fp32 reference. usage: test_prefill_lfm3.py MODEL CALIB LAYER [GPTQ]"""
import sys, json, os, time
from pathlib import Path
import numpy as np, onnx, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from build_lfm3 import build_prefill_group, MASK
from npu_common import *
from tokenizers import Tokenizer

M, CAL, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
GQ = sys.argv[4] if len(sys.argv) > 4 else None
T, TS, C = 256, 255, 16
cal = json.load(open(CAL)); ref = LFM3Ref(M); cfg = ref.c
KV, D, H = cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_attention_heads"]; rep = H // KV
typ = cfg["layer_types"][L]
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
ids = tok.encode("<|startoftext|><|im_start|>user\nWhat is the tallest mountain in Europe, and how was its height first measured?<|im_end|>\n<|im_start|>assistant\n<think>\nThe user asks about the tallest mountain in Europe. That is Mount Elbrus in the Caucasus", add_special_tokens=False).ids
P0 = len(ids) - C
_, past = ref.forward(ids[:P0])
pos = list(range(P0, P0 + C))
x = ref.embed[ids[P0:]]
for j in range(L):
    x, _ = ref.layer_forward(j, x, pos, past[j])
x_ref, st_ref = ref.layer_forward(L, x, pos, past[L])
m = build_prefill_group(ref.W, cfg, cal, [L], T, TS, C, gptq_dir=GQ, bits=int(os.environ.get("BITS", 4)))
out = Path("build/test"); mp = out / f"p{L}.onnx"; onnx.save(m, mp, save_as_external_data=True, location=mp.name + ".data")
qp = model_qparams(m)
feeds = {"x": q16(x.numpy(), qp["x"][0])}
if typ == "conv":
    feeds[f"state{L}"] = q16(past[L].T.numpy(), qp[f"state{L}"][0])
    exp = {"state_new": st_ref.T.numpy()}
else:
    n = T if typ == "full_attention" else TS
    sfx = "f" if typ == "full_attention" else "s"
    pr = cfg["rope_parameters"][typ]; ang = int(pr.get("partial_rotary_factor", 1.0) * D) // 2
    cs = np.zeros((C, D), np.float32); sn = np.zeros((C, D), np.float32)
    for t in range(C):
        c_, s_ = rope_inputs(P0 + t, D, pr["rope_theta"])
        c_[0, ang:D // 2] = 1; c_[0, D // 2 + ang:] = 1; s_[0, ang:D // 2] = 0; s_[0, D // 2 + ang:] = 0
        cs[t], sn[t] = c_[0], s_[0]
    Kp, Vp = past[L]
    kc = np.zeros((KV, D, n), np.float32); vc = np.zeros((KV, n, D), np.float32); spos = np.full(n, -10**9)
    for p in range(P0):
        sl = p % n; kc[:, :, sl] = Kp[:, p].numpy(); vc[:, sl] = Vp[:, p].numpy(); spos[sl] = p
    mask = np.full((C, n + C), -MASK, np.float32)
    for t in range(C):
        qpz = P0 + t
        ok = (spos < P0) & (spos >= 0)
        if typ == "sliding_attention":
            ok &= spos > qpz - cfg["sliding_window"]
        mask[t, :n][ok] = 0; mask[t, n:n + t + 1] = 0
    mask = np.tile(mask, (rep, 1))[None]
    feeds.update({f"cos_{sfx}": q16(cs, qp[f"cos_{sfx}"][0]), f"sin_{sfx}": q16(sn, qp[f"sin_{sfx}"][0]), f"mask_{sfx}": q16(mask, qp[f"mask_{sfx}"][0])})
    feeds.update(dict(zip([f"past_kh{L}", f"past_kl{L}"], split16(q16(kc, qp[f"k_new{L}"][0])))))
    feeds.update(dict(zip([f"past_vh{L}", f"past_vl{L}"], split16(q16(vc, qp[f"v_new{L}"][0])))))
    exp = {"k_new": st_ref[0][:, -C:].numpy(), "v_new": st_ref[1][:, -C:].numpy()}
rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
def report(tag, sess):
    r = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    msg = f"x_out {rel(dq16(r['x_out'], qp['x_out'][0]), x_ref.numpy()):.4f}"
    for k, e in exp.items():
        msg += f" | {k} {rel(dq16(r[f'{k}{L}'], qp[f'{k}{L}'][0]).reshape(e.shape), e):.4f}"
    print(f"{tag:4} prefill {typ:17} rel err {msg}", flush=True)
report("CPU", cpu_session(mp))
ctx = out / f"p{L}_ctx.onnx"
for f in out.glob(ctx.stem + "*"): f.unlink()
s, tc = npu_session(mp, ctx); report("NPU", s)
for _ in range(2): s.run(None, feeds)
t = time.time()
for _ in range(10): s.run(None, feeds)
print(f"NPU {1e3*(time.time()-t)/10:.2f} ms per {C}-token chunk per layer (compile {tc:.0f}s)")
