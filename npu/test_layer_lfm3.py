"""Compile one LFM3 decode layer for the NPU, compare with fp32 reference and the CPU QDQ simulation.
usage: test_layer_lfm3.py MODEL CALIB LAYER [GPTQ_DIR]   (env BITS=8 for int8 RTN weights)"""
import sys, json, os, time
from pathlib import Path
import numpy as np, onnx, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from build_lfm3 import build_group, MASK
from npu_common import *
from tokenizers import Tokenizer

M, CAL, LAYER = sys.argv[1], sys.argv[2], int(sys.argv[3])
GQ = sys.argv[4] if len(sys.argv) > 4 else None
T, TS = 256, 255
cal = json.load(open(CAL)); ref = LFM3Ref(M); cfg = ref.c
KV, D, K = cfg["num_key_value_heads"], cfg["head_dim"], cfg["conv_L_cache"]
typ = cfg["layer_types"][LAYER]
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
ids = tok.encode("<|startoftext|><|im_start|>user\nWhat is the tallest mountain in Europe?<|im_end|>\n<|im_start|>assistant\n<think>\nThe user asks about the tallest mountain in Europe. That is Mount Elbrus", add_special_tokens=False).ids
P = len(ids) - 1
_, past = ref.forward(ids[:-1])
x = ref.embed[ids[-1:]]
for j in range(LAYER):
    x, _ = ref.layer_forward(j, x, [P], past[j])
x_in = x.numpy()
x_ref, st_ref = ref.layer_forward(LAYER, x, [P], past[LAYER]); x_ref = x_ref.numpy()

out = Path("build/test"); out.mkdir(parents=True, exist_ok=True)
m = build_group(ref.W, cfg, cal, [LAYER], T, TS, gptq_dir=GQ, bits=int(os.environ.get("BITS", 4)))
mp = out / f"l{LAYER}.onnx"; onnx.save(m, mp, save_as_external_data=True, location=mp.name + ".data")
qp = model_qparams(m)
L = LAYER
feeds = {"x": q16(x_in, qp["x"][0])}
if typ == "conv":
    feeds[f"state{L}"] = q16(past[L].T.numpy(), qp[f"state{L}"][0])
    exp = {"state_new": st_ref.T.numpy()}
else:
    n = T if typ == "full_attention" else TS
    sfx = "f" if typ == "full_attention" else "s"
    pr = cfg["rope_parameters"][typ]
    cos, sin = rope_inputs(P, D, pr["rope_theta"])
    ang = int(pr.get("partial_rotary_factor", 1.0) * D) // 2           # zero-tail partial RoPE
    for arr, fill in [(cos, 1.0), (sin, 0.0)]:
        arr[0, ang:D // 2] = fill; arr[0, D // 2 + ang:] = fill
    Kp, Vp = past[L]
    kc = np.zeros((KV, D, n), np.float32); vc = np.zeros((KV, n, D), np.float32)
    mask = np.full((1, 1, n + 1), -MASK, np.float32); mask[..., n] = 0
    for p in range(P):
        if typ == "sliding_attention" and p <= P - cfg["sliding_window"]:
            continue
        sl = p % n
        kc[:, :, sl] = Kp[:, p].numpy(); vc[:, sl] = Vp[:, p].numpy(); mask[..., sl] = 0
    feeds.update({f"cos_{sfx}": q16(cos, qp[f"cos_{sfx}"][0]), f"sin_{sfx}": q16(sin, qp[f"sin_{sfx}"][0]), f"mask_{sfx}": q16(mask, qp[f"mask_{sfx}"][0])})
    feeds.update(dict(zip([f"past_kh{L}", f"past_kl{L}"], split16(q16(kc, qp[f"k_new{L}"][0])))))
    feeds.update(dict(zip([f"past_vh{L}", f"past_vl{L}"], split16(q16(vc, qp[f"v_new{L}"][0])))))
    exp = {"k_new": st_ref[0][:, -1:].numpy(), "v_new": st_ref[1][:, -1:].numpy()}

rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
def report(tag, sess):
    r = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    msg = f"x_out {rel(dq16(r['x_out'], qp['x_out'][0]), x_ref):.4f}"
    for k, e in exp.items():
        msg += f" | {k} {rel(dq16(r[f'{k}{L}'], qp[f'{k}{L}'][0]).reshape(e.shape), e):.4f}"
    print(f"{tag:4} {typ:17} rel err  {msg}", flush=True)
report("CPU", cpu_session(mp))
ctx = out / f"l{LAYER}_ctx.onnx"
for f in out.glob(ctx.stem + "*"): f.unlink()
s, tc = npu_session(mp, ctx)
report("NPU", s)
for _ in range(3): s.run(None, feeds)
t = time.time()
for _ in range(20): s.run(None, feeds)
print(f"NPU {1e3*(time.time()-t)/20:.2f} ms/layer (compile {tc:.0f}s)")
