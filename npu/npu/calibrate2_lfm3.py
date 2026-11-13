"""Layer-major activation-range calibration of the (optionally GPTQ-quantized) LFM3 model on chat data.
usage: calibrate2_lfm3.py MODEL CHAT_JSONL[,MORE.jsonl] OUT [GPTQ_DIR] [max_seqs] [max_len]
PT_NORM=1: also the ranges of the per-token rescaled projections (nd_*, o_n, ffn_n, cn_*; build_lfm3 PT_NORM).
max_seqs is per file; image sequences (gen_calib_vl.py) take their features at the <|image|> positions."""
import os, sys, json, time, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_ref import LFM3Ref, rms
from build_lfm3 import dequant_layer

M, CHAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GQ = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 24
ML = int(sys.argv[6]) if len(sys.argv) > 6 else 640
from vl_embed import load_jsonl, embed
seqs = []
for f in CHAT.split(","):
    s = load_jsonl(f, ML); random.Random(0).shuffle(s); seqs += s[:NS]
print(f"{len(seqs)} seqs, {sum(len(s[0]) for s in seqs)} tok", flush=True)
ref = LFM3Ref(M, device=os.environ.get("REF_DEVICE", "cpu")); ref._st = {}
PT = os.environ.get("PT_NORM") == "1"; PT_D = {}
X = [embed(ref, s, ft) for s, ft in seqs]
t0 = time.time()
for i in range(ref.L):
    w = ref.W.layer(i)
    if GQ:
        w = dequant_layer(w, np.load(f"{GQ}/layer{i}.npz"))
    w = {k: v.to(ref.dev) for k, v in w.items()}
    if PT:     # per-token rescaled projections (build_lfm3 PT_NORM): d = median ||u|| of this layer's inputs, first
        ref.pt_norms = {}
        for x in X:
            ref.layer_forward(i, x, list(range(len(x))), None, w=w)
        for (li, kind), v in ref.pt_norms.items():
            umax = ref._st[(li, "act" if kind == "ffn" else "y" if ref.types[li] == "conv" else "ctxg")]
            PT_D[(li, kind)] = float(min(torch.cat(v).median(), 0.9 * umax))
        ref.pt_norms = None; ref.pt = PT_D
    X = [ref.layer_forward(i, x, list(range(len(x))), None, w=w)[0] for x in X]
    print(f"layer {i:2d} {time.time()-t0:5.0f}s", flush=True)
for x in X:
    hf = rms(x, ref.fnorm, ref.eps); ref.rec("head", "fnorm", hf); ref.rec("head", "logits", hf @ ref.embed.T)
out = {"layers": [dict() for _ in range(ref.L)], "head": {}}
for (i, nm), v in ref._st.items():
    (out["head"] if i == "head" else out["layers"][i])[nm] = v
for (i, kind), d in PT_D.items():
    out["layers"][i][f"nd_{kind}"] = d
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
