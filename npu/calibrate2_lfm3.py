"""Layer-major activation-range calibration of the (optionally GPTQ-quantized) LFM3 model on chat data.
usage: calibrate2_lfm3.py MODEL CHAT_JSONL OUT [GPTQ_DIR] [max_seqs] [max_len]"""
import sys, json, time, random
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref, rms
from build_lfm3 import dequant_layer

M, CHAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GQ = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 24
ML = int(sys.argv[6]) if len(sys.argv) > 6 else 640
seqs = [json.loads(l)["ids"][:ML] for l in open(CHAT)]
random.Random(0).shuffle(seqs); seqs = seqs[:NS]
print(f"{len(seqs)} seqs, {sum(map(len, seqs))} tok", flush=True)
ref = LFM3Ref(M); ref._st = {}
X = [ref.embed[torch.tensor(s)] for s in seqs]
t0 = time.time()
for i in range(ref.L):
    w = ref.W.layer(i)
    if GQ:
        w = dequant_layer(w, np.load(f"{GQ}/layer{i}.npz"))
    X = [ref.layer_forward(i, x, list(range(len(x))), None, w=w)[0] for x in X]
    print(f"layer {i:2d} {time.time()-t0:5.0f}s", flush=True)
for x in X:
    hf = rms(x, ref.fnorm, ref.eps); ref.rec("head", "fnorm", hf); ref.rec("head", "logits", hf @ ref.embed.T)
out = {"layers": [dict() for _ in range(ref.L)], "head": {}}
for (i, nm), v in ref._st.items():
    (out["head"] if i == "head" else out["layers"][i])[nm] = v
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
