"""Build + compile all LFM3 NPU graphs. usage: build_all_lfm3.py MODEL CALIB GPTQ_DIR OUT [T] [GROUP]"""
import sys, json, time, gc
from pathlib import Path
import numpy as np, onnx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from build_lfm3 import build_group, build_head
from npu_common import npu_session, model_qparams
from tokenizers import Tokenizer

M, CAL, GQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
T = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
GROUP = int(sys.argv[6]) if len(sys.argv) > 6 else 6
TS = None
OUT.mkdir(parents=True, exist_ok=True)
cal = json.load(open(CAL)); ref = LFM3Ref(M); cfg = ref.c
TS = cfg["sliding_window"] - 1                         # ring of past slots for sliding layers (+ self)
L = cfg["num_hidden_layers"]
np.save(OUT / "embed_f16.npy", ref.embed.numpy().astype(np.float16))
graphs = []
for gi, a in enumerate(range(0, L, GROUP)):
    layers = list(range(a, min(a + GROUP, L)))
    name = f"decode_g{gi}"; p = OUT / f"{name}.onnx"
    t = time.time()
    m = build_group(ref.W, cfg, cal, layers, T, TS, gptq_dir=GQ)
    onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
    qp = model_qparams(m); del m; gc.collect()
    ctx = OUT / f"{name}_ctx.onnx"
    for f in OUT.glob(ctx.stem + "*"): f.unlink()
    s, tc = npu_session(p, ctx); del s; gc.collect()
    graphs.append({"name": name, "layers": layers, "onnx": p.name, "ctx": ctx.name, "qparams": qp})
    print(f"{name} layers {layers[0]}-{layers[-1]}: build {time.time()-t-tc:.0f}s compile {tc:.0f}s", flush=True)
p = OUT / "head.onnx"
m = build_head(ref.W, cfg, cal, bits=int(__import__("os").environ.get("HEAD_BITS", 8)))
onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
qp = model_qparams(m); del m
for f in OUT.glob("head_ctx*"): f.unlink()
s, tc = npu_session(p, OUT / "head_ctx.onnx"); del s
print(f"head compile {tc:.0f}s")
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
json.dump({"T": T, "TS": TS, "graphs": graphs, "head": {"onnx": "head.onnx", "ctx": "head_ctx.onnx", "qparams": qp},
           "cfg": cfg, "eos": [cfg["eos_token_id"], tok.token_to_id("<|im_end|>")]}, open(OUT / "manifest.json", "w"), indent=1)
print("done")
