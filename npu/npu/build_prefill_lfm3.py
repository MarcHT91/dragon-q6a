"""Build + compile the chunked-prefill graphs into an existing LFM3 build dir and register them in manifest.json.
usage: build_prefill_lfm3.py MODEL CALIB GPTQ_DIR BUILD_DIR [C]"""
import sys, json, time, gc
from pathlib import Path
import onnx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from build_lfm3 import build_prefill_group
from npu_common import npu_session, model_qparams
M, CAL, GQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
C = int(sys.argv[5]) if len(sys.argv) > 5 else 16
man = json.load(open(OUT / "manifest.json")); cal = json.load(open(CAL)); ref = LFM3Ref(M)
pre = []
for gi, gr in enumerate(man["graphs"]):
    name = f"prefill_g{gi}"; p = OUT / f"{name}.onnx"; t = time.time()
    m = build_prefill_group(ref.W, ref.c, cal, gr["layers"], man["T"], man["TS"], C, gptq_dir=GQ)
    onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
    qp = model_qparams(m); del m; gc.collect()
    ctx = OUT / f"{name}_ctx.onnx"
    for f in OUT.glob(ctx.stem + "*"): f.unlink()
    s, tc = npu_session(p, ctx); del s; gc.collect()
    pre.append({"name": name, "layers": gr["layers"], "onnx": p.name, "ctx": ctx.name, "qparams": qp})
    print(f"{name}: build {time.time()-t-tc:.0f}s compile {tc:.0f}s", flush=True)
man["prefill"] = {"C": C, "graphs": pre}
json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
print("done")
