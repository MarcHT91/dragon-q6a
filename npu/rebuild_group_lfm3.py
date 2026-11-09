"""Rebuild one decode + prefill group of an existing LFM3 build (e.g. with INT8_LAYERS=0,1,...) in place.
usage: [INT8_LAYERS=..] rebuild_group_lfm3.py MODEL CALIB GPTQ_DIR BUILD_DIR GROUP_INDEX
Files of that group are unlinked first, so BUILD_DIR may be a hardlinked copy (cp -al) of another build."""
import sys, json, time, gc
from pathlib import Path
import onnx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from build_lfm3 import build_group, build_prefill_group, INT8_LAYERS, INT8_MATS
from npu_common import npu_session, model_qparams
M, CAL, GQ, OUT, GI = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
man = json.load(open(OUT / "manifest.json")); cal = json.load(open(CAL)); ref = LFM3Ref(M)
gr = man["graphs"][GI]; layers = gr["layers"]
print("group", GI, "layers", layers, "int8 layers", sorted(INT8_LAYERS & set(layers)), flush=True)
for kind, entry, fn in [("decode", gr, lambda: build_group(ref.W, ref.c, cal, layers, man["T"], man["TS"], gptq_dir=GQ)),
                        ("prefill", man["prefill"]["graphs"][GI], lambda: build_prefill_group(ref.W, ref.c, cal, layers, man["T"], man["TS"], man["prefill"]["C"], gptq_dir=GQ))]:
    name = f"{kind}_g{GI}"; p = OUT / f"{name}.onnx"; t = time.time()
    for f in OUT.glob(name + "*"): f.unlink()
    m = fn()
    onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
    entry["qparams"] = model_qparams(m); del m; gc.collect()
    s, tc = npu_session(p, OUT / f"{name}_ctx.onnx"); del s; gc.collect()
    print(f"{name}: build {time.time()-t-tc:.0f}s compile {tc:.0f}s", flush=True)
man.setdefault("int8_layers", []); man["int8_layers"] = sorted(set(man["int8_layers"]) | (INT8_LAYERS & set(layers)))
man["int8_mats"] = sorted({tuple(x) for x in man.get("int8_mats", []) if x[0] not in layers} | {m for m in INT8_MATS if m[0] in layers})
(OUT / "manifest.json").unlink(); json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
print("done")
