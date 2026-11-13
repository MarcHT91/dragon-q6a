"""Build + compile d1-2B for the QCS6490 NPU (run on the PC, offline QNN compile).

usage: python build_d1.py MODEL_DIR RANGES_JSON OUT_DIR [T=2048] [C=64]

d1 only reads (one forward pass, no decoding), so there are only prefill graphs, all weights int8 (RTN per output
channel: d1_quant.py measured it ~lossless, int4 GPTQ moved answers up to 0.36). `set` attention runs stage by stage:
stage k = full-attention layer f_k (its query sees the whole cache, keys of every token already written) + the
causal layers after it + the next full-attention layer's k/v only (kv_only_last). Stage 0 = the layers before the
first full-attention layer + its k/v. The host runs each stage over every chunk before the next stage.
Writes stage{k}_ctx_qnn.bin, embed_f16.npy, fnorm.npy, manifest.json (NPU_HEAD=1: also head_ctx_qnn.bin) (+ tokenizer.json, calibration.json,
config.json copied for the runtime).
"""
import gc, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, onnx

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "npu" / "npu", HERE.parent / "npu"):
    sys.path.insert(0, str(_p))
M, CAL, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
T = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
C = int(sys.argv[5]) if len(sys.argv) > 5 else 64
OUT.mkdir(parents=True, exist_ok=True)

from lfm3_ref import LFM3Ref  # noqa: E402
ref = LFM3Ref(M); cfg = ref.c; L = cfg["num_hidden_layers"]; types = cfg["layer_types"]
os.environ["INT8_LAYERS"] = ",".join(map(str, range(L)))       # every layer RTN int8 (read at build_lfm3 import)
os.environ.setdefault("PT_NORM", "1")      # per-token rescaled projections (needs ranges from d1_calib.py, which has them)
from build_lfm3 import build_prefill_group, build_head  # noqa: E402
from npu_common import npu_session, model_qparams  # noqa: E402

cal = json.load(open(CAL))
TS = cfg["sliding_window"] - 1
full = [i for i, t in enumerate(types) if t == "full_attention"]
stages = [(list(range(0, full[0] + 1)), True)]
for a, b in zip(full, full[1:]):
    stages.append((list(range(a, b + 1)), True))
stages.append((list(range(full[-1], L)), False))
print("stages:", [(s[0], s[-1], "kv" if kv else "") for s, kv in stages], flush=True)

np.save(OUT / "embed_f16.npy", ref.embed.numpy().astype(np.float16))
np.save(OUT / "fnorm.npy", ref.fnorm.numpy().astype(np.float32))      # final norm: the runtime's CPU readout head
out = []
only = {int(x) for x in os.environ.get("ONLY_STAGES", "").split(",") if x}
for k, (layers, kv) in enumerate(stages):
    name = f"stage{k}"; p = OUT / f"{name}.onnx"; ctx = OUT / f"{name}_ctx.onnx"
    if only and k not in only:
        out.append({"name": name, "layers": layers, "kv_only_last": kv, "onnx": p.name, "ctx": ctx.name,
                    "qparams": json.load(open(OUT / "manifest.json"))["stages"][k]["qparams"]}); continue
    t = time.time()
    m = build_prefill_group(ref.W, cfg, cal, layers, T, TS, C, gptq_dir=None, kv_only_last=kv)
    onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
    qp = model_qparams(m); del m; gc.collect()
    for f in OUT.glob(ctx.stem + "*"):
        f.unlink()
    s, tc = npu_session(p, ctx); del s; gc.collect()
    out.append({"name": name, "layers": layers, "kv_only_last": kv, "onnx": p.name, "ctx": ctx.name, "qparams": qp})
    print(f"{name} layers {layers[0]}-{layers[-1]}{' (kv only)' if kv else ''}: build {time.time() - t - tc:.0f}s compile {tc:.0f}s", flush=True)
head = None
if os.environ.get("NPU_HEAD") == "1":    # the runtime reads the few readout tokens on the CPU; an NPU head is 263 MB
    p = OUT / "head.onnx"
    m = build_head(ref.W, cfg, cal, bits=8)
    onnx.save(m, p, save_as_external_data=True, location=p.name + ".data")
    qp = model_qparams(m); del m
    for f in OUT.glob("head_ctx*"):
        f.unlink()
    s, tc = npu_session(p, OUT / "head_ctx.onnx"); del s
    head = {"onnx": "head.onnx", "ctx": "head_ctx.onnx", "qparams": qp}
    print(f"head compile {tc:.0f}s", flush=True)
full_cfg = json.load(open(Path(M) / "config.json"))
json.dump({"T": T, "TS": TS, "C": C, "stages": out, "head": head,
           "cfg": cfg, "d1": {k: v for k, v in full_cfg.items() if k.startswith("d1_")}},
          open(OUT / "manifest.json", "w"), indent=1)
for f in ("tokenizer.json", "calibration.json", "config.json"):
    shutil.copy(Path(M) / f, OUT / f)
print("done")
