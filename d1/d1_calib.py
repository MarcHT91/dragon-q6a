"""Activation ranges of d1-2B for the NPU build (the format of npu/npu/calibrate2_lfm3.py), measured on d1 requests
under `set` attention with the weights as the build quantizes them (RTN int8 per output channel); and the eval set
for the board: the requests, their pictures and fp32 features, and the fp32 answers. CUDA.

usage: python d1_calib.py MODEL_DIR COCO_PARQUET WIKI_PARQUET OUT_DIR
writes OUT_DIR/ranges.json and OUT_DIR/eval/{eval.json, feat_k.npy, img_k.jpg}
"""
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d1_ref import D1Ref  # noqa: E402
from d1_data import requests  # noqa: E402
from lfm3_ref import rms  # noqa: E402

torch.set_grad_enabled(False)
M, COCO, WIKI, OUT = sys.argv[1:5]
OUT = Path(OUT); (OUT / "eval").mkdir(parents=True, exist_ok=True)
m = D1Ref(M, "cuda"); m.lm.cache_weights()
CAL, EVAL = requests(m, COCO, WIKI, OUT / "img")


def answers_probs(a):
    out = {}
    for n, v in a.items():
        if v["type"] == "noul":
            out[n] = [[v["noul"], 1 - v["noul"]]]
        elif "choices" in v:
            out[n] = [[p, 1 - p] for p in v["probabilities"].values()]
        else:
            p = v["probabilities"]; out[n] = [list(p.values()) if isinstance(p, dict) else list(p)]
    return out


# ---- eval set with fp32 answers (fp32 weights)
ev = []
for k, (state, qs, feat, img) in enumerate(EVAL):
    a = m.ask(state, qs, feat=feat)
    rec = {"state": state, "questions": qs, "answers": answers_probs(a)}
    if feat is not None:
        np.save(OUT / "eval" / f"feat_{k}.npy", feat.float().cpu().numpy())
        shutil.copy(img, OUT / "eval" / f"img_{k}.jpg")
        rec["feat"], rec["image"] = f"feat_{k}.npy", f"img_{k}.jpg"
    ev.append(rec)
if not (OUT / "eval" / "eval.json").exists() or os.environ.get("EVAL", "1") == "1":
    json.dump(ev, open(OUT / "eval" / "eval.json", "w"), indent=1, ensure_ascii=False)
print(f"eval: {len(ev)} requests", flush=True)

# ---- weights as the build makes them: int8 per output channel, absmax scale
for i in range(m.lm.L):
    for k, W in m.lm._wc[i].items():
        if W.dim() == 2 and k.endswith("proj.weight") or k.startswith("feed_forward."):
            s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
            m.lm._wc[i][k] = torch.clamp(torch.round(W / s), -128, 127) * s

# ---- ranges over the calibration requests (every packed sequence under set attention)
RANGES = os.environ.get("RANGES_OUT", "ranges.json")


def calib_pass():
    for state, qs, feat, _ in CAL:
        media = "" if feat is None else "<|vision_start|>" + "<|image|>" * len(feat) + "<|vision_end|>"
        prefix = m.prefix(state, media); ql = list(qs.values())
        for k in range(0, len(ql), m.set_size):
            ids, s, g, _ = m.pack(prefix, ql[k:k + m.set_size])
            h = m.forward(ids, s, g, feat)
            m.lm.rec("head", "fnorm", h); m.lm.rec("head", "logits", h @ m.lm.embed.T)


t0 = time.time()
# PT_NORM (default 1): per-token rescaled out_proj/o_proj and w2 (build_lfm3 PT_NORM). First pass: the inputs' per-token norms
# -> d = PT_DFAC (1.0) x median ||u|| (within u's own range, so the appended column is representable); second pass: ranges.
PT = os.environ.get("PT_NORM", "1") == "1"      # on by default: the d1 build uses it (build_d1.py)
PT_D = {}
if PT:
    m.lm.pt_norms = {}; m.lm._st = {}
    calib_pass()
    for (i, kind), v in m.lm.pt_norms.items():
        umax = m.lm._st[(i, "act" if kind == "ffn" else "y" if m.lm.types[i] == "conv" else "ctxg")]
        PT_D[(i, kind)] = float(min(float(os.environ.get("PT_DFAC", "1.0")) * torch.cat(v).median(), 0.9 * umax))
    m.lm.pt_norms, m.lm.pt = None, PT_D
    print(f"PT_NORM d: op {np.median([v for (i, k), v in PT_D.items() if k == 'op']):.3f} "
          f"ffn {np.median([v for (i, k), v in PT_D.items() if k == 'ffn']):.3f} (medians over layers)", flush=True)
m.lm._st = {}
calib_pass()
out = {"layers": [dict() for _ in range(m.lm.L)], "head": {}}
for (i, nm), v in m.lm._st.items():
    (out["head"] if i == "head" else out["layers"][i])[nm] = v
for (i, kind), d in PT_D.items():
    out["layers"][i][f"nd_{kind}"] = d
json.dump(out, open(OUT / RANGES, "w"), indent=1)
print(f"ranges over {len(CAL)} requests in {time.time() - t0:.0f}s -> {OUT / RANGES}")
print("x_in L0 %.3f | x_out L23 %.1f | head fnorm %.1f logits %.1f" % (out["layers"][0]["x_in"], out["layers"][23]["x_out"],
                                                                      out["head"]["fnorm"], out["head"]["logits"]))
