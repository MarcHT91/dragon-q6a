"""d1 on the NPU vs fp32 on the eval set of d1_calib.py: agreement and probability changes per question.
usage: python d1_validate.py BUILD_DIR EVAL_DIR [--feat-suffix ,.gpu]
--feat-suffix: comma list of feature-file suffixes, all run in this one process (feat_k{suffix}.npy; "" = the fp32
features at the picture's own size, ".gpu" = the GPU encoder's). One NPU process: each process start/exit can
leak DSP memory on this kernel."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_npu import D1NPU

B, E = sys.argv[1], Path(sys.argv[2])
SUFS = sys.argv[sys.argv.index("--feat-suffix") + 1].split(",") if "--feat-suffix" in sys.argv else [""]
m = D1NPU(B); print(f"loaded in {m.load_s:.1f}s", flush=True)
ev = json.load(open(E / "eval.json"))
for SUF in SUFS:
    print(f"=== features {SUF or 'fp32'}", flush=True)
    agree = n = 0; dps, kls, times, by = [], [], [], {}
    for k, r in enumerate(ev):
        feat = np.load(E / r["feat"].replace(".npy", f"{SUF}.npy")) if "feat" in r else None
        if SUF and feat is None:
            continue                                     # text requests: same as the fp32 pass
        t = time.time(); a = m.ask(r["state"], r["questions"], feat=feat); dt = time.time() - t; times.append(dt)
        kind = "picture" if feat is not None else "text" if r["state"] and len(r["state"]) > 200 else "command"
        worst = 0
        for name, q in r["questions"].items():
            v = a[name]
            if v["type"] == "noul":
                pb = [[v["noul"], 1 - v["noul"]]]
            elif "choices" in v:
                pb = [[p, 1 - p] for p in v["probabilities"].values()]
            else:
                p = v["probabilities"]; pb = [list(p.values()) if isinstance(p, dict) else list(p)]
            for pa, pq in zip(r["answers"][name], pb):
                pa, pq = np.clip(np.array(pa), 1e-9, 1), np.clip(np.array(pq), 1e-9, 1)
                ok = int(pa.argmax() == pq.argmax()); agree += ok; n += 1
                dp = float(np.abs(pa - pq).max()); dps.append(dp); kls.append(float((pa * np.log(pa / pq)).sum()))
                by.setdefault(kind, []).append((ok, dp)); worst = max(worst, dp)
        print(f"req {k:2d} {kind:7} {m.stats['tokens']:4d} tok {dt:5.2f}s  worst |dp| {worst:.3f}", flush=True)
    print(f"agree {agree / n:.4f} | mean |dp| {np.mean(dps):.4f} p95 {np.percentile(dps, 95):.4f} max {np.max(dps):.4f} | KL {np.mean(kls):.4f} | n {n}")
    for kd, v in by.items():
        print(f"  {kd:7} agree {np.mean([o for o, _ in v]):.4f} mean |dp| {np.mean([d for _, d in v]):.4f}")
    print(f"time per request: mean {np.mean(times):.2f}s")
