"""Validate the C++ d1 (d1 --serve, one process) on the eval set of d1_calib.py against the fp32 answers.
usage: python validate_cpp.py D1_BINARY BUILD_DIR EVAL_DIR ENC_DIR   (passes: fp32 features, then pictures through the GPU)"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
BIN, B, E, ENC = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
ev = json.load(open(E / "eval.json"))
reqs = []
for mode in ("feat", "image"):
    for r in ev:
        if mode == "image" and "image" not in r:
            continue
        q = {"state": r["state"], "questions": r["questions"]}
        if "feat" in r:
            q[mode] = str(E / (r["feat"] if mode == "feat" else r["image"]))
        reqs.append((mode, r, q))
p = subprocess.Popen([BIN, "--build", B, "--enc", ENC, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
res = {}
for mode, r, q in reqs:
    p.stdin.write(json.dumps(q, ensure_ascii=False) + "\n"); p.stdin.flush()
    a = json.loads(p.stdout.readline())
    if "error" in a:
        print("ERROR", a["error"]); continue
    res.setdefault(mode, []).append((r, a))
p.stdin.close(); p.wait()
for mode, items in res.items():
    agree = n = 0; dps = []; t = []; tn = []; tg = []
    for r, a in items:
        t.append(a["time"]["total_s"]); tn.append(a["time"]["npu_s"]); tg.append(a["time"]["gpu_s"])
        for name, v in a["answers"].items():
            pb = [[v["noul"], 1 - v["noul"]]] if v["type"] == "noul" else [[x, 1 - x] for x in v["probabilities"].values()] if "choices" in v \
                else [list(v["probabilities"].values()) if isinstance(v["probabilities"], dict) else list(v["probabilities"])]
            for pa, pq in zip(r["answers"][name], pb):
                pa, pq = np.array(pa), np.array(pq); agree += int(pa.argmax() == pq.argmax()); n += 1; dps.append(float(np.abs(pa - pq).max()))
    print(f"{mode:5}: {len(items)} requests, agree {agree / n:.4f} mean |dp| {np.mean(dps):.4f} p95 {np.percentile(dps, 95):.4f} max {np.max(dps):.4f} n {n} | "
          f"time mean {np.mean(t):.2f}s (NPU {np.mean(tn):.2f}s, GPU {np.mean(tg):.2f}s)", flush=True)
    if mode == "feat":
        pics = [a["time"]["total_s"] for r, a in items if "feat" in r]; txt = [a["time"]["total_s"] for r, a in items if "feat" not in r]
        print(f"       picture requests {np.mean(pics):.2f}s, text/command {np.mean(txt):.2f}s")
