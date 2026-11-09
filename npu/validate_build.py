"""One-process validation of an LFM3 NPU build: first-token p(<tool_call>) on the bench tool prompts (high for the
first 7, low for the last 3), decode speed, and teacher-forced top-1 / KL against fp32 logits (make_eval_ref.py).
usage: validate_build.py [BUILD_DIR] [EVAL_DIR]"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import ChatFormat, BUILD_DIR, ROOT
from bench import TASKS, TOOLS, messages
from lfm3_npu import LFM3NPU

B = Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD_DIR
E = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "build" / "eval"
cf = ChatFormat(); TC = cf.tok.token_to_id("<tool_call>")
m = LFM3NPU(B)
ps = []
for c, q, _ in TASKS:
    if c != "tool":
        continue
    msgs, _ = messages(c, q)
    m.reset(); lg = m.logits_f32(m.feed(cf.encode(msgs, TOOLS)))
    p = np.exp(lg - lg.max()); p /= p.sum(); ps.append(p[TC])
print("p(<tool_call>)", " ".join(f"{p:.3f}" for p in ps))
m.reset(); t0 = time.time(); t = 1
for _ in range(64):
    t = int(np.argmax(m.step(t)))
print(f"decode {64 / (time.time() - t0):.2f} tok/s")
if (E / "ref_logits.npz").exists():
    IDS = json.load(open(E / "eval_ids.json")); R = np.load(E / "ref_logits.npz")
    lp = lambda z: z - z.max(-1, keepdims=True) - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True))
    ag = n = 0; kls = []
    for k, ids in enumerate(IDS):
        rl = R[f"arr_{k}"].astype(np.float32); m.reset()
        nl = np.stack([m.logits_f32(m.step(t)) for t in ids]); pr, pn = lp(rl), lp(nl)
        kls.append((np.exp(pr) * (pr - pn)).sum(-1)); ag += (rl.argmax(-1) == nl.argmax(-1)).sum(); n += len(ids)
    print(f"teacher-forced top1 {ag / n:.3f} | KL {np.concatenate(kls).mean():.3f}")
else:
    print(f"(no {E}/ref_logits.npz: run make_eval_ref.py for the top-1/KL check)")
