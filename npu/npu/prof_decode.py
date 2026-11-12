"""Decode-time breakdown: per-graph run time vs host work, for either runtime.
usage: prof_decode.py [BUILD_DIR] [N_TOKENS]        (runtime: LFM3_RUNTIME=native|ort)
With the ORT runtime, QNN_PROF=basic (or detailed) QNN_PROF_DIR=dir also writes one QNN profiling CSV per graph
(npu_common.py): "Accelerator (execute) time" is the time on the HTP; "QNN (execute) time" minus that is the
host-side cost of the call (mostly copying the inputs, i.e. the KV cache); detailed adds per-op cycles.
See docs/RESULTS.md (decode profile)."""
import sys, time, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import BUILD_DIR
from runtime import load_model

bd = sys.argv[1] if len(sys.argv) > 1 else str(BUILD_DIR)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
m = load_model(bd, no_prefill=True)
print(f"{type(m).__name__}: loaded in {m.load_s:.1f}s, T={m.T} TS={m.TS}")
for i, (g, *_) in enumerate(m.graphs):
    tot = 0
    for li in g["layers"]:
        st = m.st[li]
        tot += st.nbytes if isinstance(st, np.ndarray) else sum(a.nbytes for a in st)
    print(f"graph {i} (layers {g['layers'][0]}-{g['layers'][-1]}): state {tot/1e6:.1f} MB")

acc = collections.defaultdict(float)
if hasattr(m.graphs[0][1], "run") and not getattr(m, "fast", False):   # per-graph timing (ORT or numpy step)
    def wrap(name, obj):
        orig = obj.run
        def run(*a, **k):
            t = time.perf_counter(); r = orig(*a, **k); acc[name] += time.perf_counter() - t; return r
        obj.run = run
    for i, (g, s, *_) in enumerate(m.graphs):
        wrap(f"graph {i}", s)
    wrap("head", m.head)

tok = 1
for _ in range(8):
    m.step(tok)
acc.clear()
t0 = time.perf_counter()
for _ in range(n):
    tok = int(np.argmax(m.step(tok)))
wall = time.perf_counter() - t0
print(f"{n} steps: {wall/n*1000:.1f} ms/token ({n/wall:.2f} tok/s)")
tot = 0
for k, v in acc.items():
    print(f"  {k:10s} {v/n*1000:7.2f} ms"); tot += v
if acc:
    print(f"  {'host':10s} {(wall-tot)/n*1000:7.2f} ms")
