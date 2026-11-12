"""Check the native runtime against ONNX Runtime: same prompt -> same logits and greedy tokens, plus timing.
Each runtime must run in its own process (two QNN runtimes in one process exhaust the DSP address space):
  python npu/test_native.py ort    [BUILD_DIR] [N]   # writes build/native_ref.npz
  python npu/test_native.py native [BUILD_DIR] [N]   # compares with it
Add --long to use a ~700-token prompt (45 prefill chunks; the 255-slot sliding window wraps around).
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import ChatFormat, BUILD_DIR, ROOT
from runtime import load_model

long = "--long" in sys.argv
sys.argv = [a for a in sys.argv if a != "--long"]
mode = sys.argv[1] if len(sys.argv) > 1 else "native"
bd = sys.argv[2] if len(sys.argv) > 2 else str(BUILD_DIR)
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ref = ROOT / "build" / ("native_ref_long.npz" if long else "native_ref.npz")
cf = ChatFormat()
text = ("Explain in a few sentences why the sky is blue, and mention Rayleigh scattering. "
        "Then list three colors of a sunset.")
if long:
    text = " ".join(f"Note {i}: the sky over the harbour changed colour {i % 7 + 2} times before noon." for i in range(40)) \
        + " Now, ignoring the notes above: " + text
ids = cf.encode([{"role": "user", "content": text}])

m = load_model(bd, mode)
t0 = time.time(); lq = m.feed(ids); t1 = time.time()
first = m.logits_f32(lq).copy()
out, lqs = [], []
t2 = time.time()
for _ in range(n_new):
    nxt = int(np.argmax(lq)); out.append(nxt)
    lq = m.step(nxt); lqs.append(m.logits_f32(lq))
t3 = time.time()
print(f"{type(m).__name__}: load {m.load_s:.1f}s, prefill {len(ids)} tok {t1-t0:.2f}s, decode {n_new/(t3-t2):.2f} tok/s")
if mode == "ort":
    ref.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ref, first=first, out=np.array(out), lqs=np.array(lqs))
    print("reference written to", ref)
else:
    r = np.load(ref)
    k = min(n_new, len(r["out"]))
    same = sum(int(a) == b for a, b in zip(r["out"][:k], out[:k]))
    d = max(float(np.abs(a - b).max()) for a, b in zip(r["lqs"][:k], lqs[:k]))
    print(f"greedy tokens identical: {same}/{k}; first logits max |diff| {np.abs(r['first'] - first).max():.4g}; "
          f"per-step logits max |diff| {d:.4g}")
    print(repr(cf.tok.decode(out)[:150]))
