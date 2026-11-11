"""Interactive chat with LFM3-1.7B on the NPU.
usage: chat.py [--think] [--system "..."] [--max-new 512]
Commands: /reset (new conversation), /quit. Greedy decoding with repetition penalty 1.1."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import ChatFormat, BUILD_DIR
from lfm3_npu import LFM3NPU

ap = argparse.ArgumentParser()
ap.add_argument("--think", action="store_true", help="enable the model's <think> reasoning")
ap.add_argument("--system", default=None)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--build", default=str(BUILD_DIR))
a = ap.parse_args()

cf = ChatFormat()
m = LFM3NPU(a.build)
eos = cf.stop_ids(m.man["eos"])
ban = [] if a.think else [i for i in (cf.tok.token_to_id("<think>"), cf.tok.token_to_id("</think>")) if i is not None]
print(f"LFM3-1.7B on the NPU (context {m.T}, loaded in {m.load_s:.1f}s). /reset, /quit", flush=True)
msgs = [{"role": "system", "content": a.system}] if a.system else []
hist = []                                     # token ids already in the model state


def feed(ids):
    """reuse the model state when the new prompt extends what was already fed; otherwise start over."""
    global hist
    if hist and ids[:len(hist)] == hist and len(ids) > len(hist):
        lq = m.feed(ids[len(hist):])
    else:
        m.reset(); lq = m.feed(ids)
    hist = list(ids)
    return lq


while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    if q == "/quit":
        break
    if q == "/reset":
        msgs = msgs[:1] if a.system else []; hist = []; m.reset(); print("(new conversation)"); continue
    msgs.append({"role": "user", "content": q})
    ids = cf.encode(msgs, think=a.think)
    if len(ids) + 16 >= m.T:
        print(f"(conversation is {len(ids)} tokens, context is {m.T}: /reset)"); msgs.pop(); continue
    t0 = time.time(); lg = m.logits_f32(feed(ids)); t1 = time.time()
    out, seen, shown = [], set(), ""
    for _ in range(min(a.max_new, m.T - len(ids) - 1)):
        lg = lg.astype(np.float32)
        if seen:
            ix = np.fromiter(seen, int); v = lg[ix]; lg[ix] = np.where(v > 0, v / 1.1, v * 1.1)
        if ban:
            lg[ban] = -np.inf
        t = int(np.argmax(lg))
        if t in eos:
            break
        out.append(t); seen.add(t); hist.append(t)
        text = cf.tok.decode(out, skip_special_tokens=False)
        if not text.endswith("�"):
            print(text[len(shown):], end="", flush=True); shown = text
        lg = m.logits_f32(m.step(t))
    dt = time.time() - t1
    reply = cf.tok.decode(out, skip_special_tokens=False)
    msgs.append({"role": "assistant", "content": reply.split("</think>")[-1].strip()})
    print(f"\n  [{len(ids)} prompt tok in {t1 - t0:.1f}s | {len(out)} tok at {len(out) / max(dt, 1e-9):.1f} tok/s]", flush=True)
