"""LFM3-VL NPU build vs fp32 (make_eval_vl.py): teacher-forced top-1 / KL over the question + answer positions of
held-out images, with the image features injected as extra embedding rows; then greedy answers on the NPU.
usage: validate_vl.py BUILD_DIR EVAL_DIR [--gen N] [--feat-suffix SUF]
  --feat-suffix .gpu  uses feat_k.gpu.npy (features from ncnn on the Adreno) instead of the fp32 ones."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_native import LFM3Native
from vl_embed import IMG
from tokenizers import Tokenizer

B, E = Path(sys.argv[1]), Path(sys.argv[2])
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
NGEN, SUF = int(arg("--gen", 0)), arg("--feat-suffix", "")
tok = Tokenizer.from_file(str(B / "tokenizer.json")) if (B / "tokenizer.json").exists() else None
m = LFM3Native(B)
print(f"loaded in {m.load_s:.1f}s", flush=True)
R = np.load(E / "ref_logits.npz")
lp = lambda z: z - z.max(-1, keepdims=True) - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True))
ag = n = 0; kls = []
for k, line in enumerate(open(E / "eval.jsonl")):
    d = json.loads(line)
    feat = np.load(E / d["img"].replace(".npy", f"{SUF}.npy"))
    m.reset(); m.set_extra(feat)
    ids, s = m.image_ids(d["ids"], IMG), d["start"]
    t0 = time.time()
    nl = [m.logits_f32(m.feed(ids[:s + 1]))]
    t1 = time.time()
    nl += [m.logits_f32(m.step(t)) for t in ids[s + 1:]]
    nl = np.stack(nl); rl = R[f"arr_{k}"].astype(np.float32)
    pr, pn = lp(rl), lp(nl)
    kl = (np.exp(pr) * (pr - pn)).sum(-1); kls.append(kl)
    a = (rl.argmax(-1) == nl.argmax(-1)).sum(); ag += a; n += len(rl)
    print(f"img {k} ({d['size']}px, {len(feat)} img tok, prompt {s + 1} tok, prefill {t1 - t0:.2f}s): "
          f"top1 {a / len(rl):.3f} KL {kl.mean():.3f}", flush=True)
    if NGEN and tok:
        q_end = ids.index(124901) + 2 if 124901 in ids else s     # after <think></think>
        out, st = m.generate(ids[:q_end], NGEN)
        print("   NPU:", repr(tok.decode(out)[:300]), f"[{st['decode_tok_s']:.1f} tok/s]")
        print("   fp32:", repr(tok.decode(d["ids"][q_end:])[:300]))
    m.set_extra(None)
print(f"teacher-forced top1 {ag / n:.3f} | KL {np.concatenate(kls).mean():.3f} over {n} positions")
