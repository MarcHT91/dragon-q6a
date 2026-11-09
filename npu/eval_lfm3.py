"""LFM3 NPU build vs fp32 reference (teacher-forced), then English chat with tok/s.
usage: eval_lfm3.py MODEL BUILD [--noref] [max_new]"""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_npu import LFM3NPU
from tokenizers import Tokenizer

M, B = sys.argv[1], sys.argv[2]
N = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 200
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
npu = LFM3NPU(B)
print(f"NPU graphs loaded in {npu.load_s:.1f}s", flush=True)
chat = lambda u: tok.encode(f"<|startoftext|><|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>", add_special_tokens=False).ids

if "--noref" not in sys.argv:
    from lfm3_ref import LFM3Ref
    ref = LFM3Ref(M)
    from eval_texts import TEXTS
    agree, kls, nr, nn, cnt = 0, [], 0.0, 0.0, 0
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        rl = ref.forward(ids)[0].numpy()
        npu.reset()
        if "--prefill" in sys.argv and npu.C:
            n_ch = (len(ids) - 1) // npu.C
            for c in range(n_ch):
                npu.prefill_chunk(ids[c * npu.C:(c + 1) * npu.C])
            rl = rl[n_ch * npu.C:]
            nl = np.stack([npu.logits_f32(npu.step(t)) for t in ids[n_ch * npu.C:]])
            ids = ids[n_ch * npu.C:]
        else:
            nl = np.stack([npu.logits_f32(npu.step(t)) for t in ids])
        lp = lambda z: z - z.max(-1, keepdims=True) - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True))
        pr, pn = lp(rl), lp(nl)
        kls.append((np.exp(pr) * (pr - pn)).sum(-1)); agree += (rl.argmax(-1) == nl.argmax(-1)).sum(); cnt += len(ids)
        tgt = np.array(ids[1:]); nr += -pr[np.arange(len(tgt)), tgt].sum(); nn += -pn[np.arange(len(tgt)), tgt].sum()
    kl = np.concatenate(kls)
    print(f"teacher-forced over {cnt} positions ({len(TEXTS)} texts): top1 agree {agree/cnt:.3f} | mean KL {kl.mean():.3f} "
          f"(max {kl.max():.2f}) | ppl ref {np.exp(nr/(cnt-len(TEXTS))):.2f} npu {np.exp(nn/(cnt-len(TEXTS))):.2f}", flush=True)

for q in ["Explain briefly what an NPU is.", "What is 17 * 23?", "Write a haiku about a small computer board."]:
    out, st = npu.generate(chat(q), N)
    print(f"\nQ: {q}\nA: {tok.decode(out, skip_special_tokens=False)}\n   [{st['prompt_tokens']} prompt tok, prefill {st['prefill_s']:.1f}s, "
          f"{st['decode_tokens']} tok @ {st['decode_tok_s']:.1f} tok/s]", flush=True)
