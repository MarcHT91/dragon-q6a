"""Per-matrix int8 sensitivity on top of a base mix (int8 layers 0-5 + GPTQ int4 elsewhere), then greedy budgets.
Each candidate matrix is switched alone to RTN int8 and the drop in mean KL vs fp32 is measured (weights only).
Writes OUT/sens.json (sorted by KL gain per parameter) and OUT/int8_mats_{N}M.json budget lists. CUDA, ~50 min on an RTX 3060.
usage: sens.py MODEL_DIR GPTQ_DIR WIKITEXT_PARQUET CHAT_JSONL OUT_DIR"""
import json, sys, time, numpy as np, torch
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_ref import LFM3Ref
from eval_texts import TEXTS
torch.set_grad_enabled(False)
MD, GQD, WIKI, CHAT, OUT = sys.argv[1:6]; Path(OUT).mkdir(parents=True, exist_ok=True)
tok = Tokenizer.from_file(f"{MD}/tokenizer.json"); BOS = tok.token_to_id("<|startoftext|>")
IDS = [tok.encode(t, add_special_tokens=False).ids for t in TEXTS]
wiki = tok.encode("\n".join(t for t in pq.read_table(WIKI)["text"].to_pylist() if t.strip()), add_special_tokens=False).ids
rng = np.random.default_rng(123)
IDS += [[BOS] + wiki[o:o + 255] for o in rng.integers(0, len(wiki) - 300, 4)]
chat = [json.loads(l)["ids"] for l in open(CHAT)]
IDS += [c[:300] for c in chat[-4:]]
print(len(IDS), "texts", sum(map(len, IDS)), "tok", flush=True)
r = LFM3Ref(MD, device="cuda"); r.cache_weights()
ORIG = {i: {k: v.clone() for k, v in r._wc[i].items()} for i in range(r.L)}
def gq(i):
    z = np.load(f"{GQD}/layer{i}.npz")
    return {k[:-2]: torch.from_numpy((z[k].astype(np.float32) * z[k[:-2] + "_s"][None]).T.copy()).cuda() for k in z.files if k.endswith("_q")}
def rtn(W, bits):
    qmax = 2 ** (bits - 1) - 1; Wt = W.T; amax = Wt.abs().amax(0) + 1e-12
    best_e = torch.full_like(amax, float("inf")); best_s = amax / qmax
    for rr in np.linspace(0.6, 1.0, 17):
        s = amax * float(rr) / qmax; e = ((torch.clamp(torch.round(Wt / s), -qmax - 1, qmax) * s - Wt) ** 2).sum(0)
        b = e < best_e; best_e[b], best_s[b] = e[b], s[b]
    return (torch.clamp(torch.round(Wt / best_s), -qmax - 1, qmax) * best_s).T.contiguous()
REFLP = [torch.log_softmax(r.forward(ids)[0].float(), -1).half().cpu() for ids in IDS]
REFAM = [l.argmax(-1) for l in REFLP]
G4 = {i: gq(i) for i in range(6, r.L)}
I8 = {i: {k: rtn(ORIG[i][k], 8) for k in G4.get(i, gq(i))} for i in range(r.L)}
for i in range(r.L):
    r._wc[i] = dict(ORIG[i]); r._wc[i].update(I8[i] if i < 6 else G4[i])
def score(sub=None):
    kl = []; ag = n = 0
    for k, ids in enumerate(IDS if sub is None else [IDS[j] for j in sub]):
        k = k if sub is None else sub[k]
        lp = torch.log_softmax(r.forward(ids)[0].float(), -1); pr = REFLP[k].cuda().float()
        kl.append((pr.exp() * (pr - lp)).sum(-1)); ag += int((lp.argmax(-1).cpu() == REFAM[k]).sum()); n += len(ids)
    return float(torch.cat(kl).mean()), ag / n
base, base_t1 = score(); b4 = score(list(range(4)))
print(f"base: KL {base:.4f} top1 {base_t1:.3f} | 4 texts KL {b4[0]:.3f} top1 {b4[1]:.3f}", flush=True)
cands = []
t0 = time.time()
for i in range(6, r.L):
    for nm in G4[i]:
        r._wc[i][nm] = I8[i][nm]; kl, _ = score(); r._wc[i][nm] = G4[i][nm]
        p = ORIG[i][nm].numel(); cands.append((i, nm, base - kl, p))
        print(f"L{i:2d} {nm.split('.')[-2]:9s} {p/1e6:5.1f}M  dKL {base-kl:+.4f}  per100M {1e8*(base-kl)/p:+.4f}", flush=True)
print(f"sweep {time.time()-t0:.0f}s", flush=True)
cands.sort(key=lambda c: -c[2] / c[3])
json.dump(cands, open(f"{OUT}/sens.json", "w"))
for budget in [25e6, 50e6, 100e6, 150e6, 200e6, 300e6]:
    for i in range(6, r.L):
        r._wc[i].update(G4[i])
    used = 0; sel = []; pairs = []
    for i, nm, d, p in cands:
        if d > 0 and used + p <= budget:
            r._wc[i][nm] = I8[i][nm]; used += p; sel.append(f"L{i}.{nm.split('.')[-2]}"); pairs.append([i, nm])
    kl, t1 = score(); k4, t4 = score(list(range(4)))
    ms = used * 0.5 / 16e9 * 1e3
    json.dump(pairs, open(f"{OUT}/int8_mats_{int(budget/1e6)}M.json", "w"))
    print(f"budget {budget/1e6:4.0f}M (+{ms:.1f} ms/tok): KL {kl:.4f} top1 {t1:.3f} | 4 texts KL {k4:.3f} top1 {t4:.3f} | {len(sel)} mats: {' '.join(sel)}", flush=True)
