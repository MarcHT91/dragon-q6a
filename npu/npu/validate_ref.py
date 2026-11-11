"""Validate lfm3_ref vs HF (5-layer truncation: conv,conv,full,conv,sliding), incremental path,
then full-model greedy generation and activation-range report (is there a position-0 sink?)."""
import sys, time, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

M = sys.argv[1]
NL = 5
tok = AutoTokenizer.from_pretrained(M)
msgs = [{"role": "user", "content": "Explain briefly what an NPU is, and give one example of a device that has one."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False).input_ids
print("prompt:", repr(text[:200]), "| tokens", len(ids), "| first ids", ids[:4], flush=True)

try:
    cfg = AutoConfig.from_pretrained(M)
    native = True
except Exception:
    cfg = AutoConfig.from_pretrained(M, trust_remote_code=True); native = False
cfg.num_hidden_layers = NL; cfg.layer_types = cfg.layer_types[:NL]
hf = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.float32, trust_remote_code=not native)
print("HF impl:", "native transformers" if native else "remote code", flush=True)
with torch.no_grad():
    hl = hf(torch.tensor([ids])).logits[0]
del hf
r = LFM3Ref(M, n_layers=NL)
rl, _ = r.forward(ids)
print(f"{NL}-layer logits rel err vs HF: {(rl-hl).norm()/hl.norm():.2e}  argmax agree {(rl.argmax(-1)==hl.argmax(-1)).float().mean():.3f}", flush=True)
l1, past = r.forward(ids[:-3])
for j in range(3):
    l2, past = r.forward(ids[-3 + j:len(ids) - 2 + j] if j < 2 else ids[-1:], start=len(ids) - 3 + j, past=past)
print(f"incremental vs full last logits rel err: {(l2[-1]-rl[-1]).norm()/rl[-1].norm():.2e}", flush=True)
del r

ref = LFM3Ref(M)
stats = {}
t = time.time()
logits, _ = ref.forward(ids, stats=stats)
print(f"full prefill {time.time()-t:.1f}s", flush=True)
for i in range(ref.L):
    v = stats[(i, "x_out")][0]
    print(f"layer{i:2d} {ref.types[i][:5]:5} x_out pos0 {v[0]:8.1f} | rest max {v[1:].max():7.1f} @ {1+int(v[1:].argmax()):2d} median {v[1:].median():6.2f}"
          f" | act max {stats[(i,'act')][0].max():8.1f} ffn {stats[(i,'ffn')][0].max():7.1f} op {stats[(i,'op')][0].max():7.1f}")
eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
t = time.time()
out = ref.generate(ids, 40, eos=eos)
print(f"greedy 40 tok {time.time()-t:.1f}s:", repr(tok.decode(out)))
