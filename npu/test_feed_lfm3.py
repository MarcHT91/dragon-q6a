import sys, time, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_npu import LFM3NPU
from transformers import PreTrainedTokenizerFast
M, B = sys.argv[1], sys.argv[2]
tok = PreTrainedTokenizerFast.from_pretrained(M)
tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
         for n, d in [("get_weather", "Current weather for a city"), ("search_web", "Search the web"), ("send_email", "Send an email to someone"),
                      ("create_event", "Create a calendar event"), ("calculator", "Evaluate arithmetic")]]
text = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant running on a small board. Use tools when needed."},
                                {"role": "user", "content": "What's the weather in Girona, and email the result to ana@example.com?"}],
                               tools=tools, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False).input_ids
print("prompt tokens:", len(ids))
npu = LFM3NPU(B)
lp = lambda z: z - z.max() - np.log(np.exp(z - z.max()).sum())
t = time.time(); npu.reset(); a = npu.logits_f32(npu.feed(ids)); ta = time.time() - t
C = npu.C; npu.C = 0
t = time.time(); npu.reset(); b = npu.logits_f32(npu.feed(ids)); tb = time.time() - t
npu.C = C
pa, pb = lp(a), lp(b)
print(f"chunked prefill {ta:.2f}s ({len(ids)/ta:.0f} tok/s) | token-by-token {tb:.2f}s ({len(ids)/tb:.0f} tok/s)")
print(f"last-token logits: argmax {a.argmax()} vs {b.argmax()} | KL(step||chunk) {(np.exp(pb)*(pb-pa)).sum():.4f} | top5 {tok.decode(np.argsort(-a)[:5])!r} vs {tok.decode(np.argsort(-b)[:5])!r}")
if "--ref" in sys.argv:
    from lfm3_ref import LFM3Ref
    r = LFM3Ref(M).forward(ids)[0][-1].numpy(); pr = lp(r)
    print(f"vs fp32: KL(ref||chunk) {(np.exp(pr)*(pr-pa)).sum():.4f} KL(ref||step) {(np.exp(pr)*(pr-pb)).sum():.4f} | ref top5 {tok.decode(np.argsort(-r)[:5])!r}")
out, st = npu.generate(ids, 120)
print(st); print(tok.decode(out))
