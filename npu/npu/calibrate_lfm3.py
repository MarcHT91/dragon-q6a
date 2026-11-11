"""Activation abs-max per (layer, tensor) for LFM3 a16 quantization -> calib.json
usage: calibrate_lfm3.py MODEL_DIR OUT [GPTQ_DIR]  (with GPTQ_DIR, calibrates the quantized-weight model)"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lfm3_ref import LFM3Ref
from tokenizers import Tokenizer

M, OUT = sys.argv[1], sys.argv[2]
GQ = sys.argv[3] if len(sys.argv) > 3 else None
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
def chat(u, think, a):
    return f"<|startoftext|><|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{a}<|im_end|>\n"
TEXTS = [
    chat("Explain briefly what an NPU is.", "The user wants a short explanation of NPUs. I should define it and give an example.",
         "An NPU (Neural Processing Unit) is a processor specialized for neural-network inference. Phones such as the Pixel 9 and laptops with Snapdragon X chips include one."),
    chat("Explica breument què és una NPU, en català.", "L'usuari vol una explicació curta en català.",
         "Una NPU és un processador especialitzat a executar xarxes neuronals de manera eficient i amb poc consum."),
    chat("Write a Python function that returns the n-th Fibonacci number.", "Iterative solution, O(n).",
         "```python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"),
    chat("What is 17 * 23? Show the steps.", "17*20 = 340, 17*3 = 51, total 391.", "17 × 23 = 340 + 51 = 391."),
    chat("¿Cuál es la capital de Francia?", "París.", "La capital de Francia es París, atravesada por el río Sena."),
    chat("Call the weather tool for Barcelona.", "I should emit a tool call with the city argument.",
         '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Barcelona"}}\n</tool_call>'),
]
ref = LFM3Ref(M)
ref._st = {}
if GQ:
    from build_lfm3 import dequant_layer
    orig = ref.W.layer
    ref.W.layer = lambda i: dequant_layer(orig(i), np.load(f"{GQ}/layer{i}.npz"))
t = time.time()
for txt in TEXTS:
    ids = tok.encode(txt, add_special_tokens=False).ids
    ref.forward(ids)
    print(f"{len(ids):4d} tok {time.time()-t:6.1f}s", flush=True)
out = {"layers": [dict() for _ in range(ref.L)], "head": {}}
for (i, nm), v in ref._st.items():
    (out["head"] if i == "head" else out["layers"][i])[nm] = v
json.dump(out, open(OUT, "w"), indent=1)
for nm in ["x_out", "act", "scores", "bx", "k_all", "v_all"]:
    print(f"{nm:7}", " ".join(f"{out['layers'][i].get(nm, 0):.3g}" for i in range(ref.L)))
print("head", out["head"])
