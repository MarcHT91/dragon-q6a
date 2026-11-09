"""Generate in-domain calibration data for LFM3 GPTQ: the fp32 model answers agent/tool/JSON/
instruction/coding/math prompts (with its thinking traces). Run on the Windows GPU.

usage: python gen_calib_lfm3.py MODEL_DIR OUT.jsonl [max_new=384]
Each line: {"ids": [...]}  (prompt + generated tokens, chat-formatted with the model's own template)
"""
import sys, json, time, random
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_ref import LFM3Ref
from transformers import PreTrainedTokenizerFast

M, OUT = sys.argv[1], sys.argv[2]
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 384
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = PreTrainedTokenizerFast.from_pretrained(M)   # plain tokenizer: no remote-code prompt
ref = LFM3Ref(M, device=DEV)
ref.cache_weights()                      # ~7 GB fp32 on the GPU; without this every token re-reads the checkpoint on the CPU
print("device", DEV, "| weights cached", flush=True)
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), ref.c["eos_token_id"]}

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
S = lambda d: {"type": "string", "description": d}
N = lambda d: {"type": "number", "description": d}
TOOLS = [
    fn("get_weather", "Current weather for a city", {"city": S("City name"), "unit": S("celsius or fahrenheit")}, ["city"]),
    fn("search_web", "Search the web", {"query": S("Search query")}, ["query"]),
    fn("send_email", "Send an email", {"to": S("Recipient"), "subject": S("Subject"), "body": S("Body")}, ["to", "subject", "body"]),
    fn("create_event", "Create a calendar event", {"title": S("Title"), "date": S("YYYY-MM-DD"), "time": S("HH:MM")}, ["title", "date"]),
    fn("calculator", "Evaluate an arithmetic expression", {"expression": S("Expression")}, ["expression"]),
    fn("read_file", "Read a text file", {"path": S("File path")}, ["path"]),
    fn("set_timer", "Set a timer", {"seconds": N("Duration in seconds")}, ["seconds"]),
    fn("get_stock_price", "Latest stock price", {"ticker": S("Ticker symbol")}, ["ticker"]),
]
TOOL_Q = ["What's the weather like in Barcelona right now?", "Is it raining in Tokyo? Use fahrenheit.",
          "Email alice@example.com that the meeting moved to 3pm.", "Schedule a dentist appointment on 2026-12-03 at 10:30.",
          "What is 1234 * 5678 - 91?", "Find recent news about RISC-V laptops.", "Set a timer for 7 minutes.",
          "Read the file /etc/hostname and tell me what it says.", "How much is NVDA trading at?",
          "Remind me to call mom tomorrow (2026-11-10) at 18:00 and email bob@example.com about it.",
          "Search for the population of Iceland and then compute it divided by 1000.",
          "What's the weather in Paris and in Berlin?"]
GEN_Q = ["Explain how a heat pump works in three sentences.", "Summarize the plot of Romeo and Juliet in two lines.",
         "Give me five tips for writing clean Python code.", "What are the main differences between TCP and UDP?",
         "Write a short professional reply declining a meeting invitation.", "Translate 'Where is the train station?' into Spanish, French and German.",
         "List the planets of the solar system in order from the Sun.", "Explain what a hash table is to a beginner.",
         "Write a haiku about autumn.", "What is the difference between RAM and storage?",
         "Explain recursion with a simple example.", "Give a 3-step plan to learn a new language.",
         "What causes the seasons on Earth?", "Rewrite this sentence to be more formal: 'gonna be late, sorry'.",
         "Describe the water cycle briefly.", "Why do we need sleep?"]
JSON_Q = ["Extract name, email and company as JSON from: 'Hi, I'm Marta Vidal from Acme Corp, write me at marta@acme.io'.",
          "Return a JSON array of 3 fictional books with title, author and year.",
          "Parse this into JSON with fields date, amount, currency: 'Paid 42.50 EUR on 3 March 2025'.",
          "Classify the sentiment (positive/negative/neutral) of: 'The battery life is disappointing.' Answer as JSON.",
          "Convert this list to YAML: apples 3, pears 5, kiwis 2.", "Output only valid JSON: an object describing a cat named Mio, age 4, indoor."]
IF_Q = ["Answer in exactly three bullet points: why is exercise healthy?", "Write one sentence about the sea, all in lowercase, without commas.",
        "Reply with only the word YES or NO: is 17 a prime number?", "Write a 4-line poem where every line starts with the letter S.",
        "Give two reasons to learn math. Use numbered list and keep each under ten words.", "Respond in uppercase: what is the capital of Italy?"]
CODE_Q = ["Write a Python function that checks if a string is a palindrome.", "Write a bash one-liner that counts lines in all .py files.",
          "Fix the bug: def add(a,b): return a-b", "Write a SQL query that returns the 5 most expensive products.",
          "Implement binary search in Python with a docstring.", "Explain what this does: [x*x for x in range(10) if x%2==0]"]
MATH_Q = ["What is 17 * 23?", "Solve for x: 3x + 7 = 22.", "A train travels 180 km in 2.5 hours. What is its average speed?",
          "What is 15% of 240?", "If I have 3 red and 5 blue balls, what is the probability of drawing a red one?", "Compute the derivative of x^3 + 2x."]
ML_Q = ["Explica breument què és la fotosíntesi.", "¿Qué es una red neuronal? Responde en dos frases.", "Qu'est-ce que le cloud computing ?",
        "Was ist ein Mikrocontroller?"]

convs = []
for q in TOOL_Q:
    convs.append(([{"role": "system", "content": "You are a helpful assistant with tools. Call tools when needed."},
                   {"role": "user", "content": q}], list({t["function"]["name"]: t for t in random.Random(len(convs)).sample(TOOLS, 6)}.values())))
for q in GEN_Q + JSON_Q + IF_Q + CODE_Q + MATH_Q + ML_Q:
    convs.append(([{"role": "user", "content": q}], None))
for q in GEN_Q[:8] + IF_Q[:4]:   # no-thinking variant
    convs.append(([{"role": "user", "content": q}], "nothink"))

import os
convs = convs[:int(os.environ.get("LIMIT", len(convs)))]
print(len(convs), "conversations", flush=True)

def sample(logits, prev, temp=0.1, top_k=50, rep=1.1):
    l = logits.float().clone()
    if prev:
        idx = torch.tensor(sorted(set(prev[-256:])), device=l.device)
        v = l[idx]; l[idx] = torch.where(v > 0, v / rep, v * rep)
    v, i = torch.topk(l, top_k)
    p = torch.softmax(v / temp, -1)
    return int(i[torch.multinomial(p, 1)])

torch.manual_seed(0)
t0 = time.time(); total = 0
with open(OUT, "w") as f:
    for n, (msgs, tools) in enumerate(convs):
        kw = {"enable_thinking": False} if tools == "nothink" else {}
        text = tok.apply_chat_template(msgs, tools=tools if isinstance(tools, list) else None, tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, add_special_tokens=False).input_ids
        logits, past = ref.forward(ids)
        out = []
        for _ in range(NEW):
            t = sample(logits[-1], ids + out)
            out.append(t)
            if t in EOS:
                break
            logits, past = ref.forward([t], start=len(ids) + len(out) - 1, past=past)
        f.write(json.dumps({"ids": ids + out}) + "\n"); f.flush()
        total += len(ids) + len(out)
        print(f"{n+1}/{len(convs)} {len(ids)}+{len(out)} tok  {time.time()-t0:.0f}s | {tok.decode(out[:60])!r}", flush=True)
print("done", total, "tokens")
