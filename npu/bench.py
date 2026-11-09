"""Task benchmark with automatic checks: tool calling, JSON extraction, instruction following, knowledge, math.

usage: bench.py BACKEND OUT.jsonl [--think] [--hard] [--only CAT] [--idx 3,7] [--dry]
BACKEND: lfm3-npu (the NPU build in LFM3_BUILD) | lfm3-ref (fp32 reference, CPU or REF_DEVICE=cuda)
Greedy decoding with repetition penalty 1.1 (LFM3 card's general setting, temp 0.1 ~ greedy).
lfm3-ref can simulate the NPU weights (weight-only, activations fp32):
  QW_GPTQ=gptq_dir [QW_INT8_LAYERS=0,..,5] [QW_INT8_MATS=configs/int8_mats_100M.json]"""
import sys, json, os, re, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import ChatFormat, MODEL_DIR, BUILD_DIR

def fn(name, desc, **params):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": v[0], "description": v[1]} for k, v in params.items()},
        "required": list(params)}}}

TOOLS = [
    fn("get_weather", "Get the current weather for a city.", city=("string", "City name")),
    fn("send_email", "Send an email.", to=("string", "Recipient address"), subject=("string", "Subject"), body=("string", "Body text")),
    fn("set_timer", "Start a countdown timer.", minutes=("integer", "Duration in minutes")),
    fn("convert_currency", "Convert money between currencies.", amount=("number", "Amount"),
       from_currency=("string", "ISO code, e.g. USD"), to_currency=("string", "ISO code, e.g. EUR")),
    fn("web_search", "Search the web for recent information.", query=("string", "Search query")),
]

# ---------- checks ----------
def strip_think(s):
    return s.split("</think>")[-1].strip()

def parse_calls(s):
    """LFM3 XML tool calls (and a JSON fallback) -> [(name, {arg: str})]."""
    calls = []
    for blk in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", s, re.S):
        m = re.search(r"<function=([\w\-]+)>", blk)
        if m:
            args = {k: v.strip() for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", blk, re.S)}
            calls.append((m.group(1), args))
            continue
        try:
            j = json.loads(blk.strip())
            calls.append((j["name"], {k: str(v) for k, v in j.get("arguments", {}).items()}))
        except Exception:
            calls.append(("<unparsable>", {}))
    return calls

def call_check(name, **want):
    def chk(out):
        c = parse_calls(strip_think(out))
        if not c:
            return False, "no tool call"
        n, a = c[0]
        if n != name:
            return False, f"called {n}"
        for k, f in want.items():
            if k not in a or not f(a[k]):
                return False, f"bad arg {k}={a.get(k)!r}"
        return True, ""
    return chk

def no_call(must=None):
    def chk(out):
        o = strip_think(out)
        if parse_calls(o):
            return False, "unexpected tool call"
        if must and not all(re.search(m, o, re.I) for m in must):
            return False, "missing " + str(must)
        return True, ""
    return chk

def json_check(want):
    def chk(out):
        o = strip_think(out)
        m = re.search(r"\{.*\}|\[.*\]", o, re.S)
        if not m:
            return False, "no json"
        try:
            j = json.loads(m.group(0))
        except Exception as e:
            return False, f"invalid json: {e}"
        for k, v in want.items():
            got = j.get(k) if isinstance(j, dict) else None
            ok = v(got) if callable(v) else (str(got).strip().lower() == str(v).lower())
            if not ok:
                return False, f"{k}={got!r}"
        return True, ""
    return chk

def contains(*pats):
    def chk(out):
        o = strip_think(out)
        miss = [p for p in pats if not re.search(p, o, re.I)]
        return (not miss), ("missing " + ",".join(miss) if miss else "")
    return chk

def last_number(want):
    def chk(out):
        o = strip_think(out).replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", o)
        if not nums:
            return False, "no number"
        ok = any(abs(float(n) - want) < 1e-6 for n in nums[-3:])
        return ok, "" if ok else f"last nums {nums[-3:]}"
    return chk

def fmt(pred, why):
    def chk(out):
        o = strip_think(out)
        return (pred(o), "" if pred(o) else why)
    return chk

eq = lambda v: (lambda s: s.strip().lower() == v.lower())
num = lambda v: (lambda s: re.sub(r"[^\d.\-]", "", s) not in ("", ".", "-") and abs(float(re.sub(r"[^\d.\-]", "", s)) - v) < 1e-6)
has = lambda v: (lambda s: v.lower() in s.lower())

TASKS = [
    # tool calling (system prompt with tools)
    ("tool", "What's the weather like in Lyon right now?", call_check("get_weather", city=has("lyon"))),
    ("tool", "Start a 25 minute timer please.", call_check("set_timer", minutes=num(25))),
    ("tool", "How much is 150 US dollars in euros?", call_check("convert_currency", amount=num(150), from_currency=eq("USD"), to_currency=eq("EUR"))),
    ("tool", "Email bob@example.com with the subject 'Lunch' and tell him I'll be 10 minutes late.",
     call_check("send_email", to=eq("bob@example.com"), subject=has("lunch"), body=has("late"))),
    ("tool", "Search the web for the latest Raspberry Pi release news.", call_check("web_search", query=has("raspberry"))),
    ("tool", "Convert 2000 yen to British pounds.", call_check("convert_currency", amount=num(2000), from_currency=eq("JPY"), to_currency=eq("GBP"))),
    ("tool", "I'm going for a run in Oslo, should I bring a jacket?", call_check("get_weather", city=has("oslo"))),
    ("tool", "Hi! Can you tell me what you are able to help me with?", no_call()),
    ("tool", "What is the capital of Italy?", no_call(["rome"])),
    ("tool", "Translate 'good morning' to Spanish.", no_call(["buenos d"])),
    ("tool_result", None, contains(r"-3", r"snow")),
    ("tool_result2", None, contains(r"137")),
    # JSON extraction
    ("json", "Extract name, age and city as a JSON object with keys name, age, city. Text: 'Maria Lopez, 42, moved to Valencia last year.' Output only the JSON.",
     json_check({"name": lambda v: v and "maria" in str(v).lower(), "age": lambda v: str(v) == "42", "city": "Valencia"})),
    ("json", "From this invoice line, output JSON with keys product, quantity, unit_price (number): '3 x USB-C cable @ 7.50 EUR each'. Only JSON.",
     json_check({"product": lambda v: v and "usb" in str(v).lower(), "quantity": lambda v: str(v) == "3", "unit_price": lambda v: v is not None and abs(float(v) - 7.5) < 1e-6})),
    ("json", "Classify the sentiment of this review as positive, negative or neutral and return JSON {\"sentiment\": ...}: 'The battery died after two days and support never answered.'",
     json_check({"sentiment": "negative"})),
    ("json", "Return a JSON object with keys date (YYYY-MM-DD) and time (HH:MM, 24h) for: 'see you on March 5th 2027 at 3:30 pm'. Only JSON.",
     json_check({"date": "2027-03-05", "time": "15:30"})),
    ("json", "Extract every email address from the text into JSON {\"emails\": [...]}: 'Contact ana@uni.edu or, for billing, pay@shop.io. Not: foo@'.",
     json_check({"emails": lambda v: isinstance(v, list) and sorted(map(str.lower, v)) == ["ana@uni.edu", "pay@shop.io"]})),
    ("json", "Turn this into JSON with keys title, priority (low/medium/high), due (YYYY-MM-DD): 'URGENT: renew the domain before 2026-12-01'. Only JSON.",
     json_check({"priority": "high", "due": "2026-12-01", "title": lambda v: v and "domain" in str(v).lower()})),
    # instruction following
    ("instr", "Is Paris in France? Answer with only YES or NO.", fmt(lambda o: o.strip().strip(".").upper() == "YES", "not exactly YES")),
    ("instr", "List exactly three fruits as a bulleted list using '- ' and nothing else.",
     fmt(lambda o: len([l for l in o.splitlines() if l.strip().startswith("- ")]) == 3 and len([l for l in o.splitlines() if l.strip()]) == 3, "not exactly 3 bullets")),
    ("instr", "Describe the ocean in one sentence written entirely in lowercase letters.",
     fmt(lambda o: o == o.lower() and len(o) > 10, "has uppercase")),
    ("instr", "Start your reply with 'OK:' and then say what 5 + 5 is.", fmt(lambda o: o.startswith("OK:") and "10" in o, "format")),
    ("instr", "Explain what a CPU does in fewer than 20 words.", fmt(lambda o: 0 < len(o.split()) < 20, "word count")),
    ("instr", "Reply with the word 'banana' repeated exactly 4 times, separated by spaces, nothing else.",
     fmt(lambda o: o.strip().strip(".").lower().split() == ["banana"] * 4, "format")),
    # knowledge
    ("know", "What is the capital of Australia? Answer briefly.", contains("canberra")),
    ("know", "What is the chemical symbol for gold?", contains(r"\bAu\b")),
    ("know", "Who wrote the novel '1984'?", contains("orwell")),
    ("know", "In what year did humans first land on the Moon?", contains("1969")),
    ("know", "Which planet is the largest in our solar system?", contains("jupiter")),
    ("know", "What gas do plants absorb from the air for photosynthesis?", contains(r"carbon dioxide|CO2|CO₂")),
    # math
    ("math", "A shop sells pens at 3 euros each. Anna buys 7 pens and pays with a 50 euro note. How much change does she get? End with the number.", last_number(29)),
    ("math", "What is 17 * 23?", last_number(391)),
    ("math", "A train travels 180 km in 2 hours. At the same speed, how many km does it travel in 5 hours?", last_number(450)),
    ("math", "If x + 2x + 3 = 18, what is x?", last_number(5)),
    ("math", "Tom has 12 apples. He gives a third to Sara and then eats 2. How many apples does Tom have left?", last_number(6)),
    ("math", "What is 15% of 240?", last_number(36)),
]

SYS_TOOLS = "You are a helpful assistant. Use the tools when they are needed; otherwise answer directly."

def messages(cat, q):
    if cat == "tool_result":
        return [{"role": "system", "content": SYS_TOOLS}, {"role": "user", "content": "What's the weather in Oslo?"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
                {"role": "tool", "content": '{"temperature_c": -3, "conditions": "light snow", "wind_kmh": 12}'}], True
    if cat == "tool_result2":
        return [{"role": "system", "content": SYS_TOOLS}, {"role": "user", "content": "How much is 100 euros in Swiss francs... no wait, in Japanese yen?"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "convert_currency",
                 "arguments": {"amount": 100, "from_currency": "EUR", "to_currency": "JPY"}}}]},
                {"role": "tool", "content": '{"result": 16137, "currency": "JPY"}'},
                ], True
    if isinstance(q, list):
        return [{"role": "system", "content": SYS_TOOLS}] + q, True
    if cat == "tool":
        return [{"role": "system", "content": SYS_TOOLS}, {"role": "user", "content": q}], True
    return [{"role": "user", "content": q}], False

# ---------- hard set ----------
def calls_check(want):
    """want: list of (name, {arg: pred}); all must appear (any order)."""
    def chk(out):
        c = parse_calls(strip_think(out))
        left = list(want)
        for n, a in c:
            for w in left:
                if w[0] == n and all(k in a and f(a[k]) for k, f in w[1].items()):
                    left.remove(w); break
        return (not left), ("missing " + ",".join(w[0] for w in left) + f" (got {[x[0] for x in c]})" if left else "")
    return chk

def tc(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}

LONG_EMAIL = ("Hi team,\n\nFollowing up on yesterday's call with Northwind Logistics. Their procurement lead, Priya Raman "
    "(priya.raman@northwind.example, +44 20 7946 0958), confirmed they want to move ahead with the pilot. Scope: 40 handheld "
    "scanners and 3 docking stations for the Leeds warehouse. They asked for delivery no later than 14 January 2027 and want "
    "invoicing in GBP. Budget ceiling is 18,500 pounds before tax. Note that the Bristol site is NOT part of this pilot, even "
    "though it came up during the call. Priya also mentioned that their CFO, Daniel Okafor, has to sign off on anything above "
    "15,000, so expect a short delay. Can someone prepare the quote by Friday?\n\nThanks,\nLaura")

HARD = [
    ("tool", "Check the weather in both Paris and Berlin.", calls_check([("get_weather", {"city": has("paris")}), ("get_weather", {"city": has("berlin")})])),
    ("tool", "Remind me in an hour and a half.", call_check("set_timer", minutes=num(90))),
    ("tool", "I've got 50 quid left, how many dollars is that?", call_check("convert_currency", amount=num(50), from_currency=eq("GBP"), to_currency=eq("USD"))),
    ("tool", "Book me a table for two at an Italian restaurant tonight.", fmt(lambda o: not any(n in ("send_email", "set_timer", "convert_currency", "get_weather") for n, _ in parse_calls(o)), "misused a tool")),
    ("tool", [{"role": "user", "content": "Get the weather in Lyon and email it to ana@example.com."}, tc("get_weather", city="Lyon"),
              {"role": "tool", "content": '{"temperature_c": 8, "conditions": "rain"}'}],
     call_check("send_email", to=eq("ana@example.com"), body=lambda b: "8" in b and "rain" in b.lower())),
    ("tool", [{"role": "user", "content": "What's the weather in Rome?"}, tc("get_weather", city="Rome"),
              {"role": "tool", "content": '{"temperature_c": 17, "conditions": "sunny"}'},
              {"role": "assistant", "content": "It's 17°C and sunny in Rome."},
              {"role": "user", "content": "Great. And how much are 20 euros in US dollars?"}],
     call_check("convert_currency", amount=num(20), from_currency=eq("EUR"), to_currency=eq("USD"))),
    ("tool", [{"role": "user", "content": "Weather in Tokyo?"}, tc("get_weather", city="Tokyo"),
              {"role": "tool", "content": '{"error": "service unavailable"}'}],
     fmt(lambda o: not parse_calls(o) and re.search(r"unavailable|could(n't| not)|unable|error|sorry", o, re.I), "did not report the error")),
    ("tool", "Set timers for 5 and for 12 minutes.", calls_check([("set_timer", {"minutes": num(5)}), ("set_timer", {"minutes": num(12)})])),
    ("json", "Extract the order lines from this message as JSON {\"items\": [{\"name\", \"qty\", \"price\"}]} (price per unit, number): "
             "'Hey, add two packs of AA batteries (4.99 each), one HDMI cable for 12 and, actually, make the batteries three packs.' Only JSON.",
     json_check({"items": lambda v: isinstance(v, list) and len(v) == 2 and any("batter" in str(i.get("name", "")).lower() and str(i.get("qty")) == "3" and abs(float(i.get("price", 0)) - 4.99) < 1e-6 for i in v)
                 and any("hdmi" in str(i.get("name", "")).lower() and str(i.get("qty")) == "1" and abs(float(i.get("price", 0)) - 12) < 1e-6 for i in v)})),
    ("json", "Return JSON with keys name, email, phone for this signature. Use null for anything missing. 'Best, Jordan Lee — jordan.lee@example.org'. Only JSON.",
     json_check({"name": lambda v: v and "jordan" in v.lower(), "email": "jordan.lee@example.org", "phone": lambda v: v is None})),
    ("json", "From the email below, return JSON with keys company, contact_name, contact_email, quantity_scanners, site, deadline (YYYY-MM-DD), "
             "currency, budget (number), approver. Only JSON.\n\n" + LONG_EMAIL,
     json_check({"company": lambda v: v and "northwind" in v.lower(), "contact_name": lambda v: v and "priya" in v.lower(),
                 "contact_email": "priya.raman@northwind.example", "quantity_scanners": lambda v: str(v) == "40", "site": lambda v: v and "leeds" in v.lower(),
                 "deadline": "2027-01-14", "currency": "GBP", "budget": lambda v: v is not None and abs(float(str(v).replace(",", "")) - 18500) < 1e-6,
                 "approver": lambda v: v and "okafor" in v.lower()})),
    ("json", "Is this message spam? Return JSON {\"spam\": true/false, \"reason\": \"...\"}: 'CONGRATULATIONS!!! You won a $1000 gift card, click bit.ly/xx to claim within 24h'",
     json_check({"spam": lambda v: v is True})),
    ("math", "A jacket costs 80 euros. It is discounted by 25%, then 21% VAT is added to the discounted price. What is the final price?", last_number(72.6)),
    ("math", "A father is 3 times as old as his son. In 12 years he will be twice as old as his son. How old is the son now?", last_number(12)),
    ("math", "How many minutes are there between 9:45 and 13:20?", last_number(215)),
    ("math", "Pipe A fills a tank in 6 hours and pipe B fills it in 3 hours. How many hours do they take together?", last_number(2)),
    ("math", "What is the sum of all integers from 1 to 50 that are divisible by 3?", last_number(408)),
    ("math", "If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", last_number(5)),
    ("math", "How many legs do 3 spiders and 2 ants have in total?", last_number(36)),
    ("know", "Which is heavier, a kilogram of feathers or a kilogram of steel?", contains(r"same|equal|neither|both weigh")),
    ("know", "What is the capital of Canada?", contains("ottawa")),
    ("know", "Who painted 'The Persistence of Memory'?", contains(r"dal[ií]")),
    ("know", "How many times does the letter r appear in the word 'strawberry'?", last_number(3)),
    ("know", "What do you get if you reverse the letters of the word 'stressed'?", contains("desserts")),
    ("instr", "Answer in JSON only, with a single key 'answer': what is the capital of Japan?", json_check({"answer": lambda v: v and "tokyo" in v.lower()})),
    ("instr", "Give me 5 English words that start with the letter q, comma-separated, lowercase, nothing else.",
     fmt(lambda o: len([w for w in o.strip().strip(".").split(",") if w.strip()]) == 5 and all(w.strip().startswith("q") and w.strip() == w.strip().lower() for w in o.strip().strip(".").split(",") if w.strip()), "format")),
    ("instr", "Write exactly two sentences about cats. Each sentence must end with an exclamation mark.",
     fmt(lambda o: len(re.findall(r"[.!?](\s|$)", o)) == 2 and o.count("!") == 2 and "." not in o.replace("...", ""), "format")),
    ("instr", "Write a haiku about rain. Output only the three lines.", fmt(lambda o: len([l for l in o.splitlines() if l.strip()]) == 3, "not 3 lines")),
]
if "--hard" in sys.argv:
    TASKS = HARD


def load_ref():
    import torch
    torch.set_grad_enabled(False)
    from lfm3_ref import LFM3Ref
    m = LFM3Ref(str(MODEL_DIR), device=os.environ.get("REF_DEVICE", "cpu"))
    # the checkpoint is bf16: keep it resident as bf16 and upcast one layer at a time (same numbers, half the RAM)
    m._wc = {i: {k: v.to(torch.bfloat16).to(m.dev) for k, v in m.W.layer(i).items()} for i in range(m.L)}
    # QW_GPTQ=dir [QW_INT8_LAYERS=0,1,..] [QW_INT8_MATS=file.json]: simulate the NPU weights (GPTQ int4 dequantized,
    # RTN int8 per-channel for the listed layers / matrices); activations stay fp32. Quantized matrices kept in fp32.
    import os as _os
    if _os.environ.get("QW_GPTQ"):
        import numpy as _np
        _l8 = {int(v) for v in _os.environ.get("QW_INT8_LAYERS", "").split(",") if v}
        _m8 = {(int(a), b) for a, b in json.load(open(_os.environ["QW_INT8_MATS"]))} if _os.environ.get("QW_INT8_MATS") else set()
        def _rtn8(W):
            qmax = 127; Wt = W.T; amax = Wt.abs().amax(0) + 1e-12
            best_e = torch.full_like(amax, float("inf")); best_s = amax / qmax
            for rr in _np.linspace(0.6, 1.0, 17):
                sc = amax * float(rr) / qmax; e = ((torch.clamp(torch.round(Wt / sc), -qmax - 1, qmax) * sc - Wt) ** 2).sum(0)
                b = e < best_e; best_e[b], best_s[b] = e[b], sc[b]
            return (torch.clamp(torch.round(Wt / best_s), -qmax - 1, qmax) * best_s).T.contiguous()
        n4 = n8 = 0
        for i in range(m.L):
            z = _np.load(f"{_os.environ['QW_GPTQ']}/layer{i}.npz")
            for k in z.files:
                if not k.endswith("_q"):
                    continue
                nm = k[:-2]
                if i in _l8 or (i, nm) in _m8:
                    m._wc[i][nm] = _rtn8(m.W.get(f"model.layers.{i}.{nm}").to(m.dev)); n8 += 1
                else:
                    m._wc[i][nm] = torch.from_numpy((z[k].astype(_np.float32) * z[nm + "_s"][None]).T.copy()).to(m.dev); n4 += 1
        print(f"simulated NPU weights: {n4} GPTQ int4 + {n8} RTN int8 matrices", flush=True)
    _lf = m.layer_forward
    m.layer_forward = lambda i, x, pos, past, w=None, **kw: _lf(i, x, pos, past, w={k: v.float() for k, v in m._wc[i].items()}, **kw)
    return m


def main():
    back, out_path = sys.argv[1], sys.argv[2]
    think = "--think" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    idx = [int(x) for x in sys.argv[sys.argv.index("--idx") + 1].split(",")] if "--idx" in sys.argv else None
    cf = ChatFormat()
    tok = cf.tok
    if "--dry" in sys.argv:
        for cat, q, _ in TASKS:
            msgs, tools = messages(cat, q)
            print(cat, len(cf.encode(msgs, TOOLS if tools else None, think)))
        return
    if back == "lfm3-npu":
        from lfm3_npu import LFM3NPU
        m = LFM3NPU(BUILD_DIR); eos = cf.stop_ids(m.man["eos"]); ctx = m.T
        def start(ids):
            m.reset(); return m.logits_f32(m.feed(ids))
        nxt = lambda t: m.logits_f32(m.step(t))
    elif back == "lfm3-ref":
        m = load_ref(); eos = cf.stop_ids([m.c["eos_token_id"]]); ctx = 4096
        st = {}
        def start(ids):
            l, st["past"] = m.forward(ids); st["pos"] = len(ids); return l[-1].float().cpu().numpy()
        def nxt(t):
            l, st["past"] = m.forward([t], start=st["pos"], past=st["past"]); st["pos"] += 1
            return l[-1].float().cpu().numpy()
    else:
        sys.exit(f"unknown backend {back}")

    def run(ids, n, rp=1.1):
        t0 = time.time(); lg = start(ids); t1 = time.time()
        out, seen = [], set()
        for _ in range(n):
            lg = lg.astype(np.float32).copy()
            if seen:
                ix = np.fromiter(seen, int); v = lg[ix]; lg[ix] = np.where(v > 0, v / rp, v * rp)
            t = int(np.argmax(lg))
            if t in eos:
                break
            out.append(t); seen.add(t)
            if len(ids) + len(out) >= ctx:
                break
            lg = nxt(t)
        return out, t1 - t0, time.time() - t1

    budget = 900 if think else 400
    res = []
    f = open(out_path, "w", encoding="utf-8")
    for ti, (cat, q, chk) in enumerate(TASKS):
        if idx is not None and ti not in idx:
            continue
        if only and not cat.startswith(only):
            continue
        msgs, tools = messages(cat, q)
        ids = cf.encode(msgs, TOOLS if tools else None, think)
        n = min(budget, ctx - len(ids) - 1)
        if n < 32:
            r = {"cat": cat, "q": q, "ok": False, "why": f"prompt {len(ids)} tok > ctx", "skip": True}
        else:
            out, tp, td = run(ids, n)
            text = tok.decode(out, skip_special_tokens=False)
            ok, why = chk(text)
            r = {"cat": cat, "q": q, "ok": bool(ok), "why": why, "out": text, "prompt_tok": len(ids), "gen_tok": len(out),
                 "prefill_s": round(tp, 2), "tok_s": round(len(out) / td, 2) if out else 0, "truncated": len(out) >= n}
        res.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {cat:12s} {r.get('gen_tok', 0):4d} tok {r.get('tok_s', 0):5.1f} t/s  {(q if isinstance(q, str) else cat)[:60]!r} {r['why']}", flush=True)
    cats = sorted({r["cat"].split("_")[0] for r in res})
    print("\n" + back + (" think" if think else "") + ": " + " | ".join(
        f"{c} {sum(r['ok'] for r in res if r['cat'].startswith(c))}/{sum(1 for r in res if r['cat'].startswith(c))}" for c in cats)
        + f" | total {sum(r['ok'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
