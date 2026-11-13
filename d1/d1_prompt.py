"""d1-2B prompt and readout, without torch: what the fp32 reference (d1_ref.py) and the NPU runtime (d1_npu.py)
share. A backend gives `slot_logprobs(ids, state_len, groups_per_token, feat, slots)`.

How d1 reads a request (runner.py, prompt.py, attention.py of the release):
- prompt (ChatML, no system turn):  <|startoftext|><|im_start|>user\\n{media}{state}\\n\\n\\nQUESTION:\\n{question}
  <|im_end|>\\n<|im_start|>assistant\\n<think></think>   -- the answer slot is the last token. A state that is not a
  string is written as json.dumps(state, indent=2). With no state, the question follows the media directly.
- `set` attention: up to d1_set_size (16) questions packed after one state, each further one as "QUESTION:\\n" +
  its suffix. On the 6 full-attention layers, a state token sees every token (the whole state and all questions)
  and a question token sees the state and its own question, both ways. Sliding-window and conv layers stay causal.
- readout: log-softmax at the slot; each option scores its best token form (yes/Yes/YES; option codes A, B.. and
  " A"; digits 0..n-1); the scores are divided by the calibration temperature (bucketed by type and size) and
  softmaxed. A multiple choice puts a "<json name>:" marker per option after the assistant lead, a yes/no slot each,
  P(yes) mapped through calibration table "multiple".
"""
import bisect, json, math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

IM_START, IM_END, LEAD, HEADER = "<|im_start|>", "<|im_end|>", "<think></think>", "QUESTION:\n"
LETTERS = [chr(ord("A") + i) for i in range(26)]
FALLBACK = (LETTERS + [f"{i:02d}" for i in range(100)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
            + [f"#{i}" for i in range(200)] + [a + b for a in LETTERS for b in LETTERS])
IMG, VSTART, VEND = 125030, 125009, 125010



def set_mask(n, s, g):
    """[n, n] bool: which keys each query sees on the full-attention layers under `set` attention"""
    qi = np.arange(n)[:, None]; ki = np.arange(n)[None]; gg = np.asarray(g)
    return (qi < s) | (ki < s) | (gg[:, None] == gg[None, :])


class D1Prompt:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        cal = self.dir / "calibration.json"
        self.cal = json.loads(cal.read_text()) if cal.exists() else {"default": 1.0, "temperatures": {}, "tables": {}}
        self.set_size = self.cfg.get("d1_set_size", 4)
        self.scheme = self.cfg.get("d1_codes", 1)
        self.max_pixels = self.cfg.get("d1_vision_max_pixels", 1024 * 1024)

    def slot_logprobs(self, ids, s, g, feat, slots):
        """log-softmax over the vocabulary at each slot position: [len(slots), vocab] (a backend's forward)"""
        raise NotImplementedError

    # ------------------------------------------------------------ tokens
    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def single(self, s):
        e = self.enc(s)
        return e[0] if len(e) == 1 else None

    def codes(self, labels):
        """option_codes + aliases: one distinct single-token code per option."""
        labs = [str(x).strip() for x in labels]
        if labs and all(len(k) == 1 and k.isalpha() for k in labs):
            want = labs
        elif len(labs) <= 26:
            want = LETTERS[:len(labs)]
        elif self.scheme == 2:
            want = (LETTERS + [a + b for a in LETTERS for b in LETTERS])[:len(labs)]
        else:
            want = [f"{i:02d}" for i in range(len(labs))]
        out, used = [], set()
        for c in want:
            for raw in [c] + FALLBACK:
                t = self.single(raw)
                if t is not None and t not in used:
                    out.append((raw, t)); used.add(t); break
            else:
                raise RuntimeError("no single-token code left")
        return out

    def forms(self, texts):
        out = []
        for t in texts:
            i = self.single(t)
            if i is not None and i not in out:
                out.append(i)
        return out

    # ------------------------------------------------------------ prompt
    @staticmethod
    def state_block(state):
        return f"{state}\n\n" if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"

    def prefix(self, state, media=""):
        body = "" if state is None else self.state_block(state) + "\n" + HEADER
        return f"<|startoftext|>{IM_START}user\n{media}{body}"

    def block(self, q):
        t, said = q["type"], q["instructions"]
        if t == "choice" and q.get("multiple"):
            lines = "\n".join(f"{json.dumps(n, ensure_ascii=False)}: {q['criteria'][n] or n}" for n in q["criteria"])
            return f"{said}\n\nOptions:\n{lines}\n\nReply with yes or no for each option."
        if t == "choice":
            names = list(q["criteria"])
            lines = "\n".join(f"{code} {q['criteria'][n] or n.replace('_', ' ')}" for (code, _), n in zip(self.codes(names), names))
            return f"{said}\n\nOptions:\n{lines}\n\nReply with the option code only."
        if t == "noul":
            c = q.get("criteria")
            extra = f"\nYes: {c.get('true')}\nNo: {c.get('false')}" if c else ""
            return f"{said}{extra}\n\nReply with yes or no only."
        if t == "score":
            legend = "\n".join(f"{i} {n}" for i, n in enumerate(q["criteria"]))
            return f"{said}\n\n{legend}\n\nReply with a single digit 0-{len(q['criteria']) - 1} only."
        raise ValueError(f"type {t} not covered by this reference")

    def suffix(self, q):
        """(suffix text, slot offsets from its last token)"""
        text = f"{self.block(q)}{IM_END}\n{IM_START}assistant\n{LEAD}"
        if not (q["type"] == "choice" and q.get("multiple")):
            return text, [0]
        ends = []
        for i, n in enumerate(q["criteria"]):
            text += ("\n" if i else "") + json.dumps(n, ensure_ascii=False) + ":"
            ends.append(len(text))
        e = self.tok.encode(text, add_special_tokens=False)
        slots = []
        for ch in ends:
            m = [i for i, (a, b) in enumerate(e.offsets) if a <= ch - 1 < b]
            assert len(m) == 1, "slot marker without an exact colon token"
            slots.append(m[0] - len(e.ids) + 1)
        return text, slots

    def pack(self, prefix, qs):
        """pack_set: ids, state length, per-token question (-1 state), slot positions per question"""
        sfx = [self.suffix(q) for q in qs]
        ids = self.enc(prefix + sfx[0][0])
        s = len(ids) - len(self.enc(sfx[0][0]))
        g = [-1] * s + [1] * (len(ids) - s)
        ends = [[len(ids) - 1 + o for o in sfx[0][1]]]
        for k, (text, offs) in enumerate(sfx[1:], 2):
            own = self.enc((HEADER if prefix.endswith(HEADER) else "") + text)
            ids, g = ids + own, g + [k] * len(own)
            ends.append([len(ids) - 1 + o for o in offs])
        return ids, s, g, ends

    def temperature(self, q):
        t = q["type"]
        if t == "noul":
            b = "noul"
        elif t == "score":
            b = f"score:{len(q['criteria'])}"
        else:
            k = len(q["criteria"])
            b = "choice:small" if k <= 3 else "choice:mid" if k <= 8 else "choice:large" if k <= 26 else "choice:inventory"
        return self.cal["temperatures"].get(b, self.cal.get("default", 1.0))

    def groups(self, q):
        if q["type"] == "noul" or q.get("multiple"):
            return [self.forms(["yes", "Yes", "YES"]), self.forms(["no", "No", "NO"])]
        if q["type"] == "score":
            return [self.forms([str(i)]) for i in range(len(q["criteria"]))]
        return [[t] + [i for i in self.forms([f" {c}"]) if i != t] for c, t in self.codes(list(q["criteria"]))]

    @staticmethod
    def through(v, knots):
        if not knots:
            return v
        xs = [a for a, _ in knots]
        j = bisect.bisect_left(xs, v)
        if j == 0:
            return knots[0][1]
        if j == len(knots):
            return knots[-1][1]
        (a, pa), (b, pb) = knots[j - 1], knots[j]
        return pa + (v - a) * (pb - pa) / (b - a)

    def ask(self, state, questions, picture=None, size=None, feat=None):
        """{name: answer} like model.system_one (noul / choice / score), and the raw slot log-probs."""
        names, qs = list(questions), list(questions.values())
        media = ""
        if picture is not None and feat is None:
            feat = self.picture(picture, size)          # a backend with pictures
        if feat is not None:
            media = "<|vision_start|>" + "<|image|>" * len(feat) + "<|vision_end|>"
        prefix = self.prefix(state, media)
        out = {}
        for k0 in range(0, len(qs), self.set_size):
            part = qs[k0:k0 + self.set_size]
            ids, s, g, ends = self.pack(prefix, part)
            flat = [p for e in ends for p in e]
            lz = self.slot_logprobs(ids, s, g, feat, flat)
            at = 0
            for name, q, slot in zip(names[k0:k0 + self.set_size], part, ends):
                logz = lz[at:at + len(slot)]; at += len(slot)                          # [slots, vocab]
                grp = self.groups(q)
                if q.get("multiple"):
                    py = [self.softmax([max(float(z[i]) for i in gr) for gr in grp])[0] for z in logz]   # untempered
                    py = [self.through(p, self.cal["tables"].get("multiple")) for p in py]
                    out[name] = {"type": "choice", "choices": [n for n, p in zip(q["criteria"], py) if p >= 0.5],
                                 "probabilities": dict(zip(q["criteria"], py))}
                    continue
                T = self.temperature(q)
                p = self.softmax([max(float(logz[0][i]) for i in gr) / T for gr in grp])
                if q["type"] == "noul":
                    out[name] = {"type": "noul", "noul": p[0]}
                elif q["type"] == "choice":
                    b = int(np.argmax(p)); cr = list(q["criteria"])
                    out[name] = {"type": "choice", "choice": cr[b], "confidence": p[b], "probabilities": dict(zip(cr, p))}
                else:
                    acc, lvl = 0.0, len(p) - 1
                    for i, v in enumerate(p):
                        acc += v
                        if acc >= 0.5:
                            lvl = i; break
                    out[name] = {"type": "score", "score": float(lvl), "probabilities": p,
                                 "legend": {str(i): t for i, t in enumerate(q["criteria"])}}
        return out

    @staticmethod
    def softmax(v):
        m = max(v); e = [math.exp(x - m) for x in v]; s = sum(e)
        return [x / s for x in e]


# the model card's example questions (README "Transformers quick start")
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                          "technical": "App or site faults"}},
    "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
               "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                            "login": "Signing in to the account"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]},
}
