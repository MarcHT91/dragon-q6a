"""fp32 reference of LiquidAI d1-2B (decision model on LFM3-VL-2B), written from the model card and the released
model code (read, not run): prompt, `set` attention, and the option readout. For checking an NPU port.

Covered: noul (yes/no), choice (single and multiple), score; text/JSON states and pictures. Not covered: the typed
reads (text/number/date spans), estimates on a grid, boxes.

  python d1_ref.py MODEL_DIR             # the model card's examples, with their expected answers

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
import json, math, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "npu" / "npu", HERE.parent / "npu", HERE.parent / "vl" / "encoder", HERE.parent / "vl", HERE):   # repo layout, or the PC workspace
    sys.path.insert(0, str(_p))
from lfm3_ref import LFM3Ref, rms  # noqa: E402
from d1_prompt import D1Prompt, IMG, QUESTIONS  # noqa: E402

torch.set_grad_enabled(False)


class D1Ref(D1Prompt):
    """fp32 reference backend (torch, CPU or CUDA)."""

    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir)
        self.lm = LFM3Ref(self.dir, device=device)
        self.dev = self.lm.dev
        self.vision = None

    def load_vision(self):
        import vl_enc as V
        from safetensors import safe_open
        f = safe_open(str(self.dir / "model.safetensors"), framework="pt")
        self.V = V
        self.vision = {k.replace("model.", "", 1): f.get_tensor(k).float().to(self.dev) for k in f.keys()
                       if k.startswith(("model.vision_tower", "model.multi_modal_projector"))}

    def picture(self, path, size=None):
        """features [N, 2048] (fp32 reference encoder) for a picture; size=(H, W) forces the resize."""
        from PIL import Image
        if self.vision is None:
            self.load_vision()
        img = Image.open(path).convert("RGB")
        if size is None and img.width * img.height > self.max_pixels:          # media.cap_pixels
            sc = math.sqrt(self.max_pixels / (img.width * img.height))
            img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BICUBIC)
        tmp = Path("/tmp") if Path("/tmp").exists() else Path(".")
        p = tmp / "_d1_pic.png"; img.save(p)
        return self.V.reference(self.vision, self.V.preprocess(str(p), size).to(self.dev))[1]

    # ------------------------------------------------------------ forward + readout
    def forward(self, ids, s, g, feat=None):
        """hidden states after the final norm [T, hid] under `set` attention"""
        t = torch.as_tensor(ids, device=self.dev)
        x = self.lm.embed[t].clone()
        if feat is not None:
            m = t == IMG
            x[m] = torch.as_tensor(feat, dtype=x.dtype, device=self.dev)[: int(m.sum())]
        n = len(ids)
        qi = torch.arange(n)[:, None]; ki = torch.arange(n)[None]
        gg = torch.as_tensor(g)
        self.lm.full_allow = (qi < s) | (ki < s) | (gg[:, None] == gg[None, :])
        pos = list(range(n))
        for i in range(self.lm.L):
            x, _ = self.lm.layer_forward(i, x, pos, None)
        self.lm.full_allow = None
        return rms(x, self.lm.fnorm, self.lm.eps)

    def slot_logprobs(self, ids, s, g, feat, slots):
        h = self.forward(ids, s, g, feat)
        return torch.log_softmax(h[slots] @ self.lm.embed.T, -1).cpu()


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = D1Ref(sys.argv[1], dev); m.lm.cache_weights()
    t = time.time()
    a = m.ask("I was charged twice and my parcel never arrived. Please refund one charge.", QUESTIONS)
    print(f"[{time.time() - t:.2f}s]", round(a["refund"]["noul"], 2), a["team"]["choice"], a["topics"]["choices"],
          a["urgency"]["legend"][str(int(a["urgency"]["score"]))], "   expected: 1.0 billing ['billing', 'delivery'] medium")
    for tk, exp in [("Where is my parcel? It was due Monday.", "shipping"), ("The app crashes when I open settings.", "technical")]:
        print(m.ask(tk, {"team": QUESTIONS["team"]})["team"]["choice"], "  expected:", exp)
    print(json.dumps(a, indent=1, default=float)[:1500])
