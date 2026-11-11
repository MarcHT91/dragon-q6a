"""fp32 reference logits for the teacher-forced eval texts -> build/eval/{eval_ids.json, ref_logits.npz}.
Used by validate_build.py and quant_sim_eval.py. Needs ~7 GB (GPU: REF_DEVICE=cuda).
usage: make_eval_ref.py [OUT_DIR]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import MODEL_DIR, ROOT
from eval_texts import TEXTS
from lfm3_ref import LFM3Ref
from tokenizers import Tokenizer
torch.set_grad_enabled(False)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "eval"; OUT.mkdir(parents=True, exist_ok=True)
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
IDS = [tok.encode(t, add_special_tokens=False).ids for t in TEXTS]
r = LFM3Ref(str(MODEL_DIR), device=os.environ.get("REF_DEVICE", "cpu")); r.cache_weights()
json.dump(IDS, open(OUT / "eval_ids.json", "w"))
np.savez_compressed(OUT / "ref_logits.npz", *[r.forward(ids)[0].float().cpu().numpy().astype(np.float16) for ids in IDS])
print("wrote", OUT, [len(i) for i in IDS])
