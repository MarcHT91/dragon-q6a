"""LFM3-VL eval set: held-out COCO images (not in gen_calib_vl.py's sample), fp32 features and fp32 answers, and the
fp32 logits over the question + answer positions, for validate_vl.py. Run on the PC GPU.

usage: python make_eval_vl.py MODEL_DIR COCO_PARQUET OUT_DIR [n_images=8] [answer_tokens=64] [calib_images=128]
Writes OUT_DIR/feat_{k}.npy, img_{k}.jpg, eval.jsonl ({"ids", "img", "start"}) and ref_logits.npz (fp16, arr_k =
logits at positions start..len(ids)-1).
"""
import io, json, os, random, sys
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "vl")); sys.path.insert(1, str(HERE.parent.parent / "vl" / "encoder"))   # PC workspace, or the repo
from lfm3_ref import LFM3Ref, rms
from vl_embed import IMG, VSTART, VEND, embed
import vl_enc as V
from tokenizers import Tokenizer

torch.set_grad_enabled(False)
M, PQ, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
NI = int(sys.argv[4]) if len(sys.argv) > 4 else 8
NA = int(sys.argv[5]) if len(sys.argv) > 5 else 64
NCAL = int(sys.argv[6]) if len(sys.argv) > 6 else 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Q = ["Describe this image in detail.", "What is in this picture?", "How many people are there? Explain.",
     "Descriu aquesta imatge.", "What colors dominate this image?", "Write a caption for this photo.",
     "Is this indoors or outdoors? Why?", "List the main objects."]
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
STOP = {tok.token_to_id("<|im_end|>"), 7}
W = V.load_weights(os.environ.get("VL_WEIGHTS", str(HERE.parent / "vl" / "vision.safetensors")))
ref = LFM3Ref(M, device=DEV); ref.cache_weights()
rows = pq.read_table(PQ, columns=["image"])["image"].to_pylist()
calib = set(random.Random(0).sample(range(len(rows)), NCAL))          # same draw as gen_calib_vl.py
pick = [i for i in random.Random(1).sample(range(len(rows)), NI + NCAL) if i not in calib][:NI]


def run(ids, feat):
    x = embed(ref, ids, feat)
    for i in range(ref.L):
        x, _ = ref.layer_forward(i, x, list(range(len(ids))), None)
    return rms(x, ref.fnorm, ref.eps) @ ref.embed.T


logits = []
with open(OUT / "eval.jsonl", "w") as fo:
    for k, r in enumerate(pick):
        img = Image.open(io.BytesIO(rows[r]["bytes"])).convert("RGB")
        p = OUT / f"img_{k}.jpg"; img.save(p, quality=92)
        s = [256, 384, 512][k % 3]
        feat = V.reference(W, V.preprocess(str(p), (s, s)))[1].numpy().astype(np.float32)
        np.save(OUT / f"feat_{k}.npy", feat)
        q = Q[k % len(Q)]
        ids = ([tok.token_to_id("<|startoftext|>")] + enc("<|im_start|>user\n") + [VSTART] + [IMG] * len(feat) + [VEND])
        start = len(ids)
        ids += enc(q) + enc("<|im_end|>\n<|im_start|>assistant\n<think></think>")
        for _ in range(NA):                      # greedy fp32 answer (full recompute: short and simple)
            t = int(run(ids, feat)[-1].argmax())
            ids.append(t)
            if t in STOP:
                break
        lg = run(ids, feat)[start:].float().cpu().numpy().astype(np.float16)
        logits.append(lg)
        fo.write(json.dumps({"ids": ids, "img": f"feat_{k}.npy", "start": start, "size": s}) + "\n")
        print(k, s, q, "->", repr(tok.decode(ids[start:])[:160]), flush=True)
np.savez_compressed(OUT / "ref_logits.npz", *logits)
print("wrote", OUT)
