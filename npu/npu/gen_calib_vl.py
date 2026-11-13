"""Image calibration data for LFM3-VL: COCO photos -> fp32 vision features (vl/vl_enc.py reference) -> the fp32
language model answers a question about each image. Run on the PC GPU.

usage: python gen_calib_vl.py MODEL_DIR COCO_PARQUET OUT_DIR [n_images=96] [max_new=160]
Writes OUT_DIR/feat_{k}.npy (float32 [N, 2048]) and OUT_DIR/calib_vl.jsonl with {"ids": [...], "img": "feat_k.npy"}:
the <|image|> positions of ids take the rows of the feature file, in order (see vl_embed.py).
"""
import io, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "vl")); sys.path.insert(1, str(HERE.parent.parent / "vl" / "encoder"))   # PC workspace, or the repo
from lfm3_ref import LFM3Ref, rms
from vl_embed import IMG, VSTART, VEND, embed
import vl_enc as V
from tokenizers import Tokenizer

M, PQ, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
NI = int(sys.argv[4]) if len(sys.argv) > 4 else 96
NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 160
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZES = [256, 384, 512]
QUESTIONS = [
    "Describe this image in detail.", "What is happening in this picture?", "Describe this image in one sentence.",
    "How many people are in the image?", "What colors stand out in this image?", "Is there any text in the image? What does it say?",
    "List the objects you can see.", "Where was this photo probably taken?", "What is the main subject of this photo?",
    "Describe the background of the image.", "Is this photo taken indoors or outdoors?", "What time of day does it look like?",
    "Descriu aquesta imatge en català.", "Què hi ha en aquesta foto?", "Describe esta imagen en español.",
    "¿Cuántos objetos hay en la imagen?", "Write a short caption for this image.", "Answer in JSON with keys \"objects\" and \"scene\".",
    "Is there an animal in this picture? Answer yes or no and explain.", "What would be a good title for this photo?",
]

tok = Tokenizer.from_file(f"{M}/tokenizer.json")
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
STOP = {tok.token_to_id("<|im_end|>"), 7}
W = V.load_weights(os.environ.get("VL_WEIGHTS", str(HERE.parent / "vl" / "vision.safetensors")))
ref = LFM3Ref(M, device=DEV); ref.cache_weights()
rows = pq.read_table(PQ, columns=["image"])["image"].to_pylist()
rng = random.Random(0); pick = rng.sample(range(len(rows)), NI)
t0 = time.time()
with open(OUT / "calib_vl.jsonl", "w") as fo:
    for k, r in enumerate(pick):
        img = Image.open(io.BytesIO(rows[r]["bytes"])).convert("RGB")
        s = SIZES[k % len(SIZES)]
        p = OUT / f"img_{k}.jpg"; img.save(p, quality=92)
        feat = V.reference(W, V.preprocess(str(p), (s, s)))[1].numpy().astype(np.float32)
        np.save(OUT / f"feat_{k}.npy", feat)
        q = QUESTIONS[k % len(QUESTIONS)]
        ids = ([tok.token_to_id("<|startoftext|>")] + enc("<|im_start|>user\n") + [VSTART] + [IMG] * len(feat) + [VEND]
               + enc(q) + enc("<|im_end|>\n<|im_start|>assistant\n<think></think>"))
        x = embed(ref, ids, feat)
        past = []
        for i in range(ref.L):
            x, st = ref.layer_forward(i, x, list(range(len(ids))), None); past.append(st)
        logits = rms(x, ref.fnorm, ref.eps) @ ref.embed.T
        out, pos = [], len(ids)
        for _ in range(NEW):
            t = int(logits[-1].argmax()); out.append(t)
            if t in STOP:
                break
            logits, past = ref.forward([t], start=pos, past=past); pos += 1
        fo.write(json.dumps({"ids": ids + out, "img": f"feat_{k}.npy"}) + "\n"); fo.flush()
        print(f"{k:3d} {s}px {time.time()-t0:5.0f}s | {q} -> {tok.decode(out)[:100]!r}", flush=True)
print("done")
