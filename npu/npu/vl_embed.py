"""LFM3-VL: input embeddings with image features in place of the <|image|> tokens, and calibration-file loading."""
import json
from pathlib import Path
import numpy as np, torch

IMG, VSTART, VEND = 125030, 125009, 125010   # <|image|>, <|vision_start|>, <|vision_end|>


def embed(ref, ids, feat=None):
    """ref.embed rows for ids; the k-th <|image|> position takes feat[k] (feat [N, hidden], N >= number of positions)."""
    t = torch.as_tensor(ids, device=ref.dev)
    x = ref.embed[t].clone()
    m = t == IMG
    if feat is not None and int(m.sum()):
        f = torch.as_tensor(np.asarray(feat), dtype=x.dtype, device=x.device)
        x[m] = f[: int(m.sum())]
    return x


def load_jsonl(path, max_len=None):
    """[(ids, feat or None)] from a calibration jsonl; "img" names a .npy next to the jsonl."""
    out = []
    for line in open(path):
        d = json.loads(line)
        ids = d["ids"][:max_len] if max_len else d["ids"]
        feat = np.load(Path(path).parent / d["img"]) if d.get("img") else None
        out.append((ids, feat))
    return out
