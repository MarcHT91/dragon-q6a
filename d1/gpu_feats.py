"""Picture features of d1's vision encoder on the Adreno (ncnn) for the eval set of d1_calib.py: each picture with the
exported shape closest to its aspect ratio -> EVAL_DIR/feat_k.gpu.npy (d1_validate.py --feat-suffix ,.gpu).
usage: python gpu_feats.py EVAL_DIR ENC_DIR      (ENC_DIR: vlenc_HxW.ncnn.{param,bin} exported by vl/encoder/vl_enc.py)"""
import glob, json, math, os, re, sys, time
from pathlib import Path
import numpy as np, ncnn
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vl" / "encoder"))
import vl_enc as V  # noqa: E402

E, ENC = sys.argv[1], sys.argv[2]
shapes = sorted({tuple(map(int, re.findall(r"vlenc_(\d+)x(\d+)", p)[0])) for p in glob.glob(os.path.join(ENC, "vlenc_*x*.ncnn.bin"))})
nets = {}


def net(H, W):
    if (H, W) not in nets:
        n = ncnn.Net(); n.opt.use_vulkan_compute = True; n.opt.use_subgroup_ops = False; n.opt.lightmode = False
        n.load_param(os.path.join(ENC, f"vlenc_{H}x{W}.ncnn.param")); n.load_model(os.path.join(ENC, f"vlenc_{H}x{W}.ncnn.bin"))
        nets[(H, W)] = n
    return nets[(H, W)]


for r in json.load(open(os.path.join(E, "eval.json"))):
    if "image" not in r:
        continue
    p = os.path.join(E, r["image"]); im = Image.open(p)
    H, W = min(shapes, key=lambda s: abs(math.log(s[1] / s[0]) - math.log(im.width / im.height)))
    img = np.ascontiguousarray(V.preprocess(p, (H, W)).numpy())
    t = time.time(); ex = net(H, W).create_extractor(); ex.input("in0", ncnn.Mat(img))
    f = np.array(ex.extract("out0")[1]).reshape(2048, -1).T.copy()
    np.save(os.path.join(E, r["feat"].replace(".npy", ".gpu.npy")), f)
    print(f"{r['image']} {im.width}x{im.height} -> {W}x{H}: {f.shape[0]} tokens, {(time.time() - t) * 1e3:.0f} ms", flush=True)
