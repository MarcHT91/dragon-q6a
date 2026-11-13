"""Image features with the ncnn encoder on the Adreno for every image of an eval dir (eval.jsonl: img_k.jpg, size)
-> feat_k.gpu.npy. usage: python gpu_feats.py EVAL_DIR"""
import json, os, sys, time, numpy as np, ncnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vl_enc as V
E = sys.argv[1]
nets = {}
for k, line in enumerate(open(os.path.join(E, "eval.jsonl"))):
    d = json.loads(line); s = d["size"]
    if s not in nets:
        n = ncnn.Net(); n.opt.use_vulkan_compute = True; n.opt.use_subgroup_ops = False; n.opt.lightmode = False
        n.load_param(f"vlenc_{s}x{s}.ncnn.param"); n.load_model(f"vlenc_{s}x{s}.ncnn.bin"); nets[s] = n
    img = np.ascontiguousarray(V.preprocess(os.path.join(E, f"img_{k}.jpg"), (s, s)).numpy())
    t0 = time.time(); ex = nets[s].create_extractor(); ex.input("in0", ncnn.Mat(img))
    f = np.array(ex.extract("out0")[1]).reshape(2048, -1).T.copy(); dt = time.time() - t0
    ref = np.load(os.path.join(E, d["img"]))
    cs = (f * ref).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(ref, axis=1)
    np.save(os.path.join(E, d["img"].replace(".npy", ".gpu.npy")), f)
    print(f"img {k} {s}px {dt*1000:.0f} ms (incl. first-run warmup) cos mean {cs.mean():.4f} min {cs.min():.4f}", flush=True)
