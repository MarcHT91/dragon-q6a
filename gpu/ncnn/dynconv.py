# Microbenchmark: attention scores q.k^T as a grouped 1x1 conv with dynamic weights (correct but ~24 GFLOPS).
# scores[h] = q[h] (n x HD) . k[h]^T  as grouped 1x1 conv with dynamic weight: input k as [H*HD, g, g], weight q as [H*n, HD, 1, 1], group=H
import sys, os, time, numpy as np, ncnn
H, HD, g = int(sys.argv[1]), 64, int(sys.argv[2]); n = g*g
gpu = os.environ.get("GPU", "1") == "1"; fp16 = os.environ.get("FP16", "1") == "1"
grp = f"ConvolutionDepthWise L 2 1 k q out0 0={H*n} 1=1 5=0 6=0 7={H} 19=1" if H > 1 else f"Convolution L 2 1 k q out0 0={n} 1=1 5=0 6=0 19=1"
open("dc.param","w").write(f"7767517\n3 3\nInput k 0 1 k\nInput q 0 1 q\n{grp}\n")
open("dc.bin","wb").write(b"")
net = ncnn.Net(); net.opt.use_vulkan_compute = gpu; net.opt.use_subgroup_ops = False
net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
net.load_param("dc.param"); net.load_model("dc.bin")
k = np.random.randn(H, HD, n).astype(np.float32); q = np.random.randn(H, n, HD).astype(np.float32)
ts = []
for i in range(4):
    ex = net.create_extractor(); ex.input("k", ncnn.Mat(k.reshape(H*HD, g, g))); ex.input("q", ncnn.Mat(q.reshape(H*n, HD, 1, 1)))
    t = time.time(); r, o = ex.extract("out0"); ts.append(time.time() - t)
o = np.array(o).reshape(H, n, n); ref = q @ k
dt = min(ts[1:])
print(f"dynconv H={H} n={n} {'gpu' if gpu else 'cpu'}: {dt*1000:.1f} ms, {2*H*n*n*HD/dt/1e9:.0f} GFLOPS, relerr {np.linalg.norm(o-ref)/np.linalg.norm(ref):.4f}")
