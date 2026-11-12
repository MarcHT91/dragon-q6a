# Microbenchmark: a Linear layer written as a 1x1 Convolution (input [k, sqrt(n), sqrt(n)]), the fast path on Turnip.
# usage: tiny3.py n k m   env: FP16, GPU, OFF as in tiny2.py
import sys, os, time, numpy as np, ncnn
n, k, m = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
fp16 = os.environ.get("FP16", "1") == "1"; gpu = os.environ.get("GPU", "1") == "1"
g = int(n ** 0.5)
w = (np.random.randn(m*k).astype(np.float32)*0.02)
open("t3.param","w").write(f"7767517\n2 2\nInput in0 0 1 in0\nConvolution L 1 1 in0 out0 0={m} 1=1 5=0 6={m*k}\n")
open("t3.bin","wb").write(np.uint32(0).tobytes() + w.tobytes())
net = ncnn.Net(); net.opt.use_vulkan_compute = gpu
net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
for f in os.environ.get("OFF","").split(","):
    f and setattr(net.opt, f, False)
net.load_param("t3.param"); net.load_model("t3.bin")
x = np.random.randn(k, g, g).astype(np.float32)
ts=[]
for i in range(4):
    ex = net.create_extractor(); ex.input("in0", ncnn.Mat(x)); t=time.time(); _, o = ex.extract("out0"); ts.append(time.time()-t)
o = np.array(o).reshape(m, n); ref = w.reshape(m, k) @ x.reshape(k, n)
print("conv1x1", n, k, m, "gpu" if gpu else "cpu", "fp16" if fp16 else "fp32", f"{min(ts[1:])*1000:.1f} ms", f"{2*n*k*m/min(ts[1:])/1e9:.0f} GFLOPS", "relerr", f"{np.linalg.norm(o-ref)/np.linalg.norm(ref):.4f}")
