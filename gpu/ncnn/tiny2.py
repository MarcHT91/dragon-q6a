# Microbenchmark: one ncnn layer, Gemm ("gemm") or InnerProduct ("ip"), C = A[n,k] x W[m,k]^T on the GPU (Vulkan).
# usage: tiny2.py gemm|ip n k m   env: FP16=0|1, GPU=0|1, OFF=opt1,opt2 (ncnn options to switch off, e.g. use_subgroup_ops)
import sys, os, time, numpy as np, ncnn
layer, n, k, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fp16 = os.environ.get("FP16", "1") == "1"; gpu = os.environ.get("GPU", "1") == "1"
w = (np.random.randn(m*k).astype(np.float32)*0.02)
if layer == "ip":
    p = f"InnerProduct L 1 1 in0 out0 0={m} 1=0 2={m*k}"; blob = np.uint32(0).tobytes() + w.tobytes()
else:  # Gemm as pnnx emits it (constant B, transB)
    p = f"Gemm L 1 1 in0 out0 10=-1 2=0 3=1 4=0 5=1 6=1 7={n} 8={m} 9={k}"; blob = np.uint32(0).tobytes() + w.tobytes()
open("t2.param","w").write(f"7767517\n2 2\nInput in0 0 1 in0\n{p}\n")
open("t2.bin","wb").write(blob)
net = ncnn.Net(); net.opt.use_vulkan_compute = gpu
net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
for f in os.environ.get("OFF","").split(","):
    f and setattr(net.opt, f, False)
net.load_param("t2.param"); net.load_model("t2.bin")
x = np.random.randn(n, k).astype(np.float32)
ts=[]
for i in range(4):
    ex = net.create_extractor(); ex.input("in0", ncnn.Mat(x)); t=time.time(); _, o = ex.extract("out0"); ts.append(time.time()-t)
o = np.array(o); ref = x @ w.reshape(m, k).T
print(layer, n, k, m, "gpu" if gpu else "cpu", "fp16" if fp16 else "fp32", f"{min(ts[1:])*1000:.1f} ms", f"{2*n*k*m/min(ts[1:])/1e9:.0f} GFLOPS", "relerr", f"{np.linalg.norm(o-ref)/np.linalg.norm(ref):.4f}", o.shape)
