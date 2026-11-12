"""Time an exported ncnn model on the Adreno (Vulkan fp16) and check it against the torch fp32 output.
Usage: python bench_ncnn.py WHICH GRID [runs]"""
import sys, time, numpy as np, torch, ncnn

which, g = sys.argv[1], int(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
io = torch.load(f"io_{which}_{g}.pt", weights_only=True)
x, y = io["x"][0].numpy(), io["y"][0].numpy()

net = ncnn.Net()
net.opt.use_vulkan_compute = True
import os
fp16 = os.environ.get("FP16", "1") == "1"
net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
net.opt.use_subgroup_ops = False  # Turnip hang (VK_ERROR_DEVICE_LOST) in Gemm otherwise
net.set_vulkan_device(0)
net.load_param(f"enc_{which}_{g}.ncnn.param"); net.load_model(f"enc_{which}_{g}.ncnn.bin")


def run():
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    _, out = ex.extract("out0")
    return np.array(out)


t0 = time.time(); out = run(); first = time.time() - t0
ts = []
for _ in range(runs):
    t0 = time.time(); out = run(); ts.append(time.time() - t0)
rel = np.linalg.norm(out - y) / np.linalg.norm(y)
cos = float((out * y).sum() / np.linalg.norm(out) / np.linalg.norm(y))
print(f"{which} grid={g} tokens={g*g}: first {first*1000:.0f} ms, steady median {np.median(ts)*1000:.0f} ms "
      f"(min {min(ts)*1000:.0f}); rel err {rel:.4f}, cos {cos:.5f}")
