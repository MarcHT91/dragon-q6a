"""Run the synthetic LFM3-VL encoder (enc_cf.py) through ONNX Runtime + QNN GPU backend (needs KGSL + Qualcomm OpenCL).
Usage: source ../kgsl/adreno-env.sh; python qnn_gpu_test.py GRID WHICH(full|attn1|conv1) [half]
  (Python with torch, onnx, onnxruntime==1.27 and onnxruntime-qnn==2.3; "half" exports the model in fp16)"""
import os, sys, time, numpy as np, torch
sys.argv, args = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ncnn"))
import enc_cf
import onnxruntime as ort, onnxruntime_qnn as oq
ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())

g, which = int(args[0]), args[1]
fp16 = "fp16" in args
layers = {"full": enc_cf.LAYERS, "attn1": ["attn"], "conv1": ["conv"]}[which]
torch.manual_seed(0)
m = enc_cf.Enc(g, layers).eval()
for p in m.parameters():
    if p.dim() > 1:
        torch.nn.init.normal_(p, std=0.02)
x = torch.randn(1, enc_cf.D, g, g)
with torch.no_grad():
    y = m(x).numpy()
half = "half" in args
path = f"qnn_{which}_{g}{'_half' if half else ''}.onnx"
if not os.path.exists(path):
    me, xe = (m.half(), x.half()) if half else (m, x)
    torch.onnx.export(me, (xe,), path, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
opts = {"backend_path": str(oq.get_qnn_gpu_path())}
if fp16:
    opts["enable_htp_fp16_precision"] = "1"
t0 = time.time()
so.log_severity_level = int(os.environ.get("ORT_LOG", 3))
so.add_provider_for_devices([d for d in ort.get_ep_devices() if d.ep_name == oq.get_ep_name()], opts)
sess = ort.InferenceSession(path, so)
print(f"session created in {time.time()-t0:.1f} s, providers {sess.get_providers()}")
xi = x.half().numpy() if half else x.numpy()
ts = []
for i in range(4):
    t0 = time.time(); out = sess.run(None, {"x": xi})[0]; ts.append(time.time() - t0)
out = out.astype(np.float32)
rel = np.linalg.norm(out - y) / np.linalg.norm(y)
cos = float((out * y).sum() / np.linalg.norm(out) / np.linalg.norm(y))
print(f"QNN-GPU {which} grid={g}: first {ts[0]*1000:.0f} ms, steady min {min(ts[1:])*1000:.0f} ms; rel err {rel:.4f}, cos {cos:.5f}")
