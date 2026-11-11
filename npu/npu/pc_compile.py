"""Offline QNN HTP (v68) context-binary compile on a PC without the NPU (onnxruntime-qnn x64, HtpPrepare).
usage: pc_compile.py MODEL.onnx [MODEL.onnx ...]   -> writes MODEL_ctx.onnx + MODEL_ctx_qnn.bin next to each model
The context files load on the board with npu_common.npu_session(model, ctx) like board-compiled ones."""
import sys, time, os
from pathlib import Path
import onnxruntime as ort, onnxruntime_qnn as oq

ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
devs = [d for d in ort.get_ep_devices() if d.ep_name == oq.get_ep_name()]
for mp in map(Path, sys.argv[1:]):
    ctx = mp.with_name(mp.stem + "_ctx.onnx")
    for f in mp.parent.glob(ctx.stem + "*"):
        f.unlink()
    o = ort.SessionOptions()
    o.log_severity_level = int(os.environ.get("ORT_LOG", 3))
    o.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    o.add_session_config_entry("ep.context_enable", "1")
    o.add_session_config_entry("ep.context_embed_mode", "0")
    o.add_session_config_entry("ep.context_file_path", str(ctx))
    po = {"backend_path": str(oq.get_qnn_htp_path()), "htp_arch": "68", "soc_model": os.environ.get("SOC_MODEL", "35"), "vtcm_mb": os.environ.get("VTCM_MB", "2"),
          "htp_graph_finalization_optimization_mode": "3"}
    o.add_provider_for_devices(devs, po)
    t = time.time()
    try:
        ort.InferenceSession(str(mp), o)
    except Exception as e:                 # no HTP device on the PC: the context may still have been written
        print(f"  (session init: {str(e)[:200]})")
    ok = ctx.exists()
    print(f"{mp.name}: {'OK' if ok else 'NO CONTEXT'} {time.time()-t:.0f}s -> {[f.name for f in mp.parent.glob(ctx.stem + '*')]}", flush=True)
