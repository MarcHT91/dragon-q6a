import os, time, platform
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as oq

_DEVS = None
# On a PC without the NPU (x86-64, e.g. Windows + onnxruntime-qnn 2.3.0 + onnxruntime 1.27.0) npu_session only compiles:
# it writes the HTP v68 context (ctx_path) offline via HtpPrepare and returns (None, seconds). NO_COMPILE=1 skips compiling.
OFFLINE = os.environ.get("QNN_OFFLINE", "") == "1" or platform.machine().lower() in ("amd64", "x86_64")
_SESSIONS = []          # weakrefs to every NPU session, released at exit (see _release_all)


def _release_all():
    """Drop the native QNN sessions before the interpreter exits. On kernel 7.2 fastrpc often fails to unmap DSP
    buffers around process start/exit ("unmmap ... ERROR" in dmesg); after several such leaks the CDSP hangs
    (glink "intent request timed out") until it is restarted. Releasing sessions reduces this but does not always
    prevent it, so keep the model in one long-lived process (see docs/TROUBLESHOOTING.md)."""
    import gc
    for r in _SESSIONS:
        s = r()
        if s is not None and getattr(s, "_sess", None) is not None:
            s._sess = None
    gc.collect()


import atexit, weakref
atexit.register(_release_all)


def qnn_devices():
    global _DEVS
    if _DEVS is None:
        os.environ["ADSP_LIBRARY_PATH"] = f"{Path(oq.get_library_path()).parent};/usr/lib/dsp/cdsp;/dsp"
        ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
        _DEVS = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", None) == oq.get_ep_name()]
    return _DEVS


def npu_session(model_path, ctx_path=None, perf="burst", opt_mode="3"):
    """Load a compiled EP context if present, else compile model_path (and save context)."""
    o = ort.SessionOptions()
    o.log_severity_level = int(os.environ.get("ORT_LOG", 3))
    o.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    po = {"backend_path": str(oq.get_qnn_htp_path()), "htp_performance_mode": perf,
          "htp_arch": "68", "htp_graph_finalization_optimization_mode": opt_mode}
    src = model_path
    if ctx_path and Path(ctx_path).exists():
        src = ctx_path
    elif ctx_path:
        o.add_session_config_entry("ep.context_enable", "1")
        o.add_session_config_entry("ep.context_embed_mode", "0")
        o.add_session_config_entry("ep.context_file_path", str(ctx_path))
    if OFFLINE or os.environ.get("NO_COMPILE") == "1":
        if os.environ.get("NO_COMPILE") == "1" or src == ctx_path:
            return None, 0.0
        # QCS6490: QNN soc_model 35, VTCM 2 MB. Other values compile but the board refuses the context (CREATE_FROM_BINARY).
        po["soc_model"] = os.environ.get("SOC_MODEL", "35"); po["vtcm_mb"] = os.environ.get("VTCM_MB", "2")
        o.add_provider_for_devices(qnn_devices(), po)
        t = time.time()
        try:
            ort.InferenceSession(str(src), o)
        except Exception as e:                 # expected without an HTP device; the context is written first
            if not Path(ctx_path).exists():
                raise
        if not Path(ctx_path).exists():
            raise RuntimeError(f"offline compile wrote no context for {model_path}")
        return None, time.time() - t
    o.add_provider_for_devices(qnn_devices(), po)
    t = time.time()
    s = ort.InferenceSession(str(src), o)
    _SESSIONS.append(weakref.ref(s))
    return s, time.time() - t


def cpu_session(model_path):
    o = ort.SessionOptions(); o.log_severity_level = 3
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(model_path), o, providers=["CPUExecutionProvider"])


def q16(x, s):
    return np.clip(np.round(np.asarray(x, np.float32) / s) + 32768, 0, 65535).astype(np.uint16)


def dq16(q, s, zp=32768):
    return (q.astype(np.float32) - zp) * s


def model_qparams(m):
    """name of graph I/O -> (scale, zp) read from the DQ/Q nodes touching them."""
    init = {t.name: t for t in m.graph.initializer}
    from onnx import numpy_helper
    val = lambda n: numpy_helper.to_array(init[n]).item()
    out = {}
    io = {i.name for i in m.graph.input} | {o.name for o in m.graph.output}
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in io:
            out[n.input[0]] = (val(n.input[1]), val(n.input[2]))
        if n.op_type == "QuantizeLinear" and n.output[0] in io:
            out[n.output[0]] = (val(n.input[1]), val(n.input[2]))
    return out


def rope_inputs(pos, D, theta):
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float64) / D))
    f = pos * inv
    emb = np.concatenate([f, f])
    sign = np.concatenate([-np.ones(D // 2), np.ones(D // 2)])
    return np.cos(emb)[None].astype(np.float32), (np.sin(emb) * sign)[None].astype(np.float32)


def split16(q):
    """uint16 (zp 32768) -> (hi, lo) uint8, signed bytes with zp 128: v = hi*256 + lo."""
    v = np.clip(q.astype(np.int32) - 32768, -32768, 32639)
    lo = ((v + 128) & 255) - 128
    hi = (v - lo) >> 8
    return (hi + 128).astype(np.uint8), (lo + 128).astype(np.uint8)
