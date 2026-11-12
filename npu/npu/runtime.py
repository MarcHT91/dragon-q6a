"""Pick the NPU runtime: "native" (native/libqnnrun.so, faster, see lfm3_native.py) or "ort" (ONNX Runtime QNN EP).
Default: native when native/libqnnrun.so is built, otherwise ort. Override with LFM3_RUNTIME=native|ort."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def default_runtime():
    import qnnrun
    return os.environ.get("LFM3_RUNTIME") or ("native" if qnnrun.available() else "ort")


def load_model(build_dir, runtime=None, **kw):
    rt = runtime or default_runtime()
    if rt == "native":
        from lfm3_native import LFM3Native
        return LFM3Native(build_dir, **kw)
    if rt == "ort":
        from lfm3_npu import LFM3NPU
        return LFM3NPU(build_dir, **kw)
    raise ValueError(f"unknown runtime {rt!r} (native or ort)")
