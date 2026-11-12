"""Python wrapper for native/libqnnrun.so: runs QNN HTP context binaries with persistent rpcmem I/O buffers.

Every graph input/output is exposed as a numpy array that aliases its rpcmem buffer, so writing an input is just
an assignment and nothing is copied when the graph executes. Built with native/build.sh.
Only one QNN runtime per process: do not load ONNX Runtime QNN sessions and these graphs in the same process
(the DSP runs out of address space and the process can hang; see docs/TROUBLESHOOTING.md)."""
import atexit
import ctypes as C
import os
from pathlib import Path
import numpy as np
import onnxruntime_qnn as oq

QNN_DIR = Path(oq.__file__).resolve().parent            # libQnnHtp.so, libQnnSystem.so, v68 stub/skel (QNN 2.47)
LIB = Path(__file__).resolve().parent.parent / "native" / "libqnnrun.so"


def available():
    return LIB.exists()


_lib = None
_inited = False


def _load():
    global _lib
    if _lib is not None:
        return _lib
    os.environ.setdefault("ADSP_LIBRARY_PATH", str(QNN_DIR))          # where the DSP side finds the v68 skel
    C.CDLL(str(QNN_DIR / "libQnnHtpV68Stub.so"), mode=C.RTLD_GLOBAL)   # dlopen'ed by name from libQnnHtp.so
    L = C.CDLL(str(LIB))
    L.qr_init.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
    L.qr_load.argtypes = [C.c_char_p]
    L.qr_num.argtypes = [C.c_int, C.c_int]
    L.qr_info.argtypes = [C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                          C.POINTER(C.c_int), C.POINTER(C.c_float), C.POINTER(C.c_int), C.POINTER(C.c_void_p),
                          C.POINTER(C.c_size_t)]
    L.qr_execute.argtypes = [C.c_int]
    L.qr_share_input.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int]
    L.lf_set_model.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                               C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
    L.lf_add_graph.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_int)]
    L.lf_step.argtypes = [C.c_int, C.c_int]
    L.lf_step.restype = C.c_void_p
    _lib = L
    return L


def lib():
    return _load()


def _shutdown():
    if _inited:
        _lib.qr_shutdown()          # free graphs and rpcmem buffers before the interpreter exits


# QNN data types -> numpy
_DT = {0x0408: np.uint8, 0x0416: np.uint16, 0x0108: np.uint8, 0x0116: np.uint16, 0x0132: np.uint32,
       0x0008: np.int8, 0x0016: np.int16, 0x0032: np.int32, 0x0232: np.float32, 0x0216: np.float16}


class Tensor:
    def __init__(self, name, idx, arr, scale, offset):
        self.name, self.idx, self.a, self.scale, self.offset = name, idx, arr, scale, offset


class Graph:
    """One context binary (one graph). .inputs / .outputs: name -> Tensor (numpy view .a, quant .scale/.offset)."""

    def __init__(self, ctx_bin):
        global _inited
        L = _load()
        if not _inited:
            if L.qr_init(str(QNN_DIR / "libQnnHtp.so").encode(), str(QNN_DIR / "libQnnSystem.so").encode(), 1):
                raise RuntimeError("qr_init failed")
            _inited = True
            atexit.register(_shutdown)
        self.g = L.qr_load(str(ctx_bin).encode())
        if self.g < 0:
            raise RuntimeError(f"qr_load {ctx_bin} -> {self.g}")
        self.inputs, self.outputs = {}, {}
        for is_out, d in ((0, self.inputs), (1, self.outputs)):
            for i in range(L.qr_num(self.g, is_out)):
                name = C.create_string_buffer(256); dims = (C.c_uint32 * 8)(); rank = C.c_uint32(); dt = C.c_int()
                sc = C.c_float(); off = C.c_int(); ptr = C.c_void_p(); nb = C.c_size_t()
                L.qr_info(self.g, is_out, i, name, 256, dims, C.byref(rank), C.byref(dt), C.byref(sc), C.byref(off),
                          C.byref(ptr), C.byref(nb))
                npdt = _DT[dt.value]
                shape = tuple(dims[k] for k in range(rank.value))
                arr = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(np.ctypeslib.as_ctypes_type(npdt))), shape=shape)
                nm = name.value.decode()
                d[nm] = Tensor(nm, i, arr, sc.value, off.value)

    def run(self):
        if _lib.qr_execute(self.g):
            raise RuntimeError("graphExecute failed")

    def share_input(self, name, src_graph, src_name, src_is_out=False):
        """Make input `name` use the same rpcmem buffer as src_graph's input/output `src_name`."""
        src = (src_graph.outputs if src_is_out else src_graph.inputs)[src_name]
        dst = self.inputs[name]
        if _lib.qr_share_input(self.g, dst.idx, src_graph.g, int(src_is_out), src.idx):
            raise RuntimeError(f"share {name} failed")
        dst.a = src.a
