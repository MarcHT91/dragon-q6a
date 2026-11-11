"""Generic QDQ ONNX primitives for HTP v68 (QCS6490) graphs: uint16 activations, int4/int8 per-channel weights.

- G: small graph builder (inputs, QDQ ops, packed int4/int8 weights, u16 constants)
- sym_u16: symmetric uint16 activation quantization params from an abs-max
- quant_w_int4_pc: symmetric per-output-channel weight quantization with an MSE clip search
- quant_rows_lowmem: per-row RTN for very large matrices (e.g. a 128k-row LM head)
- lp_rmsnorm: RMSNorm as LpNormalization * gamma*sqrt(N) (HTP v68 has no RMSNorm op)
MASK: magnitude of the additive attention mask.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

MASK = 100.0
U16 = TP.UINT16


def sym_u16(absmax, margin=1.25):
    s = max(float(absmax), 1e-6) * margin / 32767.0
    return s, 32768


def quant_w_int4_pc(W, bits=4, clip_search=True):
    """W [out,in] float -> (Bq int8 [in,out], scale [out]) symmetric per-output-channel."""
    qmax = 2 ** (bits - 1) - 1
    Wt = W.T.astype(np.float32)                       # [in,out]
    amax = np.abs(Wt).max(0) + 1e-12
    best_s = amax / qmax
    if clip_search:
        best_e = np.full(Wt.shape[1], np.inf, np.float32)
        for r in np.linspace(0.6, 1.0, 17):
            s = amax * r / qmax
            q = np.clip(np.round(Wt / s), -qmax - 1, qmax)
            e = ((q * s - Wt) ** 2).sum(0)
            better = e < best_e
            best_e[better], best_s[better] = e[better], s[better]
    q = np.clip(np.round(Wt / best_s), -qmax - 1, qmax).astype(np.int8)
    return q, best_s.astype(np.float32)


def quant_rows_lowmem(E, bits=8, chunk=8192):
    """per-row symmetric RTN of a big [rows, in] matrix -> (q [in,rows] int8, s [rows]); chunked, no clip search."""
    qmax = 2 ** (bits - 1) - 1
    s = (np.abs(E).max(1) / qmax).astype(np.float32) + 1e-12
    q = np.empty((E.shape[1], E.shape[0]), np.int8)
    for a in range(0, E.shape[0], chunk):
        b = min(a + chunk, E.shape[0])
        q[:, a:b] = np.clip(np.round(E[a:b] / s[a:b, None]), -qmax - 1, qmax).astype(np.int8).T
    return q, s


class G:
    def __init__(self, name):
        self.name, self.nodes, self.inits, self.inputs, self.outputs = name, [], [], [], []
        self.n = 0
        self.vi = []
        self.taps = {}

    def uid(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, name, arr, dt):
        """Initializer from raw bytes (fast, no python lists). INT4 is packed two per byte, low nibble first."""
        a = np.asarray(arr)
        t = onnx.TensorProto(name=name, data_type=dt, dims=list(a.shape))
        if dt == TP.INT4:
            f = (a.ravel().astype(np.int8) & 0x0F).astype(np.uint8)
            if f.size % 2:
                f = np.append(f, np.uint8(0))
            t.raw_data = (f[0::2] | (f[1::2] << 4)).tobytes()
        else:
            npdt = {TP.FLOAT: np.float32, TP.INT64: np.int64, TP.INT8: np.int8, TP.UINT8: np.uint8, TP.UINT16: np.uint16}[dt]
            t.raw_data = np.ascontiguousarray(a, npdt).tobytes()
        self.inits.append(t)
        return name

    def qparams(self, base, s, zp, dt=U16):
        sn = self.const(self.uid(base + "_s"), np.float32(s), TP.FLOAT)
        zn = self.const(self.uid(base + "_z"), np.array(zp, np.uint16 if dt == U16 else np.uint8), dt)
        return sn, zn

    def dq(self, qname, s, zp, dt=U16):
        sn, zn = self.qparams(qname, s, zp, dt)
        out = self.uid(qname + "_dq")
        self.nodes.append(helper.make_node("DequantizeLinear", [qname, sn, zn], [out]))
        return out

    def q(self, fname, s, zp, out=None, dt=U16):
        sn, zn = self.qparams(fname, s, zp, dt)
        out = out or self.uid(fname + "_q")
        self.nodes.append(helper.make_node("QuantizeLinear", [fname, sn, zn], [out]))
        return out

    def op(self, typ, ins, qp, name, out_q=None, **attrs):
        """Float op on dequantized inputs, then Q (and DQ unless it's a graph output)."""
        f = self.uid(name)
        self.nodes.append(helper.make_node(typ, ins, [f], **attrs))
        s, zp = qp
        qn = self.q(f, s, zp, out=out_q)
        self.taps[name] = (qn, s, zp)
        if out_q:
            return qn
        return self.dq(qn, s, zp)

    def inp(self, name, shape, qp):
        self.inputs.append(helper.make_tensor_value_info(name, U16, shape))
        return self.dq(name, *qp)

    def out(self, name, shape):
        self.outputs.append(helper.make_tensor_value_info(name, U16, shape))

    def weight_int4(self, name, W, bits=4, pre=None):
        q, s = pre if pre is not None else quant_w_int4_pc(W, bits)
        # dead channels (scale ~1e-8) overflow the HTP requantizer -> clamp scale, zero their weights
        q, s = np.array(q), np.array(s, np.float32)
        dead = s < s.max() * 1e-4
        if dead.any():
            s[dead] = s.max() * 1e-4
            q[:, dead] = 0
        dt = TP.INT4 if bits == 4 else TP.INT8
        wn = self.const(name + "_wq", q, dt)
        sn = self.const(name + "_ws", s, TP.FLOAT)
        zn = self.const(name + "_wz", np.zeros(s.shape, np.int8), dt)
        out = name + "_w"
        self.nodes.append(helper.make_node("DequantizeLinear", [wn, sn, zn], [out], axis=1))
        return out

    def weight_u16(self, name, w):
        w = np.asarray(w, np.float32)
        lo, hi = min(0.0, float(w.min())), max(0.0, float(w.max()))
        s = (hi - lo) / 65535.0 or 1e-6
        zp = int(round(-lo / s))
        q = np.clip(np.round(w / s) + zp, 0, 65535).astype(np.uint16)
        wn = self.const(name + "_gq", q, U16)
        return self.dq(wn, s, zp)

    def shape_const(self, shape):
        return self.const(self.uid("shape"), np.array(shape, np.int64), TP.INT64)

    def model(self):
        g = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs, initializer=self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])
        m.ir_version = 10
        return m


def lp_rmsnorm(g, x, gamma, qp, name):
    """RMSNorm(x)*gamma == LpNorm_2(x) * (gamma*sqrt(N)) (eps negligible). HTP v68 has no RmsNorm."""
    n = len(gamma)
    u = g.op("LpNormalization", [x], (1.0 / 32767.0, 32768), name + "_lp", axis=-1, p=2)
    gd = g.weight_u16(name + "_g", np.asarray(gamma, np.float32) * math.sqrt(n))
    return g.op("Mul", [u, gd], qp, name)
