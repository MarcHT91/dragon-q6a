"""LFM3-VL-2B vision encoder + projector: an fp32 reference that follows the HuggingFace code
(tokens in merge-block order), and a channels-first version for ncnn/Vulkan export with the same weights.

Weights: $VL_WEIGHTS (default hf/vision.safetensors): the vision_tower + multi_modal_projector tensors of
LiquidAI/LFM3-VL-2B (or of d1-2B, which fine-tunes them slightly).
Usage:
  python vl_enc.py ref IMAGE OUT.pt        reference features for an image
  python vl_enc.py export H W              export the channels-first model for an HxW input (multiples of 32) with
                                           pnnx -> vlenc_HxW.ncnn.{param,bin} in the current directory
The export calibrates the fp16 rescaling on a few photos: $VL_CALIB (comma list) or calib/*.jpg (any 3-5 ordinary
photos; only the order of magnitude of the activations matters).
"""
import math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F

D, I, H, KV, HD, P, M = 1024, 2560, 16, 8, 64, 16, 2
LAYERS = ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv"]
EPS, PEPS, THETA = 1e-5, 1e-6, 1e6
WEIGHTS = os.environ.get("VL_WEIGHTS", "hf/vision.safetensors")   # VL_WEIGHTS: another checkpoint's vision tower (d1-2B)
# photos for EncCF.calibrate: only the order of magnitude of the activations matters
CALIB = os.environ["VL_CALIB"].split(",") if os.environ.get("VL_CALIB") else sorted(__import__("glob").glob("calib/*.jpg"))


def load_weights(path=WEIGHTS):
    from safetensors.torch import load_file
    return {k.replace("model.", "", 1): t.float() for k, t in load_file(path).items()}


# ---------------------------------------------------------------- preprocessing (image processor)
def smart_resize(h, w, factor=P * M, min_px=65536, max_px=16777216):
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_px:
        b = math.sqrt(h * w / max_px)
        hb, wb = max(factor, math.floor(h / b / factor) * factor), max(factor, math.floor(w / b / factor) * factor)
    elif hb * wb < min_px:
        b = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * b / factor) * factor, math.ceil(w * b / factor) * factor
    return hb, wb


def preprocess(path, size=None):
    """Image -> normalized [3, H, W] float tensor. size=(H, W) forces the resize target (multiples of 32)."""
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    h, w = size or smart_resize(img.height, img.width)
    img = img.resize((w, h), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - 0.5) / 0.5).contiguous()


def rope_tables(gh, gw):
    """cos/sin [gh*gw, HD] for raster positions (row, col): axial RoPE, layout [h16 w16 h16 w16]."""
    sd = HD // 2
    inv = 1.0 / (THETA ** (torch.arange(0, sd, 2, dtype=torch.float) / sd))
    hp, wp = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    fh, fw = hp.reshape(-1, 1).float() * inv, wp.reshape(-1, 1).float() * inv
    f = torch.cat([fh, fw, fh, fw], dim=-1)
    return f.cos(), f.sin()


def rot_half(x, dim):
    a, b = x.chunk(2, dim=dim)
    return torch.cat([-b, a], dim=dim)


# ---------------------------------------------------------------- reference (token layout, as in transformers)
def rms(x, w, eps=EPS):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


@torch.no_grad()
def reference(W, img):
    """img [3, H, W] -> (hidden [N, 1024] in merge-block order, features [N/4, 2048])."""
    _, Hh, Ww = img.shape
    gh, gw = Hh // P, Ww // P
    # patchify like the image processor: merge-block order, (channel, temporal, py, px) per patch
    p = img.reshape(3, gh // M, M, P, gw // M, M, P).permute(1, 4, 2, 5, 0, 3, 6)
    p = p.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(gh * gw, 3 * 2 * P * P)
    x = rms(p @ W["vision_tower.patchifier.weight"].T, W["vision_tower.embedding_norm.weight"])
    # positions in merge-block order
    hp, wp = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    order = (hp * gw + wp).reshape(gh // M, M, gw // M, M).transpose(1, 2).flatten()
    cos, sin = (t[order].to(x.device) for t in rope_tables(gh, gw))
    n = gh * gw
    for i, kind in enumerate(LAYERS):
        L = lambda s: W[f"vision_tower.layers.{i}.{s}"]
        h = rms(x, L("operator_norm.weight"))
        if kind == "attn":
            q = rms((h @ L("self_attn.q_proj.weight").T).view(n, H, HD), L("self_attn.q_layernorm.weight"))
            k = rms((h @ L("self_attn.k_proj.weight").T).view(n, KV, HD), L("self_attn.k_layernorm.weight"))
            v = (h @ L("self_attn.v_proj.weight").T).view(n, KV, HD)
            c, s = cos[:, None], sin[:, None]
            q, k = q * c + rot_half(q, -1) * s, k * c + rot_half(k, -1) * s
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            k, v = k.repeat_interleave(H // KV, 0), v.repeat_interleave(H // KV, 0)
            a = torch.softmax(q @ k.transpose(1, 2) * HD ** -0.5, -1) @ v
            o = a.transpose(0, 1).reshape(n, H * HD) @ L("self_attn.out_proj.weight").T
        else:
            B, C, xx = (h @ L("conv.in_proj.weight").T).chunk(3, -1)
            g = (B * xx).view(gh // M, gw // M, M, M, D).permute(4, 0, 2, 1, 3).reshape(1, D, gh, gw)
            g = F.conv2d(g, L("conv.spatial_conv.weight"), padding=1, groups=D)
            g = g.view(D, gh // M, M, gw // M, M).permute(1, 3, 2, 4, 0).reshape(n, D)
            o = (C * g) @ L("conv.out_proj.weight").T
        x = x + o
        h = rms(x, L("ffn_norm.weight"))
        x = x + (F.silu(h @ L("feed_forward.w1.weight").T) * (h @ L("feed_forward.w3.weight").T)) @ L("feed_forward.w2.weight").T
    pj = lambda s: W[f"multi_modal_projector.{s}"]
    f = F.layer_norm(x, (D,), pj("layer_norm.weight"), pj("layer_norm.bias"), PEPS).reshape(-1, 4 * D)
    f = F.gelu(f @ pj("linear_1.weight").T + pj("linear_1.bias")) @ pj("linear_2.weight").T + pj("linear_2.bias")
    return x, f


# ---------------------------------------------------------------- channels-first model for ncnn
def c1(i, o, bias=False):
    return nn.Conv2d(i, o, 1, bias=bias)


class Norm:
    """x / rms(x) over dim 1, computed as u / rms(u) with u = k*x and eps*k^2 (identical result). k is a power of two
    chosen by EncCF.calibrate so that rms(u) ~ 1: the encoder's activations are ~0.02, and x^2 and eps=1e-5 fall below
    the fp16 normal range on the GPU (NaN) without it."""
    def __init__(s, eps):
        s.eps, s.k, s.cal = eps, 1.0, None
        s.inscale = 1.0     # the input is inscale x the true activation (residual stream scaled by EncCF.S)

    def __call__(s, x):
        if s.cal is not None:
            s.cal.append(float(x.pow(2).mean(1).sqrt().median()))
        u = x * s.k if s.k != 1.0 else x
        return u * torch.rsqrt((u * u).mean(1, keepdim=True) + s.eps * (s.k * s.inscale) ** 2)


class RMSc(nn.Module):  # RMSNorm over dim 1
    def __init__(s, n, eps=EPS):
        super().__init__()
        s.w = nn.Parameter(torch.ones(1, n, 1, 1)); s.nrm = Norm(eps)

    def forward(s, x):
        return s.nrm(x) * s.w


class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.w1, s.w3, s.w2 = c1(D, I), c1(D, I), c1(I, D)

    def forward(s, x):
        return s.w2(F.silu(s.w1(x)) * s.w3(x))


class Attn(nn.Module):
    def __init__(s, gh, gw):
        super().__init__()
        s.n = gh * gw; s.gh, s.gw = gh, gw
        s.q, s.k, s.v, s.o = c1(D, H * HD), c1(D, KV * HD), c1(D, KV * HD), c1(H * HD, D)
        s.qn = nn.Parameter(torch.ones(1, HD, 1)); s.kn = nn.Parameter(torch.ones(1, HD, 1))
        s.nq, s.nk = Norm(EPS), Norm(EPS)
        cos, sin = rope_tables(gh, gw)
        s.register_buffer("cos", cos.T.reshape(1, HD, s.n).contiguous())
        s.register_buffer("sin", sin.T.reshape(1, HD, s.n).contiguous())

    def rope(s, x, w, nrm):  # x [heads, HD, n]: per-head RMSNorm over HD, then RoPE
        x = nrm(x) * w
        return x * s.cos + rot_half(x, 1) * s.sin

    def forward(s, x):
        n = s.n
        q = s.rope(s.q(x).reshape(H, HD, n), s.qn, s.nq).transpose(1, 2)              # [H, n, HD]
        k = s.rope(s.k(x).reshape(KV, HD, n), s.kn, s.nk)                              # [KV, HD, n]
        v = s.v(x).reshape(KV, HD, n)
        k = k.reshape(KV, 1, HD, n).expand(KV, H // KV, HD, n).reshape(H, HD, n)
        v = v.reshape(KV, 1, HD, n).expand(KV, H // KV, HD, n).reshape(H, HD, n)
        a = torch.softmax((q * HD ** -0.5) @ k, dim=-1) @ v.transpose(1, 2)       # [H, n, HD]
        return s.o(a.transpose(1, 2).reshape(1, H * HD, s.gh, s.gw))


class GConv(nn.Module):
    def __init__(s):
        super().__init__()
        s.b, s.c, s.h = c1(D, D), c1(D, D), c1(D, D)   # in_proj split in (B, C, x)
        s.dw = nn.Conv2d(D, D, 3, padding=1, groups=D, bias=False); s.out = c1(D, D)

    def forward(s, x):
        return s.out(s.c(x) * s.dw(s.b(x) * s.h(x)))


class Block(nn.Module):
    def __init__(s, kind, gh, gw):
        super().__init__()
        s.n1, s.n2 = RMSc(D), RMSc(D)
        s.mix = Attn(gh, gw) if kind == "attn" else GConv(); s.mlp = MLP()

    def forward(s, x):
        x = x + s.mix(s.n1(x))
        return x + s.mlp(s.n2(x))


class EncCF(nn.Module):
    """[1, 3, H, W] normalized pixels -> [1, 2048, H/32, W/32] language-model embeddings (row-major = token order)."""

    def __init__(s, Hh, Ww, layers=LAYERS, proj=True):
        super().__init__()
        gh, gw = Hh // P, Ww // P
        s.proj = proj
        s.patch = nn.Conv2d(3, D, P, stride=P, bias=False); s.en = RMSc(D)
        s.blocks = nn.ModuleList(Block(k, gh, gw) for k in layers)
        s.lnw = nn.Parameter(torch.ones(1, D, 1, 1)); s.lnb = nn.Parameter(torch.zeros(1, D, 1, 1)); s.nln = Norm(PEPS)
        s.l1 = nn.Conv2d(D, 4 * D, M, stride=M)          # LayerNorm -> 2x2 merge -> linear_1 == 2x2/2 conv
        s.l2 = c1(4 * D, 2048, bias=True)

    def forward(s, x):
        x = s.en(s.patch(x))
        for b in s.blocks:
            x = b(x)
        if not s.proj:
            return x
        x = s.nln(x - x.mean(1, keepdim=True)) * s.lnw + s.lnb
        return s.l2(F.gelu(s.l1(x)))

    def norms(s):
        return [s.en.nrm, s.nln] + [n for b in s.blocks for n in [b.n1.nrm, b.n2.nrm] + ([b.mix.nq, b.mix.nk] if hasattr(b.mix, "nq") else [])]

    S = 64.0    # residual-stream scale: the true stream is ~0.02 at the first layers and ~30 after the last one

    @torch.no_grad()
    def calibrate(s, imgs):
        """Rescale for fp16 on the Adreno, which flushes fp16 subnormals to zero: the encoder's activations are tiny
        (residual ~0.02, gated-conv products ~1e-5). Call once, right after load(); the result is mathematically
        unchanged. 1) the residual stream is carried as S x the true value; 2) the outputs of b/h/c (gated conv) and
        q/k/v/w3 are scaled to rms ~1 by powers of two folded into their weights, and each branch's output projection
        absorbs the compensation; 3) every norm gets k (see Norm)."""
        p2 = lambda r: 2.0 ** round(-math.log2(r))
        convs = {}
        for b in s.blocks:
            mods = [b.mlp.w3] + ([b.mix.b, b.mix.h, b.mix.c] if isinstance(b.mix, GConv) else [b.mix.q, b.mix.k, b.mix.v])
            for m in mods:
                convs[m] = []
        hooks = [m.register_forward_hook(lambda m, i, o: convs[m].append(float(o.pow(2).mean(1).sqrt().median())))
                 for m in convs]
        for im in imgs:
            s(im[None])
        for h in hooks:
            h.remove()
        f = {m: p2(sorted(v)[len(v) // 2]) for m, v in convs.items()}
        for m, a in f.items():
            m.weight.mul_(a)
        s.en.w.mul_(s.S)
        for b in s.blocks:
            mx = b.mix
            if isinstance(mx, GConv):
                mx.out.weight.mul_(s.S / (f[mx.b] * f[mx.h] * f[mx.c]))
            else:
                mx.o.weight.mul_(s.S / f[mx.v])   # q, k feed norms: only their eps needs the scale
                mx.nq.inscale, mx.nk.inscale = f[mx.q], f[mx.k]
            b.mlp.w2.weight.mul_(s.S / f[b.mlp.w3])
            b.n1.nrm.inscale = b.n2.nrm.inscale = s.S
        s.nln.inscale = s.S
        for n in s.norms():
            n.k, n.cal = 1.0, []
        for im in imgs:
            s(im[None])
        for n in s.norms():
            n.k = 2.0 ** round(-math.log2(sorted(n.cal)[len(n.cal) // 2])) if n.cal else 1.0; n.cal = None
        return s

    @torch.no_grad()
    def load(s, W):
        cv = lambda t: t[:, :, None, None]
        s.patch.weight.copy_(W["vision_tower.patchifier.weight"].view(D, 3, 2, P, P).sum(2))
        s.en.w.copy_(W["vision_tower.embedding_norm.weight"].view(1, D, 1, 1))
        for i, (b, kind) in enumerate(zip(s.blocks, LAYERS)):  # first len(blocks) layers
            L = lambda n: W[f"vision_tower.layers.{i}.{n}"]
            b.n1.w.copy_(L("operator_norm.weight").view(1, D, 1, 1)); b.n2.w.copy_(L("ffn_norm.weight").view(1, D, 1, 1))
            for m, n in ((b.mlp.w1, "w1"), (b.mlp.w3, "w3"), (b.mlp.w2, "w2")):
                m.weight.copy_(cv(L(f"feed_forward.{n}.weight")))
            if kind == "attn":
                a = b.mix
                for m, n in ((a.q, "q_proj"), (a.k, "k_proj"), (a.v, "v_proj"), (a.o, "out_proj")):
                    m.weight.copy_(cv(L(f"self_attn.{n}.weight")))
                a.qn.copy_(L("self_attn.q_layernorm.weight").view(1, HD, 1))
                a.kn.copy_(L("self_attn.k_layernorm.weight").view(1, HD, 1))
            else:
                c = b.mix; Bw, Cw, Xw = L("conv.in_proj.weight").chunk(3, 0)
                c.b.weight.copy_(cv(Bw)); c.c.weight.copy_(cv(Cw)); c.h.weight.copy_(cv(Xw))
                c.dw.weight.copy_(L("conv.spatial_conv.weight")); c.out.weight.copy_(cv(L("conv.out_proj.weight")))
        pj = lambda n: W[f"multi_modal_projector.{n}"]
        s.lnw.copy_(pj("layer_norm.weight").view(1, D, 1, 1)); s.lnb.copy_(pj("layer_norm.bias").view(1, D, 1, 1))
        # linear_1 input index = (mr*2 + mc)*D + d  ->  conv weight [o, d, mr, mc]
        s.l1.weight.copy_(pj("linear_1.weight").view(4 * D, M, M, D).permute(0, 3, 1, 2))
        s.l1.bias.copy_(pj("linear_1.bias"))
        s.l2.weight.copy_(cv(pj("linear_2.weight"))); s.l2.bias.copy_(pj("linear_2.bias"))
        return s


FP32_LAYERS = ("MatMul", "Softmax", "BinaryOp", "Reduction")   # Reduction: mean(x^2) of the norms overflowed fp16 on non-square shapes (NaN)


def fp32_arith(param):
    """Mark the attention products, softmax, element-wise ops and reductions to compute in fp32 (ncnn featmask 31=1)
    while storage stays fp16: per-token cos vs fp32 goes from 0.36 to 0.96 (min) at 512 px for +1% time, and the
    norms' mean(x^2) no longer overflows fp16 (NaN) on non-square shapes."""
    lines = open(param).read().splitlines()
    out = lines[:2] + [l + " 31=1" if l.split()[0] in FP32_LAYERS and " 31=" not in l else l for l in lines[2:]]
    open(param, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ref":
        img = preprocess(sys.argv[2])
        x, f = reference(load_weights(), img)
        torch.save({"img": img, "hidden": x, "feat": f}, sys.argv[3])
        print(tuple(img.shape), tuple(x.shape), tuple(f.shape))
    elif cmd == "export":
        import pnnx
        Hh, Ww = int(sys.argv[2]), int(sys.argv[3])
        m = EncCF(Hh, Ww).eval().load(load_weights())
        if not CALIB:
            raise SystemExit("no calibration photos: put a few in calib/*.jpg or set VL_CALIB")
        m.calibrate([preprocess(p, (Hh, Ww)) for p in CALIB])
        print("norm scales:", [n.k for n in m.norms()])
        x = torch.randn(1, 3, Hh, Ww)
        with torch.no_grad():
            pnnx.export(m, f"vlenc_{Hh}x{Ww}.pt", (x,))
        fp32_arith(f"vlenc_{Hh}x{Ww}.ncnn.param")
