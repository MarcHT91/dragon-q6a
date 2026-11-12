"""Channels-first synthetic LFM3-VL vision encoder (random weights) for ncnn/Vulkan on Turnip.
Linear -> 1x1 Conv2d (ncnn's Vulkan conv1x1 is ~4x faster than Gemm/InnerProduct on the Adreno 643,
and Gemm with subgroup ops hangs the GPU). x: [1, D, g, g].
Usage: python enc_cf.py GRID WHICH(full|attn1|conv1) [sdpa]"""
import sys, torch, torch.nn as nn, torch.nn.functional as F

D, I, H, KV, HD = 1024, 2560, 16, 8, 64
LAYERS = ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv"]
SDPA = "sdpa" in sys.argv


def lin(i, o):
    return nn.Conv2d(i, o, 1, bias=False)


class LN(nn.Module):  # LayerNorm over the channel dim
    def __init__(s):
        super().__init__()
        s.w = nn.Parameter(torch.ones(1, D, 1, 1)); s.b = nn.Parameter(torch.zeros(1, D, 1, 1))

    def forward(s, x):
        m = x.mean(1, keepdim=True); xc = x - m
        return xc * torch.rsqrt((xc * xc).mean(1, keepdim=True) + 1e-5) * s.w + s.b


class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.w1 = lin(D, I); s.w3 = lin(D, I); s.w2 = lin(I, D)

    def forward(s, x):
        return s.w2(F.silu(s.w1(x)) * s.w3(x))


class Attn(nn.Module):
    def __init__(s, g):
        super().__init__()
        s.g = g; s.q = lin(D, H * HD); s.k = lin(D, KV * HD); s.v = lin(D, KV * HD); s.o = lin(H * HD, D)

    def forward(s, x):
        n = s.g * s.g
        q = s.q(x).reshape(H, HD, n).transpose(1, 2)            # [H, n, HD]
        k = s.k(x).reshape(KV, 1, HD, n).expand(KV, H // KV, HD, n).reshape(H, HD, n)
        v = s.v(x).reshape(KV, 1, HD, n).expand(KV, H // KV, HD, n).reshape(H, HD, n)
        if SDPA:
            a = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2))  # [H, n, HD]
        else:
            a = torch.softmax((q * HD ** -0.5) @ k, dim=-1) @ v.transpose(1, 2)
        return s.o(a.transpose(1, 2).reshape(1, H * HD, s.g, s.g))


class GConv(nn.Module):
    def __init__(s):
        super().__init__()
        s.b = lin(D, D); s.c = lin(D, D); s.h = lin(D, D)
        s.dw = nn.Conv2d(D, D, 3, padding=1, groups=D, bias=False); s.out = lin(D, D)

    def forward(s, x):
        return s.out(s.c(x) * s.dw(s.b(x) * s.h(x)))


class Block(nn.Module):
    def __init__(s, kind, g):
        super().__init__()
        s.n1 = LN(); s.n2 = LN(); s.mix = Attn(g) if kind == "attn" else GConv(); s.mlp = MLP()

    def forward(s, x):
        x = x + s.mix(s.n1(x))
        return x + s.mlp(s.n2(x))


class Enc(nn.Module):
    def __init__(s, g, layers):
        super().__init__()
        s.blocks = nn.ModuleList(Block(k, g) for k in layers)

    def forward(s, x):
        for b in s.blocks:
            x = b(x)
        return x


if __name__ == "__main__":
    import pnnx
    g, which = int(sys.argv[1]), sys.argv[2]
    layers = {"full": LAYERS, "attn1": ["attn"], "conv1": ["conv"]}[which]
    tag = f"cf_{which}{'_sdpa' if SDPA else ''}"
    torch.manual_seed(0)
    m = Enc(g, layers).eval()
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=0.02)
    x = torch.randn(1, D, g, g)
    with torch.no_grad():
        torch.save({"x": x, "y": m(x)}, f"io_{tag}_{g}.pt")
        pnnx.export(m, f"enc_{tag}_{g}.pt", (x,))
