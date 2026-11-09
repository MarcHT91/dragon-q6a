# Quantization of LFM3-1.7B for the v68 HTP

Constraints from the hardware ([HTP_V68.md](HTP_V68.md)): activations uint16, weights int4
per output channel or int8, no block-wise int4. Decode speed is bound by weight bandwidth,
so int4 is ~2× faster than int8 per matrix.

Quality metric used throughout: teacher-forced comparison with the fp32 model on short
chat-formatted texts (`npu/eval_texts.py`): **top-1 agreement** of the next-token argmax
and **mean KL divergence** of the next-token distributions. Weight-only variants are
simulated in fp32 on a GPU (`bench.py lfm3-ref` with `QW_*`, `sens.py`); NPU numbers come
from `validate_build.py`.

## 1. Int4 weights: GPTQ is required

| Weights (all int4, weight-only simulation) | top-1 | KL |
|---|---|---|
| Round-to-nearest, MSE clip search | 0.705 | 0.593 |
| GPTQ (act-order, MSE clip, sequential) | 0.815 | 0.256 |

GPTQ calibration data: 128 × 512 tokens of wikitext-2 plus chat-formatted sequences from
`gen_calib_lfm3.py`. Using wikitext alone gave 76.3% top-1 on the NPU (KL 0.48); adding
chat data gave 82.1% (KL 0.29).

## 2. Most of the NPU error comes from the weights

| | top-1 | KL |
|---|---|---|
| int8 layers 0-5 + GPTQ int4, weights only (fp32 activations) | 0.841 | 0.169 |
| same build on the NPU (uint16 activations) | 0.841 | 0.188 |
| all weights int8 (RTN), weights only | 0.971 | 0.004 |

16-bit activations add only ~0.02 KL. The int4 weights are the limit.

## 3. The tool-call bias and the int8 first layers

With GPTQ int4 everywhere the model, given a system prompt with tools, called a tool for
questions that need none:

| p(`<tool_call>`) as first token | fp32 | NPU, int4 everywhere | NPU, int8 layers 0-5 |
|---|---|---|---|
| "What is the capital of Italy?" | 0.000 | 0.994 | 0.004 |
| "Translate 'good morning' to Spanish." | 0.011 | 0.700 | 0.011 |

How it was located:

1. Run the prompt on the NPU and record the residual stream after each 6-layer group.
2. Splice those hidden states into the fp32 model at each group boundary and finish with
   fp32 layers. The bias is already complete after group 0 (layers 0-5): p = 0.987.
3. Weight-only test: GPTQ weights in layers 0-5 alone give p = 0.915; in layers 6-23 alone
   p = 0.017. No single layer explains it (the worst alone, the full-attention layer 2,
   gives 0.147); it is the accumulated int4 error of the early layers.

Fix: layers 0-5 in RTN int8 per channel (`INT8_LAYERS=0,1,2,3,4,5`). Cost: decode
10.5 → 9.4 tok/s (1024 context). Basic task bench: 34/36 → 36/36.

## 4. Per-matrix int8 budget

`npu/sens.py` switches each matrix of layers 6-23 alone to int8 on top of the configuration
above and measures the KL drop (12 texts: the eval texts, wikitext and chat sequences),
then greedily adds matrices by KL gain per parameter under a size budget. Results
(weight-only; ms/token estimated from the measured cost of int8 bandwidth):

| Budget (int8 params) | Matrices | Est. extra ms/token | KL | top-1 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.107 | 0.850 |
| 25M | 14 | +0.8 | 0.094 | 0.850 |
| 50M | 18 | +1.5 | 0.089 | 0.852 |
| **100M** | **26** | **+3.1** | **0.080** | **0.865** |
| 200M | 34 | +6.2 | 0.069 | 0.881 |
| 300M | 48 | +9.4 | 0.059 | 0.888 |

The best matrices per byte are attention `k_proj` / `v_proj` (small and sensitive), then
convolution `out_proj`, then some MLP `w2`/`w3`. The 100M list is used for the reference
build. Measured on the NPU: KL 0.188 → 0.156, decode 9.57 → 9.43 tok/s (−1.5%).

Task-level effect is noisy: on the 28 `--hard` tasks (weight-only simulation) 100M scored
23/28, 200M 19/28, 300M 23/28, all-int8 25/28 and fp32 about 26/28. With greedy decoding a
single different token changes a whole answer, so ±3-4 tasks is within noise; only all-int8
is clearly closer to fp32, at roughly 6-7 tok/s.

## 5. AWQ before GPTQ: no gain

`experiments/awq_gptq_lfm3.py` applies AWQ-style per-input-channel scales (folded exactly
into the norms, `v_proj` rows, the convolution's C projection rows and `w3` rows, so no op is
added; equivalence verified, top-1 1.000) and then GPTQ:

| Weight-only | GPTQ | AWQ + GPTQ |
|---|---|---|
| int4 everywhere | 0.815 / 0.256 | 0.792 / 0.251 |
| int8 layers 0-5 + int4 | 0.841 / 0.169 | 0.828 / 0.191 |
| RTN int4 | 0.705 / 0.593 | 0.679 / 0.666 |

LFM3 has no strong outlier input channels: the search picks α ≈ 0 (no scaling) for most
inputs, and where it scales (MLP `w2`, attention `o_proj`) the local error drops but the
logits do not improve.

## 6. Context length and speed

| Context T | Decode tok/s |
|---|---|
| 1024 | 9.43 |
| 2048 | 8.87 |
| 4096 | 7.81 |

The six full-attention layers read all T cache slots every token (the mask hides the
unused ones), so decode slows down as T grows.
