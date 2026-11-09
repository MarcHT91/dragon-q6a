# The QCS6490 HTP (v68) for LLMs: what works, what breaks

Measured with onnxruntime 1.27.0 + onnxruntime-qnn 2.3.0 (QNN 2.47), `htp_arch=68`, CPU
fallback disabled, by compiling small graphs and comparing NPU output with a CPU QDQ
simulation of the same graph. Every op pattern in the LFM3 graphs was checked this way
before being used.

## Data types

- **No fp16 compute on v68.** Everything is quantized: activations uint16 (QDQ, zero
  point 32768), weights int4 or int8.
- **int4 weights, per output channel, symmetric: correct** and ~2.3× faster than int8
  (they stay packed in memory; decode is bandwidth-bound).
- **int4 weights per block (`block_size`): compile but give wrong results.** Use per-channel
  only. This is why GPTQ matters: per-channel int4 with round-to-nearest loses too much.
- **int8 weights with uint16 activations:** work, but are less precise on the NPU than the
  CPU simulation of the same graph (4% vs 0.3% error on a small-range test input). In a
  full model, int8 is still far closer to fp32 than int4.
- **Dead channels.** A weight row whose per-channel scale is ~1e-8 overflows the
  requantizer and outputs 65535. Clamp scales to ≥ max·1e-4 and zero those rows
  (`G.weight_int4` in `build_graphs.py`).

## Ops

| Pattern | Status |
|---|---|
| Elementwise Mul/Add/Sub, ReduceSum, Sigmoid, SiLU, Softmax, Concat, Split, Reshape, Transpose at uint16 | precise (~1e-4) |
| MatMul activation × weight (u16 × int4/int8) | precise |
| MatMul activation × activation (u16 × u16) | **the second operand is silently reduced to 8 bits** (~4.9% error) |
| SimplifiedLayerNormalization (RMSNorm) | **not supported** |
| LpNormalization (p=2) | supported |

**A16×A16 MatMul fix.** Split the second operand into signed high and low bytes (zero
point 128), do two 16×8 MatMuls and add them (`hi·256 + lo`). With zero point 0 the error
is 0.39%; with 128 it is 0.05%. The KV caches are stored on the host already split
(`past_kh/kl`, `past_vh/vl` as uint8), so attention scores and the value product use this
scheme with no extra work per token.

**RMSNorm.** `RMSNorm(x)·γ = LpNorm₂(x) · γ·√N` (ignoring eps). When eps matters (small
vectors, eps 1e-5), append a constant `c = √(N·eps)` to the vector before the LpNorm and
drop it afterwards; this is exact.

**In-chunk attention for prefill.** For the 16 new tokens of a prefill chunk, the
attention among themselves uses elementwise Mul + ReduceSum instead of MatMul, to avoid the
A16×A16 precision loss.

## Graph layout used for LFM3

LFM3-1.7B has 24 layers: 12 gated short-convolution layers (kernel 7 cache), 6 full GQA
attention layers and 6 sliding-window (256) GQA attention layers, hidden size 2048, 16
query / 4 KV heads of 128, MLP size 7680, partial RoPE, a 128k vocabulary and a tied embedding.

- **4 decode graphs of 6 layers** (one token in, residual stream as uint16) and an **int8
  LM head** graph (per-row RTN of the 128k × 2048 tied embedding). Fewer, bigger graphs
  reduce per-call overhead; one graph for all 24 layers was not needed.
- **4 prefill graphs** with the same layer split that take 16 tokens per call and return
  the 16 new K/V rows and the new convolution states. The host feeds whole 16-token chunks
  through them and the remaining tokens through the decode graphs.
- **Host side** (`lfm3_npu.py`): embedding lookup (fp16 table), RoPE cos/sin, additive
  masks, the full-attention KV cache (T slots), the sliding-window ring (255 slots + self),
  and the convolution states. All passed as uint16/uint8 tensors.
- **Per-tensor activation scales** come from calibration (`configs/lfm3_act_ranges.json`):
  symmetric, abs-max × 1.25.
- LFM3 has no "attention sink" outlier at position 0. Other models (Qwen3: activations of
  ~14k at the first token) need that position handled separately (fp32 K/V precomputed on
  the host) because a uint16 scale cannot cover it.

## Compiling

- QNN EP options: `backend_path=libQnnHtp.so`, `htp_arch=68`,
  `htp_graph_finalization_optimization_mode=3`, `htp_performance_mode=burst`,
  `session.disable_cpu_ep_fallback=1` (so unsupported nodes fail loudly instead of running
  on the CPU).
- Contexts are saved with `ep.context_enable=1`, `ep.context_embed_mode=0` →
  `X_ctx.onnx` + `X_ctx_qnn.bin`; loading those takes seconds instead of a recompile.
- Offline (x86-64 PC) compilation needs `soc_model=35` and **`vtcm_mb=2`**: the QCS6490
  has 2 MB of VTCM (the HTP's fast on-chip memory). Contexts compiled for 4 or 8 MB
  compile fine but the board rejects them with `QNN_CONTEXT_ERROR_CREATE_FROM_BINARY`.

## Debugging tips

- To find where a graph goes wrong, prune it to each intermediate tensor, compile it alone
  and compare NPU vs CPU QDQ simulation. Keep the dangling Q/DQ nodes of Split outputs,
  otherwise QNN rejects the graph.
- A stale `*_ctx.onnx` from an earlier build is loaded silently: delete contexts whenever a
  graph is rebuilt.
- `test_layer_lfm3.py` builds one layer and reports the relative error of NPU, CPU QDQ
  simulation and fp32 for that layer's outputs.
