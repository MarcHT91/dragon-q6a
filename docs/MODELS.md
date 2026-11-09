# Other models for the QCS6490 NPU

## Rules of thumb

- **Decode is bound by weight bandwidth.** int4 weights are ~0.5 GB per billion parameters.
  LFM3-1.7B (int4 with some int8) decodes at 8-9 tok/s, so expect roughly
  **~14 tok/s at 1B, ~4-5 tok/s at 3-4B** with the same approach (estimates, not
  measurements).
- **Memory.** The board has 12 GB shared between CPU and NPU. A 4B model in int4 plus a
  2-4k context fits; 7B does not leave much room and would be slow.
- **No fp16 on v68**: everything must be quantized (int4/int8 weights, 8/16-bit activations).
  Per-channel int4 needs GPTQ (or similar) to keep quality.
- **Watch for activation outliers.** Some models (Qwen3: ~14k at the first token) have
  "attention sink" activations that a uint16 scale cannot cover; handle those positions on
  the host. LFM3 has none.
- **Big vocabularies** make the LM head expensive (LFM3: 128k × 2048 in int8). Gemma's
  262k vocabulary would cost noticeably more per token.
- **Hybrid / sliding-window models** need less KV cache traffic per token, which helps on a
  bandwidth-bound NPU.

## Tested on this board

| Model | How | Decode | Notes |
|---|---|---|---|
| **LFM3-1.7B** (Liquid AI) | this repository | 7.8-9.4 tok/s | tool calling, JSON, thinking mode; 36/36 basic tasks |
| **Qwen3-1.7B** | custom port with the same method (QDQ graphs + GPTQ, 4 groups of 7 layers, int8 head) | ~10 tok/s at 512 context | 91% top-1 vs fp32; first-token outlier handled on the host; 512 context is too short for tool prompts |
| **Llama-3.2-1B** | Radxa's precompiled Genie build (`radxa/Llama3.2-1B-4096-qairt-v68` on ModelScope) | 10.6 tok/s, prefill ~124 tok/s | needs `use-mmap: false` on kernel 7.2 (with mmap the CDSP crashes, Genie error 5005) and access to `/dev/dma_heap/system` |
| **Qwen3.5-0.8B** | [PrismPhi's port](https://github.com/PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu) | not measured here | ready-made installer with an OpenAI-compatible API |

## Candidates (not tested)

Small LLMs that should fit and run at a usable speed. Each needs its own graph builder
(attention/MLP layout, norms, RoPE variant) on top of `build_graphs.py`, plus GPTQ.

| Model | Size | Why it is interesting | Things to check |
|---|---|---|---|
| LFM2-350M / 1.2B / 2.6B (Liquid AI) | 0.35-2.6B | same hybrid short-conv + attention family as LFM3: closest to this builder | layer layout and conv details vs LFM3 |
| Qwen3-0.6B | 0.6B | very fast, tool calling, thinking mode | first-token outlier (as Qwen3-1.7B) |
| Qwen3-4B | 4B | much stronger reasoning; ~4-5 tok/s expected | memory for 4B + context; outliers |
| Llama-3.2-3B | 3B | good general model, 128k vocab | head cost |
| SmolLM2-1.7B, SmolLM3-3B | 1.7-3B | open data, small vocab (cheap head) | quality on tool calling |
| Granite-3.3-2B | 2B | tool calling, permissive license | — |
| Gemma-3-1B | 1B | sliding window + global layers | 262k vocab: expensive head |
| Phi-4-mini | 3.8B | strong at reasoning/math | upper limit for speed and memory |

## Non-LLM models that fit well

- **Text embeddings** (e.g. bge-small, all-MiniLM): small encoders, easy to quantize to
  w8a16; useful for local search / RAG next to the LLM.
- **Speech recognition** (Whisper tiny/base/small): the encoder is a good NPU fit; the
  decoder is small enough for a host loop like the one here.
- **Vision** (detection, classification, segmentation): Qualcomm AI Hub has ready w8a8
  models for the QCS6490; they run through the same onnxruntime-qnn setup.
