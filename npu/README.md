# lfm3-q6a-npu

*[Català](README.ca.md)*

Run **[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)** (Liquid AI) on the **NPU** of the
**Radxa Dragon Q6A** (Qualcomm QCS6490, Hexagon HTP v68), entirely on the NPU, with no
Qualcomm SDK and no Qualcomm account (the optional native runner only needs the public QNN
API headers, which its build script downloads from Ubuntu's public Qualcomm PPA).

The model is rebuilt by hand as quantized ONNX graphs (int4/int8 weights, 16-bit
activations) that the QNN execution provider of `onnxruntime-qnn` compiles for the HTP. A
host loop keeps the KV caches and convolution states and calls the graphs token by token,
or 16 tokens at a time for the prompt. Two runtimes run the compiled graphs: a **native
runner** (C++ calling the QNN API directly, state kept in shared NPU memory; the default) and
ONNX Runtime. Both give bit-identical outputs.

| | |
|---|---|
| Decode | **11.7 tok/s** (4096 context, native runtime); ONNX Runtime: 7.8 tok/s (4096), 8.9 (2048), 9.4 (1024) |
| Prefill | chunks of 16 tokens, 4096 context: **150 tok/s** on a 723-token prompt with the native runtime (ONNX Runtime: 94 tok/s; ~85 tok/s on a 2.8k-token prompt) |
| Load time | ~2-4 s native, ~6-9 s ONNX Runtime (compiled contexts) |
| Memory | 2.1 GB of compiled graphs + 0.5 GB embedding, ~3.6 GB process RSS (4096 context) |
| Quality vs fp32 | top-1 agreement 84.1%, mean KL 0.156 (teacher-forced) |
| Task bench | 36/36 basic tasks (tool calls, JSON, instructions, knowledge, math); 22/28 hard tasks (fp32: ~26/28) |

Everything here was measured on the board. Details in [docs/RESULTS.md](docs/RESULTS.md).

## Contents

| Path | What |
|---|---|
| `npu/` | Graph builder, NPU runtimes, quantization (GPTQ, calibration, sensitivity), evaluation, profiling, chat CLI |
| `native/` | Native QNN runner (`qnnrun.cpp`, built by `native/build.sh`) used by the native runtime |
| `configs/` | Activation ranges and int8 matrix lists used for the reference build |
| `experiments/` | AWQ + GPTQ (tried, did not help; kept for reference) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Step by step: from the Hugging Face checkpoint to a running model |
| [docs/HTP_V68.md](docs/HTP_V68.md) | What works and what breaks on the v68 HTP, and how the graphs are designed around it |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | Weight/activation quantization choices and the experiments behind them |
| [docs/PC_TOOLS.md](docs/PC_TOOLS.md) | Using an x86-64 PC with an NVIDIA GPU: GPTQ, evaluation, and **offline NPU compilation** |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | DSP hangs, QNN errors and how to recover |
| [docs/MODELS.md](docs/MODELS.md) | Other models that can be useful on this NPU |
| [docs/RESULTS.md](docs/RESULTS.md) | Speed, quality and benchmark numbers |

## Quick start

You need the board set up as in [Requirements](#requirements), the model checkpoint, GPTQ
weights and a build. The full procedure is in [docs/PIPELINE.md](docs/PIPELINE.md); in short:

```bash
# 1. environment on the board
python3 -m venv .venv && .venv/bin/pip install -r requirements-board.txt
.venv/bin/hf download LiquidAI/LFM3-1.7B --local-dir models/LFM3-1.7B

# 2. calibration data + GPTQ int4 weights (GPU PC recommended, see docs/PC_TOOLS.md)
python npu/gen_calib_lfm3.py models/LFM3-1.7B build/calib_chat.jsonl
CHAT=build/calib_chat.jsonl python npu/gptq_lfm3.py models/LFM3-1.7B data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet build/gptq 128 512

# 3. build and compile the NPU graphs (board: ~13 min, or offline on a PC: ~6 min)
export INT8_LAYERS=0,1,2,3,4,5 INT8_MATS=configs/int8_mats_100M.json
.venv/bin/python npu/build_all_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 4096 6
.venv/bin/python npu/build_prefill_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 16

# 4. native runner (optional, ~1.5x faster decode; needs g++)
native/build.sh

# 5. run
.venv/bin/python npu/chat.py            # --runtime native|ort
```

From your own code, `npu/runtime.py` `load_model(build_dir)` returns the native runtime
(`npu/lfm3_native.py`, `LFM3Native`) when `native/libqnnrun.so` is built, otherwise the ONNX Runtime one
(`npu/lfm3_npu.py`, `LFM3NPU`); `LFM3_RUNTIME=native|ort` forces one. Both have the same interface:
`feed(token_ids)` returns the logits after a prompt, `step(token)` decodes one token.

## Requirements

- Radxa Dragon Q6A. Tested on Ubuntu 26.04.1 LTS, kernel `7.2.9-edge-qcs6490`, with the
  distribution's `fastrpc` / `libcdsprpc1` packages (1.0.7) and the CDSP firmware that ships
  with the image.
- Your user in the `fastrpc` group (`/dev/fastrpc-cdsp`).
- Python 3.14 with `onnxruntime==1.27.0` and `onnxruntime-qnn==2.3.0` (QNN 2.47). The QNN
  HTP libraries, including the v68 skeleton that runs on the DSP, come inside the
  `onnxruntime-qnn` wheel. The native runner uses those same libraries; building it needs
  `g++`, `curl` and `dpkg-deb`.
- ~12 GB RAM board: building graphs and running the model fit. The fp32 reference model
  (only for validation) needs ~7 GB and is better run on a PC.
- Optional but recommended: an x86-64 PC with an NVIDIA GPU (8 GB+) for GPTQ and
  evaluation, which can also compile the NPU graphs offline
  ([docs/PC_TOOLS.md](docs/PC_TOOLS.md)).

## How it works (short)

- **Graphs.** 24 layers (12 gated short-convolution, 6 full-attention, 6 sliding-window
  attention) are split into 4 graphs of 6 layers, plus an int8 LM head. Another 4 graphs
  process 16 prompt tokens at once (chunked prefill).
- **Precision.** Activations are uint16 (QDQ), weights are symmetric per-channel int4
  (GPTQ). The first 6 layers and 26 sensitive matrices use int8 (see below).
- **Host loop.** The embedding lookup, KV caches (full attention: up to the context;
  sliding: a 255-slot ring), convolution states, RoPE tables and masks live on the host and
  are passed in and out as uint16/uint8 tensors. The native runtime keeps all of them in
  rpcmem buffers registered once with QNN, so nothing is copied per token (ONNX Runtime copies
  ~53 MB of state into QNN's buffers on every token), and runs the whole decode step in C++.
  Where the time goes: [docs/RESULTS.md](docs/RESULTS.md#decode-profile).
- **v68 workarounds.** No RMSNorm op (LpNorm × γ√N instead), 16×16-bit activation matmuls
  silently lose precision (split into hi/lo bytes), block-int4 weights give wrong results
  (per-channel only). See [docs/HTP_V68.md](docs/HTP_V68.md).

### Why some weights are int8

With GPTQ int4 everywhere, the model became strongly biased towards emitting a tool call
for trivial questions when tools were present: p(`<tool_call>`) for "What is the capital
of Italy?" went from 0.000 (fp32) to 0.994. Splicing NPU hidden states into the fp32 model
showed that the bias comes from the accumulated int4 error of the first 6 layers. Using
int8 for those layers fixes it (0.010) at a ~10% speed cost. A per-matrix sensitivity sweep
then picked 26 more matrices (mostly attention `k_proj`/`v_proj` and convolution
`out_proj`) that reduce the KL to fp32 by 17% for ~1.5% speed.
Details in [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

## Known issues

- **DSP memory leak around process start/exit.** On this kernel, fastrpc often fails to
  unmap a DSP buffer when an NPU process starts or exits (`unmmap ... ERROR` in `dmesg`).
  After several of these the CDSP stops responding and every NPU session fails until the
  CDSP is restarted. Keep the model in one long-lived process and avoid many short NPU
  processes in a row. Stop all NPU processes before restarting the CDSP. Recovery steps are in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
- **One QNN runtime per process.** Do not load the ONNX Runtime sessions and the native
  runner in the same process: the DSP runs out of address space, the process can hang, and
  the memory it had mapped stays lost until a reboot.
- **Do not read `/sys/kernel/debug/dma_buf/bufinfo`** on this kernel: it oopses and leaves
  the dma-buf lock held, so every later NPU session hangs until a reboot.

## Credits

- **[Liquid AI](https://huggingface.co/LiquidAI)** for LFM3-1.7B. The model weights are
  not included here and are under Liquid AI's own license.
- **[PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu](https://github.com/PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu)**,
  which showed that the PyPI `onnxruntime-qnn` wheel is enough to use the Q6A NPU.
- **onnxruntime** and its QNN execution provider (`onnxruntime-qnn`).
- **GPTQ** (Frantar et al., 2022) for the int4 weight quantization.
- **[rafaelwms/ubuntu-26.04.1-dragon-q6a](https://github.com/rafaelwms/ubuntu-26.04.1-dragon-q6a)**,
  the Ubuntu image this was developed and tested on.

## License

MIT for the code in this repository (see [LICENSE](LICENSE)). Model weights, tokenizer and
chat template belong to their authors and keep their own licenses.
