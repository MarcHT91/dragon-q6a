# Pipeline: from the checkpoint to LFM3 running on the NPU

All commands run from the repository root. `P` is the Python of your board venv
(`P=.venv/bin/python`). Steps marked **[GPU]** are much faster on a PC with an NVIDIA GPU
(see [PC_TOOLS.md](PC_TOOLS.md)). Steps marked **[NPU]** need the board.

Overview:

```
checkpoint ──► fp32 reference (lfm3_ref.py) ──► calibration data ──► GPTQ int4 weights
                                                      │                    │
                                                      ▼                    ▼
                                              activation ranges ──► QDQ ONNX graphs ──► QNN HTP contexts ──► runtime
                                                                    (build_lfm3.py)      (board or PC)        (lfm3_npu.py)
```

## 0. Board environment

```bash
sudo usermod -aG fastrpc $USER          # then log in again
python3 -m venv .venv                   # if venv is missing: sudo apt install python3-venv
.venv/bin/pip install -r requirements-board.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Check that the NPU is reachable (the first call loads the HTP libraries onto the DSP):

```bash
$P -c "import sys; sys.path.insert(0,'npu'); import npu_common as n; print(n.qnn_devices())"
```

## 1. Model and data

```bash
.venv/bin/hf download LiquidAI/LFM3-1.7B --local-dir models/LFM3-1.7B
.venv/bin/hf download Salesforce/wikitext --repo-type dataset \
    --include "wikitext-2-raw-v1/train-*" --local-dir data/wikitext
```

`npu/lfm3_ref.py` is a memory-light fp32 re-implementation of LFM3 (torch, CPU or CUDA,
weights loaded per layer). Everything else is checked against it. `npu/validate_ref.py`
compares it with the Hugging Face implementation (max difference ~2e-6).

## 2. Calibration data [GPU]

GPTQ and the activation ranges use two kinds of text:

- wikitext-2 (generic English);
- chat-formatted sequences in the model's own template, with tool calls, JSON,
  instructions, code, math, short multilingual answers and `<think>` traces.
  `gen_calib_lfm3.py` makes them by letting the fp32 model answer a built-in list of
  prompts:

```bash
python npu/gen_calib_lfm3.py models/LFM3-1.7B build/calib_chat.jsonl
```

Calibrating on chat data matters: with wikitext only, agreement with fp32 on chat text was
76.3% top-1 (KL 0.48); with chat data added, 82.1% (KL 0.29), before the int8 changes below.

## 3. GPTQ int4 weights [GPU]

```bash
CHAT=build/calib_chat.jsonl python npu/gptq_lfm3.py models/LFM3-1.7B \
    data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet build/gptq 128 512
```

Symmetric int4, per output channel (the only int4 layout that is correct on v68), act-order,
MSE-searched clipping, sequential (each layer is calibrated on the outputs of the already
quantized previous layers). Writes `build/gptq/layer{i}.npz`. A few minutes on an RTX 3060.

## 4. Activation ranges

Activations are uint16 with a symmetric scale per tensor, taken from the abs-max seen in
calibration plus a 25% margin.

```bash
python npu/calibrate_lfm3.py  models/LFM3-1.7B build/ranges_a.json                           # short built-in chats
python npu/calibrate2_lfm3.py models/LFM3-1.7B build/calib_chat.jsonl build/ranges_b.json build/gptq   # chat data, GPTQ weights
python npu/merge_ranges.py configs/lfm3_act_ranges.json build/ranges_a.json build/ranges_b.json
```

`configs/lfm3_act_ranges.json` in this repository is the result used for the reference
build; you can use it directly and skip this step.

## 5. Which weights stay int8

Two environment variables, read by the graph builder:

- `INT8_LAYERS=0,1,2,3,4,5`: whole layers in RTN int8 per channel instead of GPTQ int4.
- `INT8_MATS=configs/int8_mats_100M.json`: a list of `[layer, weight_name]` pairs in int8.

`configs/int8_mats_{100,200,300}M.json` come from `npu/sens.py` (per-matrix sensitivity,
[QUANTIZATION.md](QUANTIZATION.md)). The attention gate projection is always int8.

## 6. Build and compile the graphs [NPU, or PC]

```bash
export INT8_LAYERS=0,1,2,3,4,5 INT8_MATS=configs/int8_mats_100M.json
$P npu/build_all_lfm3.py     models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 4096 6
$P npu/build_prefill_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 16
```

- `build_all_lfm3.py MODEL RANGES GPTQ OUT [T] [GROUP]`: decode graphs `decode_g0..3`
  (GROUP=6 layers each, context T), the int8 LM head and `manifest.json`.
- `build_prefill_lfm3.py ... [C]`: prefill graphs that take C=16 prompt tokens per call.
- `rebuild_group_lfm3.py MODEL RANGES GPTQ BUILD GROUP_INDEX` rebuilds a single group in
  place (handy when trying int8 choices).

Each graph is compiled to `X_ctx.onnx` + `X_ctx_qnn.bin` the first time it is built. Later
loads use the context and take seconds. On the board the whole build takes ~13 minutes.
On a PC it takes ~6 minutes and does not touch the board's DSP ([PC_TOOLS.md](PC_TOOLS.md)).

Context length T: 4096 costs ~17% decode speed against 1024 (full-attention layers read
the whole cache every token). Pick the smallest T that fits your prompts.

> Delete stale `*_ctx*` files when a graph changes: an old context is loaded silently.

## 7. Validate [NPU]

```bash
python npu/make_eval_ref.py                 # [GPU or 7 GB RAM] fp32 logits of the eval texts -> build/eval/
$P npu/validate_build.py build/npu
```

Expected for the reference build:

```
p(<tool_call>) 0.999 1.000 1.000 1.000 1.000 1.000 0.692 0.000 0.010 0.003
decode 7.81 tok/s
teacher-forced top1 0.841 | KL 0.156
```

The first seven prompts should call a tool, the last three should not.

Task benchmark (36 basic tasks, `--hard` for 28 harder ones):

```bash
$P npu/bench.py lfm3-npu build/bench_npu.jsonl
$P npu/bench.py lfm3-npu build/bench_npu_hard.jsonl --hard
```

Layer-level checks used while developing the builder:

```bash
$P npu/test_layer_lfm3.py   models/LFM3-1.7B configs/lfm3_act_ranges.json 2 build/gptq   # one decode layer: NPU vs fp32 vs CPU QDQ simulation
$P npu/test_prefill_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json 2 build/gptq   # one prefill layer
$P npu/test_feed_lfm3.py    models/LFM3-1.7B build/npu                                  # chunked prefill vs token-by-token
```

## 8. Run [NPU]

Optional but recommended: build the native runner (~1.5× faster decode, same outputs). It
fetches the public QNN API headers from Ubuntu's Qualcomm PPA and needs `g++`:

```bash
native/build.sh                 # -> native/libqnnrun.so
$P npu/test_native.py ort       # reference from ONNX Runtime (own process)
$P npu/test_native.py native                     # same logits/tokens? + speed
```

Then:

```bash
$P npu/chat.py                  # interactive, no thinking
$P npu/chat.py --think          # with <think> reasoning (3-5x longer answers)
```

From Python:

```python
import sys; sys.path.insert(0, "npu")
from chat_format import ChatFormat
from runtime import load_model          # native if native/libqnnrun.so exists, else ONNX Runtime

cf = ChatFormat(); m = load_model("build/npu")
ids = cf.encode([{"role": "user", "content": "Hello!"}])
out, stats = m.generate(ids, 128)          # greedy
print(cf.tok.decode(out), stats)
```

`feed(ids)` uses the prefill graphs for whole 16-token chunks and the decode graphs for the
rest. Keep one model alive in a long-running process instead of starting many short
processes, and never load both runtimes in the same process
([TROUBLESHOOTING.md](TROUBLESHOOTING.md)). `prof_decode.py` shows where the decode time goes.

Tool calling: pass OpenAI-style tool schemas as `tools=` to `ChatFormat.encode`. The model
answers with `<tool_call><function=NAME><parameter=ARG>value</parameter>...</tool_call>`;
`bench.py`'s `parse_calls` shows how to parse it.
