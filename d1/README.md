# d1-2B decisions on the GPU + NPU

*[Català](README.ca.md)*

[d1-2B](https://huggingface.co/LiquidAI/d1-2B) (Liquid AI) is a **decision model** built on LFM3-VL-2B: given a
state (text, JSON and/or a picture) and named questions, it returns calibrated answers in **one forward pass**,
without generating tokens. Here it runs on the Dragon Q6A with the vision encoder on the **Adreno GPU** (ncnn) and the
language model on the **Hexagon NPU** (QNN, int8 weights), in one C++ process.

```
$ d1 --build npu --enc enc request.json
{"answers": {"refund": {"type": "noul", "noul": 0.996},
             "team": {"type": "choice", "choice": "billing", ...},
             "topics": {"type": "choice", "choices": ["billing", "delivery"], ...},
             "urgency": {"type": "score", "score": 1.0, ...}}, ...}
```

Question types: `noul` (yes/no: P(yes)), `choice` (one option, or `"multiple": true`), `score` (2-10 ordered levels).
Not yet: the typed reads (text/number/date spans), estimates and boxes.

## Results

Measured on 60 held-out requests (24 COCO pictures, 20 Wikipedia passages, 16 home-assistant commands; 652
answers), against d1 in fp32 with pictures at their own size:

| | Same answer | Mean probability change | 95% below |
|---|---|---|---|
| NPU, fp32 picture features | **98.9%** | 0.008 | 0.034 |
| NPU + GPU encoder (picture shape kept, 192 image tokens) | **97.1%** | 0.027 | 0.110 |

The model card's examples come out exactly (`1.0 billing ['billing', 'delivery'] medium`, `shipping`, `technical`).

Speed (`--serve`, warm):

| Request | Tokens | Time |
|---|---|---|
| Home command + 2 questions (intent, room) | 132 | 1.3 s |
| Picture + 3 questions | 274 | 3.9 s (GPU 1.7 + NPU 2.2) |
| Picture + 13 questions | 673 | 6.5 s (GPU 1.7 + NPU 4.8) |

The NPU part is bound by reading the int8 weights (every 64-token chunk reads each stage's ~250 MB); each question
adds ~45 tokens.

## How it works

- **`set` attention.** d1 packs up to 16 questions after the state. On the 6 full-attention layers every state token
  sees the whole sequence and each question token sees the state and its own question, both ways (sliding-window and
  convolution layers stay causal). The NPU graphs are prefill chunks, so the model runs **stage by stage**: a stage is
  one full-attention layer, the causal layers after it, and the next full-attention layer's keys/values only. Each
  stage runs over every chunk before the next one starts, so all the keys exist when a query needs them. 7 stages,
  64-token chunks, 1024-token context.
- **int8 weights everywhere**, 16-bit activations, and per-token rescaled projections before the post-norms
  (`PT_NORM`, see [`../npu/docs/HTP_V68.md`](../npu/docs/HTP_V68.md)).
- **Readout on the CPU.** Only the answer tokens' logits are needed (yes/no, option codes, digits): final norm and a
  few dot products, instead of a 263 MB LM head on the NPU. The full-vocabulary normalization cancels in the option
  softmax. Then d1's calibration (temperatures, the multiple-choice table) as in its release.
- **Pictures**: the encoder exported for 512x384, 384x512, 576x320, 320x576 and 448x448; each picture uses the shape
  closest to its aspect ratio.

**Decisions and the tests behind them** (all against fp32 d1 on held-out requests):

- *Weights*: int4 (RTN, GPTQ, AWQ+GPTQ), int4 with int8 sensitive layers, and int8 were compared in simulation; int8
  keeps 99.7% of the answers with a mean change of 0.003; the best mix (GPTQ int4 + int8 sensitive layers) 98.9% and
  0.012. AWQ did not help.
- *NPU precision*: the NPU first gave 97.1% / 0.030 with fp32 features. The loss was traced, op by op, to the
  output of the W8A16 MatMuls (~12 effective bits); the per-token rescaling brought it to 98.9% / 0.008. Clipping the
  activation ranges instead made it worse.
- *Picture shape*: square 384 px loses 3.4 points even in fp32; keeping the aspect ratio at ~200k pixels loses 1.7.
- *Chunks*: 64 tokens with a 1024 context is the fastest of the variants tried (128-token chunks were slower).

## Files

| File | Where | What |
|---|---|---|
| `d1_prompt.py` | any | prompt, `set` packing and readout (no torch), shared by the reference and the NPU runtime |
| `d1_ref.py` | PC | fp32 reference (torch); `python d1_ref.py MODEL` runs the model card's examples |
| `d1_quant.py` | PC | the weight-quantization comparison above |
| `d1_data.py`, `d1_calib.py` | PC | calibration/eval requests; activation ranges + eval set with fp32 answers |
| `build_d1.py` | PC | builds and compiles the stage graphs (offline QNN compile) |
| `configs/ranges.json` | | the activation ranges used for the reference build |
| `d1_npu.py`, `d1_validate.py` | board | Python runtime and validation against the fp32 answers |
| `gpu_feats.py` | board | GPU picture features for the eval set |
| [`app/`](app) | board | `d1`: the C++ program (GPU encoder + NPU stages + readout), `validate_cpp.py` |

## Build and run

1. **NPU graphs** (PC with an NVIDIA GPU; [`../npu/docs/PC_TOOLS.md`](../npu/docs/PC_TOOLS.md)), with the Hugging
   Face checkpoint in `d1-2B/`:
   ```
   python d1/build_d1.py d1-2B d1/configs/ranges.json build 1024 64
   ```
   (`d1_calib.py d1-2B COCO_PARQUET WIKI_PARQUET work` remakes the ranges and the eval set.) Copy to the board:
   `stage*_ctx_qnn.bin`, `embed_f16.npy`, `fnorm.npy`, `manifest.json`, `tokenizer.json`, `calibration.json`,
   `config.json`.
2. **Encoder** (board): `python vl/encoder/split_checkpoint.py d1-2B out/`, then for each shape
   `VL_WEIGHTS=out/vision.safetensors python vl/encoder/vl_enc.py export 384 512` (and 512 384, 320 576, 576 320,
   448 448) in the encoder directory.
3. **Program** (board, same requirements as [`../vl/app`](../vl/app)):
   `NCNN_DIR=... PY=.../python d1/app/build.sh`.
4. **Run**: `d1 --build BUILD --enc ENC request.json`, or `d1 --build BUILD --enc ENC --serve` to keep the model
   loaded and answer one JSON request per stdin line. A request:
   ```json
   {"state": "I was charged twice and my parcel never arrived.", "image": "photo.jpg",
    "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}}
   ```
   `state` may be text, any JSON value, or `null` with only a picture.

Requires `/dev/dma_heap/system` writable by the `fastrpc` group and one NPU process at a time
([`../npu/docs/TROUBLESHOOTING.md`](../npu/docs/TROUBLESHOOTING.md)). The weights are Liquid AI's (LFM Open License
v1.0) and are not redistributed here.
