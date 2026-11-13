# LFM3-VL-2B on the GPU + NPU

*[Català](README.ca.md)*

[LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) (Liquid AI) with each part on its own accelerator of the
Dragon Q6A: the **vision encoder** (14 layers, 400M parameters with the projector) on the **Adreno 643 GPU** with ncnn
over Vulkan (Mesa Turnip), and the **language model** (LFM3, 24 layers) on the **Hexagon NPU** with the same graphs
and native runner as [`../npu`](../npu). The image features enter the language model as extra embedding rows at the
`<|image|>` positions, so the NPU graphs need nothing image-specific.

## Results

| | |
|---|---|
| Encoder on the GPU (fp16) | 256 / 384 / 512 px: 0.65 / 1.2 / 2.4 s; per-token cosine vs fp32 0.99-0.998 |
| Language model on the NPU | 11.6 tok/s decode; prefill 16-token chunks |
| Time to first token (one picture) | 256 / 384 / 512 px: 1.2 / 2.4 / 4.3 s |
| Quality vs fp32 (held-out COCO pictures, teacher-forced) | top-1 85.8%, KL 0.22 (fp32 image features); 86.8%, KL 0.22 with the GPU's features |

The answers are mostly the same as fp32's (*"The image shows three giraffes reaching up to eat green leaves from a
mesh feeder hanging on a tree."*). Checks behind these numbers: the fp32 encoder reference reproduces the Hugging
Face model; the C++ tokenizer is identical to Hugging Face `tokenizers` on 85k tokens of mixed text; the image
resize is bit-identical to Pillow's.

## Contents

| Path | What |
|---|---|
| `encoder/vl_enc.py` | fp32 reference of the encoder + projector, and its channels-first version exported to ncnn (`export H W`) |
| `encoder/split_checkpoint.py` | splits the Hugging Face checkpoint into `vision.safetensors` and a plain LFM3 language model |
| `encoder/gpu_feats.py` | GPU features for the eval set (to validate the NPU with them) |
| [`app/`](app) | `lfm3vl`: the chat program in C++ (GPU encoder + NPU language model in one process) |
| `../npu/npu/vl_pipeline.bat` | PC side of the NPU build: calibration data with pictures, eval set, GPTQ, ranges, compile |
| `../npu/npu/validate_vl.py` | NPU vs fp32 on the eval set |

## Build

1. **Split the checkpoint** (any machine): `python vl/encoder/split_checkpoint.py LFM3-VL-2B/ out/` gives
   `out/vision.safetensors` and `out/lm/`.
2. **NPU graphs** (PC with an NVIDIA GPU, offline compile as in [`../npu/docs/PC_TOOLS.md`](../npu/docs/PC_TOOLS.md)):
   workspace with `models\LFM3-VL-LM` (= `out/lm`), `vl\vision.safetensors`, `data\coco_val0.parquet` (one
   validation shard of COCO, e.g. `detection-datasets/coco` on Hugging Face), the wikitext parquet, and
   `npu\` = `../npu/npu`; run `npu\vl_pipeline.bat` (~30 min on an RTX 3060). It makes 128 picture prompts with
   questions answered by the fp32 model, adds the text chat calibration, runs GPTQ, measures the activation
   ranges and compiles (int4 + int8 layers 0-5, 4096 context). Copy `build\npu` to the board.
3. **Encoder** (board, venv with torch, pnnx and the patched ncnn of [`../gpu/ncnn`](../gpu/ncnn)):
   `VL_WEIGHTS=out/vision.safetensors python vl/encoder/vl_enc.py export 384 384` (a few ordinary photos in
   `calib/*.jpg` for the fp16 rescaling).
4. **Program**: see [`app/README.md`](app/README.md).

## Notes

- The Adreno flushes fp16 subnormals, and the encoder's activations are small: `vl_enc.py` rescales the
  residual stream and the norms' inputs by powers of two, and computes the attention products, softmax,
  element-wise ops and reductions in fp32 (fp16 storage). Without it, 512 px tokens drop to cosine 0.36.
- `net.opt.lightmode = False` is needed with fp16 storage on Vulkan (blob recycling corrupts results).
- The program uses square exports; the decision model in [`../d1`](../d1) picks the export closest to each
  picture's aspect ratio, which is more accurate.

The weights are Liquid AI's (LFM Open License v1.0) and are not redistributed here.
