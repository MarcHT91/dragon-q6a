# lfm3vl: LFM3-VL-2B chat on the GPU + NPU

One C++ program that runs LFM3-VL-2B on the Radxa Dragon Q6A with each part on its own accelerator:

| Part | Where | How |
|---|---|---|
| Vision encoder + projector | Adreno 643 GPU | ncnn on Vulkan (Mesa Turnip), fp16, with our MatMul Vulkan layer |
| Language model (24 layers) | Hexagon NPU | QNN HTP context binaries, through `npu/native/qnnrun.cpp` |
| Tokenizer, image preprocessing, sampling | CPU | `tokenizer.h`, `image.h` |

The image features go into the language model as extra embedding rows at the `<|image|>` positions, so the NPU
graphs are the same ones the text model uses.

## Timings (single image, greedy, measured on the board)

| Image size | Image tokens | GPU encoder | Time to first token | Decode |
|---|---|---|---|---|
| 256 px | 64 | 0.65 s | 1.2 s | 11.5 tok/s |
| 384 px | 144 | 1.2-1.3 s | 2.3-2.4 s | 11.5 tok/s |
| 512 px | 256 | 2.4 s | 4.3 s | 11.6 tok/s |

Startup takes 8-15 s: loading the NPU contexts (2.6 GB). In parallel the GPU loads the encoder and runs it twice
as a warm-up.

While the GPU encodes the image, the NPU already processes the prompt tokens that come before it. The prefill
graphs take blocks of 16 tokens, and the last prompt token has to go on its own to produce the logits. So the
(L-1) mod 16 tokens that don't fit in a block are fed one at a time first, during the encoding, and everything
after the image goes in whole blocks plus one final step.

## Checks

- **Tokenizer:** identical to HuggingFace `tokenizers` on 4,641 lines (85,322 tokens) of mixed text, and every
  line round-trips (`test/tok_test.cpp`).
- **Resize:** bit-identical to Pillow's `Image.resize(BICUBIC)`. JPEG decoding (stb_image vs libjpeg) differs by
  up to 3 levels out of 255, with a mean of 0.05-0.09 (`test/img_test.cpp`).
- **End to end:** the answers match the Python validation (`npu/npu/validate_vl.py`) on the held-out COCO images,
  for example *"The image shows three giraffes reaching up to eat green leaves from a mesh feeder hanging on a
  tree."*
- **Encoder alone** (`test/enc_bench.cpp`): 1.13 s at 384 px and 2.24 s at 512 px, the same as from Python.

## Build

```sh
# 1. ncnn as a static library with Vulkan + our MatMul patch (~15 min on the board)
git clone --depth 1 --branch 20261020 https://github.com/Tencent/ncnn.git && cd ncnn
git submodule update --init --depth 1 glslang
python3 /path/to/dragon-q6a/gpu/ncnn/apply_patch.py .
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=OFF \
      -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/../ncnn-install
cmake --build build -j6 && cmake --install build
# 2. the app (also needs libpcre2-dev)
NCNN_DIR=/path/to/ncnn-install PY=/path/to/venv/with/onnxruntime_qnn/bin/python vl/app/build.sh
```

The QNN runtime libraries are the ones in the `onnxruntime_qnn` wheel. Their path is built in as the default,
and `--qnn` or `$QNN_DIR` overrides it.

## Run

```sh
lfm3vl --build NPU_BUILD_DIR --enc ENCODER_DIR [--size 384] [--image photo.jpg] [-p "Describe this image."]
```

- `NPU_BUILD_DIR`: the VL language model build (`npu/npu/vl_pipeline.bat` on the PC: `manifest.json`,
  `*_ctx_qnn.bin`, `embed_f16.npy`, `tokenizer.json`).
- `ENCODER_DIR`: holds `vlenc_{S}x{S}.ncnn.{param,bin}`, exported with `vl/encoder/vl_enc.py export S S`.
- Without `-p` it is an interactive chat. `/image FILE` attaches an image to the next message, `/reset` starts
  a new conversation and `/quit` exits. The conversation keeps the KV cache between turns.
- `--temp T --top-p P` turn on sampling (the default is greedy). `--think` lets the model write its reasoning.
  `--system TEXT` sets a system prompt. `-q` hides the timing lines.

**Requirement:** `/dev/dma_heap/system` must be readable and writable by the `fastrpc` group. Without it the
large NPU contexts fail to load ("fastrpc mmap ... Cannot allocate memory"). The udev rule shipped with fastrpc
uses `setfacl`, which is missing unless the `acl` package is installed. A direct rule works without it:

```
# /etc/udev/rules.d/62-dma-heap.rules
SUBSYSTEM=="dma_heap", KERNEL=="system", GROUP="fastrpc", MODE="0660"
```

## Limits

- The encoder is exported for fixed square sizes, so the image is resized to S x S and the aspect ratio is not
  kept. HF's processor keeps it with `smart_resize`. The model still describes the images correctly, but other
  shapes would need their own exports.
- One process at a time on the NPU. Don't kill it while it is running on the NPU: see `docs/TROUBLESHOOTING.md`.
