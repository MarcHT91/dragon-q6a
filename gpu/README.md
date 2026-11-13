# GPU compute on the Adreno 643

*[Català](README.ca.md)*

Experiments (2026-11-12) to run neural-network work on the Adreno 643 GPU next to the NPU, with the
open driver (Mesa: msm kernel driver, Turnip Vulkan, rusticl OpenCL) and with Qualcomm's proprietary
stack (KGSL kernel driver, Adreno OpenCL/Vulkan, QNN GPU). The target workload was the vision encoder
of LFM3-VL-2B (164M parameters: 8 gated depthwise-conv + 6 bidirectional attention layers, hidden
1024, 16 px patches), with the language model staying on the NPU ([`../npu`](../npu)).

**Outcome:** the board stays on the open driver. ncnn on Turnip, with the fixes below, matches the
best Qualcomm-stack result for the encoder and keeps Kodi working. The Qualcomm shader compiler is
better (about 1.5x on a hand-written GEMM), but nothing in Qualcomm's ML libraries works better here.

## Results

### Vision encoder (LFM3-VL, 14 layers), steady state

| Runtime | Driver | 256 px | 384 px | 512 px |
|---|---|---|---|---|
| ncnn + our Vulkan MatMul (synthetic encoder) | Turnip (open) | **0.37 s** | **0.93 s** | 1.87 s |
| ncnn, upstream (MatMul falls back to the CPU) | Turnip (open) | 0.58 s | 1.47 s | 2.94 s |
| ncnn + our Vulkan MatMul | Qualcomm Vulkan | 0.37 s | 0.94 s | **1.60 s** |
| llama.cpp, real mmproj F16 | Turnip (open) | 0.79 s | 1.80 s | 3.36 s |
| llama.cpp, real mmproj F16 | Qualcomm OpenCL | 0.51 s | 1.01 s | 1.80 s |
| llama.cpp, real mmproj F16 | Qualcomm Vulkan | 2.0 s | 4.7 s | failed |

Tokens for the LLM: 64 / 144 / 256 (one token per 32x32 px). The "synthetic" encoder has the real
shapes and layer types with random weights and no RoPE (`ncnn/enc_cf.py`). Its fp16 output matches the
fp32 PyTorch reference (cos 0.9998-1.0000; rel. error 0.4-2% with the fp16-accumulating MatMul).

### Matrix multiply (fp16, 1024x1024x1024 unless noted)

| Kernel | Driver | GFLOPS |
|---|---|---|
| hand-written OpenCL GEMM, B as image (`opencl/gemm.c img 8 16 4`) | Qualcomm OpenCL | **265** |
| same, fp32 accumulation | Qualcomm OpenCL | 176 |
| same, B as buffer | Qualcomm OpenCL | 227 |
| same kernel | rusticl (open) | 114 |
| ncnn 1x1 Convolution (= Linear), 1024x1024x3072 (`ncnn/tiny3.py`) | Turnip | 177 |
| ncnn 1x1 Convolution | Qualcomm Vulkan | 105 |
| ncnn Gemm / InnerProduct, no subgroup ops (`ncnn/tiny2.py`) | Turnip | 38-43 |
| ncnn Gemm with subgroup ops, NCNN_SUBGROUP_SIZE=64 | Turnip | ~1 |
| ncnn Gemm with subgroup ops | Qualcomm Vulkan | 27 |
| ncnn Convolution with dynamic weights (q.kᵀ, `ncnn/dynconv.py`) | Turnip | 24 |
| CPU (ncnn InnerProduct, fp32) | - | 48 |

The GPU reaches 812 MHz under load with the default `simple_ondemand` governor on both drivers, so
the gap between drivers is the shader compiler (ir3 vs Qualcomm), not clocks.

## Open driver (Mesa)

### ncnn on Turnip

- **Subgroup ops break the Gemm layer.** With `use_subgroup_ops` on (the default), ncnn's `gemm_sg`
  shader hangs the GPU at 1024x1024 (`vkWaitForFences failed -4`, VK_ERROR_DEVICE_LOST; the GPU recovers
  by itself) and gives wrong, run-to-run varying results at 128x128. `TU_DEBUG=flushall|syncdraw` and
  seven `IR3_SHADER_DEBUG` options did not change it. ncnn plans the shader for the reported subgroup
  size (128) with large tiles (16x8x8). Forcing 64 (`NCNN_SUBGROUP_SIZE=64`, our patch) makes the
  results correct, but the shader then runs at ~1 GFLOPS and still times out at 1024. Use
  `net.opt.use_subgroup_ops = False`.
- **Write Linear layers as 1x1 convolutions** (channels-first layout). ncnn's Vulkan conv1x1 path runs at
  150-180 GFLOPS. Gemm and InnerProduct run at ~40 GFLOPS, which is no faster than the CPU.
- **ncnn has no Vulkan MatMul layer.** Activation x activation products (attention q.kᵀ and scores.v)
  run on the CPU, with GPU<->CPU copies in every attention layer. `ncnn/patch` adds `MatMul_vulkan` (batched,
  elempack 1, 4x4 outputs per thread, vec4 loads). It cuts an attention block from 386 to 190 ms at 1024
  patches and the whole encoder by ~35%. It accumulates in fp16 (1-2% relative error at 512 px); fp32
  accumulation is a TODO.
- SDPA export is 2x slower than explicit MatMul + Softmax.
- **fp16 overflow in reductions on non-square inputs.** With fp16 arithmetic, the norms' `mean(x²)` (ncnn
  `Reduction`) overflowed for some tokens at 384x512 (NaN everywhere downstream), never at the square sizes
  tried before. `vl/encoder/vl_enc.py` marks Reduction, MatMul, Softmax and BinaryOp layers to compute in fp32
  (ncnn featmask `31=1`) with fp16 storage: no NaN, same speed.

### llama.cpp

llama.cpp (Vulkan) already runs the real LFM3-VL mmproj (`llamacpp/build.sh`); the encoder output is
correct (sensible image descriptions). It is about 2x slower than ncnn with our MatMul. Its OpenCL
backend refuses rusticl ("unsupported GPU 'FD643'"): it only accepts Qualcomm's driver.

### rusticl (OpenCL)

`apt install mesa-opencl-icd`, then run with `RUSTICL_ENABLE=freedreno`. It reports OpenCL 3.0 with fp16,
no subgroups and no `cl_qcom_*` extensions. The hand-written GEMM reaches 114 GFLOPS.

## Qualcomm's proprietary stack (KGSL)

Ubuntu's public "Qualcomm Addon Packages" PPA (`ppa:ubuntu-qcom-iot/qcom-ppa`, series resolute) packages
it for Ubuntu 26.04: `kgsl-dkms` 1.0.11, `qcom-adreno-*` 1.877.5 (OpenCL, Vulkan, EGL/GLES),
`qairt-*` 2.49 (QNN). Nothing has to be ported:

- `kgsl/build-kgsl.sh` builds `msm_kgsl.ko` against the running 7.2.9 kernel tree without changes. All 490
  imported symbols are exported by the kernel. It matches the upstream device tree
  (`qcom,adreno-635.0` = A643, chip id 0x06030500; `qcom,adreno-gmu-635.0`) and uses the `a660_sqe.fw` /
  `a660_gmu.bin` firmware already on the board.
- The GPU has to be left free by msm. Our kernel has `msm.skip_gpu` (msm is built in, so it is a kernel
  command-line option): `kgsl/kgsl-boot-default.sh` adds a systemd-boot entry with it and makes it the
  default. `bootctl set-oneshot` does not work because the firmware exposes no EFI variables.
- Unbinding the GPU from msm at runtime instead (`echo 3d00000.gpu > .../adreno/unbind`) does **not**
  work. msm leaves devres resources on the GMU (`3d6a000.gmu: Resources present before probing`), so
  KGSL cannot probe it; msm_gem WARNs, and the display is gone until a reboot.
- After the reboot, `sudo kgsl/kgsl-load.sh` loads the module, which creates `/dev/kgsl-3d0`.
  `kgsl/fetch-adreno.sh` unpacks the Adreno userspace into a private directory and writes
  `kgsl/adreno-env.sh`, so nothing is installed with apt and Mesa stays untouched. `kgsl/kgsl-revert.sh`
  plus a reboot goes back to the open driver.
- In this mode, **Kodi cannot render**: it needs Mesa GL on the msm GPU, and it would have to move to
  Qualcomm's GLES, which is untested.

What the stack gives (numbers in the tables above):

- **OpenCL** reports "QUALCOMM Adreno(TM) 635" and has subgroups, `cl_qcom_dot_product8`,
  `cl_qcom_ml_ops` and image paths. This was the best result: llama.cpp's Adreno OpenCL backend, and 265
  GFLOPS on the hand-written GEMM.
- **Vulkan** is slower than Turnip for ncnn: conv1x1 runs at 105 vs 177 GFLOPS. The subgroup Gemm is
  correct here, but slow.
- **CLML** (Adreno OpenCL ML SDK v4.1, `opencl/clml_bench.cpp`) is unusable. Every call returns -30
  (CL_INVALID_VALUE). gdb shows that the `libOpenCL_adreno` shim dlopens `libCB.so.1` and fails to
  resolve the ML backend: this `libCB` build exports no `mlb*` symbols. The extension is advertised but
  not shipped.
- **QNN GPU** (`qnn-gpu/qnn_gpu_test.py`, ONNX Runtime 1.27 + onnxruntime-qnn 2.3, `libQnnGpu.so`):
  a conv block takes 225 ms in fp32 (ncnn: 115 ms), and the attention block gives wrong results
  (cos 0.71). With an fp16-exported model, the conv block takes 149 ms but is wrong (cos 0.70), and
  attention gives NaN. It needs KGSL; it does not work on the open driver.

## Files

| Path | What |
|---|---|
| `ncnn/enc_cf.py` | Synthetic LFM3-VL encoder in PyTorch (channels-first, Linear as 1x1 conv), exported with pnnx |
| `ncnn/bench_ncnn.py` | Times an exported encoder on the GPU and compares with the PyTorch output |
| `ncnn/tiny2.py`, `tiny3.py`, `dynconv.py` | Single-layer microbenchmarks (Gemm/InnerProduct, conv1x1, dynamic-weight conv) |
| `ncnn/patch/`, `ncnn/apply_patch.py` | Vulkan MatMul layer + `NCNN_SUBGROUP_SIZE` override for ncnn 20261020 |
| `ncnn/build_wheel.sh` | Builds the patched ncnn wheel (cp314, aarch64) in Docker; ~1 min on an Apple Silicon Mac |
| `kgsl/` | Build KGSL, unpack the Adreno userspace, switch the boot to `msm.skip_gpu=1`, load, revert |
| `opencl/gemm.c` | Hand-written fp16 OpenCL GEMM (buffer / image variants) |
| `opencl/clml_bench.cpp` | CLML GEMM/conv benchmark (needs the SDK header; documents the -30 failure) |
| `qnn-gpu/qnn_gpu_test.py` | Encoder blocks through ONNX Runtime + QNN GPU backend |
| `llamacpp/build.sh` | llama.cpp with Vulkan + OpenCL, used as the reference encoder |

Python for the ncnn tests: `pip install ncnn pnnx torch numpy` (ncnn 1.0.20261020 from PyPI works on
Turnip if subgroup ops are disabled; the patched wheel adds the Vulkan MatMul).

Licences: our code here is MIT like the rest of the repository; `ncnn/patch` is written for ncnn (BSD-3-Clause).
Nothing from Qualcomm is included: the scripts download the KGSL source, the Adreno userspace and the QNN headers
from Ubuntu's public PPA, and the CLML SDK has to be downloaded from Qualcomm.
