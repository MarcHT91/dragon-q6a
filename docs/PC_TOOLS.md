# Using an x86-64 PC: GPU work and offline NPU compilation

An x86-64 PC with an NVIDIA GPU speeds up everything that is not running the model on the
NPU, and it can compile the NPU graphs too. Tested on Windows 11 with an RTX 3060 (12 GB) and
Python 3.12. The same should work on Linux x86-64 (`onnxruntime-qnn` publishes wheels for
both, but only Windows was tested).

| Task | Board CPU | PC (RTX 3060) |
|---|---|---|
| GPTQ, 24 layers | not practical | ~10 min |
| fp32 reference (eval logits, bench with `lfm3-ref`) | 0.2 tok/s, ~7 GB RAM | ~22 tok/s |
| Per-matrix sensitivity (`sens.py`) | not practical | ~50 min |
| Build + compile all NPU graphs | ~13 min | ~6 min, no DSP use on the board |

## 1. Environment

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements-pc.txt
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Pin `onnxruntime==1.27.0`.** `onnxruntime-qnn` 2.3.0 is a plugin execution provider
built for that version. With onnxruntime 1.31 it fails with
`OrtEp has invalid ort_version_supported=0 (expected at least 22)`.

Copy the repository (or just `npu/` and `configs/`) and the model to the PC, and download
wikitext there as in [PIPELINE.md](PIPELINE.md).

On Windows, set `PYTHONUTF8=1`; otherwise output files containing characters such as `→`
fail with `UnicodeEncodeError` (the console code page is used by default).

## 2. GPU jobs

```bat
set PYTHONUTF8=1
.venv\Scripts\python npu\gen_calib_lfm3.py models\LFM3-1.7B build\calib_chat.jsonl
set CHAT=build\calib_chat.jsonl
.venv\Scripts\python npu\gptq_lfm3.py models\LFM3-1.7B data\wikitext\wikitext-2-raw-v1\train-00000-of-00001.parquet build\gptq 128 512
set REF_DEVICE=cuda
.venv\Scripts\python npu\make_eval_ref.py
.venv\Scripts\python npu\sens.py models\LFM3-1.7B build\gptq data\wikitext\wikitext-2-raw-v1\train-00000-of-00001.parquet build\calib_chat.jsonl build\sens
```

Simulating NPU weights on tasks before compiling anything (weights quantized exactly as the
builder does, activations fp32). This reproduced the NPU's task failures one for one:

```bat
set REF_DEVICE=cuda
set QW_GPTQ=build\gptq
set QW_INT8_LAYERS=0,1,2,3,4,5
set QW_INT8_MATS=configs\int8_mats_100M.json
.venv\Scripts\python npu\bench.py lfm3-ref build\sim_hard.jsonl --hard
```

Copy `build/gptq/` and `build/eval/` to the board afterwards (`scp -r`).

## 3. Offline NPU compilation

`npu_common.npu_session()` detects an x86-64 host (or `QNN_OFFLINE=1`) and then only
compiles: it writes the HTP v68 context (`X_ctx.onnx` + `X_ctx_qnn.bin`) through QNN's
HtpPrepare and returns `(None, seconds)`. The build scripts are the same as on the board:

```bat
set PYTHONUTF8=1
set INT8_LAYERS=0,1,2,3,4,5
set INT8_MATS=configs\int8_mats_100M.json
.venv\Scripts\python npu\build_all_lfm3.py models\LFM3-1.7B configs\lfm3_act_ranges.json build\gptq build\npu 4096 6
.venv\Scripts\python npu\build_prefill_lfm3.py models\LFM3-1.7B configs\lfm3_act_ranges.json build\gptq build\npu 16
```

To recompile existing ONNX files only: `npu\pc_compile.py a.onnx b.onnx ...`.

Target options (defaults in `npu_common.py`, override with environment variables):

| Option | Value for the QCS6490 | |
|---|---|---|
| `htp_arch` | 68 | |
| `SOC_MODEL` | 35 | |
| `VTCM_MB` | **2** | 4 or 8 compile, but the board refuses the context (`QNN_CONTEXT_ERROR_CREATE_FROM_BINARY`) |

The PC must use the same QNN version as the board, which comes with `onnxruntime-qnn`
(2.3.0 → QNN 2.47.0 on both). Contexts compiled this way gave bit-identical logits and the
same evaluation results as contexts compiled on the board.

Copy to the board only what the runtime needs (contexts, manifest, embedding; ~2.6 GB for
the 4096-context build):

```bash
mkdir -p build/npu
scp "pc:path/build/npu/*_ctx*" "pc:path/build/npu/manifest.json" "pc:path/build/npu/embed_f16.npy" build/npu/
.venv/bin/python npu/validate_build.py build/npu
```

Besides speed, compiling on the PC avoids the DSP memory leak that each compile session
causes on the board ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

## Tips for a Windows PC reached over SSH

- The default OpenSSH shell on Windows is `cmd.exe`. Long commands with nested quotes break
  easily: write a `.bat`, copy it, run it and read its log.
- When Python output is redirected by `cmd`, logs may come back as UTF-16; on the Linux
  side, `tr -d '\000'` makes them readable.
- For key login of an administrator account, the key goes in
  `C:\ProgramData\ssh\administrators_authorized_keys` (permissions: Administrators and
  SYSTEM only), not in the user's `.ssh\authorized_keys`.
