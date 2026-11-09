# Results

Radxa Dragon Q6A, Ubuntu 26.04.1, kernel 7.2.9-edge-qcs6490, onnxruntime 1.27.0 +
onnxruntime-qnn 2.3.0. Reference build: GPTQ int4, int8 layers 0-5, 26 int8 matrices
(`configs/int8_mats_100M.json`), int8 LM head, chunked prefill C=16.

## Speed

| Context | Decode | Notes |
|---|---|---|
| 1024 | 9.4 tok/s | int4 everywhere (no int8 layers): 10.5 tok/s |
| 2048 | 8.9 tok/s | |
| 4096 | 7.8 tok/s | |

Prefill (16-token chunks): ~130 tok/s on prompts of a few hundred tokens (1024 context);
a 2.8k-token prompt takes 33 s at 4096 context (~85 tok/s). Reusing a saved model state
for a repeated prompt prefix (system prompt, tool schemas) avoids most of that.
Loading the compiled contexts: 7-9 s.

## Quality vs fp32 (teacher-forced, `validate_build.py`)

| Build | top-1 | KL |
|---|---|---|
| GPTQ int4 everywhere (wikitext-only calibration) | 0.763 | 0.48 |
| GPTQ int4 everywhere (wikitext + chat calibration) | 0.821 | 0.294 |
| + int8 layers 0-5 | 0.841 | 0.188 |
| + 26 int8 matrices (reference build) | 0.841 | 0.156 |

## Task benchmark (`bench.py`, greedy, no thinking)

Basic set, 36 tasks with automatic checks:

| | tool | json | instr | know | math | total |
|---|---|---|---|---|---|---|
| GPTQ int4 everywhere | 10/12 | 6/6 | 6/6 | 6/6 | 6/6 | 34/36 |
| GPTQ int4 everywhere, thinking on | 12/12 | 5/6 | 6/6 | 6/6 | 6/6 | 35/36 |
| reference build | 12/12 | 6/6 | 6/6 | 6/6 | 6/6 | **36/36** |

The two tool failures of the int4-only build were unneeded tool calls (see
[QUANTIZATION.md](QUANTIZATION.md)). Thinking mode avoided them but made answers 3-5×
longer.

Hard set, 28 tasks (parallel and chained tool calls, tool errors, long-email extraction,
multi-step math, trick questions):

| | tool | json | math | know | instr | total |
|---|---|---|---|---|---|---|
| reference build (NPU) | 7/8 | 3/4 | 5/7 | 4/5 | 3/4 | **22/28** |

The fp32 model, rerun on the 6 failed tasks, passes 4 of them (reporting a tool error,
applying a correction in an order, the "5 machines" puzzle, two sentences ending in "!"),
so it would score about 26/28. The other two (minutes between two times, reversing
"stressed") are limits of the 1.7B model itself.
