# configs

| File | What |
|---|---|
| `lfm3_act_ranges.json` | Per-tensor activation abs-max for LFM3-1.7B (element-wise max of `calibrate_lfm3.py` on short built-in chats and `calibrate2_lfm3.py` on chat data with GPTQ weights; see `merge_ranges.py`). The builder adds a 25% margin. |
| `sens_int8.json` | Output of `sens.py`: `[layer, weight, KL gain, params]` for every matrix of layers 6-23, sorted by KL gain per parameter (on top of int8 layers 0-5 + GPTQ int4). |
| `int8_mats_{100,200,300}M.json` | `[layer, weight]` lists to keep in int8 under a parameter budget (`INT8_MATS=...`). The reference build uses 100M. |

These were produced for `LiquidAI/LFM3-1.7B` with the GPTQ settings in [docs/PIPELINE.md](../docs/PIPELINE.md). They
are model-specific; regenerate them for another model or checkpoint.
