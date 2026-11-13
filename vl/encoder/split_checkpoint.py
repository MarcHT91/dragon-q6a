"""Split a Hugging Face LFM3-VL checkpoint (LiquidAI/LFM3-VL-2B, LiquidAI/d1-2B) into its two halves:
  OUT/vision.safetensors   vision_tower + multi_modal_projector (for vl_enc.py: VL_WEIGHTS)
  OUT/lm/                  the language model as a plain LFM3 checkpoint (config = text_config, keys model.*,
                           tokenizer files copied), for the NPU pipeline (npu/npu/vl_pipeline.bat, M=...)
usage: python split_checkpoint.py HF_DIR OUT_DIR"""
import json, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "lm").mkdir(parents=True, exist_ok=True)
idx = src / "model.safetensors.index.json"
files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
vision, lm = {}, {}
for fn in files:
    f = safe_open(str(src / fn), framework="pt")
    for k in f.keys():
        if k.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
            vision[k] = f.get_tensor(k)
        elif k.startswith("model.language_model."):
            lm["model." + k[len("model.language_model."):]] = f.get_tensor(k)
save_file(vision, str(out / "vision.safetensors"))
save_file(lm, str(out / "lm" / "model.safetensors"))
cfg = json.loads((src / "config.json").read_text())
(out / "lm" / "config.json").write_text(json.dumps(cfg.get("text_config", cfg), indent=1))
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if (src / f).exists():
        shutil.copy(src / f, out / "lm" / f)
print(f"{len(vision)} vision tensors -> {out / 'vision.safetensors'}, {len(lm)} language-model tensors -> {out / 'lm'}")
