"""Paths and chat-template rendering shared by the tools.

LFM3_MODEL: Hugging Face model dir (default <repo>/models/LFM3-1.7B)
LFM3_BUILD: NPU build dir (default <repo>/build/npu)
"""
import json, os, re
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("LFM3_MODEL", ROOT / "models" / "LFM3-1.7B"))
BUILD_DIR = Path(os.environ.get("LFM3_BUILD", ROOT / "build" / "npu"))


def load_template(d=MODEL_DIR):
    """The model's own chat template (jinja2 sandbox; `{% generation %}` tags stripped)."""
    d = Path(d)
    p = d / "chat_template.jinja"
    src = p.read_text() if p.exists() else json.loads((d / "tokenizer_config.json").read_text())["chat_template"]
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, separators=separators)

    def raise_exception(m):
        raise ValueError(m)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src), json.loads((d / "tokenizer_config.json").read_text())


class ChatFormat:
    def __init__(self, model_dir=MODEL_DIR):
        self.tmpl, cfg = load_template(model_dir)
        sp = lambda k: cfg.get(k) if isinstance(cfg.get(k), str) else (cfg.get(k) or {}).get("content", "")
        self.bos, self.eos_text = sp("bos_token") or "", sp("eos_token")
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def render(self, messages, tools=None, think=False):
        kw = dict(messages=messages, add_generation_prompt=True, enable_thinking=bool(think),
                  bos_token=self.bos, eos_token=self.eos_text)
        if tools:
            kw["tools"] = tools
        return self.tmpl.render(**kw)

    def encode(self, messages, tools=None, think=False):
        return self.tok.encode(self.render(messages, tools, think), add_special_tokens=False).ids

    def stop_ids(self, extra=()):
        return set(extra) | {i for i in (self.tok.token_to_id("<|im_end|>"), self.tok.token_to_id("<|endoftext|>")) if i is not None}
