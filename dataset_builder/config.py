from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class GenConfig:
    # Common
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    num_beams: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    stop: Optional[List[str]] = None
    seed: Optional[int] = 114514

    # API-only (OpenAI compatible)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Output
    return_full_text: bool = True  # if True: return prompt+completion, else only completion

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hf_generate_kwargs(self) -> Dict[str, Any]:
        # Transformers generate() kwargs
        kw: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "min_new_tokens": int(self.min_new_tokens),
            "do_sample": bool(self.do_sample),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "num_beams": int(self.num_beams),
            "repetition_penalty": float(self.repetition_penalty),
        }
        if self.top_k and self.top_k > 0:
            kw["top_k"] = int(self.top_k)
        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            kw["no_repeat_ngram_size"] = int(self.no_repeat_ngram_size)
        return kw

    def openai_kwargs(self) -> Dict[str, Any]:
        # OpenAI-compatible chat/completions kwargs
        kw: Dict[str, Any] = {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "presence_penalty": float(self.presence_penalty),
            "frequency_penalty": float(self.frequency_penalty),
        }
        # OpenAI uses max_tokens for new tokens (not including prompt)
        kw["max_tokens"] = int(self.max_new_tokens)
        if self.stop:
            kw["stop"] = list(self.stop)
        return kw


@dataclass
class BuildConfig:
    dataset_spec: str
    out_jsonl: str

    # which label to use as prompt source (default: human=0)
    prompt_from_label: int = 0

    # token prefix control
    prefix_k_tokens: int = 64
    tokenizer_strategy: str = "auto"  # auto | hf:<name_or_path> | whitespace

    # prompt templating
    prompt_template: str = "{prefix}"
    system_prompt: Optional[str] = None  # only used by chat API backends (optional)

    # generation & attacks
    gen: GenConfig = field(default_factory=GenConfig)
    attacks_config_path: Optional[str] = None

    # sampling / limits
    max_prompts: Optional[int] = None
    sample_seed: int = 114514

    # output structure
    store_original_full: bool = True           # original=[full text], not only prefix
    machine_text_mode: str = "prompt_plus"     # prompt_plus | completion_only
    write_every: int = 50                      # flush频率

    # metadata passthrough
    keep_fields: Optional[Sequence[str]] = None  # None => keep all original fields except huge ones
    attack_dataset_only: bool = False
    only_attack_machine: bool = True      # ✅ 默认只攻击机器文本
    machine_label: int = 1                # 输入数据集中“机器文本”的标签值
    # ✅ NEW: attack-only sampling (None => full)
    sample_k: Optional[int] = None