from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import os

from .base import GenerationResult
from ..config import GenConfig


@dataclass
class HFLocalBackend:
    """
    HuggingFace 本地 causal LM backend（GPT2 / LLaMA / etc.）
    """
    model_name_or_path: str
    device: str = "cuda:0"
    torch_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    trust_remote_code: bool = False

    name: str = "hf_local"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "HFLocalBackend 需要安装 torch/transformers：pip install torch transformers"
            ) from e

        self._torch = torch
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForCausalLM = AutoModelForCausalLM

        self.tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )

        dtype = None
        if self.torch_dtype != "auto":
            dtype = getattr(torch, self.torch_dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported torch_dtype={self.torch_dtype}")

        self.model = self._AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

        # GPT2 often has no pad_token; align to eos to avoid warnings
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self) -> Any:
        return getattr(self, "tokenizer", None)

    def _apply_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not text or not stop:
            return text
        cut = None
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx >= 0:
                cut = idx if cut is None else min(cut, idx)
        return text if cut is None else text[:cut]

    @staticmethod
    def _strip_prompt_from_full(full_text: str, prompt: str) -> str:
        # best-effort: if full_text startswith prompt, remove it
        if full_text.startswith(prompt):
            return full_text[len(prompt):]
        return full_text

    def generate(self, prompt: str, gen: GenConfig, system_prompt: Optional[str] = None) -> GenerationResult:
        torch = self._torch

        # ---- seed: 兼容旧 transformers：优先全局 seed（稳定），再尝试 generator（新版本支持更好） ----
        if gen.seed is not None:
            seed = int(gen.seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        g = None
        if gen.seed is not None:
            try:
                g = torch.Generator(device=self.device)
                g.manual_seed(int(gen.seed))
            except Exception:
                g = None

        # encode
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        input_len = int(input_ids.shape[-1])
        kw = gen.hf_generate_kwargs()

        if self.tokenizer.eos_token_id is not None:
            kw.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            kw.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        with torch.no_grad():
            # ---- 关键：generator 在旧 transformers 会触发 “not used by the model” ----
            if g is None:
                out_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **kw,
                )
            else:
                try:
                    out_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        generator=g,
                        **kw,
                    )
                except ValueError as e:
                    # 旧版本 transformers 把 generator 当成 model_kwargs 验证失败
                    if "not used by the model" in str(e) and "generator" in str(e):
                        out_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            **kw,
                        )
                    else:
                        raise

        out_ids_0 = out_ids[0]
        comp_ids = out_ids_0[input_len:]
        completion = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
        completion = self._apply_stop(completion, gen.stop)

        text = (prompt + completion) if gen.return_full_text else completion

        meta: Dict[str, Any] = {
            "backend": self.name,
            "model": self.model_name_or_path,
            "device": self.device,
            "gen_config": gen.to_dict(),
            "input_tokens": input_len,
            "output_tokens": int(out_ids_0.shape[-1]),
        }
        return GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta)


    def close(self) -> None:
        # nothing to close
        return
