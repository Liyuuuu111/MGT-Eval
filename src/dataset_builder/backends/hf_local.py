from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import os

from .base import GenerationResult
from ..config import GenConfig


@dataclass
class HFLocalBackend:
    """
    HuggingFace local causal LM backend (GPT2 / LLaMA / etc.).
    Optionally uses vLLM for faster generation when enabled.
    """
    model_name_or_path: str
    device: str = "cuda:0"
    torch_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    trust_remote_code: bool = False

    # vLLM (optional, faster local generation)
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: Optional[int] = None
    vllm_dtype: Optional[str] = None
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_trust_remote_code: bool = False
    vllm_disable_log_stats: bool = True
    # vLLM custom all-reduce can fail on some driver/NCCL topologies in TP mode.
    # Auto-enabled (disabled custom kernel) when tensor_parallel_size > 1.
    vllm_disable_custom_all_reduce: bool = False

    name: str = "hf_local"

    _vllm: Any = field(default=None, init=False, repr=False)
    _vllm_failed: bool = field(default=False, init=False, repr=False)

    def _resolve_device(self, raw_device: Optional[str]) -> str:
        """
        Normalize frontend/device YAML values to a valid torch device string.
        Supports "auto" and invalid CUDA indices gracefully.
        """
        torch = self._torch
        dv = str(raw_device or "").strip().lower()
        if dv in ("", "auto"):
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if dv == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if dv.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            if ":" not in dv:
                return "cuda:0"
            try:
                idx = int(dv.split(":", 1)[1])
            except Exception:
                idx = 0
            n_cuda = int(torch.cuda.device_count() or 0)
            if n_cuda <= 0:
                return "cpu"
            if idx < 0 or idx >= n_cuda:
                idx = 0
            return f"cuda:{idx}"

        if dv == "mps":
            has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            return "mps" if has_mps else ("cuda:0" if torch.cuda.is_available() else "cpu")

        if dv == "cpu":
            return "cpu"

        # Last-chance fallback for custom strings accepted by torch.device(...)
        try:
            _ = torch.device(dv)
            return dv
        except Exception:
            return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _normalize_vllm_parallelism(self) -> None:
        """
        Prevent common vLLM multi-GPU deadlocks on heterogeneous/misaligned devices.
        """
        torch = self._torch
        if not bool(self.use_vllm):
            return
        if not torch.cuda.is_available():
            print("[HFLocalBackend] CUDA is unavailable; fallback to HF backend without vLLM.")
            self.use_vllm = False
            return

        n_cuda = int(torch.cuda.device_count() or 0)
        if n_cuda <= 0:
            self.use_vllm = False
            print("[HFLocalBackend] No visible CUDA devices; fallback to HF backend without vLLM.")
            return

        tp_raw = int(self.vllm_tensor_parallel_size or 0)
        tp = n_cuda if tp_raw <= 0 else tp_raw

        if tp > n_cuda:
            print(
                "[HFLocalBackend] vLLM tensor_parallel_size "
                f"{tp} exceeds visible CUDA devices ({n_cuda}); clamp to {n_cuda}."
            )
            tp = n_cuda

        # Multi-GPU vLLM is fragile on heterogeneous GPU types; fail-safe to single GPU.
        if tp > 1:
            names: List[str] = []
            for i in range(tp):
                try:
                    names.append(str(torch.cuda.get_device_name(i)))
                except Exception:
                    names.append(f"cuda:{i}")
            if len(set(names)) > 1:
                print(
                    "[HFLocalBackend] Heterogeneous GPUs detected in visible set "
                    f"{names}. For stability, fallback vLLM tensor_parallel_size to 1."
                )
                tp = 1

        self.vllm_tensor_parallel_size = tp

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
        self.model = None
        self.device = self._resolve_device(self.device)
        self._normalize_vllm_parallelism()

        self.tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )

        # optional vLLM path
        if self.use_vllm and (self._vllm is None) and (not self._vllm_failed):
            try:
                from vllm import LLM  # type: ignore
                dtype_s = self.vllm_dtype
                if not dtype_s or str(dtype_s).strip() == "":
                    dtype_s = "auto" if self.torch_dtype == "auto" else self.torch_dtype
                kwargs = dict(
                    model=self.model_name_or_path,
                    tensor_parallel_size=max(1, int(self.vllm_tensor_parallel_size)),
                    gpu_memory_utilization=float(self.vllm_gpu_memory_utilization),
                    max_model_len=int(self.vllm_max_model_len) if (self.vllm_max_model_len is not None and int(self.vllm_max_model_len) > 0) else None,
                    dtype=dtype_s,
                    trust_remote_code=bool(self.vllm_trust_remote_code or self.trust_remote_code),
                    enforce_eager=bool(self.vllm_enforce_eager),
                    disable_log_stats=bool(self.vllm_disable_log_stats),
                    disable_custom_all_reduce=bool(
                        self.vllm_disable_custom_all_reduce
                        or max(1, int(self.vllm_tensor_parallel_size)) > 1
                    ),
                )
                print(
                    "[HFLocalBackend] Initialize vLLM with "
                    f"device={self.device}, tensor_parallel_size={kwargs['tensor_parallel_size']}, "
                    f"dtype={kwargs['dtype']}, disable_custom_all_reduce={kwargs['disable_custom_all_reduce']}"
                )
                try:
                    self._vllm = LLM(**kwargs)
                except TypeError:
                    # retry with conservative kwargs for older vLLM versions
                    for k in (
                        "trust_remote_code",
                        "enforce_eager",
                        "disable_log_stats",
                        "disable_custom_all_reduce",
                    ):
                        kwargs.pop(k, None)
                    self._vllm = LLM(**kwargs)
                if self._vllm is not None:
                    self.name = "hf_local_vllm"
            except Exception as exc:
                print(f"[HFLocalBackend] vLLM init failed, fallback to HF backend: {exc}")
                self._vllm = None
                self._vllm_failed = True

        # fallback: HF model
        if self._vllm is None:
            self._ensure_hf_model()

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

    def _ensure_hf_model(self) -> None:
        if self.model is not None:
            return
        torch = self._torch
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

    def generate(self, prompt: str, gen: GenConfig, system_prompt: Optional[str] = None) -> GenerationResult:
        torch = self._torch

        # vLLM path (if available)
        if self._vllm is not None:
            try:
                from vllm import SamplingParams  # type: ignore
            except Exception:
                SamplingParams = None
            if SamplingParams is not None:
                do_sample = bool(gen.do_sample)
                temp = float(gen.temperature) if do_sample else 0.0
                top_p = float(gen.top_p) if do_sample else 1.0
                top_k = int(gen.top_k) if (do_sample and gen.top_k and int(gen.top_k) > 0) else -1

                sp_kwargs: Dict[str, Any] = {
                    "max_tokens": int(gen.max_new_tokens),
                    "temperature": float(temp),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "repetition_penalty": float(gen.repetition_penalty),
                    "stop": list(gen.stop) if gen.stop else None,
                    "seed": int(gen.seed) if gen.seed is not None else None,
                }
                if gen.min_new_tokens and int(gen.min_new_tokens) > 0:
                    sp_kwargs["min_tokens"] = int(gen.min_new_tokens)
                if gen.num_beams and int(gen.num_beams) > 1:
                    sp_kwargs["use_beam_search"] = True
                    sp_kwargs["best_of"] = int(gen.num_beams)

                # drop None values
                sp_kwargs = {k: v for k, v in sp_kwargs.items() if v is not None}

                # build SamplingParams with compatibility fallback
                sp = None
                try:
                    sp = SamplingParams(**sp_kwargs)
                except TypeError:
                    # progressively remove optional keys for older vLLM versions
                    for k in ("min_tokens", "repetition_penalty", "stop", "seed", "top_k", "top_p", "temperature", "best_of", "use_beam_search"):
                        if k in sp_kwargs:
                            sp_kwargs.pop(k, None)
                            try:
                                sp = SamplingParams(**sp_kwargs)
                                break
                            except TypeError:
                                sp = None
                                continue

                if sp is not None:
                    try:
                        reqs = self._vllm.generate([prompt], sp)
                    except Exception:
                        # fallback to HF path if vLLM generation fails
                        self._vllm_failed = True
                        self._vllm = None
                        self.name = "hf_local"
                        reqs = None
                    if reqs is not None:
                        out_text = ""
                        out_token_ids = None
                        if reqs and getattr(reqs[0], "outputs", None):
                            out0 = reqs[0].outputs[0]
                            out_text = (out0.text or "")
                            out_token_ids = getattr(out0, "token_ids", None)
                        completion = self._apply_stop(out_text, gen.stop)

                        # token accounting
                        try:
                            input_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                        except Exception:
                            input_len = len((prompt or "").split())
                        if out_token_ids is not None:
                            output_tokens = int(input_len + len(out_token_ids))
                        else:
                            try:
                                output_tokens = int(input_len + len(self.tokenizer.encode(completion, add_special_tokens=False)))
                            except Exception:
                                output_tokens = int(input_len + len((completion or "").split()))

                        text = (prompt + completion) if gen.return_full_text else completion
                        meta: Dict[str, Any] = {
                            "backend": self.name,
                            "model": self.model_name_or_path,
                            "device": self.device,
                            "gen_config": gen.to_dict(),
                            "input_tokens": int(input_len),
                            "output_tokens": int(output_tokens),
                        }
                        return GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta)

        # ---- seed:  transformers： seed（）， generator（） ----
        self._ensure_hf_model()
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
            # ---- ：generator  transformers  “not used by the model” ----
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
                    # transformers  generator  model_kwargs
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

    def generate_batch(
        self,
        prompts: List[str],
        gen: GenConfig,
        system_prompt: Optional[str] = None,
    ) -> List[GenerationResult]:
        if not prompts:
            return []

        # vLLM path (if available)
        if self._vllm is not None:
            try:
                from vllm import SamplingParams  # type: ignore
            except Exception:
                SamplingParams = None
            if SamplingParams is not None:
                do_sample = bool(gen.do_sample)
                temp = float(gen.temperature) if do_sample else 0.0
                top_p = float(gen.top_p) if do_sample else 1.0
                top_k = int(gen.top_k) if (do_sample and gen.top_k and int(gen.top_k) > 0) else -1

                sp_kwargs: Dict[str, Any] = {
                    "max_tokens": int(gen.max_new_tokens),
                    "temperature": float(temp),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "repetition_penalty": float(gen.repetition_penalty),
                    "stop": list(gen.stop) if gen.stop else None,
                    "seed": int(gen.seed) if gen.seed is not None else None,
                }
                if gen.min_new_tokens and int(gen.min_new_tokens) > 0:
                    sp_kwargs["min_tokens"] = int(gen.min_new_tokens)
                if gen.num_beams and int(gen.num_beams) > 1:
                    sp_kwargs["use_beam_search"] = True
                    sp_kwargs["best_of"] = int(gen.num_beams)

                sp_kwargs = {k: v for k, v in sp_kwargs.items() if v is not None}

                sp = None
                try:
                    sp = SamplingParams(**sp_kwargs)
                except TypeError:
                    for k in ("min_tokens", "repetition_penalty", "stop", "seed", "top_k", "top_p", "temperature", "best_of", "use_beam_search"):
                        if k in sp_kwargs:
                            sp_kwargs.pop(k, None)
                            try:
                                sp = SamplingParams(**sp_kwargs)
                                break
                            except TypeError:
                                sp = None
                                continue

                if sp is not None:
                    try:
                        reqs = self._vllm.generate(list(prompts), sp)
                    except Exception:
                        self._vllm_failed = True
                        self._vllm = None
                        self.name = "hf_local"
                        reqs = None

                    if reqs is not None:
                        results: List[GenerationResult] = []
                        for prompt, req in zip(prompts, reqs):
                            out_text = ""
                            out_token_ids = None
                            if req is not None and getattr(req, "outputs", None):
                                out0 = req.outputs[0]
                                out_text = (out0.text or "")
                                out_token_ids = getattr(out0, "token_ids", None)
                            completion = self._apply_stop(out_text, gen.stop)

                            try:
                                input_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                            except Exception:
                                input_len = len((prompt or "").split())
                            if out_token_ids is not None:
                                output_tokens = int(input_len + len(out_token_ids))
                            else:
                                try:
                                    output_tokens = int(input_len + len(self.tokenizer.encode(completion, add_special_tokens=False)))
                                except Exception:
                                    output_tokens = int(input_len + len((completion or "").split()))

                            text = (prompt + completion) if gen.return_full_text else completion
                            meta: Dict[str, Any] = {
                                "backend": self.name,
                                "model": self.model_name_or_path,
                                "device": self.device,
                                "gen_config": gen.to_dict(),
                                "input_tokens": int(input_len),
                                "output_tokens": int(output_tokens),
                            }
                            results.append(GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta))
                        return results

        # ---- HF batch fallback ----
        self._ensure_hf_model()
        torch = self._torch
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

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        input_lens = attn.sum(dim=1).tolist() if attn is not None else [int(x.shape[-1]) for x in input_ids]
        kw = gen.hf_generate_kwargs()
        if self.tokenizer.eos_token_id is not None:
            kw.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            kw.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        with torch.no_grad():
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
                    if "not used by the model" in str(e) and "generator" in str(e):
                        out_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            **kw,
                        )
                    else:
                        raise

        results = []
        for i, prompt in enumerate(prompts):
            input_len = int(input_lens[i])
            out_ids_i = out_ids[i]
            comp_ids = out_ids_i[input_len:]
            completion = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            completion = self._apply_stop(completion, gen.stop)

            text = (prompt + completion) if gen.return_full_text else completion
            meta = {
                "backend": self.name,
                "model": self.model_name_or_path,
                "device": self.device,
                "gen_config": gen.to_dict(),
                "input_tokens": int(input_len),
                "output_tokens": int(out_ids_i.shape[-1]),
            }
            results.append(GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta))
        return results


    def close(self) -> None:
        # nothing to close
        return
