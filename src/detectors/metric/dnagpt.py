# mgt_eval/detectors/metric/dnagpt.py
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from ..base import DetectorBase
from ..registry import register

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None
    SamplingParams = None

HF_TOKEN = os.environ.get("HF_TOKEN", None)
_PUBMED_SEPARATOR_ENV = os.environ.get("MGT_EVAL_PUBMED_SEPARATOR", None)


def _select_device(user_device: Optional[str]) -> str:
    if user_device:
        d = user_device.strip().lower()
        if d.startswith("cpu"):
            return "cpu"
        if d.startswith("cuda"):
            return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _torch_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32


@register("dnagpt")
class DNAGPTDetector(DetectorBase):
    CITATION_TITLE = "DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text"
    CITATION_AUTHORS = "Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen"
    CITATION_LINK = "https://arxiv.org/abs/2305.17359"

    def __init__(
        self,
        score_model: str = "gpt2",
        dataset: str = "hc3",
        dataset_name: Optional[str] = None,
        truncate_ratio: float = 0.5,
        regen_number: int = 10,
        batch_size: int = 10,
        do_top_k: bool = False,
        top_k: int = 40,
        do_top_p: bool = True,
        top_p: float = 0.96,
        temperature: float = 1.0,
        use_bfloat16: bool = True,
        min_length_non_pubmed: int = 150,
        min_length_pubmed: int = 50,
        max_length: int = 200,
        device: Optional[str] = None,
        name: Optional[str] = None,
        max_sampling_tries: int = 5,
        detector_type: Optional[str] = "Metric-based",
        # ★ ： token ，
        min_tokens: int = 20,
        # ★ ：（）
        regen_text_batch_size: int = 1,
        # vLLM （）
        use_vllm: bool = False,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        vllm_dtype: Optional[str] = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_enforce_eager: bool = False,
        **kwargs: Any,
    ):
        effective_dataset = dataset_name if (dataset_name is not None) else dataset

        super().__init__(
            score_model=score_model,
            dataset=effective_dataset,
            truncate_ratio=truncate_ratio,
            regen_number=regen_number,
            batch_size=batch_size,
            do_top_k=do_top_k,
            top_k=top_k,
            do_top_p=do_top_p,
            top_p=top_p,
            temperature=temperature,
            use_bfloat16=use_bfloat16,
            min_length_non_pubmed=min_length_non_pubmed,
            min_length_pubmed=min_length_pubmed,
            max_length=max_length,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )

        self.score_model = score_model
        self.dataset = str(effective_dataset)
        self.truncate_ratio = float(truncate_ratio)
        self.regen_number = int(regen_number)
        self.batch_size = int(batch_size)
        self.do_top_k = bool(do_top_k)
        self.top_k = int(top_k)
        self.do_top_p = bool(do_top_p)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_sampling_tries = int(max_sampling_tries)

        self.min_length_non_pubmed = int(min_length_non_pubmed)
        self.min_length_pubmed = int(min_length_pubmed)
        self.max_length = int(max_length)

        self.min_tokens = int(min_tokens)  # ★
        self.regen_text_batch_size = int(regen_text_batch_size) if int(regen_text_batch_size) > 0 else 1

        self.user_device = device
        self.device = "cpu"
        self.dtype = _torch_dtype(use_bf16=use_bfloat16)

        # vLLM
        self.use_vllm = bool(use_vllm)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        self.vllm_max_model_len = int(vllm_max_model_len) if vllm_max_model_len is not None else None
        self.vllm_dtype = str(vllm_dtype) if vllm_dtype is not None else None
        self.vllm_tensor_parallel_size = int(vllm_tensor_parallel_size)
        self.vllm_enforce_eager = bool(vllm_enforce_eager)
        self._vllm = None
        self._vllm_failed = False

        self.name = name or f"DNAGPT[{os.path.basename(self.score_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._tok = None
        self._model = None
        self.is_loaded = False
        self.model_max_positions: Optional[int] = None

    # =====  =====
    def load(self):
        self.device = _select_device(self.user_device)
        self.dtype = _torch_dtype(self.use_bfloat16)

        # Prefer new `dtype` arg; fallback to deprecated `torch_dtype` for older transformers
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.score_model,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map={"": self.device},
                token=HF_TOKEN,
            ).eval()
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.score_model,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map={"": self.device},
                token=HF_TOKEN,
            ).eval()

        self._tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token
        # Decoder-only models should use left padding to avoid generation issues.
        try:
            if getattr(self._model.config, "is_encoder_decoder", False) is False:
                self._tok.padding_side = "left"
        except Exception:
            pass
        self._model.config.pad_token_id = self._tok.pad_token_id
        self.model_max_positions = int(getattr(self._model.config, "max_position_embeddings", 2048))

        # ：vLLM （ HF）
        if self.use_vllm and (LLM is not None) and (not self._vllm_failed):
            try:
                self._vllm = LLM(
                    model=self.score_model,
                    tensor_parallel_size=max(1, self.vllm_tensor_parallel_size),
                    gpu_memory_utilization=float(self.vllm_gpu_memory_utilization),
                    max_model_len=self.vllm_max_model_len,
                    dtype=self.vllm_dtype or "auto",
                    trust_remote_code=True,
                    enforce_eager=bool(self.vllm_enforce_eager),
                )
            except Exception as e:
                self._vllm = None
                self._vllm_failed = True
                if os.environ.get("DNAGPT_DEBUG"):
                    print(f"[DNAGPT] vLLM init failed, fallback to HF generation: {e}")
        super().load()

    # =====  =====
    @torch.inference_mode()
    def _tokenize(self, texts: List[str]):
        safe_texts = []
        for t in texts:
            s = (t or "").strip()
            safe_texts.append(s if len(s) > 0 else " ")

        max_input_len = int(min(self.max_length, self.model_max_positions or self.max_length))
        enc = self._tok(
            safe_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )

        # ★ ： BatchEncoding
        if not isinstance(enc, BatchEncoding):
            enc = BatchEncoding(dict(enc), tensor_type="pt")

        if enc["input_ids"].shape[1] == 0:
            bos = getattr(self._tok, "bos_token_id", None) or getattr(self._tok, "eos_token_id", 0)
            B = len(safe_texts)
            enc = BatchEncoding({
                "input_ids": torch.full((B, 1), bos, dtype=torch.long),
                "attention_mask": torch.ones((B, 1), dtype=torch.long),
            }, tensor_type="pt")

        return enc.to(self.device)

    def _decode_prompts_from_encoded(self, enc: BatchEncoding) -> List[str]:
        """
        Decode prompts from a tokenized BatchEncoding, trimming padding by attention_mask.
        This ensures vLLM sees the exact same (truncated) prompt content as HF generation.
        """
        if self._tok is None:
            return []
        ids = enc["input_ids"].detach().cpu()
        if "attention_mask" in enc:
            lens = enc["attention_mask"].detach().cpu().sum(dim=1).tolist()
        else:
            lens = [ids.shape[1]] * ids.shape[0]
        out: List[str] = []
        for i, l in enumerate(lens):
            out.append(self._tok.decode(ids[i, :int(l)], skip_special_tokens=True))
        return out

    def _build_prompts_and_enc(self, texts: List[str]) -> Tuple[List[str], BatchEncoding]:
        """
        Build prompt strings and tokenized inputs with the same truncation logic.
        Returns (prompts_list, encoded_inputs).
        """
        import math

        if self.dataset == "pubmed":
            pubmed_sep = " Answer:"
            proc_texts = []
            for t in texts:
                pos = t.find(pubmed_sep)
                if pos != -1:
                    proc_texts.append(t[: pos + len(pubmed_sep)])
                else:
                    toks = t.split()
                    keep = max(1, math.ceil(len(toks) * self.truncate_ratio))
                    proc_texts.append(" ".join(toks[:keep]) if keep > 0 else " ")
            enc = self._tokenize(proc_texts)
            return proc_texts, enc

        toks = [t.split() for t in texts]
        trunc = [" ".join(ts[: max(1, math.ceil(len(ts) * self.truncate_ratio))]) for ts in toks]
        enc = self._tokenize(trunc)
        return trunc, enc

    def _build_vllm_inputs(self, enc: BatchEncoding) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Build vLLM prompt_token_ids inputs from an encoded batch, plus decoded prompt strings.
        """
        if self._tok is None:
            return [], []
        ids = enc["input_ids"].detach().cpu()
        if "attention_mask" in enc:
            lens = enc["attention_mask"].detach().cpu().sum(dim=1).tolist()
        else:
            lens = [ids.shape[1]] * ids.shape[0]
        prompt_texts: List[str] = []
        vllm_prompts: List[Dict[str, Any]] = []
        for i, l in enumerate(lens):
            toks = ids[i, :int(l)].tolist()
            vllm_prompts.append({"prompt_token_ids": toks})
            prompt_texts.append(self._tok.decode(toks, skip_special_tokens=True))
        return prompt_texts, vllm_prompts

    @torch.inference_mode()
    def _token_len(self, text: str) -> int:
        """返回单条文本（按当前 tokenizer 与长度上限）的有效 token 数。"""
        enc = self._tokenize([text])
        if "attention_mask" in enc:
            return int(enc["attention_mask"][0].sum().item())
        return int(enc["input_ids"].shape[1])

    @staticmethod
    def _get_likelihood(logits: torch.Tensor, labels: torch.Tensor, pad_index: int) -> torch.Tensor:
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        vocab_size = lprobs.size(-1)
        labels = labels.clamp(min=0, max=vocab_size - 1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        mask = labels != pad_index
        log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
        return log_likelihood.squeeze(-1)

    def _log_prob_single(self, text: str) -> torch.Tensor:
        tokenized = self._tokenize([text])
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = self._model(**tokenized).logits[:, :-1]
            return self._get_likelihood(logits, labels, self._tok.pad_token_id)

    @torch.inference_mode()
    def _log_prob_batch(self, texts: List[str]) -> torch.Tensor:
        out: List[torch.Tensor] = []
        B = self.batch_size
        for i in range(0, len(texts), B):
            chunk = texts[i:i + B]
            tokenized = self._tokenize(chunk)
            labels = tokenized.input_ids[:, 1:]
            logits = self._model(**tokenized).logits[:, :-1]
            lp = self._get_likelihood(logits, labels, self._tok.pad_token_id)
            out.append(lp)
        return torch.cat(out, dim=0)

    # ——  ——
    @staticmethod
    def _trim_to_shorter_length(texta: str, textb: str) -> Tuple[str, str]:
        a_words = texta.split(" ")
        b_words = textb.split(" ")
        shorter = min(len(a_words), len(b_words))
        return " ".join(a_words[:shorter]), " ".join(b_words[:shorter])

    @staticmethod
    def _truncate_to_substring(text: str, substring: str, idx_occurrence: int) -> str:
        assert idx_occurrence > 0, "idx_occurrence must be > 0"
        idx = -1
        for _ in range(idx_occurrence):
            idx = text.find(substring, idx + 1)
            if idx == -1:
                return text
        return text[:idx]

    @torch.inference_mode()
    def _sample_from_model(
        self,
        texts: List[str],
        min_words: int,
        truncate_ratio: float,
        max_tries: Optional[int] = None,
    ) -> Tuple[List[str], bool]:
        n = len(texts)
        if n == 0:
            return [], True

        self._model.eval()
        decoded = ["" for _ in range(n)]
        ok_mask = [False for _ in range(n)]
        tries = 0
        limit = self.max_sampling_tries if max_tries is None else int(max_tries)

        while (not all(ok_mask)) and (tries < limit):
            if tries != 0:
                print(f"\nmin words: {sum(ok_mask)}/{n} done, regenerating (try {tries})")

            # only regenerate unfinished samples
            idx_map = [i for i, ok in enumerate(ok_mask) if not ok]
            cur_texts = [texts[i] for i in idx_map]

            prompts_list, all_encoded = self._build_prompts_and_enc(cur_texts)
            use_vllm = self._vllm is not None and (SamplingParams is not None)
            vllm_prompts: Optional[List[Dict[str, Any]]] = None

            if use_vllm:
                try:
                    decoded_prompts, vllm_prompts = self._build_vllm_inputs(all_encoded)
                    if decoded_prompts:
                        prompts_list = decoded_prompts
                except Exception:
                    vllm_prompts = None

            sampling_kwargs: Dict[str, Any] = {"temperature": self.temperature, "do_sample": True}
            if self.do_top_p:
                sampling_kwargs["top_p"] = self.top_p
            elif self.do_top_k:
                sampling_kwargs["top_k"] = self.top_k

            # （ attention_mask） vLLM  per-prompt
            if isinstance(all_encoded, dict) and "attention_mask" in all_encoded:
                input_lens = all_encoded["attention_mask"].sum(dim=1).tolist()
            else:
                input_lens = None

            input_len = int(all_encoded["input_ids"].shape[1])

            # ★ ： 0， 1-token prompt
            if input_len == 0:
                bos = getattr(self._tok, "bos_token_id", None) or getattr(self._tok, "eos_token_id", 0)
                B = all_encoded["input_ids"].shape[0] if isinstance(all_encoded, dict) else len(texts)
                all_encoded = {
                    "input_ids": torch.full((B, 1), bos, dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones((B, 1), dtype=torch.long, device=self.device),
                }
                input_len = 1
                input_lens = [1 for _ in range(B)]

            target_total_max = int(self.max_length)
            min_total = self.min_length_pubmed if self.dataset in ["pubmed"] else self.min_length_non_pubmed

            def _calc_min_max(il: int) -> Tuple[int, int]:
                mx = max(1, target_total_max - int(il))
                mn = max(1, int(min_total) - int(il))
                if mn > mx:
                    mn = mx
                return mn, mx

            if input_lens is not None:
                min_new_tokens_list = []
                max_new_tokens_list = []
                for il in input_lens:
                    mn, mx = _calc_min_max(il)
                    min_new_tokens_list.append(mn)
                    max_new_tokens_list.append(mx)
                # HF
                min_new_tokens = int(min(min_new_tokens_list) if min_new_tokens_list else 1)
                max_new_tokens = int(max(max_new_tokens_list) if max_new_tokens_list else 1)
            else:
                min_new_tokens, max_new_tokens = _calc_min_max(input_len)

            # vLLM generation (optional)
            if use_vllm:
                try:
                    stop_ids = None
                    if self._tok is not None and getattr(self._tok, "eos_token_id", None) is not None:
                        stop_ids = [int(self._tok.eos_token_id)]
                    # Prefer per-prompt sampling params (different max/min tokens per prompt)
                    if input_lens is not None:
                        sp_list = []
                        for mn, mx in zip(min_new_tokens_list, max_new_tokens_list):
                            sp_list.append(
                                SamplingParams(
                                    temperature=float(self.temperature),
                                    top_p=float(self.top_p) if self.do_top_p else 1.0,
                                    top_k=int(self.top_k) if self.do_top_k else -1,
                                    min_tokens=int(mn),
                                    max_tokens=int(mx),
                                    **({"stop_token_ids": stop_ids} if stop_ids else {}),
                                )
                            )
                        outs = self._vllm.generate(vllm_prompts or prompts_list, sp_list)
                    else:
                        sp = SamplingParams(
                            temperature=float(self.temperature),
                            top_p=float(self.top_p) if self.do_top_p else 1.0,
                            top_k=int(self.top_k) if self.do_top_k else -1,
                            min_tokens=int(min_new_tokens),
                            max_tokens=int(max_new_tokens),
                            **({"stop_token_ids": stop_ids} if stop_ids else {}),
                        )
                        outs = self._vllm.generate(vllm_prompts or prompts_list, sp)
                    decoded_cur = []
                    for p, out in zip(prompts_list, outs):
                        if out.outputs:
                            gen = out.outputs[0].text
                        else:
                            gen = ""
                        decoded_cur.append(p + gen)
                except Exception as e:
                    if os.environ.get("DNAGPT_DEBUG"):
                        print(f"[DNAGPT] vLLM generate failed, fallback to HF: {e}")
                    # vLLM  min_tokens； min_tokens
                    try:
                        stop_ids = None
                        if self._tok is not None and getattr(self._tok, "eos_token_id", None) is not None:
                            stop_ids = [int(self._tok.eos_token_id)]
                        if input_lens is not None:
                            sp_list = []
                            for mx in max_new_tokens_list:
                                sp_list.append(
                                    SamplingParams(
                                        temperature=float(self.temperature),
                                        top_p=float(self.top_p) if self.do_top_p else 1.0,
                                        top_k=int(self.top_k) if self.do_top_k else -1,
                                        max_tokens=int(mx),
                                        **({"stop_token_ids": stop_ids} if stop_ids else {}),
                                    )
                                )
                            outs = self._vllm.generate(vllm_prompts or prompts_list, sp_list)
                        else:
                            sp = SamplingParams(
                                temperature=float(self.temperature),
                                top_p=float(self.top_p) if self.do_top_p else 1.0,
                                top_k=int(self.top_k) if self.do_top_k else -1,
                                max_tokens=int(max_new_tokens),
                                **({"stop_token_ids": stop_ids} if stop_ids else {}),
                            )
                            outs = self._vllm.generate(vllm_prompts or prompts_list, sp)
                        decoded_cur = []
                        for p, out in zip(prompts_list, outs):
                            if out.outputs:
                                gen = out.outputs[0].text
                            else:
                                gen = ""
                            decoded_cur.append(p + gen)
                    except Exception:
                        decoded_cur = None
                    if decoded_cur is None:
                        outputs = self._model.generate(
                            **all_encoded,
                            min_new_tokens=min_new_tokens,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=self._tok.pad_token_id,
                            eos_token_id=self._tok.eos_token_id,
                            **sampling_kwargs,
                        )
                        decoded_cur = self._tok.batch_decode(outputs, skip_special_tokens=True)
            else:
                outputs = self._model.generate(
                    **all_encoded,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tok.pad_token_id,
                    eos_token_id=self._tok.eos_token_id,
                    **sampling_kwargs,
                )
                decoded_cur = self._tok.batch_decode(outputs, skip_special_tokens=True)

            # update per-sample results; keep retrying only those that are short
            for local_i, global_i in enumerate(idx_map):
                s = decoded_cur[local_i]
                if len(s.split()) >= min_words:
                    decoded[global_i] = s
                    ok_mask[global_i] = True
            tries += 1
        if not all(ok_mask):
            print(f"[DNAGPT] give up after {limit} tries (min words reached: {sum(ok_mask)}/{n}).")
        return decoded, all(ok_mask)

    def _regen_samples_for_one(self, text: str) -> List[str]:
        replicas = [text] * self.regen_number
        out: List[str] = []
        B = self.batch_size
        for b in range(0, self.regen_number, B):
            cur_b = min(B, self.regen_number - b)
            original_texts = replicas[b:b + cur_b]
            sampled_texts, ok = self._sample_from_model(
                original_texts,
                min_words=30 if self.dataset in ["pubmed"] else 55,
                truncate_ratio=self.truncate_ratio,
                max_tries=self.max_sampling_tries,
            )
            if not ok:
                return []
            for o, s in zip(original_texts, sampled_texts):
                if self.dataset == "pubmed":
                    s = self._truncate_to_substring(s, "Question:", 2)
                    if _PUBMED_SEPARATOR_ENV:
                        o = o.replace(_PUBMED_SEPARATOR_ENV, " ")
                o_trim, s_trim = self._trim_to_shorter_length(o, s)
                out.append(s_trim)
        return out

    def _regen_samples_for_many(self, texts: List[str]) -> List[List[str]]:
        """
        为多条原文一次性生成重写（跨样本批量）。
        返回：List[List[str]]，与 texts 一一对齐，每条包含 regen_number 条重写。
        """
        n = len(texts)
        if n == 0:
            return []
        out: List[List[str]] = [[] for _ in range(n)]

        # vLLM ： regen_number * n （“batch”）
        if self._vllm is not None and (SamplingParams is not None):
            batch_texts: List[str] = []
            for t in texts:
                batch_texts.extend([t] * self.regen_number)

            sampled_texts, ok = self._sample_from_model(
                batch_texts,
                min_words=30 if self.dataset in ["pubmed"] else 55,
                truncate_ratio=self.truncate_ratio,
                max_tries=self.max_sampling_tries,
            )
            if not ok:
                return []

            idx = 0
            for i in range(n):
                original = texts[i]
                for _b in range(self.regen_number):
                    s = sampled_texts[idx]
                    idx += 1
                    if self.dataset == "pubmed":
                        s = self._truncate_to_substring(s, "Question:", 2)
                        if _PUBMED_SEPARATOR_ENV:
                            original = original.replace(_PUBMED_SEPARATOR_ENV, " ")
                    o_trim, s_trim = self._trim_to_shorter_length(original, s)
                    out[i].append(s_trim)
            return out

        # HF ：， OOM
        B = self.batch_size
        for b in range(0, self.regen_number, B):
            cur_b = min(B, self.regen_number - b)
            batch_texts: List[str] = []
            for t in texts:
                batch_texts.extend([t] * cur_b)

            sampled_texts, ok = self._sample_from_model(
                batch_texts,
                min_words=30 if self.dataset in ["pubmed"] else 55,
                truncate_ratio=self.truncate_ratio,
                max_tries=self.max_sampling_tries,
            )
            if not ok:
                return []

            idx = 0
            for i in range(n):
                original = texts[i]
                for _b in range(cur_b):
                    s = sampled_texts[idx]
                    idx += 1
                    if self.dataset == "pubmed":
                        s = self._truncate_to_substring(s, "Question:", 2)
                        if _PUBMED_SEPARATOR_ENV:
                            original = original.replace(_PUBMED_SEPARATOR_ENV, " ")
                    o_trim, s_trim = self._trim_to_shorter_length(original, s)
                    out[i].append(s_trim)

        return out

    # =====  =====
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        probs = 1.0 / (1.0 + np.exp(-s))
        return np.clip(probs, 1e-6, 1.0 - 1e-6).astype(np.float32)

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        results: List[float] = []
        # ： batch_size （）
        group_size = max(1, int(self.regen_text_batch_size))
        if group_size <= 1:
            group_size = self.batch_size

        for g in range(0, len(texts), group_size):
            group = texts[g:g + group_size]
            valid_texts: List[str] = []
            valid_idx: List[int] = []

            for i, text in enumerate(group):
                try:
                    if self._token_len(text) < self.min_tokens:
                        print(f"[DNAGPT] skip short text (<{self.min_tokens} tokens).")
                        continue
                except Exception:
                    continue
                valid_texts.append(text)
                valid_idx.append(i)

            # NaN，
            group_scores: List[float] = [float("nan")] * len(group)

            if not valid_texts:
                results.extend(group_scores)
                continue

            # log_prob
            lprob_batch = self._log_prob_batch(valid_texts)
            regens_list = self._regen_samples_for_many(valid_texts)
            if not regens_list:
                print("[DNAGPT] skip texts due to insufficient generations.")
                results.extend(group_scores)
                continue

            flat_regens: List[str] = []
            for rs in regens_list:
                flat_regens.extend(rs)

            lprobs_regen = self._log_prob_batch(flat_regens)
            r = self.regen_number
            for j, idx in enumerate(valid_idx):
                start = j * r
                end = start + r
                wscore = (lprob_batch[j] - lprobs_regen[start:end].mean()).item()
                group_scores[idx] = float(wscore)

            results.extend(group_scores)

        return np.asarray(results, dtype=np.float32)
