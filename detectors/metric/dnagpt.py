# mgt_eval/detectors/metric/dnagpt.py
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 顶部补一行
from transformers.tokenization_utils_base import BatchEncoding
from ..base import DetectorBase
from ..registry import register

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
        # ★ 新增：最小 token 数阈值，低于该阈值直接跳过
        min_tokens: int = 20,
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

        self.min_tokens = int(min_tokens)  # ★ 新增

        self.user_device = device
        self.device = "cpu"
        self.dtype = _torch_dtype(use_bf16=use_bfloat16)

        self.name = name or f"DNAGPT[{os.path.basename(self.score_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._tok = None
        self._model = None
        self.is_loaded = False
        self.model_max_positions: Optional[int] = None

    # ===== 加载 =====
    def load(self):
        self.device = _select_device(self.user_device)
        self.dtype = _torch_dtype(self.use_bfloat16)

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
        self._model.config.pad_token_id = self._tok.pad_token_id
        self.model_max_positions = int(getattr(self._model.config, "max_position_embeddings", 2048))
        super().load()

    # ===== 工具函数 =====
    @torch.inference_mode()
    def _tokenize(self, texts: List[str]):
        # 空串清洗
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

        # ★ 统一兜底：始终返回 BatchEncoding
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

    # —— 后处理与原脚本一致 —— #
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
        import math

        if self.dataset == "pubmed":
            pubmed_sep = " Answer:"
            proc_texts = []
            for t in texts:
                # ★ 安全查找，找不到就不用 index 直接走回退
                pos = t.find(pubmed_sep)
                if pos != -1:
                    proc_texts.append(t[: pos + len(pubmed_sep)])
                else:
                    # 回退：按比例取前缀但至少 1 词
                    toks = t.split()
                    keep = max(1, math.ceil(len(toks) * truncate_ratio))
                    proc_texts.append(" ".join(toks[:keep]) if keep > 0 else " ")
            all_encoded = self._tokenize(proc_texts)
        else:
            toks = [t.split() for t in texts]
            # ★ 至少保留 1 词，避免空前缀
            trunc = [" ".join(ts[: max(1, math.ceil(len(ts) * truncate_ratio))]) for ts in toks]
            all_encoded = self._tokenize(trunc)

        self._model.eval()
        decoded = ["" for _ in range(len(texts))]

        tries = 0
        m = 0
        limit = self.max_sampling_tries if max_tries is None else int(max_tries)

        while m < min_words and (tries < limit):
            if tries != 0:
                print(f"\nmin words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs: Dict[str, Any] = {"temperature": self.temperature, "do_sample": True}
            if self.do_top_p:
                sampling_kwargs["top_p"] = self.top_p
            elif self.do_top_k:
                sampling_kwargs["top_k"] = self.top_k

            input_len = int(all_encoded["input_ids"].shape[1])

            # ★ 仍然保险一次：如果还为 0，构造 1-token prompt
            if input_len == 0:
                bos = getattr(self._tok, "bos_token_id", None) or getattr(self._tok, "eos_token_id", 0)
                B = all_encoded["input_ids"].shape[0] if isinstance(all_encoded, dict) else len(texts)
                all_encoded = {
                    "input_ids": torch.full((B, 1), bos, dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones((B, 1), dtype=torch.long, device=self.device),
                }
                input_len = 1

            target_total_max = int(self.max_length)
            min_total = self.min_length_pubmed if self.dataset in ["pubmed"] else self.min_length_non_pubmed

            max_new_tokens = max(1, target_total_max - input_len)
            min_new_tokens = max(1, int(min_total) - input_len)
            if min_new_tokens > max_new_tokens:
                min_new_tokens = max_new_tokens

            outputs = self._model.generate(
                **all_encoded,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,  # 可按需去掉以减少早停
                **sampling_kwargs,
            )
            decoded = self._tok.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded)
            tries += 1

        ok = (m >= min_words)
        if not ok:
            print(f"[DNAGPT] give up after {limit} tries (min words reached: {m}/{min_words}).")
        return decoded, ok

    def _regen_samples_for_one(self, text: str) -> List[str]:
        assert self.regen_number % self.batch_size == 0, \
            f"regen_number({self.regen_number}) must be divisible by batch_size({self.batch_size})"

        replicas = [text] * self.regen_number
        out: List[str] = []
        B = self.batch_size
        for b in range(0, self.regen_number, B):
            original_texts = replicas[b:b + B]
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

    # ===== 与评估器对接 =====
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        probs = 1.0 / (1.0 + np.exp(-s))
        return np.clip(probs, 1e-6, 1.0 - 1e-6).astype(np.float32)

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        results: List[float] = []
        for text in texts:
            # ★ 新增：对分词后长度小于阈值的样本直接跳过（写入 NaN）
            try:
                if self._token_len(text) < self.min_tokens:
                    print(f"[DNAGPT] skip short text (<{self.min_tokens} tokens).")
                    results.append(float("nan"))
                    continue
            except Exception:
                # 容错：若长度判定出错，也不要终止流程
                results.append(float("nan"))
                continue

            lprob = self._log_prob_single(text)[0]
            regens = self._regen_samples_for_one(text)
            if len(regens) == 0:
                print("[DNAGPT] skip this text due to insufficient generations.")
                results.append(float("nan"))
                continue
            lprobs_regen = self._log_prob_batch(regens)
            wscore = (lprob - lprobs_regen.mean()).item()
            results.append(float(wscore))
        return np.asarray(results, dtype=np.float32)
