# -*- coding: utf-8 -*-
# mgt_eval/detectors/metric/dna_detectllm.py

from __future__ import annotations
from typing import List, Optional, Union
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# mgt_eval ： fallback stub
# ---------------------------------------------------------
try:
    from ..base import DetectorBase
    from ..registry import register
except Exception:  # pragma: no cover -  import
    class DetectorBase:
        def __init__(self, **kwargs):
            pass

        def load(self):
            pass

    def register(name: str):
        def deco(cls):
            return cls
        return deco


# ---------------------------------------------------------
# Helper： tokenizer
# ---------------------------------------------------------
def assert_tokenizer_consistency(model_id_1: str, model_id_2: str):
    """
    严格沿用你给出的实现：直接比较 .vocab（即使这在部分新模型上不太合理，也不改动）
    """
    tok1 = AutoTokenizer.from_pretrained(model_id_1)
    tok2 = AutoTokenizer.from_pretrained(model_id_2)
    identical_tokenizers = tok1.vocab == tok2.vocab
    if not identical_tokenizers:
        raise ValueError(
            f"Tokenizers are not identical for {model_id_1} and {model_id_2}."
        )
    return True


# ---------------------------------------------------------
# & （）
# ---------------------------------------------------------
# torch.set_grad_enabled(False)

huggingface_config = {
    # HuggingFace ；
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
detectllm_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for F1
detectllm_FPR_THRESHOLD = 0.8536432310785527       # optimized for low-FPR (0.01%)

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

ce_loss_fn = CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


# ---------------------------------------------------------
# metrics.py ：（，）
# ---------------------------------------------------------
def min_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    # token
    max_prob_token = torch.argmax(shifted_logits, dim=-1)
    shifted_labels_max = max_prob_token

    if median:
        ce_nan = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max)
            .masked_fill(~shifted_attention_mask.bool(), float("nan"))
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max)
            * shifted_attention_mask
        ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def auc_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
    max_batch_size: int = 50,
    repair_order: str = "s",
):
    """
    完全按你贴出的“逐 token 修复 + 记录 PPL 序列再平均”的版本来写，
    只做了变量命名和循环上的轻微整理，不改变算法流程。
    """
    shifted_logits = logits[..., :-1, :] / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:]

    probs = torch.softmax(shifted_logits, dim=-1)
    max_prob_tokens = probs.argmax(dim=-1)

    input_ids = encoding.input_ids[..., 1:].clone()
    current_labels = input_ids.clone()

    # PPL
    ce_initial = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels).float()
    ppl_sequence = [
        (ce_initial * shifted_attention_mask).sum() / shifted_attention_mask.sum()
    ]

    # GT  max ，（）
    logits_diff = torch.abs(
        probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        - probs.gather(-1, max_prob_tokens.unsqueeze(-1)).squeeze(-1)
    )

    non_max_mask = input_ids != max_prob_tokens
    change_indices = non_max_mask.nonzero(as_tuple=False)

    if repair_order == "s":
        iter_indices = change_indices
    elif repair_order == "h2l":
        iter_indices = sorted(
            change_indices.tolist(), key=lambda idx: logits_diff[idx[0], idx[1]].item()
        )
    elif repair_order == "l2h":
        iter_indices = sorted(
            change_indices.tolist(),
            key=lambda idx: -logits_diff[idx[0], idx[1]].item(),
        )
    elif repair_order == "r":
        iter_indices = change_indices.tolist()
        random.shuffle(iter_indices)
    else:
        iter_indices = change_indices

    for idx in iter_indices:
        batch_idx, token_idx = idx
        current_labels[batch_idx, token_idx] = max_prob_tokens[batch_idx, token_idx]

        ce_current = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels)
        current_ppl = (
            ce_current * shifted_attention_mask
        ).sum() / shifted_attention_mask.sum()
        current_ppl = current_ppl.float()
        ppl_sequence.append(current_ppl)

    ppl_sequence_np = torch.stack(ppl_sequence).cpu().numpy()
    auc = np.mean(ppl_sequence_np)
    return auc


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
            .masked_fill(~shifted_attention_mask.bool(), float("nan"))
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
            * shifted_attention_mask
        ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


@torch.no_grad()
def sum_perplexity(encoding: transformers.BatchEncoding,
                   logits: torch.Tensor,
                   median: bool = False,
                   temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :] / temperature
    attention = encoding.attention_mask[..., 1:]
    labels_std = encoding.input_ids[..., 1:]
    labels_max = torch.argmax(shifted_logits, dim=-1)

    logits_T = shifted_logits.transpose(1, 2)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * attention).sum(dim=1) / attn_sum

    out = (ppl_std + ppl_max).float()   # ★  float32
    return out.cpu().numpy()


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (
            (ce * padding_mask).sum(1) / padding_mask.sum(1)
        ).to("cpu").float().numpy()

    return agg_ce


def entropy_pro(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]

    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    max_p_token = torch.argmax(p_proba, dim=-1)
    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=max_p_token).view(-1, total_tokens_available)

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (
            (ce * padding_mask).sum(1) / padding_mask.sum(1)
        ).to("cpu").float().numpy()

    return agg_ce


# ---------------------------------------------------------
# DetectLLM （）
# ---------------------------------------------------------
class DetectLLM(object):
    def __init__(
        self,
        observer_name_or_path: str = "./Model/falcon-7b",
        performer_name_or_path: str = "./Model/falcon-7b-instruct",
        # ，：
        # observer_name_or_path: str = "/data/llm/Llama-3-8B",
        # performer_name_or_path: str = "/data/llm/Llama-3-8B-Instruct",
        # ...
        use_bfloat16: bool = False,
        max_token_observed: int = 1024,
        mode: str = "low-fpr",
    ) -> None:
        # tokenizer
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            # token=huggingface_config["TOKEN"]
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            # token=huggingface_config["TOKEN"]
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    # ——  mode / threshold  ——
    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = detectllm_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = detectllm_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: List[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding):
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def cleanup(self):
        if getattr(self, "observer_model", None) is not None:
            del self.observer_model
            del self.performer_model
            self.observer_model = None
            self.performer_model = None
        torch.cuda.empty_cache()

    def compute_score(
        self, input_text: Union[List[str], str]
    ) -> Union[float, List[float]]:
        """
        严格按你现在的版本：
        - 输入是 str 或 list[str]；
        - 始终返回 Python list（即使输入是单条 str）。
        """
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        # ppl = auc_perplexity(encodings, performer_logits, repair_order="l2h")
        ppl = sum_perplexity(encodings, performer_logits)
        # ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )

        # detectllm_scores = (ppl + standard_ppl) / (x_ppl * 2)
        detectllm_scores = -ppl / (2 * x_ppl)
        detectllm_scores = detectllm_scores.tolist()

        return detectllm_scores

    def compute_anomaly_level(
        self, input_text: Union[List[str], str]
    ) -> Union[float, List[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl_auc = auc_perplexity(encodings, performer_logits, repair_order="h2l")
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )

        detectllm_scores = ppl_auc / x_ppl
        detectllm_scores = detectllm_scores.tolist()

        return detectllm_scores

    def predict(self, input_text: Union[List[str], str]) -> Union[List[str], str]:
        detectllm_scores = np.array(self.compute_score(input_text))
        pred = np.where(
            detectllm_scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred


# ---------------------------------------------------------
# mgt_eval Detector ：
# / +  "dnadetectllm"
# ---------------------------------------------------------
@register("dnadetectllm")
class DNADetectLLMDetector(DetectorBase):
    """
    这个类是给 mgt_eval 用的 DetectorBase 子类：
    - 内部完整调用上面的 DetectLLM（原始实现不动）；
    - 外部通过 mgt_eval 统一的 score_batch(texts) 接口来用。
    """

    CITATION_TITLE = (
        "DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm"
    )
    CITATION_AUTHORS = (
        "Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao"
    )
    CITATION_LINK = "https://openreview.net/forum?id=yQoHUijSHx"

    def __init__(
        self,
        observer_model: str,
        performer_model: Optional[str] = None,
        use_bfloat16: bool = False,
        max_token_observed: int = 1024,
        mode: str = "low-fpr",
        prob_slope: float = -6.0,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            observer_model=observer_model,
            performer_model=performer_model,
            use_bfloat16=bool(use_bfloat16),
            max_token_observed=int(max_token_observed),
            mode=mode,
            prob_slope=float(prob_slope),
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.observer_model_name = observer_model
        self.performer_model_name = performer_model or observer_model
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_token_observed = int(max_token_observed)
        self.mode = mode
        self.prob_slope = float(prob_slope)
        self.user_device = device

        base_obs = os.path.basename(self.observer_model_name)
        base_perf = os.path.basename(self.performer_model_name)
        self.name = name or f"DNA-DetectLLM[{base_obs}|{base_perf}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._core: Optional[DetectLLM] = None
        self.is_loaded = False

    def load(self):
        # ： DetectLLM  DEVICE_1 / DEVICE_2 ，
        self._core = DetectLLM(
            observer_name_or_path=self.observer_model_name,
            performer_name_or_path=self.performer_model_name,
            use_bfloat16=self.use_bfloat16,
            max_token_observed=self.max_token_observed,
            mode=self.mode,
        )
        self.is_loaded = True
        super().load()

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded or self._core is None:
            self.load()
        scores = self._core.compute_score(texts)  # list
        return np.asarray(scores, dtype=np.float32)

    def cleanup(self):
        if self._core is not None:
            self._core.cleanup()
            self._core = None
        self.is_loaded = False
