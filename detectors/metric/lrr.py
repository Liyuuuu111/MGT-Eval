# mgt_eval/detectors/metric/lrr.py
from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import DetectorBase
from ..registry import register

# 环境变量（私有模型需要）
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def _select_device(user_device: Optional[str]) -> str:
    """
    返回单设备字符串：
      - 优先使用用户指定；
      - 否则自动：cuda:0 可用则选 cuda:0，否则 cpu。
    """
    if user_device:
        d = user_device.strip().lower()
        if d.startswith("cpu"):
            return "cpu"
        if d.startswith("cuda"):
            return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _torch_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32


def _get_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    与参考实现一致的逐 token 平均对数似然（单样本）：
      - logits: (1, T, V) -> 展平为 (T, V)
      - labels: (1, T)    -> 展平为 (T,)
    返回：mean(log p(label_t | x_{<t}))
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])  # (T, V)
    labels = labels.view(-1)                    # (T,)
    log_probs = F.log_softmax(logits, dim=-1)   # (T, V)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (T,)
    return log_likelihood.mean().item()


def _get_logrank(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    与参考实现一致的 Log-Rank（单样本）：
      - 对每个时间步 t，计算真实标签在降序排序下的秩 rank_t（1-indexed），再取 log(rank_t) 的均值。
      - 要求每个时间步恰好匹配一次（与原断言一致）。
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # 1-indexed
    ranks = torch.log(ranks)
    return ranks.mean().item()


def _lrr_single_text(model, tokenizer, device: str, text: str, max_len: int) -> float:
    """
    Log-Likelihood Log-Rank Ratio（单文本）：
      LRR(text) = - likelihood(text) / logrank(text)
    逐样本与参考实现保持一致（不做批量化以避免与断言逻辑冲突）。
    """
    with torch.no_grad():
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_len,
            padding=False,
        ).to(device)

        # CausalLM 的标准 next-token 设定
        labels = tokenized.input_ids[:, 1:]            # (1, T)
        logits = model(**tokenized).logits[:, :-1, :]  # (1, T, V)

        likelihood = _get_likelihood(logits, labels)
        logrank = _get_logrank(logits, labels)
        if logrank == 0:
            return 0
        return -likelihood / logrank


@register("lrr")
class LRRDetector(DetectorBase):
    """
    逻辑型（Logic-based）检测器：LRR（Log-Likelihood Log-Rank Ratio）
    - 载入单个 CausalLM 作为评分模型（scoring model）
    - 打分：LRR = - mean_log_likelihood / mean_log_rank
    - 不进行阈值/概率映射，直接返回原始分数（保持与参考实现一致）

    Args:
        score_model: 评分模型 id/路径
        use_bfloat16: CUDA 上使用 bf16
        max_token_observed: 最大 token 长度
        device: 首选设备（如 "cuda:0" / "cpu"）
        name: 自定义展示名
        detector_type: 检测器类型元信息（默认 "Metric-based"）
    """

    # ====== 文献信息（仅用于 base.evaluate() 打印，无算法常量）======
    CITATION_TITLE = "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text"
    CITATION_AUTHORS = "Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov"
    # 若无官方固定链接，可留空或在未来替换
    CITATION_LINK = "https://arxiv.org/abs/2306.05540"

    def __init__(
        self,
        score_model: str,
        use_bfloat16: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            use_bfloat16=use_bfloat16,
            max_length=max_length,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.use_bfloat16 = bool(use_bf16 := use_bfloat16)
        self.max_len = int(max_length)
        self.user_device = device

        # 展示名/类型
        base_name = os.path.basename(self.score_model.rstrip("/"))
        self.name = name or f"LRR[{base_name}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        # 运行时对象
        self._tok = None
        self._model = None
        self._dev = "cpu"
        self._dtype = _torch_dtype(use_bf16)
        self.is_loaded = False

    def load(self):
        # 1) 设备与精度
        self._dev = _select_device(self.user_device)
        self._dtype = _torch_dtype(self.use_bfloat16)

        # 2) 模型与 tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            self.score_model,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map={"": self._dev},
            token=HF_TOKEN,
        )
        self._model.eval()

        self._tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token  # 虽然单样本不做 padding，但保持一致性

        # ★ 同步到 model.config，万一后续用到 padding
        try:
            self._model.config.pad_token_id = self._tok.pad_token_id
        except Exception:
            pass
        super().load()
    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        返回每条文本的 LRR 原始分数（float32）；
        与参考实现保持一致：逐样本文本独立计算，不进行批次 padding。
        """
        assert self.is_loaded, "Call .load() before scoring."
        out: List[float] = []
        for t in texts:
            s = _lrr_single_text(self._model, self._tok, self._dev, t, self.max_len)
            out.append(float(s))
        return np.asarray(out, dtype=np.float32)
