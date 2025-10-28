# mgt_eval/detectors/metric/baseline.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..base import DetectorBase
from ..registry import register

# ---------------------------
# 小工具：路径/ID 判定与本地大小写无关匹配
# ---------------------------
def _is_probable_hf_id(name: str) -> bool:
    """
    经验规则：包含一个或多个'/'，且不是现存本地路径，即视为 HF repo id（如 'tiiuae/falcon-7b-instruct'）。
    """
    if not isinstance(name, str) or not name:
        return False
    if os.path.exists(name):
        return False
    return "/" in name and not name.startswith((".", "/", "~"))


def _case_insensitive_dir(path: str) -> Optional[str]:
    """
    尝试做一次大小写不敏感的本地目录匹配：
    - 若 path 已存在且是目录，直接返回 path
    - 否则在其上级目录中，用大小写无关比较匹配末段目录名
    - 匹配到就返回真实大小写的绝对路径，否则返回 None
    """
    if os.path.isdir(path):
        return os.path.abspath(path)

    parent = os.path.dirname(path) or "."
    base = os.path.basename(path)
    if not os.path.isdir(parent) or not base:
        return None
    base_lower = base.lower()
    try:
        for entry in os.listdir(parent):
            if entry.lower() == base_lower and os.path.isdir(os.path.join(parent, entry)):
                return os.path.abspath(os.path.join(parent, entry))
    except Exception:
        return None
    return None


def _ensure_local_or_hf_target(name: str) -> Tuple[str, bool]:
    """
    解析目标模型/分词器来源：
    - 如果是本地存在的目录（或被大小写无关匹配修正后存在），返回 (本地目录, use_hf=False)
    - 如果看起来是 HF id（如 org/repo），返回 (原字串, use_hf=True)
    - 否则如果是本地路径但不存在 → 抛错
    """
    if os.path.isdir(name):
        return os.path.abspath(name), False

    fixed = _case_insensitive_dir(name)
    if fixed is not None:
        return fixed, False

    if _is_probable_hf_id(name):
        return name, True

    # 非 HF id，且无法解析为本地存在目录
    raise RuntimeError(
        f"[ModelNotFound] Expect a local directory or an HF repo id, but got '{name}'. "
        f"No such local path, and it doesn't look like an HF repo id."
    )


# ---------------------------
# 通用工具
# ---------------------------
def _shift_for_next_token(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """
    标准 next-token 设定：以当前位置 t 的 logits 预测 token_{t+1}。
    返回:
      shift_logits: (B, T-1, V)
      labels:       (B, T-1)
    """
    labels = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    return shift_logits, labels


@torch.no_grad()
def _mean_log_likelihood(shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = shift_logits.shape[1]
    if T == 0:
        return 0.0
    lprobs = torch.log_softmax(shift_logits, dim=-1)             # (1, T, V)
    tok_ll = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (1, T)
    v = float(tok_ll.mean().item())
    return v if np.isfinite(v) else 0.0


@torch.no_grad()
def _neg_mean_rank(shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = shift_logits.shape[1]
    if T == 0:
        return 0.0
    matches = (shift_logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    if matches.numel() == 0:
        return 0.0
    ranks = matches[:, -1].float() + 1.0
    v = float(-(ranks.mean().item()))
    return v if np.isfinite(v) else 0.0


@torch.no_grad()
def _neg_mean_logrank(shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = shift_logits.shape[1]
    if T == 0:
        return 0.0
    matches = (shift_logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    if matches.numel() == 0:
        return 0.0
    ranks = matches[:, -1].float() + 1.0
    v = float(-(torch.log(ranks).mean().item()))
    return v if np.isfinite(v) else 0.0


@torch.no_grad()
def _mean_entropy(shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = shift_logits.shape[1]
    if T == 0:
        return 0.0
    probs = torch.softmax(shift_logits, dim=-1)
    lprobs = torch.log_softmax(shift_logits, dim=-1)
    ent = -(probs * lprobs).sum(dim=-1)  # (1, T)
    v = float(ent.mean().item())
    return v if np.isfinite(v) else 0.0

# ---------------------------
# 公共基类：单模型 metric-based 统计量
# ---------------------------
class _LMTokenStatDetectorBase(DetectorBase):
    """
    以单个 CausalLM 对输入文本做一次前向，计算序列级统计量（likelihood / rank / logrank / entropy）。
    子类只需实现 _stat(shift_logits, labels) → float。
    """
    CITATION_TITLE = "N/A"
    CITATION_AUTHORS = "Community Implementations"
    CITATION_LINK = "https://github.com/"

    def __init__(
        self,
        score_model: str,
        tokenizer: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 1024,
        fp16: bool = True,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs: Any,
    ):
        super().__init__(
            score_model=score_model,
            tokenizer=tokenizer or score_model,
            device=device,
            max_length=max_length,
            fp16=fp16,
            detector_type=detector_type or "Metric-based",
            **({"name": name} if name is not None else {}),
            **kwargs,
        )
        self.model = score_model
        self.tokenizer = tokenizer or score_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.fp16 = bool(fp16)

        base = os.path.basename(str(self.model).rstrip("/\\"))
        self.name = name or f"{self.__class__.__name__}[{base}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._tokenizer = None
        self._model = None
        self.is_loaded = False

    # 子类覆盖：计算统计量
    def _stat(self, shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
        raise NotImplementedError

    # ---- 模型加载 ----
    def load(self):
        def _load_tokenizer(name: str):
            target, use_hf = _ensure_local_or_hf_target(name)
            tok = AutoTokenizer.from_pretrained(
                target,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=not use_hf,
            )
            # decoder-only 常见：无 PAD → 用 EOS 作为 PAD
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"
            return tok

        def _load_causallm(name: str):
            target, use_hf = _ensure_local_or_hf_target(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                target,
                trust_remote_code=True,
                local_files_only=not use_hf,
            )
            if self.fp16 and self.device.startswith("cuda"):
                try:
                    mdl.half()
                except Exception:
                    pass
            mdl.to(self.device).eval()
            return mdl

        self._tokenizer = _load_tokenizer(self.tokenizer)
        self._model = _load_causallm(self.model)
        super().load()

    # ---- 单条打分 ----
    @torch.no_grad()
    def _score_one(self, text: str) -> float:
        if not self.is_loaded:
            self.load()
        tok = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        tok.pop("token_type_ids", None)
        tok = tok.to(self.device)

        out = self._model(**tok)
        logits = out.logits  # (1, T, V)
        shift_logits, labels = _shift_for_next_token(logits, tok["input_ids"], tok.get("attention_mask", None))

        # NEW: 序列太短（T-1==0）直接返回一个可校准的缺省分数
        if shift_logits.shape[1] == 0:
            return 0.0

        try:
            val = self._stat(shift_logits, labels)
        except Exception:
            val = 0.0
        # NEW: 数值清洗（极端保护）
        if not np.isfinite(val):
            val = 0.0
        return float(val)

    # ---- 批量接口 ----
    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        # ★ 新增懒加载保护
        if not self.is_loaded:
            self.load()
        scores: List[float] = []
        for t in texts:
            scores.append(self._score_one(t))
        return np.array(scores, dtype=np.float32)


# ---------------------------
# 4 个具体检测器（注册名与原论文/实现一致）
# ---------------------------
@register("likelihood")
class LikelihoodDetector(_LMTokenStatDetectorBase):
    """平均 token log-likelihood（越大通常越“像模型分布”）。"""
    def _stat(self, shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
        return _mean_log_likelihood(shift_logits, labels)


@register("rank")
class RankDetector(_LMTokenStatDetectorBase):
    """负的平均 rank（与参考实现保持一致，返回 -mean(rank)）。"""
    def _stat(self, shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
        return _neg_mean_rank(shift_logits, labels)


@register("logrank")
class LogRankDetector(_LMTokenStatDetectorBase):
    """负的平均 log(rank)（与参考实现保持一致）。"""
    def _stat(self, shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
        return _neg_mean_logrank(shift_logits, labels)


@register("entropy")
class EntropyDetector(_LMTokenStatDetectorBase):
    """平均 token-level 熵 H=-∑p log p（与参考实现保持一致）。"""
    def _stat(self, shift_logits: torch.Tensor, labels: torch.Tensor) -> float:
        return _mean_entropy(shift_logits, labels)
