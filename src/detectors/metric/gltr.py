# mgt_eval/detectors/metric/gltr.py
from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

from ..base import DetectorBase
from ..registry import register

# ===========================
# GLTR
# ===========================
GLTR_BIN_EDGES = (10, 100, 1000)  # <10, <100, <1000, >=1000

# （）
HF_TOKEN = os.environ.get("HF_TOKEN", None)


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


def _shift_labels(enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shifted_labels = enc_input_ids[..., 1:].contiguous()          # (B, T-1)
    shifted_attn = enc_attention_mask[..., 1:].contiguous()       # (B, T-1)
    return shifted_labels, shifted_attn


@register("gltr")
class GLTRDetector(DetectorBase):
    """
    GLTR：以参考 CausalLM 的 token 概率排名统计（4 桶特征）作为检测特征。
    - 本实现在 `calibrate()` 中支持用少量样本（如 1000 条）训练一个 LogisticRegression 头，
      将 4 维 GLTR 特征映射为 P(MGT) 概率；其余样本作为测试集。
    - 若未提供标签或关闭标定，`score_batch()` 直接返回 4 维特征。
    """
    CITATION_TITLE = "GLTR: Statistical Detection and Visualization of Generated Text"
    CITATION_AUTHORS = "Sebastian Gehrmann, Hendrik Strobelt, Alexander M. Rush"
    CITATION_LINK = "https://arxiv.org/abs/1906.04043"

    def __init__(
        self,
        score_model: str,
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        # —— ： ——
        calibrate_k: int = 1000,
        calibrate_seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type or "Metric-based",
            **kwargs,
        )
        self.score_model = score_model
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_len = int(max_token_observed)
        self.user_device = device

        # /
        self.name = name or f"GLTR[{os.path.basename(self.score_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"
        self.calibrate_k = int(max(1, calibrate_k))
        self.calibrate_seed = int(calibrate_seed)
        self._clf: Optional[LogisticRegression] = None
        self._calib_idx: Optional[np.ndarray] = None  # （/）
        self._tok = None
        self._model = None
        self._dev = "cpu"
        self.is_loaded = False

    def load(self):
        # & dtype
        self._dev = _select_device(self.user_device)
        dtype = _torch_dtype(self.use_bf16 if hasattr(self, "use_bf16") else self.use_bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.score_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self._dev},
            token=HF_TOKEN,
        )
        self._model.eval()

        # tokenizer
        self._tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token

        super().load()

    @torch.inference_mode()
    def _encode(self, texts: List[str]):
        return self._tok(
            texts,
            return_tensors="pt",
            padding=True if len(texts) > 1 else False,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
        )

    @torch.inference_mode()
    def _get_logits(self, encodings) -> torch.Tensor:
        enc = {k: (v.to(self._dev) if hasattr(v, "to") else v) for k, v in encodings.items()}
        logits = self._model(**enc).logits  # (B, T, V)
        if self._dev.startswith("cuda"):
            torch.cuda.synchronize()
        return logits

    @staticmethod
    def _gltr_bins_from_ranks(ranks_1d: torch.Tensor) -> np.ndarray:
        edges = GLTR_BIN_EDGES
        res = np.zeros(4, dtype=np.float64)
        for i in range(int(ranks_1d.numel())):
            r = float(ranks_1d[i].item())
            if r < edges[0]:
                res[0] += 1.0
            elif r < edges[1]:
                res[1] += 1.0
            elif r < edges[2]:
                res[2] += 1.0
            else:
                res[3] += 1.0
        s = res.sum()
        if s > 0:
            res = res / s
        return res

    @staticmethod
    def _token_ranks_from_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # —— ： labels  logits  ——
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        top_idx = logits.argsort(dim=-1, descending=True)  # (1, T-1, V)
        matches = (top_idx == labels.unsqueeze(-1)).nonzero()  # (?, 3) -> [b, t, rank]
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks = matches[:, -1].to(dtype=torch.float32) + 1.0  # 1-indexed
        timesteps = matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        return ranks  # (T-1,)

    def _fit_lr(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y)
        return clf

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        - 输入 scores: (N, 4) 的 GLTR 特征（或上游传来的同维度特征）
        - 若提供 labels，则从中抽取 `calibrate_k` 条做 LR 训练，其余样本与训练样本一并输出概率。
          （评测指标如何划分由上层评测器决定；本实现返回与输入等长的一维概率）
        """
        if labels is None:
            return scores  # ：

        if scores.ndim != 2 or scores.shape[1] != 4:
            # 4 ， LR
            return scores

        n = len(scores)
        k = min(self.calibrate_k, n)
        rng = np.random.default_rng(self.calibrate_seed)

        idx = np.arange(n)
        # （），
        uniq = np.unique(labels)
        if len(uniq) == 2:
            pos = idx[labels == 1]
            neg = idx[labels == 0]
            k_pos = int(round(k * len(pos) / max(1, len(idx))))
            k_neg = k - k_pos
            calib_pos = rng.choice(pos, size=min(k_pos, len(pos)), replace=False) if len(pos) > 0 else np.array([], int)
            calib_neg = rng.choice(neg, size=min(k_neg, len(neg)), replace=False) if len(neg) > 0 else np.array([], int)
            calib_idx = np.concatenate([calib_pos, calib_neg])
            if len(calib_idx) < k:
                rest = np.setdiff1d(idx, calib_idx)
                if len(rest) > 0:
                    extra = rng.choice(rest, size=min(k - len(calib_idx), len(rest)), replace=False)
                    calib_idx = np.concatenate([calib_idx, extra])
        else:
            calib_idx = rng.choice(idx, size=k, replace=False)

        self._calib_idx = np.array(sorted(set(calib_idx.tolist())))

        # LR
        self._clf = self._fit_lr(scores[self._calib_idx], labels[self._calib_idx])

        # （）
        probs = self._clf.predict_proba(scores)[:, 1].astype(np.float32)
        return probs

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        返回 (B,4) 的 GLTR 特征（四桶比例）。若上层提供标签并调用 `calibrate()`，
        则会在上层把这四维特征映射为 1 维概率。
        """
        results: List[np.ndarray] = []
        for t in texts:
            enc = self._encode([t])
            logits = self._get_logits(enc)[:, :-1, :]  # (1, T-1, V)
            shifted_labels, _ = _shift_labels(enc.input_ids, enc.attention_mask)
            # ——  logits ， cuda/cpu  ——
            if shifted_labels.device != logits.device:
                shifted_labels = shifted_labels.to(logits.device)

            ranks_1d = self._token_ranks_from_logits_labels(logits, shifted_labels)
            features = self._gltr_bins_from_ranks(ranks_1d)
            results.append(features.astype(np.float32))

        return np.vstack(results).astype(np.float32)
