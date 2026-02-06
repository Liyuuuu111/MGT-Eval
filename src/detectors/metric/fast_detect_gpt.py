# mgt_eval/detectors/metric/fast_detect_gpt.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..base import DetectorBase
from ..registry import register


# ---------------------------
# ：/ID
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

    # HF id，
    raise RuntimeError(
        f"[ModelNotFound] Expect a local directory or an HF repo id, but got '{name}'. "
        f"No such local path, and it doesn't look like an HF repo id."
    )


# ---------------------------
# ---------------------------
def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1e-6
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _prob_from_two_normals(x: float, mu0: float, s0: float, mu1: float, s1: float) -> float:
    p0 = _normal_pdf(x, mu0, s0)
    p1 = _normal_pdf(x, mu1, s1)
    denom = p0 + p1
    if denom <= 0:
        return 0.5
    return p1 / denom


def _model_basename(name: str) -> str:
    if not name:
        return ""
    s = name.strip().rstrip("/\\")
    base = os.path.basename(s)
    return base or s


def _resolve_params_key(sampling_name: str,
                        scoring_name: str,
                        distrib_params: Dict[str, Dict[str, float]]) -> Optional[str]:
    lower2orig = {k.lower(): k for k in distrib_params.keys()}
    s_base = _model_basename(sampling_name).lower()
    c_base = _model_basename(scoring_name).lower()
    cand_exact = f"{s_base}_{c_base}"
    if cand_exact in lower2orig:
        return lower2orig[cand_exact]

    cand_full = f"{sampling_name.lower()}_{scoring_name.lower()}"
    if cand_full in lower2orig:
        return lower2orig[cand_full]

    s_full = sampling_name.lower()
    c_full = scoring_name.lower()
    for low_k, orig_k in lower2orig.items():
        if "_" not in low_k:
            continue
        a, b = low_k.split("_", 1)
        if a and b and (a in s_full) and (b in c_full):
            return orig_k
    return None


@torch.no_grad()
def _sampling_discrepancy_analytic(logits_ref: torch.Tensor,
                                   logits_score: torch.Tensor,
                                   labels: torch.Tensor) -> float:
    assert logits_ref.shape[0] == 1 and logits_score.shape[0] == 1 and labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[..., :vocab_size]
        logits_score = logits_score[..., :vocab_size]

    if labels.ndim == logits_score.ndim - 1:
        labels = labels.unsqueeze(-1)  # (1, T, 1)

    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)  # (1, T)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)                       # (1, T)
    var_ref = (probs_ref * (lprobs_score ** 2)).sum(dim=-1) - (mean_ref ** 2)

    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / (var_ref.sum(dim=-1).sqrt() + 1e-12)
    return float(discrepancy.mean().item())


def _shift_for_next_token(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    labels = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    return shift_logits, labels


# ---------------------------
# ---------------------------
@register("fastdetectgpt")
class FastDetectGPTDetector(DetectorBase):
    """
    Fast-DetectGPT（Metric-based）黑盒检测器。
    - 支持同/不同 scoring & sampling 模型（建议同族 tokenizer，避免 token 对齐问题）。
    - 若提供（或命中内置）分布参数表，将直接输出“AI 概率”；否则输出判别量分数并交由基类标定为概率。
    """
    # ====== ：（ base.evaluate() ）======
    CITATION_TITLE = "Fast-DetectGPT: Efficient Detection of Machine-Generated Text via Sampling Discrepancy"
    CITATION_AUTHORS = "Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang"
    # TODO: /（arXiv/GitHub）
    CITATION_LINK = "https://arxiv.org/abs/2310.05130"
    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load()
    def __init__(
        self,
        scoring_model_name: str,
        sampling_model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 1024,
        fp16: bool = True,
        use_analytic: bool = True,
        distrib_params: Optional[Dict[str, Dict[str, float]]] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            scoring_model_name=scoring_model_name,
            sampling_model_name=sampling_model_name,
            tokenizer_name=tokenizer_name,
            device=device,
            max_length=max_length,
            fp16=fp16,
            use_analytic=use_analytic,
            detector_type=detector_type,
            **({"name": name} if name is not None else {}),
            **({"distrib_params": distrib_params} if distrib_params is not None else {}),
            **kwargs,
        )
        self.scoring_model_name = scoring_model_name
        self.sampling_model_name = sampling_model_name or scoring_model_name
        self.tokenizer_name = tokenizer_name or scoring_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.fp16 = bool(fp16)
        self.use_analytic = bool(use_analytic)

        self.name = name or f"FastDetectGPT[{os.path.basename(self.sampling_model_name)}_{os.path.basename(self.scoring_model_name)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._default_params = {
            "gpt-j-6B_gpt-neo-2.7B": {"mu0": 0.2713, "sigma0": 0.9366, "mu1": 2.2334, "sigma1": 1.8731},
            "gpt-neo-2.7B_gpt-neo-2.7B": {"mu0": -0.2489, "sigma0": 0.9968, "mu1": 1.8983, "sigma1": 1.9935},
            "falcon-7b_falcon-7b-instruct": {"mu0": -0.0707, "sigma0": 0.9520, "mu1": 2.9306, "sigma1": 1.9039},
        }
        self.distrib_params = dict(self._default_params)
        if isinstance(distrib_params, dict):
            self.distrib_params.update(distrib_params)
        self._tokenizer = None
        self._score_model = None
        self._samp_model = None
        self._have_intrinsic_prob = False
        self.is_loaded = False

    # calibrate：“”，
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        if self._have_intrinsic_prob:
            return np.clip(scores, 1e-6, 1 - 1e-6)
        return super().calibrate(scores, labels)

    # ---------------------------
    # （；； gpt-neo ）
    # ---------------------------
    def load(self):
        def _load_tokenizer(name: str):
            target, use_hf = _ensure_local_or_hf_target(name)
            tok = AutoTokenizer.from_pretrained(
                target,
                use_fast=True,
                trust_remote_code=True,          # Falcon
                local_files_only=not use_hf,     # →；HF id→
            )
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            return tok

        def _load_causallm(name: str):
            target, use_hf = _ensure_local_or_hf_target(name)
            try:
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
                return mdl, True
            except Exception:
                return None, False

        # 1) （/）
        self._tokenizer = _load_tokenizer(self.tokenizer_name)
        self._score_model, ok_s = _load_causallm(self.scoring_model_name)

        if self.sampling_model_name == self.scoring_model_name and ok_s:
            self._samp_model, ok_a = self._score_model, True
        else:
            self._samp_model, ok_a = _load_causallm(self.sampling_model_name)

        # 2) ，（）
        if not (ok_s and ok_a):
            raise RuntimeError(
                "[ModelLoadError] Failed to load models.\n"
                f"  scoring_model_name = {self.scoring_model_name}\n"
                f"  sampling_model_name = {self.sampling_model_name}\n"
                f"  tokenizer_name      = {self.tokenizer_name}\n"
                "If you intended to use local paths, ensure the directories exist and contain:\n"
                "  - config.json\n"
                "  - model.safetensors or pytorch_model*.bin (may be sharded)\n"
                "  - tokenizer.json + tokenizer_config.json + special_tokens_map.json\n"
                "If you intended to use HF repos, pass strings like 'org/repo' (e.g., 'EleutherAI/gpt-j-6b')."
            )

        # 3) ： calibrate()，
        key = _resolve_params_key(self.sampling_model_name, self.scoring_model_name, self.distrib_params)
        self._params_key = key
        self._have_intrinsic_prob = key is not None

        # 4)
        s_show = _model_basename(self.sampling_model_name)
        c_show = _model_basename(self.scoring_model_name)
        self.name = f"FastDetectGPT[{s_show}_{c_show}]"
        self.DETECTOR_NAME = self.name

        super().load()

    # ---------------------------
    # ---------------------------
    def _score_one(self, text: str) -> float:
        self._ensure_loaded()  # ★
        # Tokenize & forward ()
        tok = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        # Falcon/decoder-only  token_type_ids
        if hasattr(tok, "pop"):
            tok.pop("token_type_ids", None)
        else:
            try:
                del tok["token_type_ids"]
            except Exception:
                pass
        tok = tok.to(self.device)

        out_score = self._score_model(**tok)
        logits_score = out_score.logits  # (1, T, V)
        shift_score, labels = _shift_for_next_token(logits_score, tok["input_ids"], tok["attention_mask"])
        if self._samp_model is self._score_model:
            shift_ref = shift_score
        else:
            tok_ref = self._tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            )
            if hasattr(tok_ref, "pop"):
                tok_ref.pop("token_type_ids", None)
            else:
                try:
                    del tok_ref["token_type_ids"]
                except Exception:
                    pass
            tok_ref = tok_ref.to(self.device)

            out_ref = self._samp_model(**tok_ref)
            shift_ref, labels_ref = _shift_for_next_token(
                out_ref.logits, tok_ref["input_ids"], tok_ref["attention_mask"]
            )

            # ：，
            if labels_ref.shape != labels.shape or not torch.all(labels_ref == labels):
                shift_ref = shift_score
        if self.use_analytic:
            disc = _sampling_discrepancy_analytic(shift_ref, shift_score, labels)
        else:
            disc = _sampling_discrepancy_analytic(shift_ref, shift_score, labels)

        # （）
        if self._have_intrinsic_prob and self._params_key is not None:
            p = self.distrib_params[self._params_key]
            prob = _prob_from_two_normals(disc, p["mu0"], p["sigma0"], p["mu1"], p["sigma1"])
            return float(np.clip(prob, 1e-6, 1 - 1e-6))
        else:
            return float(disc)

    # ---------------------------
    # （）
    # ---------------------------
    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()  # ★
        scores: List[float] = []
        for t in texts:
            try:
                s = self._score_one(t)
            except Exception:
                s = 0.0 if not self._have_intrinsic_prob else 0.5
            scores.append(float(s))
        return np.array(scores, dtype=np.float32)
