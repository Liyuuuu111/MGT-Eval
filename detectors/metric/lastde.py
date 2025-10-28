# -*- coding: utf-8 -*-
# mgt_eval/detectors/metric/lastde.py

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import DetectorBase
from ..registry import register
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 环境变量
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# -----------------------------
# 设备/精度工具
# -----------------------------
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

def _assert_tokenizer_consistency(m1: str, m2: str):
    tok1 = AutoTokenizer.from_pretrained(m1, token=HF_TOKEN)
    tok2 = AutoTokenizer.from_pretrained(m2, token=HF_TOKEN)
    keys1 = {
        str(getattr(tok1, "cls_token_id", None)),
        str(getattr(tok1, "sep_token_id", None)),
        str(getattr(tok1, "pad_token_id", None)),
        str(getattr(tok1, "eos_token_id", None)),
        str(getattr(tok1, "bos_token_id", None)),
    }
    keys2 = {
        str(getattr(tok2, "cls_token_id", None)),
        str(getattr(tok2, "sep_token_id", None)),
        str(getattr(tok2, "pad_token_id", None)),
        str(getattr(tok2, "eos_token_id", None)),
        str(getattr(tok2, "bos_token_id", None)),
    }
    if keys1 != keys2:
        raise ValueError(f"Tokenizers are not identical enough for {m1} and {m2} (special tokens differ).")

# -----------------------------
# fastMDE —— 严格按“官方”实现
# -----------------------------
def histcounts(data: torch.Tensor, epsilon: int, min_=-1, max_=1):
    data = data.float()
    hist = torch.histc(data, bins=epsilon, min=min_, max=max_)
    statistical_probabilities_sequence = hist / torch.sum(hist)
    return hist, statistical_probabilities_sequence

def DE(statistical_probabilities_sequence: torch.Tensor, epsilon: int) -> torch.Tensor:
    DE_value = -1 / torch.log(torch.tensor(epsilon)) * torch.nansum(
        statistical_probabilities_sequence * torch.log(statistical_probabilities_sequence),
        dim=0
    )
    return DE_value

# ======= 在文件顶部工具区（函数们之前）加一个小的回退工具 =======
def _fallback_de(ori_data: torch.Tensor) -> torch.Tensor:
    # 返回 [B, S] 的 1.0，保证后续 lastde = templl / aggmde 可用
    B = ori_data.shape[0]
    S = ori_data.shape[2]
    return torch.ones((B, S), device=ori_data.device, dtype=ori_data.dtype)


# ======= 替换 calculate_DE =======
def calculate_DE(ori_data: torch.Tensor, embed_size: int, epsilon: int) -> torch.Tensor:
    # ori_data: [B, T, S]
    T = int(ori_data.shape[1])
    e = int(max(2, embed_size))          # 至少 2，才能做相邻窗口的余弦
    if T < (e + 1):                      # 长度不足，无法形成至少两个 e-窗口
        return _fallback_de(ori_data)

    # e 还需满足 T - e + 1 >= 2  => e <= T - 1
    e = min(e, T - 1)

    orbits = ori_data.unfold(1, e, 1)    # [B, T-e+1, S, e]，此时 T-e+1 >= 2
    orbits_cos = F.cosine_similarity(orbits[:, :-1], orbits[:, 1:], dim=-1)  # [B, T-e, S]

    eps = int(max(2, epsilon))           # 直方图分箱至少 2
    batched_1 = torch.vmap(histcounts, in_dims=-1, out_dims=1)
    _, probs_seq = batched_1(orbits_cos, epsilon=eps)
    return DE(probs_seq, eps)            # => [B, S]


# ======= 替换 get_tau_scale_DE（增加合法性检查） =======
def get_tau_scale_DE(ori_data: torch.Tensor, embed_size: int, epsilon: int, tau: int) -> torch.Tensor:
    # 需要先保证展开窗口合法
    T = int(ori_data.shape[1])
    e = int(max(2, embed_size))
    if tau > T:
        return _fallback_de(ori_data)
    # 经过 tau 平均后，时间维长度为 T' = T - tau + 1
    T_prime = T - tau + 1
    if T_prime < (e + 1):   # 后续 calculate_DE 还需要 T' >= e + 1
        return _fallback_de(ori_data)

    windows = ori_data.unfold(1, tau, 1)         # [B, T-tau+1, S, tau]
    tau_scale_seq = torch.mean(windows, dim=3)   # [B, T-tau+1, S]
    return calculate_DE(tau_scale_seq, e, epsilon)


# ======= 替换 get_tau_multiscale_DE（自动收敛 tau 范围 + 回退） =======
def get_tau_multiscale_DE(ori_data: torch.Tensor, embed_size: int, epsilon: int, tau_prime: int) -> torch.Tensor:
    # 限定 tau 的上界，确保后续都可计算
    T = int(ori_data.shape[1])
    e = int(max(2, embed_size))
    tau_max = T - e                       # 要让 T' = T - tau + 1 >= e + 1  <=> tau <= T - e
    if tau_max < 1:
        return _fallback_de(ori_data)     # 序列太短，整体回退

    taus = range(1, min(int(tau_prime), int(tau_max)) + 1)

    mde_vals = []
    for temp_tau in taus:
        mde_vals.append(get_tau_scale_DE(ori_data, e, int(max(2, epsilon)), temp_tau))  # [B, S]

    if not mde_vals:
        return _fallback_de(ori_data)

    mde = torch.stack(mde_vals, dim=0)    # [tau, B, S]
    return torch.std(mde, dim=0)          # [B, S]

def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, )):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    # 兜底：仅当确实是“真值对象”才 True（避免任何非空字符串误判为 True）
    return bool(x)

class fastMDE:
    @staticmethod
    def get_tau_multiscale_DE(ori_data, embed_size, epsilon, tau_prime):
        return get_tau_multiscale_DE(ori_data, embed_size, epsilon, tau_prime)

# -----------------------------
# 官方 Lastde / Lastde++ 逻辑函数（严格形状）
# -----------------------------
def _official_log_likelihood_tensor(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lprobs = torch.log_softmax(logits, dim=-1)           # [1, T, V]
    if labels.ndim == logits.ndim - 1:
        labels = labels.unsqueeze(-1)                    # [1, T, 1] 或 [1, T, S]
    return lprobs.gather(dim=-1, index=labels)           # [1, T, 1] 或 [1, T, S]

def _official_lastde_from_ll(log_likelihood: torch.Tensor,
                             embed_size: int,
                             epsilon: int,
                             tau_prime: int) -> torch.Tensor:
    templl = log_likelihood.mean(dim=1)                  # [1, S]
    aggmde = fastMDE.get_tau_multiscale_DE(
        ori_data=log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime
    )                                                    # [1, S]
    return templl / aggmde                               # [1, S]

def _choose_dtype(use_bfloat16: bool) -> torch.dtype:
    """
    严格由开关决定 dtype；仅在 GPU 不支持 BF16 时回退到 FP32。
    不再用“某个别名属性”，也不看环境默认值，避免被 config 覆盖。
    """
    if use_bfloat16:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere+ 原生 BF16
                return torch.bfloat16
        # 无 GPU 或不支持 BF16 → 回退 FP32
        return torch.float32
    else:
        return torch.float32

# -----------------------------
# 超参常量
# -----------------------------
LASTDE_EMBED_SIZE_DEFAULT = 3
LASTDE_EPSILON_MULT_DEFAULT = 10.0
LASTDE_TAU_PRIME_DEFAULT = 5

LASTDEPP_EMBED_SIZE_DEFAULT = 4
LASTDEPP_EPSILON_MULT_DEFAULT = 8.0
LASTDEPP_TAU_PRIME_DEFAULT = 15
LASTDEPP_NSAMPLES_DEFAULT = 100

# -----------------------------
# Detector 1: Lastde（单模型，返回“原始分数”）
# -----------------------------
@register("lastde")
class LastdeDetector(DetectorBase):
    CITATION_TITLE = "Training-free LLM-generated Text Detection by Mining Token Probability Sequences"
    CITATION_AUTHORS = "Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu"
    CITATION_LINK = "https://openreview.net/forum?id=vo4AHjowKi"

    def __init__(
        self,
        score_model: str,
        use_bfloat16: bool = False,
        max_token_observed: int = 512,
        embed_size: int = LASTDE_EMBED_SIZE_DEFAULT,
        epsilon_mult: float = LASTDE_EPSILON_MULT_DEFAULT,
        tau_prime: int = LASTDE_TAU_PRIME_DEFAULT,
        device: Optional[str] = None,
        name: Optional[str] = None,
        add_special_tokens: bool = False,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            use_bfloat16=bool(use_bfloat16),
            max_token_observed=max_token_observed,
            embed_size=embed_size,
            epsilon_mult=epsilon_mult,
            tau_prime=tau_prime,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_len = int(max_token_observed)
        self.add_special_tokens = bool(add_special_tokens)
        self.embed_size = int(embed_size)
        self.epsilon_mult = float(epsilon_mult)
        self.tau_prime = int(tau_prime)
        self.user_device = device
        
        self.name = name or f"Lastde[{os.path.basename(self.score_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._tok = None
        self._model = None
        self._dev = "cpu"
        self.is_loaded = False

    def load(self):
        self._dev = _select_device(self.user_device)
        # 1) 明确选择 dtype（只看 self.use_bfloat16）
        dtype = _choose_dtype(self.use_bfloat16)
        # 2) 加载 tokenizer
        self._tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token

        # 3) 加载权重（按 dtype），并放到目标设备
        #    说明：即便某些 repo 的 config 声明了 bfloat16，这里也用我们传入的 dtype 覆盖
        self._model = AutoModelForCausalLM.from_pretrained(
            self.score_model,
            trust_remote_code=True,
            torch_dtype=dtype,                 # 显式 dtype
            device_map=None,                   # 先在 CPU 加载，后面再 .to(self._dev) 统一上卡和 cast
            token=HF_TOKEN,
        )
        self._model.to(self._dev)             # 上卡
        self._model.to(dtype=dtype)           # 再次确保权重被 cast 到期望 dtype（双保险）
        try:
            # 有些模型在 forward 内部会参考 config.torch_dtype，这里一并同步
            self._model.config.torch_dtype = dtype
        except Exception:
            pass

        self._model.eval()

        # 4) 显式关闭 autocast（以防 trust_remote_code 里开了 autocast）
        #    你不需要在这里写上下文，inference 时我们不启用 autocast 即可；仅做提示
        self._disable_autocast = True  # 仅做标记；真正的前向在 score_batch() 里不进入 autocast

        super().load()

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        返回“原始分数”：score_raw = - LASTDE(x)
        （方向归一化为“越大越像 AI”，便于评估器统一排序）
        """
        scores_out: List[float] = []
        for text in texts:
            enc = self._tok([text], return_tensors="pt",
                            padding=False, truncation=True, max_length=self.max_len,
                            add_special_tokens=self.add_special_tokens,
                            return_token_type_ids=False)
            enc = {k: (v.to(self._dev) if hasattr(v, "to") else v) for k, v in enc.items()}

            logits = self._model(**enc).logits[:, :-1, :]   # [1, T-1, V]
            labels = enc["input_ids"][:, 1:]               # [1, T-1]

            ll_tensor = _official_log_likelihood_tensor(logits, labels)  # [1, T-1, 1]
            T = ll_tensor.shape[1]
            epsilon = max(2, int(self.epsilon_mult * T))   # <- 加一个下限，防直方图异常

            lastde_vec = _official_lastde_from_ll(
                log_likelihood=ll_tensor,
                embed_size=self.embed_size,
                epsilon=epsilon,
                tau_prime=self.tau_prime,
            )  # [1, 1]
            lastde_val = float(lastde_vec.view(-1)[0].item())
            score_raw = lastde_val  # 方向：更大=更像AI
            scores_out.append(score_raw)

        return np.asarray(scores_out, dtype=np.float32)

# -----------------------------
# Detector 2: Lastde++（双模型 + 采样，返回“原始分数”）
# -----------------------------
@register("lastdepp")
class LastdePPDetector(DetectorBase):
    CITATION_TITLE = "Training-free LLM-generated Text Detection by Mining Token Probability Sequences"
    CITATION_AUTHORS = "Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu"
    CITATION_LINK = "https://openreview.net/forum?id=vo4AHjowKi"

    def __init__(
        self,
        score_model: str,
        reference_model: Optional[str] = None,
        use_bfloat16: bool = False,
        max_token_observed: int = 512,
        embed_size: int = LASTDEPP_EMBED_SIZE_DEFAULT,
        epsilon_mult: float = LASTDEPP_EPSILON_MULT_DEFAULT,
        tau_prime: int = LASTDEPP_TAU_PRIME_DEFAULT,
        n_samples: int = LASTDEPP_NSAMPLES_DEFAULT,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            reference_model=reference_model,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            embed_size=embed_size,
            epsilon_mult=epsilon_mult,
            tau_prime=tau_prime,
            n_samples=n_samples,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.reference_model = reference_model or score_model

        self.use_bfloat16 = bool(use_bfloat16)
        self.max_len = int(max_token_observed)
        self.embed_size = int(embed_size)
        self.epsilon_mult = float(epsilon_mult)
        self.tau_prime = int(tau_prime)
        self.n_samples = int(n_samples)
        self.user_device = device

        self.name = name or f"Lastde++[{os.path.basename(self.score_model)}|{os.path.basename(self.reference_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._tok = None
        self._scoring = None
        self._reference = None
        self._dev = "cpu"
        self.is_loaded = False

    def load(self):
        _assert_tokenizer_consistency(self.score_model, self.reference_model)
        self._dev = _select_device(self.user_device)
        dtype = _choose_dtype(self.use_bfloat16)
        self._scoring = AutoModelForCausalLM.from_pretrained(
            self.score_model, trust_remote_code=True,
            torch_dtype=dtype, device_map={"": self._dev}, token=HF_TOKEN
        ).eval()

        if self.reference_model == self.score_model:
            self._reference = self._scoring
        else:
            self._reference = AutoModelForCausalLM.from_pretrained(
                self.reference_model, trust_remote_code=True,
                torch_dtype=dtype, device_map={"": self._dev}, token=HF_TOKEN
            ).eval()

        self._tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token

        super().load()

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        返回“原始分数”：score_raw = discrepancy(x) = (LASTDE(x) - μ_tilde)/σ_tilde
        （天然方向“越大越像 AI”）
        """
        scores_out: List[float] = []
        for text in texts:
            # -- 编码 --
            enc = self._tok(
                [text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                # 为了和 Lastde 保持一致、也避免只保留特殊符号导致 T=0，这里显式不添加
                add_special_tokens=False,
            )
            enc = {k: (v.to(self._dev) if hasattr(v, "to") else v) for k, v in enc.items()}

            # 序列长度（含首 token），用于判断 T = L - 1
            L = int(enc["input_ids"].shape[1])
            T_safe = max(0, L - 1)
            if T_safe == 0:
                # 文本过短（例如只一个 token 或被截断到 1 个 token），
                # 无法构造时间维度 -> 返回一个“中性分数”（0.0），避免中断标定流程
                scores_out.append(0.0)
                continue

            # -- 前向 --
            logits_score = self._scoring(**enc).logits[:, :-1, :]  # [1, T, V]，此时 T>=1
            if logits_score.shape[1] == 0:
                scores_out.append(0.0)
                continue

            if (self._reference is self._scoring):
                logits_ref = logits_score
            else:
                logits_ref = self._reference(**enc).logits[:, :-1, :]

            labels = enc["input_ids"][:, 1:]                        # [1, T]
            # 保障 labels 同步（极端情况下再兜底一次）
            if labels.shape[1] == 0:
                scores_out.append(0.0)
                continue

            # x 的 log-likelihood 张量（scoring）
            ll_x = _official_log_likelihood_tensor(logits_score, labels)  # [1, T, 1]

            # 参考分布：当 T>=1 时才构造 Categorical；否则已在上面 continue
            lprobs_ref = torch.log_softmax(logits_ref, dim=-1)            # [1, T, V]
            # 这里 T>=1，构造合法
            cat = torch.distributions.Categorical(logits=lprobs_ref)
            samples = cat.sample([self.n_samples]).permute(1, 2, 0)       # [1, T, S]

            # 采样序列在 scoring 下的 log-likelihood
            ll_xt = _official_log_likelihood_tensor(logits_score, samples)  # [1, T, S]

            # ---- 关键修复：epsilon 设下限，避免 0/1 分箱 ----
            T = int(ll_x.shape[1])
            eps_x  = max(2, int(self.epsilon_mult * T))
            eps_xt = max(2, int(self.epsilon_mult * T))

            # 计算 LASTDE
            lastde_x = _official_lastde_from_ll(
                ll_x, embed_size=self.embed_size, epsilon=eps_x,  tau_prime=self.tau_prime
            )                     # [1, 1]
            sampled_lastde = _official_lastde_from_ll(
                ll_xt, embed_size=self.embed_size, epsilon=eps_xt, tau_prime=self.tau_prime
            )                     # [1, S]

            mu_tilde = sampled_lastde.mean()
            sigma_tilde = sampled_lastde.std()
            if float(sigma_tilde) == 0.0:
                sigma_tilde = torch.tensor(1.0, device=mu_tilde.device, dtype=mu_tilde.dtype)

            discrepancy = (lastde_x.view([]) - mu_tilde) / sigma_tilde   # 标量
            score_raw = float(discrepancy.item())                        # 更大=更像AI
            scores_out.append(score_raw)

        return np.asarray(scores_out, dtype=np.float32)