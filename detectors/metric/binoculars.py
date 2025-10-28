# mgt_eval/detectors/metric/binoculars.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import DetectorBase
from ..registry import register
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 原论文/参考实现中的常量（两种工作点）
# ===========================
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # 优化 F1
BINOCULARS_FPR_THRESHOLD      = 0.8536432310785527  # 优化低 FPR（~0.01%）

# 环境变量（私有模型需要）
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def _select_devices(user_device: Optional[str]) -> Tuple[str, str]:
    """
    返回 (DEVICE_1, DEVICE_2)：
      - 双卡：('cuda:0', 'cuda:1')
      - 单卡：('cuda:0', 'cuda:0') 或显式 'cpu'
    """
    if user_device:
        d = user_device.strip().lower()
        if d.startswith("cpu"):
            return "cpu", "cpu"
        if d.startswith("cuda"):
            # 若仅 1 张卡，回退同卡
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    return "cuda:0", "cuda:1"
                else:
                    return "cuda:0", "cuda:0"
            else:
                return "cpu", "cpu"
    # 未指定，自动
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return "cuda:0", "cuda:1"
        else:
            return "cuda:0", "cuda:0"
    return "cpu", "cpu"

def _torch_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32

def _assert_tokenizer_consistency(model_id_1: str, model_id_2: str):
    """
    验证两个 tokenizer 的词表一致（严格对齐）。
    """
    tok1 = AutoTokenizer.from_pretrained(model_id_1, token=HF_TOKEN)
    tok2 = AutoTokenizer.from_pretrained(model_id_2, token=HF_TOKEN)
    if getattr(tok1, "vocab", None) is not None and getattr(tok2, "vocab", None) is not None:
        identical = (tok1.vocab == tok2.vocab)
    else:
        # 某些 tokenizer（如 sentencepiece）没有 vocab dict，用特殊 token 集合近似判断
        keys1 = set([str(tok1.cls_token_id), str(tok1.sep_token_id), str(tok1.pad_token_id),
                     str(tok1.eos_token_id), str(tok1.bos_token_id)])
        keys2 = set([str(tok2.cls_token_id), str(tok2.sep_token_id), str(tok2.pad_token_id),
                     str(tok2.eos_token_id), str(tok2.bos_token_id)])
        identical = (keys1 == keys2)
    if not identical:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

def _shift_labels(enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 CausalLM 的标准 next-token 任务：
      - shifted_logits = logits[..., :-1, :]
      - shifted_labels = input_ids[..., 1:]
      - 同时用 attention_mask[..., 1:] 做有效 token 屏蔽
    """
    shifted_labels = enc_input_ids[..., 1:].contiguous()
    shifted_attn   = enc_attention_mask[..., 1:].contiguous()
    return shifted_labels, shifted_attn

def _perplexity(encoding, logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    与参考实现一致：
      - ppl = mean CE over valid tokens（对 batch 中每条序列分别求均值）
    """
    shifted_logits = logits[..., :-1, :].contiguous() / float(temperature)
    shifted_labels, shifted_attn = _shift_labels(encoding.input_ids, encoding.attention_mask)

    # PyTorch CE: (N, C, T) x (N, T)
    ce = F.cross_entropy(
        shifted_logits.transpose(1, 2),  # (B, V, T)
        shifted_labels,                  # (B, T)
        reduction="none"
    )
    # mask & average per-sample
    ce = (ce * shifted_attn).sum(dim=1) / shifted_attn.sum(dim=1).clamp_min(1)
    return ce.detach().cpu().float().numpy()

def _entropy(p_logits: torch.Tensor,
             q_logits: torch.Tensor,
             encoding,
             pad_token_id: int,
             temperature: float = 1.0,
             sample_p: bool = False) -> np.ndarray:
    """
    交叉熵 H_p(q)：把 observer 概率 p 作为“目标分布”，计算 q 的 cross-entropy。
    参考实现中使用了 CrossEntropyLoss(target=probabilities) 的形式，这里显式展开：
      CE(q,p) = - sum_k p_k * log_softmax(q)_k
    """
    B, T, V = q_logits.shape
    p_scores = p_logits / float(temperature)
    q_scores = q_logits / float(temperature)

    # p 概率
    p_logprobs = torch.log_softmax(p_scores, dim=-1)  # (B, T, V)
    p_probs = torch.exp(p_logprobs).view(-1, V)       # (B*T, V)

    if sample_p:
        # 采样 p 的 one-hot 目标（可选；默认 False）
        idx = torch.multinomial(p_probs, num_samples=1, replacement=True).squeeze(-1)  # (B*T,)
        one_hot = torch.zeros_like(p_probs).scatter_(1, idx.unsqueeze(-1), 1.0)
        p_probs = one_hot

    # q 的 log_softmax
    q_logprobs = torch.log_softmax(q_scores, dim=-1).view(-1, V)  # (B*T, V)

    # Token 维度逐位置 CE
    ce_flat = -(p_probs * q_logprobs).sum(dim=-1)  # (B*T,)
    ce = ce_flat.view(B, T)

    # 有效 token mask
    padding_mask = (encoding.input_ids != pad_token_id).to(ce.dtype)  # (B, T)
    agg_ce = ((ce * padding_mask).sum(dim=1) / padding_mask.sum(dim=1).clamp_min(1)).detach().cpu().float().numpy()
    return agg_ce

def _binoculars_score(observer_logits: torch.Tensor,
                      performer_logits: torch.Tensor,
                      encoding,
                      pad_token_id: int) -> np.ndarray:
    """
    binoculars score = ppl(performer) / x_ppl(observer, performer)
    数值越小 -> 越倾向“AI 生成”。
    """
    ppl  = _perplexity(encoding, performer_logits)  # (B,)
    x_ppl = _entropy(observer_logits, performer_logits, encoding, pad_token_id)  # (B,)
    score = ppl / (x_ppl + 1e-12)
    return score

def _logistic_prob_from_score(score: np.ndarray, threshold: float, slope: float = 8.0) -> np.ndarray:
    """
    将 binoculars score （越小越 AI）转为概率：
      p(ai|s) = sigmoid( slope * (threshold - s) )
    这样 s < threshold -> p(ai) > 0.5，与原“阈值判别”等价且连续可微，便于 ROC/AUROC 计算。
    """
    return score

@register("binoculars")
class BinocularsDetector(DetectorBase):
    """
    逻辑型（Logic-based）检测器：Binoculars
    - 载入两套 CausalLM：observer（“观察者”，近似真实分布）与 performer（“表演者”，近似生成分布）
    - 打分：score = ppl(performer) / cross_entropy(observer, performer)
            分数越小，越可能是 AI 文本
    - 概率：默认用以工作点为中心的 Logistic 连续映射（保持与原阈值一致的决策边界）

    Args:
        observer: 观察者模型 id/路径
        performer: 表演者模型 id/路径
        use_bfloat16: CUDA 上使用 bf16
        max_token_observed: 最大 token 长度
        mode: {"low-fpr", "accuracy"} —— 控制默认阈值
        prob_slope: Logistic 概率映射的斜率（越大越接近硬阈值）
        device: 首选设备；若可用多卡自动用 cuda:0 / cuda:1

    备注：
      - 自动校验 tokenizer 一致性；不一致则报错（保持与原实现严格一致）。
      - 双卡：observer→cuda:0，performer→cuda:1；单卡或 CPU 时两者在同一设备。
    """
    # ====== 推荐新增：文献信息（供 base.evaluate() 打印）======
    CITATION_TITLE = "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"
    CITATION_AUTHORS = "Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein"
    # TODO: 将下面替换为官方论文/项目地址（arXiv/GitHub）
    CITATION_LINK = "https://arxiv.org/abs/2401.12070"
    def __init__(
        self,
        observer: str,
        performer: str,
        use_bfloat16: bool = True,
        max_length: int = 512,
        mode: str = "low-fpr",
        prob_slope: float = 8.0,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            observer=observer,
            performer=performer,
            use_bfloat16=use_bfloat16,
            max_length=max_length,
            mode=mode,
            prob_slope=prob_slope,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.observer_name = observer
        self.performer_name = performer
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_len = int(max_length)
        self.mode = str(mode).lower()
        self.user_device = device

        # ① 先根据 mode 设置论文给的阈值
        if self.mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif self.mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # ② 再设置“回退概率映射”参数：p = sigmoid(slope * (T - s))
        #    基类实现为 sigmoid(slope * ((sign*score) - center))
        #    我们令 sign=-1（prob_invert=True），center=-T  =>  (-s) - (-T) = (T - s)
        self.outputs_prob = False
        self.prob_invert = False
        self.prob_center = -float(self.threshold)
        self.prob_slope  = float(prob_slope)

        # 展示信息
        self.name = name or f"Binoculars[{os.path.basename(self.observer_name)}|{os.path.basename(self.performer_name)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        # 运行时对象
        self._tok = None
        self._observer = None
        self._performer = None
        self._dev1 = "cpu"
        self._dev2 = "cpu"
        self.is_loaded = False

    def load(self):
        # 1) 校验 tokenizer 一致
        _assert_tokenizer_consistency(self.observer_name, self.performer_name)

        # 2) 设备选择
        self._dev1, self._dev2 = _select_devices(self.user_device)
        dtype = _torch_dtype(self.use_bfloat16)

        # 3) 模型加载（分别到两个设备；单卡/CPU 则在同一设备）
        self._observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self._dev1},
            token=HF_TOKEN,
        )
        self._performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self._dev2},
            token=HF_TOKEN,
        )
        self._observer.eval()
        self._performer.eval()

        # 4) tokenizer（与 observer 一致）
        self._tok = AutoTokenizer.from_pretrained(self.observer_name, token=HF_TOKEN)
        if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
            self._tok.pad_token = self._tok.eos_token

        super().load()

    @torch.inference_mode()
    def _encode(self, texts: List[str]):
        if not self.is_loaded:
            self.load()
        return self._tok(
            texts,
            return_tensors="pt",
            padding=True if len(texts) > 1 else False,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
        )

    @torch.inference_mode()
    def _get_logits(self, encodings) -> Tuple[torch.Tensor, torch.Tensor]:
        # encodings 发往各自设备
        enc1 = {k: (v.to(self._dev1) if hasattr(v, "to") else v) for k, v in encodings.items()}
        enc2 = {k: (v.to(self._dev2) if hasattr(v, "to") else v) for k, v in encodings.items()}
        obs_logits = self._observer(**enc1).logits  # (B, T, V) on dev1
        per_logits = self._performer(**enc2).logits # (B, T, V) on dev2
        # 若是多卡，需要同步
        if self._dev1.startswith("cuda") or self._dev2.startswith("cuda"):
            torch.cuda.synchronize()
        return obs_logits, per_logits

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        enc = self._encode(texts)
        obs_logits, per_logits = self._get_logits(enc)
        if self._dev2 != self._dev1:
            per_logits = per_logits.to(self._dev1)

        scores = _binoculars_score(
            observer_logits=obs_logits,
            performer_logits=per_logits,
            encoding=enc.to(self._dev1) if hasattr(enc, "to") else enc,
            pad_token_id=self._tok.pad_token_id if self._tok.pad_token_id is not None else 0,
        )  # 原始分数：越小越像 AI

        # ★ 关键：这里返回“原始分数”，不要自己做概率映射！
        # 映射交给父类 DetectorBase.calibrate():
        #   - 若有 calibrator_path：用 JSON 的 LR；
        #   - 否则用我们在 __init__ 配好的 fallback logistic (T - s)。
        return -scores.astype(np.float32)
