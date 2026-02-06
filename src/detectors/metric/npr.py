# mgt_eval/detectors/metric/npr.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from ..base import DetectorBase
from ..registry import register
import logging

# （）
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# DetectGPT ：T5 sentinel token （<extra_id_X>）
_EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")

# ===========================
# /
# ===========================
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


# ---- model loader with SDPA (fallback if unsupported) ----
def _from_pretrained_sdpa(cls, *args, **kwargs):
    """
    Try SDPA first; if unsupported, fall back to eager, then to default.
    Also prefer `dtype` (new) over `torch_dtype` (deprecated), with fallback.
    """
    logger = logging.getLogger("mgt_eval")
    model_id = args[0] if args else "<unknown>"
    dtype = kwargs.pop("dtype", None)
    torch_dtype = kwargs.pop("torch_dtype", None)
    dtype = dtype if dtype is not None else torch_dtype

    def _call(attn_impl: Optional[str], use_dtype: bool):
        call_kwargs = dict(kwargs)
        if attn_impl is not None:
            call_kwargs["attn_implementation"] = attn_impl
        if dtype is not None:
            if use_dtype:
                call_kwargs["dtype"] = dtype
            else:
                call_kwargs["torch_dtype"] = dtype
        return cls.from_pretrained(*args, **call_kwargs)

    # 1) SDPA
    try:
        try:
            return _call("sdpa", use_dtype=True)
        except TypeError:
            return _call("sdpa", use_dtype=False)
    except Exception as e:
        logger.warning("[mgt_eval] attn_implementation=sdpa not supported for %s; fallback to eager. (%s)", model_id, e)

    # 2) eager
    try:
        try:
            return _call("eager", use_dtype=True)
        except TypeError:
            return _call("eager", use_dtype=False)
    except Exception as e:
        logger.warning("[mgt_eval] attn_implementation=eager failed for %s; fallback to default. (%s)", model_id, e)

    # 3) default
    try:
        try:
            return _call(None, use_dtype=True)
        except TypeError:
            return _call(None, use_dtype=False)
    except Exception:
        # let the original exception bubble up
        return cls.from_pretrained(*args, **kwargs)


# —— ： tokenizer  max_len  token ——
def _truncate_by_tokenizer(text: str, tok, max_len: int) -> str:
    """
    使用给定 tokenizer 将文本截断到前 max_len 个 token（上限不超过 tokenizer.model_max_length），
    再 decode 回字符串。整个过程始终在 truncation=True 下进行，避免 tokenizer 警告。
    """
    max_len = int(max_len)
    if max_len <= 0:
        return ""
    hard_cap = getattr(tok, "model_max_length", max_len)
    cap = min(max_len, hard_cap)

    enc = tok(text, add_special_tokens=False, truncation=True, max_length=cap)
    ids = enc["input_ids"][:cap]
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


# （ DetectGPT  _hard_truncate_texts_by_tok ）
def _hard_truncate_texts_by_tok(tok: AutoTokenizer, texts: List[str], max_len: int) -> List[str]:
    if len(texts) == 0:
        return []
    enc = tok(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = enc["input_ids"]
    return [tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in input_ids]


# ===========================
# LRR/NPR
# ===========================
def _get_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    逐 token 平均对数似然（单样本）：
      - logits: (1, T, V) -> (T, V)
      - labels: (1, T)    -> (T,)
    返回：mean(log p(label_t | x_{<t}))
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])   # (T, V)
    labels = labels.view(-1)                     # (T,)
    log_probs = F.log_softmax(logits, dim=-1)    # (T, V)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (T,)
    return log_likelihood.mean().item()


def _get_logrank(logits: torch.Tensor, labels: torch.Tensor, use_log: bool = True) -> float:
    """
    Log-Rank（单样本）：
      - 对每个时间步 t，计算真实标签在降序排序下的秩 rank_t（1-indexed）。
      - 若 use_log=True，返回 mean(log(rank_t))；否则返回 mean(rank_t)。
      - 要求每个时间步恰好匹配一次。
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # 1-indexed
    if use_log:
        ranks = torch.log(ranks)
    return ranks.mean().item()


def get_rank(text: str, args, model_config: Dict[str, Any], log: bool = True) -> float:
    """
    仅对前 args.max_len 个 token 评估（上限 400）。
    若外部已截断，继续在 tokenizer 侧做一次安全截断是幂等的（不会扩张）。
    """
    with torch.no_grad():
        tok = model_config["score_tokenizer"]
        mdl = model_config["score_model"]
        device = args.DEVICE
        max_len = min(int(args.max_len), 400)

        # tokenizer  max_len tokens（）
        text = _truncate_by_tokenizer(text, tok, max_len)

        enc = tok(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.dtype not in (torch.int32, torch.int64):
            input_ids = input_ids.long()
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        tokenized = {"input_ids": input_ids}
        if attn is not None:
            tokenized["attention_mask"] = attn

        # NaN，
        if input_ids.numel() <= 1:
            return float("nan")

        labels = input_ids[:, 1:]
        logits = mdl(**tokenized).logits[:, :-1, :]
        return _get_logrank(logits, labels, use_log=log)


def get_ranks(texts: List[str], args, model_config: Dict[str, Any], log: bool = True) -> List[float]:
    out: List[float] = []
    for t in texts:
        out.append(get_rank(t, args, model_config, log=log))
    return out


# ===========================
# DetectGPT ：T5 （）
# ===========================
def _tokenize_and_mask(text: str, span_length: int, pct: float, buffer_size: int) -> str:
    """
    与 DetectGPT 保持一致的近似策略：
      - 将文本按空格切分；
      - 随机选 n_spans 个长度为 span_length 的片段，用 <extra_id_i> 掩码；
      - 避免掩码过于密集（buffer_size）。
    """
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = int(pct * len(tokens) / max(1, (span_length + buffer_size * 2)))
    n_masks = 0
    L = len(tokens)
    if L <= 0 or span_length <= 0:
        return text

    while n_masks < n_spans and L >= span_length:
        start = np.random.randint(0, max(1, L - span_length))
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end   = min(L, end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
            L = len(tokens)

    # mask  <extra_id_0>..<extra_id_{n-1}>
    num_filled = 0
    for i, tk in enumerate(tokens):
        if tk == mask_string:
            tokens[i] = f"<extra_id_{num_filled}>"
            num_filled += 1

    return " ".join(tokens)


def _count_masks(texts: List[str]) -> List[int]:
    return [sum(1 for t in txt.split(" ") if t.startswith("<extra_id_")) for txt in texts]


def _replace_masks_with_t5(
    texts: List[str],
    mask_model: AutoModelForSeq2SeqLM,
    mask_tokenizer: AutoTokenizer,
    device: str,
    top_p: float = 1.0,
    max_gen_len: int = 150,
    max_len: int = 256,  # （）
) -> List[str]:
    if len(texts) == 0:
        return []

    n_expected = _count_masks(texts)
    # DetectGPT ： <extra_id_k>  eos_token_id
    stop_ids = mask_tokenizer.encode(
        f"<extra_id_{max(n_expected) if len(n_expected) > 0 else 0}>",
        add_special_tokens=False
    )
    stop_id = stop_ids[0] if isinstance(stop_ids, list) and len(stop_ids) > 0 else mask_tokenizer.eos_token_id

    tokens = mask_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    with torch.inference_mode():
        outputs = mask_model.generate(
            **tokens,
            max_length=max_gen_len,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def _extract_fills_from_decoded(decoded_texts: List[str]) -> List[List[str]]:
    cleaned = [x.replace("<pad>", "").replace("</s>", "").strip() for x in decoded_texts]
    fills = [_EXTRA_ID_PATTERN.split(x)[1:-1] for x in cleaned]
    fills = [[seg.strip() for seg in segs] for segs in fills]
    return fills


def _apply_fills(masked_texts: List[str], extracted_fills: List[List[str]]) -> List[str]:
    tokens_list = [txt.split(" ") for txt in masked_texts]
    n_expected = _count_masks(masked_texts)

    for idx, (tokens, fills, n) in enumerate(zip(tokens_list, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens_list[idx] = []  # ，
        else:
            ok = True
            for i in range(n):
                try:
                    j = tokens.index(f"<extra_id_{i}>")
                    tokens[j] = fills[i]
                except ValueError:
                    ok = False
                    break
            if not ok:
                tokens_list[idx] = []
    return [" ".join(toks) if len(toks) > 0 else "" for toks in tokens_list]


def _perturb_texts_with_t5(
    texts: List[str],
    span_length: int,
    pct: float,
    buffer_size: int,
    mask_model: AutoModelForSeq2SeqLM,
    mask_tokenizer: AutoTokenizer,
    device: str,
    top_p: float = 1.0,
    chunk_size: int = 20,
    max_retry: int = 2,
    t5_max_len: int = 256,
) -> List[str]:
    masked_texts = [_tokenize_and_mask(x, span_length, pct, buffer_size) for x in texts]
    outputs = [""] * len(masked_texts)

    eff_chunk = max(
        1,
        chunk_size // 2 if "11b" in str(getattr(mask_model.config, "name_or_path", "")).lower() else chunk_size
    )

    remaining = list(range(len(masked_texts)))
    attempts = 0
    while len(remaining) > 0 and attempts <= max_retry:
        batch_idx = remaining
        remaining = []
        for i in range(0, len(batch_idx), eff_chunk):
            idxs = batch_idx[i: i + eff_chunk]
            sub_texts = [masked_texts[j] for j in idxs]

            decoded = _replace_masks_with_t5(
                sub_texts, mask_model, mask_tokenizer, device,
                top_p=top_p, max_len=t5_max_len
            )
            fills = _extract_fills_from_decoded(decoded)
            perturbed = _apply_fills(sub_texts, fills)

            for j, ptxt in zip(idxs, perturbed):
                if ptxt == "":
                    remaining.append(j)
                else:
                    outputs[j] = ptxt

        attempts += 1
        for j in remaining:
            masked_texts[j] = _tokenize_and_mask(texts[j], span_length, pct, buffer_size)

    # ：
    for i, o in enumerate(outputs):
        if o == "":
            outputs[i] = texts[i]
            print("Perturb failed, return to original sentence.")
    return outputs


# ===========================
# NPR Detector
# ===========================
@register("npr")
class NPRDetector(DetectorBase):
    """
    度量型（Metric-based）检测器：NPR（Normalized Log-Rank Perturbation）
    - 思路：对原文本进行掩码-填充扰动（T5），计算扰动后文本的 log-rank（取均值），
            再与未扰动文本的 log-rank 作比值：  score = mean_logrank(perturbs) / logrank(original)
    - 扰动 token 的处理已经与 DetectGPT 保持一致。

    Args:
        score_model: 评分模型（CausalLM）
        mask_model: 掩码填充模型（T5）
        pct_words_masked: 掩码比例
        span_length: 掩码 span 长度
        n_perturbation: 每条文本生成的扰动条数
        chunk_size: 批量扰动生成的块大小
        buffer_size: 掩码邻域缓冲
        mask_top_p: 生成时的 top-p
        use_bfloat16: CUDA 上是否使用 bf16（仅用于 scoring 模型）
        max_len: 评分与扰动统一使用的最大 token 数（强制 ≤ 400)
        device: 设备
    """

    # ====== （ base.evaluate() ，）======
    CITATION_TITLE = "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text"
    CITATION_AUTHORS = "Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov"
    # ，
    CITATION_LINK = "https://arxiv.org/abs/2306.05540"

    def __init__(
        self,
        score_model: str,
        mask_model: str = "t5-small",
        pct_words_masked: float = 0.3,
        span_length: int = 2,
        n_perturbation: int = 100,
        chunk_size: int = 20,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        use_bfloat16: bool = True,
        mask_use_bfloat16: Optional[bool] = None,
        max_len: int = 400,
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            mask_model=mask_model,
            pct_words_masked=pct_words_masked,
            span_length=span_length,
            n_perturbation=n_perturbation,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
            mask_top_p=mask_top_p,
            use_bfloat16=use_bfloat16,
            mask_use_bfloat16=mask_use_bfloat16,
            max_len=max_len,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.mask_model = mask_model
        self.pct_words_masked = float(pct_words_masked)
        self.span_length = int(span_length)
        self.n_perturbation = int(n_perturbation)
        self.chunk_size = int(chunk_size)
        self.buffer_size = int(buffer_size)
        self.mask_top_p = float(mask_top_p)
        self.use_bfloat16 = bool(use_bfloat16)
        self.mask_use_bfloat16 = bool(use_bfloat16) if mask_use_bfloat16 is None else bool(mask_use_bfloat16)
        # max_len  token（ ≤400）
        self.max_len = int(min(max_len, 400))
        self.user_device = device

        # /
        base_name = os.path.basename(self.score_model.rstrip("/"))
        self.name = name or f"NPR[{base_name}|{os.path.basename(self.mask_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"
        self._scoring_tok = None
        self._scoring_model = None
        self._mask_tok = None
        self._mask_model = None
        self._dev = "cpu"
        self._dtype = _torch_dtype(self.use_bfloat16)
        self._mask_dtype = _torch_dtype(self.mask_use_bfloat16)
        self.is_loaded = False

    def load(self):
        # 1)
        self._dev = _select_device(self.user_device)
        self._dtype = _torch_dtype(self.use_bfloat16)
        self._mask_dtype = _torch_dtype(self.mask_use_bfloat16)

        # 2) （CausalLM）
        self._scoring_model = _from_pretrained_sdpa(
            AutoModelForCausalLM,
            self.score_model,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map={"": self._dev},
            token=HF_TOKEN,
        )
        self._scoring_model.eval()
        self._scoring_tok = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._scoring_tok, "pad_token", None) is None and getattr(self._scoring_tok, "eos_token", None) is not None:
            self._scoring_tok.pad_token = self._scoring_tok.eos_token

        # 3) （T5）
        self._mask_model = _from_pretrained_sdpa(
            AutoModelForSeq2SeqLM,
            self.mask_model,
            trust_remote_code=True,
            dtype=self._mask_dtype,
            device_map={"": self._dev},
            token=HF_TOKEN,
        )
        self._mask_model.eval()
        self._mask_tok = AutoTokenizer.from_pretrained(self.mask_model, token=HF_TOKEN)
        if getattr(self._mask_tok, "pad_token", None) is None and getattr(self._mask_tok, "eos_token", None) is not None:
            self._mask_tok.pad_token = self._mask_tok.eos_token

        super().load()

    # NPR ， calibrate
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        return scores

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        对每条文本：
          仅取前 max_len (≤400) tokens 进行评估与扰动（与 DetectGPT 的扰动逻辑完全一致）：
            - 先用评分 tokenizer 截断得到 t_trunc；
            - 复制 K 份 → T5 扰动（硬截断 t5_max_len=max_len）；
            - 扰动结果再用评分 tokenizer 截断；
            - 计算扰动均值 log-rank 与原文 log-rank 的比值。
        返回 float32 数组。
        """
        assert self.is_loaded, "Call .load() before scoring."

        # baselines  “args” / “model_config”
        args = SimpleNamespace(
            span_length=self.span_length,
            pct_words_masked=self.pct_words_masked,
            buffer_size=self.buffer_size,
            mask_top_p=self.mask_top_p,
            chunk_size=self.chunk_size,
            mask_model=self.mask_model,
            DEVICE=self._dev,
            max_len=self.max_len,
        )
        model_config: Dict[str, Any] = {
            "score_model": self._scoring_model,
            "score_tokenizer": self._scoring_tok,
            "mask_model": self._mask_model,
            "mask_tokenizer": self._mask_tok,
        }

        out: List[float] = []
        eps = 1e-8

        for t in texts:
            # tokenizer
            t_trunc = _truncate_by_tokenizer(t, self._scoring_tok, self.max_len)

            # log-rank
            orig_logrank = get_rank(t_trunc, args, model_config, log=True)

            # DetectGPT ： T5 ，///
            # K
            expanded = [t_trunc for _ in range(self.n_perturbation)]
            # T5 ， T5 tokenizer （ self.max_len ；≤400）
            expanded = _hard_truncate_texts_by_tok(self._mask_tok, expanded, self.max_len)

            perturbed = _perturb_texts_with_t5(
                expanded,
                span_length=self.span_length,
                pct=self.pct_words_masked,
                buffer_size=self.buffer_size,
                mask_model=self._mask_model,
                mask_tokenizer=self._mask_tok,
                device=self._dev,
                top_p=self.mask_top_p,
                chunk_size=self.chunk_size,
                t5_max_len=self.max_len,
            )
            # ， tokenizer
            perturbed_eval = _hard_truncate_texts_by_tok(self._scoring_tok, perturbed, self.max_len)

            # log-rank
            p_ranks = get_ranks(perturbed_eval[: self.n_perturbation], args, model_config, log=True)
            p_vals = [v for v in p_ranks if (v is not None and math.isfinite(v))]

            # ：，（score≈1）
            p_mean = float(np.mean(p_vals)) if len(p_vals) > 0 else float(orig_logrank)

            # ：0   →  eps
            denom = orig_logrank if (orig_logrank is not None and math.isfinite(orig_logrank) and abs(orig_logrank) > eps) else eps

            score = p_mean / denom
            if not math.isfinite(score):
                score = 1.0

            out.append(float(score))

        return np.asarray(out, dtype=np.float32)
