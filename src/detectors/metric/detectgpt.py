# mgt_eval/detectors/metric/detectgpt.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import re
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from ..base import DetectorBase
from ..registry import register

# ===========================
# /（）
# ===========================
CITATION_TITLE   = "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
CITATION_AUTHORS = "Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn"
CITATION_LINK    = "https://arxiv.org/abs/2301.11305"

# （/）
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

# T5 sentinel token （<extra_id_X>）
_EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")

# ===========================
# /
# ===========================
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

def _move_to(device: str, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if k in ("input_ids", "attention_mask"):
                v = v.long()
        out[k] = v
    return out

# ===========================
# ： N  token（ tokenizer ）
# ===========================
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

def _clamp_ctx_len(x: int, lo: int = 8, hi: int = 400) -> int:
    """将用户可配置的截断长度裁剪到 [8, 400]。"""
    try:
        x = int(x)
    except Exception:
        x = lo
    return max(lo, min(x, hi))

# ===========================
# （DetectGPT  T5 ）
# ===========================
def _tokenize_and_mask(text: str, span_length: int, pct: float, buffer_size: int) -> str:
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
    max_len: int = 400,  # ★ ：
) -> List[str]:
    if len(texts) == 0:
        return []

    n_expected = _count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected) if len(n_expected) > 0 else 0}>",
                                    add_special_tokens=False)[0]

    # —— ：T5  max_len ——
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
            tokens_list[idx] = []
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
    t5_max_len: int = 400,  # ★ ：
) -> List[str]:
    masked_texts = [_tokenize_and_mask(x, span_length, pct, buffer_size) for x in texts]
    outputs = [""] * len(masked_texts)

    eff_chunk = max(1, chunk_size // 2 if "11b" in str(getattr(mask_model.config, "name_or_path", "")).lower() else chunk_size)

    remaining = list(range(len(masked_texts)))
    attempts = 0
    while len(remaining) > 0 and attempts <= max_retry:
        batch_idx = remaining
        remaining = []
        for i in range(0, len(batch_idx), eff_chunk):
            idxs = batch_idx[i : i + eff_chunk]
            sub_texts = [masked_texts[j] for j in idxs]

            decoded = _replace_masks_with_t5(sub_texts, mask_model, mask_tokenizer, device,
                                             top_p=top_p, max_len=t5_max_len)
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

    for i, o in enumerate(outputs):
        if o == "":
            outputs[i] = texts[i]  # ：
            print("Perturb failed, return to original sentence.")
    return outputs

# ===========================
# /
# ===========================
def _shift_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_attn   = attention_mask[..., 1:].contiguous()
    return shifted_labels, shifted_attn

def _mean_log_likelihood_from_logits(
    encoding: Dict[str, torch.Tensor],
    logits: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels, shifted_attn = _shift_labels(encoding["input_ids"], encoding["attention_mask"])
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    gathered  = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    denom = shifted_attn.sum(dim=1).clamp_min(1)
    ll = (gathered * shifted_attn).sum(dim=1) / denom
    return ll

def _local_files_only_flag() -> bool:
    return HF_OFFLINE

# ===========================
# DetectGPT Detector
# ===========================
@register("detectgpt")
class DetectGPTDetector(DetectorBase):
    CITATION_TITLE   = CITATION_TITLE
    CITATION_AUTHORS = CITATION_AUTHORS
    CITATION_LINK    = CITATION_LINK

    def __init__(
        self,
        score_model: str = "gpt2-medium",
        mask_model: str = "t5-large",
        pct_words_masked: float = 0.3,
        span_length: int = 2,
        n_perturbations: int = 100,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        use_bfloat16: bool = True,
        max_token_observed: int = 400,      # （）
        use_zscore: bool = False,
        prob_slope: float = 8.0,
        device: Optional[str] = None,
        chunk_size: int = 20,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        # ★★★ ： ★★★
        eval_max_tokens: int = 400,         # （≤400）
        perturb_max_tokens: int = 400,      # （≤400）
        **kwargs,
    ):
        super().__init__(
            score_model=score_model,
            mask_model=mask_model,
            pct_words_masked=pct_words_masked,
            span_length=span_length,
            n_perturbations=n_perturbations,
            buffer_size=buffer_size,
            mask_top_p=mask_top_p,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            use_zscore=use_zscore,
            prob_slope=prob_slope,
            device=device,
            chunk_size=chunk_size,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.mask_model = mask_model
        self.pct = float(pct_words_masked)
        self.span_length = int(span_length)
        self.n_perts = int(n_perturbations)
        self.buffer_size = int(buffer_size)
        self.mask_top_p = float(mask_top_p)
        self.use_bf16 = bool(use_bfloat16)
        self.use_z = bool(use_zscore)
        self.prob_slope = float(prob_slope)
        self.user_device = device
        self.chunk_size = max(1, int(chunk_size))

        # ★  ≤400（ ≥8）
        self._eval_ctx_len    = _clamp_ctx_len(eval_max_tokens)
        self._perturb_ctx_len = _clamp_ctx_len(perturb_max_tokens)

        self.threshold = 0.0
        self.name = name or f"DetectGPT[{os.path.basename(self.score_model)}|{os.path.basename(self.mask_model)}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self._device = "cpu"
        self._dtype = torch.float32
        self._base_tok: Optional[AutoTokenizer] = None
        self._base: Optional[AutoModelForCausalLM] = None
        self._mask_tok: Optional[AutoTokenizer] = None
        self._mask: Optional[AutoModelForSeq2SeqLM] = None
        self.is_loaded = False

    def load(self):
        self._device = _select_device(self.user_device)
        self._dtype = _torch_dtype(self.use_bf16)
        local_only = _local_files_only_flag()

        # （CausalLM）+ tokenizer
        self._base = AutoModelForCausalLM.from_pretrained(
            self.score_model,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map={"": self._device},
            token=HF_TOKEN,
            local_files_only=local_only,
        ).eval()

        self._base_tok = AutoTokenizer.from_pretrained(
            self.score_model,
            token=HF_TOKEN,
            local_files_only=local_only,
        )
        if getattr(self._base_tok, "pad_token", None) is None and getattr(self._base_tok, "eos_token", None) is not None:
            self._base_tok.pad_token = self._base_tok.eos_token

        # ——  tokenizer  model_max_length  eval_max_tokens ——
        try:
            self._base_tok.model_max_length = self._eval_ctx_len
        except Exception:
            pass

        # （T5）+ tokenizer
        self._mask = AutoModelForSeq2SeqLM.from_pretrained(
            self.mask_model,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map={"": self._device},
            token=HF_TOKEN,
            local_files_only=local_only,
        ).eval()

        self._mask_tok = AutoTokenizer.from_pretrained(
            self.mask_model,
            token=HF_TOKEN,
            local_files_only=local_only,
        )
        if getattr(self._mask_tok, "pad_token", None) is None and getattr(self._mask_tok, "eos_token", None) is not None:
            self._mask_tok.pad_token = self._mask_tok.eos_token

        # ——  T5 tokenizer  model_max_length  perturb_max_tokens ——
        try:
            self._mask_tok.model_max_length = self._perturb_ctx_len
        except Exception:
            pass

        super().load()

    @torch.inference_mode()
    def _encode_base(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self._base_tok(
            texts,
            return_tensors="pt",
            padding=True if len(texts) > 1 else False,
            truncation=True,
            max_length=self._eval_ctx_len,
            return_token_type_ids=False,
        )
        enc["input_ids"] = enc["input_ids"].long()
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"].long()
        else:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], dtype=torch.long)
        return enc

    @torch.inference_mode()
    def _ll_batch(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0,), dtype=np.float32)

        bs = max(1, self.chunk_size)
        outs = []
        for i in range(0, len(texts), bs):
            sub = texts[i:i+bs]
            enc = self._encode_base(sub)
            enc = _move_to(self._device, enc)
            outputs = self._base(**enc)
            logits = outputs.logits
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = enc["input_ids"][..., 1:].contiguous()
            shifted_attn   = enc["attention_mask"][..., 1:].contiguous()
            log_probs = torch.log_softmax(shifted_logits, dim=-1)
            gathered  = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
            denom = shifted_attn.sum(dim=1).clamp_min(1)
            ll = (gathered * shifted_attn).sum(dim=1) / denom
            outs.append(ll.detach().cpu().float())
            del enc, outputs, logits, shifted_logits, shifted_labels, shifted_attn, log_probs, gathered, denom, ll
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        return np.clip(scores, 1e-6, 1 - 1e-6)

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        if len(texts) == 0:
            return np.zeros((0,), dtype=np.float32)

        # —— “” ——
        # tokenizer  eval_max_tokens
        texts_eval = _hard_truncate_texts_by_tok(self._base_tok, texts, self._eval_ctx_len)
        # T5 tokenizer  perturb_max_tokens
        texts_perturb_src = _hard_truncate_texts_by_tok(self._mask_tok, texts, self._perturb_ctx_len)

        # 1)  + （ texts_perturb_src）
        expanded = [t for t in texts_perturb_src for _ in range(self.n_perts)]
        perturbed = _perturb_texts_with_t5(
            expanded,
            span_length=self.span_length,
            pct=self.pct,
            buffer_size=self.buffer_size,
            mask_model=self._mask,
            mask_tokenizer=self._mask_tok,
            device=self._device,
            top_p=self.mask_top_p,
            chunk_size=self.chunk_size,
            t5_max_len=self._perturb_ctx_len,
        )
        assert len(perturbed) == len(expanded), "perturbed size mismatch"

        # —— ， ——
        perturbed_eval = _hard_truncate_texts_by_tok(self._base_tok, perturbed, self._eval_ctx_len)

        # 2)  LL
        ll_orig = self._ll_batch(texts_eval)                    # (B,)
        ll_pert = self._ll_batch(perturbed_eval)                # (B * K,)
        ll_pert = ll_pert.reshape(len(texts_eval), self.n_perts)  # (B, K)

        mean_pert = ll_pert.mean(axis=1)                        # (B,)
        std_pert  = ll_pert.std(axis=1) + 1e-8                  # (B,)

        # 3) d  z
        d = ll_orig - mean_pert
        s = d / std_pert if self.use_z else d

        # 4)
        x = self.prob_slope * (s - 0.0)
        probs = 1.0 / (1.0 + np.exp(-x))
        return probs.astype(np.float32)
