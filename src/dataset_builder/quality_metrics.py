# mgt_eval/dataset_builder/quality_metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import re

# --------------------------
# Readability helpers
# --------------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_RE = re.compile(r"[.!?]+")
_VOWEL_RE = re.compile(r"[aeiouy]+")


def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    # simple sentence splitting
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    sents = [p.strip() for p in parts if p and p.strip()]
    if not sents:
        # fallback: punct-based
        sents = [p.strip() for p in _SENT_RE.split(t) if p.strip()]
    return sents or [t]


def _extract_words(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    words = _WORD_RE.findall(t)
    if words:
        return words
    # fallback: whitespace
    return [w for w in re.split(r"\s+", t) if w]


def _count_syllables_word(word: str) -> int:
    """
    Heuristic syllable count for English-like words.
    """
    w = (word or "").lower()
    if not w:
        return 0
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    # silent 'e'
    if w.endswith("e") and len(w) > 2:
        w = w[:-1]
    groups = _VOWEL_RE.findall(w)
    n = len(groups)
    return max(1, n)


def readability_metrics(text: str) -> Dict[str, Any]:
    """
    Returns readability-related metrics (primarily meaningful for English).
    Still returns generic stats for non-English.
    """
    t = (text or "").strip()
    if not t:
        return {
            "sentences": 0,
            "words": 0,
            "chars": 0,
            "avg_word_len": None,
            "avg_sent_len": None,
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
        }

    sents = _split_sentences(t)
    words = _extract_words(t)
    chars = len(t)

    n_sent = max(1, len(sents))
    n_word = len(words)

    avg_word_len = (sum(len(w) for w in words) / n_word) if n_word > 0 else None
    avg_sent_len = (n_word / n_sent) if n_sent > 0 else None

    # syllables (only for "word-like" tokens)
    syll = sum(_count_syllables_word(w) for w in words) if n_word > 0 else 0

    fre = None
    fkg = None
    if n_word > 0 and n_sent > 0:
        # Flesch Reading Ease / Flesch-Kincaid Grade
        fre = 206.835 - 1.015 * (n_word / n_sent) - 84.6 * (syll / max(1, n_word))
        fkg = 0.39 * (n_word / n_sent) + 11.8 * (syll / max(1, n_word)) - 15.59

    return {
        "sentences": int(len(sents)),
        "words": int(n_word),
        "chars": int(chars),
        "avg_word_len": float(avg_word_len) if avg_word_len is not None else None,
        "avg_sent_len": float(avg_sent_len) if avg_sent_len is not None else None,
        "syllables": int(syll),
        "flesch_reading_ease": float(fre) if fre is not None else None,
        "flesch_kincaid_grade": float(fkg) if fkg is not None else None,
    }


# --------------------------
# Config
# --------------------------
@dataclass
class QualityConfig:
    # enable flags
    enable_ppl: bool = False
    enable_readability: bool = False
    enable_bertscore: bool = False

    # prompt-source control (optional)
    only_human_prompts: bool = False  # handled by CLI/builder selection

    # PPL (Causal LM)
    ppl_model: str = "gpt2"
    ppl_device: str = "cuda:0"
    ppl_dtype: str = "auto"  # auto|float16|bfloat16|float32
    ppl_stride: int = 256
    ppl_max_length: int = 1024

    # BERTScore
    bertscore_model: str = "roberta-large"
    bertscore_device: str = "cuda:0"
    bertscore_lang: str = "en"
    bertscore_batch_size: int = 8
    bertscore_rescale: bool = True

    # internal: max length for encoder fallback
    encoder_max_length: int = 512

    def any_enabled(self) -> bool:
        return bool(self.enable_ppl or self.enable_readability or self.enable_bertscore)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------
# Evaluator
# --------------------------
class TextQualityEvaluator:
    """
    Computes:
      - PPL for each text
      - Readability for each text
      - BERTScore for (ref, cand) pairs (ref typically original, cand generated)

    Implementation notes:
      - BERTScore uses `bert-score` if installed; otherwise falls back to
        embedding cosine similarity (marked as impl='approx_cosine').
    """

    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg

        self._ppl_tok = None
        self._ppl_lm = None

        self._bs_impl = None  # "bert_score" | "approx_cosine" | None
        self._bs_tok = None
        self._bs_model = None

        self._torch = None

    # ---------- Torch / dtype helpers ----------
    def _lazy_import_torch(self):
        if self._torch is None:
            import torch  # type: ignore
            self._torch = torch
        return self._torch

    def _resolve_dtype(self, dtype: str):
        torch = self._lazy_import_torch()
        d = (dtype or "auto").lower().strip()
        if d == "float16":
            return torch.float16
        if d == "bfloat16":
            return torch.bfloat16
        if d == "float32":
            return torch.float32
        return None  # auto

    # ---------- PPL ----------
    def _ensure_ppl_model(self):
        if not self.cfg.enable_ppl:
            return
        if self._ppl_lm is not None and self._ppl_tok is not None:
            return

        torch = self._lazy_import_torch()
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError("PPL 需要 transformers 依赖：pip install transformers") from e

        tok = AutoTokenizer.from_pretrained(self.cfg.ppl_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dtype = self._resolve_dtype(self.cfg.ppl_dtype)
        lm = AutoModelForCausalLM.from_pretrained(
            self.cfg.ppl_model,
            torch_dtype=dtype,
        )
        lm.to(self.cfg.ppl_device)
        lm.eval()

        # some models need pad_token_id set
        try:
            if getattr(lm.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
                lm.config.pad_token_id = tok.pad_token_id
        except Exception:
            pass

        self._ppl_tok = tok
        self._ppl_lm = lm

    def ppl(self, text: str) -> Optional[float]:
        if not self.cfg.enable_ppl:
            return None
        t = (text or "").strip()
        if not t:
            return None

        self._ensure_ppl_model()

        torch = self._lazy_import_torch()
        tok = self._ppl_tok
        lm = self._ppl_lm
        assert tok is not None and lm is not None

        enc = tok(t, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.cfg.ppl_device)

        seq_len = int(input_ids.size(1))
        if seq_len <= 1:
            return None

        # determine model max length
        model_max = None
        for k in ("n_positions", "max_position_embeddings"):
            v = getattr(lm.config, k, None)
            if isinstance(v, int) and v > 0:
                model_max = v
                break
        max_len = int(min(self.cfg.ppl_max_length, model_max or self.cfg.ppl_max_length))
        stride = int(max(1, self.cfg.ppl_stride))

        nlls = []
        total_toks = 0

        with torch.no_grad():
            for i in range(0, seq_len, stride):
                begin = max(i + stride - max_len, 0)
                end = min(i + stride, seq_len)
                trg_len = end - i
                if trg_len <= 0:
                    continue

                input_slice = input_ids[:, begin:end]
                target_ids = input_slice.clone()
                # only compute loss on last trg_len tokens
                if target_ids.size(1) > trg_len:
                    target_ids[:, :-trg_len] = -100

                out = lm(input_ids=input_slice, labels=target_ids)
                loss = out.loss
                if loss is None:
                    continue
                nlls.append(loss * trg_len)
                total_toks += trg_len

        if total_toks <= 0 or not nlls:
            return None

        ppl = float(math.exp(torch.stack(nlls).sum().item() / total_toks))
        return ppl

    # ---------- Readability ----------
    def readability(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.enable_readability:
            return None
        return readability_metrics(text)

    # ---------- BERTScore ----------
    def _ensure_bertscore(self):
        if not self.cfg.enable_bertscore:
            return
        if self._bs_impl is not None:
            return

        # Try official bert-score
        try:
            import bert_score  # type: ignore
            self._bs_impl = "bert_score"
            return
        except Exception:
            pass

        # fallback: embedding cosine similarity
        self._bs_impl = "approx_cosine"

        torch = self._lazy_import_torch()
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "BERTScore fallback(approx_cosine) 需要 transformers；"
                "若要真实 BERTScore请安装 bert-score：pip install bert-score"
            ) from e

        tok = AutoTokenizer.from_pretrained(self.cfg.bertscore_model, use_fast=True)
        model = AutoModel.from_pretrained(self.cfg.bertscore_model)
        model.to(self.cfg.bertscore_device)
        model.eval()

        self._bs_tok = tok
        self._bs_model = model

    def bertscore_pairs(self, refs: List[str], cands: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        returns list of dict per pair:
          - if impl='bert_score': {"p":..., "r":..., "f1":..., "impl": "bert_score"}
          - if impl='approx_cosine': {"f1":..., "impl": "approx_cosine"}  (p/r not available)
        """
        if not self.cfg.enable_bertscore:
            return None
        if not refs or not cands:
            return None
        if len(refs) != len(cands):
            raise ValueError("refs and cands must have same length")

        self._ensure_bertscore()
        impl = self._bs_impl

        if impl == "bert_score":
            # real bert-score
            try:
                from bert_score import score as bert_score_score  # type: ignore
            except Exception as e:
                raise RuntimeError("未能导入 bert-score.score，请确认 pip install bert-score") from e

            torch = self._lazy_import_torch()
            with torch.no_grad():
                P, R, F1 = bert_score_score(
                    cands,
                    refs,
                    model_type=self.cfg.bertscore_model,
                    lang=self.cfg.bertscore_lang,
                    device=self.cfg.bertscore_device,
                    verbose=False,
                    rescale_with_baseline=bool(self.cfg.bertscore_rescale),
                )
            out: List[Dict[str, Any]] = []
            for p, r, f1 in zip(P.tolist(), R.tolist(), F1.tolist()):
                out.append({"p": float(p), "r": float(r), "f1": float(f1), "impl": "bert_score"})
            return out

        # approx cosine fallback
        torch = self._lazy_import_torch()
        tok = self._bs_tok
        model = self._bs_model
        assert tok is not None and model is not None

        def _encode(texts: List[str]):
            batch = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=int(self.cfg.encoder_max_length),
                return_tensors="pt",
            )
            for k in batch:
                batch[k] = batch[k].to(self.cfg.bertscore_device)
            with torch.no_grad():
                out = model(**batch)
                h = out.last_hidden_state  # [B, T, H]
                mask = batch.get("attention_mask", None)
                if mask is None:
                    emb = h.mean(dim=1)
                else:
                    m = mask.unsqueeze(-1).to(h.dtype)
                    emb = (h * m).sum(dim=1) / (m.sum(dim=1).clamp(min=1.0))
                # normalize
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return emb  # [B, H]

        # batch encode both sides
        ref_emb = _encode(refs)
        cand_emb = _encode(cands)
        sim = (ref_emb * cand_emb).sum(dim=-1)  # cosine due to normalize
        out = [{"f1": float(x), "impl": "approx_cosine"} for x in sim.tolist()]
        return out

    # ---------- Unified ----------
    def eval_original(self, text: str) -> Dict[str, Any]:
        """
        metrics for original text only (NEVER raise; record errors instead)
        """
        out: Dict[str, Any] = {}

        # readability never relies on HF models -> put first
        if self.cfg.enable_readability:
            try:
                out["readability"] = readability_metrics(text)
            except Exception as e:
                out["readability"] = None
                out["readability_error"] = str(e)

        if self.cfg.enable_ppl:
            try:
                out["ppl"] = self.ppl(text)
            except Exception as e:
                out["ppl"] = None
                out["ppl_error"] = str(e)

        return out

    def eval_samples(self, ref_text: str, sample_texts: List[str]) -> List[Dict[str, Any]]:
        """
        metrics for each sample text; includes bertscore vs ref_text if enabled
        NEVER raise; each metric failure won't block others
        """
        outs: List[Dict[str, Any]] = [{"_ok": True} for _ in sample_texts]
        if not sample_texts:
            return outs

        # per-sample readability/ppl
        if self.cfg.enable_readability:
            for i, t in enumerate(sample_texts):
                try:
                    outs[i]["readability"] = readability_metrics(t)
                except Exception as e:
                    outs[i]["readability"] = None
                    outs[i]["readability_error"] = str(e)

        if self.cfg.enable_ppl:
            for i, t in enumerate(sample_texts):
                try:
                    outs[i]["ppl"] = self.ppl(t)
                except Exception as e:
                    outs[i]["ppl"] = None
                    outs[i]["ppl_error"] = str(e)

        # bertscore (batch)
        if self.cfg.enable_bertscore:
            try:
                refs = [ref_text] * len(sample_texts)
                bs = self.bertscore_pairs(refs, sample_texts) or []
                for i, b in enumerate(bs):
                    outs[i]["bertscore"] = {
                        **b,
                        "ref": "original",
                        "model": self.cfg.bertscore_model,
                        "lang": self.cfg.bertscore_lang,
                    }
            except Exception as e:
                # mark all samples as failed for bertscore, but keep ppl/readability
                for i in range(len(sample_texts)):
                    outs[i]["bertscore"] = None
                    outs[i]["bertscore_error"] = str(e)

        # cleanup internal marker
        for o in outs:
            o.pop("_ok", None)
        return outs