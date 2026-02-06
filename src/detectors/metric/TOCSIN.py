# mgt_eval/detectors/metric/TOCSIN.py
# Strictly follows the provided reference script for scoring.
# Differences only in engineering:
#  - We precompute perturbations for the whole batch (like the original script does for the dataset),
#    then compute per-sample scores.
#  - Probabilities are produced with the SAME logistic mapping style as Binoculars:
#        p = sigmoid( prob_slope * (prob_threshold - score) )

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
)

from ..base import DetectorBase
from ..registry import register

# ===========================
# Constants as in the script
# ===========================
TOCSIN_MASK_PCT: float = 0.015
TOCSIN_NSAMPLES: int = 10000
TOCSIN_PERTURB_PER_TEXT: int = 10
TOCSIN_BART_CHECKPOINT: str = "facebook/bart-base"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ===========================
# Device / dtype
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

# ===========================
# Perturbation utilities (same logic as your script)
# ===========================
def _fill_and_mask(text: str, pct: float = TOCSIN_MASK_PCT) -> List[int]:
    tokens = text.split(" ")
    n_spans = int(pct * len(tokens))
    if n_spans <= 0:
        return []
    indices = np.random.choice(range(len(tokens)), size=n_spans)
    return indices.tolist()

def _apply_extracted_fills(texts: List[str], indices_list: List[List[int]]) -> List[str]:
    token_lists = [t.split(" ") for t in texts]
    for toks, idxs in zip(token_lists, indices_list):
        for i in idxs:
            if 0 <= i < len(toks):
                toks[i] = ""
    return [" ".join(toks) for toks in token_lists]

def _perturb_texts_block(texts: List[str], pct: float) -> List[str]:
    """
    Equivalent to `perturb_texts_` in your script:
    - For a block of texts, compute indices_list then apply deletions.
    """
    indices_list = [_fill_and_mask(x, pct) for x in texts]
    return _apply_extracted_fills(texts, indices_list)

def _perturb_texts_dataset_style(texts: List[str], pct: float, k: int, chunk: int = 50) -> List[str]:
    """
    EXACTLY mirrors your dataset-level pipeline:
      outputs = []
      for i in range(0, len(texts), 50):
          outputs.extend(perturb_texts_(texts[i:i + 50], pct))
      return outputs
    Here `texts` is already the repeated list [t1,t1,...,tk, t2,t2,...,tk, ...].
    """
    outputs: List[str] = []
    for i in range(0, len(texts), chunk):
        block = texts[i:i + chunk]
        outputs.extend(_perturb_texts_block(block, pct))
    return outputs

# ===========================
# Scoring utilities (same as your script)
# ===========================
def _get_samples(logits: torch.Tensor, labels: torch.Tensor, nsamples: int = TOCSIN_NSAMPLES) -> torch.Tensor:
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples_2 = distrib.sample([nsamples]).permute([1, 2, 0])  # (1, T, nsamp)
    return samples_2

def _get_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)  # (1, 1) or (1, nsamp)

def _get_logrank(logits: torch.Tensor, labels: torch.Tensor) -> float:
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dims, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
    ranks = ranks.float() + 1.0
    ranks = torch.log(ranks)
    val = (-ranks.mean()).item()  # ：≤ 0
    # NEW:  0  ，，
    if not math.isfinite(val) or abs(val) < 1e-12:
        val = -1e-12
    return float(val)

class _BARTScorer:
    def __init__(self, device: str = "cuda:0", max_length: int = 1024, checkpoint: str = TOCSIN_BART_CHECKPOINT):
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        self.loss_fct = nn.NLLLoss(reduction="none", ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path: Optional[str] = None):
        if path is None:
            path = "models/bart.pth"
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 10) -> List[float]:
        out: List[float] = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                enc_src = self.tokenizer(src_list, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
                enc_tgt = self.tokenizer(tgt_list, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
                src_tokens = enc_src["input_ids"].to(self.device)
                src_mask = enc_src["attention_mask"].to(self.device)
                tgt_tokens = enc_tgt["input_ids"].to(self.device)
                tgt_mask = enc_tgt["attention_mask"]
                tgt_len = tgt_mask.sum(dim=1).to(self.device)
                output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss.sum(dim=1) / tgt_len
                out.extend([-x.item() for x in loss])
            except RuntimeError:
                # ： 0.0 ，， values_all
                out.extend([0.0] * len(src_list))
                continue
        return out

# ===========================
# Binoculars-style probability mapping
# ===========================
def _logistic_prob_from_score(scores: np.ndarray, threshold: float, slope: float = 8.0) -> np.ndarray:
    x = slope * (float(threshold) - np.asarray(scores, dtype=np.float64))
    p = 1.0 / (1.0 + np.exp(-x))
    return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

# ===========================
# Detector
# ===========================
@register("tocsin")
class TOCSINDetector(DetectorBase):
    CITATION_TITLE = "Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness"
    CITATION_AUTHORS = "Shixuan Ma, Quan Wang"
    CITATION_LINK = "https://arxiv.org/abs/2409.16914"

    def __init__(
        self,
        score_model: str,
        reference_model: str,
        basemodel: str = "Fast",                    # {"Fast","lrr","likelihood","logrank","standalone"}
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mask_pct: float = TOCSIN_MASK_PCT,
        perturb_per_text: int = TOCSIN_PERTURB_PER_TEXT,
        bart_checkpoint: str = TOCSIN_BART_CHECKPOINT,
        dataset_file: Optional[str] = None,        # special-case for gemini+pubmed (+2)
        device: Optional[str] = None,
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",
        # Binoculars-like mapping params (center & slope)
        prob_slope: float = -2.0,
        prob_threshold: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            score_model=score_model,
            reference_model=reference_model,
            basemodel=basemodel,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            mask_pct=mask_pct,
            perturb_per_text=perturb_per_text,
            bart_checkpoint=bart_checkpoint,
            dataset_file=dataset_file,
            device=device,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type,
            **kwargs,
        )
        self.score_model = score_model
        self.reference_model = reference_model
        self.basemodel = str(basemodel)
        self.use_bfloat16 = bool(use_bfloat16)
        self.max_len = int(max_token_observed)
        self.mask_pct = float(mask_pct)
        self.k_perturb = int(perturb_per_text)
        self.bart_checkpoint = str(bart_checkpoint)
        self.dataset_file = dataset_file
        self.user_device = device
        self.name = name or f"TOCSIN[{os.path.basename(self.score_model)}|{os.path.basename(self.reference_model)}|{self.basemodel}]"
        self.DETECTOR_NAME = self.name
        self.detector_type = detector_type or "Metric-based"

        self.prob_slope = float(prob_slope)
        self.threshold = float(prob_threshold)

        # runtime
        self._tok_score = None
        self._tok_ref = None
        self._model_score = None
        self._model_ref = None
        self._bart = None
        self._dev = "cpu"
        self._dtype = torch.float32
        self.is_loaded = False

        # seeds (same spirit as your script)
        self._seed = 0
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

    def load(self):
        # device/dtype
        self._dev = _select_device(self.user_device)
        self._dtype = _torch_dtype(self.use_bf16 if hasattr(self, "use_bf16") else self.use_bfloat16)

        # scoring tokenizer/model
        self._tok_score = AutoTokenizer.from_pretrained(self.score_model, token=HF_TOKEN)
        if getattr(self._tok_score, "pad_token", None) is None and getattr(self._tok_score, "eos_token", None) is not None:
            self._tok_score.pad_token = self._tok_score.eos_token

        self._model_score = AutoModelForCausalLM.from_pretrained(
            self.score_model, trust_remote_code=True, torch_dtype=self._dtype, device_map={"": self._dev}, token=HF_TOKEN
        ).eval()

        # reference tokenizer/model (or reuse)
        if self.reference_model != self.score_model:
            self._tok_ref = AutoTokenizer.from_pretrained(self.reference_model, token=HF_TOKEN)
            if getattr(self._tok_ref, "pad_token", None) is None and getattr(self._tok_ref, "eos_token", None) is not None:
                self._tok_ref.pad_token = self._tok_ref.eos_token

            self._model_ref = AutoModelForCausalLM.from_pretrained(
                self.reference_model, trust_remote_code=True, torch_dtype=self._dtype, device_map={"": self._dev}, token=HF_TOKEN
            ).eval()
        else:
            self._tok_ref = self._tok_score
            self._model_ref = self._model_score

        # BART scorer
        self._bart = _BARTScorer(device=self._dev, max_length=self.max_len, checkpoint=self.bart_checkpoint)

        super().load()

    @torch.inference_mode()
    def _encode_one(self, text: str, use_ref: bool = False):
        tok = self._tok_ref if use_ref else self._tok_score
        enc = tok(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
        )
        return {k: (v.to(self._dev) if hasattr(v, "to") else v) for k, v in enc.items()}

    @torch.inference_mode()
    def _compute_ll_mu_std_logrank(self, text: str) -> Tuple[float, float, float, float]:
        # scoring forward
        tokenized = self._encode_one(text, use_ref=False)
        # logits ：labels  token
        labels = tokenized["input_ids"][:, 1:]
        logits_score = self._model_score(**tokenized).logits[:, :-1]

        # ====== ： ======
        # token  T==0，
        if logits_score.size(1) == 0 or labels.size(1) == 0:
            return 0.0, 0.0, 1.0, -1e-12

        # reference forward
        if self.reference_model == self.score_model:
            logits_ref = logits_score
        else:
            tokenized_ref = self._encode_one(text, use_ref=True)
            # tokenizer ；
            if tokenized_ref["input_ids"].size(1) < 2:
                return 0.0, 0.0, 1.0, 0.0
            logits_ref = self._model_ref(**tokenized_ref).logits[:, :-1]
            if logits_ref.size(1) == 0:
                return 0.0, 0.0, 1.0, 0.0

        # samples & stats（ T>=1 ）
        samples_2 = _get_samples(logits_ref, labels, nsamples=TOCSIN_NSAMPLES)
        ll_x = _get_likelihood(logits_score, labels)                      # (1,1)
        logrank_x = _get_logrank(logits_score, labels)                    # scalar
        ll_tilde = _get_likelihood(logits_score, samples_2)               # (1,nsamp)
        mu = float(ll_tilde.mean(dim=-1).item())
        std_val = float(ll_tilde.std(dim=-1).item())
        if std_val == 0.0:
            std_val = 1e-12

        return float(ll_x.squeeze(-1).item()), mu, std_val, float(logrank_x)

    def _combine_score(self, llx: float, mu: float, std: float, logrank: float, mean_sim: float) -> float:
        # STRICT branch logic from your script
        if self.basemodel == "Fast":
            if (self.dataset_file is not None) and ('gemini' in self.dataset_file and 'pubmed' in self.dataset_file):
                return (((llx - mu) / std) + 2.0) * math.pow(math.e, -mean_sim)
            else:
                return ((llx - mu) / std) * math.pow(math.e, -mean_sim)
        elif self.basemodel == "lrr":
            den = logrank
            if not math.isfinite(den) or abs(den) < 1e-12:
                # ； 0 ， _get_logrank
                den = -1e-12 if den >= 0 else den  # den==0 -> -1e-12
                if abs(den) < 1e-12:
                    den = -1e-12
            return (llx / den) * math.pow(math.e, -mean_sim)
        elif self.basemodel == "likelihood":
            return llx * math.pow(math.e, +mean_sim)
        elif self.basemodel == "logrank":
            return logrank * math.pow(math.e, +mean_sim)
        elif self.basemodel == "standalone":
            return -mean_sim
        else:
            return llx * math.pow(math.e, +mean_sim)

    # ===========================
    # Main API: score_batch
    # ===========================
    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        k = self.k_perturb

        # ---- (A) Dataset-style perturbation first (exactly like your script) ----
        # Build repeated source list: [t1]*k + [t2]*k + ...
        tgt_all: List[str] = []
        for t in texts:
            tgt_all.extend([t] * k)

        # Perturb all at once, in chunks of 50 (same as your script)
        perturbed_all = _perturb_texts_dataset_style(tgt_all, self.mask_pct, k=k, chunk=50)

        # Compute BART scores for all pairs in one go (batch_size=10 as in your script)
        values_all = self._bart.score(perturbed_all, tgt_all, batch_size=10)

        # Mean per text
        mean_sim_per_text: List[float] = []
        for i in range(n):
            vals = values_all[i * k: (i + 1) * k]
            mv = float(np.mean(vals)) if len(vals) > 0 else 0.0
            if not math.isfinite(mv):
                mv = 0.0
            mean_sim_per_text.append(mv)

        # ---- (B) Now compute logits-based stats per text and combine ----
        raw_scores: List[float] = []
        for i, t in enumerate(texts):
            llx, mu, std, logrank = self._compute_ll_mu_std_logrank(t)
            score = self._combine_score(llx, mu, std, logrank, mean_sim_per_text[i])
            raw_scores.append(float(score))

        raw = np.asarray(raw_scores, dtype=np.float32)

        # ---- (C) Binoculars-style probability mapping ----
        probs = _logistic_prob_from_score(raw, threshold=self.threshold, slope=self.prob_slope)
        return probs

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        return np.clip(scores, 1e-6, 1.0 - 1e-6)
