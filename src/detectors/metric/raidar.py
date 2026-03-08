# mgt_eval/detectors/metric/raidar.py
# -*- coding: utf-8 -*-
"""
RAIDAR Detector (GLTR-style backend)
- score_batch() returns the feature matrix (aligned with GLTR).
- calibrate() fits StandardScaler+MLP on a small subset and outputs probabilities for all samples.
- Training uses a progress bar (evaluation/feature extraction progress is handled by the outer framework).
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import warnings
import numpy as np
import torch
from tqdm import tqdm

# ========== Similarity dependencies ==========
try:
    from fuzzywuzzy import fuzz  # type: ignore
except Exception:
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        fuzz = None

# ========== Optional OpenAI fallback ==========
try:
    import openai  # type: ignore
except Exception:
    openai = None

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None
    SamplingParams = None

# ========== mgt_eval interface ==========
from ..base import DetectorBase
from ..registry import register


# ---------------------------
# RAIDAR prompt templates (as in the original script)
# ---------------------------
RAIDAR_PROMPT_LIST: List[str] = [
    "Revise this with your best effort",
    "Help me polish this",
    "Rewrite this for me",
    "Make this fluent while doing minimal change",
    "Refine this for me please",
    "Concise this for me and keep all the information",
    "Improve this in GPT way",
]

# ---------------------------
# Citation info (placeholder)
# ---------------------------
CITATION_TITLE = "Raidar: geneRative AI Detection viA Rewriting"
CITATION_AUTHORS = "Chengzhi Mao, Carl Vondrick, Hao Wang, Junfeng Yang"
CITATION_LINK = "https://arxiv.org/abs/2401.12970"


# ---------------------------
# Basic utilities: tokenization/commonality
# ---------------------------
def _tokenize_and_normalize(sentence: str) -> List[str]:
    return [w.lower().strip() for w in (sentence or "").split()]

def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _common_elements(list1: List[str], list2: List[str]) -> set:
    return set(list1) & set(list2)

def _calculate_sentence_common(sentence1: str, sentence2: str) -> List[int]:
    """
    Return [common_1gram, common_2gram, common_3gram, common_4gram]
    """
    tokens1 = _tokenize_and_normalize(sentence1)
    tokens2 = _tokenize_and_normalize(sentence2)
    feats: List[int] = [len(_common_elements(tokens1, tokens2))]
    for n in range(2, 5):
        ngrams1 = _extract_ngrams(tokens1, n)
        ngrams2 = _extract_ngrams(tokens2, n)
        feats.append(len(_common_elements(ngrams1, ngrams2)))
    return feats  # len=4


@register("raidar")
class RAIDARDetector(DetectorBase):
    """
    RAIDAR: Rewrite-then-Feature + small MLP (GLTR-style backend)
    - score_batch(texts) → feature matrix X ∈ R^{N×F} (F≈50)
    - calibrate(X, y)    → fit StandardScaler+MLP on calibrate_k samples, output p(AI|x) for all samples
    """

    # Citation info
    CITATION_TITLE = CITATION_TITLE
    CITATION_AUTHORS = CITATION_AUTHORS
    CITATION_LINK = CITATION_LINK

    def __init__(
        self,
        # Rewrite model params
        rewrite_model: Optional[str] = None,
        use_openai: bool = True,
        openai_model: str = "gpt-3.5-turbo",
        max_new_tokens_factor: float = 1.0,
        device: Optional[str] = None,

        # Input truncation
        rewrite_input_max_tokens: Optional[int] = None,  # <<<<<< new: front-end controllable k

        # vLLM acceleration (optional)
        use_vllm: bool = False,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        vllm_dtype: Optional[str] = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_enforce_eager: bool = False,

        # Evaluation / presentation
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",

        # Small-sample calibration (aligned with GLTR)
        calibrate_k: int = 1000,
        calibrate_seed: int = 42,

        # Other passthrough
        **kwargs: Any,
    ):
        super().__init__(
            rewrite_model=rewrite_model,
            use_openai=use_openai,
            openai_model=openai_model,
            max_new_tokens_factor=max_new_tokens_factor,
            device=device,
            rewrite_input_max_tokens=rewrite_input_max_tokens,  # passthrough
            calibrate_k=calibrate_k,
            calibrate_seed=calibrate_seed,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type or "Metric-based",
            **kwargs,
        )
        # Store params
        self.rewrite_model = rewrite_model
        self.use_openai = bool(use_openai)
        self.openai_model = str(openai_model)
        self.max_new_tokens_factor = float(max_new_tokens_factor)
        self.user_device = device

        self.rewrite_input_max_tokens: Optional[int] = (
            int(rewrite_input_max_tokens) if (rewrite_input_max_tokens is not None and int(rewrite_input_max_tokens) > 0) else None
        )

        self.calibrate_k = int(max(1, calibrate_k))
        self.calibrate_seed = int(calibrate_seed)

        # Runtime objects
        self._tok: Optional[AutoTokenizer] = None
        self._llama: Optional[AutoModelForCausalLM] = None
        self._vllm = None
        self._vllm_failed = False
        self._device: Optional[str] = None
        self._openai_ready: bool = False
        self._warned_no_vllm: bool = False

        # Learners
        self._scaler = None
        self._clf = None  # sklearn MLPClassifier

        # Display name
        llama_tag = os.path.basename(self.rewrite_model) if self.rewrite_model else "none"
        self.DETECTOR_NAME = name or f"RAIDAR[{llama_tag}]"
        self.detector_type = detector_type or "Metric-based"
        self.is_loaded: bool = False

        # vLLM settings
        self.use_vllm = bool(use_vllm)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        self.vllm_max_model_len = int(vllm_max_model_len) if (vllm_max_model_len is not None and int(vllm_max_model_len) > 0) else None
        self.vllm_dtype = str(vllm_dtype) if vllm_dtype is not None else None
        self.vllm_tensor_parallel_size = int(vllm_tensor_parallel_size)
        self.vllm_enforce_eager = bool(vllm_enforce_eager)

    # ========== Load rewriter ==========
    def load(self):
        # LLaMA
        if self.rewrite_model:
            try:
                self._tok = AutoTokenizer.from_pretrained(self.rewrite_model)
                # Align pad/eos
                if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
                    self._tok.pad_token = self._tok.eos_token
                # Keep: truncate from the right to preserve the prefix
                try:
                    self._tok.truncation_side = "right"
                except Exception:
                    pass

                self._llama = AutoModelForCausalLM.from_pretrained(
                    self.rewrite_model,
                    device_map="auto",
                )
                # Sync pad_token_id
                self._llama.config.pad_token_id = self._tok.eos_token_id
                if hasattr(self._llama, "generation_config") and self._llama.generation_config is not None:
                    self._llama.generation_config.pad_token_id = self._tok.eos_token_id

                self._device = str(self._llama.device) if hasattr(self._llama, "device") else (
                    self.user_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
                )
            except Exception as e:
                if self._tok or self._llama:
                    self._tok, self._llama = None, None
                self._device = self.user_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
                if os.environ.get("RAIDAR_DEBUG"):
                    print(f"[RAIDAR] LLaMA load failed: {e}")

        # vLLM acceleration (rewrite only)
        if self.use_vllm and self.rewrite_model and (LLM is not None) and (not self._vllm_failed):
            try:
                self._vllm = LLM(
                    model=self.rewrite_model,
                    tensor_parallel_size=max(1, self.vllm_tensor_parallel_size),
                    gpu_memory_utilization=float(self.vllm_gpu_memory_utilization),
                    max_model_len=self.vllm_max_model_len,
                    dtype=self.vllm_dtype or "auto",
                    trust_remote_code=True,
                    enforce_eager=bool(self.vllm_enforce_eager),
                )
            except Exception as e:
                self._vllm = None
                self._vllm_failed = True
                if os.environ.get("RAIDAR_DEBUG"):
                    print(f"[RAIDAR] vLLM init failed, fallback to HF generation: {e}")
        if self.rewrite_model and (self._vllm is None) and (not self._warned_no_vllm):
            warnings.warn(
                "[RAIDAR] vLLM is not enabled; rewriting will be slow. "
                "Install and enable it for speed: pip install -e \".[vllm]\"",
                RuntimeWarning,
            )
            self._warned_no_vllm = True

        # OpenAI fallback
        self._openai_ready = False
        if (self._llama is None) and self.use_openai and (openai is not None):
            if os.environ.get("OPENAI_API_KEY", ""):
                self._openai_ready = True

        if fuzz is None:
            raise RuntimeError("[RAIDAR] 需要 fuzzywuzzy 或 rapidfuzz 用于相似度特征。")

        super().load()

    # ========== Truncation helper ==========
    def _truncate_prompt_string(self, prompts: str) -> str:
        """
        Truncate full prompts (including instruction and original text) to at most k tokens.
        Prefer the local tokenizer for exact truncation; fall back to whitespace splitting.
        """
        k = self.rewrite_input_max_tokens
        if not k or k <= 0:
            return prompts

        if self._tok is not None:
            enc = self._tok(prompts, add_special_tokens=False)
            ids = enc["input_ids"]
            if len(ids) <= k:
                return prompts
            ids = ids[:k]
            # Decoding may lose some whitespace/quotes, but keeps the token budget.
            return self._tok.decode(ids, skip_special_tokens=True)

        # Fallback without tokenizer: split by whitespace and take first k words (approx. tokens)
        parts = prompts.split()
        if len(parts) <= k:
            return prompts
        return " ".join(parts[:k])

    # ========== Single rewrite ==========
    @torch.inference_mode()
    def _llama_self_prompt(self, prompt_str: str, content: str) -> str:
        prompts = f'{prompt_str}: "{content}"'
        # New: truncate at the string level first
        prompts = self._truncate_prompt_string(prompts)

        # max_new_tokens still follows your logic (reduce factor/remove lower bound to speed up)
        base = int(len(_tokenize_and_normalize(prompts)) * self.max_new_tokens_factor)
        max_new_tokens = max(1, base)
        # Keep your previous lower bound (remove or reduce if too slow)
        max_new_tokens = max(max_new_tokens, 256)

        # vLLM generation (optional)
        if self._vllm is not None and (SamplingParams is not None):
            try:
                sp = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=int(max_new_tokens),
                )
                outs = self._vllm.generate([prompts], sp)
                if outs and outs[0].outputs:
                    gen = outs[0].outputs[0].text
                else:
                    gen = ""
                return prompts + gen
            except Exception as e:
                if os.environ.get("RAIDAR_DEBUG"):
                    print(f"[RAIDAR] vLLM generate failed, fallback to HF: {e}")

        # HF fallback requires tokenizer/model
        if self._tok is None or self._llama is None:
            raise RuntimeError("LLaMA 未加载（且 vLLM 不可用）")

        # Apply a tokenizer-level hard truncation to ensure input <= k
        if self.rewrite_input_max_tokens:
            model_inputs = self._tok(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.rewrite_input_max_tokens),
            )
        else:
            model_inputs = self._tok(prompts, return_tensors="pt")

        model_inputs.pop("token_type_ids", None)

        if self._device:
            model_inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        output = self._llama.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tok.eos_token_id,
        )
        return self._tok.decode(output[0], skip_special_tokens=True)

    def _openai_self_prompt(self, prompt_str: str, content: str) -> str:
        if openai is None:
            raise RuntimeError("openai 包不可用。")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置。")

        # New: truncate prompts to k tokens (exact with tokenizer, else whitespace fallback)
        prompts = f'{prompt_str}: "{content}"'
        prompts = self._truncate_prompt_string(prompts)

        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompts}],
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompts}],
                temperature=0.7,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()

    def _rewrite_once(self, prompt_str: str, content: str) -> str:
        if self._llama is not None and self._tok is not None:
            return self._llama_self_prompt(prompt_str, content)
        if self._openai_ready:
            return self._openai_self_prompt(prompt_str, content)
        return content  # fallback

    def _rewrite_text(self, text: str) -> Dict[str, str]:
        return {ep: self._rewrite_once(ep, text) for ep in RAIDAR_PROMPT_LIST}

    def _rewrite_texts_vllm(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Batch rewrite using vLLM. Returns list of dicts aligned with texts.
        Falls back to per-text rewriting if vLLM errors.
        """
        if self._vllm is None or (SamplingParams is None):
            return [self._rewrite_text(t) for t in texts]

        n = len(texts)
        rewrites: List[Dict[str, str]] = [dict() for _ in range(n)]

        for prompt_str in RAIDAR_PROMPT_LIST:
            prompts: List[str] = []
            max_tokens_list: List[int] = []
            for t in texts:
                p = f'{prompt_str}: "{t}"'
                p = self._truncate_prompt_string(p)
                prompts.append(p)
                base = int(len(_tokenize_and_normalize(p)) * self.max_new_tokens_factor)
                mt = max(1, base)
                mt = max(mt, 256)
                max_tokens_list.append(mt)

            try:
                # Prefer per-prompt sampling params if supported
                sp_list = [
                    SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=int(mt),
                    )
                    for mt in max_tokens_list
                ]
                outs = self._vllm.generate(prompts, sp_list)
            except Exception:
                # Fallback: single SamplingParams with max tokens
                try:
                    sp = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=int(max(max_tokens_list) if max_tokens_list else 1),
                    )
                    outs = self._vllm.generate(prompts, sp)
                except Exception as e:
                    if os.environ.get("RAIDAR_DEBUG"):
                        print(f"[RAIDAR] vLLM batch generate failed, fallback to per-text: {e}")
                    return [self._rewrite_text(t) for t in texts]

            if not outs or len(outs) != n:
                if os.environ.get("RAIDAR_DEBUG"):
                    print("[RAIDAR] vLLM batch output mismatch, fallback to per-text.")
                return [self._rewrite_text(t) for t in texts]

            for i, out in enumerate(outs):
                if out.outputs:
                    gen = out.outputs[0].text
                else:
                    gen = ""
                rewrites[i][prompt_str] = prompts[i] + gen

        return rewrites

    # ========== Feature construction ==========
    def _build_feature_vector(self, original: str, rewrites: Dict[str, str]) -> np.ndarray:
        raw = _tokenize_and_normalize(original)
        r_len = float(len(raw)) if len(raw) > 0 else 1.0

        all_stat = [0, 0, 0, 0]
        stat_per_prompt: Dict[str, List[int]] = {}
        fuzz_per_prompt: Dict[str, Tuple[float, float]] = {}
        whole_combined = ""

        for ep in RAIDAR_PROMPT_LIST:
            rw = rewrites.get(ep, "")
            whole_combined += (" " + rw)
            res = _calculate_sentence_common(original, rw)
            stat_per_prompt[ep] = res
            all_stat = [a + b for a, b in zip(all_stat, res)]
            if fuzz is not None:
                fuzz_per_prompt[ep] = (float(fuzz.ratio(original, rw)), float(fuzz.token_set_ratio(original, rw)))
            else:
                fuzz_per_prompt[ep] = (0.0, 0.0)

        cnt = max(1, len(RAIDAR_PROMPT_LIST))
        avg_common = [a / float(cnt) for a in all_stat]
        common_vs_all = _calculate_sentence_common(original, whole_combined)

        feats: List[float] = []
        feats.extend([v / r_len for v in avg_common])           # 4
        for ep in RAIDAR_PROMPT_LIST:
            feats.extend([(v / r_len) for v in stat_per_prompt[ep]])  # 7*4
        feats.extend([(v / r_len) for v in common_vs_all])      # 4
        for ep in RAIDAR_PROMPT_LIST:
            fr, fts = fuzz_per_prompt[ep]
            feats.extend([fr, fts])                             # 7*2
        return np.asarray(feats, dtype=np.float32)              # total dim: 4 + 28 + 4 + 14 = 50

    # ========== GLTR-aligned interface ==========
    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        Return feature matrix (N, F). The evaluation framework will call
        calibrate(features, labels) after labels are available; we fit and output probabilities there.
        """
        if self._vllm is not None and (SamplingParams is not None):
            rewrites_list = self._rewrite_texts_vllm(texts)
        else:
            rewrites_list = [self._rewrite_text(t) for t in texts]

        feats = [self._build_feature_vector(t, rewrites_list[i]) for i, t in enumerate(texts)]
        return np.vstack(feats).astype(np.float32)

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        - Input scores: feature matrix (N, F) from score_batch
        - If labels provided: sample calibrate_k points, fit StandardScaler+MLP(10), output p(AI|x)
        - If no labels: return features directly (aligned with GLTR behavior)
        """
        if labels is None:
            return scores

        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(scores)
        if n == 0:
            return scores

        k = min(self.calibrate_k, n)
        rng = np.random.default_rng(self.calibrate_seed)

        idx = np.arange(n)
        # Stratified sampling when labels are available
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

        # Train (with a brief progress bar)
        with tqdm(total=1, desc="RAIDAR calibrate: fitting MLP", dynamic_ncols=True, leave=False) as pbar:
            scaler = StandardScaler()
            Xn = scaler.fit_transform(scores[calib_idx])

            clf = MLPClassifier(
                hidden_layer_sizes=(10,),
                max_iter=1000,
                activation="relu",
                solver="adam",
                random_state=self.calibrate_seed,
            )
            clf.fit(Xn, labels[calib_idx])
            pbar.update(1)

        self._scaler = scaler
        self._clf = clf

        # Output probabilities for all samples
        X_all = self._scaler.transform(scores)
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X_all)[:, 1]
        else:
            proba = self._clf.predict(X_all).astype(np.float32)
        return np.clip(proba.astype(np.float32), 1e-6, 1.0 - 1.0e-6)
