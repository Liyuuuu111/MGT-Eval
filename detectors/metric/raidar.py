# mgt_eval/detectors/metric/raidar.py
# -*- coding: utf-8 -*-
"""
RAIDAR Detector (GLTR-style backend)
- score_batch() 返回特征矩阵（与 GLTR 一致）；
- calibrate() 使用少量样本进行 StandardScaler+MLP 标定并输出全体样本概率；
- 训练阶段带进度条（评估/特征抽取由外层评测框架自带进度条，不在此重复）。
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import numpy as np
import torch
from tqdm import tqdm

# ========== 相似度依赖 ==========
try:
    from fuzzywuzzy import fuzz  # type: ignore
except Exception:
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        fuzz = None

# ========== 可选 OpenAI 回退 ==========
try:
    import openai  # type: ignore
except Exception:
    openai = None

from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== mgt_eval 接口 ==========
from ..base import DetectorBase
from ..registry import register


# ---------------------------
# RAIDAR 提示模板（按原脚本）
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
# 文献信息（占位）
# ---------------------------
CITATION_TITLE = "Raidar: geneRative AI Detection viA Rewriting"
CITATION_AUTHORS = "Chengzhi Mao, Carl Vondrick, Hao Wang, Junfeng Yang"
CITATION_LINK = "https://arxiv.org/abs/2401.12970"


# ---------------------------
# 基础工具：分词/共现
# ---------------------------
def _tokenize_and_normalize(sentence: str) -> List[str]:
    return [w.lower().strip() for w in (sentence or "").split()]

def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _common_elements(list1: List[str], list2: List[str]) -> set:
    return set(list1) & set(list2)

def _calculate_sentence_common(sentence1: str, sentence2: str) -> List[int]:
    """
    返回 [common_1gram, common_2gram, common_3gram, common_4gram]
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
    RAIDAR：Rewrite-then-Feature + 小 MLP（GLTR 风格后端）
    - score_batch(texts) → 返回特征矩阵 X ∈ R^{N×F}（F≈50）
    - calibrate(X, y)    → 选取 calibrate_k 条训练 StandardScaler+MLP，返回全体样本概率 p(AI|x)
    """

    # 文献信息
    CITATION_TITLE = CITATION_TITLE
    CITATION_AUTHORS = CITATION_AUTHORS
    CITATION_LINK = CITATION_LINK

    def __init__(
        self,
        # —— 改写器参数 —— #
        rewrite_model: Optional[str] = None,
        use_openai: bool = True,
        openai_model: str = "gpt-3.5-turbo",
        max_new_tokens_factor: float = 1.0,
        device: Optional[str] = None,

        # —— 新增：输入截断 —— #
        rewrite_input_max_tokens: Optional[int] = None,  # <<<<<< 新增：前端可控的 k

        # —— 评测/展示 —— #
        name: Optional[str] = None,
        detector_type: Optional[str] = "Metric-based",

        # —— 小样本标定控制（与 GLTR 对齐） —— #
        calibrate_k: int = 1000,
        calibrate_seed: int = 42,

        # 其他透传
        **kwargs: Any,
    ):
        super().__init__(
            rewrite_model=rewrite_model,
            use_openai=use_openai,
            openai_model=openai_model,
            max_new_tokens_factor=max_new_tokens_factor,
            device=device,
            rewrite_input_max_tokens=rewrite_input_max_tokens,  # 透传保存
            calibrate_k=calibrate_k,
            calibrate_seed=calibrate_seed,
            **({"name": name} if name is not None else {}),
            detector_type=detector_type or "Metric-based",
            **kwargs,
        )
        # 存参
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

        # 运行态对象
        self._tok: Optional[AutoTokenizer] = None
        self._llama: Optional[AutoModelForCausalLM] = None
        self._device: Optional[str] = None
        self._openai_ready: bool = False

        # 学习器
        self._scaler = None
        self._clf = None  # sklearn MLPClassifier

        # 展示名
        llama_tag = os.path.basename(self.rewrite_model) if self.rewrite_model else "none"
        self.DETECTOR_NAME = name or f"RAIDAR[{llama_tag}]"
        self.detector_type = detector_type or "Metric-based"
        self.is_loaded: bool = False

    # ========== 加载改写器 ==========
    def load(self):
        # LLaMA
        if self.rewrite_model:
            try:
                self._tok = AutoTokenizer.from_pretrained(self.rewrite_model)
                # pad/eos 对齐
                if getattr(self._tok, "pad_token", None) is None and getattr(self._tok, "eos_token", None) is not None:
                    self._tok.pad_token = self._tok.eos_token
                # 建议保留：确保从右侧截断（保留前缀）
                try:
                    self._tok.truncation_side = "right"
                except Exception:
                    pass

                self._llama = AutoModelForCausalLM.from_pretrained(
                    self.rewrite_model,
                    device_map="auto",
                )
                # 同步 pad_token_id
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

        # OpenAI 回退
        self._openai_ready = False
        if (self._llama is None) and self.use_openai and (openai is not None):
            if os.environ.get("OPENAI_API_KEY", ""):
                self._openai_ready = True

        if fuzz is None:
            raise RuntimeError("[RAIDAR] 需要 fuzzywuzzy 或 rapidfuzz 用于相似度特征。")

        super().load()

    # ========== 截断辅助 ==========
    def _truncate_prompt_string(self, prompts: str) -> str:
        """
        将完整 prompts（包含提示词与原文）截断到最多 k 个 token。
        优先使用本地 tokenizer 做精确截断；若不可用则退化为按空格粗略截断。
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
            # decode 后可能会丢失少量空格/引号格式，但满足“截断不超过 k token”
            return self._tok.decode(ids, skip_special_tokens=True)

        # 无 tokenizer 的粗略退化：按空格切分，取前 k 个“词”（近似 token）
        parts = prompts.split()
        if len(parts) <= k:
            return prompts
        return " ".join(parts[:k])

    # ========== 单次重写 ==========
    @torch.inference_mode()
    def _llama_self_prompt(self, prompt_str: str, content: str) -> str:
        assert self._tok is not None and self._llama is not None, "LLaMA 未加载"
        prompts = f'{prompt_str}: "{content}"'
        # —— 新增：先在字符串层面截断 —— #
        prompts = self._truncate_prompt_string(prompts)

        # 再做一次 tokenizer 级别的“硬”截断，确保输入 <= k
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

        # max_new_tokens 仍然按你的实现计算（如需提速，可自行调小 factor/去掉硬下限）
        base = int(len(_tokenize_and_normalize(prompts)) * self.max_new_tokens_factor)
        max_new_tokens = max(1, base)
        # 保留你之前的下限（如你觉得慢，可删掉或改小）
        max_new_tokens = max(max_new_tokens, 256)

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

        # —— 新增：对 prompts 做 k-token 截断（若本地 tokenizer 可用则精确，否则按空格粗略） —— #
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
        return content  # 兜底

    def _rewrite_text(self, text: str) -> Dict[str, str]:
        return {ep: self._rewrite_once(ep, text) for ep in RAIDAR_PROMPT_LIST}

    # ========== 特征构造 ==========
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
        return np.asarray(feats, dtype=np.float32)              # 总维度 4 + 28 + 4 + 14 = 50

    # ========== 与 GLTR 对齐的接口 ==========
    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        返回特征矩阵 (N, F)。评测框架会在拿到 labels 后调用 calibrate(features, labels)，
        我们在 calibrate() 中完成训练并输出全体概率。
        """
        feats = [self._build_feature_vector(t, self._rewrite_text(t)) for t in texts]
        return np.vstack(feats).astype(np.float32)

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        - 输入 scores: (N, F) 的特征矩阵（由 score_batch 返回）
        - 若提供 labels：从中抽取 calibrate_k 条做 StandardScaler+MLP(10) 训练，并对全体样本输出 p(AI|x)
        - 若无 labels：直接回传特征（与 GLTR 行为对齐）
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
        # 分层抽样（若齐全）
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

        # 训练（带一个简明进度条）
        with tqdm(total=1, desc="RAIDAR calibrate: fitting MLP", dynamic_ncols=True) as pbar:
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

        # 对全体样本输出概率
        X_all = self._scaler.transform(scores)
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X_all)[:, 1]
        else:
            proba = self._clf.predict(X_all).astype(np.float32)
        return np.clip(proba.astype(np.float32), 1e-6, 1.0 - 1.0e-6)
