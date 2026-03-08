# -*- coding: utf-8 -*-
"""
PECOLA (ACL 2024) — Official Logic Wrapped for mgt_eval
=======================================================

This file keeps the official PECOLA algorithmic details intact (selective perturbation
+ YAKE keyword retention + T5 mask filling; CE / SCL / margin_weight losses), and
adds a thin engineering wrapper so it can:
  1) Load datasets via `load_dataset_unified` (0=human, 1=machine) without changing
     the original logic.
  2) Cache the *perturbed* training dataset (masked deletion + T5 fills) and reuse it
     across runs. If cache is missing, generate it first, then unload the perturbation
     model from GPU/CPU memory before training.
  3) Offer a registered training entrypoint `@register_train("pecola")` and a
     simple detector class `PecolaFT` registered to `detectors.registry` for direct
     inference/evaluation within mgt_eval.
  4) Provide richer logging (sizes, params) and leave=False tqdm progress bars.

IMPORTANT: Algorithmic steps, hyper-parameters, and loss implementations follow the
published/open-sourced PECOLA reference. We only add integration glue and utilities.

Author credits (paper): Shengchao Liu, Xiaoming Liu*, Yichen Wang, Zehua Cheng,
Chengzhengxu Li, Yu Lan, Chao Shen. ACL 2024.

This integration code: © 2025, for research use with mgt_eval.
"""
from __future__ import annotations
import os, re, json, math, time, random, hashlib, platform, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from loguru import logger
import concurrent.futures as _fut
from functools import partial
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

# ---- mgt_eval glue ----
from data_utils.load import load_dataset_unified
from train.registry import register_train
from detectors.base import DetectorBase
from detectors.registry import register as register_detector
from train.train import (
    _reset_and_mark_cuda_peaks,
    _collect_cuda_peaks,
    _save_loss_plot,
    _build_data_info,
)

# =========================
# Method meta (for logs)
# =========================
DETECTOR_NAME    = "PECOLA"
detector_type    = "Model-based"
CITATION_AUTHORS = (
    "Shengchao Liu, Xiaoming Liu*, Yichen Wang, Zehua Cheng, "
    "Chengzhengxu Li, Yu Lan, Chao Shen"
)
CITATION_TITLE   = (
    "Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation "
    "to Fine-tuned Contrastive Learning Detector would be Better"
)
CITATION_LINK    = "https://arxiv.org/abs/2402.00263"

# =========================
# Utilities
# =========================

def _seed_everything(seed: int = 114514):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _is_local_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _resolve_base(model: Optional[str]) -> str:
    fallback = "roberta-base"
    if not model:
        return fallback
    spec = model.strip()
    if _is_local_hf_dir(spec):
        return spec
    try:
        AutoConfig.from_pretrained(spec)
        return spec
    except Exception:
        return fallback


def _gpu_mem_str() -> str:
    if torch.cuda.is_available():
        try:
            return f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
        except Exception:
            return "0G"
    return "0G"


# =========================
# Official perturbation pieces (kept intact)
# =========================
# Selective deletion by YAKE, then T5 fills (<extra_id_*>), sentence-level 300-token split

try:
    import jieba
    import nltk
    import yake
except Exception:
    raise Exception
    # Let import error surface at first use; we keep lazy behavior to not block training
    pass

# ---- prefer repo-local nltk_data (if present) ----
_REPO_NLTK_DATA = str(Path(__file__).resolve().parents[2] / "nltk_data")
if os.path.isdir(_REPO_NLTK_DATA):
    if _REPO_NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, _REPO_NLTK_DATA)
    cur = os.environ.get("NLTK_DATA", "")
    if _REPO_NLTK_DATA not in cur.split(os.pathsep):
        os.environ["NLTK_DATA"] = os.pathsep.join(
            [p for p in [_REPO_NLTK_DATA, cur] if p]
        )

# ---- NLTK resource check (tokenizer) ----
try:
    nltk.data.find("tokenizers/punkt")
    logger.info("[PECOLA] NLTK resource loaded: tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
        logger.info("[PECOLA] NLTK resource downloaded: tokenizers/punkt")
    except Exception:
        raise Exception


class TextAugmenter:
    """Official selective deletion with YAKE keywords.

    Note: This class mirrors the official open-source logic:
      - Tokenization via NLTK (en) / jieba (zh)
      - YAKE keyword extraction (lan='en')
      - Random deletion with keyword retention + skip window
      - Produce "<mask>" placeholders to be later filled by T5
    """

    def __init__(self, lang: str = 'en'):
        assert lang in ['zh', 'en'], "only support 'zh' or 'en'"
        self.lang = lang
        self.joint_str = ' '

    def tokenizer(self, text: str) -> List[str]:
        if self.lang == 'zh':
            return jieba.lcut(text)
        return nltk.tokenize.word_tokenize(text)

    def extract_keywords(self, text: str, ratio: float = 0.05) -> List[str]:
        # Official logic: YAKE(lan='en', n=1) and return terms only
        kw_extractor = yake.KeywordExtractor(lan='en', n=1)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def small_fix(self, text: str) -> str:
        puncs = ',.，。!?！？;；、'
        for p in puncs:
            text = text.replace(' ' + p, p)
        return text

    def aug_by_deletion(self, text: str, prob: float = 0.12, mode: str = 'selective', ratio: float = 0.05, skip_number: int = 2) -> str:
        words = self.tokenizer(text)
        if not words:
            return text
        assert mode in ['random', 'selective']
        total_words = len(words)
        num_words_to_delete = int(prob * total_words)
        actual_deleted = 0
        deleted_indices = set()
        indices = list(range(total_words))
        keywords = self.extract_keywords(text, ratio=ratio)
        while actual_deleted < num_words_to_delete and indices:
            idx = random.choice(indices)
            if words[idx] not in keywords:
                deleted_indices.add(idx)
                actual_deleted += 1
            # skip window
            for offset in range(skip_number + 1):
                indices = [i for i in indices if i not in {idx + offset, idx - offset}]
        new_words = [w if i not in deleted_indices else "<mask>" for i, w in enumerate(words)]
        return self.small_fix(self.joint_str.join(new_words))


# ---- Drop-in replacement for your original T5TextProcessor ----
import os, re, math, time
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures as _fut
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5TextProcessor:
    """
    Multi-threaded wrapper that preserves the original PECOLA T5 filling logic:
      - <mask> -> <extra_id_k> mapping
      - split_text_for_t5 -> tokenize_and_mask -> generate -> extract_fills -> apply_extracted_fills
      - Same generate() arguments as the official snippet (max_length=512, do_sample=True, top_p=1.0, ...)
    
    New engineering options:
      - num_workers: number of threads to parallelize outer workflow.
      - per_worker_model: if True (CPU-only), each worker creates its own model+tokenizer and runs generate in parallel.
                         if False (default), we keep a single model and do batched generate in the main thread (safer).
      - batch_size: batch size when doing the "single-model batched" path.
      - omp_num_threads, interop_threads: optional control for PyTorch CPU threading.
      - tokenizer_parallelism: control the HF fast tokenizer threading env var.
    """
    def __init__(
        self,
        model_name: str = 't5-large',
        device: Optional[torch.device] = None,
        random_seed: int = 114514,
        num_workers: int = 4,
        per_worker_model: bool = True,     # ：CPU
        batch_size: int = 16,               # batch
        omp_num_threads: Optional[int] = 4,
        interop_threads: int = 1,
        tokenizer_parallelism: Optional[bool] = True,
    ):
        self.model_name = model_name
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.num_workers = max(1, int(num_workers))
        self.per_worker_model = bool(per_worker_model and self.device.type == "cpu")
        self.batch_size = max(1, int(batch_size))

        # env / seeds
        self._seed_everything(random_seed)
        if omp_num_threads is not None:
            try:
                torch.set_num_threads(int(omp_num_threads))
            except Exception:
                pass
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except Exception:
            pass
        if tokenizer_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizer_parallelism else "false"

        # （）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    # ---------------- core logic preserved ----------------

    def tokenize_and_mask(self, text: str) -> str:
        tokens = text.split(' ')
        num_filled = 0
        for idx, tk in enumerate(tokens):
            if tk == "<mask>":
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        return ' '.join(tokens)

    @torch.no_grad()
    def replace_masks(self, texts: List[str]) -> List[str]:
        """
        Original single-batch behavior kept for API compatibility.
        Now internally dispatches to either:
          - _replace_masks_single_model_batched (default)
          - _replace_masks_per_worker_models (CPU-only, if per_worker_model=True)
        """
        if not texts:
            return []
        if self.per_worker_model and self.device.type == "cpu" and self.num_workers > 1:
            return self._replace_masks_per_worker_models(texts)
        else:
            return self._replace_masks_single_model_batched(texts)

    def extract_fills(self, texts: List[str]) -> List[List[str]]:
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        pattern = re.compile(r"<extra_id_\d+>")
        extracted = [pattern.split(x)[1:-1] for x in texts]
        return [[y.strip() for y in x] for x in extracted]

    def apply_extracted_fills(self, masked_texts: List[str], extracted_fills: List[List[str]]) -> List[str]:
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = [len([x for x in t.split() if x.startswith("<extra_id_")]) for t in masked_texts]
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
        return [" ".join(x) for x in tokens]

    def split_text_for_t5(self, text: str, tokenizer, max_length: int = 300) -> List[str]:
        toks = tokenizer.tokenize(text)
        start_idx = 0
        last_period_idx = 0
        segments = []
        for i, tk in enumerate(toks):
            if tk == '.':
                last_period_idx = i
            if i - start_idx >= max_length - 1:
                if last_period_idx > start_idx:
                    segments.append(tokenizer.convert_tokens_to_string(toks[start_idx:last_period_idx+1]))
                    start_idx = last_period_idx + 1
                else:
                    segments.append(tokenizer.convert_tokens_to_string(toks[start_idx:i+1]))
                    start_idx = i + 1
        if start_idx < len(toks):
            segments.append(tokenizer.convert_tokens_to_string(toks[start_idx:]))
        return segments

    def process_augmented_and_fill(self, augmented_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        保持原先“逐条样本 → 分段 → 掩码 → 生成 → 回填”的逻辑与字段不变：
          输入： [{'article':..., 'article_delete':..., 'label':...}, ...]
          输出： [{'article':..., 'article_delete':..., 'label':..., 'generater_text':...}, ...]
        仅在内部把所有分段统一收集后进行**批量/并行**调用，再按映射回填，减少生成调用次数。
        """
        if not augmented_items:
            return []

        # 1) ： & （，CPU ）
        # item  masked ，
        from concurrent.futures import ThreadPoolExecutor

        def _prep_one(item_idx: int, item: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
            masked = item['article_delete']
            segments = self.split_text_for_t5(masked, self.tokenizer)
            masked_texts = [self.tokenize_and_mask(seg) for seg in segments]
            return (item_idx, segments, masked_texts)

        prep_results: List[Tuple[int, List[str], List[str]]] = [None] * len(augmented_items)  # type: ignore
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for idx, res in enumerate(ex.map(lambda p: _prep_one(*p),
                                             [(i, augmented_items[i]) for i in range(len(augmented_items))])):
                prep_results[idx] = res

        # 2)  masked_texts ，
        flat_masked: List[str] = []
        mapping: List[Tuple[int, int, int]] = []  # (item_idx, start, count)
        cursor = 0
        for item_idx, _segs, masked_texts in prep_results:
            cnt = len(masked_texts)
            if cnt > 0:
                flat_masked.extend(masked_texts)
                mapping.append((item_idx, cursor, cnt))
                cursor += cnt
            else:
                mapping.append((item_idx, cursor, 0))

        # 3) （）
        if flat_masked:
            gen_texts = self.replace_masks(flat_masked)  # /
            extracted = self.extract_fills(gen_texts)
        else:
            extracted = []

        # 4)  item
        out: List[Dict[str, Any]] = [None] * len(augmented_items)  # type: ignore
        for (item_idx, start, cnt), (_i2, segs, masked_texts) in zip(mapping, prep_results):
            if cnt == 0:
                filled_text = augmented_items[item_idx]['article_delete']
            else:
                fills_slice = extracted[start:start+cnt]
                filled_segments = self.apply_extracted_fills(masked_texts, fills_slice)
                filled_text = " ".join(filled_segments) if filled_segments else ""
                if filled_text.strip() == "":
                    filled_text = "Failed to generate text"
                    logger.info("Failed to generate text")
            item = augmented_items[item_idx]
            out[item_idx] = {**item, "generater_text": filled_text}

        return out

    def unload(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- internals: batching / worker models ----------------

    def _replace_masks_single_model_batched(self, texts: List[str]) -> List[str]:
        outs: List[str] = []
        n_expected = [len([x for x in t.split() if x.startswith("<extra_id_")]) for t in texts]
        stop_id = self.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]

        total = (len(texts) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(range(total), desc=f"[PECOLA][T5] gen (single, bs={self.batch_size})",
                    leave=False, dynamic_ncols=True)

        for step in pbar:
            i = step * self.batch_size
            chunk = texts[i:i+self.batch_size]
            pack = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **pack,
                    max_length=512,
                    do_sample=True,
                    top_p=1.0,
                    num_return_sequences=1,
                    eos_token_id=stop_id,
                )
            batch_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outs.extend(batch_out)
            pbar.set_postfix_str(f"done={min(i+len(chunk), len(texts))}/{len(texts)}")
        return outs

    from tqdm.auto import tqdm

    def _replace_masks_per_worker_models(self, texts: List[str]) -> List[str]:
        chunks: List[List[str]] = []
        if self.num_workers >= len(texts):
            chunks = [[t] for t in texts]
        else:
            step = math.ceil(len(texts) / self.num_workers)
            for i in range(0, len(texts), step):
                chunks.append(texts[i:i+step])

        def _worker(chunk: List[str]) -> List[str]:
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to("cpu").eval()
            n_expected = [len([x for x in t.split() if x.startswith("<extra_id_")]) for t in chunk]
            stop_id = tok.encode(f"<extra_id_{max(n_expected)}>")[0]
            outs_local: List[str] = []
            for j in range(0, len(chunk), self.batch_size):
                sub = chunk[j:j+self.batch_size]
                pack = tok(sub, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = mdl.generate(
                        **pack,
                        max_length=512,
                        do_sample=True,
                        top_p=1.0,
                        num_return_sequences=1,
                        eos_token_id=stop_id,
                    )
                outs_local.extend(tok.batch_decode(outputs, skip_special_tokens=False))
            return outs_local

        # + （）
        results_map: Dict[int, List[str]] = {}
        pbar = tqdm(total=len(chunks), desc=f"[PECOLA][T5] gen (workers={self.num_workers}, bs={self.batch_size})",
                    leave=False, dynamic_ncols=True)
        with _fut.ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futs = {ex.submit(_worker, c): idx for idx, c in enumerate(chunks)}
            for fu in _fut.as_completed(futs):
                idx = futs[fu]
                results_map[idx] = fu.result()
                pbar.update(1)
        pbar.close()

        # ，
        ordered: List[str] = []
        for i in range(len(chunks)):
            ordered.extend(results_map.get(i, []))
        return ordered


    # ---------------- misc ----------------

    @staticmethod
    def _seed_everything(seed: int = 114514):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# =========================
# Classifier prep & losses (official logic preserved)
# =========================

def _prepare_seqcls(base: str, num_labels: int = 2):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    cfg.num_labels = num_labels
    # pad
    if tok.pad_token is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tok.pad_token_id
    cfg.pad_token_id = pad_id

    mdl = AutoModelForSequenceClassification.from_pretrained(base, config=cfg, trust_remote_code=True)
    if getattr(mdl.get_input_embeddings(), "num_embeddings", len(tok)) < len(tok):
        mdl.resize_token_embeddings(len(tok))
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = pad_id
    mdl.config.id2label = {0: "human", 1: "ai"}
    mdl.config.label2id = {"human": 0, "ai": 1}
    return mdl, tok


def extract_keywords_yake(text: str) -> Dict[str, float]:
    kw_extractor = yake.KeywordExtractor(lan='en', n=1)
    keywords = kw_extractor.extract_keywords(text)
    return dict(keywords)


def marginLoss(pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    max_dist = (dist * mask).max()
    cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) \
             + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    return cos_loss.mean()


def sclLoss(pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
    mask = mask - torch.diag(torch.diag(mask))
    cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
    cos_loss = -torch.log(cos_loss + 1e-5)
    cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
    return cos_loss.mean()


def marginLoss_yake(pooled: torch.Tensor, labels: torch.Tensor, keyword_scores: Dict[str, float], tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(pooled)
    for token, score in keyword_scores.items():
        tids = tokenizer.encode(token, add_special_tokens=False)
        if not tids:
            continue
        token_id = tids[0]
        token_positions = (input_ids == token_id).nonzero()
        for pos in token_positions:
            weights[pos[0], pos[1]] *= (2 - score)
    weighted_pooled = pooled * weights
    dist = ((weighted_pooled.unsqueeze(1) - weighted_pooled.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    max_dist = (dist * mask).max()
    cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) \
             + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    return cos_loss.mean()


# =========================
# Data plumbing for PECOLA
# =========================

@dataclass
class PecolaAugCfg:
    dataset_name: str
    prob: float = 0.10
    ratio: float = 0.05
    skip_number: int = 2
    t5_model: str = 't5-large'
    lang: str = 'en'
    seed: int = 114514
    cache_dir: str = os.path.join(os.path.expanduser('~'), '.cache', 'mgt_eval')  # 'pecola'


def _pecola_cache_key(examples: List[Dict[str, Any]], cfg: PecolaAugCfg) -> str:
    # Hash over labels/texts head + cfg
    h = hashlib.sha256()
    h.update(cfg.dataset_name.encode('utf-8'))
    h.update(f"prob={cfg.prob}|ratio={cfg.ratio}|skip={cfg.skip_number}|t5={cfg.t5_model}|lang={cfg.lang}".encode('utf-8'))
    # Sample a few examples to avoid hashing entire dataset (deterministic with seed)
    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(examples)); rng.shuffle(idx)
    for i in idx[:256]:
        e = examples[i]
        h.update(str(int(e["label"]) ).encode('utf-8'))
        h.update(e["text"].encode('utf-8'))
    return h.hexdigest()


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
# Utilities
def _safe_name(s: Any, maxlen: int = 80) -> str:
    name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(s))
    return name[:maxlen] if len(name) > maxlen else name

def _resolve_cache_dir(user_cache_dir: Optional[str]) -> str:
    """
    解析 PECOLA 的缓存根目录。优先级：
    1) 显式传入的 user_cache_dir（若是 'None' 或空，忽略）
    2) 环境变量 MGT_EVAL_CACHE_DIR
    3) 默认 ~/.cache/mgt_eval   （注意：不再强制加 'pecola' 子目录）
    """
    if user_cache_dir and str(user_cache_dir).lower() != "none":
        base = str(user_cache_dir)
    else:
        base = os.environ.get("MGT_EVAL_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "mgt_eval"))
    base = os.path.expandvars(os.path.expanduser(base))
    return os.path.abspath(base)


def _generate_or_load_perturbed_training(
    train_examples: List[Dict[str, Any]], cfg: PecolaAugCfg
) -> str:
    """Return path to cached JSONL with fields: article, article_delete, generater_text, label(str).
    label: 'human' or 'machine' (mapping from 0/1 by convention 0=human,1=machine).
    """
    cache_root = _resolve_cache_dir(cfg.cache_dir)          # ←
    _ensure_dir(cache_root)

    key = _pecola_cache_key(train_examples, cfg)

    dataset_dir = os.path.join(cache_root, _safe_name(cfg.dataset_name))
    model_dir   = os.path.join(dataset_dir, _safe_name(cfg.t5_model))
    _ensure_dir(model_dir)

    fname    = f"train_prob{cfg.prob}_ratio{cfg.ratio}_skip{cfg.skip_number}_{key[:10]}.jsonl"
    out_path = os.path.join(model_dir, fname)

    logger.info(f"[PECOLA] Cache root = {cache_root}")
    logger.info(f"[PECOLA] Cache file = {out_path}")

    if os.path.isfile(out_path):
        logger.info(f"[PECOLA] Found cached perturbed training set: {out_path}")
        return out_path

    logger.info("[PECOLA] No cache found. Generating selective perturbations (deletion + T5 fills)...")
    augmenter = TextAugmenter(lang=cfg.lang)
    augmented_items = []
    for e in tqdm(train_examples, desc="[PECOLA] Selective deletion", leave=False, dynamic_ncols=True):
        article = e["text"]
        label_int = int(e["label"])  # 0=human, 1=machine per unified interface
        label_str = "human" if label_int == 0 else "machine"
        article_delete = augmenter.aug_by_deletion(article, prob=cfg.prob, mode='selective', ratio=cfg.ratio, skip_number=cfg.skip_number)
        augmented_items.append({"article": article, "article_delete": article_delete, "label": label_str})

    # T5 fills
    t5 = T5TextProcessor(model_name=cfg.t5_model, random_seed=cfg.seed)
    filled = t5.process_augmented_and_fill(augmented_items)
    t5.unload()

    # Write cache
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in filled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"[PECOLA] Perturbed training set cached to: {out_path}")
    return out_path


# =========================
# Datasets & Dataloaders (follow official behavior)
# =========================

class _PecolaTrainDS(Dataset):
    """Triples the training data: original, deletion-masked, and T5-generated.
    Input: path to cached JSONL produced by _generate_or_load_perturbed_training.
    """
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj)
        # Build tripled texts & labels
        self.texts: List[str] = []
        self.labels: List[int] = []
        for o in self.items:
            lab = 0 if o['label'] == 'human' else 1
            self.texts.extend([o['article'], o['article_delete'], o['generater_text']])
            self.labels.extend([lab, lab, lab])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int) -> Tuple[str, int]:
        return self.texts[i], self.labels[i]


class _PecolaEvalDS(Dataset):
    """Evaluation DS: use only original text, label int.
    """
    def __init__(self, examples: List[Dict[str, Any]]):
        self.exs = examples

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i: int) -> Tuple[str, int]:
        e = self.exs[i]
        return e["text"], int(e["label"])


def _build_loader_textcls(ds: Dataset, tok, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    def _collate(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tok(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return enc["input_ids"], labels, enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, pin_memory=True)


# =========================
# Eval metrics (official style)
# =========================

W_EPOCH = 8
W_MEM   = 8
W_NUM   = 7
W_STEP  = 8
SEP     = " "


def test_acc(model, data_loader, metric: str = 'acc') -> Tuple[float, float, float, float, float]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []

    print("\n" + f"{'GPU_mem':>{W_MEM}}{SEP}{'Cur_acc':>{W_NUM}}{SEP}{'avg_acc':>{W_NUM}}{SEP}{'loss':>{W_NUM}}")
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), dynamic_ncols=True, leave=False)
    ce_loss = torch.nn.CrossEntropyLoss()
    avg_loss = 0.0

    with torch.no_grad():
        for i, (input_ids, batch_labels, attention_masks) in pbar:
            device = next(model.parameters()).device
            input_ids = input_ids.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(batch_preds.tolist())
            labels_np = batch_labels.detach().cpu().numpy().tolist()
            labels.extend(labels_np)

            vloss = ce_loss(logits, batch_labels)
            avg_loss = (avg_loss * i + float(vloss.item())) / (i + 1)

            cur_right = int((batch_preds == np.array(labels_np)).sum())
            cur_tot   = int(len(labels_np))
            cur_acc   = cur_right / max(1, cur_tot)
            avg_acc   = float((np.array(preds) == np.array(labels)).mean()) if len(labels) else 0.0
            desc = (
                f"{_gpu_mem_str():>{W_MEM}}{SEP}"
                f"{cur_acc:>{W_NUM}.4f}{SEP}"
                f"{avg_acc:>{W_NUM}.4f}{SEP}"
                f"{avg_loss:>{W_NUM}.4f}"
            )
            pbar.set_description(desc)

    preds_arr = np.array(preds); labels_arr = np.array(labels)
    acc = float((preds_arr == labels_arr).mean()) if len(labels_arr) else 0.0
    # Macro-F1 & recalls
    try:
        from sklearn.metrics import f1_score, confusion_matrix
        f1  = float(f1_score(labels_arr, preds_arr, average='macro'))
        tn, fp, fn, tp = confusion_matrix(labels_arr, preds_arr, labels=[0, 1]).ravel()
        machine_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # label=0
        human_recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # label=1
        recall_macro = 0.5 * (machine_recall + human_recall)
    except Exception:
        f1 = 0.0; machine_recall = 0.0; human_recall = 0.0; recall_macro = 0.0

    return float(acc), float(f1), float(recall_macro), float(human_recall), float(machine_recall)


# =========================
# Training loop (official logic, leave=False)
# =========================

def train_common_yake(
    model, optimizer, train_loader, epoch: int, *,
    pre_model=None, shift_reg: float=0.0, scl_reg: float=0.9,
    loss_type: str='ce',
    log_steps: int=50,
    save_steps: int=-1, save_dir: Optional[str]=None, step_counter: int=0,
    max_grad_norm: float=1.0,
    tok=None,
    step_indices: Optional[List[int]] = None,
    step_losses: Optional[List[float]] = None,
):
    model.train()
    if pre_model is not None:
        pre_model.eval()

    running_loss = 0.0
    step_count = 0
    loss_fct = torch.nn.CrossEntropyLoss()

    print("\n" +
          f"{'Epoch':>{W_EPOCH}}{SEP}"
          f"{'GPU_mem':>{W_MEM}}{SEP}"
          f"{'L':>{W_NUM}}{SEP}"
          f"{'CE':>{W_NUM}}{SEP}"
          f"{'shift':>{W_NUM}}{SEP}"
          f"{'aux':>{W_NUM}}{SEP}"
          f"{'avg':>{W_NUM}}{SEP}"
          f"{'lr':>{W_NUM}}{SEP}"
          f"{'step':>{W_STEP}}")

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True, leave=False)

    for i, (input_ids, labels, attention_masks) in pbar:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_masks = attention_masks.to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_masks, output_hidden_states=True)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        ce_loss = loss_fct(logits, labels)

        shift_val = 0.0
        if shift_reg > 0 and pre_model is not None:
            # numeric-only shift regularizer as in the reference snippet
            p_cur = torch.cat([p.view(-1) for n,p in model.named_parameters() if 'bert' in n.lower() or 'roberta' in n.lower()])
            p_pre = torch.cat([p.view(-1) for n,p in pre_model.named_parameters() if 'bert' in n.lower() or 'roberta' in n.lower()])
            shift_val = torch.sum(torch.abs(p_cur - p_pre) ** 2).item()

        aux_loss_val = 0.0
        if loss_type == 'margin_weight':
            pooled = outputs.hidden_states[-1][:, 0, :]
            # convert input ids back into string for YAKE keyword weighting
            # (we follow the original style: compute YAKE on decoded input)
            texts = tok.batch_decode(input_ids, skip_special_tokens=True) if tok is not None else []
            kw_scores = [extract_keywords_yake(t) for t in texts]
            total = 0.0
            for ks in kw_scores:
                total = total + marginLoss_yake(pooled, labels, ks, tok, input_ids)
            aux_loss_val = total
        elif loss_type == 'margin':
            pooled = outputs.hidden_states[-1][:, 0, :]
            aux_loss_val = marginLoss(pooled, labels)
        elif loss_type == 'scl':
            pooled = outputs.hidden_states[-1][:, 0, :]
            aux_loss_val = sclLoss(pooled, labels)

        loss = ce_loss + shift_reg * shift_val + (aux_loss_val if loss_type in {'margin_weight','margin','scl'} else 0.0)

        running_loss += float(loss.item())
        step_count += 1
        step_counter += 1
        if step_indices is not None:
            step_indices.append(int(step_counter))
        if step_losses is not None:
            step_losses.append(float(loss.item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        lr = optimizer.param_groups[0].get("lr", 0.0)
        avg_loss = running_loss / step_count

        desc = (
            f"{f'{epoch}':>{W_EPOCH}}{SEP}"
            f"{_gpu_mem_str():>{W_MEM}}{SEP}"
            f"{float(loss.item()):>{W_NUM}.4f}{SEP}"
            f"{float(ce_loss.item()):>{W_NUM}.4f}{SEP}"
            f"{(shift_reg * shift_val):>{W_NUM}.4f}{SEP}"
            f"{(aux_loss_val.item() if isinstance(aux_loss_val, torch.Tensor) else float(aux_loss_val)):>{W_NUM}.4f}{SEP}"
            f"{float(avg_loss):>{W_NUM}.4f}{SEP}"
            f"{float(lr):>{W_NUM}.2e}{SEP}"
            f"{int(step_counter):>{W_STEP}d}"
        )
        pbar.set_description(desc)

        if (step_count % log_steps) == 0:
            logger.info(
                f"[E{epoch}] step={step_counter} | CE={ce_loss.item():.6f} | shift={(shift_reg * shift_val):.6f} | "
                f"aux={(aux_loss_val.item() if isinstance(aux_loss_val, torch.Tensor) else float(aux_loss_val)):.6f} | "
                f"loss={loss.item():.6f} | avg={avg_loss:.6f} | lr={lr:.2e}"
            )

        if (save_steps != -1 and save_dir is not None) and (step_counter % save_steps == 0):
            model_to_save = model.module if hasattr(model, 'module') else model
            path = os.path.join(save_dir, f'step{step_counter}_model.pt')
            torch.save(model_to_save.state_dict(), path)
            logger.info(f"model saved to {path}")

    return step_counter


# =========================
# End-to-end trainer
# =========================

@dataclass
class PecolaTrainCfg:
    base_model: str = 'roberta-base'
    t5_model: str = 't5-large'
    lang: str = 'en'
    prob: float = 0.10
    ratio: float = 0.05
    skip_number: int = 2
    epochs: int = 30
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    loss_type: str = 'margin_weight'  # ['ce','scl','margin','margin_weight']
    scl_reg: float = 0.9
    shift_reg: float = 0.0
    seed: int = 114514
    cache_dir: str = os.path.expanduser('~/.cache/mgt_eval/pecola')


def _save_model_dir(model, tok, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def _train_pecola_core(
    *,
    train_cache_jsonl: str,
    val_examples: Optional[List[Dict[str,Any]]],
    test_examples: Optional[List[Dict[str,Any]]],
    cfg: PecolaTrainCfg,
    output_root: str,
    dataset_spec: Optional[str] = None,
) -> Dict[str, Any]:
    torch.set_grad_enabled(True)
    _seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logs
    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }

    run_dir = f"{output_root}_{_timestamp()}"
    os.makedirs(run_dir, exist_ok=True)

    # save args
    args_json_path = os.path.join(run_dir, "train_args.json")
    with open(args_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "args": cfg.__dict__,
            "env": env_info,
            "data": _build_data_info(dataset_spec, None, None),  # placeholder: we log sizes below
        }, f, ensure_ascii=False, indent=2)

    # model
    base = _resolve_base(cfg.base_model)
    model, tok = _prepare_seqcls(base, 2)
    model.to(device)

    # pre_model for optional shift regularizer
    pre_model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)
    pre_model.eval()

    # data
    train_ds = _PecolaTrainDS(train_cache_jsonl)
    val_ds   = _PecolaEvalDS(val_examples) if val_examples else None
    test_ds  = _PecolaEvalDS(test_examples) if test_examples else None

    logger.info(f"[PECOLA] Train triples: {len(train_ds)} (after tripling)")
    if val_ds:
        logger.info(f"[PECOLA] Val examples: {len(val_ds)}")
    if test_ds:
        logger.info(f"[PECOLA] Test examples: {len(test_ds)}")

    train_loader = _build_loader_textcls(train_ds, tok, 512, cfg.train_batch_size, shuffle=True)
    val_loader   = _build_loader_textcls(val_ds, tok, 512, cfg.eval_batch_size, shuffle=False) if val_ds else None
    test_loader  = _build_loader_textcls(test_ds, tok, 512, cfg.eval_batch_size, shuffle=False) if test_ds else None

    # optim
    no_decay = ["bias","LayerNorm.weight","layer_norm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":cfg.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)

    # peak mem tracking
    mem_ctx = _reset_and_mark_cuda_peaks()

    best_acc = -1.0
    best_dir = None
    last_dir = None
    global_step = 0
    history: List[Dict[str, Any]] = []
    step_indices: List[int] = []
    step_losses: List[float] = []

    for ep in range(1, cfg.epochs + 1):
        start_ep = time.time()
        global_step = train_common_yake(
            model, optimizer, train_loader, ep,
            pre_model=pre_model, shift_reg=cfg.shift_reg, scl_reg=cfg.scl_reg,
            loss_type=cfg.loss_type, log_steps=50, save_steps=-1, save_dir=None, step_counter=global_step,
            max_grad_norm=1.0, tok=tok,
            step_indices=step_indices, step_losses=step_losses,
        )
        ep_time = time.time() - start_ep

        # validation (if provided)
        epoch_record: Dict[str, Any] = {
            "epoch": int(ep),
            "epoch_wall_time_sec": float(ep_time),
            "global_step": int(global_step),
        }
        if val_loader is not None:
            acc, f1, recall_macro, human_rec, machine_rec = test_acc(model, val_loader, 'acc')
            logger.info(
                f"[PECOLA][Epoch {ep}] val_acc={acc:.4f} val_f1={f1:.4f} "
                f"recall_macro={recall_macro:.4f} human_rec={human_rec:.4f} machine_rec={machine_rec:.4f} "
                f"time={ep_time:.1f}s"
            )
            epoch_record.update({
                "val_acc": float(acc),
                "val_f1": float(f1),
                "val_recall_macro": float(recall_macro),
                "val_human_recall": float(human_rec),
                "val_machine_recall": float(machine_rec),
            })
            if acc > best_acc:
                best_acc = acc
                best_dir = os.path.join(run_dir, "best")
                _save_model_dir(model, tok, best_dir)
                logger.info(f"[PECOLA] best model saved to: {best_dir}")
        else:
            # no validation => save last each epoch
            last_dir = os.path.join(run_dir, "last")
            _save_model_dir(model, tok, last_dir)
            logger.info(f"[PECOLA] (no val) model checkpointed to: {last_dir}")
        history.append(epoch_record)

    mem_stats = _collect_cuda_peaks(mem_ctx)

    # final test (optional)
    test_metrics = None
    if test_loader is not None:
        acc, f1, recall_macro, human_rec, machine_rec = test_acc(model, test_loader, 'acc')
        test_metrics = {
            "acc": acc, "f1": f1, "recall_macro": recall_macro,
            "human_recall": human_rec, "machine_recall": machine_rec,
        }
        logger.info(f"[PECOLA][TEST] acc={acc:.4f} f1={f1:.4f} recall_macro={recall_macro:.4f} ")

    artifacts = {
        "args_json": os.path.join(run_dir, "train_args.json"),
        "summary_json": os.path.join(run_dir, "train_summary.json"),
    }
    with open(artifacts["summary_json"], 'w', encoding='utf-8') as f:
        json.dump({
            "best_dir": best_dir,
            "last_dir": last_dir,
            "best_val_acc": (None if best_acc < 0 else best_acc),
            "test_metrics": test_metrics,
            "history": history,
            "step_indices": step_indices,
            "step_losses": step_losses,
            "memory": mem_stats,
        }, f, ensure_ascii=False, indent=2)

    return {
        "best_dir": best_dir,
        "last_dir": last_dir,
        "best_val_acc": (None if best_acc < 0 else best_acc),
        "test_metrics": test_metrics,
        "history": history,
        "step_indices": step_indices,
        "step_losses": step_losses,
        "artifacts": artifacts,
        "run_dir": run_dir,
    }


# =========================
# Public training entry (registered)
# =========================

@register_train("pecola")
def train_pecola(**kwargs) -> Dict[str, Any]:
    """Train PECOLA using unified dataset interface.

    Required kwargs:
      - dataset_training: dataset spec understood by `load_dataset_unified`.

    Optional kwargs:
      - dataset_validation: dataset spec (no auto-sampling if not provided)
      - dataset_test: dataset spec (for final test report)
      - shot: int >= 32 (number of training examples to use)
      - validation_size, test_size: ints to cap evaluation sizes
      - output_dir: base output directory name (timestamp appended)
      - base_model, t5_model, lang, prob, ratio, skip_number, epochs, lr,
        train_batch_size, eval_batch_size, weight_decay, loss_type, scl_reg, shift_reg, seed
    """
    dataset_training   = kwargs.get("dataset_training", None)
    dataset_validation = kwargs.get("dataset_validation", None)
    dataset_test       = kwargs.get("dataset_test", None)
    assert dataset_training, "pecola requires dataset_training"

    shot  = kwargs.get("shot", 10000)
    if shot is not None and shot < 32:
        raise ValueError("PECOLA requires >= 32-shot for training.")

    validation_size       = kwargs.get("validation_size", None)
    test_size      = kwargs.get("test_size", None)

    output_dir_raw     = kwargs.get("output_dir", "runs_pecola")

    cfg = PecolaTrainCfg(
        base_model=kwargs.get("base_model", 'roberta-base'),
        t5_model=kwargs.get("t5_model", 't5-large'),
        lang=kwargs.get("lang", 'en'),
        prob=float(kwargs.get("prob", 0.10)),
        ratio=float(kwargs.get("ratio", 0.05)),
        skip_number=int(kwargs.get("skip_number", 2)),
        epochs=int(kwargs.get("epochs", 30)),
        train_batch_size=int(kwargs.get("train_batch_size", 16)),
        eval_batch_size=int(kwargs.get("eval_batch_size", 32)),
        lr=float(kwargs.get("lr", 1e-5)),
        weight_decay=float(kwargs.get("weight_decay", 0.01)),
        loss_type=str(kwargs.get("loss_type", 'margin_weight')),
        scl_reg=float(kwargs.get("scl_reg", 0.9)),
        shift_reg=float(kwargs.get("shift_reg", 0.0)),
        seed=int(kwargs.get("seed", 114514)),
        cache_dir=str(kwargs.get("cache_dir", os.path.join(os.path.expanduser('~'), '.cache', 'mgt_eval'))),
    )

    # User/info banners
    print(f"[mgt_eval] Using detector: {DETECTOR_NAME} (Type={detector_type})")
    print(f"[mgt_eval] Credits: {CITATION_AUTHORS} | Paper: {CITATION_TITLE} | Link: {CITATION_LINK}")
    print("[mgt_eval] Disclaimer: This wrapper preserves official PECOLA logic with only engineering glue.")

    # Load datasets via unified interface
    tr_exs, _ = load_dataset_unified(dataset=dataset_training, sample_k=shot, sample_seed=cfg.seed, group_cols=None)
    va_exs, _ = load_dataset_unified(dataset=dataset_validation, sample_k=validation_size, sample_seed=cfg.seed, group_cols=None) if dataset_validation else (None, None)
    te_exs, _ = load_dataset_unified(dataset=dataset_test, sample_k=test_size, sample_seed=cfg.seed, group_cols=None) if dataset_test else (None, None)

    logger.info(f"[PECOLA] Loaded training examples (pre-perturb): {len(tr_exs)}")
    if va_exs is not None:
        logger.info(f"[PECOLA] Loaded validation examples: {len(va_exs)}")
    if te_exs is not None:
        logger.info(f"[PECOLA] Loaded test examples: {len(te_exs)}")

    # Cache or generate perturbed training set
    aug_cfg = PecolaAugCfg(
        dataset_name=str(dataset_training), prob=cfg.prob, ratio=cfg.ratio,
        skip_number=cfg.skip_number, t5_model=cfg.t5_model, lang=cfg.lang,
        seed=cfg.seed, cache_dir=cfg.cache_dir,
    )
    train_cache_jsonl = _generate_or_load_perturbed_training(tr_exs, aug_cfg)

    # Train core
    res = _train_pecola_core(
        train_cache_jsonl=train_cache_jsonl,
        val_examples=va_exs,
        test_examples=te_exs,
        cfg=cfg,
        output_root=output_dir_raw,
        dataset_spec=str(dataset_training),
    )

    return {"train": res}


# =========================
# Simple detector for direct inference (registered)
# =========================

@register_detector("PECOLA")
class pecola(DetectorBase):
    """A simple finetuned classifier wrapper for inference using the trained PECOLA model dir.

    Usage:
        det = PecolaFT(model_dir="/path/to/pecola/best")
        scores = det.score_batch(["text1", "text2", ...])  # probability of label=1 (ai)
    """
    def __init__(self, model_dir: Optional[str] = None, base: Optional[str] = None, device: Optional[str] = None, **kwargs):
        super().__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if model_dir and _is_local_hf_dir(model_dir):
            self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            self.mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        else:
            base = _resolve_base(base or 'roberta-base')
            self.mdl, self.tok = _prepare_seqcls(base, 2)
            self.mdl.to(self.device)
        self.mdl.eval()

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> List[float]:
        enc = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k,v in enc.items()}
        logits = self.mdl(**enc).logits
        prob = torch.softmax(logits, dim=-1)[:, 1]  # label=1 (machine/ai)
        return prob.detach().cpu().tolist()

    def name(self) -> str:
        return DETECTOR_NAME

    def type(self) -> str:
        return detector_type
