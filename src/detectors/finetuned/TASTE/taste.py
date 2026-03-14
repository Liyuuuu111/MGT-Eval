# src/detectors/taste/taste.py
"""
TASTE 检测器 —— Learning From Dictionary: Enhancing Robustness of
Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training
ICLR 2026 | Authors: Yuanfan Li, Qi Zhou, Zexuan Xie

本模块实现：
  1. TasteDetector  —— 继承 DetectorBase，用于推理（evaluate_detector）
  2. train_taste    —— 对抗训练函数，通过 @register_train("taste") 注册到训练框架

参数说明：
  model1  : 基座模型 / tokenizer（默认 bert-base-multilingual-cased，即 mBERT）
  model2  : 替代/代理模型（默认 gpt2），仅用于训练阶段的 token 重要性计算
  dict_dir: 翻译词典目录（含 *-English.json），默认为本模块同级的 translation/ 目录
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

from ..registry import register
from ..base import DetectorBase
from train.registry import register_train
from data_utils.load import load_dataset_unified

# ─────────────── 元信息 ───────────────
DETECTOR_NAME    = "TASTE"
detector_type    = "Model-based"
CITATION_TITLE   = (
    "Learning From Dictionary: Enhancing Robustness of Machine-Generated Text "
    "Detection in Zero-Shot Language via Adversarial Training"
)
CITATION_AUTHORS = "Yuanfan Li, Qi Zhou, Zexuan Xie"
CITATION_LINK    = "https://openreview.net/forum?id=taste2026"  # placeholder — update if published

# 默认路径
_HERE = Path(__file__).parent
_DEFAULT_DICT_DIR = str(_HERE / "translation")
_DEFAULT_BASE_MODEL = "bert-base-multilingual-cased"
_DEFAULT_SURROGATE  = "gpt2"

# ─────────────── 工具函数 ───────────────
def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_local_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _has_hf_weight_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = (
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    return any(os.path.isfile(os.path.join(path, n)) for n in names)


def _find_state_dict_file(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    blocked = {
        "pytorch_model.bin",
        "adapter_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    }
    cand: List[Tuple[float, str]] = []
    for pat in ("*.pt", "*.bin"):
        for p in Path(path).glob(pat):
            if not p.is_file():
                continue
            name = p.name
            if name in blocked:
                continue
            try:
                mt = float(p.stat().st_mtime)
            except Exception:
                mt = -1.0
            cand.append((mt, str(p)))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]


def _unwrap_state_dict(state: Any) -> Any:
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict", "model"):
        v = state.get(key)
        if isinstance(v, dict):
            return v
    return state


def _resolve_model(model: Optional[str], fallback: str) -> str:
    if not model:
        return fallback
    spec = model.strip()
    if _is_local_hf_dir(spec):
        return spec
    # 简单别名
    _alias = {
        "mbert": "bert-base-multilingual-cased",
        "gpt-2": "gpt2",
    }
    if spec.lower() in _alias:
        return _alias[spec.lower()]
    try:
        AutoConfig.from_pretrained(spec)
        return spec
    except Exception:
        return fallback


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ─────────────── 翻译词典加载 ───────────────
def _load_dict(path: Path) -> Dict[str, str]:
    """加载单个 *-English.json 词典，返回 {原词小写: 译词} 映射。"""
    items = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for item in items:
        if "original" in item and "translated" in item:
            out[item["original"].lower()] = item["translated"]
    return out


def _make_translator(dic: Dict[str, str]):
    """返回一个 token -> 最长前缀匹配翻译 的函数。"""
    def _tr(token: str) -> str:
        low = token.lower()
        best = ""
        for src in dic:
            if low.startswith(src) and len(src) > len(best):
                best = src
        return dic[best] if best else token
    return _tr


def _load_all_dicts(dict_dir: str) -> List:
    """加载词典目录下所有 *-English.json，返回翻译器列表。"""
    d = Path(dict_dir)
    translators = []
    for p in sorted(d.glob("*-English.json")):
        try:
            dic = _load_dict(p)
            translators.append(_make_translator(dic))
        except Exception as e:
            print(f"[TASTE] WARNING: failed to load dict {p}: {e}")
    if not translators:
        raise FileNotFoundError(
            f"[TASTE] No *-English.json found in dict_dir='{dict_dir}'. "
            "Please ensure the translation directory exists."
        )
    return translators


# ─────────────── Dataset ───────────────
class _TrainDS(Dataset):
    def __init__(self, texts, labels, flags):
        self.texts  = texts
        self.labels = labels
        self.flags  = flags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i], self.flags[i]


class _ValDS(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


# ─────────────── 梯度重要性（使用代理模型 GPT-2）───────────────
@torch.no_grad()
def _topk_token_indices(
    proxy: GPT2ForSequenceClassification,
    proxy_tok: GPT2Tokenizer,
    text: str,
    k: int,
    device,
) -> List[int]:
    """利用代理模型梯度范数选出 top-k 重要 token 位置。"""
    proxy.eval()
    enc = proxy_tok(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # 需要梯度，所以单独开启
    emb = proxy.transformer.wte(enc["input_ids"])
    emb.requires_grad_(True)

    with torch.enable_grad():
        logits = proxy(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits
        loss   = F.cross_entropy(logits, torch.tensor([1], device=device))
        grad   = torch.autograd.grad(loss, emb)[0]

    grad = grad.norm(dim=-1).squeeze(0)
    k    = min(k, grad.size(0))
    return torch.topk(grad, k).indices.tolist()


# ─────────────── BPE 频率 & JSD ───────────────
def _bpe_freq(texts: List[str], tok, vocab_size: int, device) -> torch.Tensor:
    ids = tok(texts)["input_ids"]
    mat = torch.zeros(len(texts), vocab_size, device=device)
    for i, seq in enumerate(ids):
        for tid in seq:
            mat[i, tid] += 1
    mat += 1e-8
    return mat / mat.sum(dim=1, keepdim=True)


def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="batchmean")
        + F.kl_div(m.log(), q, reduction="batchmean")
    )


# ─────────────── 语言判别器（GRL）───────────────
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


class _LangDisc(nn.Module):
    def __init__(self, h: int = 768, n_lang: int = 2):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(h, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_lang),
        )

    def forward(self, h, lamb: float = 1.0):
        return self.clf(_GRL.apply(h, lamb))


# ═══════════════ TasteDetector ═══════════════
@register("taste")
class TasteDetector(DetectorBase):
    """
    TASTE 检测器（推理阶段）。

    参数
    ----
    model1 : str
        基座模型 / tokenizer，默认 ``bert-base-multilingual-cased``（mBERT）。
        若已有训练好的检测器 checkpoint，可将 model1 设为 checkpoint 目录，
        同时通过 ``model_path`` 或 ``checkpoint`` 参数指定权重路径。
    model2 : str
        替代模型（gpt2），推理时不使用，仅保留字段以与前端保持一致。
    model_path : str, optional
        训练好的检测器权重文件或目录。若为目录且含 config.json，
        则直接 from_pretrained 加载；否则作为 state_dict (.pt/.bin) 加载。
    max_length : int
        最大 token 长度（默认 512）。
    """

    DETECTOR_NAME    = DETECTOR_NAME
    CITATION_TITLE   = CITATION_TITLE
    CITATION_AUTHORS = CITATION_AUTHORS
    CITATION_LINK    = CITATION_LINK

    # 输出概率，关闭额外校准
    outputs_prob       = True
    disable_calibration = True
    detector_type      = "model-based"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model1_path: str = _resolve_model(
            kwargs.get("model1"), _DEFAULT_BASE_MODEL
        )
        self.model2_path: str = _resolve_model(
            kwargs.get("model2"), _DEFAULT_SURROGATE
        )
        # 已训练权重（目录 or .pt）
        self.model_path: Optional[str] = kwargs.get("model_path") or kwargs.get("checkpoint")
        self.max_length: int = int(kwargs.get("max_length", 512))

        self._tokenizer = None
        self._model     = None
        self._device    = None

    def load(self):
        super().load()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        tok_path = self.model1_path
        model_path = str(self.model_path).strip() if self.model_path else ""
        if model_path and os.path.isdir(model_path):
            # HF save_pretrained directory with standard weight files.
            if _is_local_hf_dir(model_path) and _has_hf_weight_files(model_path):
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                try:
                    tok = AutoTokenizer.from_pretrained(model_path)
                except Exception:
                    tok = AutoTokenizer.from_pretrained(tok_path)
            else:
                # Compatibility: directory contains only state_dict checkpoint (*.pt/*.bin).
                ckpt_file = _find_state_dict_file(model_path)
                if not ckpt_file:
                    raise FileNotFoundError(
                        f"[TASTE] model_path='{model_path}' has config/tokenizer but no HF weights "
                        f"(pytorch_model.bin/model.safetensors) and no state_dict (*.pt/*.bin)."
                    )
                cfg_src = model_path if _is_local_hf_dir(model_path) else tok_path
                cfg = AutoConfig.from_pretrained(cfg_src, num_labels=2)
                model = AutoModelForSequenceClassification.from_pretrained(tok_path, config=cfg)
                state = torch.load(ckpt_file, map_location="cpu")
                state = _unwrap_state_dict(state)
                if isinstance(state, dict):
                    state = {
                        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                        for k, v in state.items()
                    }
                missing, unexpected = model.load_state_dict(state, strict=False)
                try:
                    tok = AutoTokenizer.from_pretrained(model_path)
                except Exception:
                    tok = AutoTokenizer.from_pretrained(tok_path)
                print(
                    f"[TASTE] loaded state_dict from {ckpt_file} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
        elif model_path and os.path.isfile(model_path):
            # state_dict 文件 (.pt / .bin) — 用 model1_path 初始化架构
            cfg = AutoConfig.from_pretrained(tok_path, num_labels=2)
            model = AutoModelForSequenceClassification.from_pretrained(
                tok_path, config=cfg
            )
            state = torch.load(model_path, map_location="cpu")
            state = _unwrap_state_dict(state)
            if isinstance(state, dict):
                state = {
                    (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                    for k, v in state.items()
                }
            missing, unexpected = model.load_state_dict(state, strict=False)
            tok = AutoTokenizer.from_pretrained(tok_path)
            print(
                f"[TASTE] loaded state_dict from {model_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            # 没有指定权重 — 直接用 model1_path（可能是已 fine-tune 的目录）
            cfg = AutoConfig.from_pretrained(tok_path, num_labels=2)
            model = AutoModelForSequenceClassification.from_pretrained(
                tok_path, config=cfg
            )
            tok = AutoTokenizer.from_pretrained(tok_path)

        # ── 修复 GPT-2 类模型无 pad_token 的问题 ──
        # GPT-2 tokenizer 默认没有 pad_token，直接用 padding=True 会让
        # position_ids 超出 position_embedding 范围，导致 CUDA 越界断言。
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            model.config.pad_token_id = tok.eos_token_id
            # GPT-2 习惯左侧 padding（解码时注意力放在右侧）
            tok.padding_side = "left"
            print("[TASTE] pad_token not found, set to eos_token and padding_side=left")

        # ── 计算模型实际支持的最大序列长度 ──
        # 取 tokenizer 和模型 config 中的最小值，优先用 model_max_length，
        # 但上限不超过 position_embedding_type 对应的 max_position_embeddings。
        cfg_max = getattr(model.config, "max_position_embeddings", None)
        tok_max = getattr(tok, "model_max_length", None)
        # 有些 tokenizer 会返回极大值（如 1e30），需要过滤
        if tok_max and tok_max > 1_000_000:
            tok_max = None
        safe_max = min(
            filter(None, [cfg_max, tok_max, self.max_length])
        )
        self._safe_max_length = safe_max
        print(f"[TASTE] safe max_length={safe_max} (cfg={cfg_max}, tok={tok_max}, user={self.max_length})")

        self._tokenizer = tok
        self._model     = model.to(device).eval()
        self.is_loaded  = True
        print(f"[TASTE] Model loaded from: {self.model_path or self.model1_path}")

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """返回每条文本的 AI 类概率（label=1 的 softmax 值）。"""
        if not self.is_loaded:
            self.load()
        # 使用经过安全校验的 max_length，防止 position embedding 越界
        safe_max = getattr(self, "_safe_max_length", self.max_length)
        enc = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=safe_max,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.cuda.amp.autocast(enabled=(self._device.type == "cuda")):
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
        return probs


# ═══════════════ TASTE 训练函数 ═══════════════
@register_train("taste")
def train_taste(**kwargs) -> Dict[str, Any]:
    """
    TASTE 对抗训练函数（通过 @register_train("taste") 注册到 MGT-Eval 训练框架）。

    关键参数
    --------
    dataset_train   : str  训练数据集路径或 HF 标识符（必填）
    model1          : str  基座/tokenizer 模型（默认 bert-base-multilingual-cased）
    model2          : str  代理模型（默认 gpt2）
    dict_dir        : str  翻译词典目录（默认 src/detectors/taste/translation）
    output_dir      : str  输出目录（默认 models/runs_taste）
    batch_size      : int  批大小（默认 16）
    lr              : float 检测器学习率（默认 2e-5）
    proxy_lr        : float 代理模型学习率（默认 1e-5）
    epochs          : int  训练轮数（默认 3）
    seed            : int  随机种子（默认 42）
    device          : str  设备（默认 cuda）
    adv_sample_ratio: float 对抗样本比例（默认 0.15）
    adv_ratio       : float 对抗替换 token 比例（默认 0.4）
    sample_k        : int  最大训练样本数（默认 10000）
    """
    # ── 参数解析 ──
    dataset_train    = kwargs.get("dataset_train") or kwargs.get("dataset")
    if not dataset_train:
        raise ValueError("[TASTE] 必须提供 dataset_train 参数")

    model1 = _resolve_model(kwargs.get("model1"), _DEFAULT_BASE_MODEL)
    model2 = _resolve_model(kwargs.get("model2"), _DEFAULT_SURROGATE)
    dict_dir = kwargs.get("dict_dir", _DEFAULT_DICT_DIR)
    output_dir = kwargs.get("output_dir", "models/runs_taste")
    batch_size = int(kwargs.get("batch_size", 16))
    lr = float(kwargs.get("lr", 2e-5))
    proxy_lr = float(kwargs.get("proxy_lr", 1e-5))
    epochs = int(kwargs.get("epochs", 3))
    seed = int(kwargs.get("seed", 42))
    device_str = kwargs.get("device", "cuda")
    adv_ratio = float(kwargs.get("adv_ratio", 0.4))
    adv_s_ratio = float(kwargs.get("adv_sample_ratio", 0.15))
    sample_k = kwargs.get("sample_k", 10000)
    max_length = int(kwargs.get("max_length", 512))
    fp16_raw = kwargs.get("fp16", True)
    fp16 = str(fp16_raw).strip().lower() not in {"0", "false", "no", "off"}
    save_epoch_ckpt_raw = kwargs.get("save_epoch_ckpt", False)
    save_epoch_ckpt = str(save_epoch_ckpt_raw).strip().lower() in {"1", "true", "yes", "on"}

    _seed_everything(seed)
    ts = _timestamp()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    run_dir = f"{output_dir}_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    run_best = os.path.join(run_dir, "best")
    run_last = os.path.join(run_dir, "last")
    os.makedirs(run_best, exist_ok=True)
    os.makedirs(run_last, exist_ok=True)

    print(f"[TASTE] Training  | device={device} | model1={model1} | model2={model2}")
    print(f"[TASTE] dict_dir  = {dict_dir}")
    print(f"[TASTE] output    = {run_dir}")

    # ── 翻译词典 ──
    translators = _load_all_dicts(dict_dir)
    print(f"[TASTE] Loaded {len(translators)} translation dictionaries.")

    # ── 加载数据 ──
    examples, _ = load_dataset_unified(
        dataset=dataset_train,
        sample_k=sample_k,
        sample_seed=seed,
        group_cols=None,
    )
    n_total = len(examples)
    if n_total < 2:
        raise ValueError(f"[TASTE] dataset is too small for train/val split: n={n_total}")

    # 8:1 分 train/val
    n_train = int(n_total * 8 / 9)
    n_train = min(max(1, n_train), n_total - 1)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    train_exs = [examples[i] for i in idx[:n_train]]
    val_exs = [examples[i] for i in idx[n_train:]]

    train_texts = [e["text"] for e in train_exs]
    train_labels = [int(e["label"]) for e in train_exs]
    val_texts = [e["text"] for e in val_exs]
    val_labels = [int(e["label"]) for e in val_exs]

    print(f"[TASTE] train={len(train_texts)}, val={len(val_texts)}")

    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
    args_json_path = os.path.join(run_dir, "train_args.json")
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {
                    "dataset_train": dataset_train,
                    "model1": model1,
                    "model2": model2,
                    "dict_dir": dict_dir,
                    "output_dir": output_dir,
                    "batch_size": batch_size,
                    "lr": lr,
                    "proxy_lr": proxy_lr,
                    "epochs": epochs,
                    "seed": seed,
                    "device": str(device),
                    "adv_ratio": adv_ratio,
                    "adv_sample_ratio": adv_s_ratio,
                    "sample_k": sample_k,
                    "max_length": max_length,
                    "fp16": fp16,
                    "save_epoch_ckpt": save_epoch_ckpt,
                },
                "env": env_info,
                "data": {
                    "dataset": dataset_train,
                    "loaded_total": n_total,
                    "train_size": len(train_texts),
                    "val_size": len(val_texts),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 对抗样本 flag
    n_adv = int(adv_s_ratio * len(train_texts))
    adv_flags = [1] * n_adv + [0] * (len(train_texts) - n_adv)
    random.shuffle(adv_flags)

    # ── Tokenizer & 检测器 ──
    det_tok = AutoTokenizer.from_pretrained(model1)
    cfg = AutoConfig.from_pretrained(model1, num_labels=2)
    cfg.num_labels = 2
    cfg.output_hidden_states = True
    if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
        cfg.pad_token_id = det_tok.pad_token_id or 0
    det = AutoModelForSequenceClassification.from_pretrained(model1, config=cfg).to(device)

    lang_disc = _LangDisc(det.config.hidden_size, n_lang=2).to(device)

    # ── 代理模型 GPT-2 ──
    proxy_tok = GPT2Tokenizer.from_pretrained(model2)
    if proxy_tok.pad_token is None:
        proxy_tok.pad_token = proxy_tok.eos_token
    proxy = GPT2ForSequenceClassification.from_pretrained(model2).to(device)
    proxy.resize_token_embeddings(len(proxy_tok))
    proxy.config.pad_token_id = proxy_tok.pad_token_id
    proxy.eval()

    # ── 优化器 / Scaler ──
    opt_det = torch.optim.Adam(det.parameters(), lr=lr)
    opt_lang  = torch.optim.Adam(lang_disc.parameters(), lr=1e-3)
    opt_proxy = torch.optim.Adam(proxy.parameters(), lr=proxy_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and device.type == "cuda"))
    sc_proxy = torch.amp.GradScaler("cuda", enabled=(fp16 and device.type == "cuda"))

    # ── Dataloader ──
    def _collate_train(batch):
        txt, lab, flg = zip(*batch)
        enc = det_tok(
            list(txt), return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        return (
            {k: v.to(device) for k, v in enc.items()},
            list(txt),
            torch.tensor(lab, device=device),
            torch.tensor(flg, device=device),
        )

    def _collate_val(batch):
        txt, lab = zip(*batch)
        enc = det_tok(
            list(txt), return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        return {k: v.to(device) for k, v in enc.items()}, torch.tensor(lab, device=device)

    train_ds = _TrainDS(train_texts, train_labels, adv_flags)
    val_ds = _ValDS(val_texts, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate_train)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_val)

    # ── 训练循环 ──
    λ_BPE = 0.0
    total_start = time.perf_counter()
    history = []
    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = -1
    best_dir = None
    last_dir = None

    for ep in range(epochs):
        epoch_start = time.perf_counter()
        det.train(); lang_disc.train()
        bar = tqdm(
            enumerate(train_loader, 1), total=len(train_loader),
            desc=f"[TASTE] Ep {ep+1}/{epochs}",
        )
        ep_loss_sum, ep_steps = 0.0, 0

        for step, (tok_batch, texts, labels, flags) in bar:
            # 1) 检测器前向（原始）
            with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                out_ori = det(**tok_batch)

            pseudo = out_ori.logits.argmax(dim=-1).detach()

            # 2) 代理模型跟随（epoch > 0）
            if ep >= 1:
                proxy.train()
                with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                    px_in = proxy_tok.batch_encode_plus(
                        texts, return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length,
                    ).to(device)
                    pr_logits = proxy(**px_in).logits
                    loss_proxy = F.cross_entropy(pr_logits, pseudo)
                opt_proxy.zero_grad()
                sc_proxy.scale(loss_proxy).backward()
                sc_proxy.step(opt_proxy); sc_proxy.update()
                proxy.eval()

            # 3) 对抗样本生成
            α = min(1.0, (ep * len(train_loader) + step) / (2 * len(train_loader)))
            adv_texts = []
            for text, flag in zip(texts, flags.tolist()):
                if flag == 1:
                    toks = det_tok.tokenize(text)
                    k_max = max(1, int(adv_ratio * len(toks)))
                    k_cur = max(1, int(α * k_max))
                    idx_b = _topk_token_indices(proxy, proxy_tok, text, k_cur, device)
                    for j in idx_b:
                        if j < len(toks):
                            toks[j] = random.choice(translators)(toks[j])
                    adv_texts.append(det_tok.convert_tokens_to_string(toks))
                else:
                    adv_texts.append(text)

            tok_adv = det_tok(
                adv_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length,
            )
            tok_adv = {k: v.to(device) for k, v in tok_adv.items()}

            # 4) 检测器损失
            with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                out_adv = det(**tok_adv, output_hidden_states=True)
                loss_cls = (
                    F.cross_entropy(out_ori.logits, labels)
                    + F.cross_entropy(out_adv.logits, labels)
                ) / 2

                loss_bpe = _jsd(
                    _bpe_freq(texts, det_tok, det_tok.vocab_size, device),
                    _bpe_freq(adv_texts, det_tok, det_tok.vocab_size, device),
                )
                h_cls = out_adv.hidden_states[-1][:, 0, :]
                lang_logits = lang_disc(h_cls, 1.0)
                lang_lbl = torch.where(
                    flags == 1,
                    torch.ones_like(flags),
                    torch.zeros_like(flags),
                )
                loss_lang = F.cross_entropy(lang_logits, lang_lbl)
                λ_DA = min(0.5, 0.1 + 0.4 * ep / epochs)
                loss_det = loss_cls + λ_BPE * loss_bpe + λ_DA * loss_lang

            opt_det.zero_grad(); opt_lang.zero_grad()
            scaler.scale(loss_det).backward()
            scaler.step(opt_det); scaler.step(opt_lang); scaler.update()

            ep_loss_sum += float(loss_det.item()); ep_steps += 1
            bar.set_postfix(L=f"{loss_det.item():.4f}", α=f"{α:.2f}")

        # ── Validation ──
        det.eval()
        val_preds, val_golds = [], []
        with torch.no_grad():
            for tok_v, lab_v in val_loader:
                logits_v = det(**tok_v).logits
                val_preds.extend(logits_v.argmax(-1).tolist())
                val_golds.extend(lab_v.tolist())

        from sklearn.metrics import accuracy_score, f1_score
        if val_golds:
            acc = float(accuracy_score(val_golds, val_preds))
            f1 = float(f1_score(val_golds, val_preds, zero_division=0))
        else:
            acc = 0.0
            f1 = 0.0
        avg_loss = ep_loss_sum / max(1, ep_steps)
        epoch_time = time.perf_counter() - epoch_start
        print(
            f"[TASTE][Epoch {ep+1}] loss={avg_loss:.4f} "
            f"val_acc={acc:.4f} val_f1={f1:.4f} time={epoch_time:.1f}s"
        )

        det.save_pretrained(run_last)
        det_tok.save_pretrained(run_last)
        last_dir = run_last

        is_better = (f1 > best_val_f1) or (
            math.isclose(f1, best_val_f1) and acc >= best_val_acc
        )
        if is_better:
            best_val_acc = acc
            best_val_f1 = f1
            best_epoch = ep + 1
            det.save_pretrained(run_best)
            det_tok.save_pretrained(run_best)
            best_dir = run_best

        if save_epoch_ckpt:
            ep_dir = os.path.join(run_dir, f"epoch_{ep + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            det.save_pretrained(ep_dir)
            det_tok.save_pretrained(ep_dir)
            print(f"[TASTE] Checkpoint saved -> {ep_dir}")

        history.append({
            "epoch": ep + 1,
            "avg_train_loss": avg_loss,
            "val_acc": acc,
            "val_f1": f1,
            "epoch_time_sec": epoch_time,
        })

    if best_dir is None:
        det.save_pretrained(run_best)
        det_tok.save_pretrained(run_best)
        best_dir = run_best
        if history:
            last_h = history[-1]
            best_val_acc = float(last_h.get("val_acc", 0.0))
            best_val_f1 = float(last_h.get("val_f1", 0.0))
            best_epoch = int(last_h.get("epoch", epochs))

    total_wall_time = time.perf_counter() - total_start
    print(f"[TASTE] Training done | best_epoch={best_epoch} | total={total_wall_time / 60.0:.1f} min")

    memory: Dict[str, Any] = {}
    if device.type == "cuda":
        try:
            memory = {
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
        except Exception:
            memory = {}

    summary_path = os.path.join(run_dir, "train_summary.json")
    summary = {
        "best_dir": best_dir,
        "last_dir": last_dir or run_last,
        "model_dir": best_dir,
        "best_val_acc": (None if best_val_acc < 0 else best_val_acc),
        "best_val_f1": (None if best_val_f1 < 0 else best_val_f1),
        "best_epoch": best_epoch,
        "history": history,
        "memory": memory,
        "timing": {
            "total_wall_time_sec": total_wall_time,
        },
        "artifacts": {
            "args_json": args_json_path,
            "summary_json": summary_path,
            "best_dir": best_dir,
            "last_dir": last_dir or run_last,
        },
        "run_dir": run_dir,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "best_dir": best_dir,
        "last_dir": last_dir or run_last,
        "model_dir": best_dir,
        "best_val_acc": (None if best_val_acc < 0 else best_val_acc),
        "best_val_f1": (None if best_val_f1 < 0 else best_val_f1),
        "best_epoch": best_epoch,
        "history": history,
        "artifacts": summary["artifacts"],
        "timing": summary["timing"],
        "run_dir": run_dir,
    }


# ── 兼容 __all__ —— 供 from .taste.taste import * 使用 ──
__all__ = ["TasteDetector", "train_taste"]
