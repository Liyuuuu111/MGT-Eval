# mgt_eval/detectors/finetuned/longformer.py
from __future__ import annotations

import os
import random
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    LongformerForSequenceClassification,
    RobertaTokenizer,
)

from mgt_eval.data_utils.load import load_dataset_unified
from mgt_eval.train.train import train_model
from mgt_eval.train.registry import register_train


# ---------------- basic utils ----------------

def _seed_everything(seed: int = 114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_local_hf_dir(path: str) -> bool:
    """判断是否为本地 HF 模型目录（含 config.json）"""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _resolve_longformer_base(model: Optional[str]) -> str:
    """
    解析 Longformer 基础模型：
      1) 本地 HF 目录（含 config.json） -> 直接返回
      2) 别名 -> 对应 HF id
      3) 其它 -> 当作 HF id
      4) 全部失败 -> 回退 'allenai/longformer-base-4096'
    """
    fallback = "allenai/longformer-base-4096"
    if not model:
        return fallback

    spec = model.strip()

    # 1) 本地 HF 目录
    if _is_local_hf_dir(spec):
        return spec

    # 2) 简单别名
    alias = {
        "longformer": "allenai/longformer-base-4096",
        "longformer-base-4096": "allenai/longformer-base-4096",
    }
    if spec.lower() in alias:
        return alias[spec.lower()]

    # 3) 尝试当作 HF id
    try:
        AutoConfig.from_pretrained(spec)
        return spec
    except Exception:
        return fallback


# ---------------- dataset & split ----------------

class TLDS(Dataset):
    """
    Tiny Labeled Dataset:
    统一封装成 {"text": str, "label": int} 的形式，供 train_model 使用。
    """

    def __init__(self, exs: List[Dict[str, Any]]):
        self.exs = exs

    def __len__(self) -> int:
        return len(self.exs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.exs[idx]
        return {
            "text": e["text"],
            "label": int(e["label"]),
        }


def _stratified_split(
    examples: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 114514,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按 label=0/1 分层划分 train/val/test。
    """
    if train_ratio <= 0 and val_ratio <= 0 and test_ratio <= 0:
        train_ratio, val_ratio, test_ratio = 1.0, 0.0, 0.0

    pos = [e for e in examples if int(e["label"]) == 1]
    neg = [e for e in examples if int(e["label"]) == 0]

    def _split(lst: List[Dict[str, Any]]):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        tot = len(idx)
        S = train_ratio + val_ratio + test_ratio
        if S <= 0:
            return idx, [], []
        n_tr = int(round(tot * (train_ratio / S)))
        n_va = int(round(tot * (val_ratio / S)))
        n_tr = min(n_tr, tot)
        n_va = min(n_va, tot - n_tr)
        n_te = tot - n_tr - n_va
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    p_tr, p_va, p_te = _split(pos)
    n_tr, n_va, n_te = _split(neg)

    train = [pos[i] for i in p_tr] + [neg[i] for i in n_tr]
    val = [pos[i] for i in p_va] + [neg[i] for i in n_va]
    test = [pos[i] for i in p_te] + [neg[i] for i in n_te]

    rng = np.random.RandomState(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------- TrainCfg ----------------

@dataclass
class TrainCfg:
    """
    Longformer 序列分类任务训练配置。
    与 finetuned.py 保持风格一致，但去掉了 LoRA 相关字段。
    """
    base_model: str
    output_dir: str

    max_length: int = 2048
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    train_batch_size: int = 2
    eval_batch_size: int = 4
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    fp16: bool = True
    label_smoothing: float = 0.0
    seed: int = 114514
    device: Optional[str] = None
    name: Optional[str] = None

    # 特别：Longformer 使用的 tokenizer 路径（默认你的本地 RoBERTa）
    tokenizer_path: str = None

    # ⭐ 兼容旧接口：当未显式传入 train/val/test 数据集时，用比例做划分
    train_ratio: float = 8.0
    val_ratio: float = 1.0
    test_ratio: float = 1.0

# ---------------- model & tokenizer ----------------

def _prepare_longformer_seqcls(
    base: str,
    num_labels: int = 2,
    tokenizer_path: Optional[str] = None,
    max_length: int = 2048,
) -> Tuple[LongformerForSequenceClassification, RobertaTokenizer]:
    """
    构建 LongformerForSequenceClassification + 本地 RoBERTa tokenizer。
    - base: HF 模型 id 或本地目录
    - tokenizer_path: RoBERTa tokenizer 的本地目录
    """
    tok_dir = tokenizer_path

    # 严格按照你之前脚本的要求：用 RobertaTokenizer.from_pretrained(本地 roberta 目录)
    tokenizer = RobertaTokenizer.from_pretrained(tok_dir)
    tokenizer.model_max_length = max_length

    # 读取 Longformer 的 config
    cfg = AutoConfig.from_pretrained(base)
    cfg.num_labels = num_labels
    cfg.pad_token_id = tokenizer.pad_token_id

    # 加载 LongformerForSequenceClassification
    model = LongformerForSequenceClassification.from_pretrained(base, config=cfg)

    # 确保 embedding 大小 >= tokenizer vocab size
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # 标签映射（HC3：0=human, 1=ai）
    model.config.id2label = {0: "human", 1: "ai"}
    model.config.label2id = {"human": 0, "ai": 1}

    return model, tokenizer


# ---------------- 时间戳 & 输出路径 ----------------

from datetime import datetime


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------- 训练实现 ----------------

def _train_longformer_impl(
    *,
    model: str,
    # 旧接口：单一数据集 + 比例划分
    dataset: Optional[str] = None,
    sample_k: Optional[int] = None,
    train_ratio: float = 8.0,
    val_ratio: float = 1.0,
    test_ratio: float = 1.0,

    # 新接口：显式指定 train/val/test 数据集与各自 sample_k
    dataset_train: Optional[str] = None,
    dataset_val: Optional[str] = None,
    dataset_test: Optional[str] = None,
    sample_k_train: Optional[int] = None,
    sample_k_val: Optional[int] = None,
    sample_k_test: Optional[int] = None,

    output_dir: Optional[str] = None,
    max_length: int = 2048,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 3,
    train_batch_size: int = 2,
    eval_batch_size: int = 4,
    warmup_ratio: float = 0.06,
    grad_accum_steps: int = 1,
    fp16: bool = True,
    label_smoothing: float = 0.0,
    seed: int = 114514,
    device: Optional[str] = None,
    name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Dict[str, Any]:

    """
    Longformer 序列分类训练入口，实现与 finetuned._train_seqcls_impl 相似的逻辑，只是模型固定为 Longformer。
    """
    _seed_everything(seed)
    torch.set_grad_enabled(True)

    # 1) load data: 兼容 finetuned.py，默认 hc3
    # 1) 加载数据
    # 优先：显式提供 train/val/test 数据集，则分别加载并按各自 sample_k 控制条数；
    # 否则：退回到旧逻辑，在单一 dataset 上按比例划分。
    if dataset_train is not None or dataset_val is not None or dataset_test is not None:
        # ---- 显式数据集模式 ----
        # train
        train_spec = dataset_train or dataset or "hc3"
        train_k = sample_k_train if sample_k_train is not None else sample_k
        train_exs, _ = load_dataset_unified(
            dataset=train_spec,
            sample_k=train_k,
            sample_seed=seed,
            group_cols=None,
        )

        # val（可选）
        val_exs = []
        if dataset_val is not None:
            val_k = sample_k_val if sample_k_val is not None else None
            val_exs, _ = load_dataset_unified(
                dataset=dataset_val,
                sample_k=val_k,
                sample_seed=seed,
                group_cols=None,
            )

        # test（可选）
        test_exs = []
        if dataset_test is not None:
            test_k = sample_k_test if sample_k_test is not None else None
            test_exs, _ = load_dataset_unified(
                dataset=dataset_test,
                sample_k=test_k,
                sample_seed=seed,
                group_cols=None,
            )

    else:
        # ---- 旧模式：单一 dataset + 分层划分 ----
        if not dataset:
            examples, _ = load_dataset_unified(
                dataset="hc3",
                sample_k=sample_k,
                sample_seed=seed,
                group_cols=None,
            )
            dataset = "hc3"
        else:
            examples, _ = load_dataset_unified(
                dataset=dataset,
                sample_k=sample_k,
                sample_seed=seed,
                group_cols=None,
            )

        train_exs, val_exs, test_exs = _stratified_split(
            examples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # 将统一格式转换为 {"text": xxx, "label": 0/1}
    train_ds = TLDS(train_exs)
    val_ds = TLDS(val_exs) if val_exs else None
    test_ds = TLDS(test_exs) if test_exs else None

    # 2) resolve base
    base = _resolve_longformer_base(model)
    ts = _timestamp()

    # 3) cfg
    base_dir = output_dir or os.path.join("runs_longformer", os.path.basename(base))
    out_dir = f"{base_dir}_{ts}"
    cfg = TrainCfg(
        base_model=base,
        output_dir=out_dir,
        max_length=max_length,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        grad_accum_steps=grad_accum_steps,
        fp16=fp16,
        label_smoothing=label_smoothing,
        seed=seed,
        device=device,
        name=name,
        tokenizer_path=tokenizer_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # 4) 构建模型 & tokenizer（使用 HF Longformer + 本地 RoBERTa tokenizer）
    model_obj, tokenizer = _prepare_longformer_seqcls(
        base=base,
        num_labels=2,
        tokenizer_path=cfg.tokenizer_path,
        max_length=cfg.max_length,
    )

    # 5) 调用通用 train_model
    #   - 训练循环（包括 tqdm、loss 打印等）由 train_model 内部统一实现
    #   - 我们只需要提供模型、tokenizer、数据集和配置
    dataset_spec = {
        "train": dataset_train or dataset or "hc3",
        "val": dataset_val,
        "test": dataset_test,
    }

    result = train_model(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg,
        dataset_spec=dataset_spec,
    )

    # 训练产出目录（best 优先；否则 last）
    model_dir = result.get("best_dir") or result.get("last_dir") or cfg.output_dir
    return {
        "model_dir": model_dir,
        "best_val_acc": result.get("best_val_acc", None),
        "split_sizes": {
            "train": len(train_ds),
            "val": len(val_ds) if val_ds is not None else 0,
            "test": len(test_ds) if test_ds is not None else 0,
        },
        "test_examples": test_exs,
        "config": cfg.__dict__,
        "base_model": base,
    }

# ---------------- 注册接口 ----------------

_ALLOWED_KEYS = {
    "model",
    # 旧接口：单一数据集 + 比例划分
    "dataset",
    "sample_k",
    "train_ratio",
    "val_ratio",
    "test_ratio",

    # 新接口：显式传入三份数据集，并分别控制条数
    "dataset_train",
    "dataset_val",
    "dataset_test",
    "sample_k_train",
    "sample_k_val",
    "sample_k_test",

    "output_dir",
    "max_length",
    "lr",
    "weight_decay",
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "warmup_ratio",
    "grad_accum_steps",
    "fp16",
    "label_smoothing",
    "seed",
    "device",
    "name",
    "tokenizer_path",  # 允许从 CLI / 调用方覆写 tokenizer 目录
}


def _clean_kwargs(kwargs: dict) -> dict:
    """
    只保留本实现支持的参数键。
    """
    # 允许 base_model 作为别名覆盖 model
    base_override = kwargs.pop("base_model", None)
    if base_override is not None and str(base_override).strip():
        kwargs["model"] = base_override
    return {k: v for k, v in kwargs.items() if k in _ALLOWED_KEYS}


@register_train("longformer-base-4096", "longformer")
def train_longformer(**kwargs) -> Dict[str, Any]:
    """
    注册入口：
      - 在 CLI / 代码中传 model="longformer-base-4096" 或 "longformer" 时调用。
      - 其余参数与 finetuned 中的一致，如：
        train_ratio / val_ratio / test_ratio / sample_k / output_dir 等。
    """
    return _train_longformer_impl(**_clean_kwargs(kwargs))
