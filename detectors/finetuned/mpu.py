# mgt_eval/detectors/finetuned/mpu.py
from __future__ import annotations

import os, json, math, time, random, platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import io
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

W_EPOCH = 8
W_MEM = 8
W_NUM = 7  # 适配诸如 30.000545 和 1.23e-07
W_STEP = 8
SEP = " "  # 或者用 " | " 可读性更强

# ---- 优化 A100 上的 FP32 matmul：启用 TF32 / 提升精度等级 ----
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ---- mgt_eval 统一接口/工具 ----
from mgt_eval.data_utils.load import load_dataset_unified
from mgt_eval.train.registry import register_train
from mgt_eval.train.train import (
    _reset_and_mark_cuda_peaks,
    _collect_cuda_peaks,
    _save_loss_plot,
    _build_data_info,
)

# ============== 方法元信息（用于日志/可追溯） ==============
DETECTOR_NAME = "MPU"
detector_type = "Model-based"
CITATION_AUTHORS = "Anonymous"
CITATION_TITLE = "MPU: Multiscale Positive–Unlabeled Learning for Short AI-Text Detection"
CITATION_LINK = "https://arxiv.org/abs/2305.18149"

# ============== 环境静默设置（避免 tokenizers 多线程 + fork 死锁） ==============
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ============== 公共小工具 ==============
def _seed_everything(seed: int = 114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== 数据加载（自动对齐 mgt_eval 样式：text/label/meta） ==============
def _load_one_dataset_spec(spec: str, seed: int) -> List[Dict[str, Any]]:
    """
    返回统一样本列表：
    {"text": str, "label": int(0=human,1=ai), "meta": {...}}
    """
    exs, _ = load_dataset_unified(
        dataset=spec,
        sample_k=None,
        sample_seed=seed,
        group_cols=None
    )
    return exs


def _stratified_split(examples: List[Dict[str, Any]], tr_r: float, va_r: float, te_r: float, seed: int = 114514):
    pos = [e for e in examples if int(e["label"]) == 1]  # ai=1
    neg = [e for e in examples if int(e["label"]) == 0]  # human=0

    def _split(lst):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(lst)); rng.shuffle(idx)
        S = tr_r + va_r + te_r
        n_tr = int(round(len(idx) * (tr_r / S))) if S > 0 else len(idx)
        n_va = int(round(len(idx) * (va_r / S))) if S > 0 else 0
        n_tr = min(n_tr, len(idx)); n_va = min(n_va, len(idx) - n_tr)
        return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    p_tr, p_va, p_te = _split(pos); n_tr, n_va, n_te = _split(neg)
    tr = [pos[i] for i in p_tr] + [neg[i] for i in n_tr]
    va = [pos[i] for i in p_va] + [neg[i] for i in n_va]
    te = [pos[i] for i in p_te] + [neg[i] for i in n_te]
    rng = np.random.RandomState(seed); rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

# ---- 新增：根据给定数量随机截断样本（训练 / 验证集通用）----
def _limit_examples(examples: List[Dict[str, Any]],
                    limit: Optional[int],
                    seed: int = 114514) -> List[Dict[str, Any]]:
    """
    若 limit 为正且小于当前样本数，则按随机子集抽取 limit 个样本；
    否则直接返回原列表。
    """
    if limit is None or limit <= 0 or limit >= len(examples):
        return examples
    rng = np.random.RandomState(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    idx = idx[:limit]
    return [examples[i] for i in idx]
# ============== 句级多尺度增强（与给定实现一致） ==============
import random as _rnd
from nltk.tokenize import sent_tokenize

def _single_multi_scale_augment(data: str, min_length: int = 50, aug_mode: str = 'sentence_deletion-0.25') -> str:
    lines = sent_tokenize(data)
    if len(lines) <= 1:
        return data

    if 'sentence_deletion' in aug_mode:
        name, strp = aug_mode.split('-')
        p = float(strp)
        new_sentences = []
        for sentence in lines:
            r = _rnd.uniform(0, 1)
            if r > p:
                new_sentences.append(sentence)
        if len(new_sentences) < 1:
            return data
        return ' '.join(new_sentences)
    else:
        raise NotImplementedError(f'Multiscaling mode {aug_mode} not implemented!')

def _multi_scale_augment(data: str, min_length: int = 50, aug_mode: str | List[str] = 'sentence_deletion-0.25') -> str:
    if isinstance(aug_mode, list):
        new_data = data
        for aug in aug_mode:
            new_data = _single_multi_scale_augment(new_data, min_length, aug)
        return new_data
    return _single_multi_scale_augment(data, min_length, aug_mode)


# ============== nnPU（保持你给的实现等价行为） ==============
class PULossauto:
    def __init__(self):
        self.prior = 0
        self.label = 0

    def apply(self, input, label, prior):
        self.input = input
        self.label = label
        if isinstance(prior, float):
            prior = torch.tensor(prior)
        self.prior = prior.to(input.device).float()
        self.positive = 1
        self.unlabeled = -1
        self.loss_func = lambda x: torch.sigmoid(-x)
        self.beta = 0
        self.gamma = 1

        self.positive_x = (self.label == self.positive).float()
        self.unlabeled_x = (self.label == self.unlabeled).float()
        self.positive_num = torch.max(torch.sum(self.positive_x), torch.tensor(1).to(input.device).float())
        self.unlabeled_num = torch.max(torch.sum(self.unlabeled_x), torch.tensor(1).to(input.device).float())
        self.positive_y = self.loss_func(self.input)
        self.unlabeled_y = self.loss_func(-self.input)
        self.positive_loss = torch.sum(self.prior * self.positive_x / self.positive_num * self.positive_y.squeeze())
        self.negative_loss = torch.sum((self.unlabeled_x / self.unlabeled_num - self.prior * self.positive_x / self.positive_num) * self.unlabeled_y.squeeze())
        objective = self.positive_loss + self.negative_loss

        if self.negative_loss.data < -self.beta:
            objective = self.positive_loss - self.beta
            self.x_out = -self.gamma * self.negative_loss
        else:
            self.x_out = objective
        return objective


def _expectation_matrix(length: int, pi: float, device='cpu'):
    if length < 3:
        return torch.tensor(pi).float().to(device)
    state = torch.zeros((1, length+1)).float().to(device)
    state[0, 0] += 1.
    trans = torch.zeros((length+1, length+1)).float().to(device)
    trans[1:, :-1] += torch.eye(length).to(device)*pi
    trans[:-1, 1:] += torch.eye(length).to(device)*(1-pi)
    trans[0,0] += pi
    trans[length, length] += (1-pi)

    total_trans = torch.zeros_like(trans) + torch.eye(length+1).to(device)
    for _ in range(length):
        total_trans @= trans
    distribution = (state @ total_trans).squeeze(0)
    expectation = 1. - ((distribution * torch.arange(0, length+1).to(device)).sum()/length)
    return expectation.to(device)


class pu_loss_auto():
    def __init__(self, prior, pu_type='', max_length=512, device='cpu'):
        self.prior = prior
        self.pu_type = pu_type
        self.device = device
        if pu_type in ['dual_softmax_dyn_dtrun']:
            self.loss_mod = PULossauto()
        else:
            raise NotImplementedError(f'PU type {pu_type} not implemented...')
        if pu_type in ['dual_softmax_dyn_dtrun']:
            expectations = []
            for i in range(0, max_length+1):
                expectations.append(_expectation_matrix(i, self.prior, device))
            self.prior = torch.stack(expectations)
            print('All dynamic priors calculated...')

    def __call__(self, input, label, sentence_length):
        prior = self.prior
        if 'dyn' in self.pu_type:
            prior = self.prior[sentence_length]
        return self.loss_mod.apply(input, label, prior)

    def logits_to_scores(self, logits):
        if self.pu_type in ['dual_softmax_dyn_dtrun']:
            return F.softmax(logits, dim=-1)[..., 0]  # 以 human(0类) 作为正类分数
        else:
            raise NotImplementedError(f'PU type {self.pu_type} not implemented')


# ============== 数据集封装（mgt_eval统一 -> 张量批） ==============
class _UnifiedMPUDS(Dataset):
    """
    输出：
    - text: str
    - label: int (0=human, 1=ai)  —— 与 mgt_eval 约定一致
    """
    def __init__(self, exs: List[Dict[str, Any]]):
        self.exs = exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, idx: int):
        e = self.exs[idx]
        return {"text": str(e["text"]), "label": int(e["label"])}


# ============== 配置 ==============
@dataclass
class TrainCfg:
    dataset_training: str
    dataset_validation: Optional[str] = None
    output_dir: str = "runs_mpu"

    base_model: str = "roberta-base"
    max_length: int = 512
    train_batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 3
    num_workers: int = 4

    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.06

    # PU 关键超参
    lamb: float = 0.4
    pu_type: str = "dual_softmax_dyn_dtrun"
    prior: float = 0.2
    len_thres: int = 55

    # 多尺度增强
    aug_min_length: int = 1
    aug_mode: Optional[str] = "sentence_deletion-0.25"

    seed: int = 114514
    # ---- 新增：控制训练/验证集样本数量（None 表示不限制）----
    train_sample_limit: Optional[int] = None
    val_sample_limit: Optional[int] = None

# ============== 训练主过程（进度条样式参考你给的代码） ==============
def _train_mpu(cfg: TrainCfg, **kwargs) -> Dict[str, Any]:
    _seed_everything(cfg.seed)
    device = _device()
    torch.set_grad_enabled(True)

    print(f"[mgt_eval] Using detector: {DETECTOR_NAME} (Type={detector_type})")
    print(f"[mgt_eval] Credits: {CITATION_AUTHORS} | Paper: {CITATION_TITLE} | Link: {CITATION_LINK}")
    print("[mgt_eval] Disclaimer: This implementation may differ slightly from the original reference; "
          "results might not exactly match those reported in the paper.")
    print(f"[mgt_eval] Device: {device}")

    out_root = f"{cfg.output_dir}_{_timestamp()}"
    os.makedirs(out_root, exist_ok=True)

    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }

    args_json_path = os.path.join(out_root, "train_args.json")
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": {**cfg.__dict__},
            "env": env_info,
            "data": _build_data_info(cfg.dataset_training, cfg.dataset_validation, None),
        }, f, ensure_ascii=False, indent=2)

    # ====== 加载数据（支持逗号分隔多个训练源） ======
    train_specs = [s.strip() for s in str(cfg.dataset_training).split(",") if s.strip()]
    if len(train_specs) == 0:
        raise ValueError("dataset_training 为空；请至少提供一个数据源路径或别名。")

    train_all: List[Dict[str, Any]] = []
    for spec in train_specs:
        exs = _load_one_dataset_spec(spec, seed=cfg.seed)
        if not isinstance(exs, list) or len(exs) == 0:
            print(f"[warn] no samples loaded from: {spec}")
        else:
            print(f"[data] loaded {len(exs)} samples from: {spec}")
            train_all.extend(exs)

    if cfg.dataset_validation:
        val_specs = [s.strip() for s in str(cfg.dataset_validation).split(",") if s.strip()]
        val_all: List[Dict[str, Any]] = []
        for vs in val_specs:
            vexs = _load_one_dataset_spec(vs, seed=cfg.seed)
            print(f"[data] loaded {len(vexs)} val samples from: {vs}")
            val_all.extend(vexs)
        tr = train_all
        va = val_all
    else:
        tr, va, _ = _stratified_split(train_all, 9.0, 1.0, 0.0, seed=cfg.seed)
        print(f"[data] split merged train set -> train={len(tr)}, val={len(va)}")
    # ---- 新增：根据 train_sample_limit / val_sample_limit 随机截断样本数 ----
    tr = _limit_examples(tr, cfg.train_sample_limit, seed=cfg.seed)
    va = _limit_examples(va, cfg.val_sample_limit, seed=cfg.seed)
    print(f"[data] final train={len(tr)}, val={len(va)} (after applying sample limits)")
    # ====== 构建数据集/分词器/DataLoader ======
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

    ds_tr = _UnifiedMPUDS(tr)
    ds_va = _UnifiedMPUDS(va)

    def _collate_train(examples):
        texts = [b["text"] for b in examples]
        # 训练时做多尺度增强（按需）
        if cfg.aug_mode and cfg.aug_min_length >= 1:
            texts = [_multi_scale_augment(t, cfg.aug_min_length, cfg.aug_mode) for t in texts]

        enc = tok.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
        )
        labels = torch.tensor([b["label"] for b in examples], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    def _collate_val(examples):
        texts = [b["text"] for b in examples]
        enc = tok.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
        )
        labels = torch.tensor([b["label"] for b in examples], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        collate_fn=_collate_train,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        collate_fn=_collate_val,
        drop_last=False,
    )

    # ====== 模型/优化器/调度器 ======
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2).to(device)
    if model.config.pad_token_id is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = max(1, cfg.epochs * len(dl_tr))
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ====== 产物目录 ======
    run_dir = os.path.join(out_root, "detector")
    os.makedirs(run_dir, exist_ok=True)
    run_best = os.path.join(run_dir, "best"); os.makedirs(run_best, exist_ok=True)
    run_last = os.path.join(run_dir, "last"); os.makedirs(run_last, exist_ok=True)

    # ====== 显存峰值统计上下文 ======
    mem_ctx = _reset_and_mark_cuda_peaks()

    # ====== 训练循环（表头/列宽/样式与参考一致） ======
    global_step, best_metric, best_epoch = 0, -1.0, -1
    step_indices, step_losses = [], []
    total_wall_start = time.perf_counter()

    # PU 模块（按给定实现）
    pu_module = pu_loss_auto(prior=cfg.prior, pu_type=cfg.pu_type, max_length=cfg.max_length, device=str(device))

    def _accuracy_sum(logits, labels):
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
        TP = (classification.bool() & labels.bool()).sum().item()
        FN = (~classification.bool() & labels.bool()).sum().item()
        TN = (~classification.bool() & ~labels.bool()).sum().item()
        FP = (classification.bool() & ~labels.bool()).sum().item()
        acc = (classification == labels).float().sum().item()
        return acc, TP, FN, TN, FP

    for ep in range(1, cfg.epochs + 1):
        model.train()
        avg_loss = 0.0

        # 打印表头（与给定样式对齐）
        print("\n" +
              f"{'Epoch':>{W_EPOCH}}{SEP}"
              f"{'GPU_mem':>{W_MEM}}{SEP}"
              f"{'L':>{W_NUM}}{SEP}"
              f"{'Lce':>{W_NUM}}{SEP}"
              f"{'Lpu':>{W_NUM}}{SEP}"
              f"{'avg':>{W_NUM}}{SEP}"
              f"{'lr':>{W_NUM}}{SEP}"
              f"{'step':>{W_STEP}}")

        pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), dynamic_ncols=True)

        for i, batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_ce = out["loss"]
            logits = out["logits"]

            # 句长（基于 padding 掩码）
            sent_len = attention_mask.sum(dim=-1)  # [B]
            # 伪标签：+1=human，-1=未标注，0=忽略（与给定实现一致）
            pseudo_labels = (~labels.bool()).float()  # human(0) -> True -> +1
            U_mask = (sent_len < cfg.len_thres) & (labels.bool())     # 短 ChatGPT 视为 U
            P_short_mask = (sent_len < cfg.len_thres) & (~labels.bool())  # 短 human 在 PU 中忽略
            pseudo_labels[U_mask] = -1
            pseudo_labels[P_short_mask] = 0

            # PU 分数（给定实现：softmax[...,0]）
            scores = pu_module.logits_to_scores(logits)
            loss_pu = pu_module(scores, pseudo_labels, sent_len)

            loss = loss_ce + cfg.lamb * loss_pu
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            step_indices.append(global_step)
            step_losses.append(float(loss.item()))
            avg_loss = (avg_loss * i + float(loss.item())) / (i + 1)

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            desc = (
                f"{f'{ep}/{cfg.epochs}':>{W_EPOCH}}{SEP}"
                f"{mem:>{W_MEM}}{SEP}"
                f"{float(loss.item()):>{W_NUM}.4f}{SEP}"
                f"{float(loss_ce.item()):>{W_NUM}.4f}{SEP}"
                f"{float(loss_pu.item()):>{W_NUM}.4f}{SEP}"
                f"{float(avg_loss):>{W_NUM}.4f}{SEP}"
                f"{float(optimizer.param_groups[0]['lr']):>{W_NUM}.2e}{SEP}"
                f"{int((ep - 1) * len(dl_tr) + i):>{W_STEP}d}"
            )
            if hasattr(pbar, "set_description"):
                pbar.set_description(desc)

        # ====== 验证 ======
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            print("\n" +
                  f"{'Epoch':>{W_EPOCH}}{SEP}"
                  f"{'GPU_mem':>{W_MEM}}{SEP}"
                  f"{'Cur_acc':>{W_NUM}}{SEP}"
                  f"{'avg_acc':>{W_NUM}}{SEP}"
                  f"{'loss':>{W_NUM}}")
            pbar_val = tqdm(enumerate(dl_va), total=len(dl_va), dynamic_ncols=True)

            right_num, tot_num = 0, 0
            test_loss = 0.0
            STATS = [0, 0, 0, 0]  # TP,FN,TN,FP

            for j, batch in pbar_val:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                vloss, logits = out["loss"], out["logits"]

                # CE 验证指标（与给定实现口径一致）
                acc_raw, TP, FN, TN, FP = _accuracy_sum(logits, labels)
                right_num += int(acc_raw)
                tot_num += int(labels.numel())
                STATS[0] += TP; STATS[1] += FN; STATS[2] += TN; STATS[3] += FP
                test_loss = (test_loss * j + float(vloss.item())) / (j + 1)

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                cur_acc = float(acc_raw) / max(1, int(labels.numel()))
                avg_acc = float(right_num) / max(1, tot_num)
                desc = (
                    f"{f'{ep}/{cfg.epochs}':>{W_EPOCH}}{SEP}"
                    f"{mem:>{W_MEM}}{SEP}"
                    f"{float(cur_acc):>{W_NUM}.4f}{SEP}"
                    f"{float(avg_acc):>{W_NUM}.4f}{SEP}"
                    f"{float(vloss.item()):>{W_NUM}.4f}"
                )
                if hasattr(pbar_val, "set_description"):
                    pbar_val.set_description(desc)

        # 计算 F1/Precision/Recall（基于 TP/FN/TN/FP，与给定实现一致）
        TP, FN, TN, FP = STATS
        try:
            accuracy = (TP + TN) / max(1, TP + TN + FN + FP)
            precision = TP / max(1, TP + FP)
            recall = TP / max(1, TP + FN)
            f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        except Exception:
            accuracy = precision = recall = f1 = 0.0

        # 保存 best/last
        cur_metric = f1  # 用 F1 选优
        if cur_metric >= best_metric:
            best_metric, best_epoch = cur_metric, ep
            # 保存完整 HF 权重（便于推理复现）
            model.save_pretrained(run_best)
            tok.save_pretrained(run_best)
        model.save_pretrained(run_last)
        tok.save_pretrained(run_last)

        print(f"[{DETECTOR_NAME}][Epoch {ep}] "
              f"train_loss={avg_loss:.4f} "
              f"val_acc={accuracy:.4f} val_prec={precision:.4f} val_rec={recall:.4f} val_f1={f1:.4f} "
              f"best_f1={best_metric:.4f}@{best_epoch}")

    # ====== 结束：显存峰值、loss 图、summary ======
    mem_stats = _collect_cuda_peaks(mem_ctx)
    loss_plot = _save_loss_plot(step_indices, step_losses, out_dir=out_root, filename="train_loss.png", smooth_window=0)
    total_wall_time = time.perf_counter() - total_wall_start

    summary = {
        "best_dir": run_best,
        "last_dir": run_last,
        "best_val_f1": best_metric,
        "history": [],
        "memory": mem_stats,
        "timing": {"total_wall_time_sec": total_wall_time},
        "artifacts": {
            "args_json": args_json_path,
            "summary_json": os.path.join(out_root, "train_summary.json"),
            "loss_plot": loss_plot,
        },
    }
    with open(summary["artifacts"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "train": {
            "model_dir": run_best,
            "best_val_f1": summary.get("best_val_f1", None),
            "artifacts": summary["artifacts"],
            "output_root": out_root,
        }
    }


# ============== 统一暴露的注册入口（与 detective.py 风格一致） ==============
@register_train("mpu")
def train_mpu(**kwargs) -> Dict[str, Any]:
    cfg = TrainCfg(
        dataset_training=kwargs.get("dataset_training"),
        dataset_validation=kwargs.get("dataset_validation", None),
        output_dir=kwargs.get("output_dir", "runs_mpu"),

        base_model=kwargs.get("base_model", kwargs.get("model_name", "roberta-base")),
        max_length=kwargs.get("max_length", 512),
        train_batch_size=kwargs.get("train_batch_size", kwargs.get("batch_size", 32)),
        eval_batch_size=kwargs.get("eval_batch_size", 64),
        epochs=kwargs.get("epochs", kwargs.get("total_epoch", 3)),
        num_workers=kwargs.get("num_workers", 4),

        lr=kwargs.get("lr", kwargs.get("learning_rate", 5e-5)),
        weight_decay=kwargs.get("weight_decay", 0.0),
        warmup_ratio=kwargs.get("warmup_ratio", 0.06),

        lamb=kwargs.get("lamb", 0.4),
        pu_type=kwargs.get("pu_type", "dual_softmax_dyn_dtrun"),
        prior=kwargs.get("prior", 0.2),
        len_thres=kwargs.get("len_thres", 55),

        aug_min_length=kwargs.get("aug_min_length", 1),
        aug_mode=kwargs.get("aug_mode", "sentence_deletion-0.25"),

        seed=kwargs.get("seed", 114514),
        # ---- 新增：从 kwargs 读取样本数限制 ----
        train_sample_limit=kwargs.get("train_sample_limit", None),
        val_sample_limit=kwargs.get("val_sample_limit", None),
    )
    assert cfg.dataset_training, "mpu 需要 dataset_training 参数（可被 load_dataset_unified 解析；支持逗号分隔多源）"
    return _train_mpu(cfg, **kwargs)


# 便于脚本快速调用
def MPU(**kwargs) -> Dict[str, Any]:
    return _train_mpu(TrainCfg(**kwargs))
