from __future__ import annotations
import os, random, math, json, time, platform
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig, AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)

from mgt_eval.data_utils.load import load_dataset_unified
from mgt_eval.train.train import train_model
from mgt_eval.train.registry import register_train

# ======= 复用通用训练器里的工具，便于一致的元数据输出 =======
from mgt_eval.train.train import (
    _reset_and_mark_cuda_peaks, _collect_cuda_peaks, _save_loss_plot, _build_data_info
)

# ---- 进度条样式常量（统一列宽） ----
W_EPOCH = 8
W_MEM   = 8
W_NUM   = 8
W_STEP  = 8
SEP     = " "

# ---- 日志器：优先 loguru，回退到标准 logging ----
try:
    from loguru import logger
    _USE_LOGURU = True
except Exception:  # pragma: no cover - 回退分支
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | GREATER | %(message)s",
    )
    logger = logging.getLogger("GREATER")
    _USE_LOGURU = False

# ==== 方法元信息（用于日志） ====
DETECTOR_NAME   = "GREATER"
detector_type   = "Model-based"
CITATION_AUTHORS = "Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu"
CITATION_TITLE   = "Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training"
CITATION_LINK    = "https://arxiv.org/abs/2502.12734"

# ---------------- utils ----------------
def _seed_everything(seed: int = 114514):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _is_local_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

def _resolve_base(model: Optional[str]) -> str:
    fallback = "xlm-roberta-base"
    if not model:
        return fallback
    spec = model.strip()
    if _is_local_hf_dir(spec):
        return spec
    alias = {
        "mbert": "bert-base-multilingual-cased",
        "mdebert": "microsoft/mdeberta-v3-base",
        "mdeberta": "microsoft/mdeberta-v3-base",
        "debert": "microsoft/deberta-v3-base",
        "deberta": "microsoft/deberta-v3-base",
        "albert-base": "albert-base-v2",
        "albert-large": "albert-large-v2",
    }
    if spec.lower() in alias:
        return alias[spec.lower()]
    try:
        AutoConfig.from_pretrained(spec)
        return spec
    except Exception:
        return fallback

def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

@dataclass
class TrainCfg:
    base_model: str
    output_dir: str
    max_length: int = 512
    lr: float = 5e-5
    weight_decay: float = 0.0
    epochs: int = 3
    train_batch_size: int = 32
    eval_batch_size: int = 64
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    fp16: bool = True
    label_smoothing: float = 0.0
    seed: int = 114514
    device: Optional[str] = None
    name: Optional[str] = None

class TLDS(Dataset):
    def __init__(self, exs: List[Dict[str, Any]]):
        self.exs = exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, idx: int):
        e = self.exs[idx]
        return {"text": e["text"], "label": int(e["label"])}

def _stratified_split(
    examples: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 114514,
):
    if train_ratio <= 0 and val_ratio <= 0 and test_ratio <= 0:
        train_ratio, val_ratio, test_ratio = 1.0, 0.0, 0.0
    pos = [e for e in examples if int(e["label"]) == 1]
    neg = [e for e in examples if int(e["label"]) == 0]

    def _split(lst):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        tot = len(idx)
        S = train_ratio + val_ratio + test_ratio
        n_tr = int(round(tot * (train_ratio / S))) if S > 0 else tot
        n_va = int(round(tot * (val_ratio / S))) if S > 0 else 0
        n_tr = min(n_tr, tot)
        n_va = min(n_va, tot - n_tr)
        n_te = tot - n_tr - n_va
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    p_tr, p_va, p_te = _split(pos)
    n_tr, n_va, n_te = _split(neg)

    train = [pos[i] for i in p_tr] + [neg[i] for i in n_tr]
    val   = [pos[i] for i in p_va] + [neg[i] for i in n_va]
    test  = [pos[i] for i in p_te] + [neg[i] for i in n_te]

    rng = np.random.RandomState(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

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
    tok.padding_side = "right"
    pad_id = tok.pad_token_id
    cfg.pad_token_id = pad_id
    mdl = AutoModelForSequenceClassification.from_pretrained(
        base, config=cfg, trust_remote_code=True
    )
    if getattr(mdl.get_input_embeddings(), "num_embeddings", len(tok)) < len(tok):
        mdl.resize_token_embeddings(len(tok))
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = pad_id
    mdl.config.id2label = {0: "human", 1: "ai"}
    mdl.config.label2id = {"human": 0, "ai": 1}
    return mdl, tok

# ---------- 文本级候选（避免跨 tokenizer 复用 id） ----------
@torch.no_grad()
def _mlm_topk_candidates_on_text(
    mlm,
    mlm_tok,
    text: str,
    mlm_token_index: int,
    topk: int = 20,
    max_length: int = 512,
    device=None,
) -> List[int]:
    enc = mlm_tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    if device is None:
        device = next(mlm.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    logits = mlm(input_ids=input_ids, attention_mask=attn).logits  # [1, L, V]
    pos = int(mlm_token_index)
    if not (0 <= pos < logits.size(1)):
        return []
    probs = torch.softmax(logits[0, pos], dim=-1)
    cand = torch.topk(probs, k=min(topk, probs.numel())).indices.tolist()

    def _is_readable_piece_tok(tok_str: str) -> bool:
        t = tok_str.lstrip("Ġ▁").strip()
        if not t:
            return False
        if all(ch in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" for ch in t):
            return False
        return any(ch.isalnum() for ch in t)

    out = []
    for cid in cand:
        ts = mlm_tok.convert_ids_to_tokens([cid])[0]
        if _is_readable_piece_tok(ts):
            out.append(cid)
    return out

def _surrogate_token_importance(
    surr, tok, text: str, max_length: int, device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    surr.eval()
    enc = tok(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=False
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    emb = surr.get_input_embeddings()(input_ids)
    emb.requires_grad_(True)
    out = surr(inputs_embeds=emb, attention_mask=attn)
    logits = out.logits
    pred = torch.argmax(logits, dim=-1)
    loss = F.cross_entropy(logits, pred)
    surr.zero_grad(set_to_none=True)
    loss.backward()
    grad = emb.grad
    scores = (grad * grad).sum(dim=-1).sqrt().squeeze(0)
    return scores.detach(), {"input_ids": input_ids, "attn": attn}

def _generate_adversarial_text(
    text: str,
    surr,
    surr_tok,
    mlm,
    mlm_tok,
    max_length: int,
    k_percent: float,
    per_pos_topk: int,
    max_replacements: Optional[int],
    device,
) -> str:
    @torch.no_grad()
    def _surr_prob(txt: str, target_cls: int) -> float:
        enc = surr_tok(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = surr(**enc).logits
        return torch.softmax(logits, dim=-1)[0, target_cls].item()

    scores, pack = _surrogate_token_importance(surr, surr_tok, text, max_length, device)
    surr_enc = surr_tok(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    offsets = (
        surr_enc["offset_mapping"][0]
        if isinstance(surr_enc["offset_mapping"], list)
        else surr_enc["offset_mapping"]
    )

    input_ids = pack["input_ids"]
    seq_len = input_ids.size(1)
    specials = {
        surr_tok.cls_token_id,
        surr_tok.sep_token_id,
        surr_tok.pad_token_id,
        getattr(surr_tok, "bos_token_id", None),
        getattr(surr_tok, "eos_token_id", None),
    }
    specials = {x for x in specials if x is not None}

    k = max(1, int(math.ceil(k_percent * seq_len)))
    cand_pos = [
        i
        for i in torch.topk(scores, k=min(k, seq_len)).indices.tolist()
        if (0 < i < seq_len - 1) and int(input_ids[0, i]) not in specials
    ]
    if max_replacements is not None:
        cand_pos = cand_pos[:max_replacements]

    with torch.no_grad():
        base_logits = surr(
            input_ids=input_ids.to(device), attention_mask=pack["attn"].to(device)
        ).logits
        base_pred = int(torch.argmax(torch.softmax(base_logits, dim=-1)[0]).item())
    target_cls = 1 - base_pred

    best_text = text

    for pos in cand_pos:
        # 依据 surrogate offsets 拿字符跨度
        surr_enc = surr_tok(
            best_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        off = (
            surr_enc["offset_mapping"][0]
            if isinstance(surr_enc["offset_mapping"], list)
            else surr_enc["offset_mapping"]
        )
        if not (0 <= pos < len(off)):
            continue
        start, end = off[pos]
        if end <= start:
            continue
        center = (start + end) // 2

        # 在 MLM 的 tokenizer 空间定位对应 token
        mlm_enc = mlm_tok(
            best_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        moff = (
            mlm_enc["offset_mapping"][0]
            if isinstance(mlm_enc["offset_mapping"], list)
            else mlm_enc["offset_mapping"]
        )
        mlm_idx = None
        for i, (a, b) in enumerate(moff):
            if a <= center < b:
                mlm_idx = i
                break
        if mlm_idx is None:
            for i, (a, b) in enumerate(moff):
                if b > a:
                    mlm_idx = i
                    break
        if mlm_idx is None:
            continue

        cand_ids = _mlm_topk_candidates_on_text(
            mlm,
            mlm_tok,
            best_text,
            mlm_idx,
            topk=per_pos_topk,
            max_length=max_length,
            device=device,
        )
        if not cand_ids:
            continue

        best_score_here, best_trial_text = -1e9, best_text
        for cid in cand_ids:
            piece = mlm_tok.convert_ids_to_tokens([cid])[0]
            repl = mlm_tok.convert_tokens_to_string([piece])
            trial_text = best_text[:start] + repl + best_text[end:]
            prob = _surr_prob(trial_text, target_cls)
            if prob > best_score_here:
                best_score_here, best_trial_text = prob, trial_text

        best_text = best_trial_text

    return best_text if best_text else text

# ---------- 评测 ----------
def _build_dataloader(examples, tok, max_length, batch_size, shuffle):
    class _DS(Dataset):
        def __init__(self, exs):
            self.exs = exs

        def __len__(self):
            return len(self.exs)

        def __getitem__(self, idx):
            e = self.exs[idx]
            enc = tok(
                e["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(e["label"]), dtype=torch.long),
            }

    return DataLoader(
        _DS(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
    )

@torch.no_grad()
def _evaluate(
    mdl,
    tok,
    examples,
    max_length,
    batch_size,
    device,
    *,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
):
    """
    带进度条的评测循环：
      - 每步打印当前 loss、平均 loss、当前 batch 正确数、平均 acc 等。
      - 进度条样式与你给的示例保持一致。
    """
    mdl.eval()
    dl = _build_dataloader(examples, tok, max_length, batch_size, False)
    tot, cor = 0, 0
    loss_sum = 0.0
    n_batches = 0

    # 表头
    logger.info(
        "\n"
        + f"{'Epoch':>{W_EPOCH}}{SEP}"
        f"{'GPU_mem':>{W_MEM}}{SEP}"
        f"{'L':>{W_NUM}}{SEP}"
        f"{'avg':>{W_NUM}}{SEP}"
        f"{'acc':>{W_NUM}}{SEP}"
        f"{'acc_avg':>{W_NUM}}{SEP}"
        f"{'loss_avg':>{W_NUM}}"
    )

    pbar = tqdm(dl, dynamic_ncols=True, leave=True)

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = mdl(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        loss = F.cross_entropy(logits, batch["labels"])

        preds = logits.argmax(dim=-1)
        correct_batch = int((preds == batch["labels"]).sum().item())
        batch_total = int(batch["labels"].size(0))

        cor += correct_batch
        tot += batch_total
        loss_sum += float(loss.item())
        n_batches += 1

        acc_now = correct_batch
        acc_avg = cor / max(1, tot)
        avg_loss = loss_sum / max(1, n_batches)
        l_now = float(loss.item())

        if torch.cuda.is_available():
            mem_txt = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
        else:
            mem_txt = "0G"

        ep_txt = (
            f"{epoch}/{epochs}"
            if (epoch is not None and epochs is not None)
            else "val/test"
        )

        desc = (
            f"{ep_txt:>{W_EPOCH}}{SEP}"
            f"{mem_txt:>{W_MEM}}{SEP}"
            f"{l_now:>{W_NUM}.4f}{SEP}"
            f"{avg_loss:>{W_NUM}.4f}{SEP}"
            f"{acc_now:>{W_NUM}d}{SEP}"
            f"{acc_avg:>{W_NUM}.4f}{SEP}"
            f"{avg_loss:>{W_NUM}.4f}"
        )
        pbar.set_description(desc)

    if hasattr(pbar, "close"):
        pbar.close()

    acc = cor / max(1, tot)
    avg_loss = loss_sum / max(1, n_batches)
    return {"acc": acc, "loss": avg_loss}

# ---------- 对抗训练（含进度条 & 元数据输出） ----------
def _adv_train_detector(
    *,
    detector_base: str,
    train_examples: List[Dict[str, Any]],
    val_examples: Optional[List[Dict[str, Any]]],
    output_dir: str,  # 顶层（已含时间戳）
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.0,
    grad_accum_steps: int = 1,
    fp16: bool = True,
    label_smoothing: float = 0.01,
    seed: int = 114514,
    k_percent: float = 0.12,
    per_pos_topk: int = 20,
    max_replacements: Optional[int] = None,
    adv_prob_ai: float = 1.0,
    adv_prob_human: float = 0.0,
    surrogate_dir: str = "",
    mlm_model: Optional[str] = None,
    dataset_spec: Optional[str] = None,
):
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    det_base = _resolve_base(detector_base)
    detector, det_tok = _prepare_seqcls(det_base, 2)
    detector.to(device)

    # 允许外部传入 mlm 模型（路径或 HF hub id）
    mlm_name = _resolve_base(mlm_model) if mlm_model else det_base
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device).eval()
    mlm_tok = AutoTokenizer.from_pretrained(mlm_name, use_fast=True)
    for p in mlm.parameters():
        p.requires_grad = False

    if not surrogate_dir or not _is_local_hf_dir(surrogate_dir):
        raise ValueError(f"Invalid surrogate_dir: {surrogate_dir}")
    surr = AutoModelForSequenceClassification.from_pretrained(surrogate_dir).to(device).eval()
    surr_tok = AutoTokenizer.from_pretrained(surrogate_dir, use_fast=True)
    for p in surr.parameters():
        p.requires_grad = False

    # 优化器 & 调度器
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {
            "params": [
                p
                for n, p in detector.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in detector.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(grouped, lr=lr)
    updates_per_epoch = math.ceil(
        len(train_examples) / max(1, batch_size) / max(1, grad_accum_steps)
    )
    total_steps = epochs * max(1, updates_per_epoch)
    sched = get_linear_schedule_with_warmup(
        optim, int(warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and device.type == "cuda"))

    # 追踪 & 输出目录
    run_dir = f"{output_dir}_detector_{_timestamp()}"
    os.makedirs(run_dir, exist_ok=True)

    # ====== 元数据：训练入参、环境、数据摘要 ======
    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
    args_json_path = os.path.join(run_dir, "train_args.json")
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {
                    "detector_base": detector_base,
                    "mlm_model": mlm_model,
                    "max_length": max_length,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "warmup_ratio": warmup_ratio,
                    "weight_decay": weight_decay,
                    "grad_accum_steps": grad_accum_steps,
                    "fp16": fp16,
                    "label_smoothing": label_smoothing,
                    "seed": seed,
                    "k_percent": k_percent,
                    "per_pos_topk": per_pos_topk,
                    "max_replacements": max_replacements,
                    "adv_prob_ai": adv_prob_ai,
                    "adv_prob_human": adv_prob_human,
                    "surrogate_dir": surrogate_dir,
                },
                "env": env_info,
                "data": _build_data_info(
                    dataset_spec,
                    TLDS(train_examples),
                    TLDS(val_examples) if val_examples else None,
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 显存峰值
    mem_ctx = _reset_and_mark_cuda_peaks()

    def _iter_batches(exs, bs):
        idx = np.arange(len(exs))
        np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            yield [exs[i] for i in idx[s : s + bs]]

    best_val, best_dir = -1.0, None
    last_dir = None

    history = []
    step_indices, step_losses = [], []
    global_update = 0

    total_wall_start = time.perf_counter()
    per_epoch_train_times, per_epoch_wall_times = [], []

    # 训练
    for ep in range(1, epochs + 1):
        ep_wall_start = time.perf_counter()
        detector.train()
        running_loss, n_batches = 0.0, 0
        correct_tot, total_tot = 0, 0
        last_l_now, last_correct_batch, last_batch_total = 0.0, 0, 0

        # 表头（训练）
        logger.info(
            "\n"
            + f"{'Epoch':>{W_EPOCH}}{SEP}"
            f"{'GPU_mem':>{W_MEM}}{SEP}"
            f"{'L':>{W_NUM}}{SEP}"
            f"{'avg':>{W_NUM}}{SEP}"
            f"{'acc':>{W_NUM}}{SEP}"
            f"{'acc_avg':>{W_NUM}}{SEP}"
            f"{'loss_avg':>{W_NUM}}"
        )

        pbar = tqdm(total=updates_per_epoch, dynamic_ncols=True, leave=True)

        batch_accum = 0
        for step, mb in enumerate(_iter_batches(train_examples, batch_size), 1):
            texts, labels = [], []
            for e in mb:
                txt, y = e["text"], int(e["label"])
                p = adv_prob_ai if y == 1 else adv_prob_human
                if random.random() < p:
                    try:
                        adv_txt = _generate_adversarial_text(
                            text=txt,
                            surr=surr,
                            surr_tok=surr_tok,
                            mlm=mlm,
                            mlm_tok=mlm_tok,
                            max_length=max_length,
                            k_percent=k_percent,
                            per_pos_topk=per_pos_topk,
                            max_replacements=max_replacements,
                            device=device,
                        )
                    except Exception:
                        adv_txt = txt
                    texts.extend([txt, adv_txt])
                    labels.extend([y, y])
                else:
                    texts.append(txt)
                    labels.append(y)

            enc = det_tok(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in enc.items()}
            y = torch.tensor(labels, dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                logits = detector(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                if label_smoothing > 0:
                    logp = torch.log_softmax(logits, dim=-1)
                    nll = -logp.gather(-1, y.unsqueeze(1)).squeeze(1)
                    smooth = -logp.mean(dim=-1)
                    raw_loss = ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()
                else:
                    raw_loss = torch.nn.functional.cross_entropy(logits, y)
                loss = raw_loss / max(1, grad_accum_steps)

            # 统计 train acc / loss（以未除 grad_accum 的 raw_loss 为准）
            l_now = float(raw_loss.item())
            running_loss += l_now
            n_batches += 1

            preds = logits.argmax(dim=-1)
            correct_batch = int((preds == y).sum().item())
            batch_total = int(y.size(0))
            correct_tot += correct_batch
            total_tot += batch_total

            last_l_now = l_now
            last_correct_batch = correct_batch
            last_batch_total = batch_total

            scaler.scale(loss).backward()
            batch_accum += 1

            if batch_accum % grad_accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_update += 1
                step_indices.append(global_update)
                step_losses.append(float(loss.item()))
                batch_accum = 0

                # 更新训练进度条描述
                avg_loss = running_loss / max(1, n_batches)
                acc_now = last_correct_batch
                acc_avg = correct_tot / max(1, total_tot)

                if torch.cuda.is_available():
                    mem_txt = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
                else:
                    mem_txt = "0G"

                desc = (
                    f"{f'{ep}/{epochs}':>{W_EPOCH}}{SEP}"
                    f"{mem_txt:>{W_MEM}}{SEP}"
                    f"{last_l_now:>{W_NUM}.4f}{SEP}"
                    f"{avg_loss:>{W_NUM}.4f}{SEP}"
                    f"{acc_now:>{W_NUM}d}{SEP}"
                    f"{acc_avg:>{W_NUM}.4f}{SEP}"
                    f"{avg_loss:>{W_NUM}.4f}"
                )
                pbar.set_description(desc)
                pbar.update(1)

        if hasattr(pbar, "close"):
            pbar.close()

        train_time = time.perf_counter() - ep_wall_start
        per_epoch_train_times.append(train_time)
        avg_train_loss = running_loss / max(1, n_batches)

        # 验证（可选）：greater 的对抗阶段保留验证
        if val_examples:
            metrics = _evaluate(
                detector,
                det_tok,
                val_examples,
                max_length,
                batch_size,
                device,
                epoch=ep,
                epochs=epochs,
            )
            val_acc = metrics["acc"]
        else:
            val_acc = None

        logger.info(
            f"[GREATER][Epoch {ep}] train_loss={avg_train_loss:.4f}  "
            f"{'' if val_acc is None else f'val_acc={val_acc:.4f}  '}"
            f"train_time={train_time:.1f}s"
        )

        # 保存 last
        run_last = os.path.join(run_dir, "last")
        os.makedirs(run_last, exist_ok=True)
        detector.save_pretrained(run_last)
        det_tok.save_pretrained(run_last)
        last_dir = run_last

        # 记录最好
        cur_metric = val_acc if (val_acc is not None) else -avg_train_loss
        if cur_metric is not None and cur_metric >= best_val:
            best_val = cur_metric
            run_best = os.path.join(run_dir, "best")
            os.makedirs(run_best, exist_ok=True)
            detector.save_pretrained(run_best)
            det_tok.save_pretrained(run_best)
            best_dir = run_best

        epoch_wall_time = time.perf_counter() - ep_wall_start
        per_epoch_wall_times.append(epoch_wall_time)

        history.append(
            {
                "epoch": ep,
                "avg_train_loss": avg_train_loss,
                "train_time_sec": train_time,
                "epoch_wall_time_sec": epoch_wall_time,
                "val_acc": val_acc,
                "global_update": global_update,
            }
        )

    # 显存&墙钟
    mem_stats = _collect_cuda_peaks(mem_ctx)
    total_wall_time = time.perf_counter() - total_wall_start

    # loss 曲线
    loss_plot_path = _save_loss_plot(
        steps=step_indices,
        losses=step_losses,
        out_dir=run_dir,
        filename="train_loss.png",
        smooth_window=0,
    )

    summary = {
        "best_dir": best_dir,
        "last_dir": last_dir or run_dir,
        "best_val_acc": (None if best_val < 0 else best_val),
        "history": history,
        "memory": mem_stats,
        "timing": {
            "total_wall_time_sec": total_wall_time,
            "per_epoch_train_time_sec": per_epoch_train_times,
            "per_epoch_wall_time_sec": per_epoch_wall_times,
            "avg_epoch_train_time_sec": (sum(per_epoch_train_times) / len(per_epoch_train_times))
            if per_epoch_train_times
            else 0.0,
            "avg_epoch_wall_time_sec": (sum(per_epoch_wall_times) / len(per_epoch_wall_times))
            if per_epoch_wall_times
            else 0.0,
        },
        "artifacts": {
            "args_json": args_json_path,
            "summary_json": os.path.join(run_dir, "train_summary.json"),
            "loss_plot": loss_plot_path,
        },
    }
    with open(summary["artifacts"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "best_dir": best_dir,
        "last_dir": last_dir or run_dir,
        "best_val_acc": (None if best_val < 0 else best_val),
        "artifacts": summary["artifacts"],
        "timing": summary["timing"],
    }

# ---------- 外层：先训 surrogate（无验证），再对抗训练 detector ----------
@register_train("greater")
def train_greater(**kwargs) -> Dict[str, Any]:
    """
    必要入参：
      - dataset_surrogate: 用于替代模型训练
      - dataset_training:  用于对抗训练检测器

    关键可选入参：
      - training_sample_k:     检测器训练样本数（默认 10000）
      - validation_sample_k:   检测器验证样本数（默认 None，表示按 8:1 比例自动划分）
      - surrogate_sample_k:    surrogate 训练的采样数（默认 None=全量；无验证）
      - surrogate_base_model:  surrogate 基座模型（默认 xlm-roberta-base）
      - detector_base_model:   检测器基座模型（默认 xlm-roberta-base）
      - mlm_model:             用于对抗替换的 MLM（路径或 HF id）
      - output_dir:            顶层输出目录（会自动追加时间戳）
    """
    dataset_surrogate = kwargs.get("dataset_surrogate", None)
    dataset_training = kwargs.get("dataset_training", None)
    assert dataset_surrogate and dataset_training, "greater 需要 dataset_surrogate 与 dataset_training"

    # 可控超参
    surrogate_base_model = kwargs.get("surrogate_base_model", "xlm-roberta-base")
    surrogate_sample_k = kwargs.get("surrogate_sample_k", None)
    surrogate_epochs = kwargs.get("surrogate_epochs", 3)
    surrogate_bs = kwargs.get("surrogate_bs", 32)
    surrogate_lr = kwargs.get("surrogate_lr", 5e-5)

    detector_base_model = kwargs.get("detector_base_model", "xlm-roberta-base")
    output_dir_raw = kwargs.get("output_dir", "runs_greater")
    output_dir = f"{output_dir_raw}_{_timestamp()}"  # 顶层追加时间戳
    max_length = kwargs.get("max_length", 512)
    epochs = kwargs.get("epochs", 6)
    train_batch_size = kwargs.get("train_batch_size", 16)
    eval_batch_size = kwargs.get("eval_batch_size", 64)
    warmup_ratio = kwargs.get("warmup_ratio", 0.06)
    weight_decay = kwargs.get("weight_decay", 0.0)
    grad_accum_steps = kwargs.get("grad_accum_steps", 1)
    fp16 = kwargs.get("fp16", True)
    label_smoothing = kwargs.get("label_smoothing", 0.01)
    seed = kwargs.get("seed", 114514)

    # adversary knobs
    k_percent = kwargs.get("k_percent", 0.12)
    per_pos_topk = kwargs.get("per_pos_topk", 20)
    max_replacements = kwargs.get("max_replacements", None)
    adv_prob_ai = kwargs.get("adv_prob_ai", 1.0)
    adv_prob_human = kwargs.get("adv_prob_human", 0.2)

    mlm_model = kwargs.get("mlm_model", None)
    training_sample_k = kwargs.get("training_sample_k", 10000)   # 训练集条数（默认 10000）
    validation_sample_k = kwargs.get("validation_sample_k", None)  # 验证集条数（默认 None = 按 8:1 自动划分）

    _seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ===== 方法/论文/免责声明提示（日志） =====
    method_name = globals().get("DETECTOR_NAME", "GREATER")
    method_type = globals().get("detector_type", "Unknown")
    method_auth = globals().get("CITATION_AUTHORS", None)
    method_link = globals().get("CITATION_LINK", None)
    method_title = globals().get("CITATION_TITLE", None)
    device_hint = str(device)

    logger.info(f"[mgt_eval] Using detector: {method_name} (type={method_type})")
    if method_auth or method_link or method_title:
        logger.info(
            f"[mgt_eval] Credits: {method_auth or 'Unknown authors'} | "
            f"Paper: {method_title or 'N/A'} | Link: {method_link or 'N/A'}"
        )
    logger.info(
        "[mgt_eval] Disclaimer: This implementation may differ slightly from the original reference; "
        "results might not exactly match those reported in the paper."
    )
    logger.info(f"[mgt_eval] Device: {device_hint}")

    # 1) surrogate 训练（无验证）
    surr_exs, _ = load_dataset_unified(
        dataset=dataset_surrogate,
        sample_k=surrogate_sample_k,
        sample_seed=seed,
        group_cols=None,
    )
    s_tr = surr_exs
    surr_base = _resolve_base(surrogate_base_model)
    s_mdl, s_tok = _prepare_seqcls(surr_base, 2)
    s_out_base = os.path.join(output_dir, "surrogate")
    os.makedirs(s_out_base, exist_ok=True)
    s_cfg = TrainCfg(
        base_model=surr_base,
        output_dir=s_out_base,
        max_length=max_length,
        lr=surrogate_lr,
        weight_decay=0.0,
        epochs=surrogate_epochs,
        train_batch_size=surrogate_bs,
        eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        grad_accum_steps=1,
        fp16=fp16,
        label_smoothing=0.0,
        seed=seed,
        device=str(device),
        name="surrogate",
    )
    logger.info("Start training surrogate model...")
    s_result = train_model(
        s_mdl, s_tok, TLDS(s_tr), None, s_cfg, dataset_spec=dataset_surrogate
    )
    surrogate_dir = (
        s_result.get("best_dir") or s_result.get("last_dir") or s_cfg.output_dir
    )

    logger.info("Start training target detector...")

    # 2) detector 对抗训练（训练/验证条数可控）
    if (validation_sample_k is not None) and (training_sample_k is not None):
        total_k = int(training_sample_k) + int(validation_sample_k)
        tr_exs, _ = load_dataset_unified(
            dataset=dataset_training,
            sample_k=total_k,
            sample_seed=seed,
            group_cols=None,
        )
        # 使用“样本数”作为比值，_stratified_split 会按比例切分出接近指定条数的 train/val
        t_tr, t_va, _ = _stratified_split(
            tr_exs,
            train_ratio=float(training_sample_k),
            val_ratio=float(validation_sample_k),
            test_ratio=0.0,
            seed=seed,
        )
    else:
        tr_exs, _ = load_dataset_unified(
            dataset=dataset_training,
            sample_k=training_sample_k,
            sample_seed=seed,
            group_cols=None,
        )
        # 默认 8:1 划分（兼容原行为）
        t_tr, t_va, _ = _stratified_split(tr_exs, 8.0, 1.0, 0.0, seed=seed)

    det_out_base = os.path.join(output_dir, "detector")

    adv_res = _adv_train_detector(
        detector_base=detector_base_model,
        train_examples=t_tr,
        val_examples=t_va,
        output_dir=det_out_base,  # 内部会再追加 detector_{ts}
        max_length=max_length,
        epochs=epochs,
        batch_size=train_batch_size,
        lr=5e-5,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        fp16=fp16,
        label_smoothing=label_smoothing,
        seed=seed,
        k_percent=k_percent,
        per_pos_topk=per_pos_topk,
        max_replacements=max_replacements,
        adv_prob_ai=adv_prob_ai,
        adv_prob_human=adv_prob_human,
        surrogate_dir=surrogate_dir,
        mlm_model=mlm_model,
        dataset_spec=dataset_training,
    )

    model_dir = adv_res.get("best_dir") or adv_res.get("last_dir")
    return {
        "train": {
            "surrogate_dir": surrogate_dir,
            "model_dir": model_dir,
            "best_val_acc": adv_res.get("best_val_acc", None),
            "artifacts": adv_res.get("artifacts", {}),
            "output_root": output_dir,
        }
    }

# 便于脚本快速调用
def GREATER(**kwargs) -> Dict[str, Any]:
    return train_greater(**kwargs)
