# mgt_eval/cli.py
import argparse, json, os, time
from pathlib import Path

from .data_utils.load import load_dataset_unified
from mgt_eval.eval.evaluator import evaluate_detector as _eval
from mgt_eval.detectors.registry import get_detector_cls
from mgt_eval.calibration.runner import Calibrate as _calibrate, _build_detector

from typing import Any, Dict, Optional, List
import torch
from types import SimpleNamespace
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 复用你给的通用训练器
from mgt_eval.train.train import train_model as _train_seqcls_model

# =========================
# Built-in SeqCls backbones (ENUMERATED)
# =========================
_SEQCLS_BACKBONES = [
    # RoBERTa
    "roberta-base",
    "roberta-large",
    "distilroberta-base",

    # XLM-R
    "xlm-roberta-base",
    "xlm-roberta-large",

    # BERT family
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",

    # DeBERTa (可选，若你环境常用）
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",

    # ELECTRA（可选）
    "google/electra-base-discriminator",
]
_SEQCLS_BACKBONES_L = {x.lower(): x for x in _SEQCLS_BACKBONES}

from transformers import AutoConfig, AutoModel
import torch.nn as nn

# 可选：常用简写别名（更顺手）
_HF_ALIASES = {
    # DeBERTa v3
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    # DeBERTa v2
    "deberta-v2-xlarge": "microsoft/deberta-v2-xlarge",
    "deberta-v2-xxlarge": "microsoft/deberta-v2-xxlarge",
    # ELECTRA
    "electra-base-discriminator": "google/electra-base-discriminator",
    "electra-large-discriminator": "google/electra-large-discriminator",
}

def _parse_attack_dataset_single(v, *, data_fallback: Optional[str] = None) -> Optional[List[str]]:
    """
    支持三种写法（最终都只得到 0/1 个 attack dataset）：
      1) 不传 --attack -> None
      2) 传 --attack (无参数) -> 使用 --data 作为 attack dataset（paired-record 攻击数据集）
      3) 传 --attack path.jsonl -> 使用该 path
    同时兼容旧写法：如果 argparse 产生 list（比如历史脚本用 action=append），也会检测并强制只留一个。
    """
    if v is None:
        return None

    # 兼容：如果 v 是 list（旧脚本可能重复传），强制只能有 1 个有效项
    if isinstance(v, list):
        items = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            items.append(s)
        if not items:
            return None
        if len(items) > 1:
            raise SystemExit(f"[MGTEval][detect] --attack only supports ONE dataset, got: {items}")
        v = items[0]

    s = str(v).strip()
    # 写法 2：--attack（无参数） -> const="__SELF__"
    if s == "__SELF__":
        if not data_fallback:
            raise SystemExit("[MGTEval][detect] --attack used without a path, but --data is empty.")
        return [str(data_fallback)]

    # 写法 3：--attack a.jsonl,b.jsonl 不允许（强制一个）
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 1:
        raise SystemExit(f"[MGTEval][detect] --attack only supports ONE dataset (no commas). Got: {s}")
    return parts

def _resolve_hf_id_or_path(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # 本地目录（HF save_pretrained 目录）
    if os.path.isdir(s) and os.path.exists(os.path.join(s, "config.json")):
        return s
    # 简写别名
    return _HF_ALIASES.get(s.lower(), s)

def _infer_hidden_size(cfg) -> int:
    for k in ("hidden_size", "n_embd", "d_model", "dim"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise ValueError(f"Cannot infer hidden size from config: {type(cfg)}")

class _HFBackboneClsWrapper(nn.Module):
    """当 AutoModelForSequenceClassification 加载失败时，用 AutoModel + 线性头兜底（支持大量 decoder-only）。"""
    def __init__(self, backbone, hidden_size: int, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = out.last_hidden_state  # [B,T,H]
        if attention_mask is not None:
            idx = attention_mask.long().sum(dim=1) - 1
            idx = idx.clamp(min=0)
        else:
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
        pooled = h[torch.arange(h.size(0), device=h.device), idx]  # last non-pad token
        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)

def _has_trainer(name: str) -> bool:
    """Return True if `name` is a registered finetune trainer."""
    if not name:
        return False
    try:
        from mgt_eval.detectors import ensure_all_detectors_registered
        ensure_all_detectors_registered()
        from mgt_eval.train.registry import get_trainer
        _ = get_trainer(name)  # may raise
        return True
    except Exception:
        return False


def _safe_tag(s: str) -> str:
    s = (s or "").strip()
    return "".join([c if c.isalnum() or c in "._-+" else "-" for c in s]).strip("-") or "run"

def _stratified_split_exact(examples, train_k: int, val_k: int, seed: int):
    """
    从 examples（list[dict{text,label}]) 中做分层抽样并精确切分到 train_k / val_k。
    - train_k/val_k 可被实际数据量截断
    - 返回 (train_list, val_list)
    """
    rng = random.Random(int(seed))
    # 按 label 分桶
    buckets = {}
    for i, ex in enumerate(examples):
        y = int(ex["label"])
        buckets.setdefault(y, []).append(ex)
    for y in buckets:
        rng.shuffle(buckets[y])

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return [], []

    want_total = max(0, int(train_k) + int(val_k))
    want_total = min(want_total, total)

    # 先决定每个 label 需要多少（按占比分配 + 余数修正）
    ys = sorted(buckets.keys())
    base = {}
    fracs = []
    for y in ys:
        prop = len(buckets[y]) / total
        raw = prop * want_total
        base[y] = int(raw)
        fracs.append((raw - base[y], y))
    cur = sum(base.values())
    # 补齐到 want_total
    fracs.sort(reverse=True)
    for _, y in fracs:
        if cur >= want_total:
            break
        if base[y] < len(buckets[y]):
            base[y] += 1
            cur += 1

    # 再把每个 label 的 base[y] 切成 val/train
    val_k = min(int(val_k), want_total)
    # val 分配同理：按该 label 被选中的数量占比
    val_base = {y: 0 for y in ys}
    if val_k > 0:
        fracs2 = []
        sel_total = sum(base.values()) or 1
        for y in ys:
            raw = (base[y] / sel_total) * val_k
            val_base[y] = int(raw)
            fracs2.append((raw - val_base[y], y))
        cur2 = sum(val_base.values())
        fracs2.sort(reverse=True)
        for _, y in fracs2:
            if cur2 >= val_k:
                break
            if val_base[y] < base[y]:
                val_base[y] += 1
                cur2 += 1

    train, val = [], []
    for y in ys:
        chosen = buckets[y][:base[y]]
        v = chosen[:val_base[y]]
        t = chosen[val_base[y]:]
        val.extend(v)
        train.extend(t)

    # 最后再打散一次（可选）
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

# =========================
# Train CLI Adapter Layer
# =========================

# 统一 train 的“通用键”（CLI 侧）
# dataset_* / sample_k_* / model1-3 / tokenizer / output_dir / seed / 通用超参
_TRAIN_ADAPTERS: Dict[str, Dict[str, Any]] = {
    # ---------- CoCo ----------
    "coco": {
        "required": ["dataset_train"],
        "rename": {
            "dataset_train": "dataset_training",
            "dataset_val": "dataset_val",
            "sample_k_train": "training_sample_k",
            "sample_k_val": "validation_sample_k",
            "model1": "base_model",   # model1 -> base_model
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay", "warmup_ratio",
            "grad_accum_steps", "max_length",
            "fp16",
        ],
    },

    # ---------- GREATER ----------
    "greater": {
        "required": ["dataset_train", "dataset_aux1"],  # dataset_aux1 -> dataset_surrogate
        "rename": {
            "dataset_train": "dataset_training",
            "dataset_aux1": "dataset_surrogate",
            "sample_k_train": "training_sample_k",
            "sample_k_val": "validation_sample_k",
            "sample_k_aux1": "surrogate_sample_k",
            "model1": "surrogate_base_model",
            "model2": "detector_base_model",
            "model3": "mlm_model",
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "max_length", "fp16",
            "warmup_ratio", "weight_decay", "grad_accum_steps",
        ],
    },

    # ---------- DeTeCtive ----------
    "detective": {
        "required": ["dataset_train"],
        "rename": {
            "dataset_train": "dataset_training",
            "dataset_val": "dataset_validation",
            "sample_k_train": "train_sample_limit",
            "sample_k_val": "val_sample_limit",
            "model1": "embedding_model",  # model1 -> embedding_model
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay", "max_length",
            "num_workers", "devices",
        ],
    },

    # ---------- PECOLA ----------
    "pecola": {
        "required": ["dataset_train"],
        "rename": {
            "dataset_train": "dataset_training",
            "dataset_val": "dataset_validation",
            "dataset_test": "dataset_test",
            "sample_k_train": "shot",          # ✅ NEW: 训练集条数 -> PECOLA shot
            "dataset_aux1": "dataset_training_extra",  # aux1 -> training_extra
            "sample_k_val": "validation_size",
            "sample_k_test": "test_size",
            "model1": "base_model",
            "model2": "t5_model",
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay", "warmup_ratio",
            "grad_accum_steps", "max_length",
            "fp16",
        ],
    },

    # ---------- ImBD ----------
    "imbd": {
        "required": ["dataset_train", "dataset_val", "dataset_test"],
        "rename": {
            "dataset_train": "dataset_train",
            "dataset_val": "dataset_val",
            "dataset_test": "dataset_test",
            "sample_k_train": "train_limit",
            "sample_k_val": "val_limit",
            "sample_k_test": "test_limit",
            "model1": "base_model",
            "model2": "reference_model",
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay",
        ],
    },

    # ---------- MPU ----------
    "mpu": {
        "required": ["dataset_train"],
        "rename": {
            "dataset_train": "dataset_training",
            "dataset_val": "dataset_validation",
            "sample_k_train": "train_sample_limit",
            "sample_k_val": "val_sample_limit",
            "model1": "base_model",
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay", "warmup_ratio",
            "max_length",
            "num_workers",
        ],
    },

    # ---------- Longformer ----------
    "longformer": {
        "required": ["dataset_train"],
        "rename": {
            "dataset_train": "dataset_train",
            "dataset_val": "dataset_val",
            "dataset_test": "dataset_test",
            "sample_k_train": "sample_k_train",
            "sample_k_val": "sample_k_val",
            "sample_k_test": "sample_k_test",
            "model1": "model",             # model1 -> longformer model dir/id
            "tokenizer": "tokenizer_path", # tokenizer -> tokenizer_path
        },
        "passthrough": [
            "output_dir", "seed",
            "epochs", "train_batch_size", "eval_batch_size",
            "lr", "weight_decay", "warmup_ratio",
            "grad_accum_steps", "max_length",
            "fp16", "device", "name",
        ],
    },
}

def _pick(v):
    return None if (v is None or (isinstance(v, str) and v.strip() == "")) else v

def _require_fields(det: str, common: Dict[str, Any]) -> None:
    spec = _TRAIN_ADAPTERS.get(det, None)
    if not spec:
        return
    miss = [k for k in spec.get("required", []) if _pick(common.get(k)) is None]
    if miss:
        raise SystemExit(f"[MGTEval][train] detector='{det}' missing required args: {', '.join(miss)}")

def _adapt_train_kwargs(det: str, common: Dict[str, Any]) -> Dict[str, Any]:
    """
    将统一 train 参数 common 映射到各 trainer 的真实 kwargs。
    """
    det_l = (det or "").lower()
    spec = _TRAIN_ADAPTERS.get(det_l, None)

    # fallback：未知 trainer 的最低限度映射
    if spec is None:
        out: Dict[str, Any] = {}
        if _pick(common.get("dataset_train")) is not None:
            out["dataset_training"] = common["dataset_train"]
        if _pick(common.get("dataset_val")) is not None:
            out["dataset_val"] = common["dataset_val"]
        if _pick(common.get("model1")) is not None:
            out["base_model"] = common["model1"]
        for k in ["output_dir", "seed", "epochs", "train_batch_size", "eval_batch_size", "lr", "weight_decay", "warmup_ratio", "max_length", "fp16"]:
            if _pick(common.get(k)) is not None:
                out[k] = common[k]
        return out

    out: Dict[str, Any] = {}

    # rename
    for src, dst in spec.get("rename", {}).items():
        v = _pick(common.get(src))
        if v is not None:
            out[dst] = v

    # passthrough
    for k in spec.get("passthrough", []):
        v = _pick(common.get(k))
        if v is not None:
            out[k] = v

    return out

def _now():
    return time.strftime("%Y%m%d-%H%M%S")

def _import_all_detectors():
    """动态 import mgt_eval.detectors 包下所有子模块，触发 @register。"""
    import pkgutil, importlib
    import mgt_eval.detectors as pkg
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            print(f"[MGTEval][list] skip {m.name}: {e}")

def cmd_list(_args):
    # 1) 预加载：尝试导入所有 detectors 子模块（触发 @register）
    from mgt_eval.detectors import ensure_all_detectors_registered
    ensure_all_detectors_registered()  # 轻量：内部自行 walk_packages

    # 2) 读取注册表
    try:
        from mgt_eval.detectors.registry import list_registered_detectors
        dets = list_registered_detectors()
    except Exception:
        # 兜底：直接访问 REGISTRY
        from mgt_eval.detectors.registry import REGISTRY as _REG
        dets = sorted(_REG.keys())

    if dets:
        print("\n".join(sorted(dets)))
    else:
        print("(no detectors found)")

def cmd_run(args):
    _import_all_detectors()

    det_name = (args.detector or "").strip()
    det_l = det_name.lower()

    # evaluator/Pretrained 统一的 sample_k 规范：<=0 -> None
    sample_k = None if (args.sample_k is None or int(args.sample_k) <= 0) else int(args.sample_k)
    attack_datasets = _parse_attack_dataset_single(
        getattr(args, "attack_dataset", None),
        data_fallback=getattr(args, "data", None),
    )
    asr_save_details = bool(getattr(args, "asr_save_details", True))

    # ------------------------------------------------------------
    # Case A) HF 预训练检测器别名：openai-detector-base 等（无需 --model1）
    # ------------------------------------------------------------
    try:
        from mgt_eval.detectors.pretrained.pretrained_entrypoints import (
            DETECTOR_MODEL_MAP,
            evaluate_pretrained_detector,
        )
    except Exception:
        DETECTOR_MODEL_MAP = {}
        evaluate_pretrained_detector = None

    if det_name in DETECTOR_MODEL_MAP:
        if evaluate_pretrained_detector is None:
            raise SystemExit("[MGTEval][run] pretrained entrypoints not available in this environment.")

        evaluate_pretrained_detector(
            detector_key=det_name,
            dataset=args.data,                    # 直接把 spec/path 交给 evaluator
            device=args.device,
            batch_size=int(args.batch_size),
            max_length=int(getattr(args, "max_length", 512)),
            fp16=bool(getattr(args, "fp16", False)),
            threshold=float(args.threshold),
            sample_k=sample_k,
            sample_seed=int(args.seed),
            group_cols=None,
            out_dir=args.out,
            save_curves=bool(getattr(args, "save_curves", True)),
            k_runs=int(getattr(args, "k_runs", 1)),
            attack_datasets=attack_datasets,
            asr_save_details=asr_save_details,
            show_progress=(not args.no_progress),
        )
        return
    # ------------------------------------------------------------
    # Case Detective) embedding KNN inference (NOT seqcls)
    # ------------------------------------------------------------
    if det_l == "detective":
        from mgt_eval.detectors.pretrained.pretrained import PretrainedDetector

        # 兼容：允许复用 model1/model2（你不想多写参数时）
        train_dataset = getattr(args, "train_dataset", None)
        embedding_model_name = getattr(args, "embedding_model_name", None) or getattr(args, "model2", None)
        embedding_ckpt_path = getattr(args, "embedding_ckpt_path", None) or getattr(args, "model1", None)

        if not train_dataset:
            raise SystemExit("[MGTEval][detective] missing --train_dataset (e.g., train.jsonl,dev.jsonl).")
        if not embedding_model_name:
            raise SystemExit("[MGTEval][detective] missing --embedding_model_name (or use --model2).")
        if not embedding_ckpt_path:
            raise SystemExit("[MGTEval][detective] missing --embedding_ckpt_path (or use --model1).")

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_standard_eval = bool(getattr(args, "detective_eval", False))
        if use_standard_eval:
            det = PretrainedDetector(
                model_path=embedding_model_name,
                name=det_name,
                device=device,
                max_length=int(getattr(args, "max_length", 512)),
                fp16=bool(getattr(args, "fp16", False)) and str(device).startswith("cuda"),
                knn_train_dataset=train_dataset,
                knn_embedding_model_name=embedding_model_name,
                knn_embedding_ckpt_path=embedding_ckpt_path,
                knn_pooling=getattr(args, "pooling", "average"),
                knn_K=int(getattr(args, "max_K", 5)),
                knn_backend=str(getattr(args, "knn_backend", "torch")).lower(),
                knn_cache_root=args.cache_root,
                knn_index_name=args.index_name,
                knn_reuse_database=args.reuse_database,
                knn_save_database=args.save_database,
                knn_use_gpu_index=args.use_gpu_index,
                knn_index_batch_size=int(getattr(args, "index_batch_size", 8)),
            )
            _eval(
                detector=det,
                dataset=args.data,
                batch_size=int(args.batch_size),
                threshold=float(args.threshold),
                show_progress=(not args.no_progress),
                out_dir=args.out,
                save_curves=bool(getattr(args, "save_curves", True)),
                k_runs=int(getattr(args, "k_runs", 1)),
                attack_datasets=attack_datasets,
                asr_save_details=asr_save_details,
                sample_k=sample_k,
                sample_seed=int(args.seed),
                group_cols=None,
            )
            return

        det = PretrainedDetector(
            device=device,
            max_length=int(getattr(args, "max_length", 512)),
            fp16=bool(getattr(args, "fp16", False)) and str(device).startswith("cuda"),
        )
        sample_k_train = getattr(args, "sample_k_train", None)
        sample_k_test = getattr(args, "sample_k_test", None)
        sample_k_train = None if (sample_k_train is None or int(sample_k_train) <= 0) else int(sample_k_train)
        if sample_k_test is None:
            sample_k_test = sample_k
        sample_k_test = None if (sample_k_test is None or int(sample_k_test) <= 0) else int(sample_k_test)
        res = det.embedding_knn_infer(
            train_dataset=train_dataset,
            test_dataset=args.data,
            embedding_model_name=embedding_model_name,
            embedding_ckpt_path=embedding_ckpt_path,
            pooling=getattr(args, "pooling", "average"),
            batch_size=int(args.batch_size),
            max_K=int(getattr(args, "max_K", 5)),
            cache_root=args.cache_root,
            save_dataset=args.save_dataset,
            save_database=args.save_database,
            reuse_database=args.reuse_database,
            index_name=args.index_name,
            sample_k_train=sample_k_train,
            sample_k_test=sample_k_test,
            use_gpu_index=args.use_gpu_index,
            return_probs=args.return_probs,
            seed=int(args.seed),
            index_batch_size=int(getattr(args, "index_batch_size", 8)),
            knn_backend=str(getattr(args, "knn_backend", "torch")).lower(),   # ✅ NEW
        )

        # CLI 输出：至少把 best_K / best_metrics 打印出来
        print("best K =", res.get("best_K"))
        print("best metrics =", res.get("best_metrics"))

        # 可选落盘：--out 给目录或 json 文件
        if getattr(args, "out", None):
            out_path = args.out
            if (out_path.endswith("/") or os.path.isdir(out_path)):
                os.makedirs(out_path, exist_ok=True)
                out_file = os.path.join(out_path, "detective_embedding_knn.json")
            else:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                out_file = out_path
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

        return

    # ------------------------------------------------------------
    # Case B) “coco/mpu/longformer/...” 这类：用 Pretrained(SeqCls/LoRA) 入口跑 checkpoint
    #   - 约定：--detector 给一个名字（比如 coco），--model1 给 checkpoint 路径
    #   - 或者：--lora_dir + --base 走 LoRA
    # ------------------------------------------------------------
    FINETUNED_SEQCLS_NAMES = {
        "coco", "mpu", "longformer", "greater", "imbd", "pecola",
        "hfcls", "seqcls", "pretrained",
        # 你也可以把自己其他“本质是 SeqCls checkpoint”的 detector 名称加这里
    }

    if det_l in FINETUNED_SEQCLS_NAMES:
        from mgt_eval.detectors import Pretrained  # 走你 quickstart_simpleai 的同款入口

        # LoRA 模式
        if getattr(args, "lora_dir", None):
            if not getattr(args, "ckpt_base", None):
                raise SystemExit("[MGTEval][run] using --lora_dir requires --base/--ckpt_base.")
            model_path = args.lora_dir
            tokenizer_path = args.tokenizer or args.ckpt_base
            extra_kwargs = {"ckpt_base": args.ckpt_base, "fp16": bool(args.fp16)}
        # 普通 checkpoint / 目录
        else:
            if not getattr(args, "model1", None):
                raise SystemExit(
                    "[MGTEval][run] missing --model1 (checkpoint dir/path). "
                    "For HF pretrained aliases, use --detector openai-detector-base/... without --model1."
                )
            model_path = args.model1
            tokenizer_path = args.tokenizer or args.model1
            extra_kwargs = {"fp16": bool(args.fp16)}
            if getattr(args, "ckpt_base", None):
                extra_kwargs["ckpt_base"] = args.ckpt_base

        # 直接调用 Pretrained(...)：内部会走 evaluator 落盘 summary/curves
        Pretrained(
            data=args.data,
            sample_k=sample_k,
            batch_size=int(args.batch_size),
            threshold=float(args.threshold),
            model_path=model_path,
            out_dir=args.out,
            tokenizer_path=tokenizer_path,
            max_length=int(getattr(args, "max_length", 512)),
            name=det_name,
            show_progress=(not args.no_progress),
            k_runs=int(getattr(args, "k_runs", 1)),
            save_curves=bool(getattr(args, "save_curves", True)),
            attack_datasets=attack_datasets,
            asr_save_details=asr_save_details,
            **extra_kwargs,
        )
        return

    # ------------------------------------------------------------
    # Case C) 兜底：沿用你原来的逻辑 detector 路径（_build_detector + evaluator）
    # ------------------------------------------------------------
    if not getattr(args, "model1", None):
        raise SystemExit(
            f"[MGTEval][run] detector='{det_name}' requires --model1 in logic-detector mode.\n"
            "If you meant a finetuned seqcls checkpoint, use --detector coco/mpu/longformer (or add it to FINETUNED_SEQCLS_NAMES).\n"
            "If you meant a builtin HF alias, use --detector openai-detector-base/... without --model1."
        )

    det = _build_detector(
        detector_name=args.detector,
        model1=args.model1,
        model2=args.model2,
        device=args.device,
        use_bfloat16=bool(args.bf16),
        detector_kwargs=(json.loads(args.detector_kwargs) if args.detector_kwargs else None),
        basemodel=args.basemodel,
        bart_ckpt=args.bart_ckpt,
    )

    _eval(
        detector=det,
        dataset=args.data,
        batch_size=int(args.batch_size),
        threshold=float(args.threshold),
        show_progress=(not args.no_progress),
        out_dir=args.out,
        save_curves=bool(getattr(args, "save_curves", True)),
        k_runs=int(getattr(args, "k_runs", 1)),
        attack_datasets=attack_datasets,
        asr_save_details=asr_save_details,
        sample_k=sample_k,                # ✅ 让 evaluator 自己加载时也能采样
        sample_seed=int(args.seed),
        group_cols=None,
    )

def cmd_calibrate(args):
    _import_all_detectors()
    _ = _calibrate(
        detector=args.detector,
        model1=args.model1,
        model2=args.model2,
        data=args.data,
        batch_size=int(args.batch_size),
        sample_k=int(args.sample_k),
        seed=int(args.seed),
        device=args.device,
        bf16=bool(args.bf16),
        detector_kwargs=(json.loads(args.detector_kwargs) if args.detector_kwargs else None),
        basemodel=args.basemodel,
        bart_ckpt=args.bart_ckpt,
        calibrator_name=args.calibrator_name,
        l2=float(args.l2),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        standardize=not args.no_standardize,
        mode=args.mode,                 # NEW: 传入阈值搜索模式
        out=args.out,
        out_dir=args.out_dir,
        show_progress=(not args.no_progress),
    )

def cmd_train(args):
    from mgt_eval.detectors import ensure_all_detectors_registered
    ensure_all_detectors_registered()

    # 1) 先解析 train_kwargs JSON（trainer 原生键），作为底座
    extra: Dict[str, Any] = {}
    if getattr(args, "train_kwargs", None):
        try:
            extra = json.loads(args.train_kwargs)
            if not isinstance(extra, dict):
                raise ValueError("train_kwargs JSON must be an object/dict.")
        except Exception as e:
            raise SystemExit(f"[MGTEval][train] Failed to parse --train_kwargs as JSON: {e}")

    # 2) 收集统一参数 common
    common: Dict[str, Any] = dict(
        detector=(args.detector or "").strip(),
        dataset_train=getattr(args, "dataset_train", None),
        dataset_val=getattr(args, "dataset_val", None),
        dataset_test=getattr(args, "dataset_test", None),
        dataset_aux1=getattr(args, "dataset_aux1", None),
        dataset_aux2=getattr(args, "dataset_aux2", None),
        sample_k_train=getattr(args, "sample_k_train", None),
        sample_k_val=getattr(args, "sample_k_val", None),
        sample_k_test=getattr(args, "sample_k_test", None),
        sample_k_aux1=getattr(args, "sample_k_aux1", None),
        sample_k_aux2=getattr(args, "sample_k_aux2", None),
        model1=getattr(args, "model1", None),
        model2=getattr(args, "model2", None),
        model3=getattr(args, "model3", None),
        tokenizer=getattr(args, "tokenizer", None),

        output_dir=getattr(args, "output_dir", None),
        seed=getattr(args, "seed", None),
        epochs=getattr(args, "epochs", None),
        train_batch_size=getattr(args, "train_batch_size", None),
        eval_batch_size=getattr(args, "eval_batch_size", None),
        lr=getattr(args, "lr", None),
        weight_decay=getattr(args, "weight_decay", None),
        warmup_ratio=getattr(args, "warmup_ratio", None),
        grad_accum_steps=getattr(args, "grad_accum_steps", None),
        max_length=getattr(args, "max_length", None),
        fp16=getattr(args, "fp16", None),
        num_workers=getattr(args, "num_workers", None),
        device=getattr(args, "device", None),
        name=getattr(args, "name", None),
    )

    det_raw = common["detector"].strip()

    # 支持 hf: 前缀强制进入 HF 模式，避免与 trainer 名冲突
    force_hf = False
    if det_raw.lower().startswith("hf:"):
        force_hf = True
        det_raw = det_raw[3:].strip()

    det_l = det_raw.lower()

    # ---------------------------
    # helper: resolve nested keys from trainer result
    # ---------------------------
    def _deep_get(d: Any, path: List[str]) -> Any:
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k, None)
        return cur

    def _pick_from_result(result: Any, keys: List[str]) -> Any:
        """
        Try multiple common nesting patterns:
          - top-level: result[key]
          - result["train"][key]
          - result["train"]["train"][key]
          - result["raw"][key]
          - result["raw"]["train"][key]
        """
        if not isinstance(result, dict):
            return None
        candidates = []
        for k in keys:
            candidates.extend([
                [k],
                ["train", k],
                ["train", "train", k],
                ["raw", k],
                ["raw", "train", k],
            ])
        for p in candidates:
            v = _deep_get(result, p)
            if v is not None:
                return v
        return None

    def _is_valid_dir(p: Any) -> bool:
        try:
            return bool(p) and os.path.isdir(str(p))
        except Exception:
            return False

    def _norm_int(v, default=None):
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    def _norm_float(v, default=None):
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    # ---------------------------
    # Auto-eval helper (run detect once after training)
    # ---------------------------
    def _auto_eval_detect_once(
        *,
        detector_name_for_eval: str,
        ckpt_dir: str,
        test_spec: str,
        test_sample_k: Optional[int],
        train_run_dir: str,
        batch_size: int,
        device: Optional[str],
        fp16: bool,
        max_length: int,
        threshold: float = 0.5,
        save_curves: bool = True,
        eval_prefix: str = "eval_test",
    ) -> Optional[str]:
        """
        Run a detect-style evaluation (Pretrained entrypoint) and save under:
          {train_run_dir}/{eval_prefix}_{timestamp}/metrics/summary.json
        Return eval_out_dir if succeeded, else None.
        """
        try:
            from mgt_eval.detectors import Pretrained

            # 目录对齐你现在 ckpt 结构：eval_test_YYYYMMDD-HHMMSS
            eval_out_dir = os.path.join(train_run_dir, f"{eval_prefix}_{_now()}")
            sample_k_eval = None
            if test_sample_k is not None and int(test_sample_k) > 0:
                sample_k_eval = int(test_sample_k)

            Pretrained(
                data=test_spec,
                sample_k=sample_k_eval,
                batch_size=int(batch_size),
                threshold=float(threshold),
                model_path=ckpt_dir,
                out_dir=eval_out_dir,
                tokenizer_path=ckpt_dir,              # tokenizer follows ckpt (best/last)
                max_length=int(max_length),
                name=str(detector_name_for_eval),
                show_progress=True,
                k_runs=1,
                save_curves=bool(save_curves),
                attack_datasets=None,
                asr_save_details=True,
                device=device,
                fp16=bool(fp16),
            )
            return eval_out_dir
        except Exception as e:
            print(f"[MGTEval][train] Warning: auto-eval (detect) failed: {e}")
            return None

    # ---------------------------
    # Trainer path (registered)
    # ---------------------------
    if (not force_hf) and _has_trainer(det_l):
        from mgt_eval.train.registry import get_trainer
        trainer = get_trainer(det_l)

        _require_fields(det_l, {**common, "detector": det_l})
        adapted = _adapt_train_kwargs(det_l, {**common, "detector": det_l})

        train_kwargs: Dict[str, Any] = dict(extra)
        train_kwargs.update(adapted)

        result = trainer(**train_kwargs)

        # -------- resolve dirs robustly (fix your PECOLA case) --------
        best_dir = _pick_from_result(result, ["best_dir", "best_model_dir", "model_dir"])
        last_dir = _pick_from_result(result, ["last_dir", "model_dir"])
        run_dir  = _pick_from_result(result, ["run_dir", "output_root", "output_dir"])
        # choose train_run_dir
        train_run_dir = None
        if _is_valid_dir(run_dir):
            train_run_dir = str(run_dir)
        elif _is_valid_dir(best_dir):
            train_run_dir = os.path.dirname(str(best_dir))
        elif _is_valid_dir(last_dir):
            train_run_dir = os.path.dirname(str(last_dir))
        else:
            # fallback: user provided output_dir (may be a root)
            train_run_dir = train_kwargs.get("output_dir") or train_kwargs.get("out_dir") or "runs_train"

        os.makedirs(train_run_dir, exist_ok=True)

        # 保存训练输出（原 result 原样存）
        train_summary_path = os.path.join(train_run_dir, "train_summary.json")
        try:
            with open(train_summary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MGTEval][train] Warning: failed to write train_summary.json: {e}")

        # -------- optional auto-eval on dataset_test --------
        test_spec = common.get("dataset_test")
        if test_spec:
            ckpt_dir = None

            if _is_valid_dir(best_dir):
                ckpt_dir = str(best_dir)
            elif _is_valid_dir(last_dir):
                ckpt_dir = str(last_dir)
            else:
                # try conventional placements
                cand_best = os.path.join(train_run_dir, "best")
                cand_last = os.path.join(train_run_dir, "last")
                if os.path.isdir(cand_best):
                    ckpt_dir = cand_best
                elif os.path.isdir(cand_last):
                    ckpt_dir = cand_last

            if ckpt_dir:
                bs   = _norm_int(common.get("eval_batch_size"), 32) or 32
                ml   = _norm_int(common.get("max_length"), 512) or 512
                dev  = common.get("device", None)
                fp16 = bool(common.get("fp16", False))

                eval_out_dir = _auto_eval_detect_once(
                    detector_name_for_eval=det_l,
                    ckpt_dir=ckpt_dir,
                    test_spec=str(test_spec),
                    test_sample_k=common.get("sample_k_test"),
                    train_run_dir=train_run_dir,
                    batch_size=bs,
                    device=dev,
                    fp16=fp16,
                    max_length=ml,
                    threshold=0.5,
                    save_curves=True,
                    eval_prefix="eval_test",
                )
                if eval_out_dir:
                    print(f"[MGTEval][train] Auto-eval saved to: {eval_out_dir}")
            else:
                print("[MGTEval][train] Warning: no ckpt_dir found for auto-eval.")
        # -------- cleanup: remove empty non-timestamp output_dir (if created by fallback) --------
        base_out = train_kwargs.get("output_dir")
        try:
            if base_out:
                base_out = os.path.abspath(str(base_out))
                tr_dir = os.path.abspath(str(train_run_dir))
                # 仅当 base_out != 实际 run dir 且 base_out 为空目录时才删除
                if base_out != tr_dir and os.path.isdir(base_out) and len(os.listdir(base_out)) == 0:
                    os.rmdir(base_out)
                    print(f"[MGTEval][train] Removed empty folder: {base_out}")
        except Exception as e:
            print(f"[MGTEval][train] Warning: failed to remove empty output_dir: {e}")

        return
    # 在 _resolve_hf_id_or_path 或 normalize 之前加一层映射
    HF_ALIAS = {
        "openai-detector-base": "openai-community/roberta-base-openai-detector",
        "openai-detector-large": "openai-community/roberta-large-openai-detector",
        "simpleai-detector": "Hello-SimpleAI/chatgpt-detector-roberta",
        "radar": "TrustSafeAI/RADAR-Vicuna-7B",
    }

    det_raw = args.detector
    det_raw = HF_ALIAS.get(det_raw.lower(), det_raw)
    # ---------------------------
    # HF finetune fallback path
    # ---------------------------
    hf_id = _resolve_hf_id_or_path(det_raw)
    base_model = common["model1"] or hf_id
    tok_name = common["tokenizer"] or base_model

    seed = int(common["seed"] or 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 输出目录
    out_root = common["output_dir"] or "runs_hfcls"
    run_dir = os.path.join(out_root, f"hfcls_{_safe_tag(hf_id)}_{_now()}")
    os.makedirs(run_dir, exist_ok=True)

    # -------- load datasets + apply sample_k (支持 train/val 精确控制) --------
    train_k = common["sample_k_train"]
    val_k = common["sample_k_val"]

    if common["dataset_val"]:
        train_examples, _ = load_dataset_unified(
            dataset=common["dataset_train"],
            sample_k=(None if train_k is None else int(train_k)),
            sample_seed=seed,
            group_cols=None,
        )
        val_examples, _ = load_dataset_unified(
            dataset=common["dataset_val"],
            sample_k=(None if val_k is None else int(val_k)),
            sample_seed=seed,
            group_cols=None,
        )
    else:
        total_k = None
        if train_k is not None and val_k is not None:
            total_k = int(train_k) + int(val_k)
        elif train_k is not None:
            total_k = int(train_k)

        all_examples, _ = load_dataset_unified(
            dataset=common["dataset_train"],
            sample_k=(None if total_k is None else int(total_k)),
            sample_seed=seed,
            group_cols=None,
        )

        vk = int(val_k) if val_k is not None else max(1, int(round(0.1 * len(all_examples))))
        tk = int(train_k) if train_k is not None else max(0, len(all_examples) - vk)
        train_examples, val_examples = _stratified_split_exact(all_examples, tk, vk, seed)

    # -------- load test dataset (optional) --------
    test_examples = None
    if common.get("dataset_test"):
        test_k = common.get("sample_k_test", None)
        test_examples, _ = load_dataset_unified(
            dataset=common["dataset_test"],
            sample_k=(None if test_k is None else int(test_k)),
            sample_seed=seed,
            group_cols=None,
        )

    # -------- build tokenizer/model --------
    trust = bool(getattr(args, "trust_remote_code", False)) or bool(extra.get("trust_remote_code", False))
    num_labels = int(extra.get("num_labels", 2))

    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True, trust_remote_code=trust)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token

    cfg_hf = AutoConfig.from_pretrained(base_model, trust_remote_code=trust)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=num_labels, trust_remote_code=trust
        )
        model_kind = "auto_seqcls"
    except Exception:
        backbone = AutoModel.from_pretrained(base_model, trust_remote_code=trust)
        hidden = _infer_hidden_size(cfg_hf)
        drop = float(getattr(cfg_hf, "hidden_dropout_prob", 0.1))
        model = _HFBackboneClsWrapper(backbone, hidden_size=hidden, num_labels=num_labels, dropout=drop)
        model_kind = "wrapped_backbone"

    if getattr(model, "config", None) is not None:
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # -------- build cfg and call your trainer --------
    cfg = dict(extra)
    cfg.update({"output_dir": run_dir, "seed": seed})

    if common["epochs"] is not None: cfg["epochs"] = int(common["epochs"])
    if common["train_batch_size"] is not None: cfg["train_batch_size"] = int(common["train_batch_size"])
    if common["eval_batch_size"] is not None: cfg["eval_batch_size"] = int(common["eval_batch_size"])
    if common["lr"] is not None: cfg["lr"] = float(common["lr"])
    if common["weight_decay"] is not None: cfg["weight_decay"] = float(common["weight_decay"])
    if common["warmup_ratio"] is not None: cfg["warmup_ratio"] = float(common["warmup_ratio"])
    if common["grad_accum_steps"] is not None: cfg["grad_accum_steps"] = int(common["grad_accum_steps"])
    if common["max_length"] is not None: cfg["max_length"] = int(common["max_length"])
    if common["fp16"] is not None: cfg["fp16"] = bool(common["fp16"])
    if common["num_workers"] is not None: cfg["num_workers"] = int(common["num_workers"])
    if common["device"] is not None: cfg["device"] = common["device"]

    result = _train_seqcls_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_examples,
        val_dataset=val_examples,
        # test_dataset=test_examples,
        cfg=SimpleNamespace(**cfg),
        dataset_spec=str(common["dataset_train"]),
    )

    out = {
        "train": {
            "trainer": "hfcls",
            "model_kind": model_kind,
            "hf_id": hf_id,
            "base_model_used": base_model,
            "tokenizer_used": tok_name,
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "test_size": (len(test_examples) if test_examples is not None else 0),
            "output_dir": run_dir,
            "best_dir": result.get("best_dir"),
            "last_dir": result.get("last_dir"),
            "best_val_acc": result.get("best_val_acc"),
            "test_acc": result.get("test_acc"),
            "test_loss": result.get("test_loss"),
            "test_used": result.get("test_used"),
        },
        "raw": result,
    }

    # ---------------------------
    # Persist train summary
    # ---------------------------
    train_summary_path = os.path.join(run_dir, "train_summary.json")
    try:
        with open(train_summary_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[MGTEval][train] Warning: failed to write train_summary.json: {e}")

    # ---------------------------
    # Optional auto-eval on dataset_test (detect-style)
    # ---------------------------
    if common.get("dataset_test"):
        ckpt_dir = out["train"].get("best_dir") or out["train"].get("last_dir")
        if not (ckpt_dir and os.path.isdir(str(ckpt_dir))):
            cand_best = os.path.join(run_dir, "best")
            ckpt_dir = cand_best if os.path.isdir(cand_best) else None

        if ckpt_dir:
            bs = int(cfg.get("eval_batch_size", 32))
            ml = int(cfg.get("max_length", 512))
            dev = cfg.get("device", None)
            fp16 = bool(cfg.get("fp16", False))

            eval_out_dir = _auto_eval_detect_once(
                detector_name_for_eval="hfcls",
                ckpt_dir=str(ckpt_dir),
                test_spec=str(common["dataset_test"]),
                test_sample_k=common.get("sample_k_test"),
                train_run_dir=run_dir,
                batch_size=bs,
                device=dev,
                fp16=fp16,
                max_length=ml,
                threshold=0.5,
                save_curves=True,
                eval_prefix="eval_test",
            )
            if eval_out_dir:
                print(f"[MGTEval][train] Auto-eval saved to: {eval_out_dir}")
        else:
            print("[MGTEval][train] Warning: no ckpt_dir found for auto-eval.")

    # -------- cleanup (HF fallback): normally nothing to remove --------
    try:
        base_out = common.get("output_dir")          # 这是 root，比如 runs_hfcls
        train_run_dir = run_dir                      # 这是实际 run 目录（带时间戳）
        if base_out:
            base_out = os.path.abspath(str(base_out))
            tr_dir = os.path.abspath(str(train_run_dir))
            # 只有当 base_out 是空目录且不等于 run_dir 时才删
            if base_out != tr_dir and os.path.isdir(base_out) and len(os.listdir(base_out)) == 0:
                os.rmdir(base_out)
                print(f"[MGTEval][train] Removed empty folder: {base_out}")
    except Exception as e:
        print(f"[MGTEval][train] Warning: failed to remove empty output_dir: {e}")

    return
    
def main(argv=None):
    ap = argparse.ArgumentParser(prog="mgt-eval", description="Unified CLI for MGT-Eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # list
    ap_list = sub.add_parser("list", help="List available detectors")
    ap_list.set_defaults(_fn=cmd_list)

    # run
    ap_run = sub.add_parser("detect", help="Run detector on a dataset")
    ap_run.add_argument(
        "--detector", required=True,
        help=(
            "Detector name. \n"
            "1) Logic detectors (gltr/lastde/...): needs --model1 as scoring model.\n"
            "2) Finetuned seqcls checkpoints (e.g., coco/mpu/longformer): use --model1 as checkpoint dir (or --lora_dir).\n"
            "3) HF pretrained aliases (openai-detector-base/simpleai-detector/...): NO --model1 needed."
        )
    )
    ap_run.add_argument("--data", required=True)
        # ---- NEW: ASR (Attack Success Rate) ----
    ap_run.add_argument(
        "--attack",
        dest="attack_dataset",
        nargs="?",              # ✅ 允许可选参数
        const="__SELF__",        # ✅ 只写 --attack 时，用 --data 作为 attack dataset
        default=None,
        help=(
            "Enable ASR with ONE attack dataset.\n"
            "Usage:\n"
            "  --attack            (use --data itself as paired-record attack dataset)\n"
            "  --attack adv.jsonl  (use given adv.jsonl)\n"
            "Note: only ONE dataset is supported (no repeated flags, no commas)."
        ),
    )

    ap_run.add_argument(
        "--no-asr-details",
        dest="asr_save_details",
        action="store_false",
        help="If set, do not save detailed metrics/asr.json (only keep summary.json).",
    )
    ap_run.set_defaults(asr_save_details=True)

    # ---- detective specific (embedding KNN inference) ----
    ap_run.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Detective: train/dev dataset spec for building the FAISS database "
            "(e.g., 'train.jsonl,dev.jsonl')."
    )
    ap_run.add_argument(
        "--embedding_model_name",
        type=str,
        default=None,
        help="Detective: embedding model name/path (e.g., princeton-nlp/unsup-simcse-roberta-base). "
    )
    ap_run.add_argument(
        "--embedding_ckpt_path",
        type=str,
        default=None,
        help="Detective: trained detector ckpt path (e.g., .../model_last.pth)."
    )
    ap_run.add_argument("--pooling", type=str, default="average",
                        help="Detective: pooling method (e.g., average/cls/last).")
    ap_run.add_argument("--max_K", type=int, default=5, help="Detective: max K to search.")
    ap_run.add_argument("--cache_root", type=str, default="~/.cache/mgt_eval",
                        help="Detective: cache root for faiss db/index.")
    ap_run.add_argument("--sample_k_train", type=int, default=None,
                    help="Detective: subsample size for --train_dataset (database). Default: full.")
    ap_run.add_argument("--sample_k_test", type=int, default=None,
                        help="Detective: subsample size for --data (test). Default: follow --sample_k.")
    # 默认 True：save_dataset
    ap_run.add_argument("--save_dataset", dest="save_dataset", action="store_true",
                        help="Detective: save processed dataset cache.")
    ap_run.add_argument("--no-save-dataset", dest="save_dataset", action="store_false",
                        help="Detective: do NOT save processed dataset cache.")
    ap_run.add_argument(
        "--knn_backend",
        type=str,
        default="torch",
        choices=["torch", "faiss"],
        help="Detective: KNN backend. torch=matrix topk; faiss=FAISS index (cacheable)."
    )
    # 默认 True：save_database
    ap_run.add_argument("--save_database", dest="save_database", action="store_true",
                        help="Detective: save FAISS database/index to disk.")
    ap_run.add_argument("--no-save-database", dest="save_database", action="store_false",
                        help="Detective: do NOT save FAISS database/index to disk.")
    #   默认 True：reuse_database
    ap_run.add_argument("--reuse_database", dest="reuse_database", action="store_true",
                        help="Detective: reuse existing database if present.")
    ap_run.add_argument("--no-reuse-database", dest="reuse_database", action="store_false",
                        help="Detective: do NOT reuse existing database.")    
    ap_run.add_argument("--index_name", type=str, default=None, help="Detective: faiss index name (None=auto hash).")
    ap_run.add_argument("--use_gpu_index", action="store_true", help="Detective: use GPU faiss index.")
    ap_run.add_argument("--return_probs", action="store_true", help="Detective: return probabilities.")
    ap_run.add_argument("--index_batch_size", type=int, default=8,
                        help="Detective: retrieval batch size (avoid cublas 13).")
    # 你 run 已经有 --seed / --batch_size / --num_workers(目前只有 train 有)；
    # 这里建议 detect 也提供 num_workers
    ap_run.add_argument("--num_workers", type=int, default=0, help="Detective: num_workers for dataloading.")
    ap_run.add_argument(
        "--detective_eval",
        action="store_true",
        help="Detective: run standard evaluator (metrics/summary.json); uses --max_K as K.",
    )

    # NOTE: 不再强制 required=True（因为 openai-detector-base 等不需要）
    ap_run.add_argument("--model1", "--model", dest="model1", required=False,
                        help="For metric-base detectors: scoring model path/id. For seqcls: checkpoint dir/path.")

    ap_run.add_argument("--model2", default=None)
    ap_run.add_argument("--batch_size", type=int, default=8)
    ap_run.add_argument("--threshold", type=float, default=0.5)
    ap_run.add_argument("--seed", type=int, default=114514)
    ap_run.add_argument("--sample_k", type=int, default=None)
    ap_run.add_argument("--device", default=None)

    # ---- NEW: seqcls/Pretrained 评测用参数 ----
    ap_run.add_argument("--tokenizer", default=None,
                        help="Tokenizer path/id (seqcls/Pretrained only). Default follows --model1 or --ckpt_base.")
    ap_run.add_argument("--max_length", type=int, default=512)

    ap_run.add_argument("--lora_dir", default=None,
                        help="LoRA adapter dir (seqcls/Pretrained only).")
    ap_run.add_argument("--base", "--ckpt_base", dest="ckpt_base", default=None,
                        help="LoRA base model dir/id (seqcls/Pretrained only).")

    ap_run.add_argument("--fp16", action="store_true",
                        help="Use fp16 on CUDA (seqcls/Pretrained + HF pretrained aliases).")

    # ---- 原逻辑 detector 用 bf16（修正 argparse bool 用法：默认 True，可手动关闭）----
    ap_run.add_argument("--bf16", dest="bf16", action="store_true",
                        help="Enable bfloat16 (logic detectors).")
    ap_run.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable bfloat16 (logic detectors).")
    ap_run.set_defaults(
        save_dataset=True,
        save_database=True,
        reuse_database=True,
        use_gpu_index=True,
        return_probs=True,
    )
    ap_run.set_defaults(bf16=True)

    ap_run.add_argument("--detector_kwargs", default=None,
                        help='JSON string, e.g. \'{"max_length":512}\'')

    ap_run.add_argument("--basemodel", default=None)
    ap_run.add_argument("--bart_ckpt", default=None)
    ap_run.add_argument("--out", default=None)

    # save_curves 同样别用 type=bool
    ap_run.add_argument("--save_curves", dest="save_curves", action="store_true",
                        help="Save curves/figures (evaluator).")
    ap_run.add_argument("--no-save-curves", dest="save_curves", action="store_false")
    ap_run.set_defaults(save_curves=True)

    ap_run.add_argument("--no_progress", action="store_true")
    ap_run.add_argument("--k_runs", type=int, default=1)
    ap_run.set_defaults(_fn=cmd_run)

    # calibrate
    ap_cal = sub.add_parser("calibrate", help="Fit and save a calibrator JSON")
    ap_cal.add_argument("--detector", required=True)
    ap_cal.add_argument("--data", required=True)
    ap_cal.add_argument("--model1", required=True)
    ap_cal.add_argument("--model2", default=None)
    ap_cal.add_argument("--batch_size", type=int, default=32)
    ap_cal.add_argument("--sample_k", type=int, default=None)
    ap_cal.add_argument("--seed", type=int, default=114514)
    ap_cal.add_argument("--device", default=None)
    ap_cal.add_argument("--bf16", action="store_true")
    ap_cal.add_argument("--detector_kwargs", default=None,
                        help='JSON string for detector extra kwargs')
    ap_cal.add_argument("--basemodel", default=None)
    ap_cal.add_argument("--bart_ckpt", default=None)
    ap_cal.add_argument("--calibrator_name", default="platt_lr")
    ap_cal.add_argument("--l2", type=float, default=1e-2)
    ap_cal.add_argument("--max_iter", type=int, default=200)
    ap_cal.add_argument("--tol", type=float, default=1e-6)
    ap_cal.add_argument("--no_standardize", action="store_true")
    # NEW: 阈值搜索模式
    ap_cal.add_argument(
        "--mode",
        default="acc",
        choices=["acc", "f1", "tpr"],
        help="Decision threshold search mode for calibrator: acc / f1 / tpr (TPR@FPR<=0.01).",
    )
    ap_cal.add_argument("--out", default=None)
    ap_cal.add_argument("--out_dir", default=None)
    ap_cal.add_argument("--no_progress", action="store_true")

    # train
    ap_train = sub.add_parser("train", help="Train a finetuned detector with unified interface")
    ap_train.add_argument(
        "--detector", required=True,
        help=(
            "Trainer name (coco/greater/pecola/...) OR an HF model id/local path for classification finetuning.\n"
            "Rule: if detector matches a registered trainer -> use it; else -> HF classification finetune.\n"
            "Force HF mode with prefix: hf:<hf_id_or_path> (useful if it conflicts with a trainer name).\n"
            "Shorthands supported for convenience: deberta-v3-base, deberta-v3-large, ..."
        ),
    )

    # -------- unified datasets --------
    ap_train.add_argument("--dataset_train", "--dataset_training", dest="dataset_train", required=True,
                          help="Primary training dataset spec/path.")
    ap_train.add_argument("--dataset_val", "--dataset_validation", dest="dataset_val", default=None,
                          help="Optional validation dataset spec/path.")
    ap_train.add_argument("--dataset_test", dest="dataset_test", default=None,
                          help="Optional test dataset spec/path.")

    # two optional auxiliary datasets (for methods like GREATER/PECOLA)
    ap_train.add_argument("--dataset_aux1", "--dataset_surrogate", "--dataset_training_extra",
                          dest="dataset_aux1", default=None,
                          help="Optional auxiliary dataset #1 (GREATER: surrogate; PECOLA: training_extra).")
    ap_train.add_argument("--dataset_aux2", dest="dataset_aux2", default=None,
                          help="Optional auxiliary dataset #2 (rare; reserved).")

    # -------- unified sampling --------
    ap_train.add_argument("--sample_k_train", "--training_sample_k", dest="sample_k_train", type=int, default=None,
                          help="Subsample size for train (None = full).")
    ap_train.add_argument("--sample_k_val", "--validation_sample_k", dest="sample_k_val", type=int, default=None,
                          help="Subsample size for val (None = full).")
    ap_train.add_argument("--sample_k_test", dest="sample_k_test", type=int, default=None,
                          help="Subsample size for test (None = full).")
    ap_train.add_argument("--sample_k_aux1", "--surrogate_sample_k", dest="sample_k_aux1", type=int, default=None,
                          help="Subsample size for aux1 (e.g., GREATER surrogate_sample_k).")
    ap_train.add_argument("--sample_k_aux2", dest="sample_k_aux2", type=int, default=None,
                          help="Subsample size for aux2.")

    # -------- unified models (up to 3) --------
    ap_train.add_argument("--model1", "--base_model", "--surrogate_base_model", "--embedding_model",
                          dest="model1", default=None,
                          help="Model #1 (detector-dependent).")
    ap_train.add_argument("--model2", "--detector_base_model", "--t5_model", "--reference_model",
                          dest="model2", default=None,
                          help="Model #2 (detector-dependent).")
    ap_train.add_argument("--model3", "--mlm_model",
                          dest="model3", default=None,
                          help="Model #3 (detector-dependent, e.g., GREATER mlm_model).")
    ap_train.add_argument("--tokenizer", "--tokenizer_path", dest="tokenizer", default=None,
                          help="Tokenizer path/id (primarily for longformer; otherwise ignored).")

    # -------- common hyperparams --------
    ap_train.add_argument("--out", dest="output_dir", default=None,
                          help="Output directory root for saving checkpoints.")
    ap_train.add_argument("--seed", type=int, default=42)

    ap_train.add_argument("--epochs", type=int, default=None)
    ap_train.add_argument("--train_batch_size", type=int, default=None)
    ap_train.add_argument("--eval_batch_size", type=int, default=None)
    ap_train.add_argument("--lr", type=float, default=None)
    ap_train.add_argument("--weight_decay", type=float, default=None)
    ap_train.add_argument("--warmup_ratio", type=float, default=None)
    ap_train.add_argument("--grad_accum_steps", type=int, default=None)
    ap_train.add_argument("--max_length", type=int, default=None)
    ap_train.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading HF models with custom code (use with caution).",
    )
    # tri-state fp16: default None (don't override trainer defaults)
    ap_train.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    ap_train.add_argument("--no-fp16", dest="fp16", action="store_false")

    ap_train.add_argument("--num_workers", type=int, default=None)
    ap_train.add_argument("--devices", type=int, default=None)

    # a few extra common knobs (mostly for longformer-like trainers)
    ap_train.add_argument("--device", type=str, default=None)
    ap_train.add_argument("--name", type=str, default=None)

    # keep escape hatch
    ap_train.add_argument("--train_kwargs", default=None,
                          help='Extra JSON dict forwarded to the trainer (trainer-native keys).')

    ap_train.set_defaults(_fn=cmd_train)
    ap_cal.set_defaults(_fn=cmd_calibrate)

    args = ap.parse_args(argv)
    args._fn(args)

if __name__ == "__main__":
    main()
