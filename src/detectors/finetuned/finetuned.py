# mgt_eval/finetuned/finetuned.py
from __future__ import annotations
import os, random, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from data_utils.load import load_dataset_unified
from train.train import train_model
from train.registry import register_train

# ========= ：PEFT / LoRA （） =========
_PEFT_OK = True
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception:
    _PEFT_OK = False

# ---------------- basic utils ----------------
def _seed_everything(seed: int = 114514):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _is_local_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

def _resolve_base(model: Optional[str]) -> str:
    """
    解析基础模型：
      1) 本地 HF 目录（含 config.json） -> 直接返回
      2) 常用别名 -> 对应 HF id
      3) 其余当作 HF id 尝试
      4) 全部失败 -> 回退 'xlm-roberta-base'
    """
    fallback = "xlm-roberta-base"
    if not model:
        return fallback
    spec = model.strip()

    # 1)  HF
    if _is_local_hf_dir(spec):
        return spec

    # 2)
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

    # 3)  HF id
    try:
        AutoConfig.from_pretrained(spec)
        return spec
    except Exception:
        return fallback

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
    # ---- runtime behavior hints for detector ----
    outputs_prob: bool = True          # finetuned detector
    disable_calibration: bool = True   # runner.Calibrate/Platt （）
    force_runner_calibration: bool = False  # evaluate()  inline IRLS
    auto_calibrate: bool = False            # calibrator json`
    # ========= ：LoRA （） =========
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"              # "none" | "lora_only" | "all"
    lora_target_modules: Optional[List[str]] = None
    lora_task_type: str = "SEQ_CLS"      # peft.TaskType
    # ：
    lora_merge_on_save: bool = True      # True=“”， from_pretrained
    lora_export_tag: str = "merged"
    lora_save_adapter_copy: bool = True  # （/）

class TLDS(Dataset):
    def __init__(self, exs: List[Dict[str, Any]]): self.exs = exs
    def __len__(self): return len(self.exs)
    def __getitem__(self, idx: int):
        e = self.exs[idx]; return {"text": e["text"], "label": int(e["label"])}

def _stratified_split(examples: List[Dict[str, Any]],
                      train_ratio: float, val_ratio: float, test_ratio: float,
                      seed: int = 114514):
    if train_ratio <= 0 and val_ratio <= 0 and test_ratio <= 0:
        train_ratio, val_ratio, test_ratio = 1.0, 0.0, 0.0
    pos = [e for e in examples if int(e["label"]) == 1]
    neg = [e for e in examples if int(e["label"]) == 0]
    def _split(lst):
        rng = np.random.RandomState(seed); idx = np.arange(len(lst)); rng.shuffle(idx)
        tot = len(idx); S = train_ratio + val_ratio + test_ratio
        n_tr = int(round(tot * (train_ratio / S))) if S > 0 else tot
        n_va = int(round(tot * (val_ratio / S))) if S > 0 else 0
        n_tr = min(n_tr, tot); n_va = min(n_va, tot - n_tr); n_te = tot - n_tr - n_va
        return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    p_tr,p_va,p_te = _split(pos); n_tr,n_va,n_te = _split(neg)
    train = [pos[i] for i in p_tr] + [neg[i] for i in n_tr]
    val   = [pos[i] for i in p_va] + [neg[i] for i in n_va]
    test  = [pos[i] for i in p_te] + [neg[i] for i in n_te]
    rng = np.random.RandomState(seed); rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

# ： generation_config
try:
    from transformers import GenerationConfig
except Exception:
    GenerationConfig = None

# ========= ： LoRA  =========
def _infer_lora_targets(model_type: str) -> List[str]:
    """
    针对常见 Encoder/Decoder 名称做最常用的映射；用户也可通过 CLI 显式传入覆盖。
    """
    mt = (model_type or "").lower()
    # BERT/Roberta/DeBERTa/XLM-R： Q/K/V  Q/V  LoRA
    if any(k in mt for k in ["bert", "roberta", "deberta", "xlm-roberta", "albert"]):
        # HF ：{q_proj,k_proj,v_proj,out_proj}  {query,key,value,dense}
        return ["q_proj", "v_proj", "query", "value"]
    # GPT-2： attn  q_proj/v_proj
    if "gpt2" in mt:
        return ["c_attn", "q_proj", "v_proj"]
    return ["q_proj", "v_proj"]

def _format_trainable_params(model) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    return {"total": int(total), "trainable": int(trainable), "pct": pct}

def _prepare_seqcls(base: str, num_labels: int = 2,
                    lora_cfg: Optional[TrainCfg] = None):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    cfg.num_labels = num_labels

    # 1) tokenizer  pad_token
    if tok.pad_token is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # 2) config  pad_token_id
    cfg.pad_token_id = pad_id

    # 3)
    mdl = AutoModelForSequenceClassification.from_pretrained(base, config=cfg, trust_remote_code=True)

    # 4)  tokenizer  token， embedding
    emb = mdl.get_input_embeddings()
    if hasattr(emb, "num_embeddings") and emb.num_embeddings < len(tok):
        mdl.resize_token_embeddings(len(tok))

    # 5) ： pad_token_id  config
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = pad_id

    # 6) （）generation_config  pad_token_id
    genconf = getattr(mdl, "generation_config", None)
    if genconf is None and GenerationConfig is not None:
        try:
            mdl.generation_config = GenerationConfig.from_model_config(mdl.config)
            genconf = mdl.generation_config
        except Exception:
            genconf = None
    if genconf is not None and getattr(genconf, "pad_token_id", None) is None:
        genconf.pad_token_id = pad_id
    mdl.config.id2label = {0: "human", 1: "ai"}
    mdl.config.label2id = {"human": 0, "ai": 1}

    is_peft = False
    # ========= ： LoRA =========
    if lora_cfg is not None and getattr(lora_cfg, "use_lora", False):
        if not _PEFT_OK:
            raise ImportError("未检测到 peft，请先安装：pip install peft")
        model_type = getattr(mdl.config, "model_type", "") or ""
        targets = lora_cfg.lora_target_modules or _infer_lora_targets(model_type)
        try:
            task_enum = getattr(TaskType, lora_cfg.lora_task_type)
        except Exception:
            task_enum = TaskType.SEQ_CLS
        lc = LoraConfig(
            r=int(lora_cfg.lora_r),
            lora_alpha=int(lora_cfg.lora_alpha),
            lora_dropout=float(lora_cfg.lora_dropout),
            bias=str(lora_cfg.lora_bias),
            target_modules=list(dict.fromkeys(targets)),
            task_type=task_enum,
        )
        mdl = get_peft_model(mdl, lc)
        is_peft = True
        stat = _format_trainable_params(mdl)
        print(f"[LoRA] enabled with targets={targets}; "
              f"trainable={stat['trainable']}/{stat['total']} ({stat['pct']:.2f}%)")

    return mdl, tok, is_peft

from datetime import datetime
# ----------  ----------
def _timestamp() -> str:
    return datetime.now().strftime(f"%Y%m%d-%H%M%S")

# ========= ：（） =========
def _merge_lora_and_export(base_id_or_dir: str, adapter_dir: str, export_dir: str) -> str:
    """
    重新加载：base + adapter -> merge_and_unload -> save_pretrained(export_dir)
    返回 export_dir。
    """
    if not _PEFT_OK:
        raise ImportError("需要 peft 以执行 LoRA 合并导出。")
    os.makedirs(export_dir, exist_ok=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_id_or_dir)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(export_dir)
    # tokenizer（ base_id_or_dir ）
    try:
        tok = AutoTokenizer.from_pretrained(base_id_or_dir, use_fast=True)
        tok.save_pretrained(export_dir)
    except Exception:
        pass
    try:
        with open(os.path.join(export_dir, "lora_merged_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"base": base_id_or_dir, "adapter_dir": adapter_dir}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return export_dir

def _maybe_dump_adapter_info(dir_path: str, base_id: str):
    try:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "adapter_info.json"), "w", encoding="utf-8") as f:
            json.dump({"base_model": base_id}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _train_seqcls_impl(
    *,
    model: str,
    dataset: Optional[str] = None,
    sample_k: Optional[int] = None,
    train_ratio: float = 8.0,
    val_ratio: float = 1.0,
    test_ratio: float = 0.0,
    output_dir: Optional[str] = None,
    max_length: int = 512,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    warmup_ratio: float = 0.06,
    grad_accum_steps: int = 1,
    fp16: bool = True,
    label_smoothing: float = 0.0,
    seed: int = 114514,
    device: Optional[str] = None,
    name: Optional[str] = None,
    outputs_prob: bool = True,
    # ========= ：LoRA （ TrainCfg ） =========
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    lora_target_modules: Optional[List[str]] = None,
    lora_task_type: str = "SEQ_CLS",
    lora_merge_on_save: bool = True,
    lora_export_tag: str = "merged",
    lora_save_adapter_copy: bool = True,
) -> Dict[str, Any]:
    _seed_everything(seed)
    torch.set_grad_enabled(True)

    # 1) load data
    if not dataset:
        examples, _ = load_dataset_unified(dataset="hc3", sample_k=5000, sample_seed=seed, group_cols=None)
    else:
        examples, _ = load_dataset_unified(dataset=dataset, sample_k=sample_k, sample_seed=seed, group_cols=None)
    train, val, test = _stratified_split(examples, train_ratio, val_ratio, test_ratio, seed=seed)
    train_ds, val_ds, test_ds = TLDS(train), (TLDS(val) if val else None), (TLDS(test) if test else None)

    # 2) resolve base & build (+ LoRA)
    base = _resolve_base(model)
    ts = _timestamp()

    # 3) cfg（ LoRA ）
    base_dir = output_dir or os.path.join(f"runs_finetune", os.path.basename(base))
    out_dir  = f"{base_dir}_{ts}"
    cfg = TrainCfg(
        base_model=base, output_dir=out_dir,
        max_length=max_length, lr=lr, weight_decay=weight_decay, epochs=epochs,
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio, grad_accum_steps=grad_accum_steps,
        fp16=fp16, label_smoothing=label_smoothing, seed=seed, device=device, name=name,
        use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lora_bias=lora_bias, lora_target_modules=lora_target_modules, lora_task_type=lora_task_type,
        lora_merge_on_save=lora_merge_on_save, lora_export_tag=lora_export_tag, outputs_prob=outputs_prob,
        lora_save_adapter_copy=lora_save_adapter_copy,
    )

    mdl, tok, is_peft = _prepare_seqcls(base, num_labels=2, lora_cfg=cfg)

    # 4)
    result = train_model(
        model=mdl,
        tokenizer=tok,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg,
        dataset_spec=dataset,
    )

    # （best ； last）
    model_dir = result.get("best_dir") or result.get("last_dir") or cfg.output_dir

    # ========= ： LoRA，“” =========
    exported_dir = None
    if is_peft and cfg.lora_merge_on_save:
        tag = cfg.lora_export_tag or "merged"
        parent = os.path.dirname(model_dir.rstrip("/"))
        merged_dir = os.path.join(parent, f"{os.path.basename(model_dir)}-{tag}")
        try:
            exported_dir = _merge_lora_and_export(base_id_or_dir=base, adapter_dir=model_dir, export_dir=merged_dir)
            print(f"[LoRA] merged weights exported to: {exported_dir}")
            # ：（ model_dir ）， base_model
            if cfg.lora_save_adapter_copy:
                _maybe_dump_adapter_info(model_dir, base)
            # ： model_dir “”
            model_dir = exported_dir
        except Exception as e:
            print(f"[LoRA] merge export failed: {e}. Falling back to adapter-only dir: {model_dir}")
            if cfg.lora_save_adapter_copy:
                _maybe_dump_adapter_info(model_dir, base)

    return {
        "model_dir": model_dir,
        "best_val_acc": result.get("best_val_acc", None),
        "split_sizes": {"train": len(train_ds), "val": len(val_ds) if val_ds else 0, "test": len(test_ds) if test_ds else 0},
        "test_examples": test,
        "history": result.get("history", []),
        "step_indices": result.get("step_indices", []),
        "step_losses": result.get("step_losses", []),
        "artifacts": result.get("artifacts", {}),
        "timing": result.get("timing", {}),
        "memory": result.get("memory", {}),
        "config": cfg.__dict__,
        # （）
        "adapter_dir": None if not is_peft else (result.get("best_dir") or result.get("last_dir")),
        "merged_dir": exported_dir,
        "is_lora": bool(is_peft),
        "base_model": base,
    }

# base_model  model；
_ALLOWED_KEYS = {
    "model", "dataset", "sample_k",
    "train_ratio", "val_ratio", "test_ratio",
    "output_dir", "max_length", "lr", "weight_decay", "epochs",
    "train_batch_size", "eval_batch_size", "warmup_ratio", "grad_accum_steps",
    "fp16", "label_smoothing", "seed", "device", "name",
    # ========= ：LoRA  =========
    "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_bias",
    "lora_target_modules", "lora_task_type",
    "lora_merge_on_save", "lora_export_tag", "lora_save_adapter_copy",
    "outputs_prob",
    "disable_calibration",
    "force_runner_calibration",
    "auto_calibrate",
}

def _clean_kwargs(kwargs: dict) -> dict:
    base_override = kwargs.pop("base_model", None)
    if base_override is not None and str(base_override).strip():
        kwargs["model"] = base_override
    return {k: v for k, v in kwargs.items() if k in _ALLOWED_KEYS}

# Encoder/MLM （）
@register_train(
    "xlm-roberta-base", "xlm-roberta-large",
    "roberta-base", "roberta-large",
    "mbert", "bert-base-uncased", "bert-base-cased",
    "albert-base", "albert-large",
    "deberta-v3-base", "deberta-v3-large",
    "mdeberta-base", "mdeberta-large",
)
def train_mlm_family(**kwargs) -> Dict[str, Any]:
    return _train_seqcls_impl(**_clean_kwargs(kwargs))

# Decoder-only（GPT2）
@register_train("gpt2")
def train_gpt2(**kwargs) -> Dict[str, Any]:
    return _train_seqcls_impl(**_clean_kwargs(kwargs))

# ：/HF id
@register_train("hf")
def train_hf_any(**kwargs) -> Dict[str, Any]:
    return _train_seqcls_impl(**_clean_kwargs(kwargs))
