# mgt_eval/detectors/pretrained/pretrained.py
"""
Pretrained detector：
  - 自动适配多类模型：
      * DeTeCtive（新增）：识别 best/last 目录下 encoder_hf/ 与 model_classifier_*.pth，重建分类头进行推理
      * CoCo（检测含 coco_config.json 的目录并加载 CoCoGraphModel）
      * SequenceClassification：直接输出“AI 类别”的概率
      * MaskedLM（BERT/Roberta 等）：用伪似然 (PLL) 近似平均 NLL
      * CausalLM（GPT/LLaMA/Qwen 等）：按自回归平均 NLL
  - 当 model_path 指向 .pt/.pth/.bin/.ckpt/.safetensors 等**权重文件**时：
      * 不再从该路径读取 config；
      * 而是用 tokenizer_path（或 ckpt_base）构建 SeqCls 模型（num_labels=ckpt_num_labels），
        再把 checkpoint 的 state_dict 加载（strict=False，自动清理前缀）。
  - 新增：LoRA 适配器目录自动识别与加载（AutoPeftModel* 优先；回退 PeftModel.from_pretrained），
          安全加载包装（优先 safetensors；若 torch<2.6 且仅 .bin 则报可操作错误），
          因果族识别仅对明确的 CausalLM 模型尝试因果头，
          对 MLM 的 PLL 评分过程增加 tqdm 进度条。

  - 新增：嵌入 + KNN 推理管线（可选）
      * TextEmbeddingModel（平均/CLS 池化，支持 T5、BGE/MXBai 的 CLS 优先）
      * FAISS Indexer（CPU/GPU 自动回退）
      * 统一数据集加载（依赖 mgt_eval.data_utils.load.load_dataset_unified）
      * 数据/数据库缓存与复用（用户目录）
      * 支持返回 @K 上的分类指标与“投票比例概率”（可用于 AUROC；若无概率则跳过）
"""

from typing import List, Optional, Dict, Any, Tuple, Iterable
import os
import io
import json
import math
import pickle
import hashlib
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Token indices sequence length is longer than the specified maximum sequence length for this model",
    category=UserWarning,
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from packaging.version import parse as _V
from tqdm.auto import tqdm
from .src.index import Indexer as _FaissIndexer

from ..base import DetectorBase
from ..registry import register

# ------------ 数据统一加载 ------------
try:
    from mgt_eval.data_utils.load import load_dataset_unified
except Exception as _e_load_unified:
    load_dataset_unified = None
    _LOAD_UNIFIED_IMPORT_ERROR = _e_load_unified

# --- 可选：PRDetect shim ---
try:
    from mgt_eval.detectors.finetuned.prdetect import PRDetectDetector as _PRDShim
except Exception:
    _PRDShim = None

# --- 可选：CoCo 支持（Graph-Enhanced）---
CoCoGraphModel = None
CoCoConfig = None
CoCoGraphDataset = None
coco_collate = None
_coco_resolve_base = None
_COCO_IMPORT_ERROR = None


def _lazy_import_coco():
    """
    延迟导入 CoCo 相关组件，避免在模块导入阶段与 evaluator / registry 形成循环依赖。
    只在第一次真正需要 CoCo 模型时导入一次。
    """
    global CoCoGraphModel, CoCoConfig, CoCoGraphDataset, coco_collate, _coco_resolve_base, _COCO_IMPORT_ERROR

    # 已经导入过就直接返回
    if CoCoGraphModel is not None:
        return

    try:
        from mgt_eval.detectors.finetuned.coco import (
            CoCoGraphModel as _CoCoGraphModel,
            CoCoConfig as _CoCoConfig,
            CoCoGraphDataset as _CoCoGraphDataset,
            coco_collate as _coco_collate,
            _resolve_base as _coco_resolve_base_fn,
        )

        CoCoGraphModel = _CoCoGraphModel
        CoCoConfig = _CoCoConfig
        CoCoGraphDataset = _CoCoGraphDataset
        coco_collate = _coco_collate
        _coco_resolve_base = _coco_resolve_base_fn
        _COCO_IMPORT_ERROR = None
    except Exception as e:
        CoCoGraphModel = None
        CoCoConfig = None
        CoCoGraphDataset = None
        coco_collate = None
        _coco_resolve_base = None
        _COCO_IMPORT_ERROR = e

# --- 可选：DeTeCtive 支持（新增）---
_DETECTIVE_IMPORT_ERROR = None
_SimCLR_MultiLevel = None
try:
    # 优先使用你的实现，保证参数/层命名完全一致以便直接 load_state_dict
    from mgt_eval.detectors.finetuned.detective import SimCLR_MultiLevel as _SimCLR_MultiLevel
except Exception as _e_detective:
    _DETECTIVE_IMPORT_ERROR = _e_detective
    _SimCLR_MultiLevel = None  # fallback 会用一个轻量兼容类

# --- 可选：PEFT (LoRA) ---
_PEFT_IMPORT_ERROR = None
try:
    from peft import PeftModel
    try:
        from peft import (
            AutoPeftModelForCausalLM,
            AutoPeftModelForMaskedLM,
            AutoPeftModelForSequenceClassification,
        )
    except Exception as _e_auto:
        AutoPeftModelForCausalLM = None
        AutoPeftModelForMaskedLM = None
        AutoPeftModelForSequenceClassification = None
        _PEFT_IMPORT_ERROR = _e_auto
except Exception as _e_peft:
    PeftModel = None
    AutoPeftModelForCausalLM = None
    AutoPeftModelForMaskedLM = None
    AutoPeftModelForSequenceClassification = None
    _PEFT_IMPORT_ERROR = _e_peft

# ------------ Transformers 主体 ------------
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)


# ============================================================================
#  一些通用小工具
# ============================================================================
def _is_prdetect_dir(path: str) -> bool:
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    cands = [
        os.path.join(path, "prdetect_gcn.pt"),
        os.path.join(path, "best", "prdetect_gcn.pt"),
        os.path.join(path, "last", "prdetect_gcn.pt"),
    ]
    return any(os.path.isfile(p) for p in cands)


def _is_ckpt_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[-1].lower()
    return ext in {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}


def _is_lora_dir(path: str) -> bool:
    """简单判定 LoRA 适配器目录：含 adapter_config.json 或 adapter_model.(safetensors|bin) 即认定"""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    return any(
        os.path.isfile(os.path.join(path, fn))
        for fn in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    )


def _hash_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def _basename_like_id(p: str) -> str:
    """将本地路径或 HF id 统一取最后一段作为“名称”"""
    if not p:
        return "unknown"
    p = p.strip("/").strip("\\")
    base = os.path.basename(p)
    if "/" in p and base == "":
        base = p.split("/")[-1]
    return base or p


def _path_has_safetensors(model_path: str) -> bool:
    if not model_path:
        return False
    if os.path.isdir(model_path):
        try:
            for fn in os.listdir(model_path):
                if fn.endswith(".safetensors"):
                    return True
        except Exception:
            pass
        return False
    return str(model_path).endswith(".safetensors")


def _torch_too_old_for_bin() -> bool:
    try:
        ver = _V(torch.__version__.split("+")[0])
        return ver < _V("2.6")
    except Exception:
        return False


def _load_state_dict(ckpt_path: str, device: str = "cpu"):
    # 支持 torch 与 safetensors
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_sft
        sd = load_sft(ckpt_path, device=device)
    else:
        sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # 清理常见前缀
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    return new_sd


def _find_detective_train_args(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    base = path if os.path.isdir(path) else os.path.dirname(path)
    for _ in range(3):
        cand = os.path.join(base, "train_args.json")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None


def _read_detective_embedding_model(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = args.get("embedding_model")
    if not model:
        return None
    return str(model)


def _from_pretrained_safe(factory, model_id_or_path: str, **kwargs):
    """
    统一安全加载：
      - 若本地目录不存在 *.safetensors 且 torch>=2.6，则直接允许加载 .bin（use_safetensors=False）
      - 否则先尝试 use_safetensors=True；如果因为缺少 safetensors 报 OSError，再自动回退到 .bin
      - 若 torch<2.6 且只能 .bin，则给出清晰可操作错误
    """
    kw = dict(kwargs or {})

    local_dir = os.path.isdir(model_id_or_path)
    has_sft = _path_has_safetensors(model_id_or_path) if local_dir else False

    if local_dir and (not has_sft):
        if _torch_too_old_for_bin():
            raise RuntimeError(
                "[pretrained] 检测到 PyTorch 版本 < 2.6 且当前本地目录无 *.safetensors，"
                "Transformers 出于安全原因（CVE-2025-32434）禁止加载 .bin。\n"
                "解决方案：升级 PyTorch 至 2.6+，或将权重转换为 safetensors。"
            )
        kw.setdefault("use_safetensors", False)
        return factory.from_pretrained(model_id_or_path, **kw)

    kw_try_sft = dict(kw)
    kw_try_sft.setdefault("use_safetensors", True)
    try:
        return factory.from_pretrained(model_id_or_path, **kw_try_sft)
    except OSError as e:
        msg = str(e).lower()
        missing_sft = ("no file named" in msg and "safetensors" in msg) or ("safetensors" in msg and "not found" in msg)
        if missing_sft:
            if _torch_too_old_for_bin():
                raise RuntimeError(
                    "[pretrained] 未找到 safetensors 且当前 PyTorch < 2.6，无法安全回退到 .bin。\n"
                    "请升级 PyTorch 至 2.6+ 或提供 safetensors 权重。"
                ) from e
            kw_try_bin = dict(kw)
            kw_try_bin["use_safetensors"] = False
            return factory.from_pretrained(model_id_or_path, **kw_try_bin)
        raise
    except ValueError as e:
        msg = str(e)
        needs_new_torch = ("upgrade torch to at least v2.6" in msg) or ("require users to upgrade torch to at least v2.6" in msg)
        if needs_new_torch:
            has_sft_any = _path_has_safetensors(model_id_or_path) if os.path.isdir(model_id_or_path) else False
            if _torch_too_old_for_bin() and not has_sft_any:
                raise RuntimeError(
                    "[pretrained] 仅 .bin 权重且 PyTorch < 2.6。出于安全原因被拒绝加载。\n"
                    "请升级 PyTorch 至 2.6+ 或将权重转换为 safetensors。"
                ) from e
        raise


# ============== CoCo 目录定位 ==============
def _locate_coco_dir(path: str) -> Optional[str]:
    """
    若 path 或其 best/last 子目录含有 coco_config.json，则返回该目录路径；否则返回 None。
    """
    if not isinstance(path, str):
        return None
    cands = []
    if os.path.isdir(path):
        cands.append(path)
        cands.append(os.path.join(path, "best"))
        cands.append(os.path.join(path, "last"))
    # 若直接给到文件，尝试其父目录
    elif os.path.isfile(path):
        par = os.path.dirname(path)
        if par:
            cands.append(par)
            cands.append(os.path.join(par, "best"))
            cands.append(os.path.join(par, "last"))
    for d in cands:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "coco_config.json")):
            # 进一步确认至少有模型权重与 tokenizer 文件之一
            has_weight = os.path.isfile(os.path.join(d, "pytorch_model.bin")) or any(
                fn.endswith(".safetensors") for fn in os.listdir(d)
            )
            has_tok = any(os.path.isfile(os.path.join(d, fn)) for fn in ["tokenizer.json", "vocab.json", "merges.txt"])
            if has_weight and has_tok:
                return d
    return None


# ================== DeTeCtive 轻量回退（当无法 import 你的类时） ==================
class _MinimalDetectiveForInfer(torch.nn.Module):
    """
    当无法 import 你的 SimCLR_MultiLevel 时，用此最小推理兼容类。
    结构与参数命名对齐：
      - encoder_hf -> AutoModel
      - projection head: encoder.proj.[0].weight/bias, [2].weight/bias
      - classifier head: cls_label.weight/bias
    仅用于前向得到 logits_label（无需训练）。
    """
    def __init__(self, encoder_dir: str, proj_dim: int):
        super().__init__()
        cfg = AutoConfig.from_pretrained(encoder_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir, use_fast=True)
        self.encoder_hf = AutoModel.from_pretrained(encoder_dir, config=cfg, trust_remote_code=True)
        hid = getattr(self.encoder_hf.config, "hidden_size", 768)
        # 投影头（Linear-GELU-Linear）
        self.encoder = torch.nn.Module()
        self.encoder.model = self.encoder_hf
        self.encoder.proj = torch.nn.Sequential(
            torch.nn.Linear(hid, hid),
            torch.nn.GELU(),
            torch.nn.Linear(hid, proj_dim),
        )
        # 分类头
        self.cls_label = torch.nn.Linear(proj_dim, 2)

        # pad token 兜底
        tok = self.tokenizer
        if tok.pad_token is None:
            if getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                self.encoder_hf.resize_token_embeddings(len(tok))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        mask = attention_mask.unsqueeze(-1).float()
        x = (out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        z = self.encoder.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(input_ids, attention_mask)
        logits = self.cls_label(z)
        return {"z": z, "logits_label": logits}


# ================== TextEmbeddingModel（用于嵌入+KNN） ==================
class TextEmbeddingModel(torch.nn.Module):
    """
    轻量通用文本编码器：AutoModel + 池化（average/cls）
    - 对 BGE/MXBai 默认 'cls'；对 T5 自动注入 decoder_input_ids=[0]
    - 可选加载你外部训练好的 state_dict（会自动清理'model.'前缀）
    """
    def __init__(self, model_name: str, output_hidden_states: bool = False):
        super().__init__()
        self.model_name = model_name
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                try:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                except Exception:
                    pass

    def _pooling(self, model_output, attention_mask, use_pooling='average', hidden_states=False):
        if hidden_states:
            model_output.masked_fill_(~attention_mask[None, ..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:, :, 0]
            emb = emb.permute(1, 0, 2)
        else:
            model_output.masked_fill_(~attention_mask[..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:, 0]
        return emb

    def forward(self, encoded_batch: Dict[str, torch.Tensor], use_pooling='average', hidden_states=False):
        if "t5" in self.model_name.lower():
            input_ids = encoded_batch["input_ids"]
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            out = self.model(**encoded_batch, decoder_input_ids=decoder_input_ids)
        else:
            out = self.model(**encoded_batch)

        if "bge" in self.model_name.lower() or "mxbai" in self.model_name.lower():
            use_pooling = "cls"

        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            if hidden_states:
                feat = torch.stack(out["hidden_states"], dim=0)
            else:
                feat = out["last_hidden_state"]
        else:
            feat = out

        emb = self._pooling(feat, encoded_batch["attention_mask"], use_pooling, hidden_states)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class _DetectiveClassificationHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_dim, in_dim // 4)
        self.dense2 = torch.nn.Linear(in_dim // 4, in_dim // 16)
        self.out_proj = torch.nn.Linear(in_dim // 16, out_dim)

        torch.nn.init.xavier_uniform_(self.dense1.weight)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.normal_(self.dense1.bias, std=1e-6)
        torch.nn.init.normal_(self.dense2.bias, std=1e-6)
        torch.nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        return self.out_proj(x)


class _MinimalDetectiveClassifier(torch.nn.Module):
    """
    兼容当前 detective.py 的分类器命名（model + classifier.*）。
    用于没有 encoder_hf 目录时的推理加载。
    """
    def __init__(self, encoder_name: str, proj_dim: int, classifier_dim: int = 2):
        super().__init__()
        self.model = TextEmbeddingModel(encoder_name)
        self.classifier = _DetectiveClassificationHead(proj_dim, classifier_dim)

    def get_encoder(self) -> TextEmbeddingModel:
        return self.model

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.model(batch)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoded_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if encoded_batch is None:
            if input_ids is None or attention_mask is None:
                raise ValueError("input_ids/attention_mask or encoded_batch must be provided.")
            encoded_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        z = self.model(encoded_batch)
        logits = self.classifier(z)
        return {"z": z, "logits_label": logits}


# ================== 简易指标（与 utils.compute_metrics 等价接口） ==================
def _compute_basic_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """
    返回：
        - human_rec（对 label=0 的召回）
        - machine_rec（对 label=1 的召回）
        - avg_rec = (human_rec + machine_rec)/2
        - acc / precision / recall / f1（以“1=机器”为正类）
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    p1 = int((y_true == 1).sum())
    p0 = int((y_true == 0).sum())

    machine_rec = tp / p1 if p1 > 0 else 0.0
    human_rec = tn / p0 if p0 > 0 else 0.0
    avg_rec = 0.5 * (human_rec + machine_rec)

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = machine_rec
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(
        human_rec=float(human_rec),
        machine_rec=float(machine_rec),
        avg_rec=float(avg_rec),
        acc=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    """
    给定实数数组，返回平均名次（处理并列）。仅用于 ROC AUC 的无依赖实现。
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    # 处理并列：相同值的 rank 取平均
    vals = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    无三方依赖的二分类 ROC AUC（Mann–Whitney U 公式）
      AUC = (sum_{pos} rank(score_pos) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    当正负类或分数无区分度（例如全常数）时返回 None。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape[0] != y_score.shape[0]:
        return None
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # 若分数几乎全相等 -> 无法定义
    if np.allclose(y_score, y_score[0]):
        return None
    ranks = _rankdata_avg_ties(y_score)
    sum_pos = float(ranks[y_true == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ============================================================================
#  PretrainedDetector 主体（原能力 + 嵌入KNN）
# ============================================================================
@register("pretrained")
class PretrainedDetector(DetectorBase):
    """
    Args:
        model_path: HF 模型 id/目录，或权重文件 (.pt/.pth/.bin/.ckpt/.safetensors) 或 LoRA 目录，或 CoCo 目录，或 DeTeCtive 目录
        tokenizer_path: HF tokenizer id/目录（默认同 model_path）
        name: 预训练/基座模型名（用于元数据与输出文件名）
        device: "cuda" 或 "cpu"（默认自动）
        max_length: 截断/填充长度
        fp16: 在 CUDA 上使用半精度
        pll_stride: MLM 伪似然的并行掩码步长
        ai_label_id: 序列分类/CoCo/Detective 时“AI 类别”的 id（默认 1）
        ckpt_num_labels: 当以 checkpoint 文件加载 SeqCls 模型时使用的类别数（默认 2）
        ckpt_base: 当以 checkpoint 文件加载或 LoRA 加载时用于构建 Config/基座的模型 id/目录（默认用 tokenizer_path）
        show_progress: 在 MLM PLL 评分过程中显示进度条（默认 True）
        detector_type: 标注类型（默认 "Model-based"；CoCo/PRDetect/Detective 会覆盖）

    嵌入+KNN（仅当你显式调用 embedding_knn_infer(...) 时使用）：
        - 见 embedding_knn_infer 方法 docstring
    """

    # --------------------- 初始化 ---------------------
    def __init__(
        self,
        model_path: str = "gpt2",
        tokenizer_path: Optional[str] = None,
        name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        fp16: bool = True,
        pll_stride: int = 64,
        ai_label_id: int = 1,
        ckpt_num_labels: int = 2,
        ckpt_base: Optional[str] = None,
        show_progress: bool = True,
        detector_type: Optional[str] = "Model-based",
        # --- NEW: Embedding+KNN mode (unified evaluator compatible) ---
        knn_train_dataset: Optional[str] = None,  # 指定则启用 KNN 模式
        knn_embedding_model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        knn_embedding_ckpt_path: Optional[str] = None,
        knn_pooling: str = "average",
        knn_K: int = 51,
        knn_backend: str = "torch",  # 'torch' | 'faiss'
        knn_cache_root: Optional[str] = None,
        knn_index_name: Optional[str] = None,
        knn_reuse_database: bool = True,
        knn_save_database: bool = True,
        knn_use_gpu_index: bool = True,
        knn_db_block: int = 65536,      # torch backend: database block
        knn_index_batch_size: int = 8,  # faiss backend: query batch for indexer
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            name=name,
            device=device,
            max_length=max_length,
            fp16=fp16,
            pll_stride=pll_stride,
            ai_label_id=ai_label_id,
            ckpt_num_labels=ckpt_num_labels,
            ckpt_base=ckpt_base,
            show_progress=show_progress,
            **kwargs,
        )
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.ckpt_base = ckpt_base or self.tokenizer_path
        self._prdetect_shim = None
        self.detector_type = detector_type or "Model-based"
        # --- NEW: KNN configs ---
        self.knn_train_dataset = knn_train_dataset
        self.knn_embedding_model_name = knn_embedding_model_name
        self.knn_embedding_ckpt_path = knn_embedding_ckpt_path
        self.knn_pooling = knn_pooling
        self.knn_K = int(knn_K)
        self.knn_backend = str(knn_backend or "torch").lower()
        self.knn_cache_root = knn_cache_root
        self.knn_index_name = knn_index_name
        self.knn_reuse_database = bool(knn_reuse_database)
        self.knn_save_database = bool(knn_save_database)
        self.knn_use_gpu_index = bool(knn_use_gpu_index)
        self.knn_db_block = int(knn_db_block)
        self.knn_index_batch_size = int(knn_index_batch_size)

        # --- NEW: KNN runtime states ---
        self._knn_enc = None                      # TextEmbeddingModel
        self._knn_train_tensor = None             # torch.Tensor [N, D]  (torch backend)
        self._knn_train_labels_tensor = None      # torch.LongTensor [N] (torch backend)
        self._knn_faiss_indexer = None            # _FaissIndexer (faiss backend)
        self._knn_faiss_label_dict = None         # Dict[int, int]
        self._knn_mm_device = None                # 'cuda'|'cpu'
        # 名称推断
        if name:
            inferred = name
        elif _is_ckpt_file(model_path):
            base = _basename_like_id(self.ckpt_base)
            if base and base.lower() != _basename_like_id(model_path).lower():
                inferred = base
            else:
                inferred = os.path.splitext(os.path.basename(model_path))[0]
        else:
            inferred = _basename_like_id(model_path)

        if (self.knn_train_dataset is not None) and (not name):
            self.name = f"knn_{_basename_like_id(self.knn_embedding_model_name)}_K{self.knn_K}"
            self.DETECTOR_NAME = self.name

        self.name = inferred
        self.DETECTOR_NAME = self.name

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.fp16 = bool(fp16)
        self.pll_stride = max(1, int(pll_stride))
        self.ai_label_id = int(ai_label_id)
        self.ckpt_num_labels = int(ckpt_num_labels)
        self.show_progress = bool(show_progress)

        self._config = None
        self._tokenizer = None
        self._model = None
        self._kind = None  # "detective" | "coco" | "causal" | "mlm" | "seqcls" | "prdetect"
        # --- NEW: 标记本 detector 是否直接输出概率 ---
        # evaluator 会打印 outputs_prob=... 并据此决定是否需要 prob mapping / calibration
        self.outputs_prob: bool = False

        # --- NEW: 强制禁用任何校准（即使 evaluator 打算校准） ---
        # 需要 evaluator 侧尊重该标志；你也可以在 runner.py 里兜底
        self.disable_calibration: bool = False
    # --------------------- Load ---------------------
    def load(self):
        if self.knn_train_dataset is not None:
            self._setup_knn_database()
            self._kind = "knn"
            self.detector_type = "Model-based"
            self.is_loaded = True
            self.outputs_prob = True
            self.disable_calibration = True
            return
        # ---------- PRDetect shim ----------
        if _PRDShim is not None and _is_prdetect_dir(self.model_path):
            roberta_path = self.tokenizer_path or self.model_path
            self._prdetect_shim = _PRDShim(
                model_path=self.model_path,
                roberta_path=roberta_path,
                max_length=self.max_length,
                device=self.device,
            )
            self._prdetect_shim.load()
            self._kind = "prdetect"
            self.detector_type = "Model-based"
            self.is_loaded = True
            return

        # ---------- CoCo 目录 ----------
        coco_dir = _locate_coco_dir(self.model_path)
        if coco_dir is not None:
            _lazy_import_coco()
            if any(x is None for x in [CoCoGraphModel, CoCoConfig, CoCoGraphDataset, coco_collate, _coco_resolve_base]):
                raise RuntimeError(
                    "[pretrained] 识别到 CoCo 目录，但导入 CoCo 组件失败，请确认 "
                    "`mgt_eval.detectors.finetuned.coco` 可用。\n"
                    f"具体异常：{repr(_COCO_IMPORT_ERROR)}"
                )
            with open(os.path.join(coco_dir, "coco_config.json"), "r", encoding="utf-8") as f:
                coco_cfg = CoCoConfig(**json.load(f))
            self._tokenizer = AutoTokenizer.from_pretrained(coco_dir, use_fast=True)
            if self._tokenizer.pad_token_id is None:
                if getattr(self._tokenizer, "eos_token", None) is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
                else:
                    self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
            try:
                self._tokenizer.padding_side = "right"
            except Exception:
                pass

            base = _coco_resolve_base(coco_cfg.base_model, "roberta-base")
            self._model = CoCoGraphModel(base, coco_cfg, num_labels=2)
            weight_path = os.path.join(coco_dir, "pytorch_model.bin")
            if not os.path.isfile(weight_path):
                sft_candidates = [fn for fn in os.listdir(coco_dir) if fn.endswith(".safetensors")]
                if not sft_candidates:
                    raise FileNotFoundError(
                        f"[pretrained] 在 CoCo 目录未找到权重文件：{weight_path} 或 *.safetensors"
                    )
                weight_path = os.path.join(coco_dir, sft_candidates[0])
            state_dict = _load_state_dict(weight_path, device="cpu")
            self._model.load_state_dict(state_dict, strict=True)

            try:
                if getattr(self._model.encoder.config, "pad_token_id", None) is None:
                    self._model.encoder.config.pad_token_id = self._tokenizer.pad_token_id
            except Exception:
                pass

            if self.fp16 and self.device.startswith("cuda"):
                try:
                    self._model.half()
                except Exception:
                    pass
            self._model.to(self.device)
            self._model.eval()

            self._config = None
            self._kind = "coco"
            self.detector_type = "Model-based"
            self.is_loaded = True
            self.outputs_prob = True
            self.disable_calibration = True
            return

        # ---------- DeTeCtive 目录（新增） ----------
        detective_dir = self._is_detective_dir(self.model_path)
        if detective_dir is not None:
            enc_dir = os.path.join(detective_dir, "encoder_hf")
            if not os.path.isdir(enc_dir):
                parent_enc = os.path.join(os.path.dirname(detective_dir), "encoder_hf")
                if os.path.isdir(parent_enc):
                    enc_dir = parent_enc
            if not os.path.isdir(enc_dir):
                args_path = _find_detective_train_args(detective_dir)
                embed_model = _read_detective_embedding_model(args_path)
                if embed_model:
                    enc_dir = embed_model
                else:
                    raise RuntimeError(
                        "[pretrained/DeTeCtive] 未找到 encoder_hf，且 train_args.json 中缺少 embedding_model。"
                    )
            ckpt_path = (
                os.path.join(detective_dir, "model_classifier_best.pth")
                if os.path.isfile(os.path.join(detective_dir, "model_classifier_best.pth"))
                else os.path.join(detective_dir, "model_classifier_last.pth")
            )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"[pretrained/DeTeCtive] 未找到分类器权重：{ckpt_path}"
                )

            sd = _load_state_dict(ckpt_path, device="cpu")
            if any(k.startswith("model.") for k in sd) and not any(k.startswith("model.model.") for k in sd):
                sd = {("model." + k if k.startswith("model.") else k): v for k, v in sd.items()}
            has_cls_label = "cls_label.weight" in sd
            has_classifier = "classifier.dense1.weight" in sd
            if has_cls_label:
                proj_dim = int(sd["cls_label.weight"].shape[1])
                if _SimCLR_MultiLevel is not None:
                    mdl = _SimCLR_MultiLevel(
                        base_name=enc_dir,
                        proj_dim=proj_dim,
                        temperature=0.07,
                        num_label_classes=2,
                        num_model_classes=int(sd.get("cls_model.weight", torch.empty(0)).shape[0]) if ("cls_model.weight" in sd) else 1,
                        num_set_classes=int(sd.get("cls_set.weight", torch.empty(0)).shape[0]) if ("cls_set.weight" in sd) else 1,
                        a=0.0, b=0.0, c=0.0, d=1.0,
                        one_loss=False,
                        only_classifier=True,
                        freeze_embedding_layer=False,
                    )
                else:
                    mdl = _MinimalDetectiveForInfer(enc_dir, proj_dim)
            elif has_classifier:
                proj_dim = int(sd["classifier.dense1.weight"].shape[1])
                cls_dim = int(sd.get("classifier.out_proj.weight", torch.empty(2, 0)).shape[0]) or 2
                mdl = _MinimalDetectiveClassifier(enc_dir, proj_dim, classifier_dim=cls_dim)
            else:
                raise RuntimeError(
                    "[pretrained/DeTeCtive] 权重中缺少 cls_label.weight 或 classifier.dense1.weight，无法恢复分类器。"
                )

            mdl.load_state_dict(sd, strict=False)
            self._tokenizer = AutoTokenizer.from_pretrained(enc_dir, use_fast=True)
            try:
                if getattr(mdl, "encoder", None) and getattr(mdl.encoder, "model", None):
                    if getattr(mdl.encoder.model.config, "pad_token_id", None) is None:
                        mdl.encoder.model.config.pad_token_id = self._tokenizer.pad_token_id
            except Exception:
                pass

            if self.fp16 and self.device.startswith("cuda"):
                try:
                    mdl.half()
                except Exception:
                    pass
            mdl.to(self.device).eval()

            self._model = mdl
            self._config = None
            self._kind = "detective"
            self.detector_type = "Model-based"
            self.is_loaded = True
            self.outputs_prob = True
            self.disable_calibration = True
            self.ai_label_id = 1
            return

        # ---------- 常规分支（LoRA / SeqCls / MLM / Causal） ----------
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            if getattr(self._tokenizer, "eos_token", None) is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
            else:
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        try:
            self._tokenizer.padding_side = "right"
        except Exception:
            pass

        # ---------- LoRA 适配器 ----------
        if _is_lora_dir(self.model_path):
            if (AutoPeftModelForCausalLM is None) and (PeftModel is None):
                raise RuntimeError(
                    "[pretrained] 检测到 LoRA 目录，但导入 peft 失败。\n"
                    f"具体异常：{repr(_PEFT_IMPORT_ERROR)}\n"
                    "请确认 `pip show peft` 在当前环境可见，并尝试升级：\n"
                    "  pip install -U peft transformers accelerate\n"
                )
            base_id = self.ckpt_base or self.tokenizer_path
            if not base_id:
                raise RuntimeError(
                    "[pretrained] LoRA 加载需要指定 ckpt_base 或 tokenizer_path 作为基座模型。"
                )
            try:
                base_cfg = AutoConfig.from_pretrained(base_id)
            except Exception as e:
                raise RuntimeError(
                    f"[pretrained] 无法从基座 '{base_id}' 读取 config，请确认其为有效的 HF 目录/ID（含 config.json）。"
                ) from e

            def _has_arch_suffix_base(suffix: str) -> bool:
                archs = set(getattr(base_cfg, "architectures", []) or [])
                return any(a.endswith(suffix) for a in archs)

            model_type = (getattr(base_cfg, "model_type", "") or "").lower()
            CAUSAL_FAMILIES = {
                "gpt2", "gptj", "gpt_neo", "gpt-neox", "mpt", "llama", "falcon",
                "qwen", "qwen2", "phi", "mistral", "mixtral", "gemma", "xlnet"
            }

            def _autopeft_try():
                if AutoPeftModelForSequenceClassification and (
                    _has_arch_suffix_base("ForSequenceClassification") or getattr(base_cfg, "num_labels", None)
                ):
                    return AutoPeftModelForSequenceClassification.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "seqcls"
                if AutoPeftModelForMaskedLM and model_type in {
                    "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                    "electra", "albert", "camembert", "distilbert", "flaubert",
                    "bart", "t5"
                }:
                    return AutoPeftModelForMaskedLM.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "mlm"
                if AutoPeftModelForCausalLM and model_type in CAUSAL_FAMILIES:
                    return AutoPeftModelForCausalLM.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "causal"
                return None, None

            model, kind = _autopeft_try()
            if model is None:
                if _has_arch_suffix_base("ForSequenceClassification") or getattr(base_cfg, "num_labels", None):
                    base_model = _from_pretrained_safe(AutoModelForSequenceClassification, base_id)
                    kind = "seqcls"
                elif model_type in {
                    "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                    "electra", "albert", "camembert", "distilbert", "flaubert",
                    "bart", "t5"
                }:
                    base_model = _from_pretrained_safe(AutoModelForMaskedLM, base_id)
                    kind = "mlm"
                elif model_type in CAUSAL_FAMILIES:
                    base_model = _from_pretrained_safe(AutoModelForCausalLM, base_id)
                    kind = "causal"
                else:
                    try:
                        base_model = _from_pretrained_safe(AutoModelForSequenceClassification, base_id)
                        kind = "seqcls"
                    except Exception:
                        base_model = _from_pretrained_safe(AutoModelForMaskedLM, base_id)
                        kind = "mlm"

                if PeftModel is None:
                    raise RuntimeError(
                        "[pretrained] 需要 peft.PeftModel 来装载 LoRA 适配器，请 `pip install peft`。"
                    )
                model = PeftModel.from_pretrained(base_model, self.model_path)

            self._config = base_cfg
            self._model = model
            self._kind = kind

        # ---------- checkpoint 权重文件 ----------
        elif _is_ckpt_file(self.model_path):
            try:
                self._config = AutoConfig.from_pretrained(self.ckpt_base)
            except Exception as e:
                raise RuntimeError(
                    f"无法从 ckpt_base='{self.ckpt_base}' 读取 Config。"
                    f" 请将 ckpt_base 指向一个包含 config.json 的 HF 目录或模型 id，"
                    f"通常可与 tokenizer_path 一致。"
                ) from e

            self._config.num_labels = getattr(self._config, "num_labels", self.ckpt_num_labels)
            if self._config.num_labels != self.ckpt_num_labels:
                self._config.num_labels = self.ckpt_num_labels

            self._model = AutoModelForSequenceClassification.from_config(self._config)
            sd = _load_state_dict(self.model_path, device="cpu")
            self._model.load_state_dict(sd, strict=False)
            self._kind = "seqcls"

        # ---------- 常规 HF 目录/id ----------
        else:
            self._config = AutoConfig.from_pretrained(self.model_path)

            def _has_arch_suffix(suffix: str) -> bool:
                archs = set(getattr(self._config, "architectures", []) or [])
                return any(a.endswith(suffix) for a in archs)

            model_type = (getattr(self._config, "model_type", "") or "").lower()
            CAUSAL_FAMILIES = {
                "gpt2", "gptj", "gpt_neo", "gpt-neox", "mpt", "llama", "falcon",
                "qwen", "qwen2", "phi", "mistral", "mixtral", "gemma", "xlnet"
            }

            if _has_arch_suffix("ForSequenceClassification") or getattr(self._config, "num_labels", None):
                try:
                    self._model = _from_pretrained_safe(AutoModelForSequenceClassification, self.model_path)
                    self._kind = "seqcls"
                except Exception:
                    self._model = None

            if self._model is None and model_type in {
                "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                "electra", "albert", "camembert", "distilbert", "flaubert",
                "bart", "t5"
            }:
                try:
                    self._model = _from_pretrained_safe(AutoModelForMaskedLM, self.model_path)
                    self._kind = "mlm"
                except Exception:
                    self._model = None

            if self._model is None and model_type in CAUSAL_FAMILIES:
                try:
                    self._model = _from_pretrained_safe(AutoModelForCausalLM, self.model_path)
                    self._kind = "causal"
                except Exception:
                    self._model = None

            if self._model is None:
                try:
                    self._model = _from_pretrained_safe(AutoModelForSequenceClassification, self.model_path)
                    self._kind = "seqcls"
                except Exception:
                    self._model = _from_pretrained_safe(AutoModelForMaskedLM, self.model_path)
                    self._kind = "mlm"

        try:
            if getattr(self._model.config, "pad_token_id", None) is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
        except Exception:
            pass
        if self.fp16 and self.device.startswith("cuda"):
            try:
                self._model.half()
            except Exception:
                pass
        self._model.to(self.device)
        # --- NEW: seqcls 也是概率输出（_score_seqcls 返回 softmax/sigmoid 概率）---
        if self._kind == "seqcls":
            self.outputs_prob = True
            self.disable_calibration = True
        else:
            # MLM/causal 的 score 不是严格概率，仍允许 evaluator 做映射/校准
            self.outputs_prob = False
            self.disable_calibration = False

        self._model.eval()
        self.is_loaded = True

    # --------------------- 内部：Detective 目录匹配 ---------------------
    @staticmethod
    def _is_detective_dir(path: str) -> Optional[str]:
        if not isinstance(path, str):
            return None
        dirs = []
        if os.path.isdir(path):
            dirs.append(path)
            for sub in ("best", "last"):
                cand = os.path.join(path, sub)
                if os.path.isdir(cand):
                    dirs.append(cand)
        elif os.path.isfile(path):
            par = os.path.dirname(path)
            if par:
                dirs.append(par)
                for sub in ("best", "last"):
                    cand = os.path.join(par, sub)
                    if os.path.isdir(cand):
                        dirs.append(cand)
        for d in dirs:
            ckpt_best = os.path.join(d, "model_classifier_best.pth")
            ckpt_last = os.path.join(d, "model_classifier_last.pth")
            if not (os.path.isfile(ckpt_best) or os.path.isfile(ckpt_last)):
                continue
            enc = os.path.join(d, "encoder_hf")
            if os.path.isdir(enc):
                return d
            if _find_detective_train_args(d):
                return d
        return None
    # --------------------- NEW: KNN database builder ---------------------
    def _setup_knn_database(self) -> None:
        if load_dataset_unified is None:
            raise RuntimeError(
                "未能导入 mgt_eval.data_utils.load.load_dataset_unified，"
                f"具体异常：{repr(_LOAD_UNIFIED_IMPORT_ERROR)}"
            )

        # 1) cache dir
        cache_root = os.path.abspath(os.path.expanduser(self.knn_cache_root or "~/.cache/mgt_eval"))
        knn_dir = os.path.join(cache_root, "knn")
        _ensure_dir(knn_dir)

        # 2) index name
        if self.knn_index_name is None:
            ix_hash = _hash_str(
                "knn|"
                + "|".join([
                    str(self.knn_train_dataset),
                    str(self.knn_embedding_model_name),
                    str(self.knn_pooling),
                    str(self.max_length),
                    str(self.knn_K),
                    str(self.knn_backend),
                ])
            )
            self.knn_index_name = f"{_basename_like_id(self.knn_embedding_model_name)}-{self.knn_pooling}-{ix_hash}"
        ix_dir = os.path.join(knn_dir, self.knn_index_name)
        _ensure_dir(ix_dir)

        # 3) build embedding encoder
        enc = TextEmbeddingModel(self.knn_embedding_model_name).to(self.device)
        if self.fp16 and self.device.startswith("cuda"):
            try:
                enc.half()
            except Exception:
                pass

        # load optional embedding ckpt
        if self.knn_embedding_ckpt_path and os.path.isfile(self.knn_embedding_ckpt_path):
            sd = torch.load(self.knn_embedding_ckpt_path, map_location=self.device)
            new_sd = {}
            if isinstance(sd, dict):
                for k, v in sd.items():
                    nk = k[6:] if k.startswith("model.") else k
                    new_sd[nk] = v
            try:
                enc.load_state_dict(new_sd, strict=False)
            except Exception:
                enc.load_state_dict(sd, strict=False)

        enc.eval()
        self._knn_enc = enc
        self._tokenizer = enc.tokenizer  # 让 padding/mask 等行为与 embedding tokenizer 对齐

        # 4) load train examples
        train_examples, _ = load_dataset_unified(self.knn_train_dataset, sample_k=None, sample_seed=114514, group_cols=None)
        train_texts = [str(ex["text"]) for ex in train_examples]
        train_labels = [int(ex["label"]) for ex in train_examples]

        # 5) embed train
        def _batch_encode(texts: List[str]) -> Dict[str, torch.Tensor]:
            toks = self._tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )
            return {k: v.to(self.device) for k, v in toks.items()}

        @torch.no_grad()
        def _embed_all(texts: List[str], desc: str) -> np.ndarray:
            vecs: List[torch.Tensor] = []
            for i in tqdm(range(0, len(texts), 128), desc=desc, dynamic_ncols=True):
                batch = _batch_encode(texts[i:i+128])
                z = self._knn_enc(batch, use_pooling=self.knn_pooling)  # [B,D] 已 normalize
                vecs.append(z.detach().float().cpu())
            return torch.cat(vecs, dim=0).numpy().astype("float32", copy=False)

        # ----- backend: torch -----
        if self.knn_backend == "torch":
            emb_path = os.path.join(ix_dir, "train_emb.npy")
            lbl_path = os.path.join(ix_dir, "train_labels.json")

            if self.knn_reuse_database and os.path.isfile(emb_path) and os.path.isfile(lbl_path):
                train_emb = np.load(emb_path)
                with open(lbl_path, "r", encoding="utf-8") as f:
                    train_labels = [int(x) for x in json.load(f)]
            else:
                train_emb = _embed_all(train_texts, desc="KNN Embed(train)")
                if self.knn_save_database:
                    np.save(emb_path, train_emb)
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        json.dump([int(x) for x in train_labels], f, ensure_ascii=False)

            # 存储到 matmul device（默认跟随 detector.device；若 knn_use_gpu_index 且 cuda 可用则放 cuda）
            self._knn_mm_device = "cuda" if (self.knn_use_gpu_index and torch.cuda.is_available() and self.device.startswith("cuda")) else "cpu"
            TR = torch.from_numpy(np.ascontiguousarray(train_emb)).to(self._knn_mm_device)
            TR = TR / (TR.norm(dim=1, keepdim=True) + 1e-6)
            self._knn_train_tensor = TR

            y = torch.tensor([int(x) for x in train_labels], dtype=torch.long, device=self._knn_mm_device)
            self._knn_train_labels_tensor = y
            return

        # ----- backend: faiss -----
        if self.knn_backend == "faiss":
            # 与原 embedding_knn_infer 的目录结构兼容：在 ix_dir 下写 index.faiss 等
            label_pkl = os.path.join(ix_dir, "label_dict.pkl")
            ix_exists = all(os.path.isfile(os.path.join(ix_dir, fn)) for fn in ["index.faiss", "index_meta.faiss"]) and os.path.isfile(label_pkl)

            if self.knn_reuse_database and ix_exists:
                indexer = _FaissIndexer(1, device=("cuda" if (self.knn_use_gpu_index and torch.cuda.is_available()) else "cpu"))
                indexer.deserialize_from(ix_dir)
                with open(label_pkl, "rb") as f:
                    label_dict = pickle.load(f)
            else:
                train_emb = _embed_all(train_texts, desc="KNN Embed(train)")
                emb_dim = int(train_emb.shape[1])
                indexer = _FaissIndexer(emb_dim, device=("cuda" if (self.knn_use_gpu_index and torch.cuda.is_available()) else "cpu"))
                train_ids = list(range(len(train_labels)))
                indexer.index_data(train_ids, train_emb)
                label_dict = {int(i): int(y) for i, y in enumerate(train_labels)}
                if self.knn_save_database:
                    indexer.serialize(ix_dir)
                    with open(label_pkl, "wb") as f:
                        pickle.dump(label_dict, f)

            self._knn_faiss_indexer = indexer
            self._knn_faiss_label_dict = label_dict
            return

        raise ValueError(f"[pretrained/KNN] unknown knn_backend={self.knn_backend!r}, expected 'torch' or 'faiss'")

    # --------------------- NEW: KNN scoring (returns prob_AI) ---------------------
    @torch.no_grad()
    def _score_knn(self, texts: List[str]) -> np.ndarray:
        assert self._knn_enc is not None, "[pretrained/KNN] encoder not initialized; did you call load()?"
        K = max(1, int(self.knn_K))

        toks = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        q = self._knn_enc(toks, use_pooling=self.knn_pooling)  # [B,D] normalized
        q = q.detach().float()

        # ---- torch backend ----
        if self.knn_backend == "torch":
            assert self._knn_train_tensor is not None and self._knn_train_labels_tensor is not None
            mm_dev = self._knn_mm_device or ("cuda" if torch.cuda.is_available() else "cpu")
            Q = q.to(mm_dev)
            TR = self._knn_train_tensor
            y = self._knn_train_labels_tensor
            B = Q.size(0)
            N = TR.size(0)

            # block-scan topK to avoid huge (B,N) peak
            best_val = torch.full((B, K), -1e9, device=mm_dev)
            best_idx = torch.full((B, K), -1, dtype=torch.long, device=mm_dev)

            db_block = max(1024, int(self.knn_db_block))
            for s in range(0, N, db_block):
                e = min(s + db_block, N)
                chunk = TR[s:e]                      # [M,D]
                S = Q @ chunk.t()                    # [B,M]
                v, i = torch.topk(S, k=K, dim=1, largest=True, sorted=False)
                i = i + s

                cat_v = torch.cat([best_val, v], dim=1)
                cat_i = torch.cat([best_idx, i], dim=1)
                best_val, pos = torch.topk(cat_v, k=K, dim=1, largest=True, sorted=False)
                best_idx = cat_i.gather(1, pos)

                del S, v, i, cat_v, cat_i, pos
                if mm_dev == "cuda":
                    torch.cuda.empty_cache()

            neigh = y[best_idx]                     # [B,K] labels in {0,1}
            ones = neigh.sum(dim=1).float()
            probs = (ones / float(K)).clamp(0.0, 1.0)
            return probs.detach().cpu().numpy().astype(np.float32, copy=False)

        # ---- faiss backend ----
        if self.knn_backend == "faiss":
            assert self._knn_faiss_indexer is not None and self._knn_faiss_label_dict is not None
            q_np = q.cpu().numpy().astype("float32", copy=False)
            top = self._knn_faiss_indexer.search_knn(q_np, top_docs=K, index_batch_size=int(self.knn_index_batch_size))
            out = []
            for ids_i, _scores_i in top:
                ones = 0
                for pid in ids_i[:K]:
                    ones += int(self._knn_faiss_label_dict.get(int(pid), 0))
                out.append(float(ones / float(K)))
            return np.asarray(out, dtype=np.float32)

        raise RuntimeError(f"[pretrained/KNN] unexpected knn_backend={self.knn_backend!r}")

    # --------------------- Scoring Dispatcher ---------------------
    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if getattr(self, "_kind", None) == "knn":
            return self._score_knn(texts)
        if getattr(self, "_kind", None) == "prdetect" and self._prdetect_shim is not None:
            return self._prdetect_shim.score_batch(texts)
        if getattr(self, "_kind", None) == "coco":
            return self._score_coco(texts)
        if getattr(self, "_kind", None) == "detective":
            return self._score_detective(texts)  # 新增
        if self._kind == "seqcls":
            return self._score_seqcls(texts)
        elif self._kind == "mlm":
            return self._score_mlm_pll(texts)
        else:
            return self._score_causallm(texts)

    # --------------------- DeTeCtive: classifier 概率 ---------------------
    @torch.no_grad()
    def _score_detective(self, texts: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)

        def _encode_with(obj: Any) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if hasattr(obj, "encode") and callable(getattr(obj, "encode")):
                return obj.encode(input_ids=input_ids, attention_mask=attention_mask)
            if callable(obj):
                return obj({"input_ids": input_ids, "attention_mask": attention_mask})
            return None

        z = None
        if hasattr(self._model, "get_encoder") and callable(getattr(self._model, "get_encoder")):
            z = _encode_with(self._model.get_encoder())
        if z is None:
            z = _encode_with(self._model)

        if z is not None:
            cls_head = getattr(self._model, "cls_label", None) or getattr(self._model, "classifier", None)
            if cls_head is None:
                raise RuntimeError("[pretrained/DeTeCtive] 模型缺少 cls_label/classifier 分类头。")
            logits = cls_head(z)
        else:
            if hasattr(self._model, "__call__"):
                B = input_ids.size(0)
                dummy = torch.zeros(B, dtype=torch.long, device=self.device)
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    y_label=dummy,
                )
                logits = out["logits_label"]
            else:
                raise RuntimeError("[pretrained/DeTeCtive] 无法找到可用的 encode/forward 接口。")

        probs = torch.softmax(logits, dim=-1)
        ai_id = max(0, min(self.ai_label_id, probs.size(-1) - 1))
        return probs[:, ai_id].detach().float().cpu().numpy()

    # --------------------- CoCo ---------------------
    @torch.no_grad()
    def _score_coco(self, texts: List[str]) -> np.ndarray:
        if any(x is None for x in [CoCoGraphDataset, coco_collate]):
            raise RuntimeError(
                "[pretrained] CoCo 推理所需组件缺失，请确认 mgt_eval.detectors.finetuned.coco 可用。"
            )
        examples = [{"text": t, "label": 0} for t in texts]
        try:
            cfg = getattr(self._model, "cfg", None)
            if cfg is not None and hasattr(cfg, "max_seq_length"):
                cfg.max_seq_length = int(self.max_length)
        except Exception:
            pass

        ds = CoCoGraphDataset(examples, self._tokenizer, getattr(self._model, "cfg", None))
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=coco_collate)

        probs_ai = []
        for batch in dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self._model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nodes_index_mask=batch["nodes_index_mask"],
                adj_metric=batch["adj_metric"],
                node_mask=batch["node_mask"],
            )["logits"]
            prob = torch.softmax(logits, dim=-1)[:, max(0, min(self.ai_label_id, logits.size(-1)-1))]
            probs_ai.extend(prob.detach().float().cpu().tolist())
        return np.array(probs_ai, dtype=np.float32)

    # --------------------- SequenceClassification ---------------------
    def _score_seqcls(self, texts: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self._model(**toks)
        logits = outputs.logits  # (B, C)
        num_labels = logits.shape[-1]
        if num_labels == 1:
            probs_ai = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.softmax(logits, dim=-1)
            ai_id = self.ai_label_id
            id2label = getattr(self._config, "id2label", None) if self._config is not None else None
            if isinstance(id2label, dict):
                for k, v in id2label.items():
                    if str(v).lower() in {"ai", "machine", "fake", "generated", "mgt", "llm"}:
                        ai_id = int(k)
                        break
            ai_id = max(0, min(ai_id, num_labels - 1))
            probs_ai = probs[:, ai_id]

        return probs_ai.detach().float().cpu().numpy()

    # --------------------- CausalLM: mean NLL ---------------------
    @torch.no_grad()
    def _score_causallm(self, texts: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_attn = (shift_labels != -100)

        safe_labels = shift_labels.masked_fill(~shift_attn, 0)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * shift_attn.float()

        lengths = shift_attn.float().sum(dim=1).clamp_min(1.0)
        mean_nll = -(token_logp.sum(dim=1) / lengths)

        scores = 1.0 / (1.0 + mean_nll)
        return scores.detach().float().cpu().numpy()

    # --------------------- MaskedLM: pseudo log-likelihood (PLL) ---------------------
    @torch.no_grad()
    def _score_mlm_pll(self, texts: List[str]) -> np.ndarray:
        assert self._tokenizer.mask_token_id is not None, "MaskedLM 需要 tokenizer.mask_token_id"
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        B, T = input_ids.size()
        mask_id = self._tokenizer.mask_token_id

        special_ids = set(
            tid for tid in [
                getattr(self._tokenizer, "cls_token_id", None),
                getattr(self._tokenizer, "sep_token_id", None),
                getattr(self._tokenizer, "pad_token_id", None),
                getattr(self._tokenizer, "bos_token_id", None),
                getattr(self._tokenizer, "eos_token_id", None),
            ] if tid is not None
        )

        scores: List[float] = []
        iterator = range(B)
        if self.show_progress and B > 1:
            iterator = tqdm(iterator, desc="PLL", leave=True, dynamic_ncols=True)

        for b in iterator:
            ids = input_ids[b]
            attn = attention_mask[b]
            pos = torch.nonzero(attn == 1, as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                scores.append(float(0.5))
                continue

            if special_ids:
                mask_non_special = torch.ones_like(ids, dtype=torch.bool)
                for sid in special_ids:
                    mask_non_special &= (ids != sid)
                pos = pos[mask_non_special[pos]]
                if pos.numel() == 0:
                    scores.append(float(0.5))
                    continue

            total_logp = 0.0
            count = 0

            for start in range(0, pos.numel(), self.pll_stride):
                chunk_pos = pos[start:start + self.pll_stride]
                csz = chunk_pos.numel()
                ids_exp = ids.unsqueeze(0).repeat(csz, 1)
                attn_exp = attn.unsqueeze(0).repeat(csz, 1)

                arange_c = torch.arange(csz, device=ids.device)
                ids_exp[arange_c, chunk_pos] = mask_id

                outputs = self._model(input_ids=ids_exp, attention_mask=attn_exp)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)

                orig_tokens = ids[chunk_pos]
                logp = log_probs[arange_c, chunk_pos, orig_tokens]
                total_logp += float(logp.sum().detach().cpu())
                count += int(csz)

            mean_nll = - (total_logp / max(count, 1))
            score = 1.0 / (1.0 + mean_nll)
            scores.append(float(score))

        return np.array(scores, dtype=np.float32)

    # ============================================================================
    # 嵌入 + KNN 推理：数据库缓存 / 生成 / 复用
    # ============================================================================
    @torch.no_grad()
    def embedding_knn_infer(
        self,
        train_dataset: str,
        test_dataset: str,
        embedding_model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        embedding_ckpt_path: Optional[str] = None,
        pooling: str = "average",
        batch_size: int = 128,
        max_K: int = 51,
        cache_root: Optional[str] = None,
        save_dataset: bool = False,
        save_database: bool = False,
        reuse_database: bool = True,
        index_name: Optional[str] = None,
        use_gpu_index: bool = True,
        return_probs: bool = True,
        num_workers: int = 0,
        knn_backend: str = "torch",   # 'torch' | 'faiss'
        q_block: int = 1024,
        db_block: int = 65536,
        index_batch_size: int = 8,
        sample_k_train: Optional[int] = None,
        sample_k_test: Optional[int] = None,
        seed: int = 114514,
    ) -> Dict[str, Any]:
        """
        使用“文本嵌入 + FAISS KNN”进行检测推理，并**可选**持久化 database 到用户目录。

        Args:
            train_dataset: 训练库数据（JSON/JSONL/目录/HF-HC3 指定），结构需兼容 load_dataset_unified
            test_dataset:  测试数据（同上）
            embedding_model_name: 用于编码的 HF 模型名/目录
            embedding_ckpt_path: 可选的你的外部 state_dict 路径（按你的 DAT/SimCSE 训练产物加载）
            pooling: 'average' 或 'cls'；BGE/MXBai 会自动用 'cls'
            batch_size: 推理批大小
            max_K: 评估 1..K 的投票结果（多数投票），并返回每个 K 的指标
            cache_root: 用户缓存根目录（默认 ~/.cache/mgt_eval）
            save_dataset: 是否把“标准化后的样本”写入缓存（datasets 子目录）
            save_database: 是否把 FAISS 索引与 label_dict 写入缓存（faiss 子目录）
            reuse_database: 若缓存索引存在，是否直接复用
            index_name: database 子目录名（缺省则自动哈希）
            use_gpu_index: 若可用则使用 GPU 索引（不可用会自动回落到 CPU）
            return_probs: 返回“投票比例”作为概率（可用于 AUROC）；若不需要可设 False
            num_workers: DataLoader 工作线程数
            sample_k_train/test: 可选抽样规模
            seed: 随机种子（影响抽样）

        Returns:
            {
              "K_values": [1,2,...,max_K],
              "metrics_by_K": {
                  k: {
                      "human_rec", "machine_rec", "avg_rec",
                      "acc", "precision", "recall", "f1",
                      "auroc": float | None
                  }, ...
              },
              "best_K": int,   # 按 avg_rec 最大选取
              "best_metrics": {...同上...},
              "test_probs_by_K": Optional[List[List[float]]],  # 若 return_probs=True，返回每个K的AI概率
              "test_preds_by_K": Dict[int, List[int]],         # 0/1
              "test_ids": List[int],                           # 与输入顺序对齐
            }
        """
        if load_dataset_unified is None:
            raise RuntimeError(
                "未能导入 mgt_eval.data_utils.load.load_dataset_unified，"
                f"具体异常：{repr(_LOAD_UNIFIED_IMPORT_ERROR)}"
            )
        device = self.device

        cache_root = os.path.abspath(os.path.expanduser(cache_root or "~/.cache/mgt_eval"))
        cache_ds_dir = os.path.join(cache_root, "datasets")
        cache_ix_dir = os.path.join(cache_root, "faiss")
        _ensure_dir(cache_ds_dir); _ensure_dir(cache_ix_dir)
        print(f"[KNN] cache_root resolved to: {cache_root}")

        # ---------- 读取/缓存 标准化样本 ----------
        def _ensure_examples(spec: str, tag: str, sample_k: Optional[int]) -> Tuple[List[Dict[str, Any]], str]:
            ds_hash = _hash_str(f"{spec}|sample={sample_k}|seed={seed}")
            cache_path = os.path.join(cache_ds_dir, f"{ds_hash}-{tag}.jsonl")
            if os.path.isfile(cache_path):
                examples = _load_jsonl(cache_path)
                return examples, cache_path
            examples, _group_cols = load_dataset_unified(spec, sample_k=sample_k, sample_seed=seed, group_cols=None)
            if save_dataset:
                _save_jsonl(examples, cache_path)
            return examples, cache_path

        train_examples, train_cache_path = _ensure_examples(train_dataset, "train", sample_k_train)
        test_examples,  test_cache_path  = _ensure_examples(test_dataset, "test",  sample_k_test)

        # 样本 id：默认用序号；若存在 'uuid' 则优先使用其哈希稳定映射为 int
        def _to_int_ids(examples: List[Dict[str, Any]]) -> List[int]:
            ids: List[int] = []
            for i, ex in enumerate(examples):
                if "uuid" in ex:
                    ids.append(int(int(hashlib.md5(str(ex["uuid"]).encode("utf-8")).hexdigest(), 16) % (10**9)))
                else:
                    ids.append(i)
            return ids

        train_ids = _to_int_ids(train_examples)
        test_ids = _to_int_ids(test_examples)
        train_texts = [str(ex["text"]) for ex in train_examples]
        test_texts  = [str(ex["text"]) for ex in test_examples]
        train_labels = [int(ex["label"]) for ex in train_examples]
        test_labels  = [int(ex["label"]) for ex in test_examples]

        # ---------- 准备嵌入模型 ----------
        enc = TextEmbeddingModel(embedding_model_name).to(device)
        if self.fp16 and device.startswith("cuda"):
            try:
                enc.half()
            except Exception:
                pass

        if embedding_ckpt_path and os.path.isfile(embedding_ckpt_path):
            sd = torch.load(embedding_ckpt_path, map_location=device)
            # 清理前缀
            new_sd = {}
            for k, v in sd.items():
                nk = k[6:] if k.startswith("model.") else k
                new_sd[nk] = v
            try:
                enc.load_state_dict(new_sd, strict=False)
            except Exception:
                # 允许直接尝试原 sd（兼容其他保存方式）
                enc.load_state_dict(sd, strict=False)

        tokenizer = enc.tokenizer

        def _batch_encode(texts: List[str]) -> Dict[str, torch.Tensor]:
            toks = tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )
            return {k: v.to(device) for k, v in toks.items()}

        def _embed_all(texts: List[str], desc: str) -> np.ndarray:
            enc.eval()
            vecs: List[torch.Tensor] = []
            N = len(texts)
            for i in tqdm(range(0, N, batch_size), desc=desc, dynamic_ncols=True):
                chunk = texts[i:i+batch_size]
                batch = _batch_encode(chunk)
                z = enc(batch, use_pooling=pooling)  # [B, D], 已归一化
                vecs.append(z.detach().float().cpu())
            emb = torch.cat(vecs, dim=0).numpy()
            return emb

        # ---------- database（FAISS）准备：复用 or 生成 ----------
        # ✅ 优先：当使用 Torch-KNN 时，直接在此早返回，完全绕过 FAISS 初始化/反序列化
                # ---------- database cache dir (torch backend) ----------
        if knn_backend.lower() == "torch":
            # 给 torch backend 单独一个缓存目录，避免和 faiss 混用
            torch_ix_root = os.path.join(cache_root, "knn_torch")
            _ensure_dir(torch_ix_root)

            # index_name：必须把“抽样/seed/ckpt”等因素编码进去，否则复用可能错配
            if index_name is None:
                ckpt_sig = ""
                if embedding_ckpt_path and os.path.isfile(embedding_ckpt_path):
                    st = os.stat(embedding_ckpt_path)
                    ckpt_sig = f"{embedding_ckpt_path}|{st.st_size}|{int(st.st_mtime)}"
                ix_hash = _hash_str(
                    "torchix|"
                    + "|".join([
                        str(train_cache_path),               # ✅ 用“标准化后的缓存路径”更稳定
                        str(embedding_model_name),
                        str(pooling),
                        str(self.max_length),
                        str(sample_k_train),
                        str(seed),
                        ckpt_sig,
                    ])
                )
                index_name = f"{_basename_like_id(embedding_model_name)}-{pooling}-{ix_hash}"

            ix_dir = os.path.join(torch_ix_root, index_name)
            _ensure_dir(ix_dir)

            emb_path = os.path.join(ix_dir, "train_emb.npy")
            lbl_path = os.path.join(ix_dir, "train_labels.npy")
            meta_path = os.path.join(ix_dir, "meta.json")

            # meta 用于防止误复用（尤其是你改了 sample_k/seed/ckpt 之类）
            meta = dict(
                train_cache_path=os.path.abspath(train_cache_path),
                embedding_model_name=str(embedding_model_name),
                embedding_ckpt_path=str(embedding_ckpt_path) if embedding_ckpt_path else None,
                pooling=str(pooling),
                max_length=int(self.max_length),
                sample_k_train=sample_k_train,
                seed=int(seed),
            )

            # ---------- reuse train embeddings ----------
            can_reuse = (
                reuse_database
                and os.path.isfile(emb_path)
                and os.path.isfile(lbl_path)
                and os.path.isfile(meta_path)
            )
            if can_reuse:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    can_reuse = (old == meta)
                except Exception:
                    can_reuse = False

            if can_reuse:
                train_emb = np.load(emb_path)
                train_labels_np = np.load(lbl_path)
            else:
                train_emb = _embed_all(train_texts, desc="Embed(train)")
                train_emb = np.ascontiguousarray(train_emb.astype("float32", copy=False))
                train_labels_np = np.asarray(train_labels, dtype=np.int64)

                if save_database:
                    np.save(emb_path, train_emb)
                    np.save(lbl_path, train_labels_np)
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            # test embeddings 通常不建议跨 run 复用（因为 test_dataset 常换），这里保持现算
            test_emb = _embed_all(test_texts, desc="Embed(test)")
            test_emb = np.ascontiguousarray(test_emb.astype("float32", copy=False))

            # ---------- torch KNN (double-block topK to avoid giant matmul) ----------
            device_mm = "cuda" if (use_gpu_index and torch.cuda.is_available()) else "cpu"
            TR = torch.from_numpy(train_emb).to(device_mm, non_blocking=True)
            y  = torch.from_numpy(train_labels_np).to(device_mm, non_blocking=True)
            TE = torch.from_numpy(test_emb).to(device_mm, non_blocking=True)

            TR = TR / (TR.norm(dim=1, keepdim=True) + 1e-6)
            TE = TE / (TE.norm(dim=1, keepdim=True) + 1e-6)

            topK = int(max_K)
            K_values = list(range(1, topK + 1))
            preds_by_K = {k: [] for k in K_values}
            probs_by_K = {k: [] for k in K_values} if return_probs else None

            Nq = TE.size(0)
            N  = TR.size(0)
            qb = max(1, int(q_block))
            db = max(1024, int(db_block))

            for qs in range(0, Nq, qb):
                qe = min(qs + qb, Nq)
                Q = TE[qs:qe]  # [q, d]
                qsz = Q.size(0)

                best_val = torch.full((qsz, topK), -1e9, device=device_mm)
                best_idx = torch.full((qsz, topK), -1, dtype=torch.long, device=device_mm)

                for s in range(0, N, db):
                    e = min(s + db, N)
                    chunk = TR[s:e]            # [m, d]
                    S = Q @ chunk.t()          # [q, m]
                    v, i = torch.topk(S, k=topK, dim=1, largest=True, sorted=False)
                    i = i + s

                    cat_v = torch.cat([best_val, v], dim=1)
                    cat_i = torch.cat([best_idx, i], dim=1)
                    best_val, pos = torch.topk(cat_v, k=topK, dim=1, largest=True, sorted=False)
                    best_idx = cat_i.gather(1, pos)

                    del S, v, i, cat_v, cat_i, pos
                    if device_mm == "cuda":
                        torch.cuda.empty_cache()

                neigh = y[best_idx]  # [q, topK] in {0,1}

                # 逐 rank 累积投票（与你原来保持一致：平票偏向 1）
                ones = torch.zeros((qsz,), device=device_mm, dtype=torch.int32)
                zeros = torch.zeros((qsz,), device=device_mm, dtype=torch.int32)
                for rk in range(1, topK + 1):
                    lab = neigh[:, rk - 1]
                    ones += (lab == 1).int()
                    zeros += (lab == 0).int()
                    pred = (ones >= zeros).long()

                    preds_by_K[rk].extend(pred.detach().cpu().tolist())
                    if probs_by_K is not None:
                        probs_by_K[rk].extend((ones.float() / float(rk)).detach().cpu().tolist())

                del best_val, best_idx, neigh, ones, zeros
                if device_mm == "cuda":
                    torch.cuda.empty_cache()

            # ---------- metrics ----------
            y_true = [int(y) for y in test_labels]
            metrics_by_K = {}
            for k in K_values:
                m = _compute_basic_metrics(y_true, preds_by_K[k])
                if return_probs and (probs_by_K is not None):
                    auc_val = _roc_auc_binary(np.asarray(y_true, int), np.asarray(probs_by_K[k], float))
                    m["auroc"] = float(auc_val) if auc_val is not None else None
                else:
                    m["auroc"] = None
                metrics_by_K[k] = m

            best_K = max(K_values, key=lambda kk: metrics_by_K[kk]["avg_rec"])
            return {
                "K_values": K_values,
                "metrics_by_K": metrics_by_K,
                "best_K": int(best_K),
                "best_metrics": metrics_by_K[best_K],
                "test_preds_by_K": preds_by_K,
                "test_probs_by_K": [probs_by_K[k] for k in K_values] if (return_probs and probs_by_K is not None) else None,
                "test_ids": [int(x) for x in test_ids],
                "torch_index_dir": ix_dir,   # ✅ 方便你 debug
                "torch_reused": bool(can_reuse),
            }

        # ---------- 仅当使用 FAISS 时，才进行索引目录与缓存的初始化 ----------
        if index_name is None:
            ix_hash = _hash_str("ix|" + "|".join([train_dataset, embedding_model_name, pooling, str(self.max_length)]))
            index_name = f"{_basename_like_id(embedding_model_name)}-{pooling}-{ix_hash}"
        ix_dir = os.path.join(cache_ix_dir, index_name)
        os.makedirs(ix_dir, exist_ok=True)
        print(f"[KNN] index_dir = {ix_dir}", flush=True)

        label_pkl = os.path.join(ix_dir, "label_dict.pkl")
        ix_exists = all(os.path.isfile(os.path.join(ix_dir, fn)) for fn in ["index.faiss", "index_meta.faiss"]) \
                    and os.path.isfile(label_pkl)

        # FAISS 索引器与 label_dict
        label_dict: Dict[int, int] = {}

        if reuse_database and ix_exists:
            # 反序列化已有索引
            dim_guess = 1  # 仅为 Indexer 构造占位；实际维度会在反序列化后覆盖
            indexer = _FaissIndexer(dim_guess, device=("cuda" if (use_gpu_index and torch.cuda.is_available()) else "cpu"))
            print("[KNN] restore FAISS index from cache ...", flush=True)
            indexer.deserialize_from(ix_dir)
            with open(label_pkl, "rb") as f:
                label_dict = pickle.load(f)
        else:
            # 需要新建索引：先计算 train 嵌入
            train_emb = _embed_all(train_texts, desc="Embed(train)")
            emb_dim = int(train_emb.shape[1])

            # 构建索引器并写入向量
            indexer = _FaissIndexer(emb_dim, device=("cuda" if (use_gpu_index and torch.cuda.is_available()) else "cpu"))
            print("[KNN] build FAISS indexer ...", flush=True)
            print("[KNN] add embeddings to FAISS ...", flush=True)
            indexer.index_data(train_ids, train_emb)

            label_dict = {int(i): int(y) for i, y in zip(train_ids, train_labels)}
            if save_database:
                indexer.serialize(ix_dir)  # 写 index.faiss 与 index_meta.faiss
                with open(label_pkl, "wb") as f:
                    pickle.dump(label_dict, f)

        # 计算测试集嵌入
        test_emb = _embed_all(test_texts, desc="Embed(test)")
        test_emb = np.ascontiguousarray(test_emb.astype("float32"))

        # ---------- 使用 FAISS 进行检索 ----------
        # 官方 search_knn: List[(List[db_ids:int], List[scores:float])]
        top = indexer.search_knn(test_emb, top_docs=int(max_K), index_batch_size=int(index_batch_size))

        K_values = list(range(1, int(max_K) + 1))
        preds_by_K: Dict[int, List[int]] = {k: [] for k in K_values}
        probs_by_K: Optional[Dict[int, List[float]]] = {k: [] for k in K_values} if return_probs else None

        for i, (ids_i, scores_i) in enumerate(top):
            ones = zeros = 0
            # 已按相似度降序；逐步累积进行多数投票
            for rank_k, pid in enumerate(ids_i, start=1):
                lab = int(label_dict.get(int(pid), 0))
                if lab == 1:
                    ones += 1
                else:
                    zeros += 1

                if rank_k in preds_by_K:
                    pred = 1 if ones >= zeros else 0
                    preds_by_K[rank_k].append(pred)
                    if probs_by_K is not None:
                        prob_ai = ones / rank_k
                        probs_by_K[rank_k].append(float(prob_ai))

                if rank_k == max_K:
                    break

        # ---------- 统计指标（含可选 AUROC） ----------
        metrics_by_K: Dict[int, Dict[str, float]] = {}
        y_true = [int(y) for y in test_labels]

        for k in K_values:
            y_pred = preds_by_K[k]
            m = _compute_basic_metrics(y_true, y_pred)
            # AUROC：需要概率；若无或常数分数时返回 None
            auroc = None
            if return_probs and probs_by_K is not None:
                y_score = np.asarray(probs_by_K[k], dtype=float)
                auc_val = _roc_auc_binary(np.asarray(y_true, dtype=int), y_score)
                auroc = auc_val if auc_val is not None else None
            m["auroc"] = float(auroc) if (auroc is not None) else None
            metrics_by_K[k] = m

        # 选择 best_K：按 avg_rec 最大
        best_K = K_values[0]
        best_val = metrics_by_K[best_K]["avg_rec"]
        for k in K_values[1:]:
            if metrics_by_K[k]["avg_rec"] > best_val:
                best_val = metrics_by_K[k]["avg_rec"]
                best_K = k

        result: Dict[str, Any] = {
            "K_values": K_values,
            "metrics_by_K": metrics_by_K,
            "best_K": int(best_K),
            "best_metrics": metrics_by_K[best_K],
            "test_preds_by_K": preds_by_K,
            "test_ids": [int(x) for x in test_ids],
        }
        if return_probs and probs_by_K is not None:
            # 以列表形式返回，便于调用端保存/后处理
            result["test_probs_by_K"] = [probs_by_K[k] for k in K_values]
        else:
            result["test_probs_by_K"] = None

        return result
