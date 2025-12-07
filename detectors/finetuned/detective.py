# mgt_eval/detectors/finetuned/detective.py
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
os.environ.setdefault("FAISS_NO_GPU", "1")

# ---- 优化 A100 上的 FP32 matmul：启用 TF32 / 提升精度等级 ----
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # Ampere (SM 8.x) 及以上支持 TF32
        if major >= 8:
            # 'high' 或 'medium' 均可；一般推荐 'high'（更快，精度对多数 NLP 任务足够）
            torch.set_float32_matmul_precision('medium')
            # 这两个开关分别影响 cuBLAS 和 cuDNN 路径
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
DETECTOR_NAME = "DETECTIVE"
detector_type = "Model-based"
CITATION_AUTHORS = "Xun Guo, Shan Zhang, Yongxin He, Ting Zhang, Wanquan Feng, Haibin Huang, Chongyang Ma"
CITATION_TITLE = "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning"
CITATION_LINK = "https://arxiv.org/abs/2410.20964"

# ============== 环境静默设置（避免 tokenizers 多线程 + fork 死锁） ==============
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _unique_preserve_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _torch_knn_majority_vote(emb_db_np, labels_db_list, emb_q_np, topk=10,
                             device="cuda", q_block=2048):
    """
    用 PyTorch 在 GPU 上做 KNN（余弦相似度 + topk 多数投票）。
    - emb_db_np: (N, d) float32 contiguous
    - emb_q_np : (Q, d) float32 contiguous
    - labels_db_list: len N，元素为 {0,1}（与 DeTeCtive 内部一致：1=human, 0=machine）
    """
    import torch
    assert emb_db_np.dtype == np.float32 and emb_q_np.dtype == np.float32
    N = emb_db_np.shape[0]
    labels_t = torch.as_tensor(labels_db_list, dtype=torch.long, device=device)

    db_t = torch.as_tensor(emb_db_np, dtype=torch.float32, device=device)
    db_t = db_t / (db_t.norm(dim=1, keepdim=True) + 1e-6)

    preds = []
    Q = emb_q_np.shape[0]
    for s in range(0, Q, q_block):
        q_np = emb_q_np[s:s+q_block]
        q_t = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        q_t = q_t / (q_t.norm(dim=1, keepdim=True) + 1e-6)

        # 相似度 [b, N]，取 topk
        sims = q_t @ db_t.T
        topk_idx = sims.topk(k=topk, dim=1, largest=True, sorted=False).indices  # [b, k]
        lbl = labels_t[topk_idx]  # [b, k]

        zeros = (lbl == 0).sum(dim=1)
        ones = topk - zeros
        batch_preds = torch.where(zeros > ones,
                                  torch.zeros_like(zeros),
                                  torch.ones_like(ones)).tolist()
        preds.extend([str(int(x)) for x in batch_preds])
        del q_t, sims, topk_idx, lbl, zeros, ones
        torch.cuda.empty_cache()
    return preds

def _scan_generator_classes(examples: List[Dict[str, Any]]) -> List[str]:
    """
    从样本中提取 generator 名单（与官方 PassagesDataset 的 classes 类似）：
    - 优先 meta.generator / meta.model_name
    - 其次根级字段 src / model / generator / LLM_name
    - 人类样本回填 'human'，未知生成器回填 'default'
    - 最终过滤掉 'default'（只打印有效类别）
    """
    gens = []
    for e in examples:
        meta = e.get("meta", {}) or {}
        gen_root = e.get("src") or e.get("model") or e.get("generator") or e.get("LLM_name")
        gen_meta = meta.get("generator") or meta.get("model_name")
        gen = gen_root or gen_meta
        if not gen:
            # 0=human, 1=ai（mgt_eval 约定）
            y = int(e.get("label", 1))
            gen = "human" if y == 0 else "default"
        gens.append(str(gen))
    classes = _unique_preserve_order(gens)
    # 与官方输出风格一致，不展示 'default'
    classes = [c for c in classes if c != "default"]
    return classes


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


# ====== 官方字段探测 & 自适应读取 ======
_JSONL_PROBE_MAX = 80  # 最多探测前 N 行


def _probe_jsonl_has_official_keys(jsonl_path: str) -> bool:
    """
    判断 JSONL 是否包含官方常见字段：text/label/(src|model|generator|LLM_name)
    只要命中其中任一即可视为“含官方信息”的样式。
    """
    want_text = False
    want_label = False
    want_src_like = False
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for _ in range(_JSONL_PROBE_MAX):
                line = f.readline()
                if not line:
                    break
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                keys = set(obj.keys())
                want_text = want_text or ("text" in keys)
                want_label = want_label or ("label" in keys)
                want_src_like = want_src_like or any(k in keys for k in ["src", "model", "generator", "LLM_name"])
                if want_text and (want_label or want_src_like):
                    return True
    except Exception:
        pass
    return False


def _normalize_example_from_official_json(obj: Dict[str, Any], dataset_name_hint: str) -> Dict[str, Any]:
    """
    将官方 JSON 行样式统一规整为 mgt_eval 的样式：
    - text: str
    - label: int (mgt_eval 约定: 0=human, 1=ai)
    - meta: {generator, dataset}

    标签推断优先级：
    1) 若 src/model 指向 human => 0
    2) 若 label 存在，且 == 1 表示 human（对齐官方脚本约定） => 0，否则 1
    3) 若仍不确定，最后 fallback：label==0 视为 human
    """
    text = obj.get("text") or obj.get("content") or ""
    gen = obj.get("src") or obj.get("model") or obj.get("generator") or obj.get("LLM_name") or None
    raw_label = obj.get("label")

    # 先用生成器名判断
    y_mg = None  # 0=human, 1=ai
    if isinstance(gen, str):
        if "human" in gen.lower():
            y_mg = 0
        else:
            y_mg = 1

    # 其次用官方常见约定：label==1 -> human
    if y_mg is None and raw_label is not None:
        try:
            v = int(raw_label)
            y_mg = 0 if v == 1 else 1
        except Exception:
            pass

    # 最后兜底：假设 label==0 -> human
    if y_mg is None:
        try:
            v = int(raw_label)
            y_mg = 0 if v == 0 else 1
        except Exception:
            y_mg = 1  # 实在拿不到，就按机器做负担更保守

    meta = {
        "generator": gen or ("human" if y_mg == 0 else "default"),
        "dataset": dataset_name_hint,
    }
    return {"text": text, "label": int(y_mg), "meta": meta}


def _load_jsonl_with_official_keys(jsonl_path: str, seed: int) -> List[Dict[str, Any]]:
    """
    真正读取“含官方字段”的 JSONL，逐行规整为 mgt_eval 样式。
    """
    dsname = Path(jsonl_path).stem
    out: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                out.append(_normalize_example_from_official_json(obj, dsname))
            except Exception:
                # 跳过坏行
                continue
    return out


def _dir_looks_like_official_csv_dir(dir_path: str) -> bool:
    """
    Deepfake/TuringBench 的官方脚本目录通常有 train.csv/valid.csv/test.csv，
    且至少包含 [text-like, label-like, src-like] 列。
    """
    p = Path(dir_path)
    if not p.is_dir():
        return False
    # 任意一个存在即可判定为“可能是官方目录”
    for name in ["train.csv", "valid.csv", "test.csv"]:
        if (p / name).exists():
            return True
    return False


def _read_official_csv_split(dir_path: str, split: str) -> List[Dict[str, Any]]:
    """
    读取官方 CSV（Deepfake/TuringBench）到 mgt_eval 样式；
    自动识别列名：'text' 或 'Generation'；src 在 'src' 或 'label'（值为 'human'）里。
    """
    import pandas as pd

    csv_path = Path(dir_path) / f"{split}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)

    # 文本列
    if "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    elif "Generation" in df.columns:
        texts = df["Generation"].astype(str).tolist()
    else:
        raise RuntimeError(f"cannot find text column in {csv_path}")

    # 源（生成器）列
    if "src" in df.columns:
        srcs = df["src"].astype(str).tolist()
    elif "label" in df.columns:
        # TuringBench 的 'label' 列非数值，而是 'human' / 其他模型名
        srcs = df["label"].astype(str).tolist()
    else:
        srcs = ["default"] * len(texts)

    # 标签列（若无则由 src 推断）
    labels = []
    for i in range(len(texts)):
        lab = None
        if "label" in df.columns:
            try:
                lab = int(df.iloc[i]["label"])
            except Exception:
                lab = None
        if lab is None:
            # 根据 src 推断：human -> 0=human -> mgt_eval 0
            lab = 0 if ("human" in str(srcs[i]).lower()) else 1
        else:
            # 兼容官方：label==1 表示 human
            lab = 0 if lab == 1 else 1
        labels.append(int(lab))

    dsname = Path(dir_path).name
    out = []
    for t, y, s in zip(texts, labels, srcs):
        out.append({"text": t, "label": int(y), "meta": {"generator": str(s), "dataset": dsname}})
    return out


def _load_dataset_auto(path: str, seed: int) -> List[Dict[str, Any]]:
    """
    统一入口：
    1) 如果是目录且像官方 CSV 结构 -> 读取 train/valid/test 中最合适的 split
    2) 如果是 JSONL 且含官方字段 -> 逐行解析
    3) 否则 -> 回退 load_dataset_unified

    说明：
    - 训练集：优先 'train.csv'，否则退到 'valid.csv'/'test.csv'
    - 验证集：优先 'valid.csv'，否则退到 'test.csv'
    """
    p = Path(path)

    # 目录：Deepfake/TuringBench 风格
    if p.is_dir() and _dir_looks_like_official_csv_dir(str(p)):
        # 训练/验证的 split 选择由调用方区分，这里只做“默认训练集”的选择
        # 留给外部的包装函数决定具体 split，避免在这里写死
        # 先返回所有可用 split，供上层挑选
        pack = {
            "train": _read_official_csv_split(str(p), "train"),
            "valid": _read_official_csv_split(str(p), "valid"),
            "test": _read_official_csv_split(str(p), "test"),
        }
        # 将空的去掉
        for k in list(pack.keys()):
            if len(pack[k]) == 0:
                pack.pop(k, None)
        # 扁平化返回（给不带 split 的情况兜底用）
        flat = []
        for k in ["train", "valid", "test"]:
            if k in pack:
                flat.extend(pack[k])
        return flat

    # JSONL 文件：M4 等官方 jsonl（含 src/model/label）
    if p.is_file() and p.suffix.lower() == ".jsonl" and _probe_jsonl_has_official_keys(str(p)):
        return _load_jsonl_with_official_keys(str(p), seed)

    # 回退：使用 mgt_eval 自带的统一加载器
    exs, _ = load_dataset_unified(dataset=str(p), sample_k=None, sample_seed=seed, group_cols=None)
    return exs


# ============== 指标（复刻你原 DeTeCtive 代码里的日志口径） ==============
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)


def _compute_metrics_binary_01(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    y_true/y_pred 元素为 '0' 或 '1' 的字符串（与原代码对齐）。
    返回 human_rec, machine_rec, avg_rec, acc, precision, recall, f1

    注意：这里的人/机与 DeTeCtive 同口径：'1' 为 human，'0' 为 machine（由于前面做了翻转）
    """
    yt = np.array([int(x) for x in y_true], dtype=int)
    yp = np.array([int(x) for x in y_pred], dtype=int)

    acc = float(accuracy_score(yt, yp))
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
    # 分类别召回
    rep = classification_report(yt, yp, output_dict=True, zero_division=0)
    human_rec = float(rep.get("1", {}).get("recall", 0.0))
    machine_rec = float(rep.get("0", {}).get("recall", 0.0))
    avg_rec = (human_rec + machine_rec) / 2.0

    return dict(
        human_rec=human_rec,
        machine_rec=machine_rec,
        avg_rec=avg_rec,
        acc=acc,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1)
    )


# ============== FAISS Index（与原 DeTeCtive 一致） ==============
class Indexer(object):
    def __init__(self, vector_sz: int, device: str = "cuda"):
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(vector_sz)
        self.device = device
        try:
            if self.device == "cuda" and faiss.get_num_gpus() > 0:
                # 某些版本需要先 import 一下 torch_utils 才能正确初始化
                try:
                    import faiss.contrib.torch_utils  # noqa
                except Exception:
                    pass
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            else:
                self.device = "cpu"
        except Exception as e:
            print(f"[faiss] GPU init failed -> fallback to CPU. reason={e}")
            self.device = "cpu"
        try:
            n_threads = min(os.cpu_count() or 8, 16)
            faiss.omp_set_num_threads(n_threads)
            print(f"[faiss] omp threads = {faiss.omp_get_max_threads()}")
        except Exception:
            pass

    def index_data(self, ids: List[int], embeddings: np.ndarray, chunk: int = 20000):
        # 强制 float32 + C_CONTIGUOUS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32', copy=False)
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings, dtype='float32')

        self.index_id_to_db_id.extend(ids)
        # IndexFlat* 不需要 train；保留兼容
        if not self.index.is_trained:
            self.index.train(embeddings)

        # 分块 add，避免一次性大数组触发慢路径或内存抖动
        n = embeddings.shape[0]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            self.index.add(embeddings[s:e])

    def search_knn(self, query_vectors: np.ndarray, top_docs: int, index_batch_size: int = 8):
        query_vectors = query_vectors.astype("float32")
        result = []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        for k in tqdm(range(nbatch), leave=False, desc="KNN Search", dynamic_ncols=True):
            s = k * index_batch_size
            e = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[s:e]
            scores, indexes = self.index.search(q, top_docs)
            db_ids = [[str(self.index_id_to_db_id[i]) for i in row] for row in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def reset(self):
        self.index.reset()
        self.index_id_to_db_id = []


def _majority_vote(ids: List[str], label_dict: Dict[int, int]) -> str:
    z = sum(1 for _id in ids if label_dict[int(_id)] == 0)
    o = len(ids) - z
    return "0" if z > o else "1"


# ============== 文本编码模型（DeTeCtive 的 TextEmbeddingModel + 池化） ==============
class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, output_hidden_states: bool = False):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=output_hidden_states
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, last_hidden, attention_mask, use_pooling: str = "average", hidden_states: bool = False):
        if hidden_states:
            last_hidden.masked_fill_(~attention_mask[None, ..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = last_hidden.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = last_hidden[:, :, 0]
            emb = emb.permute(1, 0, 2)
        else:
            last_hidden.masked_fill_(~attention_mask[..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            else:  # "cls"
                emb = last_hidden[:, 0]
        return emb

    def forward(self, encoded_batch, use_pooling: str = "average", hidden_states: bool = False):
        if "t5" in self.model_name.lower():
            decoder_input_ids = torch.zeros(
                (encoded_batch["input_ids"].shape[0], 1),
                dtype=torch.long,
                device=encoded_batch["input_ids"].device,
            )
            out = self.model(**encoded_batch, decoder_input_ids=decoder_input_ids)
        else:
            out = self.model(**encoded_batch)

        if "bge" in self.model_name.lower() or "mxbai" in self.model_name.lower():
            use_pooling = "cls"

        if isinstance(out, tuple):
            last_hidden = out[0]
        elif isinstance(out, dict):
            if hidden_states:
                last_hidden = torch.stack(out["hidden_states"], dim=0)
            else:
                last_hidden = out["last_hidden_state"]
        else:
            last_hidden = out

        emb = self.pooling(last_hidden, encoded_batch["attention_mask"], use_pooling, hidden_states)
        emb = F.normalize(emb, dim=-1)
        return emb


# ============== 分类头 & DeTeCtive 两种损失的模型（AA 已删除） ==============
class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, in_dim // 4)
        self.dense2 = nn.Linear(in_dim // 4, in_dim // 16)
        self.out_proj = nn.Linear(in_dim // 16, out_dim)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)
        nn.init.normal_(self.dense2.bias, std=1e-6)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        return self.out_proj(x)


class SimCLR_Classifier_SCL(nn.Module):
    """单损失（SCL）版本"""

    def __init__(self, opt, fabric):
        super().__init__()
        self.temperature = opt.get("temperature", 0.07)
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt["embedding_model"])
        self.device = next(self.model.parameters()).device
        self.classifier = ClassificationHead(opt["projection_size"], opt["classifier_dim"])
        self.a = torch.tensor(opt.get("a", 1.0), device=self.device)
        self.d = torch.tensor(opt.get("d", 1.0), device=self.device)
        self.only_classifier = bool(opt.get("only_classifier", False))
        self.eps = torch.tensor(1e-6, device=self.device)

    def get_encoder(self):
        return self.model

    def _sim(self, q, k):
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        return (qn @ kn.T) / self.temperature

    def _compute_logits_label(self, q, q_label, k, k_label):
        logits = self._sim(q, k)  # [N, NK]
        ql = q_label.view(-1, 1)  # [N, 1]
        kl = k_label.view(1, -1)  # [1, NK]
        same = (ql == kl)         # [N, NK]
        pos = torch.sum(logits * same, dim=1) / torch.max(torch.sum(same, dim=1), self.eps)
        neg = logits * (~same)
        logits_label = torch.cat([pos.unsqueeze(1), neg], dim=1)
        return logits_label

    def forward(self, encoded_batch, labels_de):
        # q, k 聚合（Fabric all_gather）
        q = self.model(encoded_batch)
        k = q.detach().clone()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(labels_de).view(-1)

        logits_label = self._compute_logits_label(q, labels_de, k, k_label)
        out = self.classifier(q)
        loss_cls = F.cross_entropy(out, labels_de)
        gt = torch.zeros(logits_label.size(0), dtype=torch.long, device=logits_label.device)
        loss_scl = torch.tensor(0.0, device=self.device) if self.only_classifier else F.cross_entropy(logits_label, gt)
        loss = self.a * loss_scl + self.d * loss_cls

        if self.training:
            return loss, loss_scl, loss_cls, k, k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss, out, k, k_label


class SimCLR_Classifier(nn.Module):
    """多层对比学习（DeTeCtive 主体，human/machine + model + model_set + label，多头损失）"""

    def __init__(self, opt, fabric):
        super().__init__()
        self.temperature = opt.get("temperature", 0.07)
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt["embedding_model"])
        self.device = next(self.model.parameters()).device
        self.classifier = ClassificationHead(opt["projection_size"], opt["classifier_dim"])
        self.only_classifier = bool(opt.get("only_classifier", False))
        self.a = torch.tensor(opt.get("a", 1.0), device=self.device)
        self.b = torch.tensor(opt.get("b", 1.0), device=self.device)
        self.c = torch.tensor(opt.get("c", 1.0), device=self.device)
        self.d = torch.tensor(opt.get("d", 1.0), device=self.device)
        self.eps = torch.tensor(1e-6, device=self.device)

    def get_encoder(self):
        return self.model

    def _sim(self, q, k):
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        return (qn @ kn.T) / self.temperature

    def _compute_logits(self, q, qi_model, qi_set, qi_label, k, kk_model, kk_set, kk_label):
        # 余弦相似度 + 温度缩放
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        logits = (qn @ kn.T) / self.temperature  # [N, N_all]

        # 维度对齐
        qi_model = qi_model.view(-1, 1)  # [N,1]
        qi_set = qi_set.view(-1, 1)
        qi_label = qi_label.view(-1, 1)
        kk_model = kk_model.view(1, -1)  # [1,N_all]
        kk_set = kk_set.view(1, -1)
        kk_label = kk_label.view(1, -1)

        # 基本关系
        same_model = (qi_model == kk_model)  # [N,N_all]
        same_set = (qi_set == kk_set)
        same_label = (qi_label == kk_label)

        is_human = (qi_label.squeeze(1) == 1)   # 1 = human（与官方一致）
        is_machine = (qi_label.squeeze(1) == 0)

        # —— 按“官方口径”分别定义 各头 的正/负样本掩码 ——
        # # Lm: 正 = same_model；负 = ~same_model（仅机器anchor参与）
        pos_m = same_model
        neg_m = ~same_model

        # Ls: 正 = xor(same_set, same_model)；负 = ~same_set（仅机器anchor参与）
        pos_s = torch.logical_xor(same_set, same_model)
        neg_s = ~same_set

        # Ll: 正 = xor(same_set, same_label)；负 = ~same_label（仅机器anchor参与）
        pos_l = torch.logical_xor(same_set, same_label)
        neg_l = ~same_label

        # Lh: 正 = same_label；负 = ~same_label（仅人类anchor参与）
        pos_h = same_label
        neg_h = ~same_label

        def _build_logits(pos_mask, neg_mask):
            # 分子：正样本平均相似度；分母避免除0与数值不稳
            denom = torch.max(pos_mask.sum(dim=1), self.eps)
            pos = (logits * pos_mask).sum(dim=1) / denom  # [N]
            neg = logits * neg_mask                        # [N, N_all]
            return torch.cat([pos.unsqueeze(1), neg], dim=1)

        # 各头 logits，并在“anchor 侧”筛选参与者
        lg_model = _build_logits(pos_m, neg_m)[is_machine]   # 机器anchor
        lg_set = _build_logits(pos_s, neg_s)[is_machine]
        lg_label = _build_logits(pos_l, neg_l)[is_machine]
        lg_human = _build_logits(pos_h, neg_h)[is_human]     # 人类anchor

        return lg_model, lg_set, lg_label, lg_human

    def forward(self, encoded_batch, model_idx, set_idx, labels_de):
        q = self.model(encoded_batch)
        k = q.detach().clone()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(labels_de).view(-1)
        k_model = self.fabric.all_gather(model_idx).view(-1)
        k_set = self.fabric.all_gather(set_idx).view(-1)

        lg_model, lg_set, lg_label, lg_human = self._compute_logits(
            q, model_idx, set_idx, labels_de, k, k_model, k_set, k_label
        )
        out = self.classifier(q)
        loss_cls = F.cross_entropy(out, labels_de)

        gt_model = torch.zeros(lg_model.size(0), dtype=torch.long, device=lg_model.device) if lg_model.numel() else None
        gt_set = torch.zeros(lg_set.size(0), dtype=torch.long, device=lg_set.device) if lg_set.numel() else None
        gt_label = torch.zeros(lg_label.size(0), dtype=torch.long, device=lg_label.device) if lg_label.numel() else None
        gt_human = torch.zeros(lg_human.size(0), dtype=torch.long, device=lg_human.device) if lg_human.numel() else None

        loss_model = F.cross_entropy(lg_model, gt_model) if gt_model is not None else torch.tensor(0.0, device=self.device)
        loss_set = F.cross_entropy(lg_set, gt_set) if gt_set is not None else torch.tensor(0.0, device=self.device)
        loss_label = F.cross_entropy(lg_label, gt_label) if gt_label is not None else torch.tensor(0.0, device=self.device)
        loss_human = F.cross_entropy(lg_human.to(torch.float64), gt_human) if gt_human is not None else torch.tensor(0.0, device=self.device)

        loss = self.a*loss_model + self.b*loss_set + self.c*loss_label + (self.a+self.b+self.c)*loss_human + self.d*loss_cls

        if self.training:
            return loss, loss_model, loss_set, loss_label, loss_human, loss_cls, k, k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss, out, k, k_label


# ============== 统一数据集封装（由 load_dataset_unified 提供） ==============
class _UnifiedDeTDS(Dataset):
    """
    将 mgt_eval 的统一样本转为 DeTeCtive 期望的字段：
    - text: str
    - label_de: int (DeTeCtive 内部：1=human, 0=machine) <-- 由 mgt_eval 的 0(human)/1(ai) 翻转得到
    - model_idx: int (生成器/模型名索引；human 样本统一为 'human')
    - set_idx: int (模型集合/域索引；human 样本统一为 'human')
    """
    def __init__(self, exs: List[Dict[str, Any]], model_vocab: Dict[str, int], set_vocab: Dict[str, int],
                 tok: AutoTokenizer, max_length: int = 512):
        self.exs = exs
        self.model_vocab = model_vocab
        self.set_vocab = set_vocab
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, idx: int):
        e = self.exs[idx]
        text = e["text"]
        y_mg = int(e["label"])  # 0=human, 1=ai (mgt_eval 默认)
        y_de = 1 - y_mg         # DeTeCtive 内部用 1=human, 0=machine
        assert y_de in (0, 1)

        # 在 __getitem__ 里把 gen/mset 的候选键再多加几类根级字段
        meta = e.get("meta", {}) or {}

        # 先看根级
        gen_root = e.get("src") or e.get("model") or e.get("generator") or e.get("LLM_name")
        set_root = e.get("dataset") or e.get("domain") or e.get("source") or e.get("collection") or None

        # 再看 meta
        gen_meta = meta.get("generator") or meta.get("model_name")
        set_meta = meta.get("dataset") or meta.get("domain") or meta.get("source")

        gen = gen_root or gen_meta or ("human" if y_mg == 0 else "default")
        mset = set_root or set_meta or ("human" if y_mg == 0 else "default")

        model_idx = self.model_vocab.get(str(gen), self.model_vocab["default"])
        set_idx = self.set_vocab.get(str(mset), self.set_vocab["default"])

        return {
            "text": text,
            "label_de": torch.tensor(y_de, dtype=torch.long),
            "model_idx": torch.tensor(model_idx, dtype=torch.long),
            "set_idx": torch.tensor(set_idx, dtype=torch.long),
        }


def _build_vocabs(exs: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    models, sets_ = set(["default", "human"]), set(["default", "human"])
    for e in exs:
        y = int(e["label"])
        meta = e.get("meta", {}) or {}
        gen = meta.get("generator") or meta.get("model_name") or ("human" if y == 0 else "default")
        mset = meta.get("dataset") or meta.get("domain") or meta.get("source") or ("human" if y == 0 else "default")
        models.add(str(gen)); sets_.add(str(mset))
    model_vocab = {m: i for i, m in enumerate(sorted(models))}
    set_vocab = {s: i for i, s in enumerate(sorted(sets_))}
    return model_vocab, set_vocab


# === 新增：单个数据源加载助手 ===
def _load_one_dataset_spec(spec: str, seed: int) -> List[Dict[str, Any]]:
    """
    尝试按官方实现读取（包含 text/label/model 的 JSONL 或 Deepfake/Turing/OUTFOX/M4 目录）；
    失败则退回 mgt_eval 的统一加载器。
    返回统一样本列表：{\"text\": str, \"label\": int(0=human,1=ai), \"meta\": {...}}
    """
    # 若你之前已实现 _try_official_dataset_load(spec) -> Optional[List[Dict]], 这里直接复用
    try:
        official = _try_official_dataset_load(spec)  # 若无匹配则返回 None
    except NameError:
        official = None
    if official is not None:
        return official

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

# ---- 新增：按给定数量随机截断样本（训练 / 验证集通用） ----
def _limit_examples(examples: List[Dict[str, Any]],
                    limit: Optional[int],
                    seed: int = 114514) -> List[Dict[str, Any]]:
    """
    若 limit 为正且小于当前样本数，则按随机子集抽取 limit 个样本；
    否则原样返回。
    """
    if limit is None or limit <= 0 or limit >= len(examples):
        return examples
    rng = np.random.RandomState(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    idx = idx[:limit]
    return [examples[i] for i in idx]

# ============== Lightning Fabric（可选） ==============
_FABRIC_AVAILABLE = True
try:
    from lightning import Fabric
    from lightning.fabric.strategies import DDPStrategy
except Exception:
    _FABRIC_AVAILABLE = False
    Fabric = None
    DDPStrategy = None

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainCfg:
    dataset_training: str
    dataset_validation: Optional[str] = None  # <--- 新增：可选验证集
    output_dir: str = "runs_detective"
    embedding_model: str = "princeton-nlp/unsup-simcse-roberta-base"
    projection_size: int = 768
    classifier_dim: int = 2
    temperature: float = 0.07
    total_epoch: int = 50
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    warmup_steps: int = 2000
    lr: float = 2e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    d: float = 1.0
    one_loss: bool = False
    only_classifier: bool = False
    freeze_embedding_layer: bool = False
    topk: int = 10
    max_length: int = 512
    seed: int = 114514
    devices: int = 1
    # ---- 新增：限制训练/验证集样本数量（None 表示不限制） ----
    train_sample_limit: Optional[int] = None
    val_sample_limit: Optional[int] = None

# ============== 训练主过程（风格与 greater.py 对齐：进度条、元数据、产物结构） ==============
def _train_detective(cfg: TrainCfg, **kwargs) -> Dict[str, Any]:
    _seed_everything(cfg.seed)
    device = _device()
    torch.set_grad_enabled(True)

    # ===== 日志头，与 greater 对齐 =====
    print(f"[mgt_eval] Using detector: {DETECTOR_NAME} (Type={detector_type})")
    print(f"[mgt_eval] Credits: {CITATION_AUTHORS} | Paper: {CITATION_TITLE} | Link: {CITATION_LINK}")
    print("[mgt_eval] Disclaimer: This implementation may differ slightly from the original reference; "
          "results might not exactly match those reported in the paper.")
    print(f"[mgt_eval] Device: {device}")

    # ===== 输出根目录 & 环境元数据 =====
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
            "data": _build_data_info(cfg.dataset_training, None, None),
        }, f, ensure_ascii=False, indent=2)

    # ====== 数据集载入（支持 --dataset_training 逗号分隔多个数据源） ======
    # 例：--dataset_training "/path/monolingual_train.jsonl,/path/monolingual_dev.jsonl"
    train_specs = [s.strip() for s in str(cfg.dataset_training).split(",") if s.strip()]
    if len(train_specs) == 0:
        raise ValueError("dataset_training 为空；请至少提供一个数据源路径或别名。")

    # 合并多个训练源
    train_all: List[Dict[str, Any]] = []
    for spec in train_specs:
        exs = _load_one_dataset_spec(spec, seed=cfg.seed)
        if not isinstance(exs, list) or len(exs) == 0:
            print(f"[warn] no samples loaded from: {spec}")
        else:
            print(f"[data] loaded {len(exs)} samples from: {spec}")
            train_all.extend(exs)

    if cfg.dataset_validation:
        # 验证集仍按单源（也可自己传多源逗号分隔；这里一并支持）
        val_specs = [s.strip() for s in str(cfg.dataset_validation).split(",") if s.strip()]
        val_all: List[Dict[str, Any]] = []
        for vs in val_specs:
            vexs = _load_one_dataset_spec(vs, seed=cfg.seed)
            print(f"[data] loaded {len(vexs)} val samples from: {vs}")
            val_all.extend(vexs)
        tr = train_all
        va = val_all
    else:
        # 未显式提供验证集：对合并后的训练集做 9:1 分层切分
        tr, va, _ = _stratified_split(train_all, 9.0, 1.0, 0.0, seed=cfg.seed)
        print(f"[data] split merged train set -> train={len(tr)}, val={len(va)}")

    # ---- 新增：根据 train_sample_limit / val_sample_limit 随机截断样本数 ----
    tr = _limit_examples(tr, cfg.train_sample_limit, seed=cfg.seed)
    va = _limit_examples(va, cfg.val_sample_limit, seed=cfg.seed)
    print(f"[data] final train={len(tr)}, val={len(va)} (after applying sample limits)")

    # （其后保持不变：tok/model_vocab/set_vocab/dataloaders 构建）
    tok = AutoTokenizer.from_pretrained(cfg.embedding_model, use_fast=True, trust_remote_code=True)
    model_vocab, set_vocab = _build_vocabs(tr + va)
    ds_tr = _UnifiedDeTDS(tr, model_vocab, set_vocab, tok, max_length=cfg.max_length)
    ds_va = _UnifiedDeTDS(va, model_vocab, set_vocab, tok, max_length=cfg.max_length)

    # 词表 & DataLoader 构建前，先把“扫描到的模型类别”记下来，稍后在 rank0 打印
    _scanned_model_classes = _scan_generator_classes(tr + va)

    tok = AutoTokenizer.from_pretrained(cfg.embedding_model, use_fast=True, trust_remote_code=True)
    model_vocab, set_vocab = _build_vocabs(tr + va)
    ds_tr = _UnifiedDeTDS(tr, model_vocab, set_vocab, tok, max_length=cfg.max_length)
    ds_va = _UnifiedDeTDS(va, model_vocab, set_vocab, tok, max_length=cfg.max_length)

    def _collate(examples):
        texts = [b["text"] for b in examples]
        enc = tok.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_de": torch.stack([b["label_de"] for b in examples], dim=0),
            "model_idx": torch.stack([b["model_idx"] for b in examples], dim=0),
            "set_idx": torch.stack([b["set_idx"] for b in examples], dim=0),
        }

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        collate_fn=_collate,
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
        collate_fn=_collate,
        drop_last=False,
    )

    # ====== Fabric（保留原论文并行化思路） ======
    if _FABRIC_AVAILABLE and cfg.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", devices=cfg.devices, precision="bf16-mixed", strategy=strategy)
    elif _FABRIC_AVAILABLE and cfg.devices == 1 and torch.cuda.is_available():
        fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    else:
        class _DummyFabric:
            def setup(self, *objs):
                return objs
            def setup_dataloaders(self, *dls):
                return dls
            def backward(self, loss):
                loss.backward()
            def all_gather(self, x):
                return x
            @property
            def global_rank(self):
                return 0
            def launch(self):
                pass
            def barrier(self):
                pass
        fabric = _DummyFabric()

    if hasattr(fabric, "launch"):
        fabric.launch()

    # === 像官方实现一样打印扫描到的模型类别（仅 rank0 输出一次） ===
    try:
        if getattr(fabric, "global_rank", 0) == 0:
            # 尝试从路径中给个“模式”名（尽量贴近官方 'deepfake' / 'TuringBench' / 'M4' / 'OUTFOX' 风格）
            from pathlib import Path
            mode_hint = Path(str(cfg.dataset_training)).stem.lower()
            # 简单归一：遇到典型关键词就替换更干净的名字
            if "turing" in mode_hint:
                mode_hint = "Turing"
            elif "deepfake" in mode_hint:
                mode_hint = "deepfake"
            elif "outfox" in mode_hint:
                mode_hint = "OUTFOX"
            elif "m4" in mode_hint:
                mode_hint = "M4"
            classes = _scanned_model_classes
            print(f"there are {len(classes)} classes in {mode_hint} dataset")
            print(f"the classes are {classes}")
    except Exception:
        # 打印失败不影响训练
        pass

    # ====== 构建模型 ======
    opt = {
        "embedding_model": cfg.embedding_model,
        "projection_size": cfg.projection_size,
        "classifier_dim": cfg.classifier_dim,
        "temperature": cfg.temperature,
        "a": cfg.a,
        "b": cfg.b,
        "c": cfg.c,
        "d": cfg.d,
        "only_classifier": cfg.only_classifier,
    }
    
    if cfg.one_loss:
        model = SimCLR_Classifier_SCL(opt, fabric).train()
    else:
        model = SimCLR_Classifier(opt, fabric).train()

    # 冻结 embeddings（可选）
    if cfg.freeze_embedding_layer:
        for n, p in model.model.named_parameters():
            if "emb" in n:
                p.requires_grad = False

    # ====== 优化器/调度器（沿用 DeTeCtive 代码逻辑） ======
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)

    updates_per_epoch = len(dl_tr)
    total_steps = cfg.total_epoch * updates_per_epoch
    warmup_steps = int(cfg.warmup_steps)
    # 用余弦退火（warmup 手动线性上升）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.lr / 10.0)

    # Fabric setup
    model, optimizer = fabric.setup(model, optimizer)
    dl_tr, dl_va = fabric.setup_dataloaders(dl_tr, dl_va)

    # ====== 产物目录结构（与 greater.py 一致 best/last 等） ======
    run_dir = os.path.join(out_root, "detector")  # 与 greater 内部的二级目录类似
    os.makedirs(run_dir, exist_ok=True)
    run_best = os.path.join(run_dir, "best"); os.makedirs(run_best, exist_ok=True)
    run_last = os.path.join(run_dir, "last"); os.makedirs(run_last, exist_ok=True)

    # ====== 显存峰值统计上下文 ======
    mem_ctx = _reset_and_mark_cuda_peaks()

    # ====== 训练循环（进度条样式与 greater 一致） ======
    global_step, best_metric, best_epoch = 0, -1.0, -1
    step_indices, step_losses = [], []
    total_wall_start = time.perf_counter()

    # ====== 训练循环（官方 DeTeCtive tqdm 样式） ======
    global_step, best_metric, best_epoch = 0, -1.0, -1
    step_indices, step_losses = [], []
    total_wall_start = time.perf_counter()

    for ep in range(1, cfg.total_epoch + 1):
        model.train()
        avg_loss = 0.0
        # 用于 KNN 的训练期聚合
        all_emb_tr, all_lab_tr = [], []

        num_batches_per_epoch = len(dl_tr)
        warmup_steps = int(cfg.warmup_steps)

        # 官方样式：先打印表头，再用 tqdm 包裹 enumerate(dataloader)
        if getattr(fabric, "global_rank", 0) == 0:
            if cfg.one_loss:
                print("Train with one loss!")
                # Ep, Mem, L(总损失), Lscl(对比损失), Lc(分类头), avg(滑动), lr, step
                print("\n" +
                      f"{'Epoch':>{W_EPOCH}}{SEP}"
                      f"{'GPU_mem':>{W_MEM}}{SEP}"
                      f"{'L':>{W_NUM}}{SEP}"
                      f"{'Lscl':>{W_NUM}}{SEP}"
                      f"{'Lc':>{W_NUM}}{SEP}"
                      f"{'avg':>{W_NUM}}{SEP}"
                      f"{'lr':>{W_NUM}}{SEP}"
                      f"{'step':>{W_STEP}}")
            else:
                # Ep, Mem, L(总损失), Lm(模型对比), Ls(集合对比), Ll(标签对比), Lc(分类头), Lh(人类对比), avg, lr, step
                print("\n" +
                      f"{'Epoch':>{W_EPOCH}}{SEP}"
                      f"{'GPU_mem':>{W_MEM}}{SEP}"
                      f"{'L':>{W_NUM}}{SEP}"
                      f"{'Lm':>{W_NUM}}{SEP}"
                      f"{'Ls':>{W_NUM}}{SEP}"
                      f"{'Ll':>{W_NUM}}{SEP}"
                      f"{'Lc':>{W_NUM}}{SEP}"
                      f"{'Lh':>{W_NUM}}{SEP}"
                      f"{'avg':>{W_NUM}}{SEP}"
                      f"{'lr':>{W_NUM}}{SEP}"
                      f"{'step':>{W_STEP}}")
            pbar = tqdm(enumerate(dl_tr), total=num_batches_per_epoch, dynamic_ncols=True)
        else:
            pbar = enumerate(dl_tr)

        for i, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            current_step = (ep - 1) * num_batches_per_epoch + i

            # 线性 warmup
            if current_step < warmup_steps:
                lr_now = cfg.lr * float(current_step + 1) / float(max(1, warmup_steps))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_now
            current_lr = optimizer.param_groups[0]['lr']

            encoded = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            }
            lbl = batch["label_de"].to(device, non_blocking=True)

            if cfg.one_loss:
                loss, loss_scl, loss_cls, k, k_lbl = model(encoded, lbl)
            else:
                loss, loss_m, loss_s, loss_l, loss_h, loss_cls, k, k_lbl = model(
                    encoded, batch["model_idx"].to(device), batch["set_idx"].to(device), lbl
                )

            fabric.backward(loss)
            optimizer.step()

            # warmup 之后走余弦退火
            if current_step >= warmup_steps:
                scheduler.step()

            # 统计/显示
            global_step += 1
            step_indices.append(global_step)
            step_losses.append(float(loss.item()))

            if getattr(fabric, "global_rank", 0) == 0:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                avg_loss = (avg_loss * i + float(loss.item())) / (i + 1)

                # 官方 set_description 模版
                if cfg.one_loss:
                    # 单损失：L(总), Lscl(对比), Lc(分类头)
                    desc = (
                        f"{f'{ep}/{cfg.total_epoch}':>{W_EPOCH}}{SEP}"
                        f"{mem:>{W_MEM}}{SEP}"
                        f"{float(loss.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_scl.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_cls.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(avg_loss):>{W_NUM}.4f}{SEP}"
                        f"{float(current_lr):>{W_NUM}.2e}{SEP}"
                        f"{int((ep - 1) * num_batches_per_epoch + i):>{W_STEP}d}"
                    )
                    if hasattr(pbar, "set_description"):
                        pbar.set_description(desc)
                else:
                    desc = (
                        f"{f'{ep}/{cfg.total_epoch}':>{W_EPOCH}}{SEP}"
                        f"{mem:>{W_MEM}}{SEP}"
                        f"{float(loss.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_m.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_s.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_l.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_cls.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(loss_h.item()):>{W_NUM}.4f}{SEP}"
                        f"{float(avg_loss):>{W_NUM}.4f}{SEP}"
                        f"{float(current_lr):>{W_NUM}.2e}{SEP}"
                        f"{int((ep - 1) * num_batches_per_epoch + i):>{W_STEP}d}"
                    )
                    if hasattr(pbar, "set_description"):
                        pbar.set_description(desc)

            # 仅 rank0 聚合训练 embedding（用于后续 KNN 验证）
            all_emb_tr.append(k.detach().cpu())
            all_lab_tr.extend(k_lbl.detach().cpu().tolist())

        # ====== 验证（官方 DeTeCtive tqdm 样式） ======
        torch.cuda.empty_cache()
        fabric.barrier()

        with torch.no_grad():
            model.eval()
            test_loss = 0.0

            # 先构建训练索引（仅 rank0）
            if getattr(fabric, "global_rank", 0) == 0:
                print("Build torch-knn db embeddings...")
                if len(all_emb_tr) > 0:
                    emb_tr = torch.cat(all_emb_tr, dim=0)
                    emb_tr = emb_tr / (emb_tr.norm(dim=1, keepdim=True) + 1e-6)
                    emb_db_np = emb_tr.contiguous().cpu().numpy().astype(np.float32, copy=False)
                    labels_db = [int(x) for x in all_lab_tr]
                else:
                    emb_db_np, labels_db = None, []


            # 准备 tqdm 样式
            if getattr(fabric, "global_rank", 0) == 0:
                print("\n" +
                      f"{'Epoch':>{W_EPOCH}}{SEP}"
                      f"{'GPU_mem':>{W_MEM}}{SEP}"
                      f"{'Cur_acc':>{W_NUM}}{SEP}"
                      f"{'avg_acc':>{W_NUM}}{SEP}"
                      f"{'loss':>{W_NUM}}")
                pbar_val = tqdm(enumerate(dl_va), total=len(dl_va), dynamic_ncols=True)
            else:
                pbar_val = enumerate(dl_va)

            right_num, tot_num = 0, 0
            all_emb_va, all_lbl_va = [], []

            for j, batch in pbar_val:
                encoded = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                }
                lbl = batch["label_de"].to(device, non_blocking=True)

                if cfg.one_loss:
                    vloss, out, k, k_lbl = model(encoded, lbl)
                else:
                    vloss, out, k, k_lbl = model(
                        encoded, batch["model_idx"].to(device), batch["set_idx"].to(device), lbl
                    )

                preds = torch.argmax(out, dim=1)
                cur_right = (preds == k_lbl).sum().item()
                cur_tot = k_lbl.numel()
                right_num += cur_right
                tot_num += cur_tot
                test_loss = (test_loss * j + float(vloss.item())) / (j + 1)

                if getattr(fabric, "global_rank", 0) == 0:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                    cur_acc = float(cur_right) / max(1, cur_tot)
                    avg_acc = float(right_num) / max(1, tot_num)
                    desc = (
                        f"{f'{ep}/{cfg.total_epoch}':>{W_EPOCH}}{SEP}"
                        f"{mem:>{W_MEM}}{SEP}"
                        f"{float(cur_acc):>{W_NUM}.4f}{SEP}"
                        f"{float(avg_acc):>{W_NUM}.4f}{SEP}"
                        f"{float(vloss.item()):>{W_NUM}.4f}"
                    )
                    if hasattr(pbar_val, "set_description"):
                        pbar_val.set_description(desc)

                all_emb_va.append(k.detach().cpu())
                all_lbl_va.extend(k_lbl.detach().cpu().tolist())

            torch.cuda.empty_cache()
            fabric.barrier()

        if getattr(fabric, "global_rank", 0) == 0:
            clf_acc = float(right_num) / max(1, tot_num)

            print("Search knn with torch...")
            if len(all_emb_va) > 0 and emb_db_np is not None and len(labels_db) == emb_db_np.shape[0]:
                embv = torch.cat(all_emb_va, dim=0)
                embv = embv / (embv.norm(dim=1, keepdim=True) + 1e-6)
                emb_q_np = embv.contiguous().cpu().numpy().astype(np.float32, copy=False)

                device_knn = "cuda" if torch.cuda.is_available() else "cpu"
                preds = _torch_knn_majority_vote(
                    emb_db_np, labels_db, emb_q_np, topk=cfg.topk, device=device_knn, q_block=2048
                )
                reals = [str(x) for x in all_lbl_va]
                m = _compute_metrics_binary_01(reals, preds)
            else:
                m = dict(human_rec=0.0, machine_rec=0.0, avg_rec=clf_acc, acc=clf_acc,
                        precision=0.0, recall=0.0, f1=0.0)
            print("Search knn done!")

            cur_metric = float(m.get("avg_rec", clf_acc))
            if cur_metric >= best_metric:
                best_metric, best_epoch = cur_metric, ep
                torch.save(model.get_encoder().state_dict(), os.path.join(run_best, "model_best.pth"))
                torch.save(model.state_dict(), os.path.join(run_best, "model_classifier_best.pth"))
            torch.save(model.get_encoder().state_dict(), os.path.join(run_last, "model_last.pth"))
            torch.save(model.state_dict(), os.path.join(run_last, "model_classifier_last.pth"))

            print(f"[{DETECTOR_NAME}][Epoch {ep}] "
                  f"train_loss={avg_loss:.4f} "
                  f"val(clf_acc)={clf_acc:.4f} val(avg_rec)={m.get('avg_rec',0.0):.4f} "
                  f"best(avg_rec)={best_metric:.4f}@{best_epoch}")

        fabric.barrier()

    # ====== 训练结束：显存峰值、loss 图、summary ======
    mem_stats = _collect_cuda_peaks(mem_ctx)
    loss_plot = _save_loss_plot(step_indices, step_losses, out_dir=out_root, filename="train_loss.png", smooth_window=0)
    total_wall_time = time.perf_counter() - total_wall_start

    summary = {
        "best_dir": run_best,
        "last_dir": run_last,
        "best_val_metric_avg_rec": best_metric,
        "history": [],  # 为简洁，这里不逐 epoch 存表；如需可在上面记录
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
            "best_val_acc": summary.get("best_val_metric_avg_rec", None),
            "artifacts": summary["artifacts"],
            "output_root": out_root,
        }
    }


# ============== 统一暴露的注册入口（风格与 greater.py 完全一致） ==============
@register_train("detective")
def train_detective(**kwargs) -> Dict[str, Any]:
    cfg = TrainCfg(
        dataset_training=kwargs.get("dataset_training"),
        dataset_validation=kwargs.get("dataset_validation", None),  # <--- 新增
        output_dir=kwargs.get("output_dir", "runs_detective"),
        embedding_model=kwargs.get("embedding_model", "princeton-nlp/unsup-simcse-roberta-base"),
        projection_size=kwargs.get("projection_size", 768),
        classifier_dim=kwargs.get("classifier_dim", 2),
        temperature=kwargs.get("temperature", 0.07),
        total_epoch=kwargs.get("epochs", kwargs.get("total_epoch", 50)),
        train_batch_size=kwargs.get("train_batch_size", 32),
        eval_batch_size=kwargs.get("eval_batch_size", 64),
        num_workers=kwargs.get("num_workers", 4),
        warmup_steps=kwargs.get("warmup_steps", 2000),
        lr=kwargs.get("lr", 2e-5),
        weight_decay=kwargs.get("weight_decay", 1e-4),
        beta1=kwargs.get("beta1", 0.9),
        beta2=kwargs.get("beta2", 0.98),
        eps=kwargs.get("eps", 1e-6),
        a=kwargs.get("a", 1.0),
        b=kwargs.get("b", 1.0),
        c=kwargs.get("c", 1.0),
        d=kwargs.get("d", 1.0),
        one_loss=kwargs.get("one_loss", False),
        only_classifier=kwargs.get("only_classifier", False),
        freeze_embedding_layer=kwargs.get("freeze_embedding_layer", False),
        topk=kwargs.get("topk", 10),
        max_length=kwargs.get("max_length", 512),
        seed=kwargs.get("seed", 114514),
        devices=kwargs.get("devices", 1),
        # ---- 新增：从 kwargs 读取限制参数 ----
        train_sample_limit=kwargs.get("train_sample_limit", None),
        val_sample_limit=kwargs.get("val_sample_limit", None),
    )
    assert cfg.dataset_training, "detective 需要 dataset_training 参数（可被 load_dataset_unified 解析）"
    return _train_detective(cfg, **kwargs)

# 便于脚本快速调用
def DeTeCtive(**kwargs) -> Dict[str, Any]:
    return _train_detective(**kwargs)