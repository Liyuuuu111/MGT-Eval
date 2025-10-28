# mgt_eval/detectors/finetuned/prdetect.py
# -*- coding: utf-8 -*-
"""
PRDetect (严格按论文实现封装版)
- **核心函数未改动**：build_graph / GCN2 / 逐样本训练流程
- 图构建：spaCy 依存边 + RoBERTa token embedding（按论文代码）
- 训练：逐文档前向、BCELoss、Adam(1e-4)、阈值0.5、文档级均值池化
- 评测：输出概率即 p(human|x)（与论文一致，label=1 表示 human）
- 工程化：注册到 mgt_eval（@register_train/@register），支持 jsonl 和 sample_k，产出 best/prdetect_gcn.pt

依赖：
    pip install spacy transformers torch torch-geometric tqdm matplotlib
    python -m spacy download en_core_web_sm
    # torch-geometric 请按 CUDA 版本选择官方指令安装 wheels
"""

from __future__ import annotations
import os
import json
import time
import random
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import spacy
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

#（mgt_eval）注册/评测基类
try:
    from mgt_eval.detectors.base import DetectorBase, EvalResult
    from mgt_eval.detectors.registry import register
    from mgt_eval.train.registry import register_train
    from mgt_eval.eval.evaluator import evaluate_detector
except Exception as e:
    raise RuntimeError("请确保本文件置于 mgt_eval/detectors/ 下并在同一工程内运行。") from e

# 可选：绘图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ====================== 实用工具 ======================
def _seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _device_str(dev: Optional[str]) -> str:
    return dev if dev else ("cuda" if torch.cuda.is_available() else "cpu")

def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_loss_curve(steps: List[int], losses: List[float], out_dir: str, fname: str):
    if not (_HAS_MPL and losses): return None
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(steps, losses); ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_title("Training Loss (BCE)"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
    return path

def _load_jsonl_lines(jsonl_path: str) -> List[str]:
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines

def _label_human_is_one(lbl: Any) -> int:
    s = (str(lbl) if lbl is not None else "").lower()
    if "human" in s: return 1
    if "ai" in s or "machine" in s or "chatgpt" in s: return 0
    try:
        v = int(lbl); return 1 if v == 1 else 0
    except Exception:
        return 0


# ====================== 论文原始核心：全局 backbone + build_graph ======================
# ！！注意：下方 build_graph 函数**保持与论文代码一致**（不修改核心逻辑/语义）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = None
model = None
tokenizer = None

def _ensure_backbone(roberta_path: str):
    """初始化论文同款 backbone：spaCy + RoBERTa + tokenizer（vocab.json/merges.txt）"""
    global nlp, model, tokenizer, device
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    # 严格按论文：从本地目录加载
    if model is None or tokenizer is None:
        if not os.path.isdir(roberta_path):
            raise FileNotFoundError(f"[PRDetect] 期望 roberta_path 为本地目录：{roberta_path}")
        model_path = roberta_path
        vocab_file = os.path.join(roberta_path, "vocab.json")
        merges_file = os.path.join(roberta_path, "merges.txt")
        if not (os.path.isfile(vocab_file) and os.path.isfile(merges_file)):
            raise FileNotFoundError(f"[PRDetect] 未找到 {vocab_file} 或 {merges_file}")
        mdl = RobertaModel.from_pretrained(model_path).to(device)
        tok = RobertaTokenizer(vocab_file, merges_file, use_fast=False)
        # 赋到全局
        globals()["model"] = mdl
        globals()["tokenizer"] = tok

# -------------- 论文给出的 build_graph（保持核心不变）--------------
import pickle
import numpy as np
from tqdm import tqdm  # 已导入
from scipy.sparse import csr_matrix  # 保留与论文一致（未使用）
import time
import os as _os

def build_graph(json_texts):
    start_time = time.time()
    texts = list()
    y = list()
    for json_text in json_texts:
        texts.append(json.loads(json_text)['text'])
        label = 1 if "human" in json.loads(json_text)['label'] else 0
        y.append(label)
    y = torch.tensor(y, dtype=torch.float32)
    tokenized_sentences = list()
    all_token_embeddings = list()
    all_edge_index = list()
    all_sparse_adj_matrix = list()
    for text in tqdm(texts):
        try:
            doc = nlp(text)
            tokenized_sentence = [token.text for token in doc]
            tokenized_sentences.append(tokenized_sentence)
            
            max_length = 512
            chunks = [tokenized_sentence[i:i+max_length] for i in range(0, len(tokenized_sentence), max_length)]
            chunk_outputs = []
            for chunk in chunks:
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_ids)

                last_hidden_states = output.last_hidden_state
                token_embeddings = last_hidden_states[0]
                chunk_outputs.append(token_embeddings)
            token_embeddings = torch.cat(chunk_outputs, dim=0)
            all_token_embeddings.append(token_embeddings)

            node_relations = list()
            for word in doc:        
                node_relations.append([word.i,word.head.i])
            edge0 = list()
            edge1 = list()
            for edge in node_relations:
                edge0.append(edge[0])
                edge1.append(edge[1])
            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            all_edge_index.append(edge_index)
            # sparse_adj_matrix = csr_matrix((np.ones(len(edge0)),(np.array(edge0), np.array(edge1))),shape=(len(tokenized_sentence),len(tokenized_sentence)))
            # all_sparse_adj_matrix.append(sparse_adj_matrix)
        except Exception as e:
            print(text)
            print(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time} 秒")
    return all_token_embeddings, all_edge_index, y

#（论文中的 I/O 小工具，原样保留）
def read_json(file_name):
    texts = list()
    with open(f"original_text/{file_name}.json", "r", encoding="utf-8") as f:
        for line in f.readlines():
            texts.append(line)
    return texts

def save_pkl(file_name, all_token_embeddings, all_edge_index, y):
    _os.makedirs("./graph_data", exist_ok=True)
    with open(f"./graph_data/{file_name}.pkl", "wb") as f:
        pickle.dump({"all_token_embeddings": all_token_embeddings,
                     "all_edge_index": all_edge_index,
                     "y": y}, f)


# ====================== 论文原始核心：GCN 模型（保持不变） ======================
class GCN2(nn.Module):
    def __init__(self,  input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)  
        return torch.sigmoid(x)

# 可选备用（与论文一致地提供，但本封装默认用 GCN2）
class GCN4(nn.Module):
    def __init__(self,  input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim):
        super(GCN4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, output_dim)
        self.fc = nn.Linear(output_dim, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)  
        return torch.sigmoid(x)


# ====================== 训练封装（不改核心，只做数据与落盘封装） ======================
@dataclass
class _TrainCfg:
    data: str
    eval_data: Optional[str]
    out_dir: str
    roberta_path: str
    max_length: int
    lr: float
    epochs: int
    seed: int
    device: Optional[str]
    # 额外封装项（不影响核心）
    sample_k: Optional[int] = None
    loss_plot_filename: str = "train_loss.png"

def _train_impl(cfg: _TrainCfg) -> Dict[str, Any]:
    _seed_all(cfg.seed)
    # 初始化与论文一致的 backbone（全局）
    _ensure_backbone(cfg.roberta_path)

    # 读取原始 jsonl，并（可选）采样
    lines_all = _load_jsonl_lines(cfg.data)
    if cfg.sample_k is not None and cfg.sample_k > 0 and cfg.sample_k < len(lines_all):
        rng = np.random.RandomState(cfg.seed)
        idx = np.arange(len(lines_all)); rng.shuffle(idx)
        lines_all = [lines_all[i] for i in idx[:cfg.sample_k]]

    # 若有独立验证集，直接使用；否则按 8:1 切分（与之前需求一致，但不改核心训练）
    if cfg.eval_data:
        tr_lines = lines_all
        va_lines = _load_jsonl_lines(cfg.eval_data)
    else:
        rng = np.random.RandomState(cfg.seed)
        idx = np.arange(len(lines_all)); rng.shuffle(idx)
        n_tr = int(round(len(idx) * (8.0 / 9.0)))
        tr_lines = [lines_all[i] for i in idx[:n_tr]]
        va_lines = [lines_all[i] for i in idx[n_tr:]]

    # —— 使用**论文原始** build_graph ——（核心不改）
    Xtr, Etr, Ytr = build_graph(tr_lines)
    Xva, Eva, Yva = build_graph(va_lines)

    # 模型/优化（与论文保持一致）
    input_dim = 768
    hidden_dim2 = 256
    output_dim = 64
    gcnmodel = GCN2(input_dim, hidden_dim2, output_dim).to(device)
    optimizer = torch.optim.Adam(gcnmodel.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()

    # 输出目录
    ts = _timestamp()
    run_dir = os.path.join(cfg.out_dir, f"prdetect_{ts}")
    best_dir = os.path.join(run_dir, "best"); os.makedirs(best_dir, exist_ok=True)
    last_dir = os.path.join(run_dir, "last"); os.makedirs(last_dir, exist_ok=True)

    steps, losses = [], []
    val_max_acc = -1.0
    gs = 0
    t0 = time.perf_counter()

    # 训练循环（与论文保持逐样本、阈值 0.5）
    for ep in range(1, cfg.epochs + 1):
        gcnmodel.train()
        epoch_loss = 0.0
        correct_predictions = 0
        it = tqdm(range(len(Ytr)), desc=f"Train [ep {ep}/{cfg.epochs}]", dynamic_ncols=True, leave=False)
        for i in it:
            data_i = Data(x=Xtr[i], edge_index=Etr[i], y=Ytr[i]).to(device)
            optimizer.zero_grad()
            outputs = gcnmodel(data_i)
            loss = criterion(outputs, data_i.y.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == data_i.y.view(-1, 1)).sum().item()
            gs += 1; steps.append(gs); losses.append(float(loss.item()))
            it.set_postfix(loss=f"{(epoch_loss/(i+1)):.4f}")

        train_loss = epoch_loss / max(1, len(Ytr))
        train_acc  = correct_predictions / max(1, len(Ytr))

        # 验证
        gcnmodel.eval()
        v_loss = 0.0
        v_correct = 0
        with torch.no_grad():
            vit = tqdm(range(len(Yva)), desc="Valid", dynamic_ncols=True, leave=False)
            for i in vit:
                data_i = Data(x=Xva[i], edge_index=Eva[i], y=Yva[i]).to(device)
                outputs = gcnmodel(data_i)
                loss = criterion(outputs, data_i.y.float().view(-1, 1))
                v_loss += loss.item()
                predictions = (outputs >= 0.5).long()
                v_correct += (predictions == data_i.y.view(-1, 1)).sum().item()
        val_loss = v_loss / max(1, len(Yva))
        val_acc  = v_correct / max(1, len(Yva))
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # 落盘（与之前格式一致）
        torch.save(gcnmodel.state_dict(), os.path.join(last_dir, "prdetect_gcn.pt"))
        if val_acc >= val_max_acc:
            val_max_acc = val_acc
            torch.save(gcnmodel.state_dict(), os.path.join(best_dir, "prdetect_gcn.pt"))

    wall = time.perf_counter() - t0

    # 曲线/摘要
    loss_png = _save_loss_curve(steps, losses, run_dir, "train_loss.png")
    summary = {
        "detector": "PRDetect",
        "config": {
            "data": cfg.data,
            "eval_data": cfg.eval_data,
            "out_dir": cfg.out_dir,
            "roberta_path": cfg.roberta_path,
            "max_length": cfg.max_length,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "device": str(device),
            "sample_k": cfg.sample_k,
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "best_val_acc": val_max_acc if val_max_acc >= 0 else None,
        "timing": {"train_wall_sec": wall},
        "artifacts": {"loss_plot": loss_png},
    }
    os.makedirs(run_dir, exist_ok=True)
    _save_json(summary, os.path.join(run_dir, "train_summary.json"))

    return {
        "run_dir": run_dir,
        "model_dir": best_dir,
        "best_val_acc": val_max_acc if val_max_acc >= 0 else None,
        "artifacts": {"best_dir": best_dir, "last_dir": last_dir, "loss_plot": loss_png},
    }


@register_train("prdetect")
def train_prdetect(*,
                   model: str = "prdetect",
                   dataset: Optional[str] = None,
                   eval_dataset: Optional[str] = None,
                   sample_k: Optional[int] = None,
                   output_dir: str = "./runs_prdetect",
                   max_length: int = 512,
                   lr: float = 1e-4,
                   epochs: int = 10,
                   seed: int = 2024,
                   device: Optional[str] = None,
                   roberta_path: str = "roberta-base",
                   **kwargs) -> Dict[str, Any]:
    if not dataset:
        raise ValueError("[prdetect] --data / --dataset 不能为空（需要 jsonl，每行含 {'text','label'}）")
    cfg = _TrainCfg(
        data=dataset,
        eval_data=eval_dataset,
        out_dir=output_dir,
        roberta_path=roberta_path,
        max_length=max_length,
        lr=lr,
        epochs=epochs,
        seed=seed,
        device=device,
        sample_k=sample_k,
    )
    return _train_impl(cfg)


# ====================== 运行时检测器（评测） ======================
def _roberta_path_from_summary(model_dir: str) -> Optional[str]:
    run_dir = os.path.abspath(os.path.join(model_dir, os.pardir))
    cand = os.path.join(run_dir, "train_summary.json")
    try:
        with open(cand, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return (meta.get("config") or {}).get("roberta_path", None)
    except Exception:
        return None

@register("prdetect")
class PRDetectDetector(DetectorBase):
    DETECTOR_NAME = "prdetect"
    detector_type = "Graph-based (RoBERTa + Dependency-GCN)"
    CITATION_AUTHORS = "Your Team (2024–2025)"
    CITATION_TITLE = "PRDetect: Graph-based Detection via Dependency-GCN"
    CITATION_LINK = "https://github.com/your_org/PRDetect"

    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,   # 未用
                 roberta_path: Optional[str] = None,
                 max_length: int = 512,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_path = os.path.abspath(model_path or ".")
        self.max_length = int(max_length)
        self.device = torch.device(_device_str(device))
        self.roberta_path = roberta_path or _roberta_path_from_summary(self.model_path) or "roberta-base"

        self._gcn = None

    def load(self):
        if self.is_loaded:
            return
        # 与论文一致的 backbone
        _ensure_backbone(self.roberta_path)

        # 模型结构与训练一致
        input_dim = 768; hidden_dim2 = 256; output_dim = 64
        self._gcn = GCN2(input_dim, hidden_dim2, output_dim).to(device)

        # 允许三种形态：当前目录 /best /last
        cand = [
            os.path.join(self.model_path, "prdetect_gcn.pt"),
            os.path.join(self.model_path, "best", "prdetect_gcn.pt"),
            os.path.join(self.model_path, "last", "prdetect_gcn.pt"),
        ]
        ckpt = next((p for p in cand if os.path.isfile(p)), None)
        if not ckpt:
            raise FileNotFoundError(f"[prdetect] 未找到权重文件，尝试过：\n" + "\n".join(cand))
        self._gcn.load_state_dict(torch.load(ckpt, map_location=device))
        self._gcn.eval()
        self.is_loaded = True

    def unload(self):
        try:
            del self._gcn
        except Exception:
            pass
        self._gcn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False

    @torch.no_grad()
    def _score_one_p_human(self, text: str) -> float:
        """按论文标签语义：返回 p(human|x)。"""
        # 为复用论文 build_graph，这里对单样本构造一个 json 文本行
        js = json.dumps({"text": text, "label": "human"})  # label 字段在 build_graph 仅用于占位，不影响推断
        X, E, _ = build_graph([js])  # y 未用
        out = self._gcn(Data(x=X[0], edge_index=E[0])).item()
        return float(out)

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """返回 p(human|x)，与训练一致（1=human）。"""
        if not self.is_loaded:
            self.load()
        probs: List[float] = []
        for t in texts:
            probs.append(self._score_one_p_human(t))
        return np.asarray(probs, dtype=np.float32)

    def evaluate(self,
                 examples: Sequence[Dict[str, Any]],
                 batch_size: int = 1,
                 threshold: float = 0.5,
                 show_progress: bool = True) -> EvalResult:
        if not self.is_loaded:
            self.load()
        labels: List[int] = []
        probs: List[float] = []
        it = range(len(examples))
        if show_progress:
            it = tqdm(it, desc="[PRDetect] Eval", dynamic_ncols=True)
        with torch.no_grad():
            for i in it:
                ex = examples[i]
                text = ex.get("text", "")
                y = _label_human_is_one(ex.get("label", 0))
                p_h = self._score_one_p_human(text)
                probs.append(float(p_h)); labels.append(int(y))
        preds = [1 if p >= float(threshold) else 0 for p in probs]
        acc = sum(int(p == y) for p, y in zip(preds, labels)) / max(1, len(labels))
        meta = {
            "detector": self.DETECTOR_NAME,
            "detector_type": self.detector_type,
            "roberta_path": self.roberta_path,
            "model_path": self.model_path,
            "threshold": float(threshold),
            "prob_semantics": "p(human|x)",   # 明确概率语义
            "env": {"cuda_available": torch.cuda.is_available(),
                    "device": str(device),
                    "torch": torch.__version__},
        }
        metrics = {"acc": acc}
        return EvalResult(scores=list(probs), probs=list(probs), preds=list(preds),
                          labels=list(labels), meta=meta, metrics=metrics)


# ====================== 一把梭：训练 + 评测 ======================
def PRDetect(*,
             data: str,
             eval_dataset: Optional[str] = None,
             out_dir: str = "./runs_prdetect",
             roberta_path: str = "roberta-base",
             max_length: int = 512,
             lr: float = 1e-4,
             epochs: int = 10,
             seed: int = 2024,
             device: Optional[str] = None,
             batch_size: int = 1,          # 仅占位，按论文逐样本
             threshold: float = 0.5,
             sample_k: Optional[int] = None,
             save_curves: bool = True,
             name: Optional[str] = None) -> Dict[str, Any]:
    """
    封装：严格按论文核心训练+推断；保持输出/落盘结构与之前一致。
    返回：{"train": {...}, "eval": {"metrics":..., "meta":...}, "model_dir": "..."}
    """
    tr_out = train_prdetect(
        dataset=data,
        eval_dataset=eval_dataset,
        output_dir=out_dir,
        roberta_path=roberta_path,
        max_length=max_length,
        lr=lr,
        epochs=epochs,
        seed=seed,
        device=device,
        sample_k=sample_k,
    )
    model_dir = tr_out["model_dir"]

    # 评测：使用同一检测器（p(human|x)）
    res = evaluate_detector(
        detector="prdetect",
        dataset=(eval_dataset or data),
        batch_size=batch_size,
        threshold=threshold,
        sample_k=sample_k,
        out_dir=os.path.join(out_dir, "eval_results"),
        save_curves=save_curves,
        model_path=model_dir,
        roberta_path=roberta_path,
        max_length=max_length,
        show_progress=True,
    )
    eval_out = {"metrics": getattr(res, "metrics", {}), "meta": getattr(res, "meta", {})}
    return {"train": tr_out, "eval": eval_out, "model_dir": model_dir}
