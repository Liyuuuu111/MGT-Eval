# mgt_eval/detectors/finetuned/coco.py
from __future__ import annotations
import os, re, json, math, time, random, platform, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
# ——  HF “Token indices sequence length ...” ——
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Token indices sequence length is longer than the specified maximum sequence length for this model",
    category=UserWarning,
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # set_verbosity_warning()
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
)
# ==== Detective  ====
W_EPOCH = 8
W_MEM   = 8
W_NUM   = 7   # 30.000545  1.23e-07
W_STEP  = 8
SEP     = " "  # " | "

# ---- ：A100/SM8.x  TF32， FP32 matmul  ----
try:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ---------------- Env & Warnings ----------------
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# UserWarning（）
warnings.filterwarnings(
    "ignore",
    message=r"Token indices sequence length is longer than the specified maximum sequence length for this model",
    category=UserWarning,
)

# ---------------- Logger ----------------
try:
    from loguru import logger
    _USE_LOGURU = True
except Exception:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("CoCo")
    _USE_LOGURU = False

# ===== （）=====
try:
    from data_utils.load import load_dataset_unified
except Exception:
    raise ImportError("[CoCo] 请确保 mgt_eval.data_utils.load.load_dataset_unified 可用。")

try:
    from train.registry import register_train
except Exception:
    # （）
    def register_train(name):
        def deco(fn): return fn
        return deco

try:
    from train.train import (
        _reset_and_mark_cuda_peaks, _collect_cuda_peaks,
        _save_loss_plot, _build_data_info
    )
except Exception:
    # （ GPU /）
    def _reset_and_mark_cuda_peaks(): return None
    def _collect_cuda_peaks(_): return {}
    def _save_loss_plot(steps, losses, out_dir, filename="train_loss.png", smooth_window=0):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(os.path.join(out_dir, "train_loss.csv"), "w", encoding="utf-8") as f:
            for s, l in zip(steps, losses): f.write(f"{s},{l}\n")
        return path
    def _build_data_info(dataset_spec, train_ds, val_ds=None):
        return {
            "dataset": dataset_spec, "train_size": len(train_ds) if train_ds else 0,
            "val_size": len(val_ds) if val_ds else 0
        }

# ---------------- Config Patcher ----------------
import shutil
from typing import Optional
from transformers import PreTrainedTokenizerBase

def _copy_and_patch_config(
    out_dir: str,
    base_model_or_dir: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """
    鲁棒做法：
      1) 读取“原始”checkpoint的 config（base_model_or_dir）；
      2) 覆盖写入到 out_dir/config.json；
      3) 仅在缺失时增补少量下游需要的键（num_labels、id2label/label2id、pad_token_id 等）。
    """
    os.makedirs(out_dir, exist_ok=True)
    target_cfg_path = os.path.join(out_dir, "config.json")

    base_cfg = AutoConfig.from_pretrained(base_model_or_dir, trust_remote_code=True).to_dict()
    base_cfg.setdefault("num_labels", 2)

    if "id2label" not in base_cfg or not isinstance(base_cfg["id2label"], dict):
        base_cfg["id2label"] = {"0": "human", "1": "ai"}
    else:
        base_cfg["id2label"] = {str(k): v for k, v in base_cfg["id2label"].items()}
        base_cfg["id2label"].setdefault("0", "human")
        base_cfg["id2label"].setdefault("1", "ai")

    if "label2id" not in base_cfg or not isinstance(base_cfg["label2id"], dict):
        base_cfg["label2id"] = {"human": 0, "ai": 1}
    else:
        base_cfg["label2id"].setdefault("human", 0)
        base_cfg["label2id"].setdefault("ai", 1)

    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        base_cfg["pad_token_id"] = int(tokenizer.pad_token_id)

    if extra_overrides:
        for k, v in extra_overrides.items():
            base_cfg[k] = v

    tmp_path = target_cfg_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(base_cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target_cfg_path)

# =====  =====
DETECTOR_NAME     = "CoCo"
detector_type     = "Model-based"
CITATION_AUTHORS  = "Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, Chao Shen"
CITATION_TITLE    = "CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning"
CITATION_LINK     = "https://aclanthology.org/2023.emnlp-main.1005/"

# ===================  ===================
def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _resolve_base(spec: Optional[str], fallback: str = "roberta-base") -> str:
    if not spec: return fallback
    s = spec.strip()
    if os.path.isdir(s) and os.path.isfile(os.path.join(s, "config.json")):
        return s
    try:
        AutoConfig.from_pretrained(s)
        return s
    except Exception:
        return fallback

def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank()
        except Exception:
            return 0
    return 0

def _gpu_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0.0

# =================== （） ===================
import nltk
_REPO_NLTK_DATA = str(Path(__file__).resolve().parents[2] / "nltk_data")
if os.path.isdir(_REPO_NLTK_DATA):
    if _REPO_NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, _REPO_NLTK_DATA)
    cur = os.environ.get("NLTK_DATA", "")
    if _REPO_NLTK_DATA not in cur.split(os.pathsep):
        os.environ["NLTK_DATA"] = os.pathsep.join(
            [p for p in [_REPO_NLTK_DATA, cur] if p]
        )
try:
    nltk.data.find("tokenizers/punkt")
    logger.info("[CoCo] NLTK resource loaded: tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
        logger.info("[CoCo] NLTK resource downloaded: tokenizers/punkt")
    except Exception: raise Exception
try:
    nltk.data.find("corpora/stopwords")
    logger.info("[CoCo] NLTK resource loaded: corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
        logger.info("[CoCo] NLTK resource downloaded: corpora/stopwords")
    except Exception: raise Exception
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
    logger.info("[CoCo] NLTK resource loaded: taggers/averaged_perceptron_tagger")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger", quiet=True)
        logger.info("[CoCo] NLTK resource downloaded: taggers/averaged_perceptron_tagger")
    except Exception: raise Exception
try:
    nltk.data.find("chunkers/maxent_ne_chunker")
    logger.info("[CoCo] NLTK resource loaded: chunkers/maxent_ne_chunker")
except LookupError:
    try:
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("words", quiet=True)
        logger.info("[CoCo] NLTK resource downloaded: chunkers/maxent_ne_chunker")
        logger.info("[CoCo] NLTK resource downloaded: corpora/words")
    except Exception: raise Exception

from nltk.corpus import stopwords
_en_stops = set(stopwords.words("english")) if stopwords.__dict__.get("words", None) else set()

def _sent_split(text: str) -> List[str]:
    try:
        sens = nltk.sent_tokenize(text)
    except Exception:
        sens = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sens:
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        out.extend(parts)
    return out

def _extract_entities_allennlp(sents: List[str]) -> List[List[str]]:
    try:
        from allennlp.predictors.predictor import Predictor
        _cuda = 0 if torch.cuda.is_available() else -1
        predictor_ner = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz",
            cuda_device=_cuda
        )
        outputs = predictor_ner.predict_batch_json(inputs=[{"sentence": s} for s in sents])
        ents = []
        for o in outputs:
            words, tags = o["words"], o["tags"]
            e_list_final = []
            start_index = None
            for i, tag in enumerate(tags):
                if tag == "O": continue
                if tag.startswith("B"):
                    start_index = i
                elif tag.startswith("I"):
                    continue
                elif tag.startswith("L"):
                    if start_index is None: continue
                    e_list_cache = []
                    for j in range(start_index, i+1):
                        cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[j])
                        if cword: e_list_cache.append(cword)
                    if e_list_cache:
                        e_list_final.append(" ".join(e_list_cache))
                    start_index = None
                elif tag.startswith("U"):
                    cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])
                    if cword: e_list_final.append(cword)
            ents.append(e_list_final)
        return ents
    except Exception:
        ents = []
        for s in sents:
            try:
                toks = nltk.word_tokenize(s)
                pos = nltk.pos_tag(toks)
                tree = nltk.ne_chunk(pos, binary=False)
                cur = []
                for subtree in tree:
                    if hasattr(subtree, "label"):
                        phrase = " ".join(token for token, _ in subtree.leaves())
                        phrase = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", " ", phrase).strip()
                        if phrase: cur.append(phrase)
                ents.append(cur)
            except Exception:
                ents.append([])
        return ents

def _keep_node(text: str, words: List[str]) -> bool:
    if not words: return False
    if len(words) > 6: return False
    if all((w.lower() in _en_stops or not re.search(r"[A-Za-z0-9]", w)) for w in words):
        return False
    t = re.sub(r"[^A-Za-z0-9]+", "", text)
    return len(t) >= 2

def _keep_sentence_by_kws(kws: List[str]) -> bool:
    return len(kws) > 0

def _clean_token(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9,.\'!?]+", "", s)

def _first_index_list(token_list: List[str], cleaned_target: str) -> Tuple[int, int]:
    if not cleaned_target: return -1, 0
    tgt = cleaned_target
    n = len(token_list)
    for i in range(n):
        if not token_list[i]: continue
        acc = token_list[i]
        if acc == tgt: return i, 1
        j = i + 1
        while j < n and len(acc) < len(tgt):
            if token_list[j]:
                acc += token_list[j]
                if acc == tgt:
                    return i, (j - i + 1)
            j += 1
    return -1, 0

# ===================  ===================
def _build_graph_from_sent_kw(sents: List[str], sent_entities: List[List[str]]) -> Tuple[List[Dict], List[Tuple[int,int,str]], Dict[str,int], List[str], List[List[int]]]:
    nodes, edges, entity_occur, sen2node = [], [], {}, []
    last_sen_cnt = 0
    for sen_idx, kws in enumerate(sent_entities):
        kws = list({kw for kw in kws if kw and kw.strip() and kw.lower() not in _en_stops})
        if not _keep_sentence_by_kws(kws):
            sen2node.append([]); continue
        kws_cnt = 0
        sen_nodes = []
        for kw in kws:
            kw_norm = re.sub(r"[^a-zA-Z0-9,.\'\`!?]+", " ", kw)
            words = [w for w in nltk.word_tokenize(kw_norm) if (w not in _en_stops and w.capitalize() not in _en_stops)]
            if _keep_node(kw_norm, words):
                sen_nodes.append(len(nodes))
                nodes.append({"text": kw_norm, "words": words, "sentence_id": sen_idx})
                entity_occur[kw_norm] = entity_occur.get(kw_norm, 0) + 1
                kws_cnt += 1
        edges += [(last_sen_cnt + i, last_sen_cnt + i + 1, "inner") for i in list(range(max(0, kws_cnt - 1)))]
        last_sen_cnt += kws_cnt
        sen2node.append(sen_nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: continue
            if nodes[i]["text"].strip() == nodes[j]["text"].strip():
                edges.append((min(i, j), max(i, j), "inter"))
    if not nodes:
        return [], [], {}, sents, sen2node
    edges = list(set(edges))
    return nodes, edges, entity_occur, sents, sen2node

def _generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer, max_seq_length: int):
    L = int(max_seq_length)
    budget = max(0, L - 2)
    sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []
    start_ptr = 0

    for sen in sens:
        toks = tokenizer.tokenize(sen)
        if budget <= 0:
            cleaned = []
            sen_tokens.append(cleaned)
            sen_idx_pair.append((start_ptr, start_ptr))
            continue
        if len(toks) > budget:
            toks = toks[:budget]
        cleaned = [_clean_token(t) for t in toks]
        sen_tokens.append(cleaned)
        sen_idx_pair.append((start_ptr, start_ptr + len(toks)))
        all_tokens.extend(toks)
        start_ptr += len(toks)
        budget -= len(toks)

    for nidx, node in enumerate(ent_nodes):
        node_text = node["text"]
        sid = int(node.get("sentence_id", 0))
        if not (0 <= sid < len(sen_tokens)):
            ent_nodes[nidx]["spans"] = (-1, -1)
            drop_nodes.append(nidx)
            continue

        start_pos, node_len = _first_index_list(sen_tokens[sid], _clean_token(node_text))
        if start_pos != -1:
            final_start_pos = sen_idx_pair[sid][0] + start_pos
            max_pos = final_start_pos + node_len
            if 0 <= final_start_pos < L and 0 < max_pos <= L:
                ent_nodes[nidx]["spans"] = (final_start_pos, max_pos)
                ent_nodes[nidx]["spans_check"] = all_tokens[final_start_pos:max_pos]
            else:
                ent_nodes[nidx]["spans"] = (-1, -1)
                drop_nodes.append(nidx)
        else:
            ent_nodes[nidx]["spans"] = (-1, -1)
            drop_nodes.append(nidx)

    return ent_nodes, all_tokens, drop_nodes, sen_idx_pair

# =================== Dataset  Collate ===================
@dataclass
class CoCoConfig:
    base_model: str = "roberta-base"
    max_seq_length: int = 512
    max_nodes_num: int = 150
    gcn_layers: int = 2
    with_relation: int = 2  # 0: no relation, 2: inner+inter
    attention_maxscore: int = 16
    lambda_cl: float = 0.1      # （0 ）
    cl_temp: float = 0.07

class CoCoGraphDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, cfg: CoCoConfig):
        self.examples = examples
        self.tok = tokenizer
        self.cfg = cfg

    def __len__(self): return len(self.examples)

    def _build_item(self, text: str, label: int):
        sents = _sent_split(text)
        sent_entities = _extract_entities_allennlp(sents)
        nodes, edges, _, sens, sen2node = _build_graph_from_sent_kw(sents, sent_entities)

        enc = self.tok(text, truncation=True, max_length=self.cfg.max_seq_length,
                       padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)

        if not nodes:
            nodes = [{"text": "dummy", "words": ["dummy"], "sentence_id": 0, "spans": (1, 2)}]
            edges = []
            sens = sents or [text]
            sen2node = [[0]]

        nodes, all_tokens, drop_nodes, sen_idx_pair = _generate_rep_mask_based_on_graph(
            nodes, sens, self.tok, self.cfg.max_seq_length
        )
        maxN = self.cfg.max_nodes_num
        L = self.cfg.max_seq_length
        nodes_mask = np.zeros((maxN, L), dtype=np.float32)
        node_mask = np.zeros((maxN,), dtype=np.int64)
        valid_nodes = []
        for n in nodes:
            if n.get("spans", (-1, -1))[0] != -1:
                valid_nodes.append(n)
        nodes = valid_nodes[:maxN]
        for i, n in enumerate(nodes):
            a, b = n["spans"]
            a = max(0, min(a, L-1)); b = max(0, min(b, L))
            if b > a:
                nodes_mask[i, a:b] = 1.0
                node_mask[i] = 1

        R = self.cfg.with_relation
        if R > 0:
            if R >= 2:
                rel2idx = {"inner": 0, "inter": 1}
                adj = np.zeros((2, maxN, maxN), dtype=np.float32)
            else:
                rel2idx = {"inner": 0}
                adj = np.zeros((1, maxN, maxN), dtype=np.float32)
            for i in range(min(maxN, len(nodes))):
                for r in range(adj.shape[0]):
                    adj[r, i, i] = 1.0
            for (u, v, typ) in edges:
                if u >= len(valid_nodes) or v >= len(valid_nodes): continue
                if u >= maxN or v >= maxN: continue
                ridx = rel2idx.get(typ, 0)
                adj[ridx, u, v] = 1.0; adj[ridx, v, u] = 1.0
        else:
            adj = np.zeros((maxN, maxN), dtype=np.float32)
            for i in range(min(maxN, len(nodes))):
                adj[i, i] = 1.0

        item = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "label": torch.tensor(int(label), dtype=torch.long),
            "nodes_index_mask": torch.tensor(nodes_mask, dtype=torch.float32),
            "adj_metric": torch.tensor(adj, dtype=torch.float32),
            "node_mask": torch.tensor(node_mask, dtype=torch.int64),
        }
        return item

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return self._build_item(ex["text"], ex["label"])

def coco_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.stack([b[k] for b in batch], dim=0)
    return collated

# =================== ：Text Encoder + R-GCN +  ===================
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels: int, dropout=0.1):
        super().__init__()
        self.num_rels = num_rels
        self.weight = nn.Parameter(torch.randn(num_rels, in_dim, out_dim) * (1.0 / math.sqrt(in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        h:   [B, N, D]
        adj: [B, R, N, N]
        mask:[B, N]  (1=valid)
        """
        R = adj.size(1)
        out = 0.0
        for r in range(R):
            a = adj[:, r]  # [B, N, N]
            Wh = torch.einsum("bnd,df->bnf", h, self.weight[r])  # [B,N,F]
            neigh = torch.einsum("bnm,bmf->bnf", a, Wh)
            out = out + neigh
        out = out / max(1, R)
        out = out + self.bias
        out = self.act(out)
        out = self.dropout(out)
        out = out * mask.unsqueeze(-1)
        return out

class CoCoGraphModel(nn.Module):
    def __init__(self, base_model: str, cfg: CoCoConfig, num_labels: int = 2):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(base_model, add_pooling_layer=False)
        hid = self.encoder.config.hidden_size
        self.proj_tok = nn.Linear(hid, hid)
        self.gcn1 = RGCNLayer(hid, hid, max(1, cfg.with_relation), dropout=0.1)
        self.gcn2 = RGCNLayer(hid, hid, max(1, cfg.with_relation), dropout=0.1) if cfg.gcn_layers >= 2 else None
        self.pool_graph = nn.Linear(hid, hid)
        self.classifier = nn.Linear(hid * 2, num_labels)  # [CLS] + graph
        self.dropout = nn.Dropout(0.1)
        self.cl_text = nn.Linear(hid, hid)
        self.cl_graph = nn.Linear(hid, hid)

    def _nodes_from_mask(self, token_emb: torch.Tensor, nodes_mask: torch.Tensor, node_mask: torch.Tensor):
        denom = torch.clamp(nodes_mask.sum(dim=-1, keepdim=True), min=1.0)
        w = nodes_mask / denom
        nodes = torch.einsum("bnl,blh->bnh", w, token_emb)  # [B,N,H]
        nodes = nodes * node_mask.unsqueeze(-1)
        return nodes

    def _graph_readout(self, nodes: torch.Tensor, node_mask: torch.Tensor):
        denom = torch.clamp(node_mask.sum(dim=-1, keepdim=True), min=1.0)  # [B,1]
        g = (nodes.sum(dim=1) / denom)  # [B,H]
        g = self.pool_graph(g)
        return g

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        nodes_index_mask: torch.Tensor,
        adj_metric: torch.Tensor,
        node_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeds: bool = False,
    ):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        tok = enc.last_hidden_state  # [B,L,H]
        pooled = enc.pooler_output if hasattr(enc, "pooler_output") and enc.pooler_output is not None \
            else tok[:, 0]  # [CLS]

        tok = self.dropout(self.proj_tok(tok))  # [B,L,H]

        if adj_metric.dim() == 3:
            adj_metric = adj_metric.unsqueeze(1)

        nodes = self._nodes_from_mask(tok, nodes_index_mask, node_mask)  # [B,N,H]
        h = self.gcn1(nodes, adj_metric, node_mask)
        if self.gcn2 is not None:
            h = self.gcn2(h, adj_metric, node_mask)

        g = self._graph_readout(h, node_mask)  # [B,H]

        cat = torch.cat([pooled, g], dim=-1)  # [B,2H]
        logits = self.classifier(self.dropout(cat))

        out = {"logits": logits, "text_emb": pooled, "graph_emb": g}

        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits, labels)
            loss_cls = ce
            loss_scl = torch.tensor(0.0, device=logits.device)

            loss = ce
            # （ vs ）
            if self.cfg.lambda_cl > 0:
                t = F.normalize(self.cl_text(pooled), dim=-1)
                z = F.normalize(self.cl_graph(g), dim=-1)
                sim = (t @ z.t()) / max(1e-6, self.cfg.cl_temp)  # [B,B]
                pos = torch.diag(sim)
                logsumexp = torch.logsumexp(sim, dim=1)
                cl_loss = - (pos - logsumexp).mean()
                loss_scl = cl_loss
                loss = loss + self.cfg.lambda_cl * cl_loss

            out["loss"] = loss
            out["loss_cls"] = loss_cls
            out["loss_scl"] = loss_scl
        if return_embeds:
            out["token_emb"] = tok
            out["node_emb"] = h
        return out

# ===================  /  ===================
@dataclass
class TrainCfg:
    base_model: str = "roberta-base"
    output_dir: str = "runs_coco"
    max_length: int = 512
    max_nodes_num: int = 150
    with_relation: int = 2
    gcn_layers: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 6
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    fp16: bool = True
    lambda_cl: float = 0.1
    cl_temp: float = 0.07
    seed: int = 42
    device: Optional[str] = None
    dataset_spec: Optional[str] = None
    validation_sample_k: Optional[int] = None    # ：

def _stratified_split(examples: List[Dict[str, Any]], tr: float, va: float, te: float, seed: int = 42):
    pos = [e for e in examples if int(e["label"]) == 1]
    neg = [e for e in examples if int(e["label"]) == 0]
    def split(lst):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(lst)); rng.shuffle(idx)
        S = tr + va + te; n = len(idx)
        ntr = int(round(n * tr / S)) if S > 0 else n
        nva = int(round(n * va / S)) if S > 0 else 0
        ntr = min(ntr, n); nva = min(nva, n - ntr); nte = n - ntr - nva
        return idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    p_tr, p_va, p_te = split(pos)
    n_tr, n_va, n_te = split(neg)
    train = [pos[i] for i in p_tr] + [neg[i] for i in n_tr]
    val   = [pos[i] for i in p_va] + [neg[i] for i in n_va]
    test  = [pos[i] for i in p_te] + [neg[i] for i in n_te]
    rng = np.random.RandomState(seed); rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

@torch.no_grad()
def _evaluate(model: CoCoGraphModel, dl: DataLoader, device: torch.device):
    model.eval()
    tot, cor = 0, 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
            nodes_index_mask=b["nodes_index_mask"],
            adj_metric=b["adj_metric"],
            node_mask=b["node_mask"],
        )["logits"]
        pred = logits.argmax(dim=-1)
        cor += (pred == b["label"]).sum().item()
        tot += b["label"].size(0)
    return {"acc": cor / max(1, tot)}

def _linear_warmup_decay(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

# ------- （ + tqdm + logger） -------
W_EPOCH = 8
W_MEM   = 8
W_NUM   = 7
W_STEP  = 8
SEP     = " "

def _train_coco(
    *,
    train_examples: List[Dict[str, Any]],
    val_examples: Optional[List[Dict[str, Any]]],
    output_dir: str,
    cfg: TrainCfg
) -> Dict[str, Any]:
    torch.set_grad_enabled(True)
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = _resolve_base(cfg.base_model, "roberta-base")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)

    # pad token & padding side
    if tok.pad_token is None:
        if getattr(tok, "eos_token", None) is not None: tok.pad_token = tok.eos_token
        else: tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "right"
    # tokenizer  model_max_length
    try: tok.model_max_length = int(cfg.max_length)
    except Exception: pass
    try: tok.deprecation_warnings["sequence_length_is_longer_than_the_maximum_length"] = True
    except Exception: pass

    coco_cfg = CoCoConfig(
        base_model=base,
        max_seq_length=cfg.max_length,
        max_nodes_num=cfg.max_nodes_num,
        gcn_layers=cfg.gcn_layers,
        with_relation=cfg.with_relation,
        attention_maxscore=16,
        lambda_cl=cfg.lambda_cl,
        cl_temp=cfg.cl_temp
    )
    model = CoCoGraphModel(base, coco_cfg, num_labels=2).to(device)

    run_dir = f"{output_dir}_coco_{_timestamp()}"
    os.makedirs(run_dir, exist_ok=True)
    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    args_json_path = os.path.join(run_dir, "train_args.json")
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(cfg),
            "env": env_info,
            "data": _build_data_info(cfg.dataset_spec,
                                     train_ds:=CoCoGraphDataset(train_examples, tok, coco_cfg),
                                     val_ds:= (CoCoGraphDataset(val_examples, tok, coco_cfg) if val_examples else None))
        }, f, ensure_ascii=False, indent=2)

    train_ds = train_ds
    val_ds = val_ds

    train_dl = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device.type=="cuda"), collate_fn=coco_collate)
    val_dl = (DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                         num_workers=2, pin_memory=(device.type=="cuda"), collate_fn=coco_collate)
              if val_examples else None)

    # +  warmup/decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay":cfg.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.fp16 and device.type == "cuda"))

    mem_ctx = _reset_and_mark_cuda_peaks()
    global_step = 0
    best_val = -1.0
    best_dir = None
    last_dir = None
    step_indices, step_losses = [], []
    history = []
    total_wall_start = time.perf_counter()
    rank = _get_rank()
    logger.info(f"[mgt_eval] Using detector: {DETECTOR_NAME} (type={detector_type})")
    logger.info(f"[mgt_eval] Credits: {CITATION_AUTHORS} | Paper: {CITATION_TITLE} | Link: {CITATION_LINK}")
    logger.info("[mgt_eval] Disclaimer: This implementation approximates the original CoCo "
                "with a lightweight R-GCN and supervised contrastive term; results may differ from the paper.")
    logger.info(f"Train size={len(train_ds)} | Val size={len(val_ds) if val_ds else 0} | Device={device}")

    ema = None
    ema_beta = 0.9

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        num_batches_per_epoch = len(train_dl)

        # Detective ：rank0  tqdm  enumerate(dl)
        if True:  # ， rank0
            # （lambda_cl>0）， L/Lscl/Lc； L/Lc
            has_cl = cfg.lambda_cl > 0
            print("\n" +
                  f"{'Epoch':>{W_EPOCH}}{SEP}"
                  f"{'GPU_mem':>{W_MEM}}{SEP}"
                  f"{'L':>{W_NUM}}{SEP}" +
                  (f"{'Lscl':>{W_NUM}}{SEP}" if has_cl else "") +
                  f"{'Lc':>{W_NUM}}{SEP}"
                  f"{'avg':>{W_NUM}}{SEP}"
                  f"{'lr':>{W_NUM}}{SEP}"
                  f"{'step':>{W_STEP}}")
            pbar = tqdm(enumerate(train_dl), total=num_batches_per_epoch, dynamic_ncols=True, leave=False)
        else:
            pbar = enumerate(train_dl)

        accum = 0
        avg_loss = 0.0

        for i, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(cfg.fp16 and device.type == "cuda")):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    nodes_index_mask=batch["nodes_index_mask"],
                    adj_metric=batch["adj_metric"],
                    node_mask=batch["node_mask"],
                    labels=batch["label"],
                )
                loss = out["loss"] / max(1, cfg.grad_accum_steps)
                loss_cls = out.get("loss_cls", torch.tensor(0.0, device=device))
                loss_scl = out.get("loss_scl", torch.tensor(0.0, device=device))

            scaler.scale(loss).backward()
            running += float(loss.item()); n_batches += 1
            accum += 1

            if accum % cfg.grad_accum_steps == 0:
                # warmup + （）
                lr_scale = _linear_warmup_decay(global_step, total_steps, warmup_steps)
                for pg in optim.param_groups:
                    pg["lr"] = cfg.lr * lr_scale

                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                step_indices.append(global_step)
                step_losses.append(float(loss.item()))
                accum = 0

                # Detective  set_description
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                avg_loss = (avg_loss * i + float(loss.item())) / (i + 1)
                lr_now = optim.param_groups[0]["lr"]
                desc = (
                    f"{f'{ep}/{cfg.epochs}':>{W_EPOCH}}{SEP}"
                    f"{mem:>{W_MEM}}{SEP}"
                    f"{float(loss.item()):>{W_NUM}.4f}{SEP}" +
                    (f"{float(loss_scl.item()):>{W_NUM}.4f}{SEP}" if has_cl else "") +
                    f"{float(loss_cls.item()):>{W_NUM}.4f}{SEP}"
                    f"{float(avg_loss):>{W_NUM}.4f}{SEP}"
                    f"{float(lr_now):>{W_NUM}.2e}{SEP}"
                    f"{int((ep - 1) * num_batches_per_epoch + i):>{W_STEP}d}"
                )
                if hasattr(pbar, "set_description"):
                    pbar.set_description(desc)
        # ===== （Detective ） =====
        val_acc = None
        if val_dl is not None:
            model.eval()
            right_num, tot_num = 0, 0
            avg_vloss = 0.0

            # rank0
            if rank == 0:
                print("\n" +
                    f"{'Epoch':>{W_EPOCH}}{SEP}"
                    f"{'GPU_mem':>{W_MEM}}{SEP}"
                    f"{'Cur_acc':>{W_NUM}}{SEP}"
                    f"{'avg_acc':>{W_NUM}}{SEP}"
                    f"{'loss':>{W_NUM}}")
                pbar_val = tqdm(enumerate(val_dl), total=len(val_dl), dynamic_ncols=True, leave=False)
            else:
                pbar_val = enumerate(val_dl)

            with torch.no_grad():
                for j, batch in pbar_val:
                    b = {k: v.to(device) for k, v in batch.items()}
                    out = model(
                        input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"],
                        nodes_index_mask=b["nodes_index_mask"],
                        adj_metric=b["adj_metric"],
                        node_mask=b["node_mask"],
                        labels=b["label"],  # forward  loss（ + ）
                    )
                    logits = out["logits"]
                    vloss  = out.get("loss", torch.tensor(0.0, device=device))

                    # batch
                    preds = logits.argmax(dim=-1)
                    cur_right = (preds == b["label"]).sum().item()
                    cur_tot   = b["label"].size(0)
                    right_num += cur_right
                    tot_num   += cur_tot
                    avg_vloss  = (avg_vloss * j + float(vloss.item())) / (j + 1)
                    if rank == 0 and hasattr(pbar_val, "set_description"):
                        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                        cur_acc = float(cur_right) / max(1, cur_tot)
                        avg_acc = float(right_num) / max(1, tot_num)
                        desc = (
                            f"{f'{ep}/{cfg.epochs}':>{W_EPOCH}}{SEP}"
                            f"{mem:>{W_MEM}}{SEP}"
                            f"{cur_acc:>{W_NUM}.4f}{SEP}"
                            f"{avg_acc:>{W_NUM}.4f}{SEP}"
                            f"{avg_vloss:>{W_NUM}.4f}"
                        )
                        pbar_val.set_description(desc)

            if hasattr(pbar_val, "close"):
                pbar_val.close()

            val_acc = float(right_num) / max(1, tot_num)

        # —— /（ best/last ，） ——
        run_last = os.path.join(run_dir, "last")
        os.makedirs(run_last, exist_ok=True)
        model_to_save = model
        torch.save(model_to_save.state_dict(), os.path.join(run_last, "pytorch_model.bin"))
        tok.save_pretrained(run_last)
        _copy_and_patch_config(out_dir=run_last, base_model_or_dir=base, tokenizer=tok)
        with open(os.path.join(run_last, "coco_config.json"), "w") as f:
            json.dump(vars(coco_cfg), f)
        last_dir = run_last  # <<<

        cur_metric = val_acc if (val_acc is not None) else -(running / max(1, n_batches))
        if cur_metric is not None and (best_val < 0 or cur_metric >= best_val):
            best_val = cur_metric
            run_best = os.path.join(run_dir, "best")
            os.makedirs(run_best, exist_ok=True)
            torch.save(model_to_save.state_dict(), os.path.join(run_best, "pytorch_model.bin"))
            tok.save_pretrained(run_best)
            _copy_and_patch_config(out_dir=run_best, base_model_or_dir=base, tokenizer=tok)
            with open(os.path.join(run_best, "coco_config.json"), "w") as f:
                json.dump(vars(coco_cfg), f)
            best_dir = run_best 
        logger.info(f"[CoCo][Epoch {ep}] train_loss={running/max(1,n_batches):.4f} "
              f"{'' if val_acc is None else f'val_acc={val_acc:.4f}  '}")

        if isinstance(pbar, tqdm):
            pbar.close()

        avg_train = running / max(1, n_batches)

        history.append({
            "epoch": ep, "avg_train_loss": avg_train, "val_acc": val_acc,
            "global_step": global_step
        })

    mem_stats = _collect_cuda_peaks(mem_ctx)
    loss_plot_path = _save_loss_plot(step_indices, step_losses, run_dir, "train_loss.png", 0)
    total_wall = time.perf_counter() - total_wall_start
    summary = {
        "best_dir": best_dir,
        "last_dir": last_dir or run_dir,
        "best_val_acc": (None if best_val < 0 else best_val),
        "history": history,
        "memory": mem_stats,
        "timing": {"total_wall_time_sec": total_wall},
        "artifacts": {
            "args_json": os.path.join(run_dir, "train_args.json"),
            "summary_json": os.path.join(run_dir, "train_summary.json"),
            "loss_plot": loss_plot_path,
        }
    }
    with open(summary["artifacts"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"[CoCo] done. best_dir={summary['best_dir']} | last_dir={summary['last_dir']} | "
                f"best_val_acc={summary['best_val_acc']} | time={total_wall:.1f}s")

    return {
        "best_dir": best_dir,
        "last_dir": last_dir or run_dir,
        "best_val_acc": (None if best_val < 0 else best_val),
        "artifacts": summary["artifacts"],
        "timing": summary["timing"],
    }

# =================== ：register_train ===================
@register_train("coco")
def train_coco(**kwargs) -> Dict[str, Any]:
    """
    关键入参（与 GREATER 风格一致）：
      - dataset_training:  训练/验证数据集标识（传给 load_dataset_unified）
    可选入参：
      - output_dir:        顶层输出目录（自动加时间戳）
      - base_model:        预训练基座（默认 roberta-base）
      - max_length, max_nodes_num, with_relation, gcn_layers
      - train_batch_size, eval_batch_size, lr, weight_decay, epochs, warmup_ratio, grad_accum_steps
      - fp16, lambda_cl, cl_temp, seed
      - training_sample_k: 训练集抽样条数（默认 None=全量）；验证:训练=1:8
    """
    dataset_training = kwargs.get("dataset_training", None)
    assert dataset_training, "coco 需要 dataset_training （传给 load_dataset_unified）"

    output_dir_raw   = kwargs.get("output_dir", "runs_coco")
    output_dir       = f"{output_dir_raw}_{_timestamp()}"

    base_model       = kwargs.get("base_model", "roberta-base")
    max_length       = int(kwargs.get("max_length", 512))
    max_nodes_num    = int(kwargs.get("max_nodes_num", 150))
    with_relation    = int(kwargs.get("with_relation", 2))
    gcn_layers       = int(kwargs.get("gcn_layers", 2))
    validation_sample_k = kwargs.get("validation_sample_k", None)
    train_batch_size = int(kwargs.get("train_batch_size", 16))
    eval_batch_size  = int(kwargs.get("eval_batch_size", 64))
    lr               = float(kwargs.get("lr", 2e-5))
    weight_decay     = float(kwargs.get("weight_decay", 0.01))
    epochs           = int(kwargs.get("epochs", 6))
    warmup_ratio     = float(kwargs.get("warmup_ratio", 0.06))
    grad_accum_steps = int(kwargs.get("grad_accum_steps", 1))
    fp16             = bool(kwargs.get("fp16", True))
    lambda_cl        = float(kwargs.get("lambda_cl", 0.1))
    cl_temp          = float(kwargs.get("cl_temp", 0.07))
    seed             = int(kwargs.get("seed", 42))
    training_sample_k= kwargs.get("training_sample_k", None)

    _seed_everything(seed)

    # （）
    exs, _ = load_dataset_unified(
        dataset=dataset_training,
        sample_k=training_sample_k,
        sample_seed=seed,
        group_cols=None
    )
    # ：
    dataset_val = kwargs.get("dataset_val", kwargs.get("dataset_validation", None))
    if dataset_val:
        # val
        val_exs, _ = load_dataset_unified(
            dataset=str(dataset_val),
            sample_k=validation_sample_k,   # None=，>0
            sample_seed=seed,
            group_cols=None
        )
        t_tr, t_va = exs, val_exs
    else:
        # ： 8:1
        t_tr, t_va, _ = _stratified_split(exs, 8.0, 1.0, 0.0, seed=seed)

    cfg = TrainCfg(
        base_model=base_model,
        output_dir=output_dir,
        max_length=max_length,
        max_nodes_num=max_nodes_num,
        with_relation=with_relation,
        gcn_layers=gcn_layers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
        warmup_ratio=warmup_ratio, grad_accum_steps=grad_accum_steps,
        fp16=fp16, lambda_cl=lambda_cl, cl_temp=cl_temp, seed=seed,
        dataset_spec=dataset_training,
        validation_sample_k=validation_sample_k,
    )

    result = _train_coco(
        train_examples=t_tr,
        val_examples=t_va,
        output_dir=output_dir,
        cfg=cfg
    )
    # CLI ： best_dir/last_dir（cmd_train ）
    best_dir = result.get("best_dir")
    last_dir = result.get("last_dir")
    # run_dir  best/last
    run_dir = None
    if best_dir and os.path.isdir(best_dir):
        run_dir = os.path.dirname(best_dir.rstrip("/"))
    elif last_dir and os.path.isdir(last_dir):
        run_dir = os.path.dirname(last_dir.rstrip("/"))

    return {
        "best_dir": best_dir,
        "last_dir": last_dir,
        "run_dir": run_dir,
        "best_val_acc": result.get("best_val_acc"),
        "artifacts": result.get("artifacts", {}),
        "train": {
            "model_dir": best_dir or last_dir,
            "best_val_acc": result.get("best_val_acc"),
            "artifacts": result.get("artifacts", {}),
            "output_root": output_dir,
            "run_dir": run_dir,
        }
    }

def COCO(**kwargs) -> Dict[str, Any]:
    return train_coco(**kwargs)
