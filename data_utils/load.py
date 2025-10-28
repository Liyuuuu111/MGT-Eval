from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os, json, glob, re
import numpy as np

HF_HC3 = "Hello-SimpleAI/HC3"

# ---------------------------
# 基础 I/O
# ---------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def _read_json(path: str) -> List[Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "examples", "items", "records"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        return [obj]
    return []

def _list_json_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
        files += glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
        return files
    if os.path.isfile(path) and (path.endswith(".jsonl") or path.endswith(".json")):
        return [path]
    return []

def _maybe_sample_exact(items: List[Any], k: Optional[int], seed: int = 114514) -> List[Any]:
    """
    从 items 中“样本级”精确抽取 k 条（k < len(items) 时），不放回；否则返回原列表。
    """
    if not items or k is None or k <= 0:
        return items
    n = len(items)
    if k >= n:
        return items
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [items[int(i)] for i in idx.tolist()]

# ---------------------------
# 通用规范化：text/label + 保留其余字段（用于分组）
# ---------------------------
def _norm_text_label(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rec, dict):
        return None
    if "text" not in rec or "label" not in rec:
        return None
    try:
        return {
            **rec,
            "text": str(rec["text"]),
            "label": int(rec["label"]),
        }
    except Exception:
        return None

# ---------------------------
# original / sample 顶层 JSON 展开
# ---------------------------
def _flatten_original_sample_record(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    展开形如:
      {"original": [...], "sample": [...], "sampled": [...], <其他上下文字段...>}
    为标准样本 [{text, label, split, ...}]：
      - original → label = 0, split = "original"
      - sample   → label = 1, split = "sample"
      - sampled  → label = 1, split = "sampled"
    任一字段不存在时可省略。
    """
    if not isinstance(rec, dict) or (("original" not in rec) and ("sample" not in rec) and ("sampled" not in rec)):
        return None

    def _to_text_list(lst: Any) -> List[str]:
        out: List[str] = []
        if isinstance(lst, (list, tuple)):
            for x in lst:
                if isinstance(x, dict) and "text" in x:
                    t = str(x.get("text", "")).strip()
                else:
                    t = str(x).strip()
                if t:
                    out.append(t)
        return out

    ctx = {k: v for k, v in rec.items() if k not in ("original", "sample", "sampled")}

    ori = _to_text_list(rec.get("original", []))
    sam_sample  = _to_text_list(rec.get("sample", []))
    sam_sampled = _to_text_list(rec.get("sampled", []))

    out: List[Dict[str, Any]] = []
    out += [{**ctx, "text": t, "label": 0, "split": "original"} for t in ori]

    # 逐源展开，split 保留其来源，均视作机器（label=1）
    out += [{**ctx, "text": t, "label": 1, "split": "sample"}  for t in sam_sample]
    out += [{**ctx, "text": t, "label": 1, "split": "sampled"} for t in sam_sampled]

    return out

# ---------------------------
# HC3 专用展开（成对最小平衡）
# ---------------------------
_HC3_PATTERN = re.compile(r"hc3", flags=re.IGNORECASE)

def should_route_to_hc3(dataset_spec: str) -> bool:
    return _HC3_PATTERN.search(dataset_spec) is not None

def _flatten_hc3_record(
    rec: Dict[str, Any],
    pair_mode: str = "balance_min",
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, Any]]:
    """
    将单条 HC3 记录展开为若干 {text,label}：
      - balance_min：取 k = min(#human, #machine)，输出 2k 条（各 k 条）
    """
    out: List[Dict[str, Any]] = []
    human = [s.strip() for s in rec.get("human_answers", []) if isinstance(s, str) and s.strip()]
    machine = [s.strip() for s in rec.get("chatgpt_answers", []) if isinstance(s, str) and s.strip()]
    source = rec.get("source", "unknown")

    if pair_mode == "all":
        out += [{"text": t, "label": 0, "source": source} for t in human]
        out += [{"text": t, "label": 1, "source": source} for t in machine]
        return out
    if not human or not machine:
        return out

    rng = rng or np.random.RandomState(114514)
    if pair_mode == "one_each":
        out.append({"text": human[rng.randint(len(human))], "label": 0, "source": source})
        out.append({"text": machine[rng.randint(len(machine))], "label": 1, "source": source})
        return out

    # balance_min
    k = min(len(human), len(machine))
    h_idx = rng.choice(len(human), size=k, replace=False)
    m_idx = rng.choice(len(machine), size=k, replace=False)
    out += [{"text": human[int(i)], "label": 0, "source": source} for i in h_idx]
    out += [{"text": machine[int(j)], "label": 1, "source": source} for j in m_idx]
    return out

def _load_hc3_all_from_hf() -> List[Dict[str, Any]]:
    """
    从 HF 读取 HC3 全量（不在本函数内抽样），再逐条记录展开为样本。
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("需要安装 datasets 才能从 HF 加载 HC3：pip install datasets") from e
    raw = load_dataset(HF_HC3, "all")
    # 聚合记录
    records: List[Dict[str, Any]] = []
    if hasattr(raw, "keys"):
        for sp in raw.keys():
            records.extend(list(raw[sp]))
    else:
        records.extend(list(raw))
    # 展开为样本
    rng = np.random.RandomState(114514)
    exs: List[Dict[str, Any]] = []
    for rec in records:
        exs.extend(_flatten_hc3_record(rec, pair_mode="balance_min", rng=rng))
    return exs

def _load_hc3_all_from_local(path: str) -> List[Dict[str, Any]]:
    """
    从本地 JSON/JSONL 读取 HC3 全量（不在本函数内抽样），再逐条记录展开为样本。
    同时兼容已是标准 text/label 的数据。
    """
    files = _list_json_files(path)
    if not files:
        raise FileNotFoundError(f"未在路径中找到 HC3 文件：{path}")

    # 读为记录列表
    records: List[Dict[str, Any]] = []
    for p in files:
        try:
            recs = _read_jsonl(p) if p.endswith(".jsonl") else _read_json(p)
        except Exception:
            continue
        records.extend(recs)
    if not records:
        raise ValueError(f"在 {path} 未解析到 HC3 记录")

    # 展开为样本；若已经是标准化数据则直接保留
    rng = np.random.RandomState(114514)
    exs: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict) and ("human_answers" in rec or "chatgpt_answers" in rec):
            exs.extend(_flatten_hc3_record(rec, pair_mode="balance_min", rng=rng))
        else:
            nr = _norm_text_label(rec)
            if nr is not None:
                exs.append(nr)
    return exs

# ---------------------------
# 抽样辅助：HC3 场景下“按类别均衡”抽样
# ---------------------------
def _balanced_sample_two_class(examples: List[Dict[str, Any]],
                               k_per_class: int,
                               seed: int = 114514) -> List[Dict[str, Any]]:
    """
    在二分类（0/1）样本上进行“每类恰好 k_per_class 条”的均衡抽样（不放回）。
    如果某类不足，则取 min(可用数) 达到对称上限，返回 2 * k_final 条。
    """
    if not examples or k_per_class is None or k_per_class <= 0:
        return examples

    # 分桶
    cls0 = [ex for ex in examples if int(ex.get("label", 0)) == 0]
    cls1 = [ex for ex in examples if int(ex.get("label", 0)) == 1]

    n0, n1 = len(cls0), len(cls1)
    if n0 == 0 or n1 == 0:
        # 无法均衡，直接返回原样本
        return _maybe_sample_exact(examples, 2 * k_per_class, seed=114514)

    k = min(k_per_class, n0, n1)
    rng = np.random.RandomState(seed)
    idx0 = rng.choice(n0, size=k, replace=False)
    idx1 = rng.choice(n1, size=k, replace=False)
    sub0 = [cls0[int(i)] for i in idx0.tolist()]
    sub1 = [cls1[int(i)] for i in idx1.tolist()]
    merged = sub0 + sub1

    # 打乱合并的样本，保持随机性
    if merged:
        perm = rng.permutation(len(merged))
        merged = [merged[int(i)] for i in perm.tolist()]
    return merged

# ---------------------------
# 通用加载入口
# ---------------------------
_DEFAULT_GROUP_CANDIDATES = ["lang", "source", "model", "sub_source", "split"]

def _auto_group_cols(examples: List[Dict[str, Any]],
                     candidates: Sequence[str] = _DEFAULT_GROUP_CANDIDATES) -> List[str]:
    """从候选集合中，挑出在数据中“存在且非空”的列名。"""
    chosen: List[str] = []
    if not examples:
        return chosen
    for col in candidates:
        exists = False
        for ex in examples[:512]:  # 采样一部分做探测
            v = ex.get(col, None)
            if v is not None and str(v).strip() != "":
                exists = True
                break
        if exists:
            chosen.append(col)
    return chosen

def load_dataset_unified(
    dataset: Union[str, Iterable[Dict[str, Any]]],
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    group_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    统一数据集加载：
      - 若 dataset 为 str：
          * 包含 'hc3'：无论本地/HF，均先“全量读取 + 展开为样本”，
            然后若设置 sample_k，则进行“每类各 sample_k 条”的均衡抽样（总计 2*sample_k）
          * 其他：当作标准 JSON/JSONL（目录或文件），样本级抽样，目标精确 sample_k 条
      - 若 dataset 为 Iterable[dict]：标准化 {"text","label"}，样本级抽样，目标精确 sample_k 条
    返回：
      * examples: List[dict]（顺序即评测顺序）
      * group_cols: 实际用于分组统计的列名（若未显式指定则自动探测）
    """
    # 1) 读取 & 标准化 examples
    examples: List[Dict[str, Any]] = []
    routed_hc3 = isinstance(dataset, str) and should_route_to_hc3(dataset)

    if isinstance(dataset, str):
        if routed_hc3:
            # HC3：全量展开为样本
            if os.path.exists(dataset):
                examples = _load_hc3_all_from_local(dataset)
            else:
                examples = _load_hc3_all_from_hf()
        else:
            # 标准 JSON/JSONL
            files = _list_json_files(dataset)
            if not files:
                raise FileNotFoundError(f"未找到 JSON/JSONL：{dataset}")
            for p in files:
                recs = _read_jsonl(p) if p.endswith(".jsonl") else _read_json(p)
                for r in recs:
                    # 先尝试展开 original/sample 顶层 JSON
                    exs = _flatten_original_sample_record(r)
                    if exs is not None and len(exs) > 0:
                        examples.extend(exs)
                        continue
                    # 回退为标准 {text,label}
                    nr = _norm_text_label(r)
                    if nr is not None:
                        examples.append(nr)

    else:
        # Iterable 直接标准化，优先尝试 original/sample 展开
        for r in dataset:
            exs = _flatten_original_sample_record(r)
            if exs is not None and len(exs) > 0:
                examples.extend(exs)
                continue
            nr = _norm_text_label(r)
            if nr is not None:
                examples.append(nr)

    if not examples:
        raise ValueError("未加载到有效样本（需包含 'text' 与 'label'）。")

    # 2) 样本级抽样（修复：HC3 保证目标为 2*sample_k）
    if sample_k is not None and sample_k > 0:
        if routed_hc3:
            # 目标：每类各 sample_k 条（总计 2*sample_k）；若不足则取能取到的最大均衡数
            examples = _balanced_sample_two_class(examples, k_per_class=int(sample_k), seed=sample_seed)
        else:
            # 标准数据：总数为 sample_k
            examples = _maybe_sample_exact(examples, int(sample_k), seed=sample_seed)

    # 3) 决定 group_cols
    if group_cols is None or len(group_cols) == 0:
        used = _auto_group_cols(examples, _DEFAULT_GROUP_CANDIDATES)
    else:
        used = list(group_cols)

    return examples, used
