# mgt_eval/data_utils/load.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os, json, glob, re, csv
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

def _read_csv(path: str) -> List[Dict[str, Any]]:
    """
    读取带表头的CSV并转成记录列表(dict)。特别兼容以下列名：
      - 文本列：优先 text；否则回退到 answer / generation / content / article / body
      - 标签列：label（数值0/1，或字符串 '0'/'1'/'human'/'machine'/... 均可）
      - 源列：source 或 src
      - 其他列（如 id / question 等）原样保留
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        lower_to_orig = {name.lower(): name for name in (reader.fieldnames or [])}

        def get_col(*cands: str) -> Optional[str]:
            for c in cands:
                if c in lower_to_orig:
                    return lower_to_orig[c]
            return None

        col_text = get_col("text") or get_col("answer", "generation", "content", "article", "body")
        col_label = get_col("label")
        col_source = get_col("source") or get_col("src")

        for row in reader:
            rec: Dict[str, Any] = dict(row)
            txt = rec.get(col_text) if col_text else None
            if txt is None or str(txt).strip() == "":
                continue
            if col_label is None or rec.get(col_label) is None or str(rec.get(col_label)).strip() == "":
                continue

            rec["text"] = str(txt)
            rec["label"] = rec[col_label]
            if col_source is not None:
                rec["source"] = rec.get(col_source)
            out.append(rec)
    return out

def _list_json_files(path: str) -> List[str]:
    """
    递归列出数据文件；支持 .jsonl/.json/.csv
    """
    patterns = ["*.jsonl", "*.json", "*.csv"]
    files: List[str] = []
    if os.path.isdir(path):
        for pat in patterns:
            files += glob.glob(os.path.join(path, "**", pat), recursive=True)
        return files
    if os.path.isfile(path) and any(path.endswith(ext) for ext in [".jsonl", ".json", ".csv"]):
        return [path]
    return []

def _maybe_sample_exact(items: List[Any], k: Optional[int], seed: int = 114514) -> List[Any]:
    if not items or k is None or k <= 0:
        return items
    n = len(items)
    if k >= n:
        return items
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [items[int(i)] for i in idx.tolist()]

# ---------------------------
# 统一规范化：text/label + 保留其余字段
# ---------------------------
def _norm_text_label(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rec, dict):
        return None

    text = rec.get("text", None)
    if text is None:
        text = rec.get("article", None)  # CoCo
    if text is None:
        return None

    raw_label = rec.get("label", None)
    if raw_label is None:
        return None

    def _to_label(val) -> Optional[int]:
        if isinstance(val, (int, np.integer)):
            iv = int(val)
            if iv in (0, 1):
                return iv
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("0", "1"):
                return int(s)
            if s in ("human", "hum", "real", "reference", "gold", "ref"):
                return 0
            if s in ("machine", "ai", "model", "generated", "gen", "chatgpt", "gpt", "bot", "fake"):
                return 1
        if isinstance(val, bool):
            return 1 if val else 0
        return None

    y = _to_label(raw_label)
    if y is None:
        return None

    try:
        return {**rec, "text": str(text), "label": int(y)}
    except Exception:
        return None

# ---------------------------
# original / sample 顶层 JSON 展开
# ---------------------------
def _flatten_original_sample_record(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    展开形如:
      {"original": [...], "sample": [...], "sampled": [...], "rewritten": [...], <上下文...>}
    为样本 [{text, label, split, ...}]：
      - original  → label = 0, split = "original"
      - sample    → label = 1, split = "sample"
      - sampled   → label = 1, split = "sampled"
      - rewritten → label = 1, split = "rewritten"

    兼容 list 元素为：
      - str
      - {"text": "...", "id": "...", ...}  （会保留 id 及其余字段）
    """
    if not isinstance(rec, dict):
        return None

    has_original = "original" in rec
    has_any_machine = any(k in rec for k in ("sample", "sampled", "rewritten"))
    if not (has_original or has_any_machine):
        return None

    def _to_item_list(lst: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(lst, (list, tuple)):
            for x in lst:
                if isinstance(x, dict):
                    # 最低要求：必须能拿到 text
                    if "text" in x and isinstance(x["text"], str):
                        t = x["text"].strip()
                        if not t:
                            continue
                        # 保留除 label/split 之外所有字段（防止污染我们后续写入的 label/split）
                        item = {k: v for k, v in x.items() if k not in ("label", "split")}
                        item["text"] = t
                        out.append(item)
                    else:
                        # dict 但没有 text，就忽略
                        continue
                else:
                    t = str(x).strip()
                    if t:
                        out.append({"text": t})
        return out

    ctx = {k: v for k, v in rec.items() if k not in ("original", "sample", "sampled", "rewritten")}

    ori_items = _to_item_list(rec.get("original", []))
    sam_items = _to_item_list(rec.get("sample", []))
    sampled_items = _to_item_list(rec.get("sampled", []))
    rewrite_items = _to_item_list(rec.get("rewritten", []))

    out: List[Dict[str, Any]] = []
    out += [{**ctx, **it, "label": 0, "split": "original"} for it in ori_items]
    out += [{**ctx, **it, "label": 1, "split": "sample"} for it in sam_items]
    out += [{**ctx, **it, "label": 1, "split": "sampled"} for it in sampled_items]
    out += [{**ctx, **it, "label": 1, "split": "rewritten"} for it in rewrite_items]
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

    k = min(len(human), len(machine))
    h_idx = rng.choice(len(human), size=k, replace=False)
    m_idx = rng.choice(len(machine), size=k, replace=False)
    out += [{"text": human[int(i)], "label": 0, "source": source} for i in h_idx]
    out += [{"text": machine[int(j)], "label": 1, "source": source} for j in m_idx]
    return out

def _load_hc3_all_from_hf() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("需要安装 datasets 才能从 HF 加载 HC3：pip install datasets") from e
    raw = load_dataset(HF_HC3, "all")
    records: List[Dict[str, Any]] = []
    if hasattr(raw, "keys"):
        for sp in raw.keys():
            records.extend(list(raw[sp]))
    else:
        records.extend(list(raw))
    rng = np.random.RandomState(114514)
    exs: List[Dict[str, Any]] = []
    for rec in records:
        exs.extend(_flatten_hc3_record(rec, pair_mode="balance_min", rng=rng))
    return exs

def _load_hc3_all_from_local(path: str) -> List[Dict[str, Any]]:
    files = _list_json_files(path)
    if not files:
        raise FileNotFoundError(f"未在路径中找到 HC3 文件：{path}")

    records: List[Dict[str, Any]] = []
    for p in files:
        try:
            if p.endswith(".jsonl"):
                recs = _read_jsonl(p)
            elif p.endswith(".json"):
                recs = _read_json(p)
            elif p.endswith(".csv"):
                recs = _read_csv(p)
            else:
                recs = []
        except Exception:
            continue
        records.extend(recs)
    if not records:
        raise ValueError(f"在 {path} 未解析到 HC3 记录")

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
# 抽样辅助：HC3 均衡抽样
# ---------------------------
def _balanced_sample_two_class(examples: List[Dict[str, Any]],
                               k_per_class: int,
                               seed: int = 114514) -> List[Dict[str, Any]]:
    if not examples or k_per_class is None or k_per_class <= 0:
        return examples

    cls0 = [ex for ex in examples if int(ex.get("label", 0)) == 0]
    cls1 = [ex for ex in examples if int(ex.get("label", 0)) == 1]

    n0, n1 = len(cls0), len(cls1)
    if n0 == 0 or n1 == 0:
        return _maybe_sample_exact(examples, 2 * k_per_class, seed=114514)

    k = min(k_per_class, n0, n1)
    rng = np.random.RandomState(seed)
    idx0 = rng.choice(n0, size=k, replace=False)
    idx1 = rng.choice(n1, size=k, replace=False)
    sub0 = [cls0[int(i)] for i in idx0.tolist()]
    sub1 = [cls1[int(i)] for i in idx1.tolist()]
    merged = sub0 + sub1
    if merged:
        perm = rng.permutation(len(merged))
        merged = [merged[int(i)] for i in perm.tolist()]
    return merged

# ---------------------------
# 自动分组列探测
# ---------------------------
_DEFAULT_GROUP_CANDIDATES = ["id", "qid", "question_id", "question",
                             "lang", "source", "model", "sub_source", "split"]

def _auto_group_cols(examples: List[Dict[str, Any]],
                     candidates: Sequence[str] = _DEFAULT_GROUP_CANDIDATES) -> List[str]:
    chosen: List[str] = []
    if not examples:
        return chosen
    for col in candidates:
        exists = False
        for ex in examples[:512]:
            v = ex.get(col, None)
            if v is not None and str(v).strip() != "":
                exists = True
                break
        if exists:
            chosen.append(col)
    return chosen

# ---------------------------
# 解析逗号分隔的数据集规格
# ---------------------------
def _split_dataset_specs(spec: str) -> List[str]:
    if not isinstance(spec, str):
        return []
    parts = [p.strip() for p in spec.split(",")]
    return [p for p in parts if p]

# ---------------------------
# 通用加载入口（单条样本）
# ---------------------------
def load_dataset_unified(
    dataset: Union[str, Iterable[Dict[str, Any]]],
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    group_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    统一数据集加载（支持多数据集合并，且对 .csv/.jsonl/.json 一视同仁）：
      - 支持 HC3；支持 original/sample(s)/rewritten 顶层结构的展开
      - 返回 examples: [{text,label,...}], 与 group_cols（用于统计/可选配对）
    """
    examples: List[Dict[str, Any]] = []

    if isinstance(dataset, str):
        specs = _split_dataset_specs(dataset)
        if len(specs) <= 1:
            routed_hc3 = should_route_to_hc3(dataset)
            if routed_hc3:
                if os.path.exists(dataset):
                    examples = _load_hc3_all_from_local(dataset)
                else:
                    examples = _load_hc3_all_from_hf()
            else:
                files = _list_json_files(dataset)
                if not files:
                    raise FileNotFoundError(f"未找到 CSV/JSON/JSONL：{dataset}")
                for p in files:
                    if p.endswith(".jsonl"):
                        recs = _read_jsonl(p)
                    elif p.endswith(".json"):
                        recs = _read_json(p)
                    elif p.endswith(".csv"):
                        recs = _read_csv(p)
                    else:
                        recs = []
                    for r in recs:
                        exs = _flatten_original_sample_record(r)
                        if exs:
                            examples.extend(exs)
                        else:
                            nr = _norm_text_label(r)
                            if nr is not None:
                                examples.append(nr)
            only_hc3 = should_route_to_hc3(dataset)
        else:
            all_parts: List[Dict[str, Any]] = []
            parts_is_hc3: List[bool] = []
            for spec in specs:
                is_hc3 = should_route_to_hc3(spec)
                parts_is_hc3.append(is_hc3)
                if is_hc3:
                    if os.path.exists(spec):
                        part = _load_hc3_all_from_local(spec)
                    else:
                        part = _load_hc3_all_from_hf()
                    all_parts.extend(part)
                else:
                    files = _list_json_files(spec)
                    if not files:
                        raise FileNotFoundError(f"未找到 CSV/JSON/JSONL：{spec}")
                    for p in files:
                        if p.endswith(".jsonl"):
                            recs = _read_jsonl(p)
                        elif p.endswith(".json"):
                            recs = _read_json(p)
                        elif p.endswith(".csv"):
                            recs = _read_csv(p)
                        else:
                            recs = []
                        for r in recs:
                            exs = _flatten_original_sample_record(r)
                            if exs:
                                all_parts.extend(exs)
                            else:
                                nr = _norm_text_label(r)
                                if nr is not None:
                                    all_parts.append(nr)
            examples = all_parts
            only_hc3 = (len(parts_is_hc3) > 0 and all(parts_is_hc3))
    else:
        for r in dataset:
            exs = _flatten_original_sample_record(r)
            if exs:
                examples.extend(exs)
            else:
                nr = _norm_text_label(r)
                if nr is not None:
                    examples.append(nr)
        only_hc3 = False

    if not examples:
        raise ValueError("未加载到有效样本（需包含 'text' 与 'label'）。")

    if sample_k is not None and sample_k > 0:
        if only_hc3:
            examples = _balanced_sample_two_class(examples, k_per_class=int(sample_k), seed=sample_seed)
        else:
            examples = _maybe_sample_exact(examples, int(sample_k), seed=sample_seed)

    if group_cols is None or len(group_cols) == 0:
        used = _auto_group_cols(examples, _DEFAULT_GROUP_CANDIDATES)
    else:
        used = list(group_cols)

    return examples, used

# ---------------------------
# 新增：成对加载入口（ImBD-SPO 需要）
# ---------------------------
def load_dataset_unified_pairs(
    dataset: Union[str, Iterable[Dict[str, Any]]],
    sample_k_pairs: Optional[int] = None,
    sample_seed: int = 114514,
    pair_by: Optional[str] = None,
    pubmed_answer_cut: bool = True,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    将统一加载的单条样本 [{text,label,...}] 转换为 (human, machine) 成对样本：
      - 优先使用上下文键配对：'id'/'qid'/'question_id'/'question'（或显式 pair_by）
      - 若无上述键，则退化到基于组的配对（如 'source'/'lang'），再退化到全局随机配对
      - 对于包含 'pubmed' 的数据集规格，且 pubmed_answer_cut=True，则对文本做 'Answer:' 截断

    返回：
      pairs: List[Tuple[str, str]]  # (original/human, rewritten/machine)
      stats: Dict  # 包含分组/配对统计信息
    """
    # 先加载“单条样本”，不要在这里 sample，避免破坏配对
    examples, group_cols = load_dataset_unified(dataset, sample_k=None, sample_seed=sample_seed)

    # 可选的 PubMed 'Answer:' 截断（按数据集路径字符串判断）
    def _maybe_cut_answer(txt: str) -> str:
        if not isinstance(txt, str):
            return txt
        parts = txt.split("Answer:")
        if len(parts) >= 2:
            return parts[1].strip()
        return txt

    is_pubmed_like = False
    if isinstance(dataset, str):
        for spec in _split_dataset_specs(dataset):
            if re.search(r"pubmed", spec, flags=re.IGNORECASE):
                is_pubmed_like = True
                break

    if is_pubmed_like and pubmed_answer_cut:
        for ex in examples:
            ex["text"] = _maybe_cut_answer(ex["text"])

    # 分组策略：优先 id/qid/question_id/question；否则 source/lang；否则 None（全局）
    priority_keys = ["id", "qid", "question_id", "question", "source", "lang"]
    if pair_by:
        # 显式指定优先（允许传入任意存在字段）
        priority_keys = [pair_by] + [k for k in priority_keys if k != pair_by]

    # 计算每条样本的“分组键”
    def _key_for(ex: Dict[str, Any]) -> Optional[str]:
        for k in priority_keys:
            if k in ex and ex[k] is not None and str(ex[k]).strip() != "":
                return f"{k}:{str(ex[k])}"
        return None  # 无上下文 → 进入“全局配对池”

    buckets: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for ex in examples:
        buckets.setdefault(_key_for(ex), []).append(ex)

    # 在每个桶内做配对：human(0) vs machine(1) 逐一配对
    rng = np.random.RandomState(sample_seed)
    pairs: List[Tuple[str, str]] = []
    per_bucket_stats: Dict[str, Dict[str, int]] = {}

    for k, items in buckets.items():
        human = [it for it in items if int(it.get("label", 0)) == 0 and isinstance(it.get("text", None), str)]
        mach  = [it for it in items if int(it.get("label", 0)) == 1 and isinstance(it.get("text", None), str)]
        if not human or not mach:
            continue
        rng.shuffle(human)
        rng.shuffle(mach)
        m = min(len(human), len(mach))
        for i in range(m):
            pairs.append((human[i]["text"].strip(), mach[i]["text"].strip()))
        per_bucket_stats[str(k)] = {"human": len(human), "machine": len(mach), "paired": m}

    # 如果没有任何按桶配对成功，退化到全局配对
    if not pairs:
        human = [it for it in examples if int(it.get("label", 0)) == 0 and isinstance(it.get("text", None), str)]
        mach  = [it for it in examples if int(it.get("label", 0)) == 1 and isinstance(it.get("text", None), str)]
        rng.shuffle(human)
        rng.shuffle(mach)
        m = min(len(human), len(mach))
        pairs = [(human[i]["text"].strip(), mach[i]["text"].strip()) for i in range(m)]
        per_bucket_stats["__global__"] = {"human": len(human), "machine": len(mach), "paired": m}

    # 可选：对配对结果整体再抽样（对“对数”）
    if sample_k_pairs is not None and sample_k_pairs > 0:
        pairs = _maybe_sample_exact(pairs, int(sample_k_pairs), seed=sample_seed)

    stats = {
        "num_pairs": len(pairs),
        "num_buckets": len(per_bucket_stats),
        "bucket_stats": per_bucket_stats,
        "pair_by_used": priority_keys[0] if priority_keys else None,
        "group_cols_detected": group_cols,
        "pubmed_answer_cut": bool(is_pubmed_like and pubmed_answer_cut),
    }
    return pairs, stats
