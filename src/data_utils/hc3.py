# mgt_eval/data_utils/hc3.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence
import os, json, glob, re
import numpy as np

HF_DATASET_NAME = "Hello-SimpleAI/HC3"

# ：
# - "all":       human_answers  chatgpt_answers（）
# - "one_each": ， 1 ；
# - "balance_min":  min(len(h), len(c))， k ，“”
# ： sample_on='record'  sample_k>0  "balance_min"； "all"

def _flatten_hc3_record(
    rec: Dict[str, Any],
    pair_mode: str = "all",
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, Any]]:
    """
    将一条 HC3 原始记录展开为多个样本，并保留 source 字段：
      - human_answers[*] -> label=0
      - chatgpt_answers[*] -> label=1
    输出：{"text": str, "label": 0/1, "source": str}
    """
    out: List[Dict[str, Any]] = []
    human_answers = [a.strip() for a in rec.get("human_answers", []) if isinstance(a, str) and a.strip()]
    chat_answers  = [a.strip() for a in rec.get("chatgpt_answers", []) if isinstance(a, str) and a.strip()]
    source = rec.get("source", "unknown")

    if pair_mode == "all":
        out.extend({"text": t, "label": 0, "source": source} for t in human_answers)
        out.extend({"text": t, "label": 1, "source": source} for t in chat_answers)
        return out

    if not human_answers or not chat_answers:
        # ，
        return out

    rng = rng or np.random.RandomState(114514)

    if pair_mode == "one_each":
        h = human_answers[rng.randint(len(human_answers))]
        c = chat_answers[rng.randint(len(chat_answers))]
        out.append({"text": h, "label": 0, "source": source})
        out.append({"text": c, "label": 1, "source": source})
        return out

    if pair_mode == "balance_min":
        k = min(len(human_answers), len(chat_answers))
        if k <= 0:
            return out
        h_idx = rng.choice(len(human_answers), size=k, replace=False)
        c_idx = rng.choice(len(chat_answers),  size=k, replace=False)
        for i in h_idx:
            out.append({"text": human_answers[int(i)], "label": 0, "source": source})
        for j in c_idx:
            out.append({"text": chat_answers[int(j)], "label": 1, "source": source})
        return out

    # ： all
    out.extend({"text": t, "label": 0, "source": source} for t in human_answers)
    out.extend({"text": t, "label": 1, "source": source} for t in chat_answers)
    return out


def _maybe_sample_examples(examples: List[Dict[str, Any]],
                           sample_k: Optional[int],
                           sample_seed: int = 114514) -> List[Dict[str, Any]]:
    """对已展开的样本（example 级）进行全局随机抽样。"""
    if not examples or sample_k is None or sample_k <= 0 or sample_k >= len(examples):
        return examples
    rng = np.random.RandomState(sample_seed)
    idx = rng.choice(len(examples), size=sample_k, replace=False)
    return [examples[int(i)] for i in idx.tolist()]


def _maybe_sample_records(records: List[Dict[str, Any]],
                          sample_k: Optional[int],
                          sample_seed: int = 114514) -> List[Dict[str, Any]]:
    """对记录（record 级）进行随机抽样。"""
    if not records or sample_k is None or sample_k <= 0 or sample_k >= len(records):
        return records
    rng = np.random.RandomState(sample_seed)
    idx = rng.choice(len(records), size=sample_k, replace=False)
    return [records[int(i)] for i in idx.tolist()]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
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


def _expand_any_format_to_text_label(
    recs: List[Dict[str, Any]],
    pair_mode: str = "all",
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, Any]]:
    """
    将（可能混合的）记录列表统一为 [{"text","label","source"}]：
      - 已是 {"text","label"} 的，补 source="unknown" 并接收
      - 若是 HC3 原始结构（含 human_answers/chatgpt_answers），按 pair_mode 展开并保留 source
      - 其他记录类型跳过
    """
    out: List[Dict[str, Any]] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        if "text" in rec and "label" in rec:
            try:
                lbl = int(rec["label"])
                txt = str(rec["text"])
                src = rec.get("source", "unknown")
                out.append({"text": txt, "label": lbl, "source": src})
            except Exception:
                continue
        elif ("human_answers" in rec) or ("chatgpt_answers" in rec):
            out.extend(_flatten_hc3_record(rec, pair_mode=pair_mode, rng=rng))
        else:
            continue
    return out


def load_hf_hc3(
    subset: str = "all",
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    pair_mode: Optional[str] = None,
    sample_on: str = "record",   # "record" | "example"
) -> List[Dict[str, Any]]:
    """从 Hugging Face 加载 HC3 并展开为 {"text","label","source"} 列表。"""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("需要安装 `datasets` 才能从 Hugging Face 加载 HC3，请先 `pip install datasets`.") from e

    raw = load_dataset(HF_DATASET_NAME, subset)

    # split
    records: List[Dict[str, Any]] = []
    if hasattr(raw, "keys"):  # DatasetDict
        for split in raw.keys():
            records.extend(list(raw[split]))
    else:  # Dataset
        records.extend(list(raw))

    rng = np.random.RandomState(sample_seed)
    if pair_mode is None:
        pair_mode = "balance_min" if (sample_on == "record" and sample_k and sample_k > 0) else "all"

    if sample_on == "record":
        records = _maybe_sample_records(records, sample_k, sample_seed)
        examples: List[Dict[str, Any]] = []
        for rec in records:
            examples.extend(_flatten_hc3_record(rec, pair_mode=pair_mode, rng=rng))
        return examples
    else:
        examples = _expand_any_format_to_text_label(records, pair_mode=pair_mode, rng=rng)
        return _maybe_sample_examples(examples, sample_k, sample_seed)


def load_local_hc3(
    path: str,
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    pair_mode: Optional[str] = None,
    sample_on: str = "record",   # "record" | "example"
) -> List[Dict[str, Any]]:
    """从本地加载 HC3，并展开为 {"text","label","source"} 列表。"""
    files: List[str] = []
    if os.path.isdir(path):
        files.extend(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
        files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    elif os.path.isfile(path):
        if path.endswith(".jsonl") or path.endswith(".json"):
            files.append(path)
    else:
        raise FileNotFoundError(f"本地路径不存在：{path}")

    if not files:
        raise FileNotFoundError(f"未在路径中找到 .jsonl/.json 文件：{path}")

    # “”
    records: List[Dict[str, Any]] = []
    for p in files:
        try:
            recs = _read_jsonl(p) if p.endswith(".jsonl") else _read_json(p)
        except Exception:
            continue
        records.extend(recs)

    rng = np.random.RandomState(sample_seed)
    if pair_mode is None:
        pair_mode = "balance_min" if (sample_on == "record" and sample_k and sample_k > 0) else "all"

    if sample_on == "record":
        records = _maybe_sample_records(records, sample_k, sample_seed)
        examples: List[Dict[str, Any]] = []
        for rec in records:
            if isinstance(rec, dict) and ("human_answers" in rec or "chatgpt_answers" in rec):
                examples.extend(_flatten_hc3_record(rec, pair_mode=pair_mode, rng=rng))
            elif isinstance(rec, dict) and ("text" in rec and "label" in rec):
                try:
                    examples.append({"text": str(rec["text"]), "label": int(rec["label"]), "source": rec.get("source", "unknown")})
                except Exception:
                    pass
            else:
                continue
        return examples
    else:
        examples = _expand_any_format_to_text_label(records, pair_mode=pair_mode, rng=rng)
        return _maybe_sample_examples(examples, sample_k, sample_seed)


def should_route_to_hc3(dataset_spec: str) -> bool:
    """判断是否应按 HC3 规则处理：字符串中 **连续** 出现 'hc3'（大小写不敏感）"""
    return re.search(r"hc3", dataset_spec, flags=re.IGNORECASE) is not None


def get_hc3_dataset(
    dataset_spec: str,
    subset: str = "all",
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    pair_mode: Optional[str] = None,
    sample_on: str = "record",
) -> Iterable[Dict[str, Any]]:
    """
    统一入口：
      - 若本地存在 dataset_spec（目录/文件） → 本地加载
      - 否则 → 从 HF 加载 Hello-SimpleAI/HC3
    返回：包含 {"text","label","source"} 的样本列表
    """
    if os.path.exists(dataset_spec):
        return load_local_hc3(
            path=dataset_spec,
            sample_k=sample_k,
            sample_seed=sample_seed,
            pair_mode=pair_mode,
            sample_on=sample_on,
        )
    return load_hf_hc3(
        subset=subset,
        sample_k=sample_k,
        sample_seed=sample_seed,
        pair_mode=pair_mode,
        sample_on=sample_on,
    )
