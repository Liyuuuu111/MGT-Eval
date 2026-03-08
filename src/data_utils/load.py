# mgt_eval/data_utils/load.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os, json, glob, re, csv
from pathlib import Path
import numpy as np

HF_HC3 = "Hello-SimpleAI/HC3"

# ---------------------------
# I/O
# ---------------------------
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _fixed_jsonl_path(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return str(p.with_name(f"{p.stem}.fixed.jsonl"))
    return str(p.with_name(f"{p.name}.fixed.jsonl"))


def _write_jsonl_records(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    strict_mode = _env_flag("MGT_EVAL_JSONL_STRICT", False)
    export_fixed = _env_flag("MGT_EVAL_JSONL_EXPORT_FIXED", True)
    # JSONL 默认按“单行一条记录”处理，避免坏行吞掉后续正常记录。
    # 若你确实有跨行 JSON 对象，可显式设置为 true。
    multiline_recover = _env_flag("MGT_EVAL_JSONL_MULTILINE_RECOVER", False)
    try:
        max_warn_lines = int(os.getenv("MGT_EVAL_JSONL_WARN_LINES", "10") or 10)
    except Exception:
        max_warn_lines = 10
    try:
        max_recover_lines = int(os.getenv("MGT_EVAL_JSONL_RECOVER_MAX_LINES", "200") or 200)
    except Exception:
        max_recover_lines = 200
    max_recover_lines = max(1, max_recover_lines)

    out: List[Dict[str, Any]] = []
    # (start_line, end_line, err, preview)
    bad_lines: List[Tuple[int, int, str, str]] = []

    def _parse_obj(text: str) -> Dict[str, Any]:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise TypeError(f"expected JSON object per line, got {type(obj).__name__}")
        return obj

    buf_text: Optional[str] = None
    buf_start: int = 0
    buf_count: int = 0
    buf_first_err: str = ""

    def _flush_bad_buffer(end_line: int, reason: str, record_bad: bool = True) -> None:
        nonlocal buf_text, buf_start, buf_count, buf_first_err
        if buf_text is None:
            return
        if record_bad:
            preview = buf_text[:200].replace("\t", "\\t").replace("\n", "\\n")
            bad_lines.append((buf_start, end_line, reason, preview))
        buf_text = None
        buf_start = 0
        buf_count = 0
        buf_first_err = ""

    def _start_buffer(line_no: int, text: str, err: Exception) -> None:
        nonlocal buf_text, buf_start, buf_count, buf_first_err
        buf_text = text
        buf_start = line_no
        buf_count = 1
        buf_first_err = str(err)

    def _try_parse_buffer() -> Optional[Dict[str, Any]]:
        if buf_text is None:
            return None
        return _parse_obj(buf_text)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, ln in enumerate(f, start=1):
            raw = ln.rstrip("\n")
            stripped = raw.strip()

            # Buffering mode: keep accumulating until parse succeeds or we decide to resync.
            if buf_text is not None:
                standalone_err: Optional[Exception] = None
                # 关键修复：如果当前行本身可独立解析为 JSON 对象，直接断开旧坏块，
                # 避免把后续正常记录吞进前一个坏对象。
                if stripped.startswith("{"):
                    try:
                        obj = _parse_obj(stripped)
                        reason = f"discarded broken multiline fragment: {buf_first_err or 'parse error'}"
                        _flush_bad_buffer(line_no - 1, reason=reason, record_bad=True)
                        out.append(obj)
                        continue
                    except Exception as e:
                        standalone_err = e

                buf_count += 1
                buf_text = f"{buf_text}\n{raw}"
                try:
                    obj = _try_parse_buffer()
                    if obj is not None:
                        out.append(obj)
                        _flush_bad_buffer(line_no, reason="buffer_resolved", record_bad=False)
                    continue
                except Exception as e:
                    # Resync heuristic: a new object likely starts here.
                    # If current stripped line starts with '{', previous buffer likely irrecoverable.
                    if multiline_recover and stripped.startswith("{") and buf_count > 1:
                        reason = f"unrecoverable multiline object: {buf_first_err or str(e)}"
                        _flush_bad_buffer(line_no - 1, reason=reason, record_bad=True)
                        # Restart from current line.
                        restart_err = standalone_err if standalone_err is not None else e
                        _start_buffer(line_no, stripped, restart_err)
                        continue

                    if buf_count >= max_recover_lines:
                        reason = (
                            f"multiline recovery exceeded {max_recover_lines} lines; "
                            f"first_error={buf_first_err or str(e)}"
                        )
                        _flush_bad_buffer(line_no, reason=reason, record_bad=True)
                    continue

            # Normal mode (no active buffer)
            if not stripped:
                continue
            try:
                obj = _parse_obj(stripped)
                out.append(obj)
                continue
            except Exception as e:
                if strict_mode:
                    preview = stripped[:200].replace("\t", "\\t")
                    raise ValueError(
                        f"Invalid JSONL record at {path}:{line_no}: {str(e)}. "
                        f"Preview: {preview}"
                    ) from e
                if multiline_recover and stripped.startswith("{"):
                    _start_buffer(line_no, stripped, e)
                    continue
                preview = stripped[:200].replace("\t", "\\t")
                bad_lines.append((line_no, line_no, str(e), preview))
                continue

    # EOF with unresolved buffered object
    if buf_text is not None:
        reason = f"incomplete multiline object at EOF: {buf_first_err or 'unknown parse error'}"
        _flush_bad_buffer(buf_start + max(0, buf_count - 1), reason=reason, record_bad=True)

    if bad_lines:
        if not out:
            raise ValueError(
                f"All JSONL records are invalid in {path}. "
                f"First bad line range: {bad_lines[0][0]}-{bad_lines[0][1]} ({bad_lines[0][2]})."
            )

        fixed_path = _fixed_jsonl_path(path)
        if export_fixed:
            _write_jsonl_records(fixed_path, out)
            print(
                f"[load_dataset_unified] JSONL repaired: {path} | "
                f"valid={len(out)}, skipped={len(bad_lines)} | "
                f"exported={fixed_path}"
            )
        else:
            print(
                f"[load_dataset_unified] JSONL contains bad lines: {path} | "
                f"valid={len(out)}, skipped={len(bad_lines)}"
            )

        limit = max(1, max_warn_lines)
        for start_line, end_line, err_msg, preview in bad_lines[:limit]:
            line_info = str(start_line) if start_line == end_line else f"{start_line}-{end_line}"
            print(
                f"[load_dataset_unified] skipped bad JSONL line(s) {line_info}: "
                f"{err_msg} | preview={preview}"
            )
        if len(bad_lines) > limit:
            print(
                f"[load_dataset_unified] ... and {len(bad_lines) - limit} more bad lines."
            )

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

def _mgt_eval_root() -> str:
    try:
        here = Path(__file__).resolve()
        # src/data_utils/load.py -> <repo_root>
        return str(here.parents[2])
    except Exception:
        return os.getcwd()

def _looks_like_path(spec: str) -> bool:
    if not spec:
        return False
    s = spec.strip()
    if any(s.lower().endswith(ext) for ext in [".jsonl", ".json", ".csv"]):
        return True
    if s.startswith(".") or s.startswith("~"):
        return True
    if os.sep in s:
        return True
    return False

def _resolve_dataset_spec(spec: str) -> str:
    if not isinstance(spec, str) or not spec.strip():
        return spec
    s = os.path.expandvars(os.path.expanduser(spec.strip()))
    if os.path.exists(s):
        return s
    if _looks_like_path(s):
        root = _mgt_eval_root()
        cand = os.path.join(root, s)
        if os.path.exists(cand):
            return cand
    return s

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
# ：text/label +
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
# original / sample  JSON
# ---------------------------
def _flatten_original_sample_record(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    展开形如:
      {"original": [...], "sample": [...], "sampled": [...], "rewritten": [...], <上下文...>}
    为样本 [{text, label, split, ...}]。

    默认行为：
      - original  → label = 0, split = "original"
      - sample    → label = 1, split = "sample"
      - sampled   → label = 1, split = "sampled"
      - rewritten → label = 1, split = "rewritten"

    ✅ attack-only builder record（meta.attack_dataset_only=True）：
      - 只输出 sample（label=1），忽略 original/prompt
      - 支持 sample 内包含多个攻击变体（不会只留最后一个）
      - 若 sample 中包含 src/base/original 之类“非攻击”项，会自动过滤
      - attack 字段统一优先用 meta.active_attack（如 text_tran）；同时保留 attack_short（如 tran）
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
                    if "text" in x and isinstance(x["text"], str):
                        t = x["text"].strip()
                        if not t:
                            continue
                        item = {k: v for k, v in x.items() if k not in ("label", "split")}
                        item["text"] = t
                        out.append(item)
                    else:
                        continue
                else:
                    t = str(x).strip()
                    if t:
                        out.append({"text": t})
        return out

    # -------------------------
    # attack-only special case
    # -------------------------
    meta = rec.get("meta", None)
    meta = meta if isinstance(meta, dict) else {}

    is_attack_only = bool(meta.get("attack_dataset_only", False))
    if not is_attack_only:
        b = str(meta.get("builder", "") or "").strip().lower()
        if b.endswith("dataset_attack_only") or ("attack_only" in b):
            is_attack_only = True

    if is_attack_only:
        sam_items = _to_item_list(rec.get("sample", []))
        if not sam_items:
            return None

        # src/base/original “”
        filtered: List[Dict[str, Any]] = []
        for it in sam_items:
            atk = str(it.get("attack") or it.get("aug_method") or "").strip().lower()
            role = str(it.get("role") or "").strip().lower()
            if atk in ("src", "source", "base", "orig", "original") or role in ("src", "source", "base", "orig", "original"):
                continue
            filtered.append(it)
        if filtered:
            sam_items = filtered

        base_id = rec.get("id", None)
        base_id = str(base_id) if base_id is not None else ""

        # ctx
        ctx_keep_keys = ("lang", "model", "source")
        ctx: Dict[str, Any] = {}
        for k in ctx_keep_keys:
            if k in rec and rec.get(k, None) is not None and str(rec.get(k)).strip() != "":
                ctx[k] = rec.get(k)

        active_attack = meta.get("active_attack", None)
        active_attack_s = str(active_attack).strip() if active_attack is not None else ""

        # “”， text_
        has_text_attack_meta = isinstance(meta.get("text_attack_meta", None), list)

        out: List[Dict[str, Any]] = []
        for it in sam_items:
            text = it.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            # id： sample item  id（ base ）
            sid = it.get("id", None)
            ex_id = str(sid) if sid is not None and str(sid).strip() != "" else base_id

            lang = it.get("lang", None) or ctx.get("lang", None)
            model = it.get("model", None) or ctx.get("model", None)
            source = rec.get("source", None) or ctx.get("source", None)

            attack_short = it.get("attack", None) or it.get("aug_method", None) or it.get("attack_method", None) or it.get("attack_type", None)
            attack_short_s = str(attack_short).strip() if attack_short is not None else ""

            # ✅  attack ： meta.active_attack（ text_tran / text_form_zero-sp）
            attack = ""
            if active_attack_s:
                attack = active_attack_s
            elif attack_short_s:
                # active_attack ， text_*
                if has_text_attack_meta and not attack_short_s.startswith("text_"):
                    attack = "text_" + attack_short_s
                else:
                    attack = attack_short_s

            ex: Dict[str, Any] = {
                "id": ex_id,
                "base_id": base_id,
                "text": text.strip(),
                "label": 1,
                "split": "attack",
            }

            if lang is not None and str(lang).strip() != "":
                ex["lang"] = lang
            if model is not None and str(model).strip() != "":
                ex["model"] = model
            if source is not None and str(source).strip() != "":
                ex["source"] = source

            if attack:
                ex["attack"] = attack
            # ， tran/subs
            if attack_short_s and (not attack or attack_short_s != attack):
                ex["attack_short"] = attack_short_s

            out.append(ex)

        return out if out else None

    # -------------------------
    # default behavior
    # -------------------------
    ctx = {k: v for k, v in rec.items() if k not in ("original", "sample", "sampled", "rewritten")}

    ori_items = _to_item_list(rec.get("original", []))
    sam_items = _to_item_list(rec.get("sample", []))
    sampled_items = _to_item_list(rec.get("sampled", []))
    rewrite_items = _to_item_list(rec.get("rewritten", []))

    out2: List[Dict[str, Any]] = []
    out2 += [{**ctx, **it, "label": 0, "split": "original"} for it in ori_items]
    out2 += [{**ctx, **it, "label": 1, "split": "sample"} for it in sam_items]
    out2 += [{**ctx, **it, "label": 1, "split": "sampled"} for it in sampled_items]
    out2 += [{**ctx, **it, "label": 1, "split": "rewritten"} for it in rewrite_items]
    return out2

# ---------------------------
# HC3 （）
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
    try:
        raw = load_dataset(HF_HC3, "all")
    except Exception as e:
        # datasets ， data files
        msg = str(e)
        if "Dataset scripts are no longer supported" in msg or "trust_remote_code" in msg:
            raw = _load_hc3_all_from_hf_datafiles()
        else:
            raise
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

def _load_hc3_all_from_hf_datafiles():
    """
    当 datasets 禁用脚本时，尝试从 HF repo 中直接读取 data files（parquet/json/jsonl/csv）。
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("需要安装 datasets 才能从 HF 加载 HC3：pip install datasets") from e
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise RuntimeError(
            "当前 datasets 不支持脚本加载，且缺少 huggingface_hub。"
            "请先安装：pip install huggingface_hub"
        ) from e

    api = HfApi()
    try:
        repo_files = api.list_repo_files(HF_HC3, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(f"无法列出 HF 数据集文件：{HF_HC3}，请检查网络/权限。") from e

    def pick_files(exts):
        out = []
        for f in repo_files:
            lf = f.lower()
            if any(lf.endswith(ext) for ext in exts):
                out.append(f)
        return out

    # parquet， jsonl/json， csv
    for exts, loader in [
        ([".parquet"], "parquet"),
        ([".jsonl", ".json"], "json"),
        ([".csv"], "csv"),
    ]:
        files = pick_files(exts)
        if files:
            return load_dataset(loader, data_files=files)

    raise RuntimeError(
        f"HF 数据集 {HF_HC3} 未找到可直接加载的数据文件（parquet/json/jsonl/csv）。"
    )

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
# ：HC3
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
# ---------------------------
_DEFAULT_GROUP_CANDIDATES = ["id", "base_id", "qid", "question_id", "question",
                             "lang", "source", "model", "sub_source", "attack", "split"]

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
# ---------------------------
def _split_dataset_specs(spec: str) -> List[str]:
    if not isinstance(spec, str):
        return []
    parts = [p.strip() for p in spec.split(",")]
    return [p for p in parts if p]

# ---------------------------
# （）
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
        specs = [_resolve_dataset_spec(s) for s in _split_dataset_specs(dataset)]
        if len(specs) <= 1:
            routed_hc3 = should_route_to_hc3(specs[0]) if specs else should_route_to_hc3(dataset)
            if routed_hc3:
                if specs and os.path.exists(specs[0]):
                    examples = _load_hc3_all_from_local(specs[0])
                else:
                    examples = _load_hc3_all_from_hf()
            else:
                spec0 = specs[0] if specs else dataset
                files = _list_json_files(spec0)
                if not files:
                    raise FileNotFoundError(f"未找到 CSV/JSON/JSONL：{spec0}")
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
            only_hc3 = should_route_to_hc3(specs[0]) if specs else should_route_to_hc3(dataset)
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
# ：（ImBD-SPO ）
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
      stats: Dict  # /
    """
    # “”， sample，
    examples, group_cols = load_dataset_unified(dataset, sample_k=None, sample_seed=sample_seed)

    # PubMed 'Answer:' （）
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

    # ： id/qid/question_id/question； source/lang； None（）
    priority_keys = ["id", "qid", "question_id", "question", "source", "lang"]
    if pair_by:
        # （）
        priority_keys = [pair_by] + [k for k in priority_keys if k != pair_by]

    # “”
    def _key_for(ex: Dict[str, Any]) -> Optional[str]:
        for k in priority_keys:
            if k in ex and ex[k] is not None and str(ex[k]).strip() != "":
                return f"{k}:{str(ex[k])}"
        return None  # → “”

    buckets: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for ex in examples:
        buckets.setdefault(_key_for(ex), []).append(ex)

    # ：human(0) vs machine(1)
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

    # ，
    if not pairs:
        human = [it for it in examples if int(it.get("label", 0)) == 0 and isinstance(it.get("text", None), str)]
        mach  = [it for it in examples if int(it.get("label", 0)) == 1 and isinstance(it.get("text", None), str)]
        rng.shuffle(human)
        rng.shuffle(mach)
        m = min(len(human), len(mach))
        pairs = [(human[i]["text"].strip(), mach[i]["text"].strip()) for i in range(m)]
        per_bucket_stats["__global__"] = {"human": len(human), "machine": len(mach), "paired": m}

    # ：（“”）
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
