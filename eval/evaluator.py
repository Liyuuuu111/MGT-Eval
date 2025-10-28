# mgt_eval/eval/evaluator.py
from typing import Any, Dict, Iterable, Optional, Union, Sequence, List, Tuple
import os
import io
import sys
import json
import time
import math
import platform
from pathlib import Path
from datetime import datetime
import re
from ..detectors.base import DetectorBase, EvalResult
from ..detectors.registry import get_detector_cls
from ..data_utils.load import load_dataset_unified

# ---------- 可选依赖 ----------
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

# ---- 新增：显存统计/环境辅助 ----
try:
    import torch
except Exception:
    torch = None  # 允许无 torch 环境导入


def _bytes_to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)

import math
from typing import Optional, List, Dict, Any, Tuple

def _is_finite_scalar(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False

def _filter_nonfinite_examples(
    examples: List[Dict[str, Any]],
    labels: List[int],
    preds: List[int],
    *,
    ranking_vec: Optional[List[float]] = None,
    probs: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], Optional[List[float]], Optional[List[float]], int]:
    """
    统一过滤：只要 ranking 或 probs 中任一为非 finite(含 None/NaN/Inf) 就丢弃该样本。
    返回 (ex, y, pred, ranking, probs, dropped)
    """
    n = len(labels)
    keep = [True] * n
    if ranking_vec is not None and len(ranking_vec) == n:
        for i, v in enumerate(ranking_vec):
            if v is None or (not _is_finite_scalar(v)):
                keep[i] = False
    if probs is not None and len(probs) == n:
        for i, v in enumerate(probs):
            if v is None or (not _is_finite_scalar(v)):
                keep[i] = False

    ex2 = [e for e, m in zip(examples, keep) if m]
    y2  = [y for y, m in zip(labels, keep) if m]
    p2  = [p for p, m in zip(preds, keep) if m]
    r2  = None if ranking_vec is None else [s for s, m in zip(ranking_vec, keep) if m]
    pb2 = None if probs is None else [s for s, m in zip(probs, keep) if m]
    dropped = n - len(y2)
    return ex2, y2, p2, r2, pb2, dropped

def _cuda_devices() -> List[int]:
    if (torch is None) or (not hasattr(torch, "cuda")) or (not torch.cuda.is_available()):
        return []
    try:
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []

# --- Multilingual word count helper (uses `regex` if available) ---
def _word_count(text: str) -> int:
    """
    Return a language-agnostic word count.
    - If third-party `regex` is available, use Unicode-aware pattern:
        * English-like: sequences of letters with optional - or '
        * Numbers: digits
        * CJK: Han (每个汉字计 1), Hiragana/Katakana/Hangul（连续算一段）
    - Fallback: whitespace tokenization.
    """
    s = "" if text is None else str(text)
    try:
        import regex as re_u  # pip install regex
        patt = re_u.compile(
            r"(?:\p{Han})|(?:\p{Hiragana}+)|(?:\p{Katakana}+)|(?:\p{Hangul}+)"
            r"|(?:[A-Za-z]+(?:[-'][A-Za-z]+)*)|(?:\d+)",
            re_u.UNICODE
        )
        return len(patt.findall(s))
    except Exception:
        # Fallback: whitespace tokens
        return len([t for t in s.strip().split() if t])

def _reset_and_mark_cuda_peaks() -> Dict[str, Any]:
    """
    在评测前调用：重置所有可见 CUDA 设备的峰值统计，并记录设备名信息。
    返回一个上下文字典，供事后收集使用。
    """
    ctx: Dict[str, Any] = {
        "cuda_available": bool(_cuda_devices()),
        "devices": [],
    }
    devs = _cuda_devices()
    for idx in devs:
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass
        name = None
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = f"cuda:{idx}"
        ctx["devices"].append({"index": idx, "name": name})
    return ctx


def _collect_cuda_peaks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    在评测后调用：读取每张卡的峰值显存（allocated/reserved）。
    """
    out: Dict[str, Any] = {
        "cuda_available": bool(ctx.get("cuda_available", False)),
        "per_device": [],
        "total_peak_allocated_gib": 0.0,
        "total_peak_reserved_gib": 0.0,
    }
    if not out["cuda_available"]:
        return out

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    total_alloc = 0
    total_res = 0
    for d in ctx.get("devices", []):
        idx = int(d["index"])
        name = d.get("name", f"cuda:{idx}")
        try:
            peak_alloc = torch.cuda.max_memory_allocated(idx)
        except Exception:
            peak_alloc = 0
        try:
            peak_reserved = torch.cuda.max_memory_reserved(idx)
        except Exception:
            peak_reserved = 0

        out["per_device"].append({
            "device": f"cuda:{idx}",
            "name": name,
            "peak_allocated_bytes": int(peak_alloc),
            "peak_reserved_bytes": int(peak_reserved),
            "peak_allocated_gib": _bytes_to_gib(int(peak_alloc)),
            "peak_reserved_gib": _bytes_to_gib(int(peak_reserved)),
        })
        total_alloc += int(peak_alloc)
        total_res += int(peak_reserved)

    out["total_peak_allocated_gib"] = _bytes_to_gib(total_alloc)
    out["total_peak_reserved_gib"] = _bytes_to_gib(total_res)
    return out
# ---- 显存统计辅助结束 ----

# ---------- 路径与输出组织 ----------
def _timestamp() -> str:
    return datetime.now().strftime(f"%Y%m%d-%H%M%S")

def _auto_run_dir(out_dir: Optional[str], detector_display_name: str) -> Path:
    """
    规则：
      - 未提供 out_dir:  runs_{detector}_{timestamp}/
      - 提供了 out_dir:  若最后一段目录名不含 8位日期-6位时间（\\d{8}-\\d{6}），自动追加 _{timestamp}
                         若已包含时间戳，保持不变。
    """
    ts = _timestamp()
    if out_dir:
        p = Path(out_dir)
        # 只检查最后一段目录名是否有时间戳
        tail = p.name
        if not re.search(r"\d{8}-\d{6}$", tail):
            p = p.with_name(f"{tail}_{ts}")
    else:
        p = Path(f"results/runs_{detector_display_name}_{ts}")
    # 创建标准子目录
    p.mkdir(parents=True, exist_ok=True)
    (p / "metrics/curves").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    (p / "artifacts").mkdir(parents=True, exist_ok=True)
    return p

# ---------- 统计工具 ----------
def _auto_prefix(dataset: Union[str, Iterable[Dict[str, Any]]],
                 detector_display_name: str,
                 model_name: Optional[str] = None) -> str:
    def _basename(p: str) -> str:
        p = p.rstrip("/").rstrip("\\")
        return os.path.basename(p) if p else "data"
    if isinstance(dataset, str):
        d = _basename(dataset.lower())
        for suf in (".jsonl", ".json"):
            if d.endswith(suf):
                d = d[: -len(suf)]
        d = d or "data"
    else:
        d = "data"
    if model_name and model_name != detector_display_name:
        return f"{d}__{detector_display_name}__{model_name}"
    return f"{d}__{detector_display_name}"


def _basic_stats(labels: Sequence[int], preds: Sequence[int], probs: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    lbls = list(int(x) for x in labels)
    prds = list(int(x) for x in preds)
    total = len(lbls)
    pos = sum(1 for x in lbls if x == 1)
    neg = total - pos
    tp = sum(1 for y, p in zip(lbls, prds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(lbls, prds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(lbls, prds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(lbls, prds) if y == 1 and p == 0)
    acc = (tp + tn) / total if total else 0.0
    out: Dict[str, Any] = {
        "num_samples": total,
        "num_pos": pos,
        "num_neg": neg,
        "acc": acc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    if probs is not None:
        prbs = list(float(x) for x in probs)
        if prbs:
            out.update({
                "prob_mean": sum(prbs) / len(prbs),
                "prob_min": min(prbs),
                "prob_max": max(prbs),
            })
    return out


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def _by_group(
    group_values: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    *,
    include_mask: Optional[Sequence[bool]] = None,
    exclude_values: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    gs = list(str(g) for g in group_values)
    ys = list(int(y) for y in labels)
    ps = list(int(p) for p in preds)
    mask = list(bool(m) for m in include_mask) if include_mask is not None else [True] * len(gs)
    excl = set(v.lower() for v in (exclude_values or []))
    out: Dict[str, Dict[str, Any]] = {}
    for g, y, p, m in zip(gs, ys, ps, mask):
        if not m:
            continue
        if g.lower() in excl:
            continue
        d = out.setdefault(g, {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
        d["n"] += 1
        if y == 1 and p == 1: d["tp"] += 1
        elif y == 0 and p == 0: d["tn"] += 1
        elif y == 0 and p == 1: d["fp"] += 1
        elif y == 1 and p == 0: d["fn"] += 1
    # finalize
    for g, d in out.items():
        tp, tn, fp, fn = d["tp"], d["tn"], d["fp"], d["fn"]
        n = d["n"]
        d["acc"] = (tp + tn) / n if n else 0.0
        d["f1"] = _f1_from_counts(tp, fp, fn)
    return out


# ---------- 曲线与指标 ----------
def _roc_curve(labels: Sequence[int], probs_or_scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    # 排序（降序）——注意这里接受“概率或分数”，只作为排序用
    pairs = sorted(zip(probs_or_scores, labels), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    N = len(pairs) - P
    tp = fp = 0
    tpr = [0.0]; fpr = [0.0]
    last_val = None
    auc = 0.0
    prev_fpr = 0.0; prev_tpr = 0.0
    for p, y in pairs:
        if last_val is None or p != last_val:
            # 梯形积分
            auc += (fpr[-1] - prev_fpr) * (tpr[-1] + prev_tpr) / 2.0
            prev_fpr, prev_tpr = fpr[-1], tpr[-1]
            last_val = p
        if y == 1: tp += 1
        else: fp += 1
        tpr_val = tp / P if P else 0.0
        fpr_val = fp / N if N else 0.0
        tpr.append(tpr_val); fpr.append(fpr_val)
    auc += (fpr[-1] - prev_fpr) * (tpr[-1] + prev_tpr) / 2.0
    return fpr, tpr, auc


def _pr_curve(labels: Sequence[int], probs_or_scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    # 按降序
    pairs = sorted(zip(probs_or_scores, labels), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    tp = fp = 0
    precision = []; recall = []
    for p, y in pairs:
        if y == 1: tp += 1
        else: fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / P if P else 0.0
        precision.append(prec); recall.append(rec)
    # AUPR（对 recall 积分）
    aupr = 0.0
    prev_r, prev_pv = 0.0, 1.0
    for r, pv in zip(recall, precision):
        aupr += (r - prev_r) * ((pv + prev_pv) / 2.0)
        prev_r, prev_pv = r, pv
    return recall, precision, aupr


def _ece_brier(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> Dict[str, Any]:
    ys = [int(y) for y in labels]
    ps = [float(p) for p in probs]
    # Brier
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(1, len(ys))
    # ECE
    bin_sums = [0.0] * bins
    bin_accs = [0.0] * bins
    bin_cnts = [0] * bins
    for y, p in zip(ys, ps):
        b = min(bins - 1, int(p * bins))
        bin_sums[b] += p
        bin_accs[b] += (1.0 if y == 1 else 0.0)
        bin_cnts[b] += 1
    ece = 0.0
    bins_info = []
    for b in range(bins):
        if bin_cnts[b] == 0:
            bins_info.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        conf = bin_sums[b] / bin_cnts[b]
        acc = bin_accs[b] / bin_cnts[b]
        gap = abs(conf - acc)
        ece += (bin_cnts[b] / len(ys)) * gap
        bins_info.append({"bin": b, "n": bin_cnts[b], "conf": conf, "acc": acc, "gap": gap})
    return {"brier": brier, "ece": ece, "bins": bins_info}


def _risk_coverage(labels: Sequence[int], probs: Sequence[float]) -> Tuple[List[float], List[float]]:
    """
    不同覆盖率下的风险（错误率）：
      - 置信度 conf = max(p, 1-p)
      - 以 conf 降序纳入样本；coverage=k/N；risk = 错误率（1-acc）在纳入的子集上
    """
    ys = [int(y) for y in labels]
    ps = [float(p) for p in probs]
    confs = [max(p, 1.0 - p) for p in ps]
    order = sorted(range(len(ys)), key=lambda i: -confs[i])

    cov = []
    risk = []
    correct = 0
    for k, i in enumerate(order, start=1):
        pred = 1 if ps[i] >= 0.5 else 0
        if pred == ys[i]:
            correct += 1
        cov.append(k / len(ys))
        acc = correct / k
        risk.append(1.0 - acc)
    return cov, risk


def _moving_avg(xs: List[float], k: int) -> List[float]:
    if k <= 1:
        return xs[:]
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in xs:
        s += v
        q.append(v)
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _plot_curve(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, out_png: Path):
    if not _HAS_MPL:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_reliability(bins_info: List[Dict[str, Any]], out_png: Path):
    if not _HAS_MPL:
        return
    xs = []
    ys = []
    for b in bins_info:
        if b.get("conf") is None or b.get("acc") is None:
            continue
        xs.append(float(b["conf"]))
        ys.append(float(b["acc"]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.scatter(xs, ys, s=30)
    ax.set_title("Calibration (Reliability Diagram)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)

# ---------- 新增：排序向量选择器（优先 scores） ----------
def _get_ranking_values(res) -> Tuple[List[float], str]:
    """
    返回用于 ROC/PR 排序的一维向量：
      - 优先使用 res.scores（原始分数，越大越像 AI 的约定留给检测器保持一致）
      - 否则回退到 res.probs
    同时返回来源标记：'scores' / 'probs' / 'none'
    """
    scores = getattr(res, "scores", None)
    if scores is not None:
        try:
            return [float(x) for x in scores], "scores"
        except Exception:
            pass
    probs = getattr(res, "probs", None)
    if probs is not None:
        return [float(x) for x in probs], "probs"
    return [], "none"

# ---------- Bootstrap CI ----------
def _bootstrap_ci(
    labels: Sequence[int],
    probs: Sequence[float],
    *,
    iters: int = 500,
    seed: int = 114514,
) -> Dict[str, Any]:
    import random
    ys = list(int(y) for y in labels)
    ps = list(float(p) for p in probs)
    n = len(ys)
    rng = random.Random(seed)
    accs = []
    aurocs = []
    auprs = []
    f1s = []
    for _ in range(max(1, iters)):
        idxs = [rng.randrange(n) for _ in range(n)]
        yb = [ys[i] for i in idxs]
        pb = [ps[i] for i in idxs]
        preds = [1 if p >= 0.5 else 0 for p in pb]
        # 基础指标
        st = _basic_stats(yb, preds, pb)
        accs.append(st["acc"])
        tp = st["confusion"]["tp"]; fp = st["confusion"]["fp"]; fn = st["confusion"]["fn"]
        f1s.append(_f1_from_counts(tp, fp, fn))
        # 曲线
        fpr, tpr, auroc = _roc_curve(yb, pb)
        aurocs.append(auroc)
        r, p, aupr = _pr_curve(yb, pb)
        auprs.append(aupr)
    def _pct(a, q):
        k = int(round((q/100.0) * (len(a)-1)))
        return float(sorted(a)[max(0, min(len(a)-1, k))])
    return {
        "acc": {"mean": float(sum(accs)/len(accs)), "lo": _pct(accs, 2.5), "hi": _pct(accs, 97.5)},
        "auroc": {"mean": float(sum(aurocs)/len(aurocs)), "lo": _pct(aurocs, 2.5), "hi": _pct(aurocs, 97.5)},
        "aupr": {"mean": float(sum(auprs)/len(auprs)), "lo": _pct(auprs, 2.5), "hi": _pct(auprs, 97.5)},
        "f1": {"mean": float(sum(f1s)/len(f1s)), "lo": _pct(f1s, 2.5), "hi": _pct(f1s, 97.5)},
        "iters": iters,
        "seed": seed,
    }

# ---------- 运行时信息与清单 ----------
def _env_fingerprint() -> Dict[str, Any]:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import transformers as _tf
        info["transformers"] = _tf.__version__
    except Exception:
        pass
    try:
        import torch as _t
        info["torch"] = _t.__version__
        info["cuda_available"] = bool(_t.cuda.is_available())
        if _t.cuda.is_available():
            info["cuda_device_count"] = _t.cuda.device_count()
            info["cuda_devices"] = [ _t.cuda.get_device_name(i) for i in range(_t.cuda.device_count()) ]
    except Exception:
        info["torch"] = None
    return info


def _proc_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    if _HAS_PSUTIL:
        p = psutil.Process()
        with p.oneshot():
            mem = p.memory_info()
            snap["rss_bytes"] = int(getattr(mem, "rss", 0))
            snap["vms_bytes"] = int(getattr(mem, "vms", 0))
            try:
                snap["cpu_percent"] = psutil.cpu_percent(interval=0.05)
            except Exception:
                snap["cpu_percent"] = None
    return snap


def evaluate_detector(
    detector: Union[str, DetectorBase],
    dataset: Union[str, Iterable[Dict[str, Any]]],
    batch_size: int = 8,
    threshold: float = 0.5,
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    # 分组列（不指定则自动探测存在的列，如 lang/source/model/sub_source）
    group_cols: Optional[Sequence[str]] = None,
    # 输出目录：不指定则 runs_{det_name}_{timestamp}/
    out_dir: Optional[str] = None,
    out_prefix: Optional[str] = None,   # 不再用作文件前缀，仅保留兼容
    save_curves: bool = True,           # 兼容开关（图像/JSON曲线默认开启；若 False 仅写 metrics/json）
    # CI 配置
    ci_enable: Optional[bool] = None,   # None: 自动(仅采样时做)；True/False: 强制
    ci_iters: int = 200,
    ci_seed: int = 114514,
    show_progress: bool = True,      # NEW: 评测时是否显示进度条
    # --- NEW: 多次随机采样重复实验 ---
    k_runs: int = 1,                 # 当 sample_k>0 时有效；执行 k 次不同随机种子的采样评测并统计
    # 其他 detector 参数
    **detector_kwargs,
) -> EvalResult:
    """
    新增功能：
      - k_runs：当 sample_k>0（采样评测）且 k_runs>1 时，使用围绕 sample_seed 的 ±10 种子序列
        （长度不足则循环取用）重复评测，并将跨次统计的 mean/std/min/max 等写入 metrics/summary.json。
      - 全量评测（sample_k 不指定或 <=0）时，k_runs 无效，行为与旧版一致。
    """
    # 0) 初始化 detector
    if isinstance(detector, str):
        Det = get_detector_cls(detector)
        det = Det(**detector_kwargs)
    else:
        det = detector
        for k, v in detector_kwargs.items():
            setattr(det, k, v)

    display_name = getattr(det, "name", getattr(det, "DETECTOR_NAME", "detector"))

    # ----------- 工具：统计聚合 -----------
    def _stat_pack(vals: List[float]) -> Dict[str, Any]:
        vals = [float(x) for x in vals if x is not None]
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        m = sum(vals) / n
        if n > 1:
            var = sum((x - m) ** 2 for x in vals) / (n - 1)  # 样本方差
            sd = math.sqrt(var)
        else:
            sd = 0.0
        return {"n": n, "mean": m, "std": sd, "min": min(vals), "max": max(vals)}

    # ----------- 采样多次运行的分支判定 -----------
    is_sampling = (sample_k is not None and sample_k > 0)
    multi_run = (is_sampling and (k_runs is not None) and (int(k_runs) > 1))

    # 提前创建 run_dir 与 run-config
    run_dir = _auto_run_dir(out_dir, display_name)
    run_cfg = {
        "detector": display_name,
        "detector_type": getattr(det, "detector_type", "Unknown"),
        "args": {
            "batch_size": batch_size,
            "threshold": threshold,
            "sample_k": sample_k,
            "sample_seed": sample_seed,
            "group_cols": list(group_cols) if group_cols is not None else None,
            "save_curves": bool(save_curves),
            "ci_enable": ci_enable,
            "ci_iters": int(ci_iters),
            "ci_seed": int(ci_seed),
            "k_runs": int(k_runs),
        },
        "dataset_hint": str(dataset) if isinstance(dataset, str) else "iterable",
    }
    cfg_path = run_dir / "run-config.yaml"
    try:
        if _HAS_YAML:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(run_cfg, f, allow_unicode=True, sort_keys=False)
        else:
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(run_cfg, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # =============== 多次随机采样评测 ===============
    if multi_run:
        # 组装种子序列：围绕 sample_seed 的 ±10 共 21 个偏移，长度不足则循环
        base = int(sample_seed)
        offsets = list(range(-10, 11))  # [-10, ..., 0, ..., +10]
        k = int(k_runs)
        seeds = [base + offsets[i % len(offsets)] for i in range(k)]

        # 资源/目录
        metrics_dir = run_dir / "metrics"
        curves_dir = metrics_dir / "curves"
        figures_dir = run_dir / "figures"
        k_runs_dir = metrics_dir / "k_runs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        k_runs_dir.mkdir(parents=True, exist_ok=True)

        # 写 seeds
        with open(k_runs_dir / "seeds.json", "w", encoding="utf-8") as f:
            json.dump({"base": base, "seeds": seeds}, f, ensure_ascii=False, indent=2)

        # 聚合容器
        acc_list, auroc_list, aupr_list, f1_list = [], [], [], []
        ece_list, brier_list = [], []
        eval_secs, load_secs = [], []
        per_run_briefs = []

        # 保存“首轮”的详细制品（曲线/图像/预测等），便于复现与可视化
        first_res: Optional[EvalResult] = None
        first_examples: Optional[List[Dict[str, Any]]] = None
        first_used_group_cols: List[str] = []
        first_probs_seq: Optional[List[float]] = None
        first_ranking_vec: Optional[List[float]] = None
        first_ranking_src: str = "none"

        # 执行 k 次
        for ridx, seed in enumerate(seeds):
            # 1) 负载采样
            t0 = time.perf_counter()
            proc0 = _proc_snapshot()
            examples, used_group_cols = load_dataset_unified(
                dataset=dataset,
                sample_k=sample_k,
                sample_seed=int(seed),
                group_cols=group_cols,
            )
            load_time = time.perf_counter() - t0

            # 2) 显存峰值统计（每次 run 独立重置）
            cuda_ctx = _reset_and_mark_cuda_peaks()

            # 3) 评测
            t1 = time.perf_counter()
            res_i = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
            eval_time = time.perf_counter() - t1

            # 4) 指标计算（总体）
            labels_seq = list(res_i.labels)
            preds_seq  = list(res_i.preds)
            probs_seq  = (list(res_i.probs) if getattr(res_i, "probs", None) is not None else None)
            ranking_vec, ranking_src = _get_ranking_values(res_i)

            examples, labels_seq, preds_seq, ranking_vec, probs_seq, dropped = _filter_nonfinite_examples(
                examples, labels_seq, preds_seq, ranking_vec=ranking_vec, probs=probs_seq
            )
            if dropped > 0:
                print(f"[MGTEval] evaluator(run {ridx}): dropped {dropped} non-finite samples before curves/plots.")
            if len(labels_seq) == 0:
                raise RuntimeError("Empty set after dropping non-finite samples in run; aborting.")

            overall_basic = _basic_stats(labels_seq, preds_seq, probs_seq)

            # ROC / PR 使用 ranking_vec
            fpr, tpr, auroc = _roc_curve(labels_seq, ranking_vec)
            rec, prec, aupr = _pr_curve(labels_seq, ranking_vec)

            # 只有存在概率时才做校准与 RC
            if probs_seq is not None:
                calib = _ece_brier(labels_seq, probs_seq, bins=10)
                cov, risk = _risk_coverage(labels_seq, probs_seq)
            else:
                calib = {"ece": None, "brier": None, "bins": []}
                cov, risk = ([], [])

            # 记录聚合
            acc_list.append(overall_basic["acc"])
            tp = overall_basic["confusion"]["tp"]; fp = overall_basic["confusion"]["fp"]; fn = overall_basic["confusion"]["fn"]
            f1_list.append(_f1_from_counts(tp, fp, fn))
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            ece_list.append(calib["ece"])
            brier_list.append(calib["brier"])
            load_secs.append(load_time)
            eval_secs.append(eval_time)

            # 单次简报
            brief = {
                "run_index": ridx,
                "seed": int(seed),
                "counts": overall_basic,
                "metrics": {
                    "auroc": auroc,
                    "aupr": aupr,
                    "ece": calib["ece"],
                    "brier": calib["brier"],
                    "rank_source": ranking_src,
                },
                "timing_sec": {
                    "dataset_load": load_time,
                    "evaluate": eval_time,
                    "throughput_eps": len(examples) / eval_time if eval_time > 0 else None,
                    "latency_ms_per_sample": (eval_time / len(examples) * 1000.0) if len(examples) else None,
                },
                "n_samples": len(examples),
            }
            per_run_briefs.append(brief)
            with open(k_runs_dir / f"run_{ridx:02d}.json", "w", encoding="utf-8") as f:
                json.dump(brief, f, ensure_ascii=False, indent=2)

            # 保存第一轮的制品（曲线/图像/预测），其余轮只做统计（避免大量重复文件）
            if ridx == 0:
                first_res = res_i
                first_examples = examples
                first_used_group_cols = list(used_group_cols)
                first_probs_seq = probs_seq
                first_ranking_vec = ranking_vec
                first_ranking_src = ranking_src

                # 显存/进程统计
                mem_stats = _collect_cuda_peaks(cuda_ctx)
                proc1 = _proc_snapshot()
                env = _env_fingerprint()
                manifest = {
                    "env": env,
                    "resources": {
                        "gpu_memory": mem_stats,
                        "process_before": proc0,
                        "process_after": proc1,
                    },
                    "timing": {
                        "dataset_load_sec": load_time,
                        "evaluate_sec": eval_time,
                        "throughput_eps": len(examples) / eval_time if eval_time > 0 else None,
                        "latency_ms_per_sample": (eval_time / max(1, len(examples)) * 1000.0) if len(examples) else None,
                    },
                    "detector": {
                        "name": display_name,
                        "type": getattr(det, "detector_type", "Unknown"),
                    },
                    "dataset": {
                        "size": len(examples),
                        "group_cols": list(used_group_cols),
                    },
                    "notes": "Auto-generated run manifest for auditability and reproducibility. (first run artifacts)",
                }
                with open(run_dir / "run-manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                # ★★★ 新增：首轮 meta 增强 + 落盘（供审计）
                meta_first = None
                try:
                    if hasattr(first_res, "meta") and isinstance(first_res.meta, dict):
                        meta_first = dict(first_res.meta)  # 浅拷贝
                        # 与单次分支保持一致：补齐 timing/memory（便于独立查看）
                        meta_first.setdefault("memory", mem_stats)
                        meta_first.setdefault("timing", {})
                        meta_first["timing"].update({
                            "dataset_load_sec": load_time,
                            "evaluate_sec": eval_time,
                            "throughput_eps": len(examples) / eval_time if eval_time > 0 else None,
                            "latency_ms_per_sample": (eval_time / max(1, len(examples)) * 1000.0) if len(examples) else None,
                        })
                        with open(run_dir / "meta_first_run.json", "w", encoding="utf-8") as f:
                            json.dump(meta_first, f, ensure_ascii=False, indent=2)
                except Exception:
                    meta_first = None
                # —— 第一轮：分组文件 + 曲线 + 预测 + 模型卡（保持与原版一致，适配 ranking/probs）——
                labels_seq_1 = res_i.labels
                preds_seq_1 = res_i.preds

                # per_lang
                if "lang" in used_group_cols:
                    col_vals = [str(ex.get("lang", "unknown")) for ex in examples]
                    g = _by_group(col_vals, labels_seq_1, preds_seq_1)
                    with open(metrics_dir / "per_lang.json", "w", encoding="utf-8") as f:
                        json.dump(g, f, ensure_ascii=False, indent=2)

                    if save_curves:
                        langs = sorted(set(col_vals))
                        for lg in langs:
                            idxs = [i for i, v in enumerate(col_vals) if v == lg]
                            if not idxs:
                                continue
                            yl = [labels_seq_1[i] for i in idxs]
                            # ranking 子集
                            rv = [first_ranking_vec[i] for i in idxs]
                            fpr_l, tpr_l, auroc_l = _roc_curve(yl, rv)
                            rec_l, prec_l, aupr_l = _pr_curve(yl, rv)
                            with open(curves_dir / f"roc_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"fpr": fpr_l, "tpr": tpr_l, "auroc": auroc_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            with open(curves_dir / f"pr_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"recall": rec_l, "precision": prec_l, "aupr": aupr_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            _plot_curve(fpr_l, tpr_l, f"ROC ({lg}) AUC={auroc_l:.3f}", "FPR", "TPR", figures_dir / f"roc_{lg}.png")
                            _plot_curve(rec_l, prec_l, f"PR ({lg}) AUPR={aupr_l:.3f}", "Recall", "Precision", figures_dir / f"pr_{lg}.png")
                        # RC / Reliability 只在有概率时
                        if first_probs_seq is not None:
                            cov_lg, risk_lg = _risk_coverage(labels_seq_1, first_probs_seq)
                            with open(curves_dir / f"rc_abstain_overall.json", "w", encoding="utf-8") as f:
                                json.dump({"coverage": cov_lg, "risk": risk_lg}, f, ensure_ascii=False, indent=2)
                            _plot_curve(cov_lg, risk_lg, f"Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / f"rc_overall.png")

                # per_domain
                domain_key = "source" if "source" in used_group_cols else ("sub_source" if "sub_source" in used_group_cols else None)
                if domain_key:
                    col_vals = [str(ex.get(domain_key, "unknown")) for ex in examples]
                    g = _by_group(col_vals, labels_seq_1, preds_seq_1)
                    with open(metrics_dir / "per_domain.json", "w", encoding="utf-8") as f:
                        json.dump(g, f, ensure_ascii=False, indent=2)

                # per_model
                if "model" in used_group_cols:
                    col_vals = [str(ex.get("model", "unknown")) for ex in examples]
                    g = _by_group(col_vals, labels_seq_1, preds_seq_1)
                    with open(metrics_dir / "per_model.json", "w", encoding="utf-8") as f:
                        json.dump(g, f, ensure_ascii=False, indent=2)

                # per_length
                lengths = [_word_count(ex.get("text", "")) for ex in examples]
                if lengths:
                    bins = []
                    for L in lengths:
                        if L <= 50: bins.append("0-50")
                        elif L <= 100: bins.append("50-100")
                        elif L <= 200: bins.append("100-200")
                        elif L <= 300: bins.append("200-300")
                        elif L <= 400: bins.append("300-400")
                        elif L <= 500: bins.append("400-500")
                        else: bins.append(">500")
                    g = _by_group(bins, labels_seq_1, preds_seq_1)
                    with open(metrics_dir / "per_length.json", "w", encoding="utf-8") as f:
                        json.dump(g, f, ensure_ascii=False, indent=2)

                # overall 曲线与图像（ROC/PR 用 ranking，RC/Calibration 仅在 probs 存在）
                if save_curves:
                    fpr_o, tpr_o, auroc_o = _roc_curve(labels_seq_1, first_ranking_vec)
                    rec_o, prec_o, aupr_o = _pr_curve(labels_seq_1, first_ranking_vec)
                    with open(curves_dir / f"roc_overall.json", "w", encoding="utf-8") as f:
                        json.dump({"fpr": fpr_o, "tpr": tpr_o, "auroc": auroc_o, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                    with open(curves_dir / f"pr_overall.json", "w", encoding="utf-8") as f:
                        json.dump({"recall": rec_o, "precision": prec_o, "aupr": aupr_o, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                    _plot_curve(fpr_o, tpr_o, f"ROC (overall) AUC={auroc_o:.3f}", "FPR", "TPR", figures_dir / "roc_overall.png")
                    _plot_curve(rec_o, prec_o, f"PR (overall) AUPR={aupr_o:.3f}", "Recall", "Precision", figures_dir / "pr_overall.png")

                    if first_probs_seq is not None:
                        calib_o = _ece_brier(labels_seq_1, first_probs_seq, bins=10)
                        cov_o, risk_o = _risk_coverage(labels_seq_1, first_probs_seq)
                        with open(curves_dir / f"rc_abstain_overall.json", "w", encoding="utf-8") as f:
                            json.dump({"coverage": cov_o, "risk": risk_o}, f, ensure_ascii=False, indent=2)
                        _plot_curve(cov_o, risk_o, "Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / "rc_overall.png")
                        _plot_reliability(calib_o["bins"], figures_dir / "calibration_overall.png")

                # predictions（仅首轮，text 截断为前 10 个词以减小体积）
                preds_out = []
                for i, ex in enumerate(examples):
                    raw_text = str(ex.get("text", "") or "").strip()
                    words = raw_text.split()
                    text_preview = " ".join(words[:10]) if words else ""
                    rec_pred = {
                        "text": text_preview,
                        "label": int(res_i.labels[i]),
                        "prob": (float(first_probs_seq[i]) if first_probs_seq is not None else None),
                        "score": (float(first_ranking_vec[i]) if first_ranking_src == "scores" else None),
                        "pred": int(res_i.preds[i]),
                        "length": len(words),
                    }
                    for gc in used_group_cols:
                        rec_pred[gc] = ex.get(gc, None)
                    rec_pred["id"] = ex.get("id", i)
                    preds_out.append(rec_pred)
                with open(run_dir / "predictions.json", "w", encoding="utf-8") as f:
                    json.dump(preds_out, f, ensure_ascii=False, indent=2)

                # 模型卡
                card = {
                    "detector_name": display_name,
                    "detector_type": getattr(det, "detector_type", "Unknown"),
                }
                try:
                    if hasattr(det, "scoring_model_name"): card["scoring_model_name"] = getattr(det, "scoring_model_name")
                    if hasattr(det, "sampling_model_name"): card["sampling_model_name"] = getattr(det, "sampling_model_name")
                    if hasattr(det, "tokenizer_name"): card["tokenizer_name"] = getattr(det, "tokenizer_name")
                    if hasattr(det, "model_path"): card["model_path"] = getattr(det, "model_path")
                    if hasattr(det, "tokenizer_path"): card["tokenizer_path"] = getattr(det, "tokenizer_path")
                except Exception:
                    pass
                with open(run_dir / "artifacts" / "model_card.json", "w", encoding="utf-8") as f:
                    json.dump(card, f, ensure_ascii=False, indent=2)

        # —— 跨次统计与 summary.json —— #
        # 是否做 bootstrap CI：仅在有概率时
        do_ci = (ci_enable if ci_enable is not None else True)  # 在多次采样场景，默认做一次首轮 CI
        ci = None
        if do_ci and first_res is not None and hasattr(first_res, "probs") and (first_res.probs is not None) and len(first_res.labels) > 5:
            ci = _bootstrap_ci(first_res.labels, first_res.probs, iters=int(ci_iters), seed=int(ci_seed))

        # 汇总统计
        summary = {
            "threshold": threshold,
            "detector": display_name,
            "k_runs": int(k_runs),
            "k_runs_seeds": seeds,
            "counts_first_run": _basic_stats(first_res.labels, first_res.preds, getattr(first_res, "probs", None)) if first_res else None,
            "metrics_first_run": {
                "auroc": auroc_list[0] if auroc_list else None,
                "aupr": aupr_list[0] if aupr_list else None,
                "ece": ece_list[0] if ece_list else None,
                "brier": brier_list[0] if brier_list else None,
                "rank_source": first_ranking_src,
            },
            "k_runs_stats": {
                "acc": _stat_pack(acc_list),
                "f1": _stat_pack(f1_list),
                "auroc": _stat_pack(auroc_list),
                "aupr": _stat_pack(aupr_list),
                "ece": _stat_pack(ece_list),
                "brier": _stat_pack(brier_list),
                "dataset_load_sec": _stat_pack(load_secs),
                "evaluate_sec": _stat_pack(eval_secs),
            },
            "ci_95_first_run": ci,
            "meta_first_run": meta_first,
        }
        with open(run_dir / "metrics" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[MGTEval] (multi-run) results saved to: {str(run_dir)}")
        return first_res  # 返回首轮 EvalResult，便于向后兼容

    # =============== 单次评测（原逻辑，适配 ranking/probs） ===============
    # 1) 统一加载数据（并确定分组列）
    t0 = time.perf_counter()
    proc0 = _proc_snapshot()
    examples, used_group_cols = load_dataset_unified(
        dataset=dataset,
        sample_k=sample_k,
        sample_seed=sample_seed,
        group_cols=group_cols,
    )
    load_time = time.perf_counter() - t0

    # 2) 写回配置（单次）
    run_cfg["args"]["group_cols"] = list(used_group_cols)
    try:
        if _HAS_YAML:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(run_cfg, f, allow_unicode=True, sort_keys=False)
        else:
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(run_cfg, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # 3) 显存峰值统计：评测前 reset
    cuda_ctx = _reset_and_mark_cuda_peaks()

    # 4) 评测（顺序一致）
    t1 = time.perf_counter()
    res = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    eval_time = time.perf_counter() - t1

    # 5) 运行资源统计
    mem_stats = _collect_cuda_peaks(cuda_ctx)
    proc1 = _proc_snapshot()

    # 6) 写回 meta
    '''
    try:
        if hasattr(res, "meta") and isinstance(res.meta, dict):
            res.meta.setdefault("memory", mem_stats)
            res.meta.setdefault("timing", {})
            res.meta["timing"].update({
                "dataset_load_sec": load_time,
                "evaluate_sec": eval_time,
                "throughput_eps": len(examples) / eval_time if eval_time > 0 else None,
                "latency_ms_per_sample": (eval_time / len(examples) * 1000.0) if len(examples) else None,
            })
    except Exception:
        pass
    '''
    labels_seq = list(res.labels)
    preds_seq = list(res.preds)
    probs_seq = (list(res.probs) if getattr(res, "probs", None) is not None else None)
    ranking_vec, ranking_src = _get_ranking_values(res)

    # ★ 统一过滤非有限值（NaN/Inf/None）
    examples, labels_seq, preds_seq, ranking_vec, probs_seq, dropped = _filter_nonfinite_examples(
        examples, labels_seq, preds_seq, ranking_vec=ranking_vec, probs=probs_seq
    )
    if dropped > 0:
        print(f"[MGTEval] evaluator: dropped {dropped} non-finite samples before curves/plots.")
    if len(labels_seq) == 0:
        raise RuntimeError("Empty set after dropping non-finite samples; aborting evaluation.")

    by_groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for col in used_group_cols:
        col_vals = [str(ex.get(col, "unknown")) for ex in examples]
        if col.lower() == "model":
            mask = [int(y) == 1 for y in labels_seq]
            by_groups[col] = _by_group(col_vals, labels_seq, preds_seq, include_mask=mask, exclude_values=["human"])
        else:
            by_groups[col] = _by_group(col_vals, labels_seq, preds_seq)

    # 8) 其它工程化清单（run-manifest）
    env = _env_fingerprint()
    manifest = {
        "env": env,
        "resources": {
            "gpu_memory": mem_stats,
            "process_before": proc0,
            "process_after": proc1,
        },
        "timing": {
            "dataset_load_sec": load_time,
            "evaluate_sec": eval_time,
            "throughput_eps": len(examples) / eval_time if eval_time > 0 else None,
            "latency_ms_per_sample": (eval_time / max(1, len(examples)) * 1000.0) if len(examples) else None,
        },
        "detector": {
            "name": display_name,
            "type": getattr(det, "detector_type", "Unknown"),
        },
        "dataset": {
            "size": len(examples),
            "group_cols": list(used_group_cols),
        },
        "notes": "Auto-generated run manifest for auditability and reproducibility.",
    }
    with open(run_dir / "run-manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 9) 聚合指标、CI、校准与曲线
    # 基础总体
    overall_basic = _basic_stats(labels_seq, preds_seq, probs_seq)
    # ROC/PR 使用 ranking_vec
    fpr, tpr, auroc = _roc_curve(labels_seq, ranking_vec)
    rec, prec, aupr = _pr_curve(labels_seq, ranking_vec)
    # Calibration (ECE/Brier)：仅在有概率时
    if probs_seq is not None:
        calib = _ece_brier(labels_seq, probs_seq, bins=10)
        cov, risk = _risk_coverage(labels_seq, probs_seq)
    else:
        calib = {"ece": None, "brier": None, "bins": []}
        cov, risk = ([], [])

    # CI：默认仅在“采样测试”时开启；且仅当有概率
    do_ci = (ci_enable if ci_enable is not None else (sample_k is not None and sample_k > 0))
    ci = None
    if do_ci and (probs_seq is not None) and len(labels_seq) > 5:
        ci = _bootstrap_ci(labels_seq, probs_seq, iters=int(ci_iters), seed=int(ci_seed))

    # 10) 保存 metrics 概要（metrics/summary.json）
    metrics_dir = run_dir / "metrics"
    curves_dir = metrics_dir / "curves"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 总体概要（单次）
    summary = {
        "threshold": threshold,
        "detector": display_name,
        "counts": overall_basic,
        "metrics": {
            "auroc": auroc,
            "aupr": aupr,
            "ece": calib["ece"],
            "brier": calib["brier"],
            "rank_source": ranking_src,
        },
        "ci_95": ci,  # 可能为 None
        "timing": manifest["timing"],
        "memory": mem_stats,
        "k_runs": 1,                # 明确标注单次
        "k_runs_stats": None,       # 单次无跨次统计
        "k_runs_seeds": [sample_seed] if is_sampling else None,
        "meta": (res.meta if hasattr(res, "meta") else None),
    }
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 11) 分组指标文件
    if "lang" in used_group_cols:
        col_vals = [str(ex.get("lang", "unknown")) for ex in examples]
        g = _by_group(col_vals, labels_seq, preds_seq)
        with open(metrics_dir / "per_lang.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

        if save_curves:
            langs = sorted(set(col_vals))
            for lg in langs:
                idxs = [i for i, v in enumerate(col_vals) if v == lg]
                if not idxs:
                    continue
                yl = [labels_seq[i] for i in idxs]
                rv = [ranking_vec[i] for i in idxs]
                fpr_l, tpr_l, auroc_l = _roc_curve(yl, rv)
                rec_l, prec_l, aupr_l = _pr_curve(yl, rv)
                with open(curves_dir / f"roc_{lg}.json", "w", encoding="utf-8") as f:
                    json.dump({"fpr": fpr_l, "tpr": tpr_l, "auroc": auroc_l, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
                with open(curves_dir / f"pr_{lg}.json", "w", encoding="utf-8") as f:
                    json.dump({"recall": rec_l, "precision": prec_l, "aupr": aupr_l, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
                _plot_curve(fpr_l, tpr_l, f"ROC ({lg}) AUC={auroc_l:.3f}", "FPR", "TPR", figures_dir / f"roc_{lg}.png")
                _plot_curve(rec_l, prec_l, f"PR ({lg}) AUPR={aupr_l:.3f}", "Recall", "Precision", figures_dir / f"pr_{lg}.png")

            # RC / Reliability 只在有概率时
            if probs_seq is not None:
                cov_l, risk_l = _risk_coverage(labels_seq, probs_seq)
                with open(curves_dir / f"rc_abstain_overall.json", "w", encoding="utf-8") as f:
                    json.dump({"coverage": cov_l, "risk": risk_l}, f, ensure_ascii=False, indent=2)
                _plot_curve(cov_l, risk_l, f"Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / f"rc_overall.png")
                _plot_reliability(calib["bins"], figures_dir / "calibration_overall.png")

    domain_key = "source" if "source" in used_group_cols else ("sub_source" if "sub_source" in used_group_cols else None)
    if domain_key:
        col_vals = [str(ex.get(domain_key, "unknown")) for ex in examples]
        g = _by_group(col_vals, labels_seq, preds_seq)
        with open(metrics_dir / "per_domain.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

    if "model" in used_group_cols:
        col_vals = [str(ex.get("model", "unknown")) for ex in examples]
        g = _by_group(col_vals, labels_seq, preds_seq)
        with open(metrics_dir / "per_model.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

    lengths = [_word_count(ex.get("text", "")) for ex in examples]
    if lengths:
        bins = []
        for L in lengths:
            if L <= 50: bins.append("0-50")
            elif L <= 100: bins.append("50-100")
            elif L <= 200: bins.append("100-200")
            elif L <= 300: bins.append("200-300")
            elif L <= 400: bins.append("300-400")
            elif L <= 500: bins.append("400-500")
            else: bins.append(">500")
        g = _by_group(bins, labels_seq, preds_seq)
        with open(metrics_dir / "per_length.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

    # 12) 总体曲线与图像
    if save_curves:
        with open(curves_dir / f"roc_overall.json", "w", encoding="utf-8") as f:
            json.dump({"fpr": fpr, "tpr": tpr, "auroc": auroc, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
        with open(curves_dir / f"pr_overall.json", "w", encoding="utf-8") as f:
            json.dump({"recall": rec, "precision": prec, "aupr": aupr, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
        _plot_curve(fpr, tpr, f"ROC (overall) AUC={auroc:.3f}", "FPR", "TPR", figures_dir / "roc_overall.png")
        _plot_curve(rec, prec, f"PR (overall) AUPR={aupr:.3f}", "Recall", "Precision", figures_dir / "pr_overall.png")

        if probs_seq is not None:
            with open(curves_dir / f"rc_abstain_overall.json", "w", encoding="utf-8") as f:
                json.dump({"coverage": cov, "risk": risk}, f, ensure_ascii=False, indent=2)
            _plot_curve(cov, risk, "Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / "rc_overall.png")
            _plot_reliability(calib["bins"], figures_dir / "calibration_overall.png")

    # 13) 导出全量用例预测（predictions.json）
    preds_out = []
    for i, ex in enumerate(examples):
        raw_text = str(ex.get("text", "") or "").strip()
        words = raw_text.split()  # 基于空白分词
        text_preview = " ".join(words[:10]) if words else ""
        rec = {
            "text": text_preview,                  # 仅保存前 10 个单词
            "label": int(labels_seq[i]),
            "prob": (float(probs_seq[i]) if probs_seq is not None else None),
            "score": (float(ranking_vec[i]) if ranking_src == "scores" else None),
            "pred": int(preds_seq[i]),
        }
        for gc in used_group_cols:
            rec[gc] = ex.get(gc, None)
        rec["id"] = ex.get("id", i)
        rec["length"] = len(words)
        preds_out.append(rec)
    with open(run_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)

    # 14) 可选模型卡（若 detector 暴露信息）
    card = {
        "detector_name": display_name,
        "detector_type": getattr(det, "detector_type", "Unknown"),
    }
    try:
        if hasattr(det, "scoring_model_name"): card["scoring_model_name"] = getattr(det, "scoring_model_name")
        if hasattr(det, "sampling_model_name"): card["sampling_model_name"] = getattr(det, "sampling_model_name")
        if hasattr(det, "tokenizer_name"): card["tokenizer_name"] = getattr(det, "tokenizer_name")
        if hasattr(det, "model_path"): card["model_path"] = getattr(det, "model_path")
        if hasattr(det, "tokenizer_path"): card["tokenizer_path"] = getattr(det, "tokenizer_path")
    except Exception:
        pass
    with open(run_dir / "artifacts" / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    # 15) 兼容旧打印：简单提示
    print(f"[MGTEval] results saved to: {str(run_dir)}")
    return res
