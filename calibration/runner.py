# mgt_eval/calibration/runner.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import os, json, math, inspect
import numpy as np
from tqdm.auto import tqdm

from mgt_eval.detectors.registry import get_detector_cls
from mgt_eval.data_utils.load import load_dataset_unified
from .registry import get_calibrator  # 可选：支持将来扩展的校准器

# ================= 小工具：指标 =================
def _confusion_and_basic_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5):
    y = np.asarray(y).astype(int).reshape(-1)
    p = np.asarray(p).astype(float).reshape(-1)
    pred = (p >= thr).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    n = max(1, len(y))
    acc = (tp + tn) / n
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "num_samples": int(n),
        "num_pos": int(np.sum(y == 1)),
        "num_neg": int(np.sum(y == 0)),
    }

def _roc_auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    ys = np.asarray(y).astype(int)
    ss = np.asarray(s).astype(float)
    order = np.argsort(-ss)
    ys = ys[order]
    P = int(np.sum(ys == 1))
    N = len(ys) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    last = None
    for i in range(len(ys)):
        if (last is None) or (ss[order[i]] != last):
            auc += (fp / N - prev_fpr) * (tp / P + prev_tpr) / 2.0
            prev_fpr, prev_tpr = (fp / N), (tp / P)
            last = ss[order[i]]
        if ys[i] == 1: tp += 1
        else: fp += 1
    auc += (fp / N - prev_fpr) * (tp / P + prev_tpr) / 2.0
    return float(auc)

def _pr_auc_from_probs(y: np.ndarray, p: np.ndarray) -> float:
    ys = np.asarray(y).astype(int)
    ps = np.asarray(p).astype(float)
    order = np.argsort(-ps)
    ys = ys[order]
    tp = fp = 0
    P = int(np.sum(ys == 1))
    if P == 0:
        return 0.0
    aupr = 0.0
    prev_r, prev_prec = 0.0, 1.0
    for i in range(len(ys)):
        if ys[i] == 1: tp += 1
        else: fp += 1
        r = tp / P
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        aupr += (r - prev_r) * ((prec + prev_prec) / 2.0)
        prev_r, prev_prec = r, prec
    return float(aupr)

def _scan_tpr_at_fpr(
    y: np.ndarray,
    p: np.ndarray,
    fpr_levels: List[float],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    对校准后的概率 p 扫描阈值，得到在 FPR < target 时 TPR 最大的阈值。
    这里 FPR / TPR 的定义与二分类常规定义一致。
    返回结构：
    {
      "1":   {"threshold": float or None, "tpr": float or None, "fpr": float or None},
      "0.1": {...},
      ...
    }
    """
    y = np.asarray(y).astype(int).reshape(-1)
    p = np.asarray(p).astype(float).reshape(-1)
    assert y.shape[0] == p.shape[0]
    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))

    res: Dict[str, Dict[str, Optional[float]]] = {}
    # 没有正样本或负样本时无法定义 TPR / FPR
    if P == 0 or N == 0:
        for lvl in fpr_levels:
            res[f"{lvl:g}"] = {"threshold": None, "tpr": None, "fpr": None}
        return res

    order = np.argsort(-p)
    p_sorted = p[order]
    y_sorted = y[order]

    cum_tp = 0
    cum_fp = 0

    # 为每个 FPR 水平记录当前最优 (TPR 最大、满足 FPR <= level) 的点
    best: Dict[float, Dict[str, Optional[float]]] = {
        float(lvl): {"tpr": -1.0, "fpr": None, "thr": None} for lvl in fpr_levels
    }

    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            cum_tp += 1
        else:
            cum_fp += 1

        # 同一 score 块只在最后一个样本处更新（因为阈值无法区分块内样本）
        if i + 1 < len(p_sorted) and p_sorted[i + 1] == p_sorted[i]:
            continue

        thr = p_sorted[i]
        tpr = cum_tp / P
        fpr = cum_fp / N

        for lvl, rec in best.items():
            # 这里采用 FPR <= lvl + 1e-12，数值上等价于“FPR 小于等于该值”
            if fpr <= lvl + 1e-12 and tpr > (rec["tpr"] if rec["tpr"] is not None else -1.0):
                rec["tpr"] = tpr
                rec["fpr"] = fpr
                rec["thr"] = thr

    for lvl, rec in best.items():
        key = f"{lvl:g}"
        if rec["tpr"] is None or rec["tpr"] < 0:
            res[key] = {"threshold": None, "tpr": None, "fpr": None}
        else:
            res[key] = {
                "threshold": float(rec["thr"]),
                "tpr": float(rec["tpr"]),
                "fpr": float(rec["fpr"]),
            }
    return res

# ============== NEW: 通用阈值搜索（acc / f1 / tpr@0.01） ==============
def _select_decision_threshold(
    y: np.ndarray,
    p: np.ndarray,
    mode: str,
    tpr_at_fpr: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    """
    根据 mode 选择决策阈值：
      - mode == "acc":  在所有阈值上搜索，选择 accuracy 最大的阈值；
      - mode == "f1":   在所有阈值上搜索，选择 F1 最大的阈值；
      - mode == "tpr":  使用 TPR@FPR<=0.01 扫描得到的最佳点（若存在）。
    返回结构（JSON 可序列化）：
      {
        "mode": "acc" / "f1" / "tpr",
        "threshold": float or None,
        "metric_value": float or None,   # 对应优化目标上的取值（acc 或 f1 或 TPR）
        "metrics": {...} or None,        # 在该阈值下完整的指标（_confusion_and_basic_metrics）
        # 仅 mode == "tpr" 时额外包含：
        "target_fpr": 0.01,
        "operating_point": { "threshold": ..., "tpr": ..., "fpr": ... } or None
      }
    """
    mode = str(mode).lower().strip()
    y = np.asarray(y).astype(int).reshape(-1)
    p = np.asarray(p).astype(float).reshape(-1)
    n = y.shape[0]

    if n == 0:
        return {
            "mode": mode,
            "threshold": None,
            "metric_value": None,
            "metrics": None,
        }

    if mode not in {"acc", "f1", "tpr"}:
        raise ValueError(f"Unknown threshold search mode: {mode} (expected one of: acc, f1, tpr)")

    # --- 模式 3：TPR@FPR<=0.01，直接复用 _scan_tpr_at_fpr 的结果 ---
    if mode == "tpr":
        op = (tpr_at_fpr or {}).get("0.01")
        if op is None or op.get("threshold") is None:
            return {
                "mode": mode,
                "threshold": None,
                "metric_value": None,
                "metrics": None,
                "target_fpr": 0.01,
                "operating_point": op,
            }
        thr = float(op["threshold"])
        metrics = _confusion_and_basic_metrics(y, p, thr=thr)
        # 这里的 metric_value 直接取 TPR（即 op["tpr"]）
        return {
            "mode": mode,
            "threshold": thr,
            "metric_value": float(op.get("tpr", 0.0)) if op.get("tpr") is not None else None,
            "metrics": metrics,
            "target_fpr": 0.01,
            "operating_point": op,
        }

    # --- 模式 1 / 2：在所有阈值上扫描 acc / f1 ---
    # 按概率从大到小排序，阈值从 high -> low 依次移动
    order = np.argsort(-p)
    ps = p[order]
    ys = y[order]

    P = int(np.sum(ys == 1))
    N = n - P
    # 对于 acc 和 f1，即使 P=0 或 N=0 也仍然可以定义（只是 F1 会退化）
    tp = fp = 0
    fn = P
    tn = N

    best_thr: Optional[float] = None
    best_val: float = -1.0

    for i in range(n):
        if ys[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        # 同一 score 块只在最后一个样本处评估一次
        if i + 1 < n and ps[i + 1] == ps[i]:
            continue

        if mode == "acc":
            val = (tp + tn) / (P + N) if (P + N) > 0 else 0.0
        else:  # mode == "f1"
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if (prec + rec) > 0:
                val = 2.0 * prec * rec / (prec + rec)
            else:
                val = 0.0

        if val > best_val + 1e-12:
            best_val = val
            best_thr = float(ps[i])

    if best_thr is None:
        return {
            "mode": mode,
            "threshold": None,
            "metric_value": None,
            "metrics": None,
        }

    metrics = _confusion_and_basic_metrics(y, p, thr=best_thr)
    return {
        "mode": mode,
        "threshold": best_thr,
        "metric_value": float(best_val),
        "metrics": metrics,
    }

# ============== IRLS（Platt & 多特征 LR） ==============
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-z))

def _irls_fit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-6,
    standardize: bool = True,
    show_progress: bool = False,
) -> Dict[str, Any]:
    x = scores.astype(np.float64).reshape(-1)
    y = labels.astype(np.float64).reshape(-1)

    if standardize:
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        if abs(sd) < 1e-12: sd = 1.0
        xs = (x - mu) / sd
    else:
        mu, sd = 0.0, 1.0
        xs = x

    n = xs.shape[0]
    X = np.c_[np.ones((n, 1), dtype=np.float64), xs.reshape(-1, 1)]
    beta = np.zeros(2, dtype=np.float64)
    lam = float(max(0.0, l2))

    pbar = tqdm(range(int(max_iter)), desc="Calibrate: fitting Platt (IRLS)",
                dynamic_ncols=True, disable=not show_progress, leave=False)
    for _ in pbar:
        z = X @ beta
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-9, None)

        g = X.T @ (p - y)
        g[1] += lam * beta[1]

        Xw = X * w.reshape(-1, 1)
        H = X.T @ Xw
        H[1, 1] += lam
        H[0, 0] += 1e-8

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + 1e-6*np.eye(2), g, rcond=None)[0]
        beta_new = beta - step
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    pbar.close()

    beta0, beta1 = float(beta[0]), float(beta[1])
    thr_raw = float("nan") if abs(beta1) < 1e-12 else (mu + (-beta0 / beta1) * sd)

    return {
        "name": "platt_lr",
        "beta0": beta0, "beta1": beta1,
        "standardize": bool(standardize),
        "mean": float(mu), "std": float(sd),
        "threshold_raw_p05": float(thr_raw),
    }

def _irls_fit_logreg_multi(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-6,
    standardize: bool = True,
    show_progress: bool = False,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        X = X.reshape(len(X), -1)
    n, d = X.shape

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
        Xs = (X - mu) / sd
    else:
        mu = np.zeros(d, dtype=np.float64)
        sd = np.ones(d, dtype=np.float64)
        Xs = X

    # [1, X]
    Xmat = np.concatenate([np.ones((n, 1), dtype=np.float64), Xs], axis=1)  # (n, d+1)
    beta = np.zeros(d + 1, dtype=np.float64)

    lam = float(max(0.0, l2))
    # 正则：对偏置很小的 L2，其余系数 L2=lam
    reg = np.diag([1e-8] + [lam] * d)

    pbar = tqdm(range(int(max_iter)), desc="Calibrate: fitting Linear LR (IRLS)",
                dynamic_ncols=True, disable=not show_progress, leave=False)
    for _ in pbar:
        z = Xmat @ beta
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-9, None)

        g = Xmat.T @ (p - y) + reg @ beta
        Xw = Xmat * w.reshape(-1, 1)
        H = Xmat.T @ Xw + reg

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + 1e-6*np.eye(d+1), g, rcond=None)[0]
        beta_new = beta - step
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    pbar.close()

    return {
        "name": "linear_lr",
        "beta": beta.tolist(),                 # [b0, b1, ..., bd]
        "standardize": bool(standardize),
        "mean": mu.tolist(),
        "std": sd.tolist(),
        # 多维没有唯一阈值，留空（兼容字段）
        "threshold_raw_p05": None,
    }

def _apply_platt_1d(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    if params.get("standardize", True):
        mu = float(params.get("mean", 0.0))
        sd = float(params.get("std", 1.0)) or 1.0
        x = (x - mu) / sd
    z = float(params["beta0"]) + float(params["beta1"]) * x
    return _sigmoid(z).astype(np.float32)

def _apply_linear_lr(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    X = np.asarray(scores, dtype=np.float64)
    if X.ndim != 2:
        X = X.reshape(len(X), -1)
    b = np.asarray(params["beta"], dtype=np.float64).reshape(-1)
    b0, bw = b[0], b[1:]
    if params.get("standardize", True):
        mu = np.asarray(params.get("mean"), dtype=np.float64)
        sd = np.asarray(params.get("std"), dtype=np.float64)
        sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
        X = (X - mu) / sd
    z = b0 + X @ bw
    return _sigmoid(z).astype(np.float32)

# ============== 杂项 ==============
def _basename(p: str) -> str:
    p = str(p).rstrip("/\\")
    return os.path.basename(p) if p else "path"

from mgt_eval.utils.paths import user_calib_dir

def _auto_out_path(detector: str, model1: str, data: str, out_dir: Optional[str]) -> str:
    m = _basename(model1); d = _basename(data)
    od = Path(out_dir) if out_dir else user_calib_dir()
    od.mkdir(parents=True, exist_ok=True)
    return str(od / f"calibrator_{detector}_{m}_{d}.json")

def _maybe_set(kwargs: Dict[str, Any], params, key: str, value):
    if key in params and value is not None:
        kwargs[key] = value

def _build_detector(
    detector_name: str,
    model1: str,
    model2: Optional[str],
    *,
    device: Optional[str],
    use_bfloat16: bool,
    detector_kwargs: Optional[Dict[str, Any]],
    basemodel: Optional[str] = None,
    bart_ckpt: Optional[str] = None,
):
    # 1) 取类
    try:
        Det = get_detector_cls(str(detector_name).lower())
    except KeyError:
        Det = get_detector_cls(str(detector_name))

    import inspect
    sig = inspect.signature(Det.__init__)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    def _accepts(key: str) -> bool:
        return has_varkw or (key in params)
    raw_kwargs = dict(detector_kwargs or {})
    kwargs = dict(raw_kwargs) if has_varkw else {k: v for k, v in raw_kwargs.items() if k in params}

    def _set_if_absent(key, value):
       if _accepts(key) and key not in kwargs and value is not None:
            kwargs[key] = value
    # 通用校准开关（允许从 runner 侧强制关闭校准）
    _set_if_absent("outputs_prob", raw_kwargs.get("outputs_prob", None))
    _set_if_absent("disable_calibration", raw_kwargs.get("disable_calibration", None))
    _set_if_absent("auto_calibrate", raw_kwargs.get("auto_calibrate", None))
    _set_if_absent("force_runner_calibration", raw_kwargs.get("force_runner_calibration", None))
    _set_if_absent("device", device)
    for k in ("use_bfloat16", "bf16", "use_bf16"):
        _set_if_absent(k, bool(use_bfloat16) if k in params else None)

    # =========================
    # 模型名注入（统一映射）
    # =========================
    # 1) *_name_or_path 系列（DetectGPT/Lastde++ 等）
    if "scoring_name_or_path" in params or "scoring_model_name_or_path" in params:
        _set_if_absent("scoring_name_or_path", model1)
        _set_if_absent("scoring_model_name_or_path", model1)
        _set_if_absent("reference_name_or_path", model2 or model1)
        _set_if_absent("reference_model_name_or_path", model2 or model1)

    # 2) NEW: Fast-DetectGPT 常见签名（scoring_model_name / sampling_model_name 等）
    #    也兼容简写 scoring_name / sampling_name 以及 fdg_* 前缀
    #    —— 只有当这些键确实出现在 __init__ 签名里时才会注入
    #    —— 未给 model2 时，sampling 默认回落到 model1
    # --------------------------------------------------------
    if ("score_model" in params) or ("scoring_name" in params) or ("scoring_model_name" in params):
        _set_if_absent("score_model", model1 if "score_model" in params else None)
        _set_if_absent("scoring_model_name", model1 if "scoring_model_name" in params else None)
        _set_if_absent("fdg_scoring_model",  model1 if "fdg_scoring_model" in params else None)

        _set_if_absent("sampling_model_name", model2 or model1 if "sampling_model_name" in params else None)
        _set_if_absent("sampling_name",       model2 or model1 if "sampling_name" in params else None)
        _set_if_absent("fdg_sampling_model",  model2 or model1 if "fdg_sampling_model" in params else None)
        # 一些实现用 “reference_*” 指代 sampling
        _set_if_absent("reference_model", model2 or model1 if "reference_model" in params else None)
        _set_if_absent("reference_name",       model2 or model1 if "reference_name" in params else None)

    # 3) 单模型检测器（GLTR/Lastde）
    elif "model_name_or_path" in params:
        _set_if_absent("model_name_or_path", model1)
    elif "model" in params:
        _set_if_absent("model", model1)
        _set_if_absent("tokenizer", model1 if "tokenizer" in params else None)
    elif "observer" in params:
        _set_if_absent("observer", model1)
    elif "observer_model" in params:
        _set_if_absent("observer_model", model1)
    elif "base_model_name_or_path" in params:
        _set_if_absent("base_model_name_or_path", model1)
    elif "rewrite_model" in params:
        _set_if_absent("rewrite_model", model1)
    elif "base_model_name" in params:
        _set_if_absent("base_model_name", model1)
    elif "base_model_name" in params:
        _set_if_absent("base_model_name", model1)
    # —— 若以上都不匹配，就完全依赖 detector_kwargs —— #

    # DetectGPT 的 T5（仅当该键确实在签名中）
    if model2 is not None:
        _set_if_absent("mask_filling_model_name", model2)
        _set_if_absent("mask_model", model2)
        _set_if_absent("mask_name_or_path", model2)
        _set_if_absent("performer", model2)
        _set_if_absent("performer_model", model2)

    # TOCSIN 私有参数
    _set_if_absent("basemodel", basemodel)
    if bart_ckpt is not None:
        if "bart_checkpoint" in params and "bart_checkpoint" not in kwargs:
            kwargs["bart_checkpoint"] = bart_ckpt
    if "bart_ckpt" in raw_kwargs and "bart_checkpoint" in params and "bart_checkpoint" not in kwargs:
        kwargs["bart_checkpoint"] = raw_kwargs["bart_ckpt"]

    det = Det(**kwargs)
    if not getattr(det, "is_loaded", False):
        det.load()
    return det

def _score_with_detector(det, texts: List[str], batch_size: int, show_progress: bool = False) -> np.ndarray:
    outs: List[np.ndarray] = []
    total = len(texts)
    bs = int(max(1, batch_size))
    total_batches = (total + bs - 1) // bs
    iterator = range(0, total, bs)
    pbar = tqdm(
        iterator,
        total=total_batches,
        desc=f"Calibrate[{getattr(det, 'DETECTOR_NAME', 'detector')}] scoring",
        dynamic_ncols=True,
        disable=not show_progress,
        leave=False,
    )
    expected_ndim: Optional[int] = None
    for i in pbar:
        s = det.score_batch(texts[i:i+bs])
        a = np.asarray(s, dtype=np.float64)
        if expected_ndim is None:
            expected_ndim = a.ndim
        if a.ndim != expected_ndim:
            raise RuntimeError(f"Inconsistent score shape across batches: expect ndim={expected_ndim}, got {a.ndim}")
        outs.append(a)
    pbar.close()

    if not outs:
        return np.array([], dtype=np.float64)
    if outs[0].ndim == 1:
        return np.concatenate(outs, axis=0)
    else:
        return np.concatenate(outs, axis=0)  # (N, D)

def Calibrate(
    *,
    # 必需
    model1: str,
    data: str,
    # 可选/统一接口
    detector: str = "lastde",
    model2: Optional[str] = None,
    batch_size: int = 32,
    sample_k: int = 10000,
    seed: int = 114514,
    device: Optional[str] = None,
    bf16: bool = True,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    # NEW: TOCSIN 常用两个前端参数
    basemodel: Optional[str] = None,
    bart_ckpt: Optional[str] = None,
    # 校准器
    calibrator_name: str = "platt_lr",
    l2: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-6,
    standardize: bool = True,
    # NEW: 阈值搜索模式（acc / f1 / tpr）
    mode: str = "acc",
    # 输出
    out: Optional[str] = None,
    out_dir: Optional[str] = None,
    # 进度条
    show_progress: bool = True,
) -> Dict[str, Any]:
    det = _build_detector(
        detector_name=detector,
        model1=model1, model2=model2,
        device=device, use_bfloat16=bool(bf16),
        detector_kwargs=detector_kwargs,
        # NEW: 透传
        basemodel=basemodel,
        bart_ckpt=bart_ckpt,
    )

    if bf16:
        print("[Calibrate] Note: BF16 enabled for detector scoring; ensure this matches your evaluation setting.")
    # --- NEW: 对已经输出概率的检测器禁用校准拟合 ---
    det_type = (getattr(det, "detector_type", "") or "").strip().lower()
    if getattr(det, "outputs_prob", False) or getattr(det, "disable_calibration", False):
        print(f"[Calibrate] Detector '{getattr(det, 'DETECTOR_NAME', detector)}' outputs probability; "
            f"skip calibration fitting and save identity calibrator.")

        out_path = out or _auto_out_path(detector, model1, data, out_dir)
        payload = {
            "calibrator": {
                "name": "none",
                "standardize": False,
                "decision_mode": "fixed",
                "decision_threshold": 0.5,
            },
            "meta": {
                "detector": getattr(det, "DETECTOR_NAME", detector),
                "detector_type": getattr(det, "detector_type", "Unknown"),
                "dev": None,
                "models": {"model1": model1, "model2": model2},
                "dataset": str(data),
                "fit": {"name": "none"},
            }
        }
        Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Calibrate] Calibrator saved to: {out_path}")
        return {"path": out_path, "params": payload["calibrator"], "meta": payload["meta"]}

    # 载入校准数据
    examples, _ = load_dataset_unified(
        dataset=data,
        sample_k=int(sample_k) if sample_k is not None else None,
        sample_seed=int(seed),
        group_cols=None,
    )
    texts = [ex["text"] for ex in examples]
    labels = np.array([int(ex["label"]) for ex in examples], dtype=np.int32)
    if len(texts) == 0:
        raise RuntimeError("Empty calibration dataset after sampling.")

    # 打分（带进度条）
    scores = _score_with_detector(det, texts, batch_size=int(batch_size), show_progress=show_progress)
    if scores.ndim == 1:
        # 行过滤：1D 就是逐样本一个数
        mask_rows = np.isfinite(scores)
    elif scores.ndim == 2:
        # 行过滤：每行必须全为有限值
        mask_rows = np.all(np.isfinite(scores), axis=1)
    else:
        # 高维（不期望）——拍成二维再按行过滤
        scores = scores.reshape(len(labels), -1)
        mask_rows = np.all(np.isfinite(scores), axis=1)

    bad = int((~mask_rows).sum())
    if bad > 0:
        print(f"[Calibrate] drop {bad}/{len(labels)} rows with non-finite scores.")
        scores = scores[mask_rows]
        labels = labels[mask_rows]

    assert scores.shape[0] == labels.shape[0], f"shapes mismatch: scores={scores.shape}, labels={labels.shape}"
    n = scores.shape[0]
    pos = int(labels.sum()); neg = n - pos

    # 原始分数的 AUROC（仅 1D 有意义）
    auroc_raw = _roc_auc_from_scores(labels, scores) if scores.ndim == 1 else None

    # 选择并拟合校准器
    if scores.ndim == 1:
        eff_name = "platt_lr"
        params = _irls_fit_platt(
            scores, labels,
            l2=float(l2), max_iter=int(max_iter), tol=float(tol),
            standardize=bool(standardize), show_progress=show_progress,
        )
        probs_dev = _apply_platt_1d(scores, params)
    else:
        eff_name = "linear_lr" if calibrator_name == "platt_lr" else calibrator_name
        if eff_name == "linear_lr":
            params = _irls_fit_logreg_multi(
                scores, labels,
                l2=float(l2), max_iter=int(max_iter), tol=float(tol),
                standardize=bool(standardize), show_progress=show_progress,
            )
            probs_dev = _apply_linear_lr(scores, params)
        else:
            # 自定义校准器
            cal = get_calibrator(eff_name)
            cfg = {
                "l2": float(l2), "max_iter": int(max_iter),
                "tol": float(tol), "standardize": bool(standardize),
                "show_progress": bool(show_progress),
            }
            params = cal["fit"](scores, labels, cfg)
            probs_dev = np.asarray(cal["apply"](scores, params), dtype=np.float32)

    # dev 上指标（基于概率，固定 thr=0.5，保证与旧版兼容）
    dev_counts = _confusion_and_basic_metrics(labels, probs_dev, thr=0.5)
    auroc_prob = _roc_auc_from_scores(labels, probs_dev)  # 这里按“概率排序”的 AUROC
    aupr_prob = _pr_auc_from_probs(labels, probs_dev)

    # NEW: 扫描阈值，计算 TPR@FPR=1, 0.1, 0.01, 0.05（满足 FPR <= 目标时 TPR 最大的阈值）
    fpr_levels = [0.05, 0.01, 0.001, 0.0001]
    tpr_at_fpr = _scan_tpr_at_fpr(labels, probs_dev, fpr_levels=fpr_levels)

    # 控制台打印一眼可见（TPR@FPR）
    print("[Calibrate] Best thresholds for TPR@FPR:")
    for lvl in sorted(tpr_at_fpr.keys(), key=lambda x: float(x)):
        info = tpr_at_fpr[lvl]
        thr = info["threshold"]
        if thr is None:
            print(f"  FPR<{lvl}: no valid threshold (check label distribution).")
        else:
            print(f"  FPR<{lvl}: thr={thr:.6f}, TPR={info['tpr']:.4f}, FPR={info['fpr']:.4f}")

    # NEW: 三种模式的决策阈值搜索
    mode_eff = str(mode).lower().strip()
    if mode_eff not in {"acc", "f1", "tpr"}:
        raise ValueError(f"Unknown threshold search mode: {mode} (expected one of: acc, f1, tpr).")

    selected_threshold = _select_decision_threshold(
        labels,
        probs_dev,
        mode=mode_eff,
        tpr_at_fpr=tpr_at_fpr,
    )

    print(f"[Calibrate] Decision threshold search (mode={mode_eff}):")
    if selected_threshold["threshold"] is None:
        print("  No valid decision threshold found for the requested mode; "
              "dev metrics above still use thr=0.5.")
    else:
        m = selected_threshold["metrics"]
        print(
            "  thr={thr:.6f}, acc={acc:.4f}, f1={f1:.4f}, "
            "precision={prec:.4f}, recall={rec:.4f}".format(
                thr=selected_threshold["threshold"],
                acc=m["acc"],
                f1=m["f1"],
                prec=m["precision"],
                rec=m["recall"],
            )
        )
        if mode_eff == "tpr":
            op = selected_threshold.get("operating_point")
            if op is not None and op.get("threshold") is not None:
                print(
                    f"  (TPR@FPR<=0.01 operating point: "
                    f"TPR={op['tpr']:.4f}, FPR={op['fpr']:.4f})"
                )

    # 持久化
    out_path = out or _auto_out_path(detector, model1, data, out_dir)
    payload = {
        "calibrator": {
            **params,
            "name": eff_name,
            # NEW: 记录决策阈值及其模式
            "decision_mode": mode_eff,
            "decision_threshold": (
                float(selected_threshold["threshold"])
                if selected_threshold.get("threshold") is not None
                else None
            ),
        },
        "meta": {
            "detector": getattr(det, "DETECTOR_NAME", detector),
            "detector_type": getattr(det, "detector_type", "Unknown"),
            "dev": {
                "num_samples": int(n),
                "pos": pos, "neg": neg,
                "auroc_on_scores": (float(auroc_raw) if auroc_raw is not None else None),
                "metrics_on_probs": {
                    **dev_counts,
                    "auroc": float(auroc_prob),
                    "aupr": float(aupr_prob),
                },
                "feature_dim": (int(scores.shape[1]) if scores.ndim == 2 else 1),
                "tpr_at_fpr": tpr_at_fpr,            # NEW: 阈值扫描结果（原有）
                "threshold_mode": mode_eff,          # NEW: 决策阈值搜索模式
                "selected_threshold": selected_threshold,  # NEW: 最终选择的阈值及指标
            },
            "models": {"model1": model1, "model2": model2},
            "dataset": str(data),
            "fit": {
                "name": eff_name,
                "l2": float(l2),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "standardize": bool(params.get("standardize", standardize)),
            },
        }
    }
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Calibrate] Calibrator saved to: {out_path}")

    return {
        "path": out_path,
        "params": payload["calibrator"],
        "meta": payload["meta"],
    }
