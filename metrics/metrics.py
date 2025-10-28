from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

def _as_np1d(x, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr

def _filter_by_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    仅依据 y_score 的有限性(NaN/Inf/None)进行过滤，同步裁剪 y_true/y_pred。
    返回 (y_true, y_score, y_pred, dropped_count)。
    """
    y_true = _as_np1d(y_true, dtype=int)
    y_score = _as_np1d(y_score, dtype=float)
    y_pred = None if y_pred is None else _as_np1d(y_pred, dtype=int)

    mask = np.isfinite(y_score)
    if not np.all(mask):
        dropped = int((~mask).sum())
        y_true = y_true[mask]
        y_score = y_score[mask]
        if y_pred is not None:
            y_pred = y_pred[mask]
    else:
        dropped = 0
    return y_true, y_score, y_pred, dropped

def _tpr_at_fpr(y_true, y_score, target_fpr: float) -> float:
    # 过滤非有限值
    y_true, y_score, _, _ = _filter_by_score(y_true, y_score, None)
    if y_true.size == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))

def roc_pr_curves(y_true, y_score):
    # 过滤非有限值
    y_true, y_score, _, _ = _filter_by_score(y_true, y_score, None)
    if y_true.size == 0:
        return {"fpr": [], "tpr": [], "precision": [], "recall": []}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": prec.tolist(), "recall": rec.tolist()}

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray],
    tpr_at_fpr: Tuple[float, ...] = (0.01, 0.05, 0.10)
) -> Dict[str, float]:
    """
    对 y_score 做 NaN/Inf 过滤，同步裁剪 y_true/y_pred，剩余样本计算指标。
    注意：如果 y_pred 为 None（或长度不匹配），会用 y_score>=0.5 自动阈值生成。
    """
    y_true, y_score, y_pred, dropped = _filter_by_score(y_true, y_score, y_pred)

    metrics: Dict[str, float] = {}
    if y_true.size == 0:
        # 全部被丢弃时，返回 NaN/空
        metrics["accuracy"] = float("nan")
        metrics["f1"] = float("nan")
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        for f in tpr_at_fpr:
            metrics[f"tpr@fpr={int(f*100)}%"] = float("nan")
        metrics["dropped_nonfinite"] = float(dropped)
        return metrics

    if y_pred is None or y_pred.size != y_true.size:
        y_pred = (y_score >= 0.5).astype(int)

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["auprc"] = float("nan")

    for f in tpr_at_fpr:
        try:
            metrics[f"tpr@fpr={int(f*100)}%"] = _tpr_at_fpr(y_true, y_score, f)
        except Exception:
            metrics[f"tpr@fpr={int(f*100)}%"] = float("nan")

    # 可用于日志/排查
    metrics["dropped_nonfinite"] = float(dropped)
    return metrics
