# mgt_eval/calibration/platt.py
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union
import json, time, math
import numpy as np

from .registry import register_calibrator, attach_apply
# 可选：用于直接从数据集拟合（外部脚本可调用）
try:
    from ..data_utils.load import load_dataset_unified
except Exception:
    load_dataset_unified = None

SIGMOID = lambda z: 1.0 / (1.0 + np.exp(-z))

def _standardize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0
    return (x - mu) / sd, mu, sd

def _irls_fit(scores: np.ndarray, labels: np.ndarray,
              l2: float = 1e-2, max_iter: int = 200, tol: float = 1e-6,
              standardize: bool = True) -> Dict[str, Any]:
    """
    单变量逻辑回归（Platt）：p = sigmoid(b0 + b1 * s_std)
    - 仅对斜率 b1 做 L2 正则（bias 不正则）。
    - 使用 IRLS / Newton 更新，2x2 线性方程可闭式解。
    """
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    assert x.shape[0] == y.shape[0] and x.shape[0] > 1

    if standardize:
        xs, mu, sd = _standardize(x)
    else:
        xs, mu, sd = x.copy(), 0.0, 1.0

    # 设计矩阵: [1, x]
    X0 = np.ones_like(xs)
    X1 = xs
    b0, b1 = 0.0, 1.0  # 初值

    for _ in range(int(max_iter)):
        z = b0 + b1 * X1
        p = SIGMOID(z)
        # 权重与梯度
        W = p * (1.0 - p) + 1e-12
        g0 = np.sum(p - y)                           # 对 b0 的梯度
        g1 = np.sum((p - y) * X1) + l2 * b1          # 对 b1 的梯度（含 L2）
        # 2x2 Hessian
        H00 = np.sum(W * (X0 * X0))                  # = sum(W)
        H01 = np.sum(W * (X0 * X1))                  # = sum(W*X1)
        H11 = np.sum(W * (X1 * X1)) + l2             # 加正则
        # 求解牛顿步
        det = H00 * H11 - H01 * H01
        if abs(det) < 1e-12:
            break
        db0 = -( H11 * g0 - H01 * g1) / det
        db1 = -(-H01 * g0 + H00 * g1) / det
        # 更新
        b0_new = b0 + db0
        b1_new = b1 + db1
        if max(abs(db0), abs(db1)) < tol:
            b0, b1 = b0_new, b1_new
            break
        b0, b1 = b0_new, b1_new

    # p=0.5 的原始分数阈值（方便导出）
    if abs(b1) < 1e-12:
        thr_raw = float("nan")
    else:
        thr_std = -b0 / b1
        thr_raw = mu + sd * thr_std

    return {
        "beta0": float(b0),
        "beta1": float(b1),
        "standardize": bool(standardize),
        "mean": float(mu),
        "std": float(sd),
        "threshold_raw_p05": float(thr_raw),
    }

@register_calibrator("platt_lr")
def fit_platt(scores: np.ndarray, labels: np.ndarray,
              *, l2: float = 1e-2, max_iter: int = 200, tol: float = 1e-6,
              standardize: bool = True) -> Dict[str, Any]:
    params = _irls_fit(scores, labels, l2=l2, max_iter=max_iter, tol=tol, standardize=standardize)
    params["name"] = "platt_lr"
    params["version"] = 1
    params["fitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    params["pos_rate"] = float(np.mean(labels))
    return params

@attach_apply("platt_lr")
def apply_platt(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if params.get("standardize", True):
        mu = float(params.get("mean", 0.0))
        sd = float(params.get("std", 1.0))
        sd = (sd if abs(sd) > 1e-12 else 1.0)
        s = (s - mu) / sd
    b0 = float(params["beta0"])
    b1 = float(params["beta1"])
    z = b0 + b1 * s
    # 数值安全
    z = np.clip(z, -80.0, 80.0)
    p = SIGMOID(z)
    # 开区间裁剪，便于 ECE/Brier
    return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

# 在 mgt_eval/calibration/platt.py 内替换此函数
def fit_from_dataset_and_save(
    detector,
    dev_dataset: Union[str, Iterable[Dict[str, Any]]],
    out_json: str,
    *,
    batch_size: int = 32,
    calibrator_name: str = "platt_lr",
    l2: float = 1e-2, max_iter: int = 200, tol: float = 1e-6,
    standardize: bool = True,
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    group_cols: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    外部脚本可直接调用：
        params = fit_from_dataset_and_save(det, dev_path, "calibrator.json", sample_k=10000)

    - detector: 任何实现了 score_batch(texts) 的检测器（Metric-based：只需返回分数）
    - dev_dataset: 标注好的开发集（str 或 iterable of dict）
    - out_json: 输出校准 JSON 文件路径
    - sample_k/sample_seed: 从开发集抽样拟合（与 load_dataset_unified 对齐）
    """
    if not hasattr(detector, "score_batch"):
        raise TypeError("detector must provide score_batch(texts)->np.ndarray")

    # —— 载入或展开开发集 —— #
    if isinstance(dev_dataset, str):
        if load_dataset_unified is None:
            raise RuntimeError("load_dataset_unified not available.")
        examples, _ = load_dataset_unified(
            dataset=dev_dataset,
            sample_k=sample_k,
            sample_seed=sample_seed,
            group_cols=group_cols,
        )
    else:
        examples = list(dev_dataset)
        if sample_k is not None and sample_k > 0 and len(examples) > sample_k:
            import random
            rnd = random.Random(int(sample_seed))
            examples = [examples[i] for i in sorted(rnd.sample(range(len(examples)), k=sample_k))]

    texts = [ex["text"] for ex in examples]
    labels = np.array([int(ex["label"]) for ex in examples], dtype=np.int32)

    # —— 批量打分（分数；概率稍后由校准器负责） —— #
    scores_all: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        scores_all.append(detector.score_batch(texts[i:i+batch_size]).astype(np.float64))
    scores = np.concatenate(scores_all, axis=0).reshape(-1)

    # —— 拟合 —— #
    if calibrator_name != "platt_lr":
        from .registry import get_calibrator
        cal = get_calibrator(calibrator_name)
        params = cal["fit"](scores, labels)
    else:
        params = fit_platt(scores, labels, l2=l2, max_iter=max_iter, tol=tol, standardize=standardize)

    # —— 保存 JSON（含元信息、p=0.5 的原始阈值） —— #
    meta = {
        "detector_name": getattr(detector, "DETECTOR_NAME", "detector"),
        "detector_type": getattr(detector, "detector_type", "Unknown"),
        "num_samples": int(len(scores)),
        "sample_k": int(sample_k) if sample_k is not None else None,
        "sample_seed": int(sample_seed),
    }
    payload = {"calibrator": params, "meta": meta}

    Path(os.path.dirname(out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

