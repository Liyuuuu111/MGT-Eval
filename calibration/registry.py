# mgt_eval/calibration/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

_REG: Dict[str, Dict[str, Callable]] = {}

def register_calibrator(name: str):
    def deco(builder: Callable[[], Dict[str, Callable]]):
        entry = builder()
        assert "fit" in entry and "apply" in entry
        _REG[name] = entry
        return builder
    return deco

def get_calibrator(name: str) -> Dict[str, Callable]:
    if name not in _REG:
        raise KeyError(f"Calibrator '{name}' not registered.")
    return _REG[name]

# ========= 默认内置：多特征线性逻辑回归（与 runner 的 IRLS 对齐） =========
@register_calibrator("linear_lr")
def _build_linear_lr():
    def _sigmoid(z: np.ndarray):
        z = np.clip(z, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(scores: np.ndarray, labels: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        n, d = X.shape
        l2 = float(cfg.get("l2", 1e-2))
        max_iter = int(cfg.get("max_iter", 200))
        tol = float(cfg.get("tol", 1e-6))
        standardize = bool(cfg.get("standardize", True))

        if standardize:
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=0)
            sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
            Xs = (X - mu) / sd
        else:
            mu = np.zeros(d, dtype=np.float64)
            sd = np.ones(d, dtype=np.float64)
            Xs = X

        Xmat = np.concatenate([np.ones((n, 1), dtype=np.float64), Xs], axis=1)
        beta = np.zeros(d + 1, dtype=np.float64)
        reg = np.diag([1e-8] + [float(max(0.0, l2))] * d)

        for _ in range(max_iter):
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

        return {
            "name": "linear_lr",
            "beta": beta.tolist(),
            "standardize": bool(standardize),
            "mean": mu.tolist(),
            "std": sd.tolist(),
            "threshold_raw_p05": None,
        }

    def apply(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
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
        z = np.clip(z, -80.0, 80.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

    return {"fit": fit, "apply": apply}
