# mgt_eval/calibration/Calibrate.py
from __future__ import annotations
import argparse, json, os, inspect, math, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from detectors.registry import get_detector_cls
from data_utils.load import load_dataset_unified

# ============== （） ==============

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-z))

def _auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    # ：-
    order = np.argsort(-scores)
    y = labels[order]
    P = float(y.sum())
    N = float(len(y) - P)
    if P == 0.0 or N == 0.0:
        return float("nan")
    tp = 0.0
    auc = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1.0
        else:
            auc += tp
    return auc / (P * N)

def _irls_fit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-6,
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    拟合二项逻辑回归：p = sigmoid(beta0 + beta1 * x)
      - 仅对 beta1 加 L2（intercept 不惩罚）
      - 可选对 x 做 z-score 标准化
    返回：
      {
        name: "platt_lr",
        beta0, beta1,
        standardize: bool,
        mean, std,
        threshold_raw_p05: float (p=0.5 对应的“原始分数阈值”)
      }
    """
    x = scores.astype(np.float64).reshape(-1)
    y = labels.astype(np.float64).reshape(-1)
    assert x.shape[0] == y.shape[0] and x.ndim == 1

    if standardize:
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        if abs(sd) < 1e-12:
            sd = 1.0
        xs = (x - mu) / sd
    else:
        mu, sd = 0.0, 1.0
        xs = x

    # ：X = [1, xs]
    n = xs.shape[0]
    X0 = np.ones((n, 1), dtype=np.float64)
    X1 = xs.reshape(-1, 1)
    X = np.concatenate([X0, X1], axis=1)  # (n,2)
    beta = np.zeros(2, dtype=np.float64)

    # ： slope (beta1)  L2
    # Hessian  diag([0, l2])
    lam = float(max(0.0, l2))

    for _ in range(int(max_iter)):
        z = X @ beta
        p = _sigmoid(z)
        # W = diag(p*(1-p))
        w = p * (1.0 - p)
        # ：X^T (p - y) + reg
        g = X.T @ (p - y)
        g[1] += lam * beta[1]  # beta1  L2

        # Hessian：X^T W X + reg
        Xw = X * w.reshape(-1, 1)
        H = X.T @ Xw
        H[1, 1] += lam

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # ，
            H_damped = H + 1e-6 * np.eye(2)
            step = np.linalg.solve(H_damped, g)

        beta_new = beta - step
        if np.linalg.norm(beta_new - beta, ord=2) < tol:
            beta = beta_new
            break
        beta = beta_new

    beta0, beta1 = float(beta[0]), float(beta[1])

    # p=0.5  z=0 → （）x  x_thr = -beta0/beta1
    if abs(beta1) < 1e-12:
        thr_raw = float("nan")
    else:
        x_thr_std = -beta0 / beta1
        # “”
        thr_raw = mu + x_thr_std * sd

    return {
        "name": "platt_lr",
        "beta0": beta0,
        "beta1": beta1,
        "standardize": bool(standardize),
        "mean": float(mu),
        "std": float(sd),
        "threshold_raw_p05": float(thr_raw),
    }

def _basename(p: str) -> str:
    p = str(p).rstrip("/\\")
    return os.path.basename(p) if p else "path"

def _dataset_basename(p: str) -> str:
    name = _basename(p)
    # strip common data file extensions (possibly stacked)
    while True:
        lower = name.lower()
        for ext in (".jsonl", ".json", ".csv", ".parquet"):
            if lower.endswith(ext):
                name = name[: -len(ext)]
                break
        else:
            break
    return name or "data"

def _safe_tag(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._+\-]+", "-", s)
    s = s.strip("-")
    return s or "x"

def _auto_out_path(
    detector: str,
    model1: str,
    model2: Optional[str],
    data: str,
    sample_k: Optional[int],
    seed: Optional[int],
    out_dir: Optional[str],
) -> str:
    m1 = _safe_tag(_basename(model1))
    m2 = _safe_tag(_basename(model2)) if model2 else None
    d = _safe_tag(_dataset_basename(data))
    sk = "all"
    if sample_k is not None:
        try:
            sk = str(int(sample_k)) if int(sample_k) > 0 else "all"
        except Exception:
            sk = str(sample_k)
    sd = seed if seed is not None else 114514
    try:
        sd = int(sd)
    except Exception:
        sd = str(sd)

    parts = [str(detector), m1]
    if m2:
        parts.append(m2)
    parts.extend([d, str(sk), str(sd)])
    fname = "_".join(_safe_tag(p) for p in parts) + ".json"

    od = Path(out_dir or "calibrators")
    od.mkdir(parents=True, exist_ok=True)
    return str(od / fname)

def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    detector_args_json: Optional[str],
):
    Det = get_detector_cls(detector_name)
    sig = inspect.signature(Det.__init__)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    extra = _load_json(detector_args_json)

    kwargs: Dict[str, Any] = dict(extra)

    # device/bf16
    _maybe_set(kwargs, params, "device", device)
    for k in ("use_bfloat16", "bf16", "use_bf16"):
        if k in params:
            kwargs[k] = bool(use_bfloat16)

    # （）
    if "scoring_model_name_or_path" in params:
        kwargs["scoring_model_name_or_path"] = model1
        if "reference_model_name_or_path" in params:
            kwargs["reference_model_name_or_path"] = (model2 or model1)
    elif "model_name_or_path" in params:
        kwargs["model_name_or_path"] = model1
    elif "model" in params:
        kwargs["model"] = model1
        if "tokenizer" in params and "tokenizer" not in kwargs:
            kwargs["tokenizer"] = model1
    else:
        kwargs["model_name_or_path"] = model1

    if str(detector_name).strip().lower() == "taste":
        if model1 is not None:
            kwargs.setdefault("model1", model1)
        if model2 is not None:
            kwargs.setdefault("model2", model2)
    elif has_varkw and model1 is not None:
        kwargs.setdefault("model1", model1)

    det = Det(**kwargs)
    return det

def _score_with_detector(det, texts: List[str], batch_size: int) -> np.ndarray:
    scores: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        s = det.score_batch(texts[i:i+batch_size])
        scores.append(np.asarray(s, dtype=np.float64).reshape(-1))
    return np.concatenate(scores, axis=0)

# ============== ：Calibrate(args...) ==============

def Calibrate(argv=None):
    ap = argparse.ArgumentParser(
        description="Calibration runner: fit a logistic-regression (Platt) calibrator for any metric-based detector."
    )
    # ——  ——
    ap.add_argument("--model1", type=str, required=True, help="Primary model path / HF id.")
    ap.add_argument("--data",   type=str, required=True, help="Calibration dataset (json/jsonl).")

    # ——  ——
    ap.add_argument("--model2", type=str, default=None, help="Reference model for 2-model detectors (e.g., lastdepp).")
    ap.add_argument("--detector", type=str, default="lastde",
                    help="Detector registry name (e.g., lastde, lastdepp, likelihood, rank, logrank, entropy). Default: lastde")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring.")
    ap.add_argument("--sample-k", type=int, default=10000, help="Number of samples drawn from the dev set.")
    ap.add_argument("--seed", type=int, default=114514, help="Sampling seed.")
    ap.add_argument("--device", type=str, default=None, help='Device hint, e.g., "cuda:0" or "cpu".')
    ap.add_argument("--bf16", action="store_true", help="Enable bfloat16 if supported (default off).")
    ap.add_argument("--detector-args-json", type=str, default=None,
                    help="Optional JSON file of extra kwargs passed to detector constructor (used only if provided).")

    # —— （Platt） ——
    ap.add_argument("--l2", type=float, default=1e-2, help="L2 weight on slope (beta1).")
    ap.add_argument("--max-iter", type=int, default=200, help="Max IRLS iterations.")
    ap.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance.")
    ap.add_argument("--no-standardize", action="store_true", help="Disable z-score on scores before LR.")

    # ——  ——
    ap.add_argument("--out", type=str, default=None, help="Output calibrator JSON path.")
    ap.add_argument("--out-dir", type=str, default="calibrators", help="If --out missing, save to this directory with auto name.")

    args = ap.parse_args(argv)

    # 1)  detector
    det = _build_detector(
        detector_name=args.detector,
        model1=args.model1,
        model2=args.model2,
        device=args.device,
        use_bfloat16=bool(args.bf16),
        detector_args_json=args.detector_args_json,
    )
    if not getattr(det, "is_loaded", False):
        det.load()

    # 2) （）
    examples, _ = load_dataset_unified(
        dataset=args.data,
        sample_k=int(args.sample_k) if args.sample_k is not None else None,
        sample_seed=int(args.seed),
        group_cols=None,
    )
    texts = [ex["text"] for ex in examples]
    labels = np.array([int(ex["label"]) for ex in examples], dtype=np.int32)
    if len(texts) == 0:
        raise RuntimeError("Empty calibration dataset after sampling.")

    # 3)  detector （****）
    scores = _score_with_detector(det, texts, batch_size=int(args.batch_size))
    assert scores.shape[0] == labels.shape[0]

    # 4)  AUROC（，）
    auroc_raw = _auroc_from_scores(labels, scores)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)

    # 5)  Platt（ IRLS）
    params = _irls_fit_platt(
        scores, labels,
        l2=float(args.l2),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        standardize=(not args.no_standardize),
    )

    # 6)  JSON
    out_path = args.out or _auto_out_path(
        args.detector,
        args.model1,
        args.model2,
        args.data,
        args.sample_k,
        args.seed,
        args.out_dir,
    )
    payload = {
        "calibrator": params,
        "meta": {
            "detector": getattr(det, "DETECTOR_NAME", args.detector),
            "detector_type": getattr(det, "detector_type", "Unknown"),
            "dev": {
                "num_samples": int(len(labels)),
                "pos": pos,
                "neg": neg,
                "auroc_on_scores": float(auroc_raw),
            },
            "models": {
                "model1": args.model1,
                "model2": args.model2,
            },
            "dataset": str(args.data),
            "fit": {
                "l2": float(args.l2),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
                "standardize": bool(params["standardize"]),
            },
        }
    }
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Calibrate] Saved calibrator to: {out_path}")
    print(f"[Calibrate] beta0={params['beta0']:.6f}  beta1={params['beta1']:.6f}  "
          f"std={params['standardize']}  mean={params['mean']:.6f}  stddev={params['std']:.6f}")
    thr = params.get("threshold_raw_p05", None)
    if thr is not None and math.isfinite(thr):
        print(f"[Calibrate] Raw-score threshold @ p=0.5 : {thr:.6f}")
    print(f"[Calibrate] Dev AUROC (scores only): {auroc_raw:.4f}")

if __name__ == "__main__":
    Calibrate()
