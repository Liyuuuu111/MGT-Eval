from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._metrics_basic import _basic_stats, _f1_from_counts


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
    y2 = [y for y, m in zip(labels, keep) if m]
    p2 = [p for p, m in zip(preds, keep) if m]
    r2 = None if ranking_vec is None else [s for s, m in zip(ranking_vec, keep) if m]
    pb2 = None if probs is None else [s for s, m in zip(probs, keep) if m]
    dropped = n - len(y2)
    return ex2, y2, p2, r2, pb2, dropped


def _get_ranking_values(res) -> Tuple[List[float], str]:
    """
    返回用于 ROC/PR 排序的一维向量：
      - 优先 res.scores（越大越像 AI 的约定由 detector 自行保证）
      - 否则回退 res.probs
    """
    scores = getattr(res, "scores", None)
    if scores is not None:
        try:
            return [float(x) for x in scores], "scores"
        except Exception:
            pass
    probs = getattr(res, "probs", None)
    if probs is not None:
        try:
            return [float(x) for x in probs], "probs"
        except Exception:
            pass
    return [], "none"


def _roc_curve(labels: Sequence[int], probs_or_scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    pairs = sorted(zip(probs_or_scores, labels), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    N = len(pairs) - P
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    last_val = None
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for p, y in pairs:
        if last_val is None or p != last_val:
            auc += (fpr[-1] - prev_fpr) * (tpr[-1] + prev_tpr) / 2.0
            prev_fpr, prev_tpr = fpr[-1], tpr[-1]
            last_val = p
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr_val = tp / P if P else 0.0
        fpr_val = fp / N if N else 0.0
        tpr.append(float(tpr_val))
        fpr.append(float(fpr_val))
    auc += (fpr[-1] - prev_fpr) * (tpr[-1] + prev_tpr) / 2.0
    return fpr, tpr, float(auc)


def _pr_curve(labels: Sequence[int], probs_or_scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    pairs = sorted(zip(probs_or_scores, labels), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    tp = fp = 0
    precision = []
    recall = []
    for p, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / P if P else 0.0
        precision.append(float(prec))
        recall.append(float(rec))

    aupr = 0.0
    prev_r, prev_pv = 0.0, 1.0
    for r, pv in zip(recall, precision):
        aupr += (r - prev_r) * ((pv + prev_pv) / 2.0)
        prev_r, prev_pv = r, pv
    return recall, precision, float(aupr)


def _tpr_at_fpr_points(
    labels: Sequence[int],
    scores: Sequence[float],
    fpr_targets: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
) -> Dict[str, Any]:
    ys = [int(y) for y in labels]
    ss = [float(s) for s in scores]
    n = len(ys)
    if n == 0:
        return {}

    P = sum(1 for y in ys if y == 1)
    N = n - P
    out: Dict[str, Any] = {}
    if P == 0 or N == 0:
        for t in fpr_targets:
            out[str(float(t))] = {"threshold": None, "tpr": None, "fpr": None, "confusion": None, "acc": None, "f1": None}
        return out

    order = sorted(range(n), key=lambda i: (-ss[i], i))

    points = []
    tp = fp = 0
    points.append({"threshold": float("inf"), "tp": 0, "fp": 0, "tn": N, "fn": P, "tpr": 0.0, "fpr": 0.0})

    i = 0
    while i < n:
        thr = ss[order[i]]
        j = i
        while j < n and ss[order[j]] == thr:
            y = ys[order[j]]
            if y == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tn = N - fp
        fn = P - tp
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        points.append({
            "threshold": float(thr),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "tpr": float(tpr),
            "fpr": float(fpr),
        })
        i = j

    for tgt in fpr_targets:
        tgt = float(tgt)
        cand = [p for p in points if p["fpr"] <= tgt]
        if not cand:
            out[str(tgt)] = {"threshold": None, "tpr": None, "fpr": None, "confusion": None, "acc": None, "f1": None}
            continue
        best = cand[-1]
        tp, fp, tn, fn = best["tp"], best["fp"], best["tn"], best["fn"]
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        f1 = _f1_from_counts(tp, fp, fn)
        out[str(tgt)] = {
            "threshold": best["threshold"],
            "tpr": best["tpr"],
            "fpr": best["fpr"],
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "acc": float(acc),
            "f1": float(f1),
        }
    return out


def _ece_brier(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> Dict[str, Any]:
    ys = [int(y) for y in labels]
    ps = [float(p) for p in probs]
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(1, len(ys))

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
        bins_info.append({"bin": b, "n": int(bin_cnts[b]), "conf": float(conf), "acc": float(acc), "gap": float(gap)})

    return {"brier": float(brier), "ece": float(ece), "bins": bins_info}


def _risk_coverage(labels: Sequence[int], probs: Sequence[float]) -> Tuple[List[float], List[float]]:
    ys = [int(y) for y in labels]
    ps = [float(p) for p in probs]
    confs = [max(p, 1.0 - p) for p in ps]
    order = sorted(range(len(ys)), key=lambda i: -confs[i])

    cov: List[float] = []
    risk: List[float] = []
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

    accs: List[float] = []
    aurocs: List[float] = []
    auprs: List[float] = []
    f1s: List[float] = []

    for _ in range(max(1, int(iters))):
        idxs = [rng.randrange(n) for _ in range(n)]
        yb = [ys[i] for i in idxs]
        pb = [ps[i] for i in idxs]
        preds = [1 if p >= 0.5 else 0 for p in pb]

        st = _basic_stats(yb, preds, pb)
        accs.append(float(st["acc"]))
        tp = st["confusion"]["tp"]
        fp = st["confusion"]["fp"]
        fn = st["confusion"]["fn"]
        f1s.append(float(_f1_from_counts(tp, fp, fn)))

        _fpr, _tpr, auroc = _roc_curve(yb, pb)
        aurocs.append(float(auroc))
        _r, _p, aupr = _pr_curve(yb, pb)
        auprs.append(float(aupr))

    def _pct(a: List[float], q: float) -> float:
        k = int(round((q / 100.0) * (len(a) - 1)))
        sa = sorted(a)
        return float(sa[max(0, min(len(sa) - 1, k))])

    return {
        "acc": {"mean": float(sum(accs) / len(accs)), "lo": _pct(accs, 2.5), "hi": _pct(accs, 97.5)},
        "auroc": {"mean": float(sum(aurocs) / len(aurocs)), "lo": _pct(aurocs, 2.5), "hi": _pct(aurocs, 97.5)},
        "aupr": {"mean": float(sum(auprs) / len(auprs)), "lo": _pct(auprs, 2.5), "hi": _pct(auprs, 97.5)},
        "f1": {"mean": float(sum(f1s) / len(f1s)), "lo": _pct(f1s, 2.5), "hi": _pct(f1s, 97.5)},
        "iters": int(iters),
        "seed": int(seed),
    }
