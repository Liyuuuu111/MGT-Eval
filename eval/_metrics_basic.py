from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def _basic_stats(labels: Sequence[int], preds: Sequence[int], probs: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    lbls = [int(x) for x in labels]
    prds = [int(x) for x in preds]
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
        prbs = [float(x) for x in probs]
        if prbs:
            out.update({
                "prob_mean": sum(prbs) / len(prbs),
                "prob_min": min(prbs),
                "prob_max": max(prbs),
            })
    return out


def _by_group(
    group_values: Sequence[str],
    labels: Sequence[int],
    preds: Sequence[int],
    *,
    include_mask: Optional[Sequence[bool]] = None,
    exclude_values: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    gs = [str(g) for g in group_values]
    ys = [int(y) for y in labels]
    ps = [int(p) for p in preds]
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
        if y == 1 and p == 1:
            d["tp"] += 1
        elif y == 0 and p == 0:
            d["tn"] += 1
        elif y == 0 and p == 1:
            d["fp"] += 1
        elif y == 1 and p == 0:
            d["fn"] += 1

    for g, d in out.items():
        tp, tn, fp, fn = d["tp"], d["tn"], d["fp"], d["fn"]
        n = d["n"]
        d["acc"] = (tp + tn) / n if n else 0.0
        d["f1"] = _f1_from_counts(tp, fp, fn)
    return out
