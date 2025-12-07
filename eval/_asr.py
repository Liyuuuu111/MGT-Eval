from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from ..detectors.base import DetectorBase
from ._utils_common import _norm_id
from ._utils_loader import _load_examples_auto


def _attack_method_name(e: Dict[str, Any]) -> str:
    for k in ("aug_method", "attack_method", "attack_type"):
        v = e.get(k, None)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return "unknown"


def _match_key(e: Dict[str, Any]) -> Optional[str]:
    for k in ("orig_id", "base_id", "source_id"):
        v = _norm_id(e.get(k))
        if v is not None:
            return v

    sid = _norm_id(e.get("id"))
    if sid is None:
        return None

    am = str(e.get("aug_method") or "").strip()
    if am:
        token = f"-{am}-"
        if token in sid:
            return sid.rsplit(token, 1)[0]
    return sid


def _summarize_asr_attacks(attacks_out: Dict[str, Any]) -> Dict[str, Any]:
    vals: List[float] = []
    wts: List[int] = []
    for _, rec in (attacks_out or {}).items():
        a = rec.get("asr", None)
        if a is None:
            continue
        try:
            af = float(a)
        except Exception:
            continue
        vals.append(af)

        w = rec.get("attack_eval_n", None)
        if not isinstance(w, int):
            w = rec.get("base_correct_n", 0)
        wts.append(max(0, int(w)))

    asr_mean = (sum(vals) / len(vals)) if vals else None
    denom = sum(wts)
    asr_weighted_mean = (sum(v * w for v, w in zip(vals, wts)) / denom) if (vals and denom > 0) else None
    return {
        "n_attacks": int(len(attacks_out or {})),
        "n_valid_asr": int(len(vals)),
        "asr_mean": asr_mean,
        "asr_weighted_mean": asr_weighted_mean,
        "weighting": "attack_eval_n (fallback base_correct_n)",
    }


def _align_pairs(base_exs: List[Dict[str, Any]], atk_exs: List[Dict[str, Any]]):
    base_keys = [_match_key(e) for e in base_exs]
    atk_keys = [_match_key(e) for e in atk_exs]
    can_id_match = (
        len(base_exs) > 0 and len(atk_exs) > 0
        and all(x is not None for x in base_keys)
        and all(x is not None for x in atk_keys)
    )

    if can_id_match:
        amap: Dict[str, Dict[str, Any]] = {}
        dup = 0
        for e in atk_exs:
            k = _match_key(e)
            if k is None:
                continue
            if k in amap:
                dup += 1
                continue
            amap[k] = e

        b2, a2, miss = [], [], 0
        for b in base_exs:
            k = _match_key(b)
            if k is None or k not in amap:
                miss += 1
                continue
            b2.append(b)
            a2.append(amap[k])

        stats = {
            "base_n": len(base_exs),
            "atk_n": len(atk_exs),
            "matched_n": len(b2),
            "missing_in_attack": miss,
            "attack_id_duplicates_dropped": dup,
            "note": "attack ids normalized by aug_method suffix",
        }
        return b2, a2, "id", stats

    n = min(len(base_exs), len(atk_exs))
    stats = {"base_n": len(base_exs), "atk_n": len(atk_exs), "matched_n": n}
    return base_exs[:n], atk_exs[:n], "order", stats


def _precompute_base_correct_cache(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
) -> Dict[str, Any]:
    if not base_exs:
        return {"use_key": False, "correct_mask_by_index": [], "correct_by_key": {}, "base_n": 0}

    res0 = det.evaluate(base_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y0 = [int(x) for x in res0.labels]
    p0 = [int(x) for x in res0.preds]
    correct_mask = [yy == pp for yy, pp in zip(y0, p0)]

    keys = [_match_key(e) for e in base_exs]
    use_key = all(k is not None for k in keys)

    correct_by_key: Dict[str, bool] = {}
    if use_key:
        for k, ok in zip(keys, correct_mask):
            if k is None:
                continue
            if k in correct_by_key:
                continue
            correct_by_key[k] = bool(ok)

    return {
        "use_key": bool(use_key),
        "correct_mask_by_index": correct_mask,
        "correct_by_key": correct_by_key,
        "base_n": int(len(base_exs)),
    }


def _base_correct_cache_from_preds(
    base_exs: List[Dict[str, Any]],
    labels: List[int],
    preds: List[int],
) -> Dict[str, Any]:
    n = min(len(base_exs), len(labels), len(preds))
    base_exs = base_exs[:n]
    labels = labels[:n]
    preds = preds[:n]

    correct_mask = [int(y) == int(p) for y, p in zip(labels, preds)]
    keys = [_match_key(e) for e in base_exs]
    use_key = all(k is not None for k in keys)

    correct_by_key: Dict[str, bool] = {}
    if use_key:
        for k, ok in zip(keys, correct_mask):
            if k in correct_by_key:
                continue
            correct_by_key[k] = bool(ok)

    return {
        "use_key": bool(use_key),
        "correct_mask_by_index": correct_mask,
        "correct_by_key": correct_by_key,
        "base_n": int(n),
        "source": "main_eval_preds",
    }


def _compute_asr(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
    base_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    ASR = 1 - Acc(attack | correct_before_attack)
    """
    base_aligned, atk_aligned, mode, align_stats = _align_pairs(base_exs, atk_exs)
    if len(base_aligned) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": 0,
            "attack_eval_n": 0,
            "attack_acc": None,
            "asr": None,
        }

    correct_idx: List[int] = []
    if base_cache is not None:
        if mode == "id" and base_cache.get("use_key", False):
            correct_by_key = base_cache.get("correct_by_key", {})
            for i, b in enumerate(base_aligned):
                k = _match_key(b)
                if k is not None and bool(correct_by_key.get(k, False)):
                    correct_idx.append(i)
        else:
            mask = base_cache.get("correct_mask_by_index", [])
            for i in range(len(base_aligned)):
                if i < len(mask) and bool(mask[i]):
                    correct_idx.append(i)
    else:
        res0 = det.evaluate(base_aligned, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y0 = [int(x) for x in res0.labels]
        p0 = [int(x) for x in res0.preds]
        correct_idx = [i for i, (y, p) in enumerate(zip(y0, p0)) if y == p]

    if len(correct_idx) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": 0,
            "attack_eval_n": 0,
            "attack_acc": None,
            "asr": None,
        }

    atk_subset: List[Dict[str, Any]] = []
    for i in correct_idx:
        b = base_aligned[i]
        a = atk_aligned[i]
        ex = dict(a)
        ex["label"] = int(b.get("label", 1))
        if _norm_id(b.get("id")) is not None:
            ex["id"] = b.get("id")
        atk_subset.append(ex)

    res1 = det.evaluate(atk_subset, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y1 = [int(x) for x in res1.labels]
    p1 = [int(x) for x in res1.preds]
    acc1 = sum(1 for y, p in zip(y1, p1) if y == p) / max(1, len(y1))
    asr = 1.0 - float(acc1)

    return {
        "match_mode": mode,
        "align": align_stats,
        "base_correct_n": int(len(correct_idx)),
        "attack_eval_n": int(len(y1)),
        "attack_acc": float(acc1),
        "asr": float(asr),
    }


def _align_base_to_attacks_one_to_many(
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], str, Dict[str, Any]]:
    base_keys = [_match_key(e) for e in base_exs]
    atk_keys = [_match_key(e) for e in atk_exs]
    can_id_match = (
        len(base_exs) > 0 and len(atk_exs) > 0
        and all(x is not None for x in base_keys)
        and all(x is not None for x in atk_keys)
    )

    if not can_id_match:
        n = min(len(base_exs), len(atk_exs))
        stats = {
            "base_n": len(base_exs),
            "atk_n": len(atk_exs),
            "matched_n": n,
            "note": "fallback order matching (one-to-one); cannot do one-to-many without ids",
        }
        return base_exs[:n], [[atk_exs[i]] for i in range(n)], "order", stats

    amap: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in atk_exs:
        k = _match_key(a)
        if k is None:
            continue
        amap[k].append(a)

    base_aligned: List[Dict[str, Any]] = []
    atk_lists: List[List[Dict[str, Any]]] = []
    miss = 0
    for b in base_exs:
        k = _match_key(b)
        if k is None or k not in amap:
            miss += 1
            continue
        base_aligned.append(b)
        atk_lists.append(amap[k])

    total_variants = sum(len(vs) for vs in atk_lists)
    stats = {
        "base_n": len(base_exs),
        "atk_n": len(atk_exs),
        "matched_n": len(base_aligned),
        "missing_in_attack": miss,
        "attack_keys_unique": int(len(amap)),
        "attack_total_variants_aligned": int(total_variants),
        "avg_variants_per_matched_base": (total_variants / len(base_aligned)) if base_aligned else 0.0,
        "note": "one-to-many id matching via _match_key (aug_method suffix normalized)",
    }
    return base_aligned, atk_lists, "id_one_to_many", stats


def _compute_asr_any_success_one_method(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    atk_exs_same_method: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
    base_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_aligned, atk_lists, mode, align_stats = _align_base_to_attacks_one_to_many(base_exs, atk_exs_same_method)
    if len(base_aligned) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": 0,
            "attack_eval_n": 0,
            "attack_variant_n": 0,
            "attack_variant_acc": None,
            "attack_acc": None,
            "asr": None,
        }

    correct_idx: List[int] = []
    if base_cache is not None:
        if mode.startswith("id") and base_cache.get("use_key", False):
            correct_by_key = base_cache.get("correct_by_key", {})
            for i, b in enumerate(base_aligned):
                k = _match_key(b)
                if k is not None and bool(correct_by_key.get(k, False)):
                    correct_idx.append(i)
        else:
            mask = base_cache.get("correct_mask_by_index", [])
            for i in range(len(base_aligned)):
                if i < len(mask) and bool(mask[i]):
                    correct_idx.append(i)
    else:
        res0 = det.evaluate(base_aligned, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y0 = [int(x) for x in res0.labels]
        p0 = [int(x) for x in res0.preds]
        correct_idx = [i for i, (y, p) in enumerate(zip(y0, p0)) if y == p]

    if len(correct_idx) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": 0,
            "attack_eval_n": 0,
            "attack_variant_n": 0,
            "attack_variant_acc": None,
            "attack_acc": None,
            "asr": None,
        }

    atk_eval_exs: List[Dict[str, Any]] = []
    base_pos_for_each_variant: List[int] = []

    for i in correct_idx:
        b = base_aligned[i]
        y = int(b.get("label", 1))
        bid = b.get("id", None)
        variants = atk_lists[i] or []
        for a in variants:
            ex = dict(a)
            ex["label"] = y
            if _norm_id(bid) is not None:
                ex["id"] = bid
            ex["_asr_base_pos"] = int(i)
            atk_eval_exs.append(ex)
            base_pos_for_each_variant.append(int(i))

    if len(atk_eval_exs) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": int(len(correct_idx)),
            "attack_eval_n": int(len(correct_idx)),
            "attack_variant_n": 0,
            "attack_variant_acc": None,
            "attack_acc": None,
            "asr": None,
        }

    res1 = det.evaluate(atk_eval_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y1 = [int(x) for x in res1.labels]
    p1 = [int(x) for x in res1.preds]

    var_acc = sum(1 for y, p in zip(y1, p1) if y == p) / max(1, len(y1))

    fail_by_base: Dict[int, bool] = {}
    for base_pos, y, p in zip(base_pos_for_each_variant, y1, p1):
        if base_pos not in fail_by_base:
            fail_by_base[base_pos] = False
        if y != p:
            fail_by_base[base_pos] = True

    base_eval_positions = sorted(set(base_pos_for_each_variant))
    base_eval_n = len(base_eval_positions)
    success_n = sum(1 for bp in base_eval_positions if fail_by_base.get(bp, False))
    asr = (success_n / base_eval_n) if base_eval_n > 0 else None
    attack_acc_base = (1.0 - asr) if asr is not None else None

    return {
        "match_mode": mode,
        "align": align_stats,
        "base_correct_n": int(len(correct_idx)),
        "attack_eval_n": int(base_eval_n),
        "attack_variant_n": int(len(atk_eval_exs)),
        "attack_variant_acc": float(var_acc),
        "attack_acc": float(attack_acc_base) if attack_acc_base is not None else None,
        "asr": float(asr) if asr is not None else None,
        "aggregation": "any-success over multiple variants per base (method-specific)",
    }


def _compute_asr_by_method(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
    base_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in atk_exs:
        buckets[_attack_method_name(a)].append(a)

    if base_cache is None:
        base_cache = _precompute_base_correct_cache(
            det,
            base_exs,
            batch_size=batch_size,
            threshold=threshold,
            show_progress=show_progress,
        )

    by_method: Dict[str, Any] = {}
    for m in sorted(buckets.keys()):
        group = buckets[m]
        by_method[m] = _compute_asr_any_success_one_method(
            det,
            base_exs,
            group,
            batch_size=batch_size,
            threshold=threshold,
            show_progress=show_progress,
            base_cache=base_cache,
        )
        by_method[m]["attack_method"] = m
        by_method[m]["attack_n_raw"] = int(len(group))

    return {
        "by_method": by_method,
        "summary": _summarize_asr_attacks(by_method),
        "note": "ASR computed per attack method; base correctness reuses one cached base evaluation.",
        "base_cache": {"base_n": int(base_cache.get("base_n", len(base_exs))), "use_key": bool(base_cache.get("use_key", False))},
    }
