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
from detectors.base import DetectorBase, EvalResult
from detectors.registry import get_detector_cls
from data_utils.load import load_dataset_unified
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Token indices sequence length is longer than the specified maximum sequence length for this model",
    category=UserWarning,
)
# ---------- Optional dependencies ----------
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

# ---- Added: GPU memory stats / env helpers ----
try:
    import torch
except Exception:
    torch = None  # allow import without torch installed


def _bytes_to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)

import math
from typing import Optional, List, Dict, Any, Tuple

def _is_finite_scalar(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False

def _summarize_asr_attacks(attacks_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize ASR for each attack and report means:
      - asr_mean: simple mean (requested)
      - asr_weighted_mean: weighted by attack_eval_n (more reasonable; does not change simple mean)
    """
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

def _filter_nonfinite_examples(
    examples: List[Dict[str, Any]],
    labels: List[int],
    preds: List[int],
    *,
    ranking_vec: Optional[List[float]] = None,
    probs: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], Optional[List[float]], Optional[List[float]], int]:
    """
    Unified filter: drop a sample if ranking or probs is non-finite (None/NaN/Inf).
    Returns (ex, y, pred, ranking, probs, dropped).
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
        * CJK: Han (count each character), Hiragana/Katakana/Hangul (count contiguous runs)
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

def _precompute_base_correct_cache(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Run det.evaluate once on base_exs and cache:
      - correct_mask_by_index: bool list aligned with base_exs (for order fallback)
      - correct_by_key: correctness map keyed by _match_key(base_ex) (for id matching)
    """
    if not base_exs:
        return {"use_key": False, "correct_mask_by_index": [], "correct_by_key": {}}

    res0 = det.evaluate(base_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y0 = [int(x) for x in res0.labels]
    p0 = [int(x) for x in res0.preds]
    correct_mask = [yy == pp for yy, pp in zip(y0, p0)]

    keys = [_match_key(e) for e in base_exs]
    use_key = all(k is not None for k in keys)

    correct_by_key: Dict[str, bool] = {}
    if use_key:
        # If keys repeat, keep the first occurrence (stable enough)
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


def _reset_and_mark_cuda_peaks() -> Dict[str, Any]:
    """
    Before evaluation: reset peak stats for all visible CUDA devices and record device names.
    Returns a context dict for later collection.
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
    After evaluation: read peak memory usage per device (allocated/reserved).
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
# ---- End of GPU memory helpers ----

# ---------- Paths & output organization ----------
def _timestamp() -> str:
    return datetime.now().strftime(f"%Y%m%d-%H%M%S")

def _auto_run_dir(out_dir: Optional[str], detector_display_name: str) -> Path:
    """
    Rules:
      - out_dir not provided: runs_{detector}_{timestamp}/
      - out_dir provided: if the last path segment lacks a timestamp (\\d{8}-\\d{6}),
        append _{timestamp}; if already present, keep as-is.
    """
    ts = _timestamp()
    if out_dir:
        p = Path(out_dir)
        # Only check timestamp in the last path segment
        tail = p.name
        if not re.search(r"\d{8}-\d{6}$", tail):
            p = p.with_name(f"{tail}_{ts}")
    else:
        p = Path(f"results/runs_{detector_display_name}_{ts}")
    # Create standard subdirectories
    p.mkdir(parents=True, exist_ok=True)
    (p / "metrics/curves").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    (p / "artifacts").mkdir(parents=True, exist_ok=True)
    return p

# ---------- Stats helpers ----------
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

from collections import defaultdict
from typing import DefaultDict

def _attack_method_name(e: Dict[str, Any]) -> str:
    """
    Extract attack method name from attack samples. Prefer aug_method (current data format),
    and fall back to attack_method / attack_type.
    """
    for k in ("aug_method", "attack_method", "attack_type"):
        v = e.get(k, None)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return "unknown"


def _align_base_to_attacks_one_to_many(
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], str, Dict[str, Any]]:
    """
    Align base with attacks in a one-to-many manner:
      - keys use existing _match_key (supports {id}-{aug_method}-{hash} normalized to base id)
      - atk_lists: same length as base_aligned; each element is a list of attack variants for that base
    """
    base_keys = [_match_key(e) for e in base_exs]
    atk_keys  = [_match_key(e) for e in atk_exs]

    can_id_match = (
        len(base_exs) > 0 and len(atk_exs) > 0
        and all(x is not None for x in base_keys)
        and all(x is not None for x in atk_keys)
    )

    if not can_id_match:
        # Fallback: order alignment (degenerates to one-to-one, drops extra variants)
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

    # Alignment stats
    total_variants = sum(len(vs) for vs in atk_lists)
    uniq_keys = len(amap)
    stats = {
        "base_n": len(base_exs),
        "atk_n": len(atk_exs),
        "matched_n": len(base_aligned),
        "missing_in_attack": miss,
        "attack_keys_unique": int(uniq_keys),
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
    base_cache: Optional[Dict[str, Any]] = None,   # ✅ NEW
) -> Dict[str, Any]:
    """
    Compute ASR for a single attack method (supports one-to-many):
      - First filter base samples that were correctly predicted before attack
      - For each base, evaluate all attack variants for this method
      - If any variant flips the prediction => attack succeeds for that base (any-success)
    Returns:
      - asr: any-success definition (one text with multiple attack variants)
      - attack_variant_acc: variant-level accuracy (for analysis only)
    """
    base_aligned, atk_lists, mode, align_stats = _align_base_to_attacks_one_to_many(base_exs, atk_exs_same_method)
    if len(base_aligned) == 0:
        return {
            "match_mode": mode,
            "align": align_stats,
            "base_correct_n": 0,
            "attack_eval_n": 0,
            "attack_variant_n": 0,
            "attack_variant_acc": None,
            "attack_acc": None,  # compatibility field (same as 1 - asr here)
            "asr": None,
        }

    # 1) Pre-attack: keep base samples predicted correctly (reuse base_cache if available)
    correct_idx: List[int] = []

    if base_cache is not None:
        if (mode.startswith("id") or mode.startswith("id_one_to_many")) and base_cache.get("use_key", False):
            correct_by_key = base_cache.get("correct_by_key", {})
            for i, b in enumerate(base_aligned):
                k = _match_key(b)
                if k is not None and bool(correct_by_key.get(k, False)):
                    correct_idx.append(i)
        else:
            # Order fallback: base_aligned == base_exs[:n]; indices align to base_cache mask
            mask = base_cache.get("correct_mask_by_index", [])
            for i in range(len(base_aligned)):
                if i < len(mask) and bool(mask[i]):
                    correct_idx.append(i)
    else:
        # Original behavior: re-evaluate base per method (slow)
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


    # 2) Build attack eval set: expand all variants for each correct base
    atk_eval_exs: List[Dict[str, Any]] = []
    base_pos_for_each_variant: List[int] = []  # which base_aligned index each variant belongs to

    for i in correct_idx:
        b = base_aligned[i]
        y = int(b.get("label", 1))
        bid = b.get("id", None)

        variants = atk_lists[i] or []
        for a in variants:
            ex = dict(a)
            ex["label"] = y
            if _norm_id(bid) is not None:
                ex["id"] = bid  # normalize to base id for auditing
            # Extra fields are for aggregation only; detector uses text/label only
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

    # 3) Post-attack: evaluate all variants
    res1 = det.evaluate(atk_eval_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y1 = [int(x) for x in res1.labels]
    p1 = [int(x) for x in res1.preds]

    # Variant-level accuracy
    var_acc = sum(1 for y, p in zip(y1, p1) if y == p) / max(1, len(y1))

    # Base-level any-success: if any variant is wrong => success
    fail_by_base: Dict[int, bool] = {}
    for base_pos, y, p in zip(base_pos_for_each_variant, y1, p1):
        if base_pos not in fail_by_base:
            fail_by_base[base_pos] = False
        if y != p:
            fail_by_base[base_pos] = True

    # Only count bases that participated in attack eval (correct_idx and has variants)
    base_eval_positions = sorted(set(base_pos_for_each_variant))
    base_eval_n = len(base_eval_positions)
    success_n = sum(1 for bp in base_eval_positions if fail_by_base.get(bp, False))

    asr = (success_n / base_eval_n) if base_eval_n > 0 else None
    attack_acc_base = (1.0 - asr) if asr is not None else None

    return {
        "match_mode": mode,
        "align": align_stats,
        "base_correct_n": int(len(correct_idx)),
        "attack_eval_n": int(base_eval_n),              # base-level eval sample count
        "attack_variant_n": int(len(atk_eval_exs)),     # total variants
        "attack_variant_acc": float(var_acc),
        "attack_acc": float(attack_acc_base) if attack_acc_base is not None else None,
        "asr": float(asr) if asr is not None else None,
        "aggregation": "any-success over multiple variants per base (method-specific)",
    }

def _base_correct_cache_from_preds(
    base_exs: List[Dict[str, Any]],
    labels: List[int],
    preds: List[int],
) -> Dict[str, Any]:
    """
    Build base_correct cache from precomputed labels/preds to avoid re-running det.evaluate(base).
    Supports:
      - id matching: correct_by_key
      - order matching: correct_mask_by_index
    """
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

def _compute_asr_by_method(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
    base_cache: Optional[Dict[str, Any]] = None,   # ✅ NEW (allow external reuse)
) -> Dict[str, Any]:
    """
    Group samples in the same attack file by attack method and compute ASR per method.
    base_correct det.evaluate runs only once (cache reuse).
    """
    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in atk_exs:
        m = _attack_method_name(a)
        buckets[m].append(a)

    # ✅ NEW: compute base cache once
    if base_cache is None:
        base_cache = _precompute_base_correct_cache(
            det, base_exs,
            batch_size=batch_size, threshold=threshold, show_progress=show_progress
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
            base_cache=base_cache,  # ✅ NEW
        )
        by_method[m]["attack_method"] = m
        by_method[m]["attack_n_raw"] = int(len(group))

    return {
        "by_method": by_method,
        "summary": _summarize_asr_attacks(by_method),
        "note": "ASR computed per attack method; base correctness reuses one cached base evaluation.",
        "base_cache": {
            "base_n": int(base_cache.get("base_n", len(base_exs))),
            "use_key": bool(base_cache.get("use_key", False)),
        },
    }

def _tpr_at_fpr_points(
    labels: Sequence[int],
    scores: Sequence[float],
    fpr_targets: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
) -> Dict[str, Any]:
    """
    Pick multiple operating points from scores (higher = more AI-like):
    maximize TPR subject to FPR<=target.
    Returns threshold / tpr / fpr / confusion / acc / f1 for each target.
    """
    ys = [int(y) for y in labels]
    ss = [float(s) for s in scores]
    n = len(ys)
    if n == 0:
        return {}

    P = sum(1 for y in ys if y == 1)
    N = n - P
    if P == 0 or N == 0:
        # Single-class: FPR/TPR undefined
        out = {}
        for t in fpr_targets:
            out[str(t)] = {"threshold": None, "tpr": None, "fpr": None, "confusion": None, "acc": None, "f1": None}
        return out

    # sort by score desc, stable for ties
    order = sorted(range(n), key=lambda i: (-ss[i], i))

    # Scan threshold points grouped by score: predict positive iff score >= thr
    points = []
    tp = fp = 0

    # Initial point: thr=+inf -> no positives
    points.append({
        "threshold": float("inf"),
        "tp": 0, "fp": 0,
        "tn": N, "fn": P,
        "tpr": 0.0,
        "fpr": 0.0,
    })

    i = 0
    while i < n:
        thr = ss[order[i]]
        # consume all with same score
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
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": float(tpr),
            "fpr": float(fpr),
        })
        i = j

    # For each target: pick the last point with fpr <= target (max tpr)
    out: Dict[str, Any] = {}
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


# ---------- Curves & metrics ----------
def _roc_curve(labels: Sequence[int], probs_or_scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    # Sort descending — accepts probs or scores, used only for ranking
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
            # Trapezoid integration
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
    # Sort descending
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
    # AUPR (integrate over recall)
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
    Risk (error rate) under different coverage levels:
      - confidence conf = max(p, 1-p)
      - include samples in descending conf; coverage=k/N; risk = error rate (1-acc) on the included subset
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

# ---------- New: ranking vector selector (prefer scores) ----------
def _get_ranking_values(res) -> Tuple[List[float], str]:
    """
    Return a 1D vector for ROC/PR ranking:
      - prefer res.scores (raw scores; higher = more AI-like)
      - otherwise fall back to res.probs
    Also returns source tag: 'scores' / 'probs' / 'none'
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
        # Basic metrics
        st = _basic_stats(yb, preds, pb)
        accs.append(st["acc"])
        tp = st["confusion"]["tp"]; fp = st["confusion"]["fp"]; fn = st["confusion"]["fn"]
        f1s.append(_f1_from_counts(tp, fp, fn))
        # Curves
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

# ---------- Runtime info & manifest ----------
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

def _norm_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_builder_samples(rec: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Compatible with different builder versions:
      - common names: sample / samples / pair / pairs
      - list elements must be dicts
    Returns (samples, key_name)
    """
    for key in ("sample", "samples", "pair", "pairs"):
        v = rec.get(key, None)
        if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            return v, key
    return None, None


def _extract_text(obj: Dict[str, Any]) -> str:
    """
    Compatible text extraction across field names: text/content/response/output, etc.
    """
    for k in ("text", "content", "response", "output", "generation", "gen", "final_text"):
        if k in obj and obj.get(k) is not None:
            s = str(obj.get(k) or "")
            return s.strip()
    return ""


def _extract_role(obj: Dict[str, Any]) -> Optional[str]:
    """
    Compatible role fields: role/source/type
    """
    for k in ("role", "source", "type"):
        if k in obj and obj.get(k) is not None:
            s = str(obj.get(k)).strip()
            return s if s else None
    return None


def _looks_like_builder_record(rec: Dict[str, Any]) -> bool:
    samples, _ = _extract_builder_samples(rec)
    if not (isinstance(samples, list) and len(samples) >= 1 and isinstance(samples[0], dict)):
        return False
    # If we can extract non-empty text from the first sample element, treat as builder format
    return bool(_extract_text(samples[0]))

def _infer_label_from_role(role: Any, default: int = 1) -> int:
    r = str(role or "").strip().lower()
    if r == "machine":
        return 1
    if r == "human":
        return 0
    return int(default)


def _load_examples_auto(
    dataset_spec: Union[str, Iterable[Dict[str, Any]]],
    *,
    sample_k: Optional[int],
    sample_seed: int,
    group_cols: Optional[Sequence[str]],
    # builder_view:
    #   - "flat": use load_dataset_unified directly (regular dataset)
    #   - "pre":  take sample[0] from builder record
    #   - "post": take sample[-1] from builder record (requires sample>=2, else skip)
    builder_view: str = "flat",
) -> List[Dict[str, Any]]:
    # Iterable goes through the unified loader
    if not isinstance(dataset_spec, str):
        exs, _ = load_dataset_unified(
            dataset=dataset_spec,
            sample_k=sample_k,
            sample_seed=sample_seed,
            group_cols=group_cols,
        )
        return exs

    p = str(dataset_spec)
    if (p.endswith(".jsonl") or p.endswith(".json")) and os.path.exists(p):
        # Try to detect builder jsonl first
        try:
            # Read only a few lines for detection (avoid large-file overhead)
            with open(p, "r", encoding="utf-8") as f:
                head = []
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        head.append(json.loads(line))
            is_builder = any(isinstance(r, dict) and _looks_like_builder_record(r) for r in head)
        except Exception:
            is_builder = False

        if is_builder and builder_view in ("pre", "post"):
            recs = _read_jsonl(p)
            out: List[Dict[str, Any]] = []
            for i, r in enumerate(recs):
                if not isinstance(r, dict) or (not _looks_like_builder_record(r)):
                    continue
                samples, _k = _extract_builder_samples(r)
                if not samples:
                    continue

                if builder_view == "post" and len(samples) < 2:
                    continue

                obj = samples[0] if builder_view == "pre" else samples[-1]
                text = _extract_text(obj)
                if not text:
                    continue

                rid = (
                    _norm_id(r.get("id")) or _norm_id(r.get("record_id")) or _norm_id(r.get("qid"))
                    or _norm_id(r.get("question_id")) or str(i)
                )

                # label: prefer sample obj role/type/source; else r["label"]; else default 1
                if "label" in r and isinstance(r.get("label"), (int, float, str)):
                    try:
                        lb_default = int(r.get("label"))
                    except Exception:
                        lb_default = 1
                else:
                    lb_default = 1

                role = _extract_role(obj)
                label = _infer_label_from_role(role, default=lb_default)

                ex: Dict[str, Any] = {"id": rid, "text": text, "label": int(label)}

                # Keep common grouping fields where possible: first rec, then obj
                for k in ("lang", "source", "sub_source", "model"):
                    if k in r and r.get(k) is not None:
                        ex[k] = r.get(k)
                    elif isinstance(obj, dict) and (k in obj) and (obj.get(k) is not None):
                        ex[k] = obj.get(k)

                out.append(ex)

            # builder_view does not apply sample_k (original sample_k is in load_dataset_unified)
            # If needed, resample here by seed for builder_view as well.
            return out

    # Fallback: regular dataset via unified loader (supports sample_k)
    exs, _ = load_dataset_unified(
        dataset=dataset_spec,
        sample_k=sample_k,
        sample_seed=sample_seed,
        group_cols=group_cols,
    )
    return exs


def _match_key(e: Dict[str, Any]) -> Optional[str]:
    # 1) If attack file explicitly stores original id, prefer it (most stable)
    for k in ("orig_id", "base_id", "source_id"):
        v = _norm_id(e.get(k))
        if v is not None:
            return v

    # 2) Otherwise use id and try to strip suffix by aug_method: {id}-{aug_method}-{hash}
    sid = _norm_id(e.get("id"))
    if sid is None:
        return None

    am = str(e.get("aug_method") or "").strip()
    if am:
        token = f"-{am}-"
        if token in sid:
            # Only strip the last suffix occurrence to avoid false matches
            return sid.rsplit(token, 1)[0]

    return sid


def _align_pairs(base_exs, atk_exs):
    base_keys = [_match_key(e) for e in base_exs]
    atk_keys  = [_match_key(e) for e in atk_exs]

    can_id_match = (
        len(base_exs) > 0 and len(atk_exs) > 0
        and all(x is not None for x in base_keys)
        and all(x is not None for x in atk_keys)
    )

    if can_id_match:
        amap = {}
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

    # Fallback order matching
    n = min(len(base_exs), len(atk_exs))
    stats = {"base_n": len(base_exs), "atk_n": len(atk_exs), "matched_n": n}
    return base_exs[:n], atk_exs[:n], "order", stats


def _compute_asr(
    det: DetectorBase,
    base_exs: List[Dict[str, Any]],
    atk_exs: List[Dict[str, Any]],
    *,
    batch_size: int,
    threshold: float,
    show_progress: bool,
    base_cache: Optional[Dict[str, Any]] = None,   # ✅ NEW
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

    # 1) Pre-attack: keep correctly predicted samples (reuse base_cache if possible)
    correct_idx: List[int] = []

    if base_cache is not None:
        if mode == "id" and base_cache.get("use_key", False):
            correct_by_key = base_cache.get("correct_by_key", {})
            for i, b in enumerate(base_aligned):
                k = _match_key(b)
                if k is not None and bool(correct_by_key.get(k, False)):
                    correct_idx.append(i)
        else:
            # Order fallback: base_aligned == base_exs[:n]; indices align to base_cache mask
            mask = base_cache.get("correct_mask_by_index", [])
            for i in range(len(base_aligned)):
                if i < len(mask) and bool(mask[i]):
                    correct_idx.append(i)
    else:
        # Original behavior: caused the second 63/63 you observed
        res0 = det.evaluate(base_aligned, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y0 = list(int(x) for x in res0.labels)
        p0 = list(int(x) for x in res0.preds)
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

    # 2) Post-attack: evaluate only on correct_subset
    atk_subset = []
    for i in correct_idx:
        b = base_aligned[i]
        a = atk_aligned[i]
        ex = dict(a)
        ex["label"] = int(b.get("label", 1))
        if _norm_id(b.get("id")) is not None:
            ex["id"] = b.get("id")
        atk_subset.append(ex)

    res1 = det.evaluate(atk_subset, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    y1 = list(int(x) for x in res1.labels)
    p1 = list(int(x) for x in res1.preds)
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


def evaluate_detector(
    detector: Union[str, DetectorBase],
    dataset: Union[str, Iterable[Dict[str, Any]]],
    batch_size: int = 8,
    threshold: float = 0.5,
    fpr_targets: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    # Grouping columns (auto-detect if not specified, e.g., lang/source/model/sub_source)
    group_cols: Optional[Sequence[str]] = None,
    # Output dir: defaults to runs_{det_name}_{timestamp}/
    out_dir: Optional[str] = None,
    out_prefix: Optional[str] = None,   # kept for compatibility (no longer used as file prefix)
    save_curves: bool = True,           # compatibility switch (plots/JSON curves on by default)
    # CI config
    ci_enable: Optional[bool] = None,   # None: auto (only when sampling); True/False: force
    ci_iters: int = 200,
    ci_seed: int = 114514,
    show_progress: bool = True,      # NEW: show progress during evaluation
    # --- NEW: repeated random sampling runs ---
    k_runs: int = 1,                 # effective when sample_k>0; run k seeds and aggregate stats
    # Other detector params
    # --- NEW: ASR (Attack Success Rate) ---
    attack_datasets: Optional[Union[str, Sequence[str]]] = None,  # allow one or multiple attack files
    asr_save_details: bool = True,  # save detailed asr.json
    **detector_kwargs,
) -> EvalResult:
    """
    New features:
      - k_runs: when sample_k>0 (sampled eval) and k_runs>1, use a ±10 seed sequence
        around sample_seed (cycle if needed) and write mean/std/min/max to metrics/summary.json.
      - Full eval (sample_k not set or <=0): k_runs ignored, same behavior as before.
    """
    # 0) Initialize detector
    if isinstance(detector, str):
        Det = get_detector_cls(detector)
        det = Det(**detector_kwargs)
    else:
        det = detector
        for k, v in detector_kwargs.items():
            setattr(det, k, v)

    display_name = getattr(det, "name", getattr(det, "DETECTOR_NAME", "detector"))

    # ----------- Tools: aggregation -----------
    def _stat_pack(vals: List[float]) -> Dict[str, Any]:
        vals = [float(x) for x in vals if x is not None]
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        m = sum(vals) / n
        if n > 1:
            var = sum((x - m) ** 2 for x in vals) / (n - 1)  # sample variance
            sd = math.sqrt(var)
        else:
            sd = 0.0
        return {"n": n, "mean": m, "std": sd, "min": min(vals), "max": max(vals)}

    # ----------- Branching for repeated sampling runs -----------
    is_sampling = (sample_k is not None and sample_k > 0)
    multi_run = (is_sampling and (k_runs is not None) and (int(k_runs) > 1))

    # Create run_dir and run-config early
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
            "attack_datasets": (
                [attack_datasets] if isinstance(attack_datasets, str)
                else (list(attack_datasets) if attack_datasets is not None else None)
            ),
            "asr_save_details": bool(asr_save_details),
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

    # =============== Repeated random sampling evaluation ===============
    if multi_run:
        # Build seed sequence: ±10 offsets around sample_seed (21 total), cycle if shorter
        base = int(sample_seed)
        offsets = list(range(-10, 11))  # [-10, ..., 0, ..., +10]
        k = int(k_runs)
        seeds = [base + offsets[i % len(offsets)] for i in range(k)]

        # Resources/dirs
        metrics_dir = run_dir / "metrics"
        curves_dir = metrics_dir / "curves"
        figures_dir = run_dir / "figures"
        k_runs_dir = metrics_dir / "k_runs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        k_runs_dir.mkdir(parents=True, exist_ok=True)

        # Write seeds
        with open(k_runs_dir / "seeds.json", "w", encoding="utf-8") as f:
            json.dump({"base": base, "seeds": seeds}, f, ensure_ascii=False, indent=2)

        # Aggregation containers
        acc_list, auroc_list, aupr_list, f1_list = [], [], [], []
        ece_list, brier_list = [], []
        eval_secs, load_secs = [], []
        per_run_briefs = []

        # Save detailed artifacts for the first run (curves/plots/preds) for reproducibility
        first_res: Optional[EvalResult] = None
        first_examples: Optional[List[Dict[str, Any]]] = None
        first_used_group_cols: List[str] = []
        first_probs_seq: Optional[List[float]] = None
        first_ranking_vec: Optional[List[float]] = None
        first_ranking_src: str = "none"

        # Run k times
        for ridx, seed in enumerate(seeds):
            # 1) Load sampling
            t0 = time.perf_counter()
            proc0 = _proc_snapshot()
            examples, used_group_cols = load_dataset_unified(
                dataset=dataset,
                sample_k=sample_k,
                sample_seed=int(seed),
                group_cols=group_cols,
            )
            load_time = time.perf_counter() - t0

            # 2) GPU peak stats (reset per run)
            cuda_ctx = _reset_and_mark_cuda_peaks()

            # 3) Evaluate
            t1 = time.perf_counter()
            res_i = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
            eval_time = time.perf_counter() - t1

            # 4) Metrics computation (overall)
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

            # ROC / PR use ranking_vec
            fpr, tpr, auroc = _roc_curve(labels_seq, ranking_vec)
            rec, prec, aupr = _pr_curve(labels_seq, ranking_vec)
            tpr_at_fpr = _tpr_at_fpr_points(labels_seq, ranking_vec, fpr_targets=fpr_targets)

            # Calibration/RC only when probabilities exist
            if probs_seq is not None:
                calib = _ece_brier(labels_seq, probs_seq, bins=10)
                cov, risk = _risk_coverage(labels_seq, probs_seq)
            else:
                calib = {"ece": None, "brier": None, "bins": []}
                cov, risk = ([], [])

            # Record aggregation
            acc_list.append(overall_basic["acc"])
            tp = overall_basic["confusion"]["tp"]; fp = overall_basic["confusion"]["fp"]; fn = overall_basic["confusion"]["fn"]
            f1_list.append(_f1_from_counts(tp, fp, fn))
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            ece_list.append(calib["ece"])
            brier_list.append(calib["brier"])
            load_secs.append(load_time)
            eval_secs.append(eval_time)

            # Single-run summary
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

            # Save artifacts for the first run only; other runs keep stats to avoid duplicates
            if ridx == 0:
                first_res = res_i
                first_examples = examples
                first_used_group_cols = list(used_group_cols)
                first_probs_seq = probs_seq
                first_ranking_vec = ranking_vec
                first_ranking_src = ranking_src

                # GPU/process stats
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
                # ★★★ New: first-run meta enrichment + persist (audit)
                meta_first = None
                try:
                    if hasattr(first_res, "meta") and isinstance(first_res.meta, dict):
                        meta_first = dict(first_res.meta)  # shallow copy
                        # Keep consistent with single-run branch: fill timing/memory for standalone viewing
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
                # —— First run: grouped files + curves + predictions + model card (aligned with original) ——
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
                            # ranking subset
                            rv = [first_ranking_vec[i] for i in idxs]
                            fpr_l, tpr_l, auroc_l = _roc_curve(yl, rv)
                            rec_l, prec_l, aupr_l = _pr_curve(yl, rv)
                            with open(curves_dir / f"roc_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"fpr": fpr_l, "tpr": tpr_l, "auroc": auroc_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            with open(curves_dir / f"pr_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"recall": rec_l, "precision": prec_l, "aupr": aupr_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            _plot_curve(fpr_l, tpr_l, f"ROC ({lg}) AUC={auroc_l:.3f}", "FPR", "TPR", figures_dir / f"roc_{lg}.png")
                            _plot_curve(rec_l, prec_l, f"PR ({lg}) AUPR={aupr_l:.3f}", "Recall", "Precision", figures_dir / f"pr_{lg}.png")
                        # RC / Reliability only when probabilities exist
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

                # overall curves/plots (ROC/PR use ranking; RC/Calibration only when probs exist)
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

                # predictions (first run only; truncate text to 10 words to reduce size)
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

                # Model card
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

        # —— Cross-run stats and summary.json —— #
        # Bootstrap CI only when probabilities exist
        do_ci = (ci_enable if ci_enable is not None else True)  # default: CI once for first run in multi-sample mode
        ci = None
        if do_ci and first_res is not None and hasattr(first_res, "probs") and (first_res.probs is not None) and len(first_res.labels) > 5:
            ci = _bootstrap_ci(first_res.labels, first_res.probs, iters=int(ci_iters), seed=int(ci_seed))
        tpr_at_fpr_first = (
            _tpr_at_fpr_points(first_res.labels, first_ranking_vec, fpr_targets=fpr_targets)
            if (first_res is not None and first_ranking_vec is not None)
            else None
        )
        # NEW (multi-run): ASR only for first run (optional)
        asr_results = None
        atk_specs: List[str] = []
        if attack_datasets is not None:
            atk_specs = [attack_datasets] if isinstance(attack_datasets, str) else list(attack_datasets)

        if atk_specs and (first_examples is not None):
            base_for_asr = first_examples
            attacks_out: Dict[str, Any] = {}
            for atk_path in atk_specs:
                atk_key = Path(str(atk_path)).stem
                atk_exs = _load_examples_auto(
                    atk_path, sample_k=None, sample_seed=base, group_cols=group_cols, builder_view="post"
                )
                if not atk_exs:
                    atk_exs = _load_examples_auto(
                        atk_path, sample_k=None, sample_seed=base, group_cols=group_cols, builder_view="flat"
                    )
            has_method = any(isinstance(x, dict) and _attack_method_name(x) != "unknown" for x in atk_exs)

            if has_method:
                attacks_out[atk_key] = _compute_asr_by_method(
                    det,
                    base_for_asr,
                    atk_exs,
                    batch_size=batch_size,
                    threshold=threshold,
                    show_progress=show_progress,
                )
            else:
                attacks_out[atk_key] = _compute_asr(
                    det,
                    base_for_asr,
                    atk_exs,
                    batch_size=batch_size,
                    threshold=threshold,
                    show_progress=show_progress,
                )

            attacks_out[atk_key]["attack_dataset"] = str(atk_path)

            asr_results = {
                "definition": "ASR = 1 - Acc(attack | correct_before_attack)",
                "base_dataset": (str(dataset) if isinstance(dataset, str) else "iterable"),
                "base_used_n": len(base_for_asr),
                "attacks": attacks_out,
            }
            asr_results["summary"] = _summarize_asr_attacks(attacks_out)

            if asr_save_details:
                try:
                    with open(run_dir / "metrics" / "asr.json", "w", encoding="utf-8") as f:
                        json.dump(asr_results, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass


        # Persist (optional)
        if asr_results is not None and asr_save_details:
            try:
                with open(metrics_dir / "asr.json", "w", encoding="utf-8") as f:
                    json.dump(asr_results, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Summary stats
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
            "tpr_at_fpr_first_run": tpr_at_fpr_first,
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
            "asr": asr_results,
        }
        with open(run_dir / "metrics" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[MGTEval] (multi-run) results saved to: {str(run_dir)}")
        return first_res  # return first EvalResult for backward compatibility

    # =============== Single-run evaluation (original logic, ranking/probs aware) ===============
    # 1) Load data and determine grouping columns
    t0 = time.perf_counter()
    proc0 = _proc_snapshot()
    examples, used_group_cols = load_dataset_unified(
        dataset=dataset,
        sample_k=sample_k,
        sample_seed=sample_seed,
        group_cols=group_cols,
    )
    load_time = time.perf_counter() - t0

    # 2) Write back config (single run)
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

    # 3) Reset GPU peak stats before evaluation
    cuda_ctx = _reset_and_mark_cuda_peaks()

    # 4) Evaluate (consistent order)
    t1 = time.perf_counter()
    res = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    eval_time = time.perf_counter() - t1

    # 5) Runtime resource stats
    mem_stats = _collect_cuda_peaks(cuda_ctx)
    proc1 = _proc_snapshot()

    # 6) Write back meta
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

    # ★ Filter non-finite values (NaN/Inf/None)
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

    # 8) Other run-manifest items
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

    # 9) Aggregate metrics, CI, calibration, curves
    # Basic overall
    overall_basic = _basic_stats(labels_seq, preds_seq, probs_seq)
    # ROC/PR use ranking_vec
    fpr, tpr, auroc = _roc_curve(labels_seq, ranking_vec)
    rec, prec, aupr = _pr_curve(labels_seq, ranking_vec)
    tpr_at_fpr = _tpr_at_fpr_points(labels_seq, ranking_vec, fpr_targets=fpr_targets)

    # Calibration (ECE/Brier): only when probabilities exist
    if probs_seq is not None:
        calib = _ece_brier(labels_seq, probs_seq, bins=10)
        cov, risk = _risk_coverage(labels_seq, probs_seq)
    else:
        calib = {"ece": None, "brier": None, "bins": []}
        cov, risk = ([], [])

    # CI: default only for sampled eval and only when probabilities exist
    do_ci = (ci_enable if ci_enable is not None else (sample_k is not None and sample_k > 0))
    ci = None
    if do_ci and (probs_seq is not None) and len(labels_seq) > 5:
        ci = _bootstrap_ci(labels_seq, probs_seq, iters=int(ci_iters), seed=int(ci_seed))

    # 10) Save metrics summary (metrics/summary.json)
    metrics_dir = run_dir / "metrics"
    curves_dir = metrics_dir / "curves"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # NEW: compute metrics across multiple threshold fields
    threshold_metrics: Dict[str, Any] = {}
    if probs_seq is not None:
        # 1) Initial set: current threshold & 0.5 baseline
        thr_map: Dict[str, float] = {"eval": float(threshold)}
        if abs(float(threshold) - 0.5) > 1e-9:
            thr_map["p0.5"] = 0.5

        # 2) Calibration thresholds parsed from detector (DetectorBase.load_calibrator)
        calib_thrs = getattr(det, "_calibrator_thresholds", None)
        if isinstance(calib_thrs, dict):
            for name, v in calib_thrs.items():
                if isinstance(v, (int, float)):
                    thr_map[str(name)] = float(v)

        # 3) Deduplicate thresholds by value
        used: Dict[str, float] = {}
        for name, tv in thr_map.items():
            if any(abs(tv - vv) < 1e-9 for vv in used.values()):
                continue
            used[name] = tv

        # 4) Compute acc / f1 / tpr / fpr per threshold
        for name, tv in used.items():
            preds_thr = [1 if p >= tv else 0 for p in probs_seq]
            st = _basic_stats(labels_seq, preds_thr, probs_seq)
            tp = st["confusion"]["tp"]; tn = st["confusion"]["tn"]
            fp = st["confusion"]["fp"]; fn = st["confusion"]["fn"]
            P = tp + fn; N = tn + fp
            thr_rec = {
                "threshold": tv,
                "acc": st["acc"],
                "confusion": st["confusion"],
                "f1": _f1_from_counts(tp, fp, fn),
                "tpr": (tp / P) if P > 0 else None,
                "fpr": (fp / N) if N > 0 else None,
            }
            threshold_metrics[name] = thr_rec
    base_cache = _base_correct_cache_from_preds(examples, labels_seq, preds_seq)

    # =========================
    # NEW: ASR (Attack Success Rate)
    # =========================
    asr_results = None
    atk_specs: List[str] = []
    if attack_datasets is not None:
        atk_specs = [attack_datasets] if isinstance(attack_datasets, str) else list(attack_datasets)

    # auto: if attack_datasets not provided but dataset is builder paired (sample>=2), compute ASR
    auto_paired_asr = False
    if (not atk_specs) and isinstance(dataset, str) and (dataset.endswith(".jsonl") or dataset.endswith(".json")) and os.path.exists(dataset):
        try:
            with open(dataset, "r", encoding="utf-8") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if isinstance(r, dict) and _looks_like_builder_record(r) and isinstance(r.get("sample"), list) and len(r["sample"]) >= 2:
                        auto_paired_asr = True
                        break
        except Exception:
            auto_paired_asr = False

    if auto_paired_asr:
        # dataset contains pre/post: pre=sample[0], post=sample[-1]
        base_for_asr = _load_examples_auto(dataset, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="pre")
        atk_for_asr  = _load_examples_auto(dataset, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="post")
        asr_results = {
            "definition": "ASR = 1 - Acc(attack | correct_before_attack)",
            "auto_paired_dataset": str(dataset),
            "attacks": {
                "paired": _compute_asr(
                    det, base_for_asr, atk_for_asr,
                    batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                )
            },
        }
        asr_results["summary"] = _summarize_asr_attacks(asr_results["attacks"])

    elif atk_specs:
        # base uses current evaluation examples (aligned with sample_k/seed), attack aligned from file
        base_for_asr = examples  # current eval examples (not full raw file)
        attacks_out: Dict[str, Any] = {}

        for atk_path in atk_specs:
            atk_key = Path(str(atk_path)).stem

            # Try builder post; fallback to flat
            atk_exs = _load_examples_auto(
                atk_path,
                sample_k=None,                # no sampling on attack set; align to base subset by id/order
                sample_seed=sample_seed,
                group_cols=group_cols,
                builder_view="post",
            )
            if not atk_exs:
                atk_exs = _load_examples_auto(
                    atk_path,
                    sample_k=None,
                    sample_seed=sample_seed,
                    group_cols=group_cols,
                    builder_view="flat",
                )

            # If attack file has aug_method (or attack_method/type), group by method (supports one-to-many)
            has_method = any(isinstance(x, dict) and _attack_method_name(x) != "unknown" for x in atk_exs)

            if has_method:
                attacks_out[atk_key] = _compute_asr_by_method(
                    det,
                    base_for_asr,
                    atk_exs,
                    batch_size=batch_size,
                    threshold=threshold,
                    show_progress=show_progress,
                    base_cache=base_cache,   # ✅ NEW
                )
            else:
                attacks_out[atk_key] = _compute_asr(
                    det,
                    base_for_asr,
                    atk_exs,
                    batch_size=batch_size,
                    threshold=threshold,
                    show_progress=show_progress,
                    base_cache=base_cache,   # ✅ NEW
                )


            attacks_out[atk_key]["attack_dataset"] = str(atk_path)

        asr_results = {
            "definition": "ASR = 1 - Acc(attack | correct_before_attack)",
            "base_dataset": (str(dataset) if isinstance(dataset, str) else "iterable"),
            "base_used_n": len(base_for_asr),
            "attacks": attacks_out,
        }
        asr_results["summary"] = _summarize_asr_attacks(attacks_out)

    # Persist (optional)
    if asr_results is not None and asr_save_details:
        try:
            with open(metrics_dir / "asr.json", "w", encoding="utf-8") as f:
                json.dump(asr_results, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Overall summary (single run)
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
        # ✅ NEW：TPR@FPR
        "tpr_at_fpr_targets": [float(x) for x in fpr_targets],
        "tpr_at_fpr_rank_source": ranking_src,
        "tpr_at_fpr": tpr_at_fpr,
        "ci_95": ci,  # may be None
        "timing": manifest["timing"],
        "memory": mem_stats,
        "k_runs": 1,                # explicitly single run
        "k_runs_stats": None,       # no cross-run stats for single run
        "k_runs_seeds": [sample_seed] if is_sampling else None,
        "meta": (res.meta if hasattr(res, "meta") else None),
        "asr": asr_results,
        # NEW: metrics summary across threshold fields (eval / p0.5 / decision / acc / f1 / tpr@fpr<=0.01, etc.)
        "threshold_metrics": threshold_metrics or None,
    }
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(metrics_dir / "tpr_at_fpr.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "targets": [float(x) for x in fpr_targets],
                "rank_source": ranking_src,
                "tpr_at_fpr": tpr_at_fpr,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # 11) Grouped metrics files
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

            # RC / Reliability only when probabilities exist
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

    # 12) Overall curves and plots
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

    # 13) Export all predictions (predictions.json)
    preds_out = []
    for i, ex in enumerate(examples):
        raw_text = str(ex.get("text", "") or "").strip()
        words = raw_text.split()  # whitespace tokenization
        text_preview = " ".join(words[:10]) if words else ""
        rec = {
            "text": text_preview,                  # keep first 10 words only
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

    # 14) Optional model card (if detector exposes info)
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

    # 15) Legacy print: simple hint
    print(f"[MGTEval] results saved to: {str(run_dir)}")
    return res
