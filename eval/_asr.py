from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from ..detectors.base import DetectorBase
from ._utils_common import _norm_id
from ._utils_loader import _load_examples_auto


def _attack_method_name(e: Dict[str, Any]) -> str:
    """
    兼容多种输入形态：
      1) 变体样本（flatten 后）：attack / attack_short / aug_method ...
      2) 原始 record：meta.active_attack 或 sample[0].attack
      3) 兜底：从 id 中解析 -ta_xxx-（如 ...-ta_tran-...）
    并尽量统一成 text_* 命名（与你 builder 的 meta.active_attack 对齐）。
    """
    # 1) 直接字段（flatten 样本最常见）
    for k in ("attack", "attack_full", "attack_name", "aug_method", "attack_method", "attack_type", "attack_short"):
        v = e.get(k, None)
        if v is not None:
            s = str(v).strip()
            if s:
                # 如果是短名（tran/subs/...），尽量补 text_ 前缀
                if not s.startswith("text_"):
                    meta = e.get("meta")
                    if isinstance(meta, dict):
                        aa = str(meta.get("active_attack") or "").strip()
                        if aa.startswith("text_") and aa.lower().endswith(s.lower()):
                            return aa
                        if isinstance(meta.get("text_attack_meta", None), list):
                            return "text_" + s
                    # flatten 后通常没有 meta，但你也可以接受短名
                return s

    # 2) record-level: meta.active_attack
    meta = e.get("meta")
    if isinstance(meta, dict):
        aa = meta.get("active_attack", None)
        if aa is not None:
            s = str(aa).strip()
            if s:
                return s

    # 3) record-level: sample[0].attack（你给的 jsonl 就是这种）
    smp = e.get("sample")
    if isinstance(smp, list) and smp:
        obj = smp[0]
        if isinstance(obj, dict):
            v = obj.get("attack") or obj.get("aug_method") or obj.get("attack_method") or obj.get("attack_type")
            if v is not None:
                s = str(v).strip()
                if s:
                    if not s.startswith("text_"):
                        if isinstance(meta, dict) and str(meta.get("active_attack") or "").strip().startswith("text_"):
                            return "text_" + s
                    return s

    # 4) 兜底：从 id 里解析 -ta_xxx-
    sid = str(e.get("id") or "").strip()
    if sid:
        token = "-ta_"
        j = sid.find(token)
        if j >= 0:
            j2 = sid.find("-", j + len(token))
            frag = sid[j + len(token): (j2 if j2 >= 0 else len(sid))].strip()
            if frag:
                # 解析到的是短名，默认补 text_，更符合你 meta.active_attack 的风格
                return frag if frag.startswith("text_") else ("text_" + frag)

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

# ===== NEW: paired-record helpers =====

def _is_paired_record(r: Any) -> bool:
    if not isinstance(r, dict):
        return False
    if not isinstance(r.get("original"), list) or not isinstance(r.get("sample"), list):
        return False
    if not r["original"] or not r["sample"]:
        return False
    o0 = r["original"][0]
    s0 = r["sample"][0]
    return isinstance(o0, dict) and isinstance(o0.get("text"), str) and isinstance(s0, dict) and isinstance(s0.get("text"), str)

def _record_label(r: Dict[str, Any], default: int = 1) -> int:
    # attack-only: label 常在 meta.base_source.orig_label / original[0].orig_label / sample[0].orig_label
    for path in (
        ("label",),
        ("meta", "base_source", "orig_label"),
        ("original", 0, "orig_label"),
        ("sample", 0, "orig_label"),
    ):
        cur: Any = r
        ok = True
        for k in path:
            try:
                cur = cur[k] if isinstance(k, int) else cur.get(k)
            except Exception:
                ok = False
                break
        if ok and cur is not None:
            try:
                return int(cur)
            except Exception:
                pass
    return int(default)

def _record_id(r: Dict[str, Any]) -> Optional[str]:
    # 你示例里顶层 id 就是 source_id
    v = _norm_id(r.get("id"))
    if v is not None:
        return v
    meta = r.get("meta")
    if isinstance(meta, dict):
        bs = meta.get("base_source")
        if isinstance(bs, dict):
            v2 = _norm_id(bs.get("source_id"))
            if v2 is not None:
                return v2
    return None

def _extract_original_text(r: Dict[str, Any]) -> Optional[str]:
    try:
        o0 = r["original"][0]
        t = o0.get("text")
        if isinstance(t, str) and t.strip():
            return t
    except Exception:
        pass
    return None

def _extract_attack_variants(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 record.sample 中取攻击变体：
    - 若 sample=[src, attacked, ...]：跳过 attack=='src'
    - 若 sample=[attacked]：直接用
    返回的是 “variant dict（原始对象）” 列表
    """
    out: List[Dict[str, Any]] = []
    smp = r.get("sample")
    if not isinstance(smp, list):
        return out
    for obj in smp:
        if not isinstance(obj, dict):
            continue
        t = obj.get("text")
        if not isinstance(t, str) or not t.strip():
            continue
        atk = str(obj.get("attack") or "").strip().lower()
        if atk == "src":
            continue
        out.append(obj)
    return out

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
    # ===== NEW: paired-record ASR (base from original[0], attacks from sample[*]) =====
    if atk_exs and any(_is_paired_record(x) for x in atk_exs):
        # build base originals + all variants (one-to-many), then do:
        # 1) eval originals; keep only base-correct
        # 2) eval variants of base-correct; any-success => ASR

        base_exs: List[Dict[str, Any]] = []
        var_exs: List[Dict[str, Any]] = []
        var_base_pos: List[int] = []

        for r in atk_exs:
            if not _is_paired_record(r):
                continue
            rid = _record_id(r) or None
            y = _record_label(r, default=1)
            ot = _extract_original_text(r)
            if ot is None:
                continue

            bpos = len(base_exs)
            base_exs.append({"id": rid, "text": ot, "label": int(y)})

            variants = _extract_attack_variants(r)
            for v in variants:
                vt = v.get("text")
                if not isinstance(vt, str) or not vt.strip():
                    continue
                # 保留 attack 字段给统计/分桶用；label 强制对齐到 original 的 label
                ex = dict(v)
                ex["label"] = int(y)
                if rid is not None:
                    ex["id"] = rid
                var_exs.append(ex)
                var_base_pos.append(int(bpos))

        # no usable pairs
        if not base_exs or not var_exs:
            return {
                "match_mode": "paired_record",
                "align": {"base_n": len(base_exs), "attack_variant_n_raw": len(var_exs), "note": "no valid paired records"},
                "base_correct_n": 0,
                "attack_eval_n": 0,
                "attack_acc": None,
                "asr": None,
            }

        # (1) evaluate originals
        res0 = det.evaluate(base_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y0 = [int(x) for x in res0.labels]
        p0 = [int(x) for x in res0.preds]
        base_correct_mask = [yy == pp for yy, pp in zip(y0, p0)]

        # (2) filter variants by base-correct
        var_eval: List[Dict[str, Any]] = []
        var_eval_base_pos: List[int] = []
        for ex, bp in zip(var_exs, var_base_pos):
            if 0 <= bp < len(base_correct_mask) and bool(base_correct_mask[bp]):
                var_eval.append(ex)
                var_eval_base_pos.append(int(bp))

        if not var_eval:
            return {
                "match_mode": "paired_record",
                "align": {"base_n": len(base_exs), "attack_variant_n_raw": len(var_exs), "attack_variant_n_eval": 0},
                "base_correct_n": int(sum(1 for ok in base_correct_mask if ok)),
                "attack_eval_n": 0,
                "attack_acc": None,
                "asr": None,
            }

        # (3) evaluate variants
        res1 = det.evaluate(var_eval, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y1 = [int(x) for x in res1.labels]
        p1 = [int(x) for x in res1.preds]
        var_ok = [yy == pp for yy, pp in zip(y1, p1)]
        var_acc = sum(1 for ok in var_ok if ok) / max(1, len(var_ok))

        # (4) any-success aggregation per base
        fail_by_base: Dict[int, bool] = {}
        for bp, ok in zip(var_eval_base_pos, var_ok):
            if bp not in fail_by_base:
                fail_by_base[bp] = False
            if not ok:
                fail_by_base[bp] = True

        base_eval_positions = sorted(fail_by_base.keys())
        base_eval_n = len(base_eval_positions)
        success_n = sum(1 for bp in base_eval_positions if fail_by_base.get(bp, False))
        asr = (success_n / base_eval_n) if base_eval_n > 0 else None
        attack_acc_base = (1.0 - asr) if asr is not None else None

        return {
            "match_mode": "paired_record",
            "align": {
                "base_n": int(len(base_exs)),
                "attack_variant_n_raw": int(len(var_exs)),
                "attack_variant_n_eval": int(len(var_eval)),
                "base_eval_n": int(base_eval_n),
                "note": "base correctness from record.original[0].text; variants from record.sample[*] (skip attack=='src'); any-success aggregation",
            },
            "base_correct_n": int(sum(1 for ok in base_correct_mask if ok)),
            "attack_eval_n": int(base_eval_n),
            "attack_variant_n": int(len(var_eval)),
            "attack_variant_acc": float(var_acc),
            "attack_acc": float(attack_acc_base) if attack_acc_base is not None else None,
            "asr": float(asr) if asr is not None else None,
            "aggregation": "any-success over record-local variants",
        }

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
    # ===== NEW: paired-record ASR-by-method (base from original[0]) =====
    if atk_exs and any(_is_paired_record(x) for x in atk_exs):
        base_exs: List[Dict[str, Any]] = []
        var_exs: List[Dict[str, Any]] = []
        var_base_pos: List[int] = []
        var_method: List[str] = []

        for r in atk_exs:
            if not _is_paired_record(r):
                continue
            rid = _record_id(r) or None
            y = _record_label(r, default=1)
            ot = _extract_original_text(r)
            if ot is None:
                continue

            bpos = len(base_exs)
            base_exs.append({"id": rid, "text": ot, "label": int(y)})

            variants = _extract_attack_variants(r)
            for v in variants:
                vt = v.get("text")
                if not isinstance(vt, str) or not vt.strip():
                    continue
                ex = dict(v)
                ex["label"] = int(y)
                if rid is not None:
                    ex["id"] = rid
                var_exs.append(ex)
                var_base_pos.append(int(bpos))
                var_method.append(_attack_method_name(v) or "unknown")

        if not base_exs or not var_exs:
            return {
                "by_method": {},
                "summary": _summarize_asr_attacks({}),
                "note": "paired-record mode: no valid records/variants",
                "base_cache": {"base_n": int(len(base_exs)), "use_key": False},
            }

        # 1) eval originals once
        res0 = det.evaluate(base_exs, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y0 = [int(x) for x in res0.labels]
        p0 = [int(x) for x in res0.preds]
        base_correct_mask = [yy == pp for yy, pp in zip(y0, p0)]

        # 2) eval ALL variants of base-correct once
        var_eval: List[Dict[str, Any]] = []
        var_eval_bp: List[int] = []
        var_eval_m: List[str] = []
        for ex, bp, m in zip(var_exs, var_base_pos, var_method):
            if 0 <= bp < len(base_correct_mask) and bool(base_correct_mask[bp]):
                var_eval.append(ex)
                var_eval_bp.append(int(bp))
                var_eval_m.append(m)

        by_method: Dict[str, Any] = {}
        if not var_eval:
            return {
                "by_method": {},
                "summary": _summarize_asr_attacks({}),
                "note": "paired-record mode: no variants after filtering by base-correct originals",
                "base_cache": {"base_n": int(len(base_exs)), "use_key": False},
            }

        res1 = det.evaluate(var_eval, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
        y1 = [int(x) for x in res1.labels]
        p1 = [int(x) for x in res1.preds]
        var_ok = [int(yy) == int(pp) for yy, pp in zip(y1, p1)]

        # 3) aggregate per method (any-success per base within that method)
        methods = sorted(set(var_eval_m))
        for m in methods:
            idxs = [i for i, mm in enumerate(var_eval_m) if mm == m]
            if not idxs:
                continue

            # variant-level acc for this method
            var_acc = sum(1 for i in idxs if var_ok[i]) / max(1, len(idxs))

            fail_by_base: Dict[int, bool] = {}
            for i in idxs:
                bp = var_eval_bp[i]
                if bp not in fail_by_base:
                    fail_by_base[bp] = False
                if not var_ok[i]:
                    fail_by_base[bp] = True

            base_eval_positions = sorted(fail_by_base.keys())
            base_eval_n = len(base_eval_positions)
            success_n = sum(1 for bp in base_eval_positions if fail_by_base.get(bp, False))
            asr = (success_n / base_eval_n) if base_eval_n > 0 else None
            attack_acc_base = (1.0 - asr) if asr is not None else None

            by_method[m] = {
                "match_mode": "paired_record",
                "align": {
                    "base_n": int(len(base_exs)),
                    "attack_variant_n_eval": int(len(idxs)),
                    "base_eval_n": int(base_eval_n),
                    "note": "base correctness from record.original[0]; method variants from record.sample[*]; any-success per base within method",
                },
                "base_correct_n": int(sum(1 for ok in base_correct_mask if ok)),
                "attack_eval_n": int(base_eval_n),
                "attack_variant_n": int(len(idxs)),
                "attack_variant_acc": float(var_acc),
                "attack_acc": float(attack_acc_base) if attack_acc_base is not None else None,
                "asr": float(asr) if asr is not None else None,
                "aggregation": "any-success over record-local variants (method-specific)",
                "attack_method": m,
                "attack_n_raw": int(len(idxs)),
            }

        return {
            "by_method": by_method,
            "summary": _summarize_asr_attacks(by_method),
            "note": "ASR(by_method) in paired-record mode; originals evaluated first, then variants filtered by base-correct originals.",
            "base_cache": {"base_n": int(len(base_exs)), "use_key": False},
        }

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
