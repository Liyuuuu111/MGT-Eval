from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union, Sequence, List

import os
import io
import sys
import json
import time
import math
from pathlib import Path
import platform
import re
from datetime import datetime
import warnings
import random

from tqdm.auto import tqdm

# optional: torch only for GPU mem display (presentation only)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from ..detectors.base import DetectorBase, EvalResult
from ..detectors.registry import get_detector_cls
from ..data_utils.load import load_dataset_unified

from ._utils_common import _auto_run_dir
from ._utils_env import _env_fingerprint, _proc_snapshot, _reset_and_mark_cuda_peaks, _collect_cuda_peaks
from ._utils_text import _word_count, _looks_like_builder_record
from ._utils_loader import _load_examples_auto
from ._metrics_basic import _basic_stats, _by_group, _f1_from_counts
from ._metrics_curves import (
    _get_ranking_values,
    _filter_nonfinite_examples,
    _roc_curve,
    _pr_curve,
    _ece_brier,
    _risk_coverage,
    _tpr_at_fpr_points,
    _bootstrap_ci,
)
from ._plotting import _plot_curve, _plot_reliability
from ._asr import (
    _compute_asr,
    _compute_asr_by_method,
    _summarize_asr_attacks,
    _base_correct_cache_from_preds,
    _attack_method_name,
    _compute_asr_any_success_one_method
)

# ---------- warning filters ----------
warnings.filterwarnings(
    "ignore",
    message=r"Token indices sequence length is longer than the specified maximum sequence length for this model",
    category=UserWarning,
)

# ---------- optional yaml ----------
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

# =========================
# Progress/print style (presentation only)
# =========================
W_RUN  = 6
W_SEED = 10
W_MEM  = 8
W_N    = 7
W_NUM  = 7
W_TIME = 7
SEP    = " "

def _tpr_points_str(fpr_targets: Sequence[float], tpr_at_fpr: Any) -> str:
    if not tpr_at_fpr:
        return "-"

    def _extract_tpr(v: Any) -> Optional[float]:
        # v 可能是 float；也可能是 {"tpr":..., "threshold":..., ...}
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            # 常见字段名
            for key in ("tpr", "TPR", "value"):
                if key in v and v[key] is not None:
                    try:
                        return float(v[key])
                    except Exception:
                        return None
        return None

    # list/tuple：允许元素是 float 或 dict
    if isinstance(tpr_at_fpr, (list, tuple)):
        parts = []
        for i, tgt in enumerate(fpr_targets):
            v = None
            if i < len(tpr_at_fpr):
                v = _extract_tpr(tpr_at_fpr[i])
            parts.append(f"{tgt:g}->{(f'{v:.4f}' if v is not None else '-')}")
        return ", ".join(parts)

    # dict：value 允许是 float 或 dict
    if isinstance(tpr_at_fpr, dict):
        parts = []
        for tgt in fpr_targets:
            v_raw = None
            for k in (tgt, float(tgt), str(tgt), f"{tgt:g}", f"{tgt:.0e}"):
                if k in tpr_at_fpr:
                    v_raw = tpr_at_fpr.get(k)
                    break
            v = _extract_tpr(v_raw)
            parts.append(f"{tgt:g}->{(f'{v:.4f}' if v is not None else '-')}")
        return ", ".join(parts)

    return str(tpr_at_fpr)

def _fmt_num(x: Optional[float], w: int = W_NUM, nd: int = 4) -> str:
    if x is None:
        return f"{'-':>{w}}"
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return f"{'-':>{w}}"
        return f"{xf:>{w}.{nd}f}"
    except Exception:
        return f"{'-':>{w}}"

def _fmt_int(x: Optional[int], w: int = W_N) -> str:
    if x is None:
        return f"{'-':>{w}}"
    try:
        return f"{int(x):>{w}d}"
    except Exception:
        return f"{'-':>{w}}"

def _gpu_mem_str(w: int = W_MEM) -> str:
    # match train.py vibe: reserved in GB with 3 sig figs, e.g., "3.21G"
    if torch is None or not hasattr(torch, "cuda") or (not torch.cuda.is_available()):
        return f"{'0G':>{w}}"
    try:
        mem_g = float(torch.cuda.memory_reserved()) / 1e9
        return f"{mem_g:.3g}G".rjust(w)
    except Exception:
        return f"{'-':>{w}}"

def _is_paired_builder_file(path: str, *, peek_lines: int = 10) -> bool:
    if not (isinstance(path, str) and os.path.exists(path)):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(peek_lines):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (
                    isinstance(r, dict)
                    and _looks_like_builder_record(r)
                    and isinstance(r.get("sample"), list)
                    and len(r["sample"]) >= 1  # ✅ 原来是 >=2
                ):
                    return True
    except Exception:
        return False
    return False

from pathlib import Path

def _attack_key_from_sample(s: dict) -> str:
    """
    给 paired-builder 的 sample 条目生成一个“细化后的攻击键”。
    - 主要看 s["attack"] / s["aug_method"]
    - 对 typo 尽量拼上 meta 里的 subtype / op / rate 等，做到细粒度
    """
    if not isinstance(s, dict):
        return "attack"

    key = (s.get("attack") or s.get("aug_method") or s.get("name") or "attack")
    key = str(key).strip()
    if not key:
        key = "attack"

    meta = s.get("meta")
    if isinstance(meta, dict):
        kl = key.lower()
        if kl.startswith("typo"):
            # 细分 subtype/op（任选其一即可）
            for mk in ("subtype", "variant", "mode", "op", "edit_type", "edit"):
                if mk in meta and meta[mk] is not None:
                    key = f"{key}/{meta[mk]}"
                    break
            # 细分强度（任选其一即可）
            for mk in ("rate", "p", "eps", "level", "severity"):
                if mk in meta and meta[mk] is not None:
                    key = f"{key}@{mk}={meta[mk]}"
                    break
    return key


def _extract_attack_groups_from_paired_file(path: str, *, base_ids: set | None = None) -> dict[str, list[dict]]:
    """
    paired-builder attack 文件：每个 record 里 sample 有 src + 多个 attack 变体。
    返回：{attack_key: [examples...]}，examples 的 id 与 base 的 id 对齐（用于 ASR 对齐）
    """
    groups: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not (isinstance(rec, dict) and _looks_like_builder_record(rec)):
                continue

            base_ex = _builder_record_to_base_example(rec)
            if base_ex is None:
                continue

            base_id = base_ex.get("id", rec.get("id"))
            if base_id is None:
                continue
            if base_ids is not None and base_id not in base_ids:
                continue

            label = int(base_ex.get("label", 1))

            smp = rec.get("sample")
            if not isinstance(smp, list) or len(smp) == 0:
                continue

            for a in smp:
                if not isinstance(a, dict):
                    continue
                atk = _attack_key_from_sample(a).strip()
                atk_l = atk.lower()
                # 跳过 src/orig
                if atk_l in ("src", "source", "orig", "original", ""):
                    continue
                text = a.get("text")
                if text is None:
                    continue

                ex = {
                    "id": base_id,                 # ✅ 与 base 对齐
                    "text": str(text),
                    "label": label,
                    "attack": atk,
                }
                # 可选保留 meta/lang 等
                for k in ("lang", "split", "source", "sub_source", "model"):
                    if k in rec and rec[k] is not None:
                        ex[k] = rec[k]
                groups.setdefault(atk, []).append(ex)

    return groups


def _extract_attack_groups_from_flat_file(path: str, *, base_ids: set | None = None) -> dict[str, list[dict]]:
    """
    flat attack 文件：同一行可能出现多个攻击字段，例如：
      - text_typo_swap, text_typo_del, ...
      - 或者 text + attack 字段
    返回：{attack_key: [examples...]}
    """
    groups: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue

            base_id = rec.get("id")
            if base_id is None:
                continue
            if base_ids is not None and base_id not in base_ids:
                continue

            label = rec.get("label", rec.get("orig_label", None))
            if label is None:
                continue
            label = int(label)

            # 形式1：每行一个 attack（用 rec["attack"] 分组）
            if "attack" in rec and "text" in rec and rec["text"] is not None:
                atk = str(rec["attack"]).strip() or "attack"
                groups.setdefault(atk, []).append(
                    {"id": base_id, "text": str(rec["text"]), "label": label, "attack": atk}
                )
                continue

            # 形式2：多字段 text_xxx
            _SKIP_SUFFIX = {"src", "source", "orig", "original", "base", "clean"}  # ✅ base字段别当攻击

            for k, v in rec.items():
                if not isinstance(k, str):
                    continue
                if not k.startswith("text_"):
                    continue
                if v is None:
                    continue

                suffix = k[len("text_"):].strip()
                if suffix.lower() in _SKIP_SUFFIX:
                    continue  # ✅ 跳过 base/source 字段

                atk = suffix or "attack"
                groups.setdefault(atk, []).append(
                    {"id": base_id, "text": str(v), "label": label, "attack": atk}
                )

    return groups

_BASE_TEXT_KEYS = ("text_src", "text_orig", "text_original", "original_text", "src_text", "text")

def _is_flat_multi_attack_file(path: str, *, peek_lines: int = 10) -> bool:
    """
    判断 flat jsonl/json 是否是“一个record含多个 text_* 攻击字段”的容器文件。
    条件：存在至少一个 text_XXX(非src/orig/base/clean) 字段，同时能找到某个 base 文本字段。
    """
    if not (isinstance(path, str) and os.path.exists(path)):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(peek_lines):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue

                # 有 base text 字段？
                has_base = any((k in rec and rec.get(k) not in (None, "")) for k in _BASE_TEXT_KEYS)

                # 有 attack text_ 字段？
                has_attack = False
                for k, v in rec.items():
                    if not (isinstance(k, str) and k.startswith("text_")):
                        continue
                    if v in (None, ""):
                        continue
                    suffix = k[len("text_"):].strip().lower()
                    if suffix in {"src", "source", "orig", "original", "base", "clean"}:
                        continue
                    has_attack = True
                    break

                if has_base and has_attack:
                    return True
    except Exception:
        return False
    return False


def _flat_record_to_base_example(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    flat 多字段 record -> base example（主评测用）
    base text 优先级：text_src/text_orig/... > text
    """
    if not isinstance(rec, dict):
        return None

    base_text = None
    for k in _BASE_TEXT_KEYS:
        v = rec.get(k)
        if v not in (None, ""):
            base_text = str(v)
            break
    if base_text is None:
        return None

    # label / id
    y = rec.get("orig_label", rec.get("label", None))
    if y is None:
        return None

    ex: Dict[str, Any] = {
        "id": rec.get("id", None),
        "text": base_text,
        "label": int(y),
    }
    for k in ("lang", "split", "source", "sub_source", "model"):
        if k in rec and rec[k] is not None:
            ex[k] = rec[k]
    return ex


def _load_base_examples_from_flat_multi_attack_file(
    path: str,
    *,
    sample_k: Optional[int],
    sample_seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rng = random.Random(int(sample_seed))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            b = _flat_record_to_base_example(rec)
            if b is not None:
                out.append(b)

    if sample_k is not None and int(sample_k) > 0 and len(out) > int(sample_k):
        rng.shuffle(out)
        out = out[: int(sample_k)]
    return out

def _extract_attack_groups(path: str, *, base_ids: set | None = None) -> dict[str, list[dict]]:
    # paired-builder 优先
    if _is_paired_builder_file(path):
        return _extract_attack_groups_from_paired_file(path, base_ids=base_ids)
    return _extract_attack_groups_from_flat_file(path, base_ids=base_ids)

def _builder_record_to_base_example(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    把一个 builder record 转成 “用于主评测的 base 样本”：
      - text: 优先 rec["original"][0]["text"], 否则找 sample 中 attack=src 的 text
      - id:   优先 rec["id"]（稳定 id），否则 fallback
      - label: 优先 orig_label，再 fallback 到 label
    """
    if not (isinstance(rec, dict) and _looks_like_builder_record(rec)):
        return None

    base_src = None

    # 1) prefer original[0]
    orig = rec.get("original")
    if isinstance(orig, list) and orig and isinstance(orig[0], dict) and (orig[0].get("text") is not None):
        base_src = orig[0]

    # 2) fallback: sample attack=="src"
    if base_src is None:
        smp = rec.get("sample")
        if isinstance(smp, list):
            for a in smp:
                if not isinstance(a, dict):
                    continue
                atk = str(a.get("attack") or a.get("aug_method") or "").strip().lower()
                if atk in ("src", "source", "orig", "original", "") and (a.get("text") is not None):
                    base_src = a
                    break
            if base_src is None and smp and isinstance(smp[0], dict) and (smp[0].get("text") is not None):
                base_src = smp[0]

    if base_src is None:
        return None

    ex: Dict[str, Any] = {}

    ex["text"] = str(base_src.get("text") or "")
    # label priority: base_src.orig_label -> base_src.label -> rec.orig_label -> rec.label
    y = base_src.get("orig_label", base_src.get("label", rec.get("orig_label", rec.get("label", 1))))
    ex["label"] = int(y)

    # stable id: rec.id preferred
    ex["id"] = rec.get("id", base_src.get("id", None))

    # keep common grouping/meta fields
    for k in ("lang", "split", "source", "sub_source", "model"):
        if k in rec:
            ex[k] = rec.get(k)
        elif k in base_src:
            ex[k] = base_src.get(k)

    # keep ids if present (help alignment)
    for k in ("orig_id", "base_id", "source_id"):
        if k in rec:
            ex[k] = rec.get(k)

    return ex


def _load_base_examples_from_paired_file(
    path: str,
    *,
    sample_k: Optional[int],
    sample_seed: int,
) -> List[Dict[str, Any]]:
    """
    关键：主评测只返回 “非 builder 样本（通常是 human） + builder record 的 base(original/src)”
    不把 sample 里的攻击变体摊平到主评测里。
    """
    out: List[Dict[str, Any]] = []
    rng = random.Random(int(sample_seed))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue

            b = _builder_record_to_base_example(rec)
            if b is not None:
                out.append(b)
            else:
                # 普通样本（通常 human / 或者已是 flat 格式）
                if "text" in rec and ("label" in rec or "orig_label" in rec):
                    ex = dict(rec)
                    if "label" not in ex and "orig_label" in ex:
                        ex["label"] = int(ex["orig_label"])
                    out.append(ex)

    # 可选采样：简单 shuffle+截断（如果你非常在意分层，可以后续再换成 stratified）
    if sample_k is not None and int(sample_k) > 0 and len(out) > int(sample_k):
        rng.shuffle(out)
        out = out[: int(sample_k)]
    # ✅ dedup by id (important for one-to-many files where base repeats across attacks)
    seen = set()
    dedup = []
    for ex in out:
        _id = ex.get("id", None) if isinstance(ex, dict) else None
        if _id is None:
            dedup.append(ex)
            continue
        if _id in seen:
            continue
        seen.add(_id)
        dedup.append(ex)
    out = dedup

    return out

def _print_eval_header(prefix: str = "") -> None:
    # columns: Run Seed GPU_mem N Acc F1 AUROC AUPR ECE t(s)
    hdr = (
        f"{prefix}"
        f"{'Run':>{W_RUN}}{SEP}"
        f"{'Seed':>{W_SEED}}{SEP}"
        f"{'GPU_mem':>{W_MEM}}{SEP}"
        f"{'N':>{W_N}}{SEP}"
        f"{'Acc':>{W_NUM}}{SEP}"
        f"{'F1':>{W_NUM}}{SEP}"
        f"{'AUROC':>{W_NUM}}{SEP}"
        f"{'AUPR':>{W_NUM}}{SEP}"
        f"{'ECE':>{W_NUM}}{SEP}"
        f"{'t(s)':>{W_TIME}}"
    )
    print("\n" + hdr)

def evaluate_detector(
    detector: Union[str, DetectorBase],
    dataset: Union[str, Iterable[Dict[str, Any]]],
    batch_size: int = 8,
    threshold: float = 0.5,
    fpr_targets: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    group_cols: Optional[Sequence[str]] = None,
    out_dir: Optional[str] = None,
    out_prefix: Optional[str] = None,   # 兼容保留，不再用作文件名前缀
    save_curves: bool = True,
    ci_enable: Optional[bool] = None,
    ci_iters: int = 200,
    ci_seed: int = 114514,
    show_progress: bool = True,
    k_runs: int = 1,
    attack_datasets: Optional[Union[str, Sequence[str]]] = None,
    asr_save_details: bool = True,
    **detector_kwargs,
) -> EvalResult:
    """
    - 对外接口保持不变（evaluate_detector(...)）。
    - 内部实现已模块化：curves / plots / env / loader / asr / stats 分离。
    - 本版本仅增强提示语句 & 外层进度条样式（不改指标计算逻辑）。
    """
    # 0) init detector
    if isinstance(detector, str):
        Det = get_detector_cls(detector)
        det = Det(**detector_kwargs)
    else:
        det = detector
        for k, v in detector_kwargs.items():
            setattr(det, k, v)

    display_name = getattr(det, "name", getattr(det, "DETECTOR_NAME", "detector"))
    det_type = getattr(det, "detector_type", "Unknown")

    def _stat_pack(vals: List[float]) -> Dict[str, Any]:
        vals = [float(x) for x in vals if x is not None]
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        m = sum(vals) / n
        if n > 1:
            var = sum((x - m) ** 2 for x in vals) / (n - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        return {"n": n, "mean": m, "std": sd, "min": min(vals), "max": max(vals)}

    is_sampling = (sample_k is not None and sample_k > 0)
    multi_run = (is_sampling and (k_runs is not None) and (int(k_runs) > 1))

    # run dir
    run_dir = _auto_run_dir(out_dir, display_name)
    if show_progress:
        print(f"[MGTEval] detector='{display_name}' (type={det_type})")
        print(f"[MGTEval] run_dir -> {str(run_dir)}")
        if isinstance(dataset, str):
            print(f"[MGTEval] dataset -> {dataset}")
        else:
            print(f"[MGTEval] dataset -> iterable")
        if is_sampling:
            print(f"[MGTEval] sampling -> sample_k={sample_k} seed={sample_seed} k_runs={k_runs}")
        if group_cols is not None:
            print(f"[MGTEval] group_cols -> {list(group_cols)}")
        if attack_datasets is not None:
            atk_list = [attack_datasets] if isinstance(attack_datasets, str) else list(attack_datasets)
            print(f"[MGTEval] ASR enabled -> {len(atk_list)} attack dataset(s)")

    run_cfg = {
        "detector": display_name,
        "detector_type": det_type,
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

    # =========================
    # Multi-run sampling branch
    # =========================
    if multi_run:
        base = int(sample_seed)
        offsets = list(range(-10, 11))
        k = int(k_runs)
        seeds = [base + offsets[i % len(offsets)] for i in range(k)]

        metrics_dir = run_dir / "metrics"
        curves_dir = metrics_dir / "curves"
        figures_dir = run_dir / "figures"
        k_runs_dir = metrics_dir / "k_runs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        k_runs_dir.mkdir(parents=True, exist_ok=True)

        with open(k_runs_dir / "seeds.json", "w", encoding="utf-8") as f:
            json.dump({"base": base, "seeds": seeds}, f, ensure_ascii=False, indent=2)

        acc_list, auroc_list, aupr_list, f1_list = [], [], [], []
        ece_list, brier_list = [], []
        eval_secs, load_secs = [], []
        per_run_briefs = []

        first_res: Optional[EvalResult] = None
        first_examples: Optional[List[Dict[str, Any]]] = None
        first_used_group_cols: List[str] = []
        first_probs_seq: Optional[List[float]] = None
        first_ranking_vec: Optional[List[float]] = None
        first_ranking_src: str = "none"
        meta_first = None

        if show_progress:
            _print_eval_header(prefix="")

        runs_iter = tqdm(
            range(len(seeds)),
            desc="Runs",
            leave=True,
            dynamic_ncols=True,
            disable=(not show_progress),
        )

        for ridx in runs_iter:
            seed = seeds[ridx]
            t0 = time.perf_counter()
            proc0 = _proc_snapshot()

            if show_progress:
                print(f"[MGTEval] [run {ridx}/{len(seeds)-1}] loading dataset (seed={seed})...")

            examples, used_group_cols = load_dataset_unified(
                dataset=dataset,
                sample_k=sample_k,
                sample_seed=int(seed),
                group_cols=group_cols,
            )
            load_time = time.perf_counter() - t0

            cuda_ctx = _reset_and_mark_cuda_peaks()

            if show_progress:
                print(f"[MGTEval] [run {ridx}] evaluating detector on N={len(examples)} (batch_size={batch_size}, thr={threshold})...")

            t1 = time.perf_counter()
            res_i = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
            eval_time = time.perf_counter() - t1

            labels_seq = list(res_i.labels)
            preds_seq = list(res_i.preds)
            probs_seq = (list(res_i.probs) if getattr(res_i, "probs", None) is not None else None)
            ranking_vec, ranking_src = _get_ranking_values(res_i)

            examples, labels_seq, preds_seq, ranking_vec, probs_seq, dropped = _filter_nonfinite_examples(
                examples, labels_seq, preds_seq, ranking_vec=ranking_vec, probs=probs_seq
            )
            if dropped > 0 and show_progress:
                print(f"[MGTEval] evaluator(run {ridx}): dropped {dropped} non-finite samples before curves/plots.")
            if len(labels_seq) == 0:
                raise RuntimeError("Empty set after dropping non-finite samples in run; aborting.")

            overall_basic = _basic_stats(labels_seq, preds_seq, probs_seq)
            _fpr, _tpr, auroc = _roc_curve(labels_seq, ranking_vec)
            _rec, _prec, aupr = _pr_curve(labels_seq, ranking_vec)
            _tpr_at_fpr = _tpr_at_fpr_points(labels_seq, ranking_vec, fpr_targets=fpr_targets)

            if probs_seq is not None:
                calib = _ece_brier(labels_seq, probs_seq, bins=10)
            else:
                calib = {"ece": None, "brier": None, "bins": []}

            acc = overall_basic["acc"]
            tp = overall_basic["confusion"]["tp"]
            fp = overall_basic["confusion"]["fp"]
            fn = overall_basic["confusion"]["fn"]
            f1 = _f1_from_counts(tp, fp, fn)

            acc_list.append(acc)
            f1_list.append(f1)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            ece_list.append(calib["ece"])
            brier_list.append(calib["brier"])
            load_secs.append(load_time)
            eval_secs.append(eval_time)

            # ---- Update outer progress bar description (train.py style) ----
            mem = _gpu_mem_str()
            desc = (
                f"{ridx:>{W_RUN}d}{SEP}"
                f"{int(seed):>{W_SEED}d}{SEP}"
                f"{mem:>{W_MEM}}{SEP}"
                f"{len(examples):>{W_N}d}{SEP}"
                f"{_fmt_num(acc)}{SEP}"
                f"{_fmt_num(f1)}{SEP}"
                f"{_fmt_num(auroc)}{SEP}"
                f"{_fmt_num(aupr)}{SEP}"
                f"{_fmt_num(calib['ece'])}{SEP}"
                f"{eval_time:>{W_TIME}.1f}"
            )
            runs_iter.set_description(desc)

            if show_progress:
                tpr_points = _tpr_points_str(fpr_targets, tpr_at_fpr)
                print(f"[MGTEval] [run {ridx}] done: acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f} aupr={aupr:.4f} ece={calib['ece']} "
                      f"(rank={ranking_src}) tpr@fprs=[{tpr_points}] time={eval_time:.2f}s")

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

            if ridx == 0:
                first_res = res_i
                first_examples = examples
                first_used_group_cols = list(used_group_cols)
                first_probs_seq = probs_seq
                first_ranking_vec = ranking_vec
                first_ranking_src = ranking_src

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
                    "detector": {"name": display_name, "type": det_type},
                    "dataset": {"size": len(examples), "group_cols": list(used_group_cols)},
                    "notes": "Auto-generated run manifest for auditability and reproducibility. (first run artifacts)",
                }
                with open(run_dir / "run-manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)

                try:
                    if hasattr(first_res, "meta") and isinstance(first_res.meta, dict):
                        meta_first = dict(first_res.meta)
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

                # --- first run artifacts: group metrics, curves, plots, predictions, model card ---
                labels_seq_1 = list(res_i.labels)
                preds_seq_1 = list(res_i.preds)

                # per_lang
                if "lang" in used_group_cols:
                    col_vals = [str(ex.get("lang", "unknown")) for ex in examples]
                    g = _by_group(col_vals, labels_seq_1, preds_seq_1)
                    with open(metrics_dir / "per_lang.json", "w", encoding="utf-8") as f:
                        json.dump(g, f, ensure_ascii=False, indent=2)

                    if save_curves and first_ranking_vec is not None:
                        langs = sorted(set(col_vals))
                        for lg in langs:
                            idxs = [i for i, v in enumerate(col_vals) if v == lg]
                            if not idxs:
                                continue
                            yl = [labels_seq_1[i] for i in idxs]
                            rv = [first_ranking_vec[i] for i in idxs]
                            fpr_l, tpr_l, auroc_l = _roc_curve(yl, rv)
                            rec_l, prec_l, aupr_l = _pr_curve(yl, rv)
                            with open(curves_dir / f"roc_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"fpr": fpr_l, "tpr": tpr_l, "auroc": auroc_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            with open(curves_dir / f"pr_{lg}.json", "w", encoding="utf-8") as f:
                                json.dump({"recall": rec_l, "precision": prec_l, "aupr": aupr_l, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                            _plot_curve(fpr_l, tpr_l, f"ROC ({lg}) AUC={auroc_l:.3f}", "FPR", "TPR", figures_dir / f"roc_{lg}.png")
                            _plot_curve(rec_l, prec_l, f"PR ({lg}) AUPR={aupr_l:.3f}", "Recall", "Precision", figures_dir / f"pr_{lg}.png")

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

                # overall curves/plots
                if save_curves and first_ranking_vec is not None:
                    fpr_o, tpr_o, auroc_o = _roc_curve(labels_seq_1, first_ranking_vec)
                    rec_o, prec_o, aupr_o = _pr_curve(labels_seq_1, first_ranking_vec)
                    with open(curves_dir / "roc_overall.json", "w", encoding="utf-8") as f:
                        json.dump({"fpr": fpr_o, "tpr": tpr_o, "auroc": auroc_o, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                    with open(curves_dir / "pr_overall.json", "w", encoding="utf-8") as f:
                        json.dump({"recall": rec_o, "precision": prec_o, "aupr": aupr_o, "rank_source": first_ranking_src}, f, ensure_ascii=False, indent=2)
                    _plot_curve(fpr_o, tpr_o, f"ROC (overall) AUC={auroc_o:.3f}", "FPR", "TPR", figures_dir / "roc_overall.png")
                    _plot_curve(rec_o, prec_o, f"PR (overall) AUPR={aupr_o:.3f}", "Recall", "Precision", figures_dir / "pr_overall.png")

                    if first_probs_seq is not None:
                        calib_o = _ece_brier(labels_seq_1, first_probs_seq, bins=10)
                        cov_o, risk_o = _risk_coverage(labels_seq_1, first_probs_seq)
                        with open(curves_dir / "rc_abstain_overall.json", "w", encoding="utf-8") as f:
                            json.dump({"coverage": cov_o, "risk": risk_o}, f, ensure_ascii=False, indent=2)
                        _plot_curve(cov_o, risk_o, "Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / "rc_overall.png")
                        _plot_reliability(calib_o["bins"], figures_dir / "calibration_overall.png")

                # predictions.json (first run)
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

                # model card
                card = {"detector_name": display_name, "detector_type": det_type}
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

        # CI: first run only
        do_ci = (ci_enable if ci_enable is not None else True)
        ci = None
        if do_ci and first_res is not None and getattr(first_res, "probs", None) is not None and len(first_res.labels) > 5:
            if show_progress:
                print(f"[MGTEval] computing bootstrap CI (iters={ci_iters}, seed={ci_seed}) on first run...")
            ci = _bootstrap_ci(first_res.labels, first_res.probs, iters=int(ci_iters), seed=int(ci_seed))

        tpr_at_fpr_first = None
        if first_res is not None and first_ranking_vec is not None:
            tpr_at_fpr_first = _tpr_at_fpr_points(first_res.labels, first_ranking_vec, fpr_targets=fpr_targets)

        # ASR: first run only (optional)
        asr_results = None
        atk_specs: List[str] = []
        if attack_datasets is not None:
            atk_specs = [attack_datasets] if isinstance(attack_datasets, str) else list(attack_datasets)

        if atk_specs and (first_examples is not None):
            base_for_asr = first_examples
            attacks_out: Dict[str, Any] = {}

            if show_progress:
                print(f"[MGTEval] ASR: evaluating {len(atk_specs)} attack dataset(s) on first run base...")

            atk_iter = tqdm(atk_specs, desc="ASR", leave=True, dynamic_ncols=True, disable=(not show_progress))
            for atk_path in atk_iter:
                atk_key = Path(str(atk_path)).stem
                atk_iter.set_description(f"ASR [{atk_key}]")

                atk_exs = _load_examples_auto(atk_path, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="flat")
                if not atk_exs:
                    atk_exs = _load_examples_auto(atk_path, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="post")

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

                if show_progress:
                    s = attacks_out[atk_key].get("summary", {})
                    asr_val = s.get("asr", None)
                    print(f"[MGTEval] ASR[{atk_key}] -> asr={asr_val}")

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

        if show_progress:
            print(f"[MGTEval] (multi-run) results saved to: {str(run_dir)}")
        return first_res  # 保持兼容：返回首轮 EvalResult

    # ======================
    # Single-run branch
    # ======================
    if show_progress:
        print(f"[MGTEval] [1/4] loading dataset...")

    t0 = time.perf_counter()
    proc0 = _proc_snapshot()

    paired_base_mode = False
    flat_multi_mode = False
    if isinstance(dataset, str) and (dataset.endswith(".jsonl") or dataset.endswith(".json")) and os.path.exists(dataset):
        paired_base_mode = _is_paired_builder_file(dataset)
        if not paired_base_mode:
            flat_multi_mode = _is_flat_multi_attack_file(dataset)

    if paired_base_mode:
        # ✅ 主评测：只评估 original/src + 非 builder（human）
        examples = _load_base_examples_from_paired_file(
            dataset,
            sample_k=sample_k,
            sample_seed=sample_seed,
        )
        # group cols：尽量保持你现在风格（id/lang/split 常见）
        if group_cols is not None:
            used_group_cols = list(group_cols)
        else:
            used_group_cols = []
            for k in ("id", "lang", "split", "source", "sub_source", "model"):
                if any(isinstance(ex, dict) and (k in ex) for ex in examples):
                    used_group_cols.append(k)
    elif flat_multi_mode:
        examples = _load_base_examples_from_flat_multi_attack_file(dataset, sample_k=sample_k, sample_seed=sample_seed)
        # used_group_cols 同你 paired 分支那套推断
        if group_cols is not None:
            used_group_cols = list(group_cols)
        else:
            used_group_cols = []
            for k in ("id", "lang", "split", "source", "sub_source", "model"):
                if any(isinstance(ex, dict) and (k in ex) for ex in examples):
                    used_group_cols.append(k)
    else:
        examples, used_group_cols = load_dataset_unified(
            dataset=dataset,
            sample_k=sample_k,
            sample_seed=sample_seed,
            group_cols=group_cols,
        )

    load_time = time.perf_counter() - t0

    if show_progress:
        print(f"[MGTEval] loaded: N={len(examples)} groups={list(used_group_cols)} load_time={load_time:.3f}s")

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

    cuda_ctx = _reset_and_mark_cuda_peaks()

    if show_progress:
        print(f"[MGTEval] [2/4] evaluating detector (batch_size={batch_size}, thr={threshold})...")

    t1 = time.perf_counter()
    res = det.evaluate(examples, batch_size=batch_size, threshold=threshold, show_progress=show_progress)
    eval_time = time.perf_counter() - t1

    mem_stats = _collect_cuda_peaks(cuda_ctx)
    proc1 = _proc_snapshot()

    labels_seq = list(res.labels)
    preds_seq = list(res.preds)
    probs_seq = (list(res.probs) if getattr(res, "probs", None) is not None else None)
    ranking_vec, ranking_src = _get_ranking_values(res)

    examples, labels_seq, preds_seq, ranking_vec, probs_seq, dropped = _filter_nonfinite_examples(
        examples, labels_seq, preds_seq, ranking_vec=ranking_vec, probs=probs_seq
    )
    if dropped > 0 and show_progress:
        print(f"[MGTEval] evaluator: dropped {dropped} non-finite samples before curves/plots.")
    if len(labels_seq) == 0:
        raise RuntimeError("Empty set after dropping non-finite samples; aborting evaluation.")

    if show_progress:
        print(f"[MGTEval] detector eval done: eval_time={eval_time:.3f}s rank_source={ranking_src}")

    # group metrics
    by_groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for col in used_group_cols:
        col_vals = [str(ex.get(col, "unknown")) for ex in examples]
        if col.lower() == "model":
            mask = [int(y) == 1 for y in labels_seq]
            by_groups[col] = _by_group(col_vals, labels_seq, preds_seq, include_mask=mask, exclude_values=["human"])
        else:
            by_groups[col] = _by_group(col_vals, labels_seq, preds_seq)

    # manifest
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
        "detector": {"name": display_name, "type": det_type},
        "dataset": {"size": len(examples), "group_cols": list(used_group_cols)},
        "notes": "Auto-generated run manifest for auditability and reproducibility.",
    }
    with open(run_dir / "run-manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if show_progress:
        print(f"[MGTEval] [3/4] computing metrics...")

    # overall metrics
    overall_basic = _basic_stats(labels_seq, preds_seq, probs_seq)
    fpr, tpr, auroc = _roc_curve(labels_seq, ranking_vec)
    rec, prec, aupr = _pr_curve(labels_seq, ranking_vec)
    tpr_at_fpr = _tpr_at_fpr_points(labels_seq, ranking_vec, fpr_targets=fpr_targets)

    if probs_seq is not None:
        calib = _ece_brier(labels_seq, probs_seq, bins=10)
        cov, risk = _risk_coverage(labels_seq, probs_seq)
    else:
        calib = {"ece": None, "brier": None, "bins": []}
        cov, risk = ([], [])

    do_ci = (ci_enable if ci_enable is not None else (sample_k is not None and sample_k > 0))
    ci = None
    if do_ci and (probs_seq is not None) and len(labels_seq) > 5:
        if show_progress:
            print(f"[MGTEval] bootstrap CI enabled -> iters={ci_iters} seed={ci_seed}")
        ci = _bootstrap_ci(labels_seq, probs_seq, iters=int(ci_iters), seed=int(ci_seed))

    metrics_dir = run_dir / "metrics"
    curves_dir = metrics_dir / "curves"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # threshold_metrics
    threshold_metrics: Dict[str, Any] = {}
    if probs_seq is not None:
        thr_map: Dict[str, float] = {"eval": float(threshold)}
        if abs(float(threshold) - 0.5) > 1e-9:
            thr_map["p0.5"] = 0.5

        calib_thrs = getattr(det, "_calibrator_thresholds", None)
        if isinstance(calib_thrs, dict):
            for name, v in calib_thrs.items():
                if isinstance(v, (int, float)):
                    thr_map[str(name)] = float(v)

        used: Dict[str, float] = {}
        for name, tv in thr_map.items():
            if any(abs(tv - vv) < 1e-9 for vv in used.values()):
                continue
            used[name] = tv

        for name, tv in used.items():
            preds_thr = [1 if p >= tv else 0 for p in probs_seq]
            st = _basic_stats(labels_seq, preds_thr, probs_seq)
            tp = st["confusion"]["tp"]; tn = st["confusion"]["tn"]
            fp = st["confusion"]["fp"]; fn = st["confusion"]["fn"]
            P = tp + fn; N = tn + fp
            threshold_metrics[name] = {
                "threshold": tv,
                "acc": st["acc"],
                "confusion": st["confusion"],
                "f1": _f1_from_counts(tp, fp, fn),
                "tpr": (tp / P) if P > 0 else None,
                "fpr": (fp / N) if N > 0 else None,
            }

    # base cache for ASR reuse
    base_cache = _base_correct_cache_from_preds(examples, labels_seq, preds_seq)

    # ASR
    asr_results = None
    atk_specs: List[str] = []
    if attack_datasets is not None:
        atk_specs = [attack_datasets] if isinstance(attack_datasets, str) else list(attack_datasets)

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
                    if isinstance(r, dict) and _looks_like_builder_record(r) and isinstance(r.get("sample"), list) and len(r["sample"]) >= 1:
                        auto_paired_asr = True
                        break
        except Exception:
            auto_paired_asr = False

    if auto_paired_asr:
        if show_progress:
            print(f"[MGTEval] ASR: auto paired dataset detected -> {dataset}")

        # ✅ 复用主评测的 base（已经是 original/src + human 的版本）
        base_for_asr = examples

        # attack side 仍然从 post 读（这里就是“攻击后的文本”）
        atk_for_asr = _load_examples_auto(
            dataset,
            sample_k=None,
            sample_seed=sample_seed,
            group_cols=group_cols,
            builder_view="post",
        )

        if auto_paired_asr:

            base_for_asr = examples

            # ✅ 用 base_ids 过滤（尤其 sample_k 时很重要）
            base_ids = {ex.get("id") for ex in base_for_asr if isinstance(ex, dict) and ex.get("id") is not None}
            groups = _extract_attack_groups(str(dataset), base_ids=base_ids)

            attacks_out: Dict[str, Any] = {}

            if groups:
                # ✅ 每个字段/每个变体一个结果
                for gk, atk_exs in sorted(groups.items(), key=lambda x: x[0]):
                    out_key = f"paired/{gk}"
                    attacks_out[out_key] = _compute_asr_any_success_one_method(
                        det,
                        base_for_asr,
                        atk_exs,  # 同一攻击键下的所有 variants（允许一对多）
                        batch_size=batch_size,
                        threshold=threshold,
                        show_progress=show_progress,
                        base_cache=base_cache,  # ✅ 复用 base 的 correct cache
                    )
                    attacks_out[out_key]["attack_dataset"] = str(dataset)
                    attacks_out[out_key]["attack_key"] = gk
            else:
                # 兜底：保持旧行为
                atk_for_asr = _load_examples_auto(
                    dataset, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="post"
                )
                attacks_out["paired"] = _compute_asr(
                    det, base_for_asr, atk_for_asr,
                    batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                    base_cache=base_cache,
                )
                attacks_out["paired"]["attack_dataset"] = str(dataset)

            asr_results = {
                "definition": "ASR = 1 - Acc(attack | correct_before_attack)",
                "auto_paired_dataset": str(dataset),
                "attacks": attacks_out,
            }
            asr_results["summary"] = _summarize_asr_attacks(attacks_out)


    elif atk_specs:
        if show_progress:
            print(f"[MGTEval] ASR: evaluating {len(atk_specs)} attack dataset(s)...")

        base_for_asr = examples
        base_ids = {ex.get("id") for ex in base_for_asr if isinstance(ex, dict) and ex.get("id") is not None}
        attacks_out: Dict[str, Any] = {}

        atk_iter = tqdm(atk_specs, desc="ASR", leave=True, dynamic_ncols=True, disable=(not show_progress))
        for atk_path in atk_iter:
            stem = Path(str(atk_path)).stem
            atk_iter.set_description(f"ASR [{stem}]")

            groups = _extract_attack_groups(str(atk_path), base_ids=base_ids)

            if groups and len(groups) > 1:
                for gk, atk_exs in sorted(groups.items(), key=lambda x: x[0]):
                    out_key = f"{stem}/{gk}"
                    attacks_out[out_key] = _compute_asr(
                        det, base_for_asr, atk_exs,
                        batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                        base_cache=base_cache,
                    )
                    attacks_out[out_key]["attack_dataset"] = str(atk_path)
                    attacks_out[out_key]["attack_key"] = gk
            elif groups:
                (gk, atk_exs), = groups.items()
                out_key = stem
                attacks_out[out_key] = _compute_asr(
                    det, base_for_asr, atk_exs,
                    batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                    base_cache=base_cache,
                )
                attacks_out[out_key]["attack_dataset"] = str(atk_path)
                attacks_out[out_key]["attack_key"] = gk
            else:
                # fallback：沿用你原来的 loader + by_method
                atk_exs = _load_examples_auto(atk_path, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="flat")
                if not atk_exs:
                    atk_exs = _load_examples_auto(atk_path, sample_k=None, sample_seed=sample_seed, group_cols=group_cols, builder_view="post")

                has_method = any(isinstance(x, dict) and _attack_method_name(x) != "unknown" for x in atk_exs)
                if has_method:
                    attacks_out[stem] = _compute_asr_by_method(
                        det, base_for_asr, atk_exs,
                        batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                        base_cache=base_cache,
                    )
                else:
                    attacks_out[stem] = _compute_asr(
                        det, base_for_asr, atk_exs,
                        batch_size=batch_size, threshold=threshold, show_progress=show_progress,
                        base_cache=base_cache,
                    )
                attacks_out[stem]["attack_dataset"] = str(atk_path)

        asr_results = {
            "definition": "ASR = 1 - Acc(attack | correct_before_attack)",
            "base_dataset": (str(dataset) if isinstance(dataset, str) else "iterable"),
            "base_used_n": len(base_for_asr),
            "attacks": attacks_out,
        }
        asr_results["summary"] = _summarize_asr_attacks(attacks_out)

    if asr_results is not None and asr_save_details:
        try:
            with open(metrics_dir / "asr.json", "w", encoding="utf-8") as f:
                json.dump(asr_results, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---- A train.py-like one-line metric row (presentation only) ----
    if show_progress:
        tp = overall_basic["confusion"]["tp"]
        fp = overall_basic["confusion"]["fp"]
        fn = overall_basic["confusion"]["fn"]
        f1 = _f1_from_counts(tp, fp, fn)
        _print_eval_header(prefix="")
        row = (
            f"{0:>{W_RUN}d}{SEP}"
            f"{int(sample_seed):>{W_SEED}d}{SEP}"
            f"{_gpu_mem_str():>{W_MEM}}{SEP}"
            f"{len(labels_seq):>{W_N}d}{SEP}"
            f"{_fmt_num(overall_basic['acc'])}{SEP}"
            f"{_fmt_num(f1)}{SEP}"
            f"{_fmt_num(auroc)}{SEP}"
            f"{_fmt_num(aupr)}{SEP}"
            f"{_fmt_num(calib['ece'])}{SEP}"
            f"{eval_time:>{W_TIME}.1f}"
        )
        print(row)
        tpr_points = _tpr_points_str(fpr_targets, tpr_at_fpr)
        print(f"[MGTEval] tpr@fpr targets: [{tpr_points}] (rank_source={ranking_src})")

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
        "tpr_at_fpr_targets": [float(x) for x in fpr_targets],
        "tpr_at_fpr_rank_source": ranking_src,
        "tpr_at_fpr": tpr_at_fpr,
        "ci_95": ci,
        "timing": manifest["timing"],
        "memory": mem_stats,
        "k_runs": 1,
        "k_runs_stats": None,
        "k_runs_seeds": [sample_seed] if is_sampling else None,
        "meta": (res.meta if hasattr(res, "meta") else None),
        "asr": asr_results,
        "threshold_metrics": threshold_metrics or None,
    }

    if show_progress:
        print(f"[MGTEval] [4/4] saving artifacts (metrics/curves/figures/predictions)...")

    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(metrics_dir / "tpr_at_fpr.json", "w", encoding="utf-8") as f:
        json.dump({"targets": [float(x) for x in fpr_targets], "rank_source": ranking_src, "tpr_at_fpr": tpr_at_fpr},
                  f, ensure_ascii=False, indent=2)

    # group metric files
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

            if probs_seq is not None:
                cov_l, risk_l = _risk_coverage(labels_seq, probs_seq)
                with open(curves_dir / "rc_abstain_overall.json", "w", encoding="utf-8") as f:
                    json.dump({"coverage": cov_l, "risk": risk_l}, f, ensure_ascii=False, indent=2)
                _plot_curve(cov_l, risk_l, "Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / "rc_overall.png")
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

    # overall curves/plots
    if save_curves:
        with open(curves_dir / "roc_overall.json", "w", encoding="utf-8") as f:
            json.dump({"fpr": fpr, "tpr": tpr, "auroc": auroc, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
        with open(curves_dir / "pr_overall.json", "w", encoding="utf-8") as f:
            json.dump({"recall": rec, "precision": prec, "aupr": aupr, "rank_source": ranking_src}, f, ensure_ascii=False, indent=2)
        _plot_curve(fpr, tpr, f"ROC (overall) AUC={auroc:.3f}", "FPR", "TPR", figures_dir / "roc_overall.png")
        _plot_curve(rec, prec, f"PR (overall) AUPR={aupr:.3f}", "Recall", "Precision", figures_dir / "pr_overall.png")

        if probs_seq is not None:
            with open(curves_dir / "rc_abstain_overall.json", "w", encoding="utf-8") as f:
                json.dump({"coverage": cov, "risk": risk}, f, ensure_ascii=False, indent=2)
            _plot_curve(cov, risk, "Risk-Coverage (overall)", "Coverage", "Risk", figures_dir / "rc_overall.png")
            _plot_reliability(calib["bins"], figures_dir / "calibration_overall.png")

    # predictions.json
    preds_out = []
    for i, ex in enumerate(examples):
        raw_text = str(ex.get("text", "") or "").strip()
        words = raw_text.split()
        text_preview = " ".join(words[:10]) if words else ""
        rec_pred = {
            "text": text_preview,
            "label": int(labels_seq[i]),
            "prob": (float(probs_seq[i]) if probs_seq is not None else None),
            "score": (float(ranking_vec[i]) if ranking_src == "scores" else None),
            "pred": int(preds_seq[i]),
        }
        for gc in used_group_cols:
            rec_pred[gc] = ex.get(gc, None)
        rec_pred["id"] = ex.get("id", i)
        rec_pred["length"] = len(words)
        preds_out.append(rec_pred)

    with open(run_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)

    # model card
    card = {"detector_name": display_name, "detector_type": det_type}
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

    if show_progress:
        print(f"[MGTEval] results saved to: {str(run_dir)}")
    return res
