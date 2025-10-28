# mgt_eval/utils/calib_bootstrap.py
from __future__ import annotations
from typing import Optional, List, Tuple, Iterable
from pathlib import Path
import re
import json
from importlib import resources

from .paths import user_calib_dir, pkg_calib_dir


# -------------------- 内部工具 --------------------

def _norm_token(s: str) -> str:
    """标准化：仅保留 a-z0-9._-，转小写；用于文件名/模型名匹配。"""
    s = (s or "").strip().replace("\\", "/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._\-]+", "_", s)


def _canon(s: str) -> str:
    """进一步“极简”标准化（去掉非字母数字），用于鲁棒匹配。"""
    return re.sub(r"[^a-z0-9]+", "", s or "")


def _detector_variants(detector_key: str) -> List[str]:
    """
    给定检测器 key（如 'lastde' 或 'fast_detect_gpt'），返回一组可接受的名字变体：
    - 原样
    - '_' 与 '-' 互换
    - 去掉分隔符
    """
    key = _norm_token(detector_key)
    v = {
        key,
        key.replace("_", "-"),
        key.replace("-", "_"),
        key.replace("_", ""),
        key.replace("-", ""),
    }
    return sorted(v)


def _extract_model_hints_from_obj(obj) -> List[str]:
    """
    从检测器对象粗略提取模型名线索，避免 import 循环依赖。
    与 DetectorBase._collect_model_hints 的逻辑保持一致（独立实现）。
    """
    keys_of_interest = [
        # binoculars / lastde 等
        "observer", "observer_name", "observer_name_or_path",
        "performer", "performer_name", "performer_name_or_path",
        "scoring_name", "scoring_name_or_path",
        "model", "model_name", "model_name_or_path", "model_path",
        "tokenizer", "tokenizer_name", "tokenizer_name_or_path", "tokenizer_path",
        # Fast-DetectGPT
        "scoring_model_name", "sampling_model_name",
        "scoring_model_path", "sampling_model_path",
        "sampling_name_or_path",
        "reference_model",
        "mask_model", "rewrite_model",
        # baseline 系列
        "score_model",
    ]
    hints: List[str] = []
    # 先看实例属性
    try:
        for k, v in list(getattr(obj, "__dict__", {}).items()):
            if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                hints.append(v)
    except Exception:
        pass
    # 再看 kwargs（若存在）
    try:
        kwargs = getattr(obj, "kwargs", {})
        if isinstance(kwargs, dict):
            for k, v in list(kwargs.items()):
                if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                    hints.append(v)
    except Exception:
        pass
    return [_norm_token(x) for x in hints if isinstance(x, str) and x.strip()]


def _iter_user_jsons() -> Iterable[Path]:
    """用户目录下的所有候选 JSON 文件。"""
    udir = user_calib_dir()
    if not udir.exists():
        return []
    got = list(udir.glob("*.json")) + list(udir.glob("*.jsonl.json"))
    # 去重
    seen = set()
    out: List[Path] = []
    for p in got:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out


def _iter_pkg_json_names() -> Iterable[str]:
    """
    包内（read-only）校准资源的文件名迭代器（字符串文件名，不含路径）。
    """
    root = pkg_calib_dir()  # Traversable
    try:
        for p in root.iterdir():
            name = p.name
            if name.endswith(".json") or name.endswith(".jsonl.json"):
                yield name
    except Exception:
        return []


def _score_filename(
    filename: str,
    det_variants: List[str],
    model_tokens: List[str],
) -> int:
    """
    对候选文件名打分：
      +2 若文件名含有检测器名（任意变体）
      +1 若文件名含有任一模型名 token
    """
    stem = _norm_token(filename)
    toks = [t for t in re.split(r"[^a-z0-9]+", stem) if t]

    # 检测器匹配
    det_ok = False
    det_canon_set = {_canon(x) for x in det_variants}
    for t in toks:
        tc = _canon(t)
        if tc in det_canon_set:
            det_ok = True
            break
    if not det_ok:
        return -1  # 直接淘汰

    score = 2
    # 模型匹配（加分项）
    model_canon = {_canon(x) for x in model_tokens}
    for t in toks:
        if _canon(t) in model_canon:
            score += 1
            break
    return score


def _best_match_in_user(
    detector_key: str,
    model_hints: List[str],
) -> Optional[Path]:
    """在用户目录里找最佳匹配（若存在就返回路径）。"""
    det_vars = _detector_variants(detector_key)
    models = [_norm_token(x) for x in model_hints]
    best_path: Optional[Path] = None
    best_score = -1
    for p in _iter_user_jsons():
        sc = _score_filename(p.name, det_vars, models)
        if sc > best_score:
            best_score = sc
            best_path = p
    return best_path if best_score >= 0 else None


def _best_match_in_pkg(
    detector_key: str,
    model_hints: List[str],
) -> Optional[str]:
    """在包内置目录里找最佳匹配（返回文件名）。"""
    det_vars = _detector_variants(detector_key)
    models = [_norm_token(x) for x in model_hints]
    best_name: Optional[str] = None
    best_score = -1
    for name in _iter_pkg_json_names():
        sc = _score_filename(name, det_vars, models)
        if sc > best_score:
            best_score = sc
            best_name = name
    return best_name if best_score >= 0 else None


def _copy_pkg_to_user(filename: str, *, overwrite: bool = False) -> str:
    """
    将包内置的 calibration JSON 复制到用户目录，返回目标绝对路径字符串。
    若目标已存在且 overwrite=False，则直接返回现有路径。
    """
    udir = user_calib_dir()
    udir.mkdir(parents=True, exist_ok=True)
    dst = udir / filename

    if dst.exists() and not overwrite:
        return str(dst)

    # 从包资源读取并原样写出（不做任何修改）
    with resources.files("mgt_eval").joinpath("calibration_results").joinpath(filename).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # 简单校验（避免写入非 JSON 或结构异常）
    if not isinstance(payload, dict):
        raise ValueError(f"[MGTEval] Bad calibrator payload (not a dict): {filename}")
    if "calibrator" not in payload:
        # 有些历史文件可能直接是参数体，这里也允许，但推荐含 'calibrator'
        pass

    with dst.open("w", encoding="utf-8") as g:
        json.dump(payload, g, ensure_ascii=False, indent=2)

    return str(dst)


# -------------------- 对外 API --------------------

def ensure_calibrator_for(
    detector_key: str,
    model_hints: List[str] | Tuple[str, ...] = (),
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    """
    确保用户目录下存在一个适配 `detector_key` 与 `model_hints` 的校准 JSON。
    - 若用户目录已有匹配项：直接返回其路径；
    - 否则从包内置目录挑选最佳匹配，复制到用户目录，返回复制后的路径；
    - 若包内也没有可匹配项：返回 None。

    参数
    ----
    detector_key: 检测器 key，如 'lastde', 'binoculars', 'fast_detect_gpt', 'raidar' 等
    model_hints: 可能包含的模型名线索（会做归一化）
    overwrite: 复制时是否覆盖同名文件
    verbose: 是否打印提示

    返回
    ----
    str | None: 用户目录下校准器 JSON 的绝对路径
    """
    # 1) 先看用户目录是否已有合适 JSON
    user_hit = _best_match_in_user(detector_key, list(model_hints))
    if user_hit is not None:
        if verbose:
            print(f"[MGTEval] Found user calibrator: {user_hit}")
        return str(user_hit)

    # 2) 包内挑选最佳并复制
    pkg_name = _best_match_in_pkg(detector_key, list(model_hints))
    if pkg_name is None:
        if verbose:
            print(f"[MGTEval] No packaged calibrator matched detector='{detector_key}'.")
        return None

    out_path = _copy_pkg_to_user(pkg_name, overwrite=overwrite)
    if verbose:
        print(f"[MGTEval] Bootstrapped calibrator to user dir: {out_path}")
    return out_path


def ensure_calibrator_for_detector(
    detector_obj,
    *,
    override_builtin: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    """
    给“检测器实例”用的便捷函数。

    场景 A：你已经通过 _auto_find_calibrator_path() 得到了 'builtin:xxx.json'
      → 传入 override_builtin='xxx.json'，将精确复制该文件到用户目录并返回路径。

    场景 B：没有任何路径，但你想“尽力匹配”
      → 本函数会按 (检测器名 + 模型线索) 在包内寻找最佳文件，复制到用户目录并返回路径。

    返回用户目录下 JSON 的绝对路径；若失败则返回 None。
    """
    # 1) 若给了 builtin 精确文件名，直接复制
    if override_builtin:
        if verbose:
            print(f"[MGTEval] Using builtin calibrator: {override_builtin}")
        try:
            out_path = _copy_pkg_to_user(override_builtin, overwrite=overwrite)
            if verbose:
                print(f"[MGTEval] Bootstrapped calibrator to user dir: {out_path}")
            return out_path
        except FileNotFoundError:
            if verbose:
                print(f"[MGTEval] builtin file not found in package: {override_builtin}")
            # 回退到场景 B 的“尽力匹配”

    # 2) 没有指定具体文件名，做“尽力匹配”
    #    检测器 key：去掉尾部 'Detector'，再标准化
    cls = type(detector_obj).__name__
    base = re.sub(r"detector$", "", cls, flags=re.IGNORECASE)
    detector_key = _norm_token(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base))  # CamelCase -> snake-like

    model_hints = _extract_model_hints_from_obj(detector_obj)
    return ensure_calibrator_for(detector_key, model_hints, overwrite=overwrite, verbose=verbose)
