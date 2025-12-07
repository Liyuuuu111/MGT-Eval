# mgt_eval/detectors/base.py
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple
import json
import time
import os
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from importlib import resources
from mgt_eval.utils.paths import user_calib_dir, pkg_calib_dir, dev_calib_dir_from
from mgt_eval.utils.calib_bootstrap import ensure_calibrator_for_detector

Text = str
Label = int  # 0 = human, 1 = AI

@dataclass
class EvalResult:
    scores: List[float]   # 原始分数（越大越像 AI，供 ROC/AUPR 排序用）
    probs: List[float]    # 概率（用于阈值/校准/ECE 等）
    preds: List[int]
    labels: List[int]
    meta: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self):
        return asdict(self)

# =========================
# Display helpers (presentation only)
# =========================
W_PHASE = 8
W_MEM   = 8
W_N     = 4
W_NUM   = 8
W_STEP  = 8
SEP     = " "

def _gpu_mem_str() -> str:
    """Current GPU reserved memory (GB-like, consistent with train.py style)."""
    try:
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
    except Exception:
        pass
    return "0G"

def _fmt_float(x, fmt: str = ".4f") -> str:
    if x is None:
        return "-"
    try:
        return format(float(x), fmt)
    except Exception:
        return "-"

def _summarize_thresholds(thrs: Dict[str, float], max_items: int = 8) -> str:
    if not thrs:
        return "-"
    items = sorted(thrs.items(), key=lambda kv: str(kv[0]))
    parts = []
    for k, v in items[:max_items]:
        try:
            parts.append(f"{k}={float(v):.4f}")
        except Exception:
            parts.append(f"{k}=?")
    if len(items) > max_items:
        parts.append("...")
    return ", ".join(parts)

def _tpr_at_fpr_str(metrics: Dict[str, Any], targets: Tuple[float, ...]) -> str:
    """
    Presentation-only: try to extract TPR@FPR from metrics in a robust way.
    Supports either:
      - metrics["tpr_at_fpr"] as dict, or
      - flat keys like "tpr@fpr=0.01", "tpr@fpr<=0.01", etc.
    """
    if not isinstance(metrics, dict) or not targets:
        return "-"

    # case 1: a dict field exists
    tdict = metrics.get("tpr_at_fpr", None)
    if isinstance(tdict, dict):
        parts = []
        for t in targets:
            v = None
            for k in (t, float(t), str(t), f"{t:g}", f"{t:.0e}"):
                if k in tdict:
                    v = tdict.get(k)
                    break
                # nested: {"0.01": {"tpr":..., "value":...}} etc
                if k in tdict and isinstance(tdict.get(k), dict):
                    cand = tdict[k]
                    for kk in ("tpr", "value"):
                        if isinstance(cand.get(kk), (int, float)):
                            v = cand[kk]
                            break
            parts.append(f"{t:g}->{_fmt_float(v, '.4f')}")
        return ", ".join(parts)

    # case 2: search flat keys
    parts = []
    for t in targets:
        v = None
        key_cands = [
            f"tpr@fpr={t}",
            f"tpr@fpr<={t}",
            f"tpr@fpr<{t}",
            f"tpr@fpr={t:g}",
            f"tpr@fpr<={t:g}",
            f"tpr@fpr<{t:g}",
            f"tpr@fpr={t:.0e}",
            f"tpr@fpr<={t:.0e}",
            f"tpr@fpr<{t:.0e}",
        ]
        for kc in key_cands:
            if kc in metrics and isinstance(metrics[kc], (int, float)):
                v = metrics[kc]
                break
        parts.append(f"{t:g}->{_fmt_float(v, '.4f')}")
    return ", ".join(parts)

class DetectorBase:
    """
    通用检测器基类：
      - 子类需实现 score_batch(texts) -> np.ndarray
      - 若子类输出的是原始分数（默认），本类会在 evaluate() 中调用 calibrate() 将分数映射为概率
      - calibrate() 优先使用 JSON 学习到的逻辑回归参数（platt_lr / linear_lr），否则回退到 Binoculars 风格 Sigmoid
    """
    DETECTOR_NAME: str = "base"

    def __init__(self, **kwargs):
        # === 新增：可选的校准参数（从 JSON 加载） ===
        self.calibrator_path: Optional[str] = kwargs.get("calibrator_path", None)
        # 缺省校准器名：一维分数 → Platt（逻辑回归）
        self.calibrator_name: str = kwargs.get("calibrator_name", "platt_lr")
        self._calibrator_params: Optional[Dict[str, Any]] = None
        # NEW: 显式禁止 runner 校准（给 finetuned / 已输出概率的 detector 用）
        # 允许子类通过 class attribute 设默认值，也允许 kwargs 覆盖
        _cls_disable = bool(getattr(self, "disable_calibration", False))
        self.disable_calibration: bool = bool(kwargs.get("disable_calibration", _cls_disable))
        # NEW: 保留完整校准 meta + 从 meta 中解析出来的“推荐阈值们”
        self._calibrator_full_meta: Optional[Dict[str, Any]] = None
        self._calibrator_thresholds: Dict[str, float] = {}
        # 推荐的单一决策阈值（如 dev 上按 acc/f1/tpr 模式选出来的阈值）
        self.decision_threshold: Optional[float] = None
        # --- Binoculars 风格参数（作为回退方案）---
        self.prob_slope: float = float(kwargs.get("prob_slope", 8.0))
        self.prob_center: float = float(kwargs.get("prob_center", 0.0))
        self.prob_invert: bool = bool(kwargs.get("prob_invert", False))

        # 检测器原生是否输出概率：支持子类 class 默认值 + kwargs 覆盖
        _cls_outputs = bool(getattr(self, "outputs_prob", False))
        self.outputs_prob: bool = bool(kwargs.get("outputs_prob", _cls_outputs))

        # 自动校准：默认开启；可通过构造参数关闭
        self.auto_calibrate: bool = bool(kwargs.get("auto_calibrate", True))
        self.force_runner_calibration: bool = bool(kwargs.get("force_runner_calibration", True))

        self.kwargs = kwargs
        self.is_loaded = False

    # ===== Runner-style helpers in base (Platt / Linear LR) =====
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _apply_platt_1d(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        if params.get("standardize", True):
            mu = float(params.get("mean", 0.0))
            sd = float(params.get("std", 1.0)) or 1.0
            x = (x - mu) / sd
        z = float(params["beta0"]) + float(params["beta1"]) * x
        return DetectorBase._sigmoid(z).astype(np.float32)

    @staticmethod
    def _apply_linear_lr(scores: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
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
        return DetectorBase._sigmoid(z).astype(np.float32)

    @staticmethod
    def _irls_fit_platt(scores: np.ndarray, labels: np.ndarray,
                        *, l2: float = 1e-2, max_iter: int = 200,
                        tol: float = 1e-6, standardize: bool = True) -> Dict[str, Any]:
        x = scores.astype(np.float64).reshape(-1)
        y = labels.astype(np.float64).reshape(-1)
        if standardize:
            mu = float(x.mean()); sd = float(x.std(ddof=0)); sd = 1.0 if abs(sd) < 1e-12 else sd
            xs = (x - mu) / sd
        else:
            mu, sd, xs = 0.0, 1.0, x
        n = xs.shape[0]
        X = np.c_[np.ones((n, 1), dtype=np.float64), xs.reshape(-1, 1)]
        beta = np.zeros(2, dtype=np.float64)
        lam = float(max(0.0, l2))
        for _ in range(int(max_iter)):
            z = X @ beta
            p = DetectorBase._sigmoid(z)
            w = np.clip(p * (1.0 - p), 1e-9, None)
            g = X.T @ (p - y); g[1] += lam * beta[1]
            Xw = X * w.reshape(-1, 1)
            H = X.T @ Xw; H[1, 1] += lam; H[0, 0] += 1e-8
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H + 1e-6*np.eye(2), g, rcond=None)[0]
            beta_new = beta - step
            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new; break
            beta = beta_new
        beta0, beta1 = float(beta[0]), float(beta[1])
        return {"name": "platt_lr", "beta0": beta0, "beta1": beta1,
                "standardize": bool(standardize), "mean": float(mu), "std": float(sd)}

    @staticmethod
    def _irls_fit_linear_lr(X: np.ndarray, y: np.ndarray,
                            *, l2: float = 1e-2, max_iter: int = 200,
                            tol: float = 1e-6, standardize: bool = True) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        n, d = X.shape
        if standardize:
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=0); sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
            Xs = (X - mu) / sd
        else:
            mu = np.zeros(d, dtype=np.float64); sd = np.ones(d, dtype=np.float64); Xs = X
        Xmat = np.concatenate([np.ones((n, 1), dtype=np.float64), Xs], axis=1)
        beta = np.zeros(d + 1, dtype=np.float64)
        lam = float(max(0.0, l2))
        reg = np.diag([1e-8] + [lam] * d)
        for _ in range(int(max_iter)):
            z = Xmat @ beta
            p = DetectorBase._sigmoid(z)
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
                beta = beta_new; break
            beta = beta_new
        return {"name": "linear_lr", "beta": beta.tolist(),
                "standardize": bool(standardize),
                "mean": mu.tolist(), "std": sd.tolist()}

    def _runner_apply_loaded_calibrator(self, scores: np.ndarray) -> Tuple[np.ndarray, str]:
        if self._calibrator_params is None:
            raise RuntimeError("[MGTEval] No calibrator params loaded.")
        name = str(self._calibrator_params.get("name", "platt_lr")).lower()
        X = np.asarray(scores)
        if name in ("platt_lr", "platt", "plattlr"):
            if X.ndim != 1:
                # 容错：有些 JSON 写错了名字但存的是 linear 的参数
                if "beta" in self._calibrator_params:
                    probs = self._apply_linear_lr(X, self._calibrator_params)
                    return probs, "learned_linear_lr"
                raise RuntimeError("[MGTEval] platt_lr expects 1-D scores.")
            probs = self._apply_platt_1d(X, self._calibrator_params)
            return probs, "learned_platt_lr"
        else:
            # 统一按线性 LR 处理
            probs = self._apply_linear_lr(X, self._calibrator_params)
            return probs, "learned_linear_lr"
        
    def _runner_fit_and_apply_inline(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, str]:
        X = np.asarray(scores)
        if X.ndim == 1:
            params = self._irls_fit_platt(X, labels, l2=1e-2, max_iter=200, tol=1e-6, standardize=True)
            probs = self._apply_platt_1d(X, params)
            return probs, "inline_platt_lr"
        else:
            params = self._irls_fit_linear_lr(X, labels, l2=1e-2, max_iter=200, tol=1e-6, standardize=True)
            probs = self._apply_linear_lr(X, params)
            return probs, "inline_linear_lr"

    def _fallback_sigmoid_1d_only(self, scores: np.ndarray) -> np.ndarray:
        x = np.asarray(scores)
        if x.ndim != 1:
            raise RuntimeError(
                f"[MGTEval] Multi-dimensional scores {x.shape} without calibrator/labels. "
                f"Provide a calibrator JSON or labels for inline LR."
            )
        x1d = -x if self.prob_invert else x
        x1d = np.clip(x1d, -1e6, 1e6)
        p = 1.0 / (1.0 + np.exp(-self.prob_slope * (x1d - self.prob_center)))
        return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

    # ----------------- 自动查找校准器 -----------------
    @staticmethod
    def _norm_token(s: str) -> str:
        """
        归一化文件名/标识用于匹配：
          - 仅保留字母数字和连字符/下划线
          - 全部小写
        """
        s = (s or "").strip()
        # 替换斜杠为下划线（HF repo e.g. EleutherAI/gpt-neo-2.7B）
        s = s.replace("\\", "/")
        s = s.split("/")[-1]  # basename-like
        s = s.lower()
        s = re.sub(r"[^a-z0-9._\-]+", "_", s)
        return s

    def _word_match(self, haystack: str, needle: str) -> bool:
        """
        在 haystack 中用“词边界”匹配 needle：
        - 左右两侧都不能是字母或数字（避免 'lastde' 命中 'lastdepp'）
        - 传入的 haystack/needle 应该是 _norm_token 之后的字符串
        """
        if not haystack or not needle:
            return False
        s = haystack.lower()
        n = needle.lower()
        return re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", s) is not None

    def _algo_key_from_meta(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        从校准 JSON 里提取算法 key：
        - 优先 meta.detector_key；否则用 meta.detector / 顶层 detector
        - 去掉尾部的 'Detector' 再规范化
        例：'LogRankDetector[GPT-Neo-2.7B]' -> 'logrank'
        """
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        cand = None
        for k in ("detector_key", "detector"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                cand = v.strip()
                break
        if cand is None:
            v = payload.get("detector")
            if isinstance(v, str) and v.strip():
                cand = v.strip()
        if not cand:
            return None
        cand = cand.split("[", 1)[0]  # 去掉 [Model] 尾缀
        cand = re.sub(r"detector$", "", cand, flags=re.IGNORECASE)  # 去掉 Detector 后缀
        # 归一化成 token
        return self._norm_token(cand)


    def _calibrators_dir(self) -> Path:
        """
        缺省校准目录：<repo_root>/calibration_results
        若不存在则回退到 <repo_root>/calibrators（可选）
        """
        here = Path(__file__).resolve()
        mgt_eval_root = here.parent.parent.parent  # mgt_eval/
        primary = mgt_eval_root / "calibration_results"
        print(f"[MGTEval] Attempt to get params in {primary} dir.")
        if primary.exists():
            return primary
        fallback = mgt_eval_root / "calibrators"
        if fallback.exists():
            print(f"[MGTEval] calibration_results not found, fallback to {fallback}")
            return fallback
        return primary  # 即使不存在也返回，后续会给出提示
    
    def _collect_model_hints(self) -> List[str]:
        hints: List[str] = []
        keys_of_interest = [
            # binoculars / lastde 等
            "observer", "observer_name", "observer_name_or_path",
            "performer", "performer_name", "performer_name_or_path",
            "scoring_name", "scoring_name_or_path",
            "model", "model_name", "model_name_or_path", "model_path",
            "tokenizer", "tokenizer_name", "tokenizer_name_or_path", "tokenizer_path",
            # FastDetectGPT
            "scoring_model_name", "sampling_model_name",
            "scoring_model_path", "sampling_model_path",
            "sampling_name_or_path",
            # ★ baseline 系列需要这个：
            "score_model",
        ]
        for k, v in list(self.__dict__.items()):
            if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                hints.append(v)
        for k, v in list(self.kwargs.items()):
            if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                hints.append(v)
        return hints

    from importlib import resources
    from mgt_eval.utils.paths import user_calib_dir, pkg_calib_dir, dev_calib_dir_from

    def _auto_find_calibrator_path(self) -> Optional[str]:
        """
        搜索顺序：
        1) 环境变量/用户数据目录  (~/.local/share/mgt_eval/calibration_results 或同等)
        2) 包内内置校准器 (read-only)
        3) 开发树 <repo_root>/calibration_results （若存在）
        返回：
        - 文件系统路径字符串，或
        - 形如 'builtin:filename.json' 的伪路径（表示包内资源）
        """
        # ---- 构造候选目录 ----
        dirs_fs = []      # 文件系统 Path
        dirs_pkg = None   # 包资源 Traversable

        # 用户数据目录
        udir = user_calib_dir()
        if udir.exists():
            dirs_fs.append(udir)

        # 包内资源
        dirs_pkg = pkg_calib_dir()

        # 开发树 fallback
        dev = dev_calib_dir_from(Path(__file__))
        if dev.exists():
            dirs_fs.append(dev)

        # ---- 收集候选文件（去重）----
        def iter_fs_jsons(root: Path):
            if not root.exists():
                return []
            got = list(root.glob("*.json")) + list(root.glob("*.jsonl.json"))
            # 去重
            seen = set()
            out = []
            for p in got:
                if p.name not in seen:
                    seen.add(p.name)
                    out.append(p)
            return out

        cand = []
        for d in dirs_fs:
            for p in iter_fs_jsons(d):
                cand.append(("fs", p))

        if dirs_pkg is not None:
            try:
                for p in dirs_pkg.iterdir():
                    name = p.name
                    if not (name.endswith(".json") or name.endswith(".jsonl.json")):
                        continue
                    cand.append(("pkg", name))  # 只记录文件名；读取时再用 resources 打开
            except Exception:
                pass

        if not cand:
            return None

        # ---- 评分（沿用你原来的匹配逻辑，略精简）----
        import re
        def norm(s: str) -> str:
            s = s.replace("\\", "/").split("/")[-1].lower()
            return re.sub(r"[^a-z0-9._\-]+", "_", s)

        cls_name = type(self).__name__
        base = re.sub(r"detector$", "", cls_name, flags=re.IGNORECASE)
        det_key = norm(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base))  # 'lastde' / 'fast_detect_gpt' 等
        det_variants = {det_key, det_key.replace("_", "-"), det_key.replace("-", "_"), det_key.replace("_", "")}
        def canon(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", s)

        model_tokens = []
        try:
            model_tokens = [norm(t) for t in self._collect_model_hints()]
        except Exception:
            pass
        model_canon = {canon(t) for t in model_tokens}

        best = None
        best_score = -1

        for kind, v in cand:
            name = v.name if kind == "fs" else v  # fs: Path；pkg: str
            stem = norm(name)
            toks = [t for t in re.split(r"[^a-z0-9]+", stem) if t]

            file_algo_ok = any((t in det_variants) or (canon(t) in {canon(x) for x in det_variants}) for t in toks)
            if not file_algo_ok:
                continue

            score = 2
            if any((t in model_tokens) or (canon(t) in model_canon) for t in toks):
                score += 1

            if score > best_score:
                best_score = score
                best = (kind, v)

        if best is None:
            return None
        kind, v = best
        if kind == "fs":
            return str(v)  # 文件系统绝对路径
        else:
            return f"builtin:{v}"  # 包内资源



    def load(self):
        self.is_loaded = True

        # ===== 1) 显式路径优先 =====
        if self.calibrator_path:
            try:
                # 支持普通文件路径 或 'builtin:xxx.json'
                self.load_calibrator(self.calibrator_path)
                print(f"[MGTEval] Loaded calibrator from: {self.calibrator_path}")
                return
            except Exception as e:
                print(f"[MGTEval] WARNING: failed to load calibrator '{self.calibrator_path}': {e}")

        # ===== 2) 自动发现 + bootstrap 到用户目录 =====
        try:
            det_type = (getattr(self, "detector_type", "") or "").strip().lower()
            need_auto = self.auto_calibrate and (det_type == "metric-based") and (not self.outputs_prob)
            if not need_auto:
                return

            auto_path = self._auto_find_calibrator_path()
            if auto_path:
                # 2a) 若是包内资源（builtin:xxx.json），先复制到用户目录再加载
                if isinstance(auto_path, str) and auto_path.startswith("builtin:"):
                    name = auto_path.split(":", 1)[1]  # e.g. 'calibrator_xxx.json'
                    copied = ensure_calibrator_for_detector(self, override_builtin=name, verbose=True)
                    if copied:
                        self.load_calibrator(copied)
                        self.calibrator_path = copied
                        self.kwargs["calibrator_path"] = copied
                        print(f"[MGTEval] Auto-loaded calibrator from: {copied}")
                        return
                    else:
                        # 极端兜底：如果复制失败，直接用 builtin 读（仍可工作，但不可写）
                        self.load_calibrator(auto_path)
                        self.calibrator_path = auto_path
                        self.kwargs["calibrator_path"] = auto_path
                        print(f"[MGTEval] Auto-loaded calibrator from builtin: {auto_path}")
                        return
                else:
                    # 2b) 文件系统路径：直接加载
                    self.load_calibrator(auto_path)
                    self.calibrator_path = auto_path
                    self.kwargs["calibrator_path"] = auto_path
                    print(f"[MGTEval] Auto-loaded calibrator from: {auto_path}")
                    return

            # 2c) 完全找不到：尝试“尽力匹配”从包内挑一个最合适的复制到用户目录
            copied = ensure_calibrator_for_detector(self, verbose=True)
            if copied:
                self.load_calibrator(copied)
                self.calibrator_path = copied
                self.kwargs["calibrator_path"] = copied
                print(f"[MGTEval] Auto-loaded calibrator from: {copied}")
            else:
                print("[MGTEval] No calibrator found or packaged default to bootstrap; will use sigmoid fallback.")
        except Exception as e:
            print(f"[MGTEval] WARNING: auto-calibrate probing/bootstrapping error: {e}")


    def load_calibrator(self, source: str | dict):
        from importlib import resources
        if isinstance(source, dict):
            payload = source
        elif isinstance(source, str) and source.startswith("builtin:"):
            name = source.split(":", 1)[1]
            with resources.files("mgt_eval").joinpath("calibration_results").joinpath(name)\
                    .open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            with open(str(source), "r", encoding="utf-8") as f:
                payload = json.load(f)

        # NEW: 记录完整 meta，并从 meta.dev 中解析各种阈值字段
        if isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict):
            self._calibrator_full_meta = payload["meta"]
            dev_meta = (self._calibrator_full_meta.get("dev") or {}) if isinstance(self._calibrator_full_meta, dict) else {}
            thresholds: Dict[str, float] = {}

            # 1) dev.decision：常见结构：
            #    {
            #      "mode": "acc" / "f1" / "tpr",
            #      "threshold": 0.73,
            #      "thresholds": {
            #          "acc":      0.70,
            #          "f1":       0.72,
            #          "tpr@0.01": 0.80,
            #          ...
            #      }
            #    }
            decision = (dev_meta.get("decision") or {}) if isinstance(dev_meta, dict) else {}
            thr_main = decision.get("threshold", None)
            if isinstance(thr_main, (int, float)):
                thresholds["decision"] = float(thr_main)

            dec_thrs = decision.get("thresholds") or {}
            if isinstance(dec_thrs, dict):
                for name, val in dec_thrs.items():
                    v = None
                    if isinstance(val, (int, float)):
                        v = float(val)
                    elif isinstance(val, dict):
                        # 兼容 {"thr": x} 或 {"threshold": x}
                        if isinstance(val.get("thr"), (int, float)):
                            v = float(val["thr"])
                        elif isinstance(val.get("threshold"), (int, float)):
                            v = float(val["threshold"])
                    if v is not None:
                        thresholds[str(name)] = v

            # 2) dev.tpr_at_fpr: {"0.01": {"threshold": ...}, ...}
            tpr_at_fpr = dev_meta.get("tpr_at_fpr") or {}
            if isinstance(tpr_at_fpr, dict):
                for fpr_k, info in tpr_at_fpr.items():
                    if not isinstance(info, dict):
                        continue
                    v = info.get("threshold", None)
                    if isinstance(v, (int, float)):
                        key = f"tpr@fpr<={fpr_k}"
                        thresholds[key] = float(v)

            # 保存解析结果
            self._calibrator_thresholds = thresholds

            # 3) 取出一个“首选决策阈值”
            self.decision_threshold = None
            if isinstance(thr_main, (int, float)):
                # 直接使用 dev.decision.threshold
                self.decision_threshold = float(thr_main)
            else:
                # 没有显式决策阈值时，优先用 FPR<=0.01 的点
                for candidate_key in ("tpr@fpr<=0.01", "tpr@fpr<=0.010", "tpr@fpr<=1e-02"):
                    if candidate_key in thresholds:
                        self.decision_threshold = thresholds[candidate_key]
                        break

        # 原有逻辑：提取参数
        params = payload["calibrator"] if isinstance(payload, dict) and "calibrator" in payload else payload
        if "name" not in params:
            params["name"] = self.calibrator_name
        self._calibrator_params = params

    # —— 子类必须实现：返回“原始分数”或“概率”（由 self.outputs_prob 决定）——
    def score_batch(self, texts: List[Text]) -> np.ndarray:
        """
        返回形状 (B,) 的 1D 向量：
          - 若 self.outputs_prob=False（默认）：返回“原始分数”（越大越像 AI）
          - 若 self.outputs_prob=True：返回“概率”（位于 [0,1]）
        """
        raise NotImplementedError

    # —— 概率映射（优先用学习到的 LR；否则固定 Sigmoid 回退）——
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        统一概率映射：
          1) 若已加载校准参数：
             - name == 'platt_lr'  : 期望 1D 分数 → 概率
             - 其它（如 'linear_lr'): 支持多维特征，调用注册校准器的 apply
          2) 否则回退到 Binoculars 的固定 Sigmoid（仅 1D 合理；多维则拍平并给出警告）
        """
        x = np.asarray(scores, dtype=np.float64)

        # —— 首选：已存在逻辑回归/线性 LR 校准 —— #
        if self._calibrator_params is not None:
            try:
                name = str(self._calibrator_params.get("name", self.calibrator_name)).lower()
                if name in ("platt_lr", "platt", "plattlr"):
                    # 1D Platt
                    x1d = x.reshape(-1)
                    b0 = float(self._calibrator_params["beta0"])
                    b1 = float(self._calibrator_params["beta1"])
                    if bool(self._calibrator_params.get("standardize", True)):
                        mu = float(self._calibrator_params.get("mean", 0.0))
                        sd = float(self._calibrator_params.get("std", 1.0)) or 1.0
                        x1d = (x1d - mu) / sd
                    z = np.clip(b0 + b1 * x1d, -80.0, 80.0)
                    p = 1.0 / (1.0 + np.exp(-z))
                    return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

                # 其它自定义校准器（多维线性 LR 等）
                from ..calibration.registry import get_calibrator
                cal = get_calibrator(name)
                p = cal["apply"](x, self._calibrator_params)  # 支持多维
                p = np.asarray(p, dtype=np.float64)
                return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)
            except Exception as e:
                print(f"[MGTEval] WARNING: calibrate() using learned params failed, fallback to sigmoid. err={e}")

        # —— 回退：Binoculars 固定 Sigmoid —— #
        x1d = x.reshape(-1)
        if x.ndim > 1:
            print("[MGTEval] WARNING: no calibrator loaded for multi-feature scores; "
                  "falling back to Binoculars sigmoid on the flattened 1D scores.")
        # 方向：若你的分数“越小越像 AI”，请在检测器里把 self.prob_invert=True
        if self.prob_invert:
            x1d = -x1d
        x1d = np.clip(x1d, -1e6, 1e6)
        p = 1.0 / (1.0 + np.exp(-self.prob_slope * (x1d - self.prob_center)))
        return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

    # —— 二分类预测（基于概率阈值）——
    def predict(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (probs >= threshold).astype(int)

    def evaluate(
        self,
        dataset: Iterable[Dict[str, Any]],
        batch_size: int = 8,
        threshold: float = 0.5,
        tpr_at_fpr: Tuple[float, ...] = (0.01, 0.05, 0.10),
        show_progress: bool = True,
    ) -> EvalResult:
        from ..metrics.metrics import compute_metrics

        if not self.is_loaded:
            self.load()

        # ===== 基本信息提示（presentation only）=====
        method_name = getattr(self, "DETECTOR_NAME", self.__class__.__name__)
        method_type = getattr(self, "detector_type", "Unknown")
        device_hint = getattr(self, "device", None)

        print(f"[MGTEval] Detector: {method_name} | type={method_type} | outputs_prob={bool(self.outputs_prob)}")
        if device_hint:
            print(f"[MGTEval] Device: {device_hint}")
        print(f"[MGTEval] Eval config: batch_size={int(batch_size)} | threshold={float(threshold):.4f} | TPR@FPR targets={list(tpr_at_fpr)}")

        # 校准器提示
        cal_name = (self._calibrator_params or {}).get("name") if self._calibrator_params else None
        print(f"[MGTEval] Calibration: auto={bool(self.auto_calibrate)} | force_runner={bool(self.force_runner_calibration)} | calibrator={cal_name or 'None'}")
        if getattr(self, "calibrator_path", None):
            print(f"[MGTEval] Calibrator path: {self.calibrator_path}")
        if getattr(self, "_calibrator_thresholds", None):
            print(f"[MGTEval] Recommended thresholds: {_summarize_thresholds(self._calibrator_thresholds)}")
        if getattr(self, "decision_threshold", None) is not None and abs(float(threshold) - float(self.decision_threshold)) > 1e-9:
            print(f"[MGTEval] Note: decision_threshold={float(self.decision_threshold):.4f} (you are using threshold={float(threshold):.4f}).")

        # ===== 数据展开 =====
        texts: List[str] = []
        labels: List[int] = []
        for ex in dataset:
            texts.append(ex["text"])
            labels.append(int(ex["label"]))
        labels_np = np.array(labels, dtype=int)

        n_total = len(texts)
        n_pos = int(np.sum(labels_np == 1))
        n_neg = int(np.sum(labels_np == 0))
        print(f"[MGTEval] Loaded {n_total} samples (AI=1:{n_pos}, Human=0:{n_neg}).")

        # ===== 批量打分（只改 tqdm 展示）=====
        t0 = time.perf_counter()
        bs = max(1, int(batch_size))
        total_batches = (n_total + bs - 1) // bs
        iterator = range(0, n_total, bs)

        if show_progress:
            # 表头（train.py风格）
            print("\n" +
                f"{'Phase':>{W_PHASE}}{SEP}"
                f"{'GPU_mem':>{W_MEM}}{SEP}{SEP}{SEP}{SEP}"
                f"{'done':>{W_N}}{SEP}{SEP}"
                f"{'eps':>{W_NUM}}{SEP}{SEP}"
                f"{'ms/ex':>{W_NUM}}{SEP}{SEP}{SEP}{SEP}"
                f"{'batch':>{W_STEP}}")

        pbar = tqdm(
            iterator,
            total=total_batches,
            desc=f"Eval[{method_name}]",
            dynamic_ncols=True,
            disable=(not show_progress),
            leave=True,
            mininterval=0.5,
        )

        all_scores: List[np.ndarray] = []
        expected_ndim: Optional[int] = None

        with torch.inference_mode():
            for bidx, start in enumerate(pbar, start=1):
                s = self.score_batch(texts[start:start + bs])
                a = np.asarray(s, dtype=np.float64)  # keep (B,) or (B,D)
                if expected_ndim is None:
                    expected_ndim = a.ndim
                elif a.ndim != expected_ndim:
                    raise RuntimeError(
                        f"Inconsistent score shape across batches: expected ndim={expected_ndim}, got {a.ndim}"
                    )
                all_scores.append(a)

                # ---- tqdm 描述（presentation only）----
                if show_progress:
                    done = min(start + bs, n_total)
                    elapsed = time.perf_counter() - t0
                    eps = (done / elapsed) if elapsed > 0 else 0.0
                    ms = (1000.0 / eps) if eps > 0 else 0.0
                    mem = _gpu_mem_str()
                    desc = (
                        f"{'Eval':>{W_PHASE}}{SEP}"
                        f"{mem:>{W_MEM}}{SEP}"
                        f"{done:>{W_N}d}/{n_total:<{W_N}d}{SEP}"
                        f"{eps:>{W_NUM}.2f}{SEP}"
                        f"{ms:>{W_NUM}.2f}{SEP}"
                        f"{bidx:>{W_STEP}d}/{total_batches:<{W_STEP}d}"
                    )
                    pbar.set_description(desc)

        infer_sec = time.perf_counter() - t0

        if all_scores:
            scores = np.concatenate(all_scores, axis=0)  # (N,) or (N,D)
        else:
            scores = np.zeros((0,), dtype=np.float64)

        avg_infer_ms = (infer_sec / max(1, n_total)) * 1e3
        throughput = (n_total / infer_sec) if infer_sec > 0 else 0.0
        print(f"[MGTEval] Scoring done: time={infer_sec:.3f}s | avg={avg_infer_ms:.3f} ms/ex | throughput={throughput:.2f} ex/s")

        # ===== 概率映射（只补充 mode 提示，不改逻辑）=====
        prob_mode = None
        if self.outputs_prob or method_type in {"Model-based"}:
            probs = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6).astype(np.float32)
            prob_mode = "native_prob"
        else:
            if self.force_runner_calibration:
                if self._calibrator_params is not None:
                    probs, used_mode = self._runner_apply_loaded_calibrator(np.asarray(scores))
                    prob_mode = used_mode
                elif labels_np is not None and len(labels_np) == len(scores):
                    probs, used_mode = self._runner_fit_and_apply_inline(np.asarray(scores), labels_np)
                    prob_mode = used_mode
                else:
                    probs = self._fallback_sigmoid_1d_only(np.asarray(scores))
                    prob_mode = "binoculars_sigmoid"
            else:
                probs = self.calibrate(np.asarray(scores), labels_np)
                prob_mode = "learned_lr" if (self._calibrator_params is not None) else "binoculars_sigmoid"

        probs = np.asarray(probs).reshape(-1).astype(np.float32)
        print(f"[MGTEval] Prob mapping mode: {prob_mode}")

        # ===== 预测与指标（不改计算，只改展示）=====
        preds = self.predict(probs, threshold=threshold).astype(int)
        metrics = compute_metrics(labels_np, probs, preds, tpr_at_fpr=tpr_at_fpr)

        # 只展示常用几项（存在则显示）
        acc = metrics.get("acc", None)
        f1  = metrics.get("f1", None)
        auroc = metrics.get("auroc", None)
        aupr  = metrics.get("aupr", None)
        ece   = metrics.get("ece", None)
        brier = metrics.get("brier", None)
        tpr_line = _tpr_at_fpr_str(metrics, tpr_at_fpr)

        print(
            "[MGTEval] Metrics: "
            f"Acc={_fmt_float(acc, '.4f')} | "
            f"F1={_fmt_float(f1, '.4f')} | "
            f"AUROC={_fmt_float(auroc, '.4f')} | "
            f"AUPR={_fmt_float(aupr, '.4f')} | "
            f"ECE={_fmt_float(ece, '.4f')} | "
            f"Brier={_fmt_float(brier, '.4f')}"
        )
        print(f"[MGTEval] TPR@FPR: {tpr_line}")

        # ===== meta（不改字段结构，只补充展示相关可审计信息）=====
        meta: Dict[str, Any] = {
            "detector": getattr(self, "DETECTOR_NAME", "detector"),
            "num_examples": n_total,
            "threshold": threshold,
            "tpr_at_fpr_targets": list(tpr_at_fpr),
            "kwargs": self.kwargs,
            "type": getattr(self, "detector_type", "Unknown"),
            "model_name": getattr(self, "name", getattr(self, "DETECTOR_NAME", "detector")),
            "timing": {
                "total_infer_sec": round(float(infer_sec), 6),
                "avg_infer_ms": round(float(avg_infer_ms), 3),
                "throughput_samples_per_sec": round(float(throughput), 3),
                "batch_size": int(batch_size),
                **({"device": getattr(self, "device", None)} if hasattr(self, "device") else {}),
            },
            "prob_mapping": {
                "mode": prob_mode,
                "slope": float(self.prob_slope),
                "center": float(self.prob_center),
                "invert": bool(self.prob_invert),
                "calibrator_params_present": bool(self._calibrator_params is not None),
                "calibrator_name": (self._calibrator_params or {}).get("name") if self._calibrator_params else None,
                "calibrator_path": self.calibrator_path,
            },
        }

        if getattr(self, "_calibrator_full_meta", None) is not None:
            meta["calibrator_meta"] = self._calibrator_full_meta
        if getattr(self, "_calibrator_thresholds", None):
            meta.setdefault("thresholds", {})
            for k, v in self._calibrator_thresholds.items():
                if k not in meta["thresholds"]:
                    meta["thresholds"][k] = float(v)
        if getattr(self, "decision_threshold", None) is not None:
            meta.setdefault("thresholds", {})
            if "decision" not in meta["thresholds"]:
                meta["thresholds"]["decision"] = float(self.decision_threshold)

        # ===== return（不改结构）=====
        scores_out = np.asarray(scores).astype(float).tolist()
        probs_out = np.asarray(probs).astype(float).tolist()

        return EvalResult(
            scores=scores_out,
            probs=probs_out,
            preds=preds.tolist(),
            labels=labels_np.tolist(),
            metrics=metrics,
            meta=meta,
        )

