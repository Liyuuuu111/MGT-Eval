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

        # --- Binoculars 风格参数（作为回退方案）---
        self.prob_slope: float = float(kwargs.get("prob_slope", 8.0))
        self.prob_center: float = float(kwargs.get("prob_center", 0.0))
        self.prob_invert: bool = bool(kwargs.get("prob_invert", False))

        # 检测器原生是否输出概率（Metric-based 一律 False；若子类已输出概率则置 True）
        self.outputs_prob: bool = bool(kwargs.get("outputs_prob", False))

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

    # —— 统一评测：批量打分 → 概率（若需要）→ 预测 → 指标 —— #
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

        # ===== 日志/元信息 =====
        method_name = getattr(self, "DETECTOR_NAME", self.__class__.__name__)
        method_type = getattr(self, "detector_type", "Unknown")
        method_auth = getattr(self, "CITATION_AUTHORS", None)
        method_link = getattr(self, "CITATION_LINK", None)
        method_title = getattr(self, "CITATION_TITLE", None)
        device_hint = getattr(self, "device", None)

        print(f"[MGTEval] You are using detector: {method_name} (Detector's Type: {method_type})")
        if method_auth or method_link:
            print(f"[MGTEval] Credits: {method_auth or 'Unknown authors'} | Paper: {method_title} | Link: {method_link or 'N/A'}")
        print("[MGTEval] Disclaimer: This implementation may differ slightly from the original reference; "
              "results might not exactly match those reported in the paper.")
        if device_hint:
            print(f"[MGTEval] You are using device: {device_hint}")
        print(f"[MGTEval] Batch size: {batch_size}")

        # ===== 数据展开 =====
        texts: List[str] = []
        labels: List[int] = []
        for ex in dataset:
            texts.append(ex["text"])
            labels.append(int(ex["label"]))
        labels_np = np.array(labels, dtype=int)
        print(f"[MGTEval] Loaded {len(texts)} samples for evaluation.")

        # ===== 批量打分 =====
        t0 = time.perf_counter()
        bs = max(1, batch_size)
        total_batches = (len(texts) + bs - 1) // bs
        iterator = range(0, len(texts), bs)

        pbar = tqdm(
            iterator,
            total=total_batches,
            desc=f"Eval[{getattr(self, 'DETECTOR_NAME', self.__class__.__name__)}]",
            dynamic_ncols=True,
            disable=(not show_progress),
            leave=False, position=0, mininterval=0.5
        )

        all_scores: List[np.ndarray] = []
        expected_ndim: Optional[int] = None

        with torch.inference_mode():
            for start in pbar:
                s = self.score_batch(texts[start:start+bs])
                a = np.asarray(s, dtype=np.float64)   # 不 reshape，保留 (B,) 或 (B,D)
                if expected_ndim is None:
                    expected_ndim = a.ndim
                elif a.ndim != expected_ndim:
                    raise RuntimeError(
                        f"Inconsistent score shape across batches: expected ndim={expected_ndim}, got {a.ndim}"
                    )
                all_scores.append(a)

        infer_sec = time.perf_counter() - t0

        if all_scores:
            scores = np.concatenate(all_scores, axis=0)   # (N,) 或 (N,D)
        else:
            scores = np.zeros((0,), dtype=np.float64)

        infer_sec = time.perf_counter() - t0
        avg_infer_ms = (infer_sec / max(1, len(texts))) * 1e3
        throughput = (len(texts) / infer_sec) if infer_sec > 0 else 0.0
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float64)

        # ===== 概率（兼容“原生概率”与“分数→概率”）=====
        prob_mode = None
        if self.outputs_prob:
            probs = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6).astype(np.float32)
            prob_mode = "native_prob"
        else:
            if self.force_runner_calibration:
                # 1) 优先用已加载的 JSON 校准器（platt 或 linear）
                if self._calibrator_params is not None:
                    probs, used_mode = self._runner_apply_loaded_calibrator(np.asarray(scores))
                    prob_mode = used_mode
                # 2) 否则若有标签 → 在线拟合（Platt/Linear）
                elif labels_np is not None and len(labels_np) == len(scores):
                    probs, used_mode = self._runner_fit_and_apply_inline(np.asarray(scores), labels_np)
                    prob_mode = used_mode
                # 3) 最后兜底（仅 1D）
                else:
                    probs = self._fallback_sigmoid_1d_only(np.asarray(scores))
                    prob_mode = "binoculars_sigmoid"
            else:
                # 保留旧行为（可通过 force_runner_calibration=False 关闭）
                probs = self.calibrate(np.asarray(scores), labels_np)
                prob_mode = "learned_lr" if (self._calibrator_params is not None) else "binoculars_sigmoid"

        # 统一成 1-D
        probs = np.asarray(probs).reshape(-1).astype(np.float32)

        # ===== 预测与指标 =====
        preds = self.predict(probs, threshold=threshold).astype(int)
        metrics = compute_metrics(labels_np, probs, preds, tpr_at_fpr=tpr_at_fpr)

        # ===== 元信息 =====
        meta: Dict[str, Any] = {
            "detector": getattr(self, "DETECTOR_NAME", "detector"),
            "num_examples": len(texts),
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
            # 记录概率映射配置，便于审计
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

        # —— 统一返回 —— #
        scores_out = scores.astype(float).tolist()
        probs_out = probs.astype(float).tolist()

        return EvalResult(
            scores=scores_out,
            probs=probs_out,
            preds=preds.tolist(),
            labels=labels_np.tolist(),
            metrics=metrics,
            meta=meta,
        )
