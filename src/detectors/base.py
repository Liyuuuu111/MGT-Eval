# mgt_eval/detectors/base.py
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple
import json
import time
import os
import re
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from utils.paths import user_calib_dir, pkg_calib_dir, dev_calib_dir_from
from utils.calib_bootstrap import ensure_calibrator_for_detector

Text = str
Label = int  # 0 = human, 1 = AI

@dataclass
class EvalResult:
    scores: List[float]   # Raw scores (higher = more AI-like; used for ROC/AUPR ranking)
    probs: List[float]    # Probabilities (for thresholds/calibration/ECE, etc.)
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

def _color_num(val, color: str = "33") -> str:
    """
    Highlight numeric values with ANSI color when stdout is a TTY.
    Default color: yellow (33).
    """
    s = str(val)
    try:
        if sys.stdout.isatty():
            return f"\033[{color}m{s}\033[0m"
    except Exception:
        pass
    return s

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

def _print_method_citation(obj) -> None:
    """
    Print paper/authors/disclaimer if available on the detector class.
    """
    try:
        if getattr(obj, "_mgt_eval_citation_logged", False):
            return
        cls = obj if isinstance(obj, type) else obj.__class__
        if getattr(cls, "_mgt_eval_citation_logged", False):
            return
        title = getattr(cls, "CITATION_TITLE", None)
        authors = getattr(cls, "CITATION_AUTHORS", None)
        link = getattr(cls, "CITATION_LINK", None)
        if title or authors or link:
            print(
                "[MGTEval] Credits: "
                f"{authors or 'Unknown authors'} | "
                f"Paper: {title or 'N/A'} | "
                f"Link: {link or 'N/A'}"
            )
        print(
            "[MGTEval] Disclaimer: This implementation may differ slightly from the original reference; "
            "results might not exactly match those reported in the paper."
        )
        try:
            setattr(obj, "_mgt_eval_citation_logged", True)
        except Exception:
            pass
        try:
            setattr(cls, "_mgt_eval_citation_logged", True)
        except Exception:
            pass
    except Exception:
        # Keep silent if anything goes wrong
        pass

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
    Generic detector base class:
      - Subclasses must implement score_batch(texts) -> np.ndarray
      - If the subclass outputs raw scores (default), evaluate() calls calibrate() to map scores to probabilities
      - calibrate() prefers learned LR params from JSON (platt_lr / linear_lr), otherwise falls back to Binoculars-style sigmoid
    """
    DETECTOR_NAME: str = "base"

    def __init__(self, **kwargs):
        # Optional: calibrator params loaded from JSON
        self.calibrator_path: Optional[str] = kwargs.get("calibrator_path", None)
        # Default calibrator name: 1D scores → Platt (logistic regression)
        self.calibrator_name: str = kwargs.get("calibrator_name", "platt_lr")
        self._calibrator_params: Optional[Dict[str, Any]] = None
        # NEW: explicitly disable runner calibration (for finetuned / probability-output detectors)
        # Allow subclasses to set defaults via class attribute; kwargs can override.
        _cls_disable = bool(getattr(self, "disable_calibration", False))
        self.disable_calibration: bool = bool(kwargs.get("disable_calibration", _cls_disable))
        # NEW: keep full calibrator meta + parse recommended thresholds from meta
        self._calibrator_full_meta: Optional[Dict[str, Any]] = None
        self._calibrator_thresholds: Dict[str, float] = {}
        # Recommended single decision threshold (e.g., selected on dev by acc/f1/tpr)
        self.decision_threshold: Optional[float] = None
        # --- Binoculars-style params (fallback) ---
        self.prob_slope: float = float(kwargs.get("prob_slope", 8.0))
        self.prob_center: float = float(kwargs.get("prob_center", 0.0))
        self.prob_invert: bool = bool(kwargs.get("prob_invert", False))

        # Whether the detector outputs probabilities natively: supports class defaults + kwargs override
        _cls_outputs = bool(getattr(self, "outputs_prob", False))
        self.outputs_prob: bool = bool(kwargs.get("outputs_prob", _cls_outputs))

        # Auto-calibration: on by default; can be disabled via ctor args
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
                # Tolerant: some JSONs mislabel but store linear params
                if "beta" in self._calibrator_params:
                    probs = self._apply_linear_lr(X, self._calibrator_params)
                    return probs, "learned_linear_lr"
                raise RuntimeError("[MGTEval] platt_lr expects 1-D scores.")
            probs = self._apply_platt_1d(X, self._calibrator_params)
            return probs, "learned_platt_lr"
        else:
            # Treat everything else as linear LR
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

    # ----------------- Auto-find calibrator -----------------
    @staticmethod
    def _norm_token(s: str) -> str:
        """
        Normalize file name / identifier for matching:
          - keep only alphanumerics and dash/underscore
          - lowercase
        """
        s = (s or "").strip()
        # Replace slashes with underscores (HF repo e.g. EleutherAI/gpt-neo-2.7B)
        s = s.replace("\\", "/")
        s = s.split("/")[-1]  # basename-like
        s = s.lower()
        s = re.sub(r"[^a-z0-9._\-]+", "_", s)
        return s

    def _word_match(self, haystack: str, needle: str) -> bool:
        """
        Match needle in haystack using word boundaries:
        - neither side can be alphanumeric (avoid matching 'lastde' in 'lastdepp')
        - haystack/needle should already be normalized via _norm_token
        """
        if not haystack or not needle:
            return False
        s = haystack.lower()
        n = needle.lower()
        return re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", s) is not None

    def _algo_key_from_meta(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Extract algorithm key from calibrator JSON:
        - prefer meta.detector_key; else use meta.detector / top-level detector
        - strip trailing 'Detector' before normalization
        Example: 'LogRankDetector[GPT-Neo-2.7B]' -> 'logrank'
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
        cand = cand.split("[", 1)[0]  # strip [Model] suffix
        cand = re.sub(r"detector$", "", cand, flags=re.IGNORECASE)  # strip Detector suffix
        # Normalize to token
        return self._norm_token(cand)


    def _calibrators_dir(self) -> Path:
        """
        Default calibration dir: <repo_root>/calibration_results
        If missing, fall back to <repo_root>/calibrators (optional)
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
        return primary  # return even if missing; later code will warn
    
    def _collect_model_hints(self) -> List[str]:
        hints: List[str] = []
        keys_of_interest = [
            # binoculars / lastde, etc.
            "observer", "observer_name", "observer_name_or_path",
            "performer", "performer_name", "performer_name_or_path",
            "scoring_name", "scoring_name_or_path",
            "model", "model_name", "model_name_or_path", "model_path",
            "tokenizer", "tokenizer_name", "tokenizer_name_or_path", "tokenizer_path",
            # FastDetectGPT
            "scoring_model_name", "sampling_model_name",
            "scoring_model_path", "sampling_model_path",
            "sampling_name_or_path",
            # Baseline series needs this:
            "score_model",
        ]
        for k, v in list(self.__dict__.items()):
            if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                hints.append(v)
        for k, v in list(self.kwargs.items()):
            if isinstance(v, str) and any(tk in k.lower() for tk in keys_of_interest):
                hints.append(v)
        return hints

    from utils.paths import user_calib_dir, pkg_calib_dir, dev_calib_dir_from

    def _auto_find_calibrator_path(self) -> Optional[str]:
        """
        Search order:
        1) env/user data dir (~/.local/share/mgt_eval/calibration_results or equivalent)
        2) built-in calibrators inside the package (read-only)
        3) dev tree <repo_root>/calibration_results (if present)
        Returns:
        - filesystem path string, or
        - pseudo path like 'builtin:filename.json' (package resource)
        """
        # ---- Build candidate directories ----
        dirs_fs = []      # filesystem Path
        dirs_pkg = None   # package resource Traversable

        # User data dir
        udir = user_calib_dir()
        if udir.exists():
            dirs_fs.append(udir)

        # Package resources
        dirs_pkg = pkg_calib_dir()

        # Dev tree fallback
        dev = dev_calib_dir_from(Path(__file__))
        if dev.exists():
            dirs_fs.append(dev)

        # ---- Collect candidate files (dedupe) ----
        def iter_fs_jsons(root: Path):
            if not root.exists():
                return []
            got = list(root.glob("*.json")) + list(root.glob("*.jsonl.json"))
            # Dedupe
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
                    cand.append(("pkg", name))  # store name only; open with resources later
            except Exception:
                pass

        if not cand:
            return None

        # ---- Scoring (same matching logic, slightly simplified) ----
        import re
        def norm(s: str) -> str:
            s = s.replace("\\", "/").split("/")[-1].lower()
            return re.sub(r"[^a-z0-9._\-]+", "_", s)

        cls_name = type(self).__name__
        base = re.sub(r"detector$", "", cls_name, flags=re.IGNORECASE)
        det_key = norm(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base))  # 'lastde' / 'fast_detect_gpt', etc.
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
            return str(v)  # absolute filesystem path
        else:
            return f"builtin:{v}"  # package resource



    def load(self):
        self.is_loaded = True

        # ===== 1) explicit path first =====
        if self.calibrator_path:
            try:
                resolved = self._resolve_calibrator_path(self.calibrator_path)
                if resolved:
                    self.calibrator_path = resolved
                    self.kwargs["calibrator_path"] = resolved
                # Support file path or 'builtin:xxx.json'
                self.load_calibrator(self.calibrator_path)
                print(f"[MGTEval] Loaded calibrator from: {self.calibrator_path}")
                return
            except Exception as e:
                print(f"[MGTEval] WARNING: failed to load calibrator '{self.calibrator_path}': {e}")

        # ===== 2) auto-discovery + bootstrap to user dir =====
        try:
            det_type = (getattr(self, "detector_type", "") or "").strip().lower()
            need_auto = self.auto_calibrate and (det_type == "metric-based") and (not self.outputs_prob)
            if not need_auto:
                return

            auto_path = self._auto_find_calibrator_path()
            if auto_path:
                # 2a) package resource: copy to user dir then load
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
                        # Last-resort: if copy fails, read builtin directly (works but not writable)
                        self.load_calibrator(auto_path)
                        self.calibrator_path = auto_path
                        self.kwargs["calibrator_path"] = auto_path
                        print(f"[MGTEval] Auto-loaded calibrator from builtin: {auto_path}")
                        return
                else:
                    # 2b) filesystem path: load directly
                    self.load_calibrator(auto_path)
                    self.calibrator_path = auto_path
                    self.kwargs["calibrator_path"] = auto_path
                    print(f"[MGTEval] Auto-loaded calibrator from: {auto_path}")
                    return

            # 2c) nothing found: best-effort pick from package and copy to user dir
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

    def _resolve_calibrator_path(self, p: Optional[str]) -> Optional[str]:
        if not isinstance(p, str) or not p.strip():
            return None
        p = os.path.expandvars(os.path.expanduser(p.strip()))
        if p.startswith("builtin:"):
            return p

        # regex mode: re:<pattern> (pattern can include a directory)
        if p.startswith("re:"):
            pattern = p[3:]
            return self._select_calibrator_by_regex(pattern)

        path = Path(p)
        # try resolve relative paths against common roots
        if not path.is_absolute() and not path.exists():
            for base in (self._calibrators_dir(), user_calib_dir()):
                cand = Path(base) / path
                if cand.exists():
                    path = cand
                    break

        # wildcard glob
        if any(ch in str(path) for ch in ["*", "?", "["]):
            matches = [Path(x) for x in glob.glob(str(path))]
            chosen = self._select_best_calibrator(matches)
            return str(chosen) if chosen is not None else None

        # directory: auto pick by detector name
        if path.is_dir():
            chosen = self._select_best_calibrator(list(path.glob("*.json*")), require_det=True)
            return str(chosen) if chosen is not None else None

        return str(path)

    def _select_calibrator_by_regex(self, pattern: str) -> Optional[str]:
        pattern = (pattern or "").strip()
        if not pattern:
            return None
        base_dir = os.path.dirname(pattern) or "."
        rx = os.path.basename(pattern) or ".*"
        try:
            prog = re.compile(rx, flags=re.IGNORECASE)
        except re.error:
            return None

        cand_dir = Path(base_dir)
        if not cand_dir.exists() or not cand_dir.is_dir():
            # try common roots
            for base in (self._calibrators_dir(), user_calib_dir()):
                d = Path(base) / base_dir
                if d.exists() and d.is_dir():
                    cand_dir = d
                    break
        files = [p for p in cand_dir.glob("*.json*") if prog.search(p.name)]
        chosen = self._select_best_calibrator(files)
        return str(chosen) if chosen is not None else None

    def _select_best_calibrator(
        self,
        paths: List[Path],
        *,
        require_det: bool = False,
    ) -> Optional[Path]:
        if not paths:
            return None

        def norm(s: str) -> str:
            s = s.replace("\\", "/").split("/")[-1].lower()
            return re.sub(r"[^a-z0-9._\\-]+", "_", s)

        def canon(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        cls_name = type(self).__name__
        base = re.sub(r"detector$", "", cls_name, flags=re.IGNORECASE)
        det_key = norm(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base))
        det_variants = {det_key, det_key.replace("_", "-"), det_key.replace("-", "_"), det_key.replace("_", "")}
        det_canon = {canon(x) for x in det_variants}

        model_tokens = []
        try:
            model_tokens = [norm(t) for t in self._collect_model_hints()]
        except Exception:
            pass
        model_canon = {canon(t) for t in model_tokens}

        best = None
        best_score = -1
        best_mtime = -1.0

        def det_match(name: str) -> bool:
            toks = [t for t in re.split(r"[^a-z0-9]+", name) if t]
            return any((t in det_variants) or (canon(t) in det_canon) for t in toks)

        def model_match(name: str) -> bool:
            toks = [t for t in re.split(r"[^a-z0-9]+", name) if t]
            return any((t in model_tokens) or (canon(t) in model_canon) for t in toks)

        # first pass: enforce detector match if required
        candidates = paths
        if require_det:
            candidates = [p for p in paths if det_match(norm(p.name))]
            if not candidates:
                candidates = paths

        for p in candidates:
            if not p.is_file():
                continue
            name = norm(p.name)
            score = 0
            if det_match(name):
                score += 2
            if model_match(name):
                score += 1
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            if score > best_score or (score == best_score and mtime > best_mtime):
                best_score = score
                best_mtime = mtime
                best = p
        return best


    def load_calibrator(self, source: str | dict):
        if isinstance(source, dict):
            payload = source
        elif isinstance(source, str) and source.startswith("builtin:"):
            name = source.split(":", 1)[1]
            root = pkg_calib_dir()
            if root is None:
                raise FileNotFoundError("[MGTEval] Packaged calibration_results not found.")
            with root.joinpath(name).open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            with open(str(source), "r", encoding="utf-8") as f:
                payload = json.load(f)

        # NEW: keep full meta and parse threshold fields from meta.dev
        if isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict):
            self._calibrator_full_meta = payload["meta"]
            dev_meta = (self._calibrator_full_meta.get("dev") or {}) if isinstance(self._calibrator_full_meta, dict) else {}
            thresholds: Dict[str, float] = {}

            # 1) dev.decision: common structure:
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
                        # Support {"thr": x} or {"threshold": x}
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

            # Save parsed results
            self._calibrator_thresholds = thresholds

            # 3) Pick a preferred decision threshold
            self.decision_threshold = None
            if isinstance(thr_main, (int, float)):
                # Use dev.decision.threshold directly
                self.decision_threshold = float(thr_main)
            else:
                # If no explicit decision threshold, prefer FPR<=0.01 point
                for candidate_key in ("tpr@fpr<=0.01", "tpr@fpr<=0.010", "tpr@fpr<=1e-02"):
                    if candidate_key in thresholds:
                        self.decision_threshold = thresholds[candidate_key]
                        break

        # Original logic: extract params
        params = payload["calibrator"] if isinstance(payload, dict) and "calibrator" in payload else payload
        if "name" not in params:
            params["name"] = self.calibrator_name
        self._calibrator_params = params

    # —— Subclass must implement: return raw scores or probabilities (controlled by self.outputs_prob) ——
    def score_batch(self, texts: List[Text]) -> np.ndarray:
        """
        Return a 1D vector of shape (B,):
          - if self.outputs_prob=False (default): return raw scores (higher = more AI-like)
          - if self.outputs_prob=True: return probabilities in [0,1]
        """
        raise NotImplementedError

    # —— Probability mapping (prefer learned LR; fallback to fixed sigmoid) ——
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Unified probability mapping:
          1) If calibrator params are loaded:
             - name == 'platt_lr'  : expects 1D scores → probabilities
             - otherwise (e.g., 'linear_lr'): supports multi-dim features via calibrator apply
          2) Else fall back to Binoculars fixed sigmoid (1D only; multi-dim will be flattened with warning)
        """
        x = np.asarray(scores, dtype=np.float64)

        # —— Preferred: existing LR calibration params ——
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

                # Other custom calibrators (multi-dim linear LR, etc.)
                from calibration.registry import get_calibrator
                cal = get_calibrator(name)
                p = cal["apply"](x, self._calibrator_params)  # supports multi-dim
                p = np.asarray(p, dtype=np.float64)
                return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)
            except Exception as e:
                print(f"[MGTEval] WARNING: calibrate() using learned params failed, fallback to sigmoid. err={e}")

        # —— Fallback: Binoculars fixed sigmoid ——
        x1d = x.reshape(-1)
        if x.ndim > 1:
            print("[MGTEval] WARNING: no calibrator loaded for multi-feature scores; "
                  "falling back to Binoculars sigmoid on the flattened 1D scores.")
        # Direction: if lower score means more AI-like, set self.prob_invert=True in detector
        if self.prob_invert:
            x1d = -x1d
        x1d = np.clip(x1d, -1e6, 1e6)
        p = 1.0 / (1.0 + np.exp(-self.prob_slope * (x1d - self.prob_center)))
        return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)

    # —— Binary prediction (probability threshold) ——
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
        from metrics.metrics import compute_metrics

        if not self.is_loaded:
            self.load()

        # ===== Basic info (presentation only) =====
        method_name = getattr(self, "DETECTOR_NAME", self.__class__.__name__)
        method_type = getattr(self, "detector_type", "Unknown")
        device_hint = getattr(self, "device", None)

        header_logged = bool(getattr(self, "_mgt_eval_eval_header_logged", False))
        if show_progress and not header_logged:
            print(f"[MGTEval] Detector: {method_name} | type={method_type} | outputs_prob={bool(self.outputs_prob)}")
            if device_hint:
                print(f"[MGTEval] Device: {device_hint}")
            _print_method_citation(self)
            print(f"[MGTEval] Eval config: batch_size={int(batch_size)} | threshold={float(threshold):.4f} | TPR@FPR targets={list(tpr_at_fpr)}")

            # Calibrator info
            cal_name = (self._calibrator_params or {}).get("name") if self._calibrator_params else None
            print(f"[MGTEval] Calibration: auto={bool(self.auto_calibrate)} | force_runner={bool(self.force_runner_calibration)} | calibrator={cal_name or 'None'}")
            if getattr(self, "calibrator_path", None):
                print(f"[MGTEval] Calibrator path: {self.calibrator_path}")
            if getattr(self, "_calibrator_thresholds", None):
                print(f"[MGTEval] Recommended thresholds: {_summarize_thresholds(self._calibrator_thresholds)}")
            if getattr(self, "decision_threshold", None) is not None and abs(float(threshold) - float(self.decision_threshold)) > 1e-9:
                print(f"[MGTEval] Note: decision_threshold={float(self.decision_threshold):.4f} (you are using threshold={float(threshold):.4f}).")
            try:
                setattr(self, "_mgt_eval_eval_header_logged", True)
            except Exception:
                pass

        # ===== Flatten dataset =====
        texts: List[str] = []
        labels: List[int] = []
        for ex in dataset:
            texts.append(ex["text"])
            labels.append(int(ex["label"]))
        labels_np = np.array(labels, dtype=int)

        n_total = len(texts)
        n_pos = int(np.sum(labels_np == 1))
        n_neg = int(np.sum(labels_np == 0))
        print(f"[MGTEval] Loaded {_color_num(n_total)} samples (AI=1:{n_pos}, Human=0:{n_neg}).")

        # ===== Batch scoring (tqdm display only) =====
        t0 = time.perf_counter()
        bs = max(1, int(batch_size))
        total_batches = (n_total + bs - 1) // bs
        iterator = range(0, n_total, bs)

        if show_progress:
            # Header (train.py style)
            print("\n" +
                f"{'Phase':>{W_PHASE}}{SEP}"
                f"{'GPU_mem':>{W_MEM}}{SEP}{SEP}{SEP}{SEP}"
                f"{'done':>{W_N}}{SEP}{SEP}"
                f"{'eps':>{W_NUM}}{SEP}{SEP}"
                f"{'batch':>{W_STEP}}")

        pbar = tqdm(
            iterator,
            total=total_batches,
            desc=f"Eval[{method_name}]",
            dynamic_ncols=True,
            disable=(not show_progress),
            leave=False,
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

                # ---- tqdm description (presentation only) ----
                if show_progress:
                    done = min(start + bs, n_total)
                    elapsed = time.perf_counter() - t0
                    eps = (done / elapsed) if elapsed > 0 else 0.0
                    mem = _gpu_mem_str()
                    desc = (
                        f"{'Eval':>{W_PHASE}}{SEP}"
                        f"{mem:>{W_MEM}}{SEP}"
                        f"{done:>{W_N}d}/{n_total:<{W_N}d}{SEP}"
                        f"{eps:>{W_NUM}.2f}{SEP}"
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

        # ===== Probability mapping (add mode info only) =====
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

        # ===== Predictions & metrics (display only) =====
        preds = self.predict(probs, threshold=threshold).astype(int)
        metrics = compute_metrics(labels_np, probs, preds, tpr_at_fpr=tpr_at_fpr)

        # Only show common metrics (when available)
        acc = metrics.get("acc", None)
        f1  = metrics.get("f1", None)
        auroc = metrics.get("auroc", None)
        aupr  = metrics.get("aupr", None)
        ece   = metrics.get("ece", None)
        brier = metrics.get("brier", None)
        tpr_line = _tpr_at_fpr_str(metrics, tpr_at_fpr)

        # Only print when all common metrics are available (avoid partial "-" values).
        if all(v is not None for v in (acc, f1, auroc, aupr, ece, brier)):
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

        # ===== meta (keep structure; add display/audit fields only) =====
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

        # ===== return (keep structure) =====
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
