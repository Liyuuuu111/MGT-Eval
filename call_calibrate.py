# call_calibrate.py
import argparse
import json
import os
from typing import Dict, Any, List, Tuple, Optional

from mgt_eval.calibration import Calibrate


def _coerce_val(s: str) -> Any:
    """Try to coerce CLI string into bool/int/float/None, else keep string."""
    if not isinstance(s, str):
        return s
    t = s.strip()
    low = t.lower()
    if low in ("true", "t", "yes", "y", "on"):
        return True
    if low in ("false", "f", "no", "n", "off"):
        return False
    if low in ("none", "null"):
        return None
    # int
    try:
        if t.isdigit() or (t.startswith(("+", "-")) and t[1:].isdigit()):
            return int(t)
    except Exception:
        pass
    # float
    try:
        return float(t)
    except Exception:
        return t


def _parse_unknown_as_kwargs(unknown: List[str]) -> Dict[str, Any]:
    """
    Convert unknown CLI tokens into a kwargs dict for detector-specific args.
    Examples:
      --basemodel lrr --bart_ckpt /path/to/bart -> {"basemodel": "lrr", "bart_ckpt": "/path/to/bart"}
      --flag_only -> {"flag_only": True}
    """
    det_kws: Dict[str, Any] = {}
    key: Optional[str] = None
    for tok in unknown:
        if tok.startswith("--"):
            # Start a new key. Hyphens → underscores for Python kwargs.
            key = tok.lstrip("-").replace("-", "_")
            # Default as True; if a following value arrives, it will overwrite.
            det_kws[key] = True
        else:
            if key is None:
                # Orphan value; ignore gracefully.
                continue
            det_kws[key] = _coerce_val(tok)
            key = None
    return det_kws


def _load_detector_args_json(path_or_json: Optional[str]) -> Dict[str, Any]:
    if not path_or_json:
        return {}
    s = path_or_json.strip()
    # If it's a file path that exists, read it; else try to parse as JSON string.
    if os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(s)
    except Exception:
        raise ValueError(
            f"--detector_args_json is neither an existing file nor a valid JSON string: {path_or_json}"
        )


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser("Calibrate quickstart")

    # Common/runner args
    ap.add_argument("--detector", type=str, default="lastde")
    ap.add_argument("--model1", type=str, required=True)
    ap.add_argument("--model2", type=str, default=None)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sample_k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=114514)
    ap.add_argument("--device", type=str, default=None)
    # Keep backward compatible: bool via "True/False"
    ap.add_argument("--bf16", type=bool, default=True)

    # Detector kwargs via JSON or inline-JSON
    ap.add_argument("--detector_args_json", type=str, default=None)

    # Calibrator args
    ap.add_argument("--calibrator", type=str, default="platt_lr")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--no_standardize", action="store_true")

    # Output
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)

    # IMPORTANT: use parse_known_args to capture detector-specific unknowns
    return ap.parse_known_args()


if __name__ == "__main__":
    args, unknown = parse_args()

    # 1) Load detector args from JSON/file (if any)
    det_kwargs = _load_detector_args_json(args.detector_args_json)

    # 2) Merge unknown CLI tokens as detector kwargs (CLI overrides JSON)
    #    This covers: --basemodel, --bart_ckpt, --tau_prime, --epsilon_mult, ...
    det_kwargs_cli = _parse_unknown_as_kwargs(unknown)
    det_kwargs.update(det_kwargs_cli)

    # 3) Call runner. The runner will:
    #    - Map model1/model2 into the detector's expected __init__ signature
    #    - Handle aliases like 'bart_ckpt' -> 'bart_checkpoint' (in _build_detector)
    ret = Calibrate(
        detector=args.detector,
        model1=args.model1,
        model2=args.model2,
        data=args.data,
        batch_size=args.batch_size,
        sample_k=args.sample_k,
        seed=args.seed,
        device=args.device,
        bf16=args.bf16,
        detector_kwargs=det_kwargs,
        calibrator_name=args.calibrator,
        l2=args.l2,
        max_iter=args.max_iter,
        tol=args.tol,
        standardize=(not args.no_standardize),
        out=args.out,
        out_dir=args.out_dir,
    )

    print(f"[Calibrate] saved: {ret['path']}")
