
"""
mgt_eval: A modular library for black-box evaluation of Machine-Generated-Text (MGT) detectors.
Compatibility shim for src/ layout when repo root is on sys.path.
"""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    _src_str = str(_SRC)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)
    try:
        if _src_str not in __path__:
            __path__.append(_src_str)
    except Exception:
        pass

from eval.evaluator import evaluate_detector  # noqa: E402
from detectors.registry import get_detector_cls, list_detectors  # noqa: E402

__all__ = ["evaluate_detector", "get_detector_cls", "list_detectors"]
