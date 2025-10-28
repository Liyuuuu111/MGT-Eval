
"""
mgt_eval: A modular library for black-box evaluation of Machine-Generated-Text (MGT) detectors.
"""
from .eval.evaluator import evaluate_detector
from .detectors.registry import get_detector_cls, list_detectors
