# mgt_eval/detectors/registry.py
from __future__ import annotations
from typing import Dict, Type
import re

from .base import DetectorBase

_REGISTRY: Dict[str, Type[DetectorBase]] = {}
REGISTRY = _REGISTRY  # ← 对外兼容：CLI/旧代码可以 from ... import REGISTRY

def _ensure_populated_once():
    if _REGISTRY:
        return
    try:
        from . import ensure_all_detectors_registered
        ensure_all_detectors_registered()
    except Exception:
        pass

def register(name: str):
    def _decorator(cls: Type[DetectorBase]):
        key = name.strip().lower()
        _REGISTRY[key] = cls
        _REGISTRY.setdefault(cls.__name__.lower(), cls)
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        _REGISTRY.setdefault(snake, cls)
        return cls
    return _decorator


def get_detector_cls(name: str) -> Type[DetectorBase]:
    _ensure_populated_once()
    """
    获取注册的检测器类。支持：
      - 显式注册名（如 "fast_detect_gpt"）
      - 类名小写（如 "fastdetectgptdetector"）
      - 类名驼峰转蛇形（如 "fast_detect_gpt_detector"）
      - 直接传入驼峰名时，也会自动转换（如 "FastDetectGPT" -> "fast_detect_gpt"）
    """
    if not isinstance(name, str) or not name:
        raise KeyError("Empty detector name.")

    key = name.strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]

    # 驼峰 -> 蛇形 再查一次
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    if snake in _REGISTRY:
        return _REGISTRY[snake]

    # 再试一次去掉后缀 Detector
    if key.endswith("detector"):
        short = key[:-8]  # remove 'detector'
        if short in _REGISTRY:
            return _REGISTRY[short]
        snake_short = re.sub(r"(?<!^)(?=[A-Z])", "_", name[:-8]).lower()
        if snake_short in _REGISTRY:
            return _REGISTRY[snake_short]

    available = ", ".join(sorted(_REGISTRY.keys())) or "N/A"
    raise KeyError(f"Detector '{name}' is not registered. Available: {available}")


def list_registered_detectors():
    """返回所有可用的注册键（小写）。"""
    _ensure_populated_once()
    return sorted(_REGISTRY.keys())

def list_detectors():
    """Backward compatibility alias."""
    return list_registered_detectors()

def available_detectors():
    """兼容旧调用名：返回已注册检测器名称列表。"""
    return list_registered_detectors()

# 如果你愿意，也可以导出一个不以下划线开头的 REGISTRY 只读视图：
REGISTRY = _REGISTRY  # 兼容旧 import 名；注意不要在别处写 REGISTRY = {}，保持引用同一个对象
