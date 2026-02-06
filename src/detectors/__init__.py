# mgt_eval/detectors/__init__.py
"""
Detector public API.

用法 1：一行跑评测（推荐）
    from detectors import Binoculars
    Binoculars(
        data="path_or_hf_spec",
        batch_size=8,
        sample_k=100,
        out="runs_dir",
        # ...  detector  kwargs
        observer_name_or_path="EleutherAI/gpt-neo-2.7B",
        performer_name_or_path="EleutherAI/gpt-neo-2.7B",
        calibrator="/path/to/calibrator.json",   # ★ （ calibrator_path）
    )

用法 2：拿到原始类（自定义实例化）
    from detectors import get_detector_cls
    Det = get_detector_cls("binoculars")
    det = Det(observer_name_or_path=..., performer_name_or_path=..., calibrator_path=".../cal.json")
    from eval import evaluate_detector
    evaluate_detector(detector=det, dataset="...", ...)

实现细节：
- 通过 __getattr__ 动态生成“运行器函数”（runner），名称为驼峰类名，如 Binoculars、FastDetectGPT、Pretrained。
- runner 的参数中，评测相关用统一前缀/名称（data、batch_size、threshold、sample_k、out 等），
  其余参数全部当作 detector kwargs 传入。
- 支持 'calibrator' 别名（被映射为 detector 的 'calibrator_path'），方便命令行直接指定 JSON。
"""

from __future__ import annotations
from typing import Any, Dict
import inspect
import re
import importlib
import pkgutil
# API
from .registry import get_detector_cls, list_registered_detectors  # noqa: F401

# ——  import  register(...)（）——
# import （）， try
try:
    from .metric.dnagpt import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.dnadetectllm import *  # noqa: F401,F403
except Exception:
    pass
try:
    from .metric.lastde import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.raidar import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.TOCSIN import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.gltr import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.detectgpt import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.fast_detect_gpt import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.lrr import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.npr import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.binoculars import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .metric.baseline import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .pretrained.pretrained import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .finetuned.finetuned import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .finetuned.greater import *  # noqa: F401,F403
except Exception:
    pass

try:
    from .finetuned.detective import *  # noqa: F401,F403
except Exception:
    pass
def _camel_to_snake(name: str) -> str:
    # Step1: FooBAR -> Foo_BAR
    s1 = re.sub(r'(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    # Step2: FooBARBaz -> FooBAR_Baz -> FooBAR_Baz -> Foo_BAR_Baz
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def _make_runner(detector_key: str):
    """
    生成一个“同名运行器”函数：
    - 自动根据 evaluate_detector 的签名，把传入的 kwargs 拆分为
      * eval_kwargs: 传给 evaluate_detector
      * detector_kwargs: 传给 detector 构造器
    - 同时兼容参数别名：
      * data/dataset -> evaluate_detector(dataset=...)
      * out/out_dir  -> evaluate_detector(out_dir=...)
      * calibrator  -> detector(calibrator_path=...)   ★ 新增
    - 这样 evaluate_detector 新增参数时，runner 会零改动自动支持。
    """
    from eval import evaluate_detector as _eval

    # evaluate_detector （）
    _eval_sig = inspect.signature(_eval)
    _eval_param_names = set(_eval_sig.parameters.keys())  # e.g. {"detector","dataset","batch_size",...}

    # （ runner ）
    _aliases_eval = {
        "data": "dataset",
        "out": "out_dir",
    }

    def runner(**all_kwargs: Dict[str, Any]):
        # ---- 1)  dataset（ data/dataset ） ----
        dataset = None
        if "dataset" in all_kwargs:
            dataset = all_kwargs.pop("dataset")
        elif "data" in all_kwargs:
            dataset = all_kwargs.pop("data")
        if dataset is None:
            raise TypeError("Missing required argument: 'data' (or 'dataset').")

        # ---- 2)  out/out_dir （） ----
        if "out_dir" not in all_kwargs and "out" in all_kwargs:
            all_kwargs["out_dir"] = all_kwargs.pop("out")

        # ---- 3)  kwargs  eval_kwargs / detector_kwargs ----
        eval_kwargs: Dict[str, Any] = {}
        detector_kwargs: Dict[str, Any] = {}

        for k, v in list(all_kwargs.items()):
            target_k = _aliases_eval.get(k, k)  # （ evaluator）
            if target_k in _eval_param_names and target_k != "detector":
                eval_kwargs[target_k] = v
            else:
                detector_kwargs[k] = v

        # ---- 4)  'calibrator' （ detector ） ----
        # calibrator="/path/to/cal.json"， calibrator_path
        if "calibrator" in detector_kwargs and "calibrator_path" not in detector_kwargs:
            detector_kwargs["calibrator_path"] = detector_kwargs.pop("calibrator")

        # ---- 5)  detector  evaluator ----
        Det = get_detector_cls(detector_key)
        det = Det(**detector_kwargs)

        return _eval(detector=det, dataset=dataset, **eval_kwargs)
    runner.__name__ = detector_key
    # doc： evaluator （，）
    sorted_params = ", ".join(sorted(p for p in _eval_param_names if p != "detector"))
    runner.__doc__ = (
        f"Runner for detector='{detector_key}'.\n"
        f"Accepts any kwargs. Known evaluator params (auto-synced): {sorted_params}\n"
        f"Aliases: data->dataset, out->out_dir, calibrator->calibrator_path.\n"
        f"Other kwargs are passed to the detector constructor."
    )
    return runner


def __getattr__(name: str):
    """
    - 当以驼峰名（Binoculars/FastDetectGPT/Pretrained）访问时，返回“同名运行器”（runner）。
    - 如需原始类，请使用：get_detector_cls('<registered_key>')。
    """
    # →  key（snake）
    key = _camel_to_snake(name)

    # ：
    try:
        _ = get_detector_cls(key)
        return _make_runner(key)
    except Exception:
        pass

    # ： runner
    try:
        _ = get_detector_cls(name)
        return _make_runner(name)
    except Exception:
        pass

    # →
    try:
        available = ", ".join(sorted(list_registered_detectors()))
    except Exception:
        available = "N/A"
    raise AttributeError(
        f"Detector runner '{name}' is not available. "
        f"Available detectors: {available}"
    )

def ensure_all_detectors_registered() -> None:
    """
    递归导入 detectors 包下除 base/registry 外的所有模块，
    以触发 @register 装饰器，填充全局注册表。
    """
    prefix = __name__ + "."
    for mod in pkgutil.walk_packages(__path__, prefix=prefix):
        name = mod.name
        parts = name.split(".")
        # ，
        if any(p in ("base", "registry") for p in parts):
            continue
        try:
            importlib.import_module(name)
        except Exception as e:
            # ：，（）
            print(f"[MGTEval] WARNING: failed to import detector module '{name}': {e}")
            pass

__all__ = [
    "get_detector_cls",
    "list_registered_detectors",
]
