# mgt_eval/train/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any, Optional

_TRAIN_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {}

def register_train(*names: str):
    """
    装饰器：注册训练方法。
    用法：
        @register_train("roberta-base", "roberta-large")
        def train_roberta(...): ...
    """
    def _decorator(fn: Callable[..., Dict[str, Any]]):
        for n in names:
            if not n:
                continue
            _TRAIN_REGISTRY[n.lower()] = fn
        return fn
    return _decorator

def get_trainer(name: str) -> Callable[..., Dict[str, Any]]:
    fn = get_train_fn(name)
    if fn is None:
        available = ", ".join(sorted(_TRAIN_REGISTRY.keys())) or "N/A"
        raise KeyError(
            f"[MGTEval][train] No trainer registered for '{name}'. "
            f"Available: {available}"
        )
    return fn

def get_train_fn(name: str) -> Optional[Callable[..., Dict[str, Any]]]:
    return _TRAIN_REGISTRY.get((name or "").lower(), None)

def model_train(model: str, **kwargs) -> Dict[str, Any]:
    """
    统一入口：根据 `model` 路由到对应的训练函数。
    若找不到，回退到 'xlm-roberta-base' 的训练函数（若依然缺失则报错）。
    返回值必须包含至少：
      - "model_dir": 可被 evaluate_detector 直接加载的目录（HF save_pretrained）
    """
    fn = get_train_fn(model)
    if fn is None:
        fn = get_train_fn("xlm-roberta-base")
        if fn is None:
            raise ValueError(f"No train function registered for '{model}', and fallback 'xlm-roberta-base' missing.")
        # 将原始 model 作为 base_model 传入，留给实现自行解析/回退
        return fn(model=model, **kwargs)
    return fn(model=model, **kwargs)

def model_train_and_eval(model: str, **kwargs) -> Dict[str, Any]:
    """
    在 model_train 基础上，额外调用评估（evaluate_detector）。
    额外可接受关键字参数（若提供则透传）：
      - eval_dataset / eval_sample_k / eval_batch_size / eval_threshold
      - eval_out_dir / eval_save_curves / eval_group_cols
      - detector_name（默认 "pretrained"）
    返回：
      {
        "train": {...train_out...},
        "eval":  {...可能包含 evaluate_detector 的结果摘要/路径...}
      }
    """
    from mgt_eval.eval import evaluate_detector
    train_out = model_train(model=model, **kwargs)
    print("开始评估模型...")
    detector_name = kwargs.pop("detector_name", "pretrained")
    eval_dataset = kwargs.pop("eval_dataset", kwargs.get("dataset", "hc3"))
    eval_sample_k = kwargs.pop("eval_sample_k", None)
    eval_batch_size = kwargs.pop("eval_batch_size", 64)
    eval_threshold = kwargs.pop("eval_threshold", 0.5)
    eval_out_dir = kwargs.pop("eval_out_dir", f"./eval_{detector_name}")
    eval_group_cols = kwargs.pop("eval_group_cols", None)
    max_length = kwargs.get("max_length", 512)

    # 兼容两种返回形式：
    if "model_dir" in train_out:
        model_dir = train_out["model_dir"]
    elif "train" in train_out and isinstance(train_out["train"], dict) and "model_dir" in train_out["train"]:
        model_dir = train_out["train"]["model_dir"]
    else:
        raise KeyError("train_out 中找不到 model_dir（既没有顶层 model_dir，也没有 train['model_dir']）")

    dataset_for_eval = train_out.get("test_examples", None) or eval_dataset    # 允许直接用训练产出的测试集进行评测（若实现返回了 test 列表）

    _ = evaluate_detector(
        detector=detector_name,
        dataset=dataset_for_eval,
        sample_k=eval_sample_k,
        batch_size=eval_batch_size,
        threshold=eval_threshold,
        model_path=model_dir,
        tokenizer_path=model_dir,
        max_length=max_length,
        out_dir=eval_out_dir,
        group_cols=eval_group_cols,
        name=kwargs.get("name", None),
    )
    return {
        "train": train_out,
        "eval": {"out_dir": eval_out_dir, "detector": detector_name}
    }
