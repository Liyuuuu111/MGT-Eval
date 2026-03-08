# mgt_eval/detectors/pretrained/pretrained_entrypoints.py
# -*- coding: utf-8 -*-
"""
统一封装预训练检测器的推理入口 + 自定义 HF 下载工具 + 评估集成。

暴露给外部的 4 个检测接口（建议通过 console_scripts 暴露为命令）：
    - openai-detector-base  -> openai-community/roberta-base-openai-detector
    - openai-detector-large -> openai-community/roberta-large-openai-detector
    - simpleai-detector     -> Hello-SimpleAI/chatgpt-detector-roberta
    - radar        -> TrustSafeAI/RADAR-Vicuna-7B

统一推理接口：
    run_detector_on_samples(detector_key, samples, ...)

统一评估接口（复用 mgt_eval.eval.evaluator 的完整评估与绘图逻辑）：
    evaluate_pretrained_detector(detector_key, dataset, ...)

前端可调用的自定义 HF 下载接口：
    download_hf_repo_files(
        repo_id, required_files=None, local_dir=None,
        hf_endpoint="https://hf-mirror.com", proxies=DEFAULT_PROXIES, ...
    )

数据格式（前端传入的 samples）：
    samples: List[Dict[str, Any]]
        - 必需字段: "text": str
        - 可选字段: "id": Any  （若没有则自动用索引 0..N-1）
        - 可选字段: "label": int （真实标签，若存在会在输出中保留为 true_label）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable, Sequence

import torch
from huggingface_hub import hf_hub_download

from data_utils.load import load_dataset_unified
from eval.evaluator import evaluate_detector
from .pretrained import PretrainedDetector


# -------------------------------------------------------------
# → HF  ID （）
# -------------------------------------------------------------

DETECTOR_MODEL_MAP: Dict[str, str] = {
    "openai-detector-base": "openai-community/roberta-base-openai-detector",
    "openai-detector-large": "openai-community/roberta-large-openai-detector",
    "simpleai-detector": "Hello-SimpleAI/chatgpt-detector-roberta",
    "radar": "TrustSafeAI/RADAR-Vicuna-7B",
}

# -------------------------------------------------------------
# ： &
# -------------------------------------------------------------

DEFAULT_REQUIRED_FILES: List[str] = [
    # /
    "pytorch_model.bin",
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "generation_config.json",
    # /
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    # SentencePiece / BPE
    "spm.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    # （ Falcon）
    "modeling_falcon.py",
]

DEFAULT_PROXIES: Dict[str, str] = {
    "http": "socks5h://127.0.0.1:1080",
    "https": "socks5h://127.0.0.1:1080",
}


# -------------------------------------------------------------
# EvalResult： evaluator.evaluate_detector
# -------------------------------------------------------------

@dataclass
class SimpleEvalResult:
    labels: List[int]
    preds: List[int]
    probs: Optional[List[float]] = None
    scores: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------
# ：/
# -------------------------------------------------------------

def download_hf_repo_files(
    repo_id: str,
    required_files: Optional[List[str]] = None,
    local_dir: Optional[str] = None,
    hf_endpoint: Optional[str] = "https://hf-mirror.com",
    proxies: Optional[Dict[str, str]] = DEFAULT_PROXIES,
    token: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    手动下载指定 HF 仓库中的若干关键文件，默认：
      - 使用 HF 镜像地址（通过 HF_ENDPOINT 环境变量）
      - 使用 socks5 代理（127.0.0.1:1080），可在前端自定义覆盖
      - 下载到 HF 默认缓存路径（local_dir=None）或用户指定位置
    """
    # HF_ENDPOINT： hf_hub_download  transformers.from_pretrained
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    # ： + （）
    if required_files is None:
        files = list(DEFAULT_REQUIRED_FILES)
    else:
        files = []
        seen = set()
        for fname in list(DEFAULT_REQUIRED_FILES) + list(required_files):
            if fname and fname not in seen:
                files.append(fname)
                seen.add(fname)

    results: Dict[str, Dict[str, Any]] = {}

    for file_name in files:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=local_dir,      # None -> HF
                token=token,
                proxies=proxies,
            )
            results[file_name] = {
                "status": "ok",
                "path": path,
            }
        except Exception as e:
            results[file_name] = {
                "status": "error",
                "error": str(e),
            }

    return results


# -------------------------------------------------------------
# PretrainedDetector（ / ）
# -------------------------------------------------------------

def build_pretrained_detector(
    detector_key: str,
    device: Optional[str] = None,
    max_length: int = 512,
    fp16: bool = True,
    show_progress: bool = False,
) -> PretrainedDetector:
    """
    根据 detector_key 构建并 load 一个 PretrainedDetector.
    """
    if detector_key not in DETECTOR_MODEL_MAP:
        raise ValueError(
            f"Unknown detector_key={detector_key!r}, "
            f"valid choices: {sorted(DETECTOR_MODEL_MAP.keys())}"
        )

    model_id = DETECTOR_MODEL_MAP[detector_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_id  model_pathtokenizer_path，name
    det = PretrainedDetector(
        model_path=model_id,
        tokenizer_path=model_id,
        name=detector_key,
        device=device,
        max_length=max_length,
        fp16=fp16 if device.startswith("cuda") else False,
        show_progress=show_progress,
    )
    det.load()
    return det


# -------------------------------------------------------------
# ： / （， per-sample ）
# -------------------------------------------------------------

def run_detector_on_samples(
    detector_key: str,
    samples: List[Dict[str, Any]],
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    fp16: bool = True,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    使用指定的 detector 对前端传入的数据集进行推理（统一接口）。
    """
    detector = build_pretrained_detector(
        detector_key=detector_key,
        device=device,
        max_length=max_length,
        fp16=fp16,
        show_progress=False,
    )
    norm_ids: List[Any] = []
    texts: List[str] = []
    for idx, ex in enumerate(samples):
        if "text" not in ex:
            raise KeyError(f"Sample #{idx} missing 'text' field: {ex}")
        texts.append(str(ex["text"]))
        norm_ids.append(ex.get("id", idx))

    results: List[Dict[str, Any]] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_ids = norm_ids[start:end]

        scores = detector.score_batch(batch_texts)  # np.ndarray, AI

        # zip ， true_label
        for sid, s_text, s_score, raw in zip(
            batch_ids, batch_texts, scores.tolist(), samples[start:end]
        ):
            pred_label = int(float(s_score) >= float(threshold))
            out: Dict[str, Any] = {
                "id": sid,
                "text": s_text,
                "score": float(s_score),   # AI
                "pred_label": pred_label,  # （1=）
                "detector": detector_key,
            }
            # （）
            if "label" in raw:
                try:
                    out["true_label"] = int(raw["label"])
                except Exception:
                    out["true_label"] = raw["label"]
            results.append(out)

    return results


# -------------------------------------------------------------
# evaluator.evaluate_detector
# -------------------------------------------------------------

class HFPretrainedDetectorForEval:
    """
    轻量包装：把 PretrainedDetector 适配成 evaluator 可调用的 detector。
    只需要实现 .evaluate(examples, batch_size, threshold, show_progress) 即可。
    """

    def __init__(self, base: PretrainedDetector):
        self.base = base
        # evaluator  name/DETECTOR_NAME
        self.name = getattr(base, "name", "pretrained-detector")
        # ：，evaluator  model_card.json
        self.detector_type = getattr(base, "detector_type", "HFPretrained")
        self.scoring_model_name = getattr(base, "model_path", None)
        self.tokenizer_name = getattr(base, "tokenizer_path", None)
        self.model_path = getattr(base, "model_path", None)
        self.tokenizer_path = getattr(base, "tokenizer_path", None)

    def evaluate(
        self,
        examples: List[Dict[str, Any]],
        batch_size: int = 8,
        threshold: float = 0.5,
        show_progress: bool = True,
    ) -> SimpleEvalResult:
        """
        复用 base.score_batch 做 scoring，再组装成 SimpleEvalResult，
        提供 labels / preds / probs / scores / meta 五个字段。
        """
        texts: List[str] = []
        labels: List[int] = []
        for idx, ex in enumerate(examples):
            if "text" not in ex:
                raise KeyError(f"Example #{idx} missing 'text' field: {ex}")
            if "label" not in ex:
                raise KeyError(
                    f"Example #{idx} missing 'label' field (需要真实标签做评估): {ex}"
                )
            texts.append(str(ex["text"]))
            labels.append(int(ex["label"]))

        n = len(texts)
        probs: List[float] = []

        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    range(0, n, batch_size),
                    desc=f"[{self.name}] eval",
                    leave=False,
                )
            except Exception:
                iterator = range(0, n, batch_size)
        else:
            iterator = range(0, n, batch_size)

        for start in iterator:
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            scores = self.base.score_batch(batch_texts)  # np.ndarray  list
            probs.extend([float(s) for s in scores])

        if len(probs) != n:
            raise RuntimeError(
                f"内部错误：得到的概率数量 {len(probs)} 与样本数 {n} 不一致。"
            )

        preds: List[int] = [1 if p >= float(threshold) else 0 for p in probs]

        meta = {
            "detector_name": self.name,
            "detector_type": self.detector_type,
            "scoring_model_name": self.scoring_model_name,
        }

        return SimpleEvalResult(
            labels=labels,
            preds=preds,
            probs=probs,
            scores=probs,  # / probs
            meta=meta,
        )


def evaluate_pretrained_detector(
    detector_key: str,
    dataset: str | Iterable[Dict[str, Any]],
    *,
    device: Optional[str] = None,
    batch_size: int = 8,
    max_length: int = 512,
    fp16: bool = True,
    threshold: float = 0.5,
    sample_k: Optional[int] = None,
    sample_seed: int = 114514,
    group_cols: Optional[Sequence[str]] = None,
    out_dir: Optional[str] = None,
    save_curves: bool = True,
    k_runs: int = 1,
    ci_enable: Optional[bool] = None,
    ci_iters: int = 200,
    ci_seed: int = 114514,
    show_progress: bool = True,
    # ---- NEW: ASR ----
    attack_datasets: Optional[Sequence[str | Iterable[Dict[str, Any]]]] = None,
    asr_save_details: bool = True,
):
    base = build_pretrained_detector(
        detector_key=detector_key,
        device=device,
        max_length=max_length,
        fp16=fp16,
        show_progress=show_progress,
    )
    wrapper = HFPretrainedDetectorForEval(base)

    return evaluate_detector(
        detector=wrapper,
        dataset=dataset,
        batch_size=batch_size,
        threshold=threshold,
        sample_k=sample_k,
        sample_seed=sample_seed,
        group_cols=group_cols,
        out_dir=out_dir,
        save_curves=save_curves,
        ci_enable=ci_enable,
        ci_iters=ci_iters,
        ci_seed=ci_seed,
        show_progress=show_progress,
        k_runs=k_runs,
        # ---- NEW: ASR ----
        attack_datasets=attack_datasets,
        asr_save_details=asr_save_details,
    )


# -------------------------------------------------------------
# CLI ：（ mode=predict ）
# -------------------------------------------------------------

def _load_dataset_for_cli(
    input_arg: str,
    sample_k: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    使用 load_dataset_unified 加载数据：
      - 若 input_arg 为路径/通配符/逗号分隔 spec，则直接丢给 load_dataset_unified
      - 若 input_arg 为 '-'，则从 stdin 读取 JSONL，对象列表丢给 load_dataset_unified
    """
    # 1： stdin  JSONL， load_dataset_unified
    if input_arg == "-" or input_arg == "":
        raw_records: List[Dict[str, Any]] = []
        for idx, line in enumerate(sys.stdin):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"[stdin] 第 {idx + 1} 行不是合法 JSON；"
                    f"在使用统一数据加载时，请保证 stdin 为 JSONL，其中每行是一个 dict，"
                    f"至少包含 'text' 与 'label' 字段。"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"[stdin] 第 {idx + 1} 行解析得到 {type(obj)}，"
                    f"期望为 JSON 对象(dict)。"
                )
            raw_records.append(obj)

        examples, _group_cols = load_dataset_unified(
            raw_records,
            sample_k=sample_k,
            sample_seed=seed,
            group_cols=None,
        )
    else:
        # 2：/spec（）， load_dataset_unified
        examples, _group_cols = load_dataset_unified(
            input_arg,
            sample_k=sample_k,
            sample_seed=seed,
            group_cols=None,
        )

    if not examples:
        raise ValueError(
            "load_dataset_unified 未加载到有效样本（需要至少包含 'text' 与 'label'）。"
        )

    # id（）
    for i, ex in enumerate(examples):
        ex.setdefault("id", i)

    return examples


def _save_results_to_stream(results: List[Dict[str, Any]], fp) -> None:
    """
    以 JSON Lines 写出推理结果，便于前端/下游解析。
    """
    for ex in results:
        fp.write(json.dumps(ex, ensure_ascii=False) + "\n")


# -------------------------------------------------------------
# CLI （ mode=predict / eval）
# -------------------------------------------------------------

def _build_argparser(detector_key: Optional[str] = None) -> argparse.ArgumentParser:
    """
    若 detector_key 为 None，则需要在命令行指定 --detector；
    否则该 parser 只针对某一个固定的 detector（适合 console_scripts 包装）。
    """
    if detector_key is None:
        prog = os.path.basename(sys.argv[0]) or "pretrained-detector-cli"
        desc = "Unified CLI for multiple pretrained MGT detectors."
    else:
        prog = detector_key
        desc = f"CLI wrapper for detector '{detector_key}'."

    parser = argparse.ArgumentParser(prog=prog, description=desc)

    if detector_key is None:
        parser.add_argument(
            "--detector",
            type=str,
            required=True,
            choices=sorted(DETECTOR_MODEL_MAP.keys()),
            help="Detector alias to use.",
        )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["predict", "eval"],
        default="predict",
        help="运行模式：predict 输出逐样本 JSONL；eval 走 evaluator 生成完整指标与图表。",
    )

    parser.add_argument(
        "--sample-k",
        type=int,
        default=None,
        help="可选：在 load_dataset_unified / evaluator 中对样本进行 subsampling（k 条样本）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=114514,
        help="随机种子（用于 subsampling / 平衡抽样等）。",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="-",
        help=(
            "数据集规格（dataset spec），将直接传给 load_dataset_unified / evaluator：\n"
            "  - 可以是目录/文件路径（自动递归 *.csv/*.json/*.jsonl）\n"
            "  - 可以是逗号分隔的多个路径/spec\n"
            "  - 若包含 'hc3'，会自动走 HC3 专用展开逻辑\n"
            "  - 在 mode=predict 时，若为 '-'，则从 stdin 读取 JSONL（每行一个 dict，含 text/label）"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="-",
        help="在 mode=predict 下：输出 JSONL 路径；使用 '-' 输出到 stdout。mode=eval 下忽略。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'cuda', 'cpu', or leave empty for auto。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference/evaluation。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization。",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even on CUDA devices。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on AI probability。",
    )

    # eval
    parser.add_argument(
        "--eval-out-dir",
        type=str,
        default=None,
        help="mode=eval 时 evaluator 的输出目录（若不指定则使用默认 runs_{detector}_{timestamp}/）。",
    )
    parser.add_argument(
        "--k-runs",
        type=int,
        default=1,
        help="mode=eval 且 sample-k>0 时：多次随机采样重复评测次数。",
    )
    parser.add_argument(
        "--no-save-curves",
        action="store_true",
        help="mode=eval 时：只保存 summary.json，不保存 ROC/PR/RC/Calibration 曲线与图像。",
    )

    return parser


def _cli_run_predict(
    detector_key: str, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if detector_key not in DETECTOR_MODEL_MAP:
        parser.error(
            f"Unknown detector_key={detector_key}, "
            f"valid: {sorted(DETECTOR_MODEL_MAP.keys())}"
        )
    try:
        samples = _load_dataset_for_cli(
            input_arg=args.input,
            sample_k=args.sample_k,
            seed=args.seed,
        )
    except Exception as e:
        sys.stderr.write(f"[ERROR] load_dataset_unified 失败：{e}\n")
        return

    if not samples:
        sys.stderr.write(
            "No valid examples loaded by load_dataset_unified "
            "(need at least 'text' and 'label').\n"
        )
        return
    fp16 = not args.no_fp16
    results = run_detector_on_samples(
        detector_key=detector_key,
        samples=samples,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=fp16,
        threshold=args.threshold,
    )
    if args.output == "-" or args.output == "":
        out_fp = sys.stdout
    else:
        out_fp = open(args.output, "w", encoding="utf-8")

    try:
        _save_results_to_stream(results, out_fp)
    finally:
        if out_fp is not sys.stdout:
            out_fp.close()


def _cli_run_eval(
    detector_key: str, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if detector_key not in DETECTOR_MODEL_MAP:
        parser.error(
            f"Unknown detector_key={detector_key}, "
            f"valid: {sorted(DETECTOR_MODEL_MAP.keys())}"
        )

    fp16 = not args.no_fp16

    # ：mode=eval ，dataset  evaluator（ load_dataset_unified）
    try:
        evaluate_pretrained_detector(
            detector_key=detector_key,
            dataset=args.input,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=fp16,
            threshold=args.threshold,
            sample_k=args.sample_k,
            sample_seed=args.seed,
            group_cols=None,
            out_dir=args.eval_out_dir,
            save_curves=(not args.no_save_curves),
            k_runs=args.k_runs,
            # CI  evaluator
        )
    except Exception as e:
        sys.stderr.write(f"[ERROR] evaluator 运行失败：{e}\n")
        return


def _cli_main_fixed(detector_key: str) -> None:
    """
    固定 detector_key 的 CLI 主函数，用于 console_scripts：
        openai-detector-base = ...:entrypoint_openai_detector_base
    """
    parser = _build_argparser(detector_key=detector_key)
    args = parser.parse_args()

    if args.mode == "eval":
        _cli_run_eval(detector_key, args, parser)
    else:
        _cli_run_predict(detector_key, args, parser)


def cli_main_generic() -> None:
    """
    通用 CLI：需要 --detector 指定具体的检测器。
    可以作为一个总入口脚本使用，例如：
        python -m mgt_eval.detectors.pretrained.pretrained_entrypoints \
            --detector openai-detector-base --mode eval -i data.jsonl
    """
    parser = _build_argparser(detector_key=None)
    args = parser.parse_args()

    detector_key = args.detector
    if detector_key not in DETECTOR_MODEL_MAP:
        parser.error(
            f"Unknown detector_key={detector_key}, "
            f"valid: {sorted(DETECTOR_MODEL_MAP.keys())}"
        )

    if args.mode == "eval":
        _cli_run_eval(detector_key, args, parser)
    else:
        _cli_run_predict(detector_key, args, parser)


# -------------------------------------------------------------
# 4  entrypoint
# （ setup.cfg / pyproject.toml  console_scripts ）
# -------------------------------------------------------------

def entrypoint_openai_detector_base() -> None:
    _cli_main_fixed("openai-detector-base")


def entrypoint_openai_detector_large() -> None:
    _cli_main_fixed("openai-detector-large")


def entrypoint_simpleai_detector() -> None:
    _cli_main_fixed("simpleai-detector")


def entrypoint_radar_detector() -> None:
    _cli_main_fixed("radar")


# python -m ，：
if __name__ == "__main__":
    cli_main_generic()
