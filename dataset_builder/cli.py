# mgt_eval/dataset_builder/cli.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BuildConfig, GenConfig
from .builder import DatasetBuilder
from .quality_metrics import QualityConfig


def _read_text_file(path: str) -> str:
    return open(path, "r", encoding="utf-8").read()


def _silence_hf_logs():
    import os
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings("ignore", message=r".*loss_type=None.*")
    warnings.filterwarnings("ignore", message=r".*Some weights of RobertaModel were not initialized.*")
    warnings.filterwarnings("ignore", message=r".*You should probably TRAIN this model.*")

    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        try:
            tlog.disable_progress_bar()
        except Exception:
            pass
    except Exception:
        pass

    try:
        from huggingface_hub import logging as hlog
        hlog.set_verbosity_error()
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
    except Exception:
        pass


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("mgt_eval.dataset_build")

    ap.add_argument("--data", required=True, help="dataset spec: path or comma-separated specs; HC3 supported")
    ap.add_argument("--out", required=True, help="output jsonl path")

    ap.add_argument("--prompt_from_label", type=int, default=0, help="default: 0 (human)")
    ap.add_argument("--only_human_prompts", type=int, default=1, help="1: force label=0 as prompt source")
    ap.add_argument("--max_prompts", type=int, default=None, help="limit number of prompts after filtering label")
    ap.add_argument("--seed", type=int, default=114514)

    ap.add_argument("--prefix_k_tokens", type=int, default=64)
    ap.add_argument("--tokenizer_strategy", type=str, default="auto", help="auto | whitespace | hf:<tok_name>")

    ap.add_argument("--prompt_template", type=str, default="{prefix}", help="python .format template")
    ap.add_argument("--prompt_template_file", type=str, default=None)
    ap.add_argument("--system_prompt", type=str, default=None)
    ap.add_argument("--machine_text_mode", type=str, default="prompt_plus", choices=["prompt_plus", "completion_only"])

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--do_sample", type=int, default=1, help="1/0")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--stop", type=str, default=None, help="json list of stop strings, e.g. '[\"\\n\\n\"]'")
    ap.add_argument("--return_full_text", type=int, default=1, help="1: prompt+completion, 0: completion only")
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)

    # attacks
    ap.add_argument("--attacks_config", type=str, default=None, help="path to attacks json")
    ap.add_argument(
        "--attack",
        action="append",
        default=None,
        help="指定攻击类型（可重复）：span, para, typo, inse, dele, subs, tran, homo, form, syno, back_trans",
    )
    ap.add_argument(
        "--attack_types",
        type=str,
        default=None,
        help="逗号分隔的攻击类型列表，用于快速指定（等价于多次 --attack）",
    )
    # ✅ 兼容你当前的命令（--attacks dele）
    ap.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="(alias of --attack_types) e.g. --attacks dele,tran",
    )

    ap.add_argument("--backend", type=str, default=None, choices=["hf", "openai"])

    # hf backend args
    ap.add_argument("--hf_model", type=str, default=None, help="HF model name/path (for --backend hf)")
    ap.add_argument("--hf_device", type=str, default="cuda:0")
    ap.add_argument("--hf_dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--hf_trust_remote_code", type=int, default=1)

    # openai backend args
    ap.add_argument("--api_model", type=str, default=None, help="API model (for --backend openai)")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--api_base", type=str, default=None, help="OpenAI-compatible base_url, e.g. http://localhost:8000/v1")
    ap.add_argument("--api_endpoint", type=str, default="chat", choices=["chat", "completions"])
    ap.add_argument("--api_timeout", type=int, default=120)

    # quality metrics
    ap.add_argument("--metric_ppl", type=int, default=1, help="1: compute perplexity on original & samples")
    ap.add_argument("--ppl_model", type=str, default="gpt2", help="HF causal LM for perplexity")
    ap.add_argument("--ppl_device", type=str, default="cuda:0")
    ap.add_argument("--ppl_dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--ppl_stride", type=int, default=256)
    ap.add_argument("--ppl_max_length", type=int, default=1024)

    ap.add_argument("--metric_readability", type=int, default=1, help="1: compute readability for original & samples")

    ap.add_argument("--metric_bertscore", type=int, default=1, help="1: compute BERTScore(original, sample)")
    ap.add_argument("--bertscore_model", type=str, default="roberta-large")
    ap.add_argument("--bertscore_device", type=str, default="cuda:0")
    ap.add_argument("--bertscore_lang", type=str, default="en")
    ap.add_argument("--bertscore_batch_size", type=int, default=8)
    ap.add_argument("--bertscore_rescale", type=int, default=1, help="1: rescale_with_baseline (bert-score)")

    # ========== NEW: attack-only mode ==========
    ap.add_argument(
        "--attack_dataset_only",
        type=int,
        default=0,
        help="1: do NOT generate; only apply attacks on input dataset texts. "
             "All attacked texts will be saved as label=1 (machine).",
    )
    ap.add_argument(
        "--only_attack_machine",
        type=int,
        default=1,  # ✅ 默认开启：只攻击机器文本
        help="1: only attack examples whose label == --machine_label. 0: attack all examples.",
    )
    ap.add_argument(
        "--machine_label",
        type=int,
        default=1,
        help="Label value treated as machine in the input dataset (used when --only_attack_machine=1).",
    )
    ap.add_argument(
        "--sample_k",
        type=int,
        default=None,
        help="(attack-only) randomly sample K eligible examples to attack; default: full. Controlled by --seed.",
    )
    return ap


def _default_attack_item(tp: str) -> Dict[str, Any]:
    t = (tp or "").strip().lower()

    if t in ("ptb", "span"):
        return {"type": "span"}
    if t in ("paraphrase", "para", "pegasus"):
        return {"type": "para", "backend": "pegasus"}
    if t in ("dipper",):
        return {"type": "para", "backend": "dipper"}
    if t in ("hf_prompt_para", "hf"):
        return {"type": "para", "backend": "hf"}
    if t in ("api_prompt_para", "api"):
        return {"type": "para", "backend": "api"}
    if t in ("chatgpt_para", "chatgpt"):
        return {"type": "para", "backend": "chatgpt"}

    # typo + meta
    if t in ("typo", "inse", "dele", "subs", "tran"):
        return {"type": t}

    if t in ("homo", "homoglyph"):
        return {"type": "homo", "variant": "ECES"}

    if t in ("form",):
        return {"type": "form", "variant": "zero-sp"}

    if t in ("syno", "word_subst_modelfree"):
        return {"type": "syno", "backend": "modelfree"}
    if t in ("word_subst_modelbase",):
        return {"type": "syno", "backend": "modelbase"}

    if t in ("back_trans", "back_translate"):
        return {"type": "back_trans", "pivot_lang": "de", "n_rounds": 1}

    raise ValueError(f"Unknown attack type: {tp}")


def _collect_attack_types(args) -> List[str]:
    attack_types: List[str] = []

    # --attack_types "a,b,c"
    if args.attack_types:
        attack_types.extend([x.strip() for x in str(args.attack_types).split(",") if x.strip()])

    # --attacks "a,b,c" (alias)
    if args.attacks:
        attack_types.extend([x.strip() for x in str(args.attacks).split(",") if x.strip()])

    # repeated --attack a --attack b
    if args.attack:
        attack_types.extend([x.strip() for x in args.attack if x and str(x).strip()])

    # dedup keep order
    seen = set()
    out: List[str] = []
    for x in attack_types:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_attacks_config_path(args) -> Optional[str]:
    """
    优先级逻辑：
    - 只有 --attacks_config：直接用该文件
    - 只有命令行 attack types：写 out 同目录的 *.attacks_from_cli.json
    - 两者同时提供：merge 到 *.attacks_merged.json（文件配置 + cli 默认项）
    """
    attack_types = _collect_attack_types(args)
    file_path = args.attacks_config

    if (not file_path) and (not attack_types):
        return None

    out_p = Path(args.out)
    if file_path and (not attack_types):
        return str(file_path)

    # helper: write json
    def _dump(path: Path, obj: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return str(path)

    if (not file_path) and attack_types:
        cfg_obj = {"text_attacks": [_default_attack_item(t) for t in attack_types], "regen_attacks": []}
        tmp_path = out_p.with_name(out_p.stem + ".attacks_from_cli.json")
        return _dump(tmp_path, cfg_obj)

    # file_path + attack_types => merge
    with open(file_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    if not isinstance(base_cfg, dict):
        raise ValueError(f"attacks_config must be a json object, got: {type(base_cfg)}")

    base_cfg.setdefault("text_attacks", [])
    base_cfg.setdefault("regen_attacks", [])

    if not isinstance(base_cfg["text_attacks"], list):
        base_cfg["text_attacks"] = []
    if not isinstance(base_cfg["regen_attacks"], list):
        base_cfg["regen_attacks"] = []

    base_cfg["text_attacks"].extend([_default_attack_item(t) for t in attack_types])

    tmp_path = out_p.with_name(out_p.stem + ".attacks_merged.json")
    return _dump(tmp_path, base_cfg)


def main():
    ap = build_argparser()
    args = ap.parse_args()
    _silence_hf_logs()

    attack_only = bool(int(getattr(args, "attack_dataset_only", 0)))

    template = args.prompt_template
    if args.prompt_template_file:
        template = _read_text_file(args.prompt_template_file)

    stop = None
    if args.stop:
        stop = json.loads(args.stop)

    gen = GenConfig(
        max_new_tokens=int(args.max_new_tokens),
        min_new_tokens=int(args.min_new_tokens),
        do_sample=bool(int(args.do_sample)),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_beams=int(args.num_beams),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        stop=stop,
        seed=int(args.seed),
        return_full_text=bool(int(args.return_full_text)),
        presence_penalty=float(args.presence_penalty),
        frequency_penalty=float(args.frequency_penalty),
    )

    prompt_from_label = int(args.prompt_from_label)
    if bool(int(args.only_human_prompts)):
        prompt_from_label = 0

    attacks_config_path = _resolve_attacks_config_path(args)

    # parse sample_k (None or <=0 => full)
    sample_k = args.sample_k
    if sample_k is not None:
        try:
            sample_k = int(sample_k)
            if sample_k <= 0:
                sample_k = None
        except Exception:
            sample_k = None

    cfg = BuildConfig(
        dataset_spec=args.data,
        out_jsonl=args.out,
        prompt_from_label=prompt_from_label,
        prefix_k_tokens=int(args.prefix_k_tokens),
        tokenizer_strategy=str(args.tokenizer_strategy),
        prompt_template=template,
        system_prompt=args.system_prompt,
        gen=gen,
        attacks_config_path=attacks_config_path,
        max_prompts=args.max_prompts,
        sample_seed=int(args.seed),
        machine_text_mode=args.machine_text_mode,

        # attack-only flags
        attack_dataset_only=attack_only,
        only_attack_machine=bool(int(args.only_attack_machine)),
        machine_label=int(args.machine_label),
        sample_k=sample_k,  # ✅ NEW（要求：默认全量）
    )

    # -------- backend: ONLY when NOT attack-only --------
    backend = None
    if not attack_only:
        if args.backend == "hf":
            if not args.hf_model:
                raise ValueError("--hf_model is required when --backend hf")
            from .backends.hf_local import HFLocalBackend  # ✅ lazy import
            backend = HFLocalBackend(
                model_name_or_path=args.hf_model,
                device=args.hf_device,
                torch_dtype=args.hf_dtype,
                trust_remote_code=bool(int(args.hf_trust_remote_code)),
            )
        elif args.backend == "openai":
            if not args.api_model:
                raise ValueError("--api_model is required when --backend openai")
            from .backends.openai_compat import OpenAICompatBackend  # ✅ lazy import
            backend = OpenAICompatBackend(
                model=args.api_model,
                api_key=args.api_key,
                base_url=args.api_base,
                endpoint=args.api_endpoint,
                timeout_s=int(args.api_timeout)),
        else:
            raise ValueError("--backend is required unless --attack_dataset_only 1")

    # -------- quality: ONLY when NOT attack-only --------
    quality_cfg = None
    if not attack_only:
        quality_cfg = QualityConfig(
            enable_ppl=bool(int(args.metric_ppl)),
            enable_readability=bool(int(args.metric_readability)),
            enable_bertscore=bool(int(args.metric_bertscore)),
            only_human_prompts=bool(int(args.only_human_prompts)),
            ppl_model=str(args.ppl_model),
            ppl_device=str(args.ppl_device),
            ppl_dtype=str(args.ppl_dtype),
            ppl_stride=int(args.ppl_stride),
            ppl_max_length=int(args.ppl_max_length),
            bertscore_model=str(args.bertscore_model),
            bertscore_device=str(args.bertscore_device),
            bertscore_lang=str(args.bertscore_lang),
            bertscore_batch_size=int(args.bertscore_batch_size),
            bertscore_rescale=bool(int(args.bertscore_rescale)),
        )
        if not quality_cfg.any_enabled():
            quality_cfg = None

    builder = DatasetBuilder(backend=backend, cfg=cfg, quality_cfg=quality_cfg)
    stats = builder.build()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
