from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from typing import ClassVar

from .base import AttackBundle
# factory.py 里：from .text_attacks import (...) 需要补齐导入
from .text_attacks import (
  NoAttack,
  WordDropAttack, WordSwapAttack, CharSwapAttack, PunctuationNoiseAttack, DictionarySubstitutionAttack,
  TypoAttack, FormatAttack, ViperHomoglyphAttack,
  PerturbationT5Attack, PegasusParaphraseAttack, DipperParaphraseAttack,
  HFPromptParaphraseAttack, APIPromptParaphraseAttack, ChatGPTParaphraseAttack,
  SynonymSubstitutionAttack, WordSubstModelBaseAttack,
  BackTranslationAttack,
  HumanizeAttackAPI, HumanizeAttackHF,
)
from .regen_attacks import NoRegen, TemperatureSweep, TopPSweep, GreedyVsSample
# factory.py 里：from .text_attacks import (...) 需要补齐导入

DEFAULT_SYSTEM_PROMPT = """
You are a ruthless paraphraser. You will aggressively and completely rewrite the user's text while preserving the core meaning and factual content. Keep the SAME LANGUAGE as the input. Do NOT add disclaimers, meta-comments, safety notes, citations, or new facts. Do NOT shorten excessively; keep roughly similar length unless instructed otherwise. Produce ONLY the rewritten text as plain text, with no surrounding quotes or markers. 
"""

def _pick_cache_dir(item: Dict[str, Any]) -> Optional[str]:
    """
    约定：
    - 不写 cache_dir / cache_dir=null/none/""/hf/default/global => 返回 None（走 HF 全局用户 cache）
    - 其他字符串 => 作为显式 cache_dir 路径
    """
    raw = item.get("cache_dir", None)
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"hf", "default", "global", "none", "null"}:
        return None
    return s

def load_attacks_from_config(path: Optional[str]) -> AttackBundle:
    """
    attacks_config 示例：
    {
      "text_attacks": [
        {"type":"word_drop", "p":0.05, "n_variants":2},
        {"type":"punct_noise", "p":0.02, "n_variants":1},
        {"type":"dict_subst", "mapping_path":"./dict.json", "p":0.2}
      ],
      "regen_attacks": [
        {"type":"temp_sweep", "temps":[0.3, 0.7, 1.2]},
        {"type":"greedy_vs_sample", "sample_temperature":1.1}
      ]
    }
    """
    if not path:
        return AttackBundle(text_attackers=[NoAttack()], regen_attackers=[NoRegen()])

    cfg = json.load(open(path, "r", encoding="utf-8"))
    tcfg = cfg.get("text_attacks", []) or []
    rcfg = cfg.get("regen_attacks", []) or []
    def _pick(item: dict, keys: List[str], default=None):
        for k in keys:
            if k in item and item[k] is not None and str(item[k]) != "":
                return item[k]
        return default

    def _normalize_type(tp_raw: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        return: (type, backend, variant_or_mode)
        - type: span/para/typo/homo/form/syno/back_trans 或 meta typo: inse/dele/subs/tran
        - backend: for para/syno
        - variant_or_mode: for homo/form/typo
        """
        tp = (tp_raw or "").strip().lower()

        # ✅ 兼容旧写法 -> 新类别/别名
        alias = {
            "ptb": "span",
            "span_perturbation": "span",
            "paraphrase": "para",
            "pegasus": "para",
            "dipper": "para",
            "hf_prompt_para": "para",
            "api_prompt_para": "para",
            "chatgpt_para": "para",
            "word_subst_modelfree": "syno",
            "word_subst_modelbase": "syno",
            "back_translate": "back_trans",
            "backtranslation": "back_trans",

            # ✅ NEW: accept short aliases for meta-typo attacks
            "del": "dele",
            "delete": "dele",
            "ins": "inse",
            "insert": "inse",
            "sub": "subs",
            "subst": "subs",
            "trans": "tran",
            "translate": "tran",
        }
        alias.update({
            "humanize": "humanize",
            "humanization": "humanize",
            "anthropomorphic": "humanize",
            "personify": "humanize",
            "persona": "humanize",
        })
        tp2 = alias.get(tp, tp)

        # ✅ 元攻击：四个类别独立存在
        if tp2 in ("inse", "dele", "subs", "tran"):
            return tp2, None, None

        # ✅ old “typo_xxx”
        if tp2.startswith("typo_"):
            return "typo", None, tp2.split("_", 1)[1]   # mode

        # ✅ old “homo_eces / homo_ices”
        if tp2.startswith("homo_"):
            return "homo", None, tp2.split("_", 1)[1].upper()

        # ✅ old “form_zero-sp / form_shift-u”
        if tp2.startswith("form_"):
            return "form", None, tp2.split("_", 1)[1]

        return tp2, None, None

    text_attackers = []
    for item in tcfg:
        tp_raw = (item.get("type") or "").strip()
        if not tp_raw:
            continue

        tp, backend_hint, var_hint = _normalize_type(tp_raw)

        if tp in ("none", "no", "noop", "noattack"):
            continue

        # --------------------
        # span (Span Perturbation)
        # --------------------
        if tp == "span":
            # ✅ model 可是本地目录，也可是 HF id；不填默认 t5-large
            model = str(_pick(item, ["model", "mask_filling_model_name"], "t5-large"))
            text_attackers.append(PerturbationT5Attack(
                mask_filling_model_name=model,
                pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                span_length=int(item.get("span_length", 2)),
                buffer_size=int(item.get("buffer_size", 1)),
                mask_top_p=float(item.get("mask_top_p", 1.0)),
                chunk_size=int(item.get("chunk_size", 20)),
                n_variants=int(item.get("n_variants", 1)),
                device=str(item.get("device", "cuda")),
                cache_dir=_pick_cache_dir(item),
                dtype=str(item.get("dtype", "bf16")),
            ))
            continue

        # --------------------
        # para (Paraphrase)
        #   backend: pegasus / dipper / hf / api / chatgpt
        # --------------------
        if tp == "para":
            backend = str(item.get("backend") or item.get("engine") or backend_hint or "").strip().lower()
            if not backend:
                # 兼容旧：type=pegasus/dipper/api_prompt_para/... 会通过 normalize_type 进来，但 backend_hint 为空
                # 给一个合理默认
                backend = "pegasus"

            if backend == "pegasus":
                model = str(_pick(item, ["model", "model_name"], "tuner007/pegasus_paraphrase"))
                text_attackers.append(PegasusParaphraseAttack(
                    model_name=model,
                    top_p=float(item.get("top_p", 0.96)),
                    temperature=float(item.get("temperature", 1.0)),
                    max_length=int(item.get("max_length", 60)),
                    sent_batch_size=int(item.get("sent_batch_size", 64)),
                    n_variants=int(item.get("n_variants", 1)),
                    device=str(item.get("device", "cuda")),
                    cache_dir=_pick_cache_dir(item),
                    dtype=str(item.get("dtype", "bf16")),
                ))

            elif backend == "dipper":
                model = str(_pick(item, ["model", "model_name"], "kalpeshk2011/dipper-paraphraser-xxl"))
                text_attackers.append(DipperParaphraseAttack(
                    model_name=model,
                    lex_diversity=int(item.get("lex_diversity", 60)),
                    order_diversity=int(item.get("order_diversity", 60)),
                    n_variants=int(item.get("n_variants", 1)),
                    device=str(item.get("device", "cuda")),
                    dtype=str(item.get("dtype", "bf16")),
                ))

            elif backend == "hf":
                # ✅ 可指定本地模型目录；不指定默认加载 HF（这里给一个稳妥默认）
                model = str(_pick(item, ["model", "model_name_or_path"], "google/flan-t5-large"))
                text_attackers.append(HFPromptParaphraseAttack(
                    model_name_or_path=model,
                    max_new_tokens=int(item.get("max_new_tokens", 256)),
                    temperature=float(item.get("temperature", 0.9)),
                    top_p=float(item.get("top_p", 0.95)),
                    do_sample=bool(item.get("do_sample", True)),
                    n_variants=int(item.get("n_variants", 1)),
                    device=str(item.get("device", "cuda")),
                    cache_dir=_pick_cache_dir(item),
                    dtype=str(item.get("dtype", "bf16")),
                    system_prompt=str(item.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
                ))

            elif backend in ("api", "chatgpt"):
                # ✅ model 字段用于 API 的 model 名；不填给默认
                model = str(_pick(item, ["model"], "deepseek-chat"))
                cls = ChatGPTParaphraseAttack if backend == "chatgpt" else APIPromptParaphraseAttack
                text_attackers.append(cls(
                    model=model,
                    base_url=str(item.get("base_url", "https://api.deepseek.com")),
                    api_key=str(item.get("api_key", "")),
                    system_prompt=str(item.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
                    temperature=float(item.get("temperature", 0.9)),
                    max_tokens=int(item.get("max_tokens", 1024)),
                    timeout=float(item.get("timeout", 60.0)),
                    retries=int(item.get("retries", 5)),
                    sleep=float(item.get("sleep", 0.2)),
                    n_variants=int(item.get("n_variants", 1)),
                ))
            else:
                raise ValueError(f"Unknown paraphrase backend: {backend}")

            continue

        # --------------------
        # typo & its 4 meta attacks: inse/dele/subs/tran
        # --------------------
        if tp in ("typo", "inse", "dele", "subs", "tran"):
            # ✅ 规则：只有 tp=="typo" 才归为 typo 类
            # ✅ tp in (inse/dele/subs/tran) 时，输出类别就是它们自己（attack 名称与分文件 key 都会随之变化）
            mode = tp
            if tp == "typo":
                # 不写 mode -> mix；也支持显式 mode（mix/insert/delet/subst/trans 或 inse/dele/subs/tran）
                mode = str(item.get("mode") or var_hint or "mix").strip().lower()

            atk = TypoAttack(
                mode=mode,
                pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                n_variants=int(item.get("n_variants", 1)),
            )

            # ✅ 关键：让 ta.name 变成 'dele'/'inse'/'subs'/'tran'
            # 这样 builder 里写出的 attack/meta/active_attack 都会变成 text_dele / text_inse / ...
            if tp in ("inse", "dele", "subs", "tran"):
                try:
                    atk.name = tp
                except Exception:
                    # 如果 TypoAttack.name 是只读 property，就退化为加一个新属性（尽量不报错）
                    try:
                        setattr(atk, "name", tp)
                    except Exception:
                        pass

            text_attackers.append(atk)
            continue

        # --------------------
        # homo (Homoglyph Alteration)
        # --------------------
        if tp == "homo":
            variant = str(item.get("variant") or var_hint or "ECES").strip().upper()
            text_attackers.append(ViperHomoglyphAttack(
                mode=variant,
                pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                n_variants=int(item.get("n_variants", 1)),
            ))
            continue

        # --------------------
        # form (Format Character Editing)
        # --------------------
        if tp == "form":
            variant = str(item.get("variant") or var_hint or "zero-sp").strip().lower()
            text_attackers.append(FormatAttack(
                mode=variant,
                pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                n_variants=int(item.get("n_variants", 1)),
            ))
            continue

        # --------------------
        # syno (Synonyms Substitution)
        #   backend: modelfree / modelbase
        # --------------------
        if tp == "syno":
            backend = str(item.get("backend") or "modelfree").strip().lower()
            if backend == "modelfree":
                text_attackers.append(SynonymSubstitutionAttack(
                    pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                    n_variants=int(item.get("n_variants", 1)),
                ))
            elif backend == "modelbase":
                # ✅ 可指定本地目录；不指定默认加载 HF（这里给 gpt2 兜底）
                model = str(_pick(item, ["model", "replacement_model_name_or_path"], "gpt2"))
                text_attackers.append(WordSubstModelBaseAttack(
                    replacement_model_name_or_path=model,
                    pct_words_masked=float(item.get("pct_words_masked", 0.6)),
                    num_replacement_retry=int(item.get("num_replacement_retry", 3)),
                    n_variants=int(item.get("n_variants", 1)),
                    device=str(item.get("device", "cuda")),
                    dtype=str(item.get("dtype", "bf16")),
                    cache_dir=_pick_cache_dir(item),
                ))
            else:
                raise ValueError(f"Unknown syno backend: {backend}")
            continue

        # --------------------
        # back_trans (Back Translation)
        # --------------------
        if tp == "back_trans":
            # ✅ 新增约定：pivot_lang 不写 => "auto"
            pivot_lang = str(item.get("pivot_lang", "auto")).strip().lower()

            text_attackers.append(BackTranslationAttack(
                pivot_lang=pivot_lang,                 # "auto" | "en" | "de" | ...
                src_lang=item.get("src_lang", None),

                # ✅ 新增：auto 策略参数（见下面 text_attacks.py 的改动）
                pivot_for_en=str(item.get("pivot_for_en", "de")).strip().lower(),
                pivot_for_non_en=str(item.get("pivot_for_non_en", "en")).strip().lower(),

                n_rounds=int(item.get("n_rounds", 1)),
                n_variants=int(item.get("n_variants", 1)),
                num_beams=int(item.get("num_beams", 5)),
                do_sample=bool(item.get("do_sample", False)),
                temperature=float(item.get("temperature", 1.0)),
                top_p=float(item.get("top_p", 1.0)),
                max_length=int(item.get("max_length", 512)),
                device=str(item.get("device", "cuda")),
                cache_dir=_pick_cache_dir(item),
                dtype=str(item.get("dtype", "bf16")),
            ))
            continue
        # --------------------
        # humanize (Anthropomorphic / Humanization rewrite)
        #   backend: api / hf
        # --------------------
        if tp == "humanize":
            backend = str(item.get("backend", "api")).strip().lower()

            # 数据集路径：你说“接受传入的一个数据集”，这里用 attack_dataset_path
            ds = _pick(item, ["attack_dataset_path", "dataset", "attack_dataset", "dataset_path", "data"], None)
            if not ds:
                raise ValueError("humanize attack requires `attack_dataset_path` (or `dataset`).")
            import os 
        
            # 相对路径：以 config 文件所在目录为基准
            if isinstance(ds, str) and path:
                base = os.path.dirname(os.path.abspath(path))
                if not os.path.isabs(ds):
                    ds = os.path.join(base, ds)

            # 公共参数
            common = dict(
                attack_dataset_path=str(ds),
                n_pairs=int(item.get("n_pairs", 3)),
                max_input_tokens=int(item.get("max_input_tokens", 4096)),
                max_output_tokens=int(item.get("max_output_tokens", 512)),
                temperature=float(item.get("temperature", 0.9)),
                top_p=float(item.get("top_p", 0.95)),
                n_variants=int(item.get("n_variants", 1)),
            )

            if backend == "api":
                text_attackers.append(HumanizeAttackAPI(
                    **common,
                    model=str(item.get("model", "deepseek-chat")),
                    base_url=str(item.get("base_url", "https://api.deepseek.com")),
                    api_key=str(item.get("api_key", "")),
                    top_k=item.get("top_k", None),
                    frequency_penalty=float(item.get("frequency_penalty", 0.0)),
                    presence_penalty=float(item.get("presence_penalty", 0.0)),
                    timeout=float(item.get("timeout", 60.0)),
                    retries=int(item.get("retries", 5)),
                    sleep=float(item.get("sleep", 0.2)),
                ))
            elif backend == "hf":
                model = _pick(item, ["model", "model_name_or_path"], None)
                if not model:
                    raise ValueError("humanize backend=hf requires `model` / `model_name_or_path`.")
                text_attackers.append(HumanizeAttackHF(
                    **common,
                    model_name_or_path=str(model),
                    do_sample=bool(item.get("do_sample", True)),
                    top_k=item.get("top_k", 50),
                    device=str(item.get("device", "cuda")),
                    cache_dir=_pick_cache_dir(item),
                    dtype=str(item.get("dtype", "bf16")),
                ))
            else:
                raise ValueError(f"Unknown humanize backend: {backend}")

            continue

        else:
            raise ValueError(f"Unknown text attack type: {tp}")

    regen_attackers = []
    for item in rcfg:
        tp = (item.get("type") or "").strip().lower()
        if tp in ("none", "no", "noop", "noregen"):
            continue
        if tp == "temp_sweep":
            temps = item.get("temps", [])
            regen_attackers.append(TemperatureSweep(temps=[float(x) for x in temps]))
        elif tp == "top_p_sweep":
            top_ps = item.get("top_ps", [])
            regen_attackers.append(TopPSweep(top_ps=[float(x) for x in top_ps]))
        elif tp == "greedy_vs_sample":
            regen_attackers.append(GreedyVsSample(sample_temperature=float(item.get("sample_temperature", 1.0))))
        else:
            raise ValueError(f"Unknown regen attack type: {tp}")

    if not text_attackers:
        text_attackers = [NoAttack()]
    if not regen_attackers:
        regen_attackers = [NoRegen()]

    return AttackBundle(text_attackers=text_attackers, regen_attackers=regen_attackers)
