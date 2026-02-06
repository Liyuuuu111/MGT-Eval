# mgt_eval/dataset_builder/builder.py
from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import hashlib
import uuid
import re
import numpy as np
import time

from data_utils.load import load_dataset_unified

from .config import BuildConfig, GenConfig
from .token_utils import take_first_k_tokens
from .io_utils import JsonlWriter
from .attacks.factory import load_attacks_from_config
from .backends.base import LLMBackend, GenerationResult

from .quality_metrics import QualityConfig, TextQualityEvaluator
from .quality_summary import RunningQualityStats

# ---- tqdm desc style ----
W_TOK = 12   # genTok: total generated new tokens (cumulative)
W_RATE = 10  # tok/s
W_MEM = 8    # GPU_mem (reserved), train-style
W_DPPL = 10  # avg delta ppl
W_DREA = 10  # avg delta readability (FRE)
W_BERT = 9   # avg bertscore f1
SEP = " "

# ---- logger ----
logger = logging.getLogger("mgt_eval.dataset_builder")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _attack_abbr(name: str, max_len: int = 12) -> str:
    """
    Abbreviate the attack name, ensuring:
      - no '-'
      - only [a-zA-Z0-9_]
      - length <= max_len
    """
    if not name:
        return "na"
    s = name.strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if not s:
        return "na"
    return s[:max_len]


def _make_text_obj(text: str, attack_abbr: str, base_id: str, **extra: Any) -> Dict[str, Any]:
    """
    One unique id per text segment: uuid-attackabbr-base_id
    """
    uid = uuid.uuid4().hex
    atk = _attack_abbr(attack_abbr)
    tid = f"{uid}-{atk}-{base_id}"
    obj: Dict[str, Any] = {"id": tid, "text": text}
    if extra:
        obj.update(extra)
    return obj


def _safe_get_id(ex: Dict[str, Any]) -> str:
    for k in ("id", "qid", "question_id"):
        if k in ex and ex[k] is not None and str(ex[k]).strip() != "":
            return str(ex[k])
    return _stable_hash(ex.get("text", ""))


def _filter_keep_fields(ex: Dict[str, Any], keep_fields: Optional[Sequence[str]]) -> Dict[str, Any]:
    if keep_fields is None:
        return dict(ex)
    out: Dict[str, Any] = {}
    for k in keep_fields:
        if k in ex:
            out[k] = ex[k]
    return out

def _infer_lang(text: str, lang_hint: Optional[str] = None) -> str:
    """
    More robust language ID (trust dataset hint first; clean text + confidence/consistency checks).
    Returns: 'en','zh','de','ru',...; fallback 'und' or 'en'.
    """
    def _norm(x: str) -> str:
        x = (x or "").strip().lower()
        if x in ("zh-cn", "zh_cn", "zh-hans", "zh_hans", "zh-tw", "zh_tw", "zh-hant", "zh_hant"):
            return "zh"
        return x

    if isinstance(lang_hint, str) and lang_hint.strip():
        return _norm(lang_hint)

    s0 = (text or "").strip()
    if not s0:
        return "und"

    # --- script stats (for robust override) ---
    def _script_stats(s: str) -> Dict[str, float]:
        letters = sum(1 for ch in s if ch.isalpha())
        latin = sum(1 for ch in s if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        cyr = sum(1 for ch in s if 0x0400 <= ord(ch) <= 0x04FF)
        arab = sum(1 for ch in s if 0x0600 <= ord(ch) <= 0x06FF)
        cjk = sum(1 for ch in s if (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF) or (0x3040 <= ord(ch) <= 0x30FF))
        denom = max(1, letters)
        return {
            "letters": float(letters),
            "latin_r": latin / denom,
            "cyr_r": cyr / denom,
            "arab_r": arab / denom,
            "cjk_r": cjk / denom,
        }

    st = _script_stats(s0)

    # --- quick heuristic: strong script signals ---
    if st["cjk_r"] >= 0.20:
        return "zh"
    if st["arab_r"] >= 0.20:
        return "ar"
    if st["cyr_r"] >= 0.20:
        return "ru"

    # --- clean text for statistical LID (remove urls/noise) ---
    s = s0
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)
    s = re.sub(r"[\d_]+", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    # too short => not reliable
    if len(s) < 40:
        # Prefer conservative 'en' here to avoid false 'ru'.
        return "en" if st["latin_r"] >= 0.50 else "und"

    cands: List[Tuple[str, float]] = []

    # --- langdetect with prob ---
    try:
        from langdetect import detect_langs  # type: ignore
        rs = detect_langs(s)
        if rs:
            top = rs[0]
            cands.append((_norm(top.lang), float(getattr(top, "prob", 0.0))))
    except Exception:
        pass

    # --- langid with margin heuristic ---
    try:
        import langid  # type: ignore
        ranked = langid.rank(s)
        if ranked:
            lang1, score1 = ranked[0]
            lang2, score2 = ranked[1] if len(ranked) > 1 else ("", score1 - 999.0)
            margin = float(score1 - score2)
            # Map margin to a pseudo-confidence (0~1) for merging with langdetect.
            conf = max(0.0, min(1.0, 0.5 + margin / 20.0))
            cands.append((_norm(lang1), conf))
    except Exception:
        pass

    # --- pick best ---
    if not cands:
        return "en" if st["latin_r"] >= 0.50 else "und"

    # If both candidates agree, use it; otherwise pick the higher-confidence one.
    cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)
    best_lang = cands_sorted[0][0]

    # --- final script-consistency override (THIS fixes your ru false positive) ---
    if best_lang == "ru" and st["cyr_r"] < 0.05 and st["latin_r"] >= 0.60:
        return "en"
    if best_lang == "zh" and st["cjk_r"] < 0.05:
        return "en"
    if best_lang == "ar" and st["arab_r"] < 0.05:
        return "en"
    return best_lang or "en"


def _infer_model_short_name(model_ref: Optional[str]) -> Optional[str]:
    """
    Turn
      '/data/dai/model/EleutherAI/pythia-70m' -> 'pythia-70m'
      'EleutherAI/pythia-70m'               -> 'pythia-70m'
      'gpt2'                                 -> 'gpt2'
    """
    if model_ref is None:
        return None
    s = str(model_ref).strip()
    if not s:
        return None
    # normalize trailing slash
    s = s.rstrip("/")

    # path-like
    try:
        p = Path(s)
        # even if path doesn't exist, Path.name still works
        name = p.name
        if name:
            return name
    except Exception:
        pass

    # hf_id-like
    if "/" in s:
        tail = s.split("/")[-1].strip()
        return tail or s

    return s

class DatasetBuilder:
    def __init__(
        self,
        backend: Optional[LLMBackend],   # ✅ allow None
        cfg: BuildConfig,
        quality_cfg: Optional[QualityConfig] = None,
    ) -> None:
        self.backend = backend
        self.cfg = cfg

        self.attacks = load_attacks_from_config(cfg.attacks_config_path)
        self.rng = np.random.RandomState(int(cfg.sample_seed))

        # tokenizer
        self.tokenizer = None
        if bool(getattr(cfg, "attack_dataset_only", False)) and backend is None:
            # attack-only: do not load any tokenizer/model
            self.tokenizer = None
        else:
            if cfg.tokenizer_strategy == "auto":
                self.tokenizer = getattr(backend, "get_tokenizer", lambda: None)()
            elif cfg.tokenizer_strategy.startswith("hf:"):
                name = cfg.tokenizer_strategy[len("hf:") :].strip()
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            elif cfg.tokenizer_strategy == "whitespace":
                self.tokenizer = None
            else:
                raise ValueError(f"Unknown tokenizer_strategy: {cfg.tokenizer_strategy}")

        # quality evaluator (CLI sets None in attack-only mode; won't load)
        self.quality_cfg = quality_cfg
        self.quality = None
        if quality_cfg is not None and quality_cfg.any_enabled():
            self.quality = TextQualityEvaluator(quality_cfg)

    def _render_prompt(self, prefix: str, ex: Dict[str, Any], lang_override: Optional[str] = None) -> str:
        meta = {
            "id": _safe_get_id(ex),
            "lang": lang_override if (isinstance(lang_override, str) and lang_override.strip()) else ex.get("lang", None),
            "source": ex.get("source", None),
        }
        meta_json = json.dumps(meta, ensure_ascii=False)
        return self.cfg.prompt_template.format(
            prefix=prefix,
            text=ex.get("text", ""),
            id=meta["id"],
            lang=meta.get("lang", ""),
            source=meta.get("source", ""),
            label=str(ex.get("label", "")),
            meta_json=meta_json,
        )

    def _generate_one(self, prompt: str, gen: GenConfig) -> GenerationResult:
        return self.backend.generate(prompt=prompt, gen=gen, system_prompt=self.cfg.system_prompt)

    def _generate_batch(self, prompts: List[str], gen: GenConfig) -> List[GenerationResult]:
        if not prompts:
            return []
        if hasattr(self.backend, "generate_batch"):
            try:
                return self.backend.generate_batch(prompts, gen=gen, system_prompt=self.cfg.system_prompt)  # type: ignore[attr-defined]
            except Exception:
                pass
        return [self._generate_one(p, gen) for p in prompts]

    def build(self) -> Dict[str, Any]:
        examples, group_cols = load_dataset_unified(
            self.cfg.dataset_spec,
            sample_k=None,
            sample_seed=self.cfg.sample_seed,
        )
        
        # ========== NEW: attack-only path ==========
        if bool(getattr(self.cfg, "attack_dataset_only", False)):
            return self._build_attack_only(examples=examples, group_cols=group_cols)
        # prompt sources by label
        src = [ex for ex in examples if int(ex.get("label", -1)) == int(self.cfg.prompt_from_label)]
        if not src:
            raise ValueError(
                f"No examples found with label={self.cfg.prompt_from_label} in dataset={self.cfg.dataset_spec}"
            )

        # optional sampling
        if self.cfg.max_prompts is not None and self.cfg.max_prompts > 0 and len(src) > self.cfg.max_prompts:
            idx = self.rng.choice(len(src), size=int(self.cfg.max_prompts), replace=False)
            src = [src[int(i)] for i in idx.tolist()]

        # main writer (only base gen; attacks go to per-attack files)
        writer = JsonlWriter(self.cfg.out_jsonl, append=False)

        # --- attack outputs (single folder file by default; optional per-attack split) ---
        save_attack_outputs = bool(getattr(self.cfg, "save_attack_outputs", True))
        save_attack_folder = bool(getattr(self.cfg, "save_attack_folder", True)) if save_attack_outputs else False
        split_by_attack = (not save_attack_folder) if save_attack_outputs else False

        attack_writers: Dict[str, JsonlWriter] = {}
        attack_written: Dict[str, int] = {}
        attack_dir: Optional[Path] = None
        attack_all_path: Optional[str] = None

        def _attack_dir_path() -> Path:
            nonlocal attack_dir
            if attack_dir is None:
                p = Path(self.cfg.out_jsonl)
                attack_dir = p.with_name(f"{p.stem}.attacks")
                attack_dir.mkdir(parents=True, exist_ok=True)
            return attack_dir

        def _attack_out_path(attack_key: str) -> str:
            # default: single attacks.jsonl under out.attacks/
            if save_attack_folder:
                return str(_attack_dir_path() / "attacks.jsonl")
            # legacy: out.<attack>.jsonl
            p = Path(self.cfg.out_jsonl)
            abbr = _attack_abbr(attack_key, max_len=24)
            return str(p.with_name(f"{p.stem}.{abbr}{p.suffix}"))

        def _writer_key(attack_key: str) -> str:
            return "__all__" if save_attack_folder else attack_key

        def _get_attack_writer(attack_key: str) -> Tuple[JsonlWriter, str]:
            nonlocal attack_all_path
            key = _writer_key(attack_key)
            if key not in attack_writers:
                out_path = _attack_out_path(attack_key)
                attack_writers[key] = JsonlWriter(out_path, append=False)
                attack_written[key] = 0
                if save_attack_folder:
                    attack_all_path = out_path
            return attack_writers[key], key

        n_written = 0
        n_base_gen = 0
        n_regen = 0
        n_text_attack = 0
        n_quality = 0

        # ---- quality columns toggles (only show when enabled) ----
        show_dppl = bool(self.quality_cfg is not None and getattr(self.quality_cfg, "enable_ppl", False))
        show_drea = bool(self.quality_cfg is not None and getattr(self.quality_cfg, "enable_readability", False))
        show_bert = bool(self.quality_cfg is not None and getattr(self.quality_cfg, "enable_bertscore", False))

        qstats: Optional[RunningQualityStats] = None
        if self.quality is not None and (show_dppl or show_drea or show_bert):
            qstats = RunningQualityStats(enable_dppl=show_dppl, enable_drea=show_drea, enable_bert=show_bert)

        try:
            from tqdm.auto import tqdm as tqdm_cls  # type: ignore

            # ---- train.py-style header (dynamic, NO prompt column) ----
            header = (
                "\n"
                f"{'genTok':>{W_TOK}}{SEP}"
                f"{'tok/s':>{W_RATE}}{SEP}"
                f"{'GPU_mem':>{W_MEM}}"
            )
            if show_dppl:
                header += f"{SEP}{'dPPL':>{W_DPPL}}"
            if show_drea:
                header += f"{SEP}{'dFRE':>{W_DREA}}"
            if show_bert:
                header += f"{SEP}{'BERT_F1':>{W_BERT}}"
            print(header)

            it = tqdm_cls(
                total=len(src),
                desc="Build",
                leave=True,
                dynamic_ncols=True,
                disable=(len(src) == 0),
            )
        except Exception:
            it = None

        # ---- token counters for progress ----
        t0 = time.time()
        total_gen_tokens = 0  # cumulative generated NEW tokens (completion only), base+regen

        def _fmt_float(v: Optional[float], width: int, prec: int) -> str:
            if v is None:
                return f"{'-':>{width}}"
            try:
                return f"{float(v):>{width}.{prec}f}"
            except Exception:
                return f"{'-':>{width}}"

        def _gpu_mem_str() -> str:
            try:
                import torch
                if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                    return "0G"
                return f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
            except Exception:
                return "0G"

        def _count_prompt_tokens_fallback(text: str) -> int:
            if self.tokenizer is not None:
                try:
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    pass
            return len((text or "").split())

        def _count_text_tokens(text: str) -> int:
            if self.tokenizer is not None:
                try:
                    return len(self.tokenizer.encode(text or "", add_special_tokens=False))
                except Exception:
                    pass
            return len((text or "").split())

        def _count_new_tokens_from_meta(res_meta: Dict[str, Any], completion_text: str) -> int:
            try:
                inp = res_meta.get("input_tokens", None)
                out = res_meta.get("output_tokens", None)
                if isinstance(inp, int) and isinstance(out, int) and out >= inp:
                    return int(out - inp)
            except Exception:
                pass
            if self.tokenizer is not None:
                try:
                    return len(self.tokenizer.encode(completion_text, add_special_tokens=False))
                except Exception:
                    pass
            return len((completion_text or "").split())

        def _dedup_texts_with_meta(texts: List[str], metas: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
            assert len(texts) == len(metas)
            seen = set()
            out_t: List[str] = []
            out_m: List[Dict[str, Any]] = []
            for t, m in zip(texts, metas):
                if t in seen:
                    continue
                seen.add(t)
                out_t.append(t)
                out_m.append(m)
            return out_t, out_m

        def _pbar_update(n: int = 1) -> None:
            if it is not None:
                try:
                    it.update(n)
                except Exception:
                    pass

        def _pbar_desc() -> None:
            if it is None or not hasattr(it, "set_description"):
                return
            elapsed = max(1e-6, time.time() - t0)
            tok_per_s = total_gen_tokens / elapsed
            avg_dppl = qstats.mean_dppl() if qstats is not None else None
            avg_dfre = qstats.mean_dfre() if qstats is not None else None
            avg_bert = qstats.mean_bert_f1() if qstats is not None else None

            mem = _gpu_mem_str()
            desc = (
                f"{total_gen_tokens:>{W_TOK}d}{SEP}"
                f"{tok_per_s:>{W_RATE}.2f}{SEP}"
                f"{mem:>{W_MEM}}"
            )
            if show_dppl:
                desc += f"{SEP}{_fmt_float(avg_dppl, W_DPPL, 2)}"
            if show_drea:
                desc += f"{SEP}{_fmt_float(avg_dfre, W_DREA, 2)}"
            if show_bert:
                desc += f"{SEP}{_fmt_float(avg_bert, W_BERT, 3)}"
            it.set_description(desc)

        gen_bs = max(1, int(getattr(self.cfg, "gen_batch_size", 1) or 1))

        for b_start in range(0, len(src), gen_bs):
            batch = src[b_start: b_start + gen_bs]
            batch_prompts: List[str] = []
            batch_items: List[Dict[str, Any]] = []

            for ex in batch:
                full_text = ex.get("text", "")
                if not isinstance(full_text, str) or not full_text.strip():
                    _pbar_update(1)
                    continue

                ex_id = _safe_get_id(ex)
                ex_lang = _infer_lang(full_text, ex.get("lang", None))

                prefix = take_first_k_tokens(
                    full_text,
                    k=int(self.cfg.prefix_k_tokens),
                    tokenizer=self.tokenizer,
                    strategy=self.cfg.tokenizer_strategy,
                )

                prompt = self._render_prompt(prefix=prefix, ex=ex, lang_override=ex_lang)

                batch_prompts.append(prompt)
                batch_items.append({
                    "ex": ex,
                    "full_text": full_text,
                    "ex_id": ex_id,
                    "ex_lang": ex_lang,
                    "prefix": prefix,
                    "prompt": prompt,
                })

            if not batch_items:
                continue

            base_results = self._generate_batch(batch_prompts, self.cfg.gen)

            for item, base_res in zip(batch_items, base_results):
                ex = item["ex"]
                full_text = item["full_text"]
                ex_id = item["ex_id"]
                ex_lang = item["ex_lang"]
                prefix = item["prefix"]
                prompt = item["prompt"]

                # base generation
                n_base_gen += 1

                if self.cfg.machine_text_mode == "prompt_plus":
                    base_machine_text = base_res.text
                elif self.cfg.machine_text_mode == "completion_only":
                    base_machine_text = base_res.completion
                else:
                    raise ValueError("machine_text_mode must be prompt_plus or completion_only")

                base_model_ref = base_res.meta.get("model", None)
                base_model_name = _infer_model_short_name(base_model_ref)  # e.g., 'pythia-70m'

                # token accounting: base
                base_new = _count_new_tokens_from_meta(base_res.meta, base_res.completion)
                total_gen_tokens += int(base_new)

                # regen attacks
                sampled_variants: List[str] = []
                sampled_meta: List[Dict[str, Any]] = []
                for ra in self.attacks.regen_attackers:
                    for gen2 in ra.propose(self.cfg.gen):
                        res2 = self._generate_one(prompt, gen2)
                        n_regen += 1

                        regen_new = _count_new_tokens_from_meta(res2.meta, res2.completion)
                        total_gen_tokens += int(regen_new)

                        txt2 = res2.text if self.cfg.machine_text_mode == "prompt_plus" else res2.completion
                        sampled_variants.append(txt2)
                        sampled_meta.append({"attack": ra.name, "gen_config": gen2.to_dict(), "meta": res2.meta})

                # text attacks (post-hoc)
                rewritten_variants: List[str] = []
                rewritten_meta: List[Dict[str, Any]] = []
                for ta in self.attacks.text_attackers:
                    outs = ta.apply(base_machine_text, rng=self.rng, meta={"id": ex_id, "lang": ex_lang})
                    if outs:
                        for t in outs:
                            rewritten_variants.append(t)
                            rewritten_meta.append({"attack": ta.name})
                        n_text_attack += len(outs)

                # dedup variants (keep order) + keep aligned metas
                sampled_variants, sampled_meta = _dedup_texts_with_meta(sampled_variants, sampled_meta)
                rewritten_variants, rewritten_meta = _dedup_texts_with_meta(rewritten_variants, rewritten_meta)

                # prompt_tokens (kept for prompt_obj only; NOT shown in tqdm)
                prompt_tokens = base_res.meta.get("input_tokens", None)
                if not isinstance(prompt_tokens, int):
                    prompt_tokens = _count_prompt_tokens_fallback(prompt)

                ctx = _filter_keep_fields(ex, self.cfg.keep_fields)

                # drop duplicating fields
                for k in ("text", "article", "label", "record_id"):
                    ctx.pop(k, None)

                # avoid schema collisions
                for k in ("original", "prompt", "sample", "sampled", "rewritten", "meta", "lang", "model", "tokens"):
                    ctx.pop(k, None)

                record_id = str(ex.get("id", ex_id))
                base_id = record_id

                # --------- three segments: original / prompt / sample ----------
                original_text = full_text if getattr(self.cfg, "store_original_full", False) else prefix

                original_obj = _make_text_obj(
                    original_text,
                    attack_abbr="ori",
                    base_id=base_id,
                    role="human",
                    lang=ex_lang,
                    model=None,
                    tokens=_count_text_tokens(original_text),
                )

                prompt_obj = _make_text_obj(
                    prompt,
                    attack_abbr="prm",
                    base_id=base_id,
                    role="prompt",
                    lang=ex_lang,
                    model=None,
                    tokens=_count_text_tokens(prompt),
                    prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
                )

                base_gen_obj = _make_text_obj(
                    base_machine_text,
                    attack_abbr="gen",
                    base_id=base_id,
                    role="machine",
                    lang=ex_lang,
                    model=base_model_name,
                    tokens=_count_text_tokens(base_machine_text),
                    attack="gen",
                    backend=base_res.meta.get("backend", None),
                    gen_config=self.cfg.gen.to_dict(),
                    gen_new_tokens=int(base_new),
                )

                # regen objs
                regen_objs: List[Dict[str, Any]] = []
                for txt2, m2 in zip(sampled_variants, sampled_meta):
                    atk_name = str(m2.get("attack", "regen"))
                    m2_meta = m2.get("meta", {}) or {}
                    regen_objs.append(
                        _make_text_obj(
                            txt2,
                            attack_abbr=f"rg_{atk_name}",
                            base_id=base_id,
                            role="machine",
                            lang=ex_lang,
                            model=m2_meta.get("model", base_model_name),
                            tokens=_count_text_tokens(txt2),
                            attack=atk_name,
                            gen_config=m2.get("gen_config", None),
                        )
                    )

                # text attack objs
                textatk_objs: List[Dict[str, Any]] = []
                for txt3, m3 in zip(rewritten_variants, rewritten_meta):
                    atk_name = str(m3.get("attack", "text"))
                    textatk_objs.append(
                        _make_text_obj(
                            txt3,
                            attack_abbr=f"ta_{atk_name}",
                            base_id=base_id,
                            role="machine",
                            lang=ex_lang,
                            model=base_model_name,
                            tokens=_count_text_tokens(txt3),
                            attack=atk_name,
                        )
                    )

                # all candidates for quality computation (base + all attacks)
                all_sample_objs = [base_gen_obj] + regen_objs + textatk_objs

                # --------- quality metrics (optional) ----------
                if self.quality is not None:
                    # original
                    original_obj["quality"] = self.quality.eval_original(full_text)

                    # samples (ppl/readability) + bertscore(original, sample)
                    sample_texts = [o.get("text", "") for o in all_sample_objs]
                    q_list = self.quality.eval_samples(full_text, sample_texts)

                    for o, q in zip(all_sample_objs, q_list):
                        o["quality"] = q

                    # ---- update running means for tqdm (sample-average) ----
                    if qstats is not None:
                        qstats.update_from_quality(
                            original_quality=original_obj.get("quality", {}),
                            sample_qualities=[o.get("quality", {}) for o in all_sample_objs],
                        )

                    n_quality += 1

                # ---- progress bar desc update (AFTER quality update; NO prompt column) ----
                _pbar_desc()

                # --------- top-level extra fields: lang/model/tokens ----------
                record_tokens = {
                    "original": _count_text_tokens(full_text),
                    "prompt": int(prompt_tokens) if isinstance(prompt_tokens, int) else _count_text_tokens(prompt),
                    "gen_new": int(base_new),
                    "gen_text": _count_text_tokens(base_machine_text),
                }

                meta_quality = None
                if self.quality_cfg is not None and self.quality_cfg.any_enabled():
                    meta_quality = {
                        "enabled": True,
                        "config": self.quality_cfg.to_dict(),
                    }

                base_meta: Dict[str, Any] = {
                    "builder": "mgt_eval.dataset_build",
                    "group_cols_detected": group_cols,
                    "prompt_from_label": int(self.cfg.prompt_from_label),
                    "prefix_k_tokens": int(self.cfg.prefix_k_tokens),
                    "tokenizer_strategy": self.cfg.tokenizer_strategy,
                    "prompt_template": self.cfg.prompt_template,
                    "system_prompt": self.cfg.system_prompt,
                    "machine_text_mode": self.cfg.machine_text_mode,
                    "base_generation": base_res.meta,
                    "regen_meta": sampled_meta,
                    "text_attack_meta": rewritten_meta,
                    "quality": meta_quality,
                    "split_by_attack": split_by_attack,
                }

                raw_label = ex.get("label", None)
                try:
                    src_label = int(raw_label)
                except Exception:
                    src_label = int(self.cfg.prompt_from_label)

                human_record: Dict[str, Any] = {
                    **ctx,
                    "id": f"{record_id}-human",
                    "text": original_text,
                    "role": "human",
                    "label": int(src_label),
                    "lang": ex_lang,
                    "model": ex.get("model", None),
                    "tokens": {
                        "original": record_tokens["original"],
                        "prompt": record_tokens["prompt"],
                    },
                    "meta": {**base_meta, "record_type": "human"},
                }

                machine_record: Dict[str, Any] = {
                    **ctx,
                    "id": f"{record_id}-gen",
                    "text": base_machine_text,
                    "role": "machine",
                    "label": 1,
                    "lang": ex_lang,
                    "model": base_model_name,
                    "tokens": {
                        "gen_new": record_tokens["gen_new"],
                        "gen_text": record_tokens["gen_text"],
                    },
                    "attack": "gen",
                    "backend": base_res.meta.get("backend", None),
                    "gen_config": self.cfg.gen.to_dict(),
                    "gen_new_tokens": int(base_new),
                    "human_id": human_record["id"],
                    "meta": {**base_meta, "record_type": "machine"},
                }

                writer.write(human_record)
                writer.write(machine_record)
                n_written += 2
                if (n_written % int(self.cfg.write_every)) == 0:
                    writer.flush()

                base_rec: Dict[str, Any] = {
                    **ctx,
                    "id": record_id,
                    "label": 1,
                    "lang": ex_lang,
                    "model": base_model_name,
                    "tokens": record_tokens,
                    "original": [original_obj],
                    "prompt": [prompt_obj],
                    "sample": [base_gen_obj],
                    "meta": {**base_meta, "record_type": "attack_base"},
                }

                # ---- write per-attack files: each record has [base_gen, attacked_variant] ----
                if save_attack_outputs:
                    for obj in regen_objs:
                        atk_name = str(obj.get("attack", "regen"))
                        atk_key = f"regen_{atk_name}"
                        w, wkey = _get_attack_writer(atk_key)
                        rec = dict(base_rec)
                        rec["sample"] = [base_gen_obj, obj]
                        rec_meta = dict(base_rec.get("meta", {}))
                        rec_meta["active_attack"] = atk_key
                        rec["meta"] = rec_meta
                        w.write(rec)

                        attack_written[wkey] += 1
                        if (attack_written[wkey] % int(self.cfg.write_every)) == 0:
                            try:
                                w.flush()
                            except Exception:
                                pass

                    for obj in textatk_objs:
                        atk_name = str(obj.get("attack", "text"))
                        atk_key = f"text_{atk_name}"
                        w, wkey = _get_attack_writer(atk_key)
                        rec = dict(base_rec)
                        rec["sample"] = [base_gen_obj, obj]
                        rec_meta = dict(base_rec.get("meta", {}))
                        rec_meta["active_attack"] = atk_key
                        rec["meta"] = rec_meta
                        w.write(rec)

                        attack_written[wkey] += 1
                        if (attack_written[wkey] % int(self.cfg.write_every)) == 0:
                            try:
                                w.flush()
                            except Exception:
                                pass

                _pbar_update(1)

        writer.close()
        for w in attack_writers.values():
            try:
                w.close()
            except Exception:
                pass

        # ---- save global quality means to file (sample-average) ----
        quality_means_path: Optional[str] = None
        quality_means: Optional[Dict[str, Any]] = None
        if qstats is not None:
            quality_means_path = self.cfg.out_jsonl + ".quality_means.json"
            qstats.save(quality_means_path)
            quality_means = qstats.to_dict()

        # collect per-attack output paths (for convenience)
        per_attack_files: Dict[str, str] = {}
        attack_folder: Optional[str] = None
        attack_file: Optional[str] = None
        if save_attack_outputs:
            if save_attack_folder:
                attack_folder = str(attack_dir) if attack_dir is not None else None
                attack_file = attack_all_path
            else:
                per_attack_files = {k: _attack_out_path(k) for k in attack_writers.keys()}

        return {
            "out_jsonl": self.cfg.out_jsonl,
            "attack_folder": attack_folder,
            "attack_file": attack_file,
            "per_attack_files": per_attack_files,
            "num_prompts": len(src),
            "num_written": n_written,
            "num_base_generations": n_base_gen,
            "num_regen_generations": n_regen,
            "num_text_attack_variants": n_text_attack,
            "total_generated_new_tokens": int(total_gen_tokens),
            "num_quality_records": int(n_quality),
            "quality_means_path": quality_means_path,
            "quality_means": quality_means,
        }
    
    def _build_attack_only(self, examples: List[Dict[str, Any]], group_cols: Any) -> Dict[str, Any]:
        """
        Attack-only mode (schema-aligned with normal build):
        - Do NOT generate.
        - Apply text attacks on existing dataset texts.
        - Each attacked variant is written as a normal-format record:
            {id, lang, model, tokens, original:[...], prompt:[...], sample:[...], meta:{...}}
        - attack outputs are saved to <out>.attacks/attacks.jsonl by default
          (or per-attack files when save_attack_folder=False), with sample=[base_src, attacked].
        - Optional sampling: cfg.sample_k on eligible pool (seed=cfg.sample_seed).
        """
        writer = JsonlWriter(self.cfg.out_jsonl, append=False)

        # --- attack outputs (single folder file by default; optional per-attack split) ---
        save_attack_outputs = bool(getattr(self.cfg, "save_attack_outputs", True))
        save_attack_folder = bool(getattr(self.cfg, "save_attack_folder", True)) if save_attack_outputs else False
        split_by_attack = (not save_attack_folder) if save_attack_outputs else False

        attack_writers: Dict[str, JsonlWriter] = {}
        attack_written: Dict[str, int] = {}
        attack_dir: Optional[Path] = None
        attack_all_path: Optional[str] = None

        def _attack_dir_path() -> Path:
            nonlocal attack_dir
            if attack_dir is None:
                p = Path(self.cfg.out_jsonl)
                attack_dir = p.with_name(f"{p.stem}.attacks")
                attack_dir.mkdir(parents=True, exist_ok=True)
            return attack_dir

        def _attack_out_path(attack_key: str) -> str:
            # default: single attacks.jsonl under out.attacks/
            if save_attack_folder:
                return str(_attack_dir_path() / "attacks.jsonl")
            # legacy: out.<attack>.jsonl
            p = Path(self.cfg.out_jsonl)
            abbr = _attack_abbr(attack_key, max_len=24)
            return str(p.with_name(f"{p.stem}.{abbr}{p.suffix}"))

        def _writer_key(attack_key: str) -> str:
            return "__all__" if save_attack_folder else attack_key

        def _get_attack_writer(attack_key: str) -> Tuple[JsonlWriter, str]:
            nonlocal attack_all_path
            key = _writer_key(attack_key)
            if key not in attack_writers:
                out_path = _attack_out_path(attack_key)
                attack_writers[key] = JsonlWriter(out_path, append=False)
                attack_written[key] = 0
                if save_attack_folder:
                    attack_all_path = out_path
            return attack_writers[key], key

        def _dedup_texts_with_meta(texts: List[str], metas: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
            assert len(texts) == len(metas)
            seen = set()
            out_t: List[str] = []
            out_m: List[Dict[str, Any]] = []
            for t, m in zip(texts, metas):
                if t in seen:
                    continue
                seen.add(t)
                out_t.append(t)
                out_m.append(m)
            return out_t, out_m

        # token helpers (copy the same fallback style as build())
        def _count_text_tokens(text: str) -> int:
            if getattr(self, "tokenizer", None) is not None:
                try:
                    return len(self.tokenizer.encode(text or "", add_special_tokens=False))
                except Exception:
                    pass
            return len((text or "").split())

        def _count_prompt_tokens_fallback(text: str) -> int:
            return _count_text_tokens(text)

        only_attack_machine = bool(getattr(self.cfg, "only_attack_machine", True))
        machine_label = int(getattr(self.cfg, "machine_label", 1))
        sample_k = getattr(self.cfg, "sample_k", None)
        seed = int(getattr(self.cfg, "sample_seed", 114514))

        # -------------------------
        # eligible pool (filtered)
        # -------------------------
        eligible: List[Dict[str, Any]] = []
        for ex in examples:
            t = ex.get("text", "")
            if not isinstance(t, str) or not t.strip():
                continue
            raw_lab = ex.get("label", None)
            try:
                lab = int(raw_lab)
            except Exception:
                lab = None
            if only_attack_machine and (lab is None or lab != machine_label):
                continue
            eligible.append(ex)

        # -------------------------
        # optional sampling on eligible
        # -------------------------
        selected = eligible
        if sample_k is not None:
            try:
                k = int(sample_k)
            except Exception:
                k = None
            if k is not None and k > 0 and len(eligible) > k:
                rng_sample = np.random.RandomState(seed)
                idx = rng_sample.choice(len(eligible), size=k, replace=False)
                idx = np.sort(idx)
                selected = [eligible[int(i)] for i in idx.tolist()]

        n_in_total = len(examples)
        n_eligible = len(eligible)
        n_selected = len(selected)

        n_written = 0
        n_attack_variants = 0

        def _gpu_mem_str() -> str:
            try:
                import torch
                if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                    return "0G"
                return f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
            except Exception:
                return "0G"

        text_attackers = list(self.attacks.text_attackers or [])
        if not text_attackers:
            logger.info("[attack-only] no text attacks configured; nothing to do.")
        else:
            # tqdm header (keep your style)
            header = (
                "\n"
                f"{'n_out':>8}{SEP}"
                f"{'GPU_mem':>{W_MEM}}"
            )
            print(header)

            def _emit_attack_records(ex: Dict[str, Any], attacked_texts: List[str], *, atk_name: str, atk_key: str) -> None:
                nonlocal n_written, n_attack_variants

                full_text = ex.get("text", "")
                if not isinstance(full_text, str) or not full_text.strip():
                    return

                if not attacked_texts:
                    return

                raw_lab = ex.get("label", None)
                try:
                    lab = int(raw_lab)
                except Exception:
                    lab = None

                ex_id = _safe_get_id(ex)
                record_id = str(ex.get("id", ex_id))
                base_id = record_id

                ex_lang = _infer_lang(full_text, ex.get("lang", None))
                base_model_name = _infer_model_short_name(ex.get("model", None))

                # build prefix + prompt (so prompt/original fields exist exactly like normal build)
                prefix = take_first_k_tokens(
                    full_text,
                    k=int(getattr(self.cfg, "prefix_k_tokens", 64)),
                    tokenizer=getattr(self, "tokenizer", None),
                    strategy=str(getattr(self.cfg, "tokenizer_strategy", "whitespace")),
                )
                prompt = self._render_prompt(prefix=prefix, ex=ex, lang_override=ex_lang)
                prompt_tokens = _count_prompt_tokens_fallback(prompt)

                # ✅ attack-only: keep ONLY minimal provenance fields from the original file
                ctx: Dict[str, Any] = {}
                for k in ("id", "lang", "model", "source"):
                    if k in ex and ex.get(k, None) is not None:
                        ctx[k] = ex.get(k)

                for k in (
                    "text", "article", "label", "record_id",
                    "original", "prompt", "sample", "sampled", "rewritten", "meta", "tokens"
                ):
                    ctx.pop(k, None)

                # --------- original/prompt objects (schema aligned) ----------
                original_text = full_text if getattr(self.cfg, "store_original_full", False) else prefix
                original_obj = _make_text_obj(
                    original_text,
                    attack_abbr="ori",
                    base_id=base_id,
                    role="machine" if (lab == machine_label) else "human",
                    lang=ex_lang,
                    model=base_model_name,
                    tokens=_count_text_tokens(original_text),
                    orig_label=lab,
                )

                prompt_obj = _make_text_obj(
                    prompt,
                    attack_abbr="prm",
                    base_id=base_id,
                    role="prompt",
                    lang=ex_lang,
                    model=None,
                    tokens=_count_text_tokens(prompt),
                    prompt_tokens=int(prompt_tokens),
                )

                base_src_obj = _make_text_obj(
                    full_text,
                    attack_abbr="src",
                    base_id=base_id,
                    role="machine",
                    lang=ex_lang,
                    model=base_model_name,
                    tokens=_count_text_tokens(full_text),
                    attack="src",
                    backend="dataset",
                    gen_config=None,
                    gen_new_tokens=0,
                    orig_label=lab,
                )

                attacked_texts, attacked_meta = _dedup_texts_with_meta(
                    attacked_texts, [{"attack": atk_name} for _ in range(len(attacked_texts))]
                )
                if not attacked_texts:
                    return

                for t in attacked_texts:
                    attacked_obj = _make_text_obj(
                        t,
                        attack_abbr=f"ta_{atk_name}",
                        base_id=base_id,
                        role="machine",
                        lang=ex_lang,
                        model=base_model_name,
                        tokens=_count_text_tokens(t),
                        attack=atk_name,
                    )

                    record_tokens = {
                        "original": _count_text_tokens(full_text),
                        "prompt": int(prompt_tokens),
                        "gen_new": 0,
                        "gen_text": _count_text_tokens(t),
                    }

                    base_rec: Dict[str, Any] = {
                        **ctx,
                        "id": record_id,
                        "label": int(lab) if lab is not None else int(machine_label),
                        "lang": ex_lang,
                        "model": base_model_name,
                        "tokens": record_tokens,
                        "original": [original_obj],
                        "prompt": [prompt_obj],
                        "sample": [attacked_obj],
                        "meta": {
                            "builder": "mgt_eval.dataset_attack_only",
                            "group_cols_detected": group_cols,
                            "attack_dataset_only": True,
                            "only_attack_machine": only_attack_machine,
                            "machine_label": machine_label,
                            "sample_k": sample_k,
                            "seed": seed,
                            "active_attack": atk_key,
                            "text_attack_meta": attacked_meta,
                            "split_by_attack": split_by_attack,
                            "base_source": {
                                "orig_label": lab,
                                "source_id": record_id,
                            },
                        },
                    }

                    writer.write(base_rec)
                    n_written += 1
                    n_attack_variants += 1
                    if (n_written % int(getattr(self.cfg, "write_every", 100))) == 0:
                        writer.flush()

                    if save_attack_outputs:
                        w, wkey = _get_attack_writer(atk_key)
                        rec2 = dict(base_rec)
                        rec2_meta = dict(base_rec.get("meta", {}))
                        rec2_meta["active_attack"] = atk_key
                        rec2["meta"] = rec2_meta
                        rec2["sample"] = [base_src_obj, attacked_obj]
                        w.write(rec2)

                        attack_written[wkey] += 1
                        if (attack_written[wkey] % int(getattr(self.cfg, "write_every", 100))) == 0:
                            try:
                                w.flush()
                            except Exception:
                                pass

            for ta in text_attackers:
                atk_name = str(getattr(ta, "name", None) or "text")
                atk_key = f"text_{atk_name}"

                logger.info("Processing attack: %s ...", atk_name)
                batch_size = int(getattr(ta, "vllm_batch_size", 1) or 1)
                use_batch = bool(hasattr(ta, "apply_batch") and batch_size > 1)

                try:
                    from tqdm.auto import tqdm as tqdm_cls  # type: ignore
                    if use_batch:
                        it = tqdm_cls(total=len(selected), desc=f"Attack[{atk_name}]", leave=True, dynamic_ncols=True)
                    else:
                        it = tqdm_cls(selected, desc=f"Attack[{atk_name}]", leave=True, dynamic_ncols=True, total=len(selected))
                except Exception:
                    it = None

                if use_batch:
                    for s in range(0, len(selected), batch_size):
                        batch = selected[s:s + batch_size]
                        texts = [ex.get("text", "") for ex in batch]
                        outs_list = ta.apply_batch(texts, rng=self.rng, meta=None)  # type: ignore[attr-defined]
                        for ex, outs in zip(batch, outs_list):
                            _emit_attack_records(ex, outs, atk_name=atk_name, atk_key=atk_key)
                            if it is not None:
                                it.update(1)
                                if hasattr(it, "set_description"):
                                    mem = _gpu_mem_str()
                                    it.set_description(f"{n_attack_variants:>8d}{SEP}{mem:>{W_MEM}}")
                    if it is not None:
                        it.close()
                else:
                    for ex in (it if it is not None else selected):
                        outs = ta.apply(ex.get("text", ""), rng=self.rng, meta={"id": _safe_get_id(ex)})
                        _emit_attack_records(ex, outs, atk_name=atk_name, atk_key=atk_key)
                        if it is not None and hasattr(it, "set_description"):
                            mem = _gpu_mem_str()
                            it.set_description(f"{n_attack_variants:>8d}{SEP}{mem:>{W_MEM}}")

                logger.info("Finished attack: %s", atk_name)

        writer.close()
        for w in attack_writers.values():
            try:
                w.close()
            except Exception:
                pass

        per_attack_files: Dict[str, str] = {}
        attack_folder: Optional[str] = None
        attack_file: Optional[str] = None
        if save_attack_outputs:
            if save_attack_folder:
                attack_folder = str(attack_dir) if attack_dir is not None else None
                attack_file = attack_all_path
            else:
                per_attack_files = {k: _attack_out_path(k) for k in attack_writers.keys()}

        return {
            "out_jsonl": self.cfg.out_jsonl,
            "attack_folder": attack_folder,
            "attack_file": attack_file,
            "per_attack_files": per_attack_files,
            "num_in": int(n_in_total),
            "num_eligible": int(n_eligible),
            "num_selected": int(n_selected),
            "num_written": int(n_written),
            "num_attack_variants": int(n_attack_variants),
            "note": "attack-only outputs are schema-aligned with normal build (original/prompt/sample/meta/tokens/lang/model).",
        }
