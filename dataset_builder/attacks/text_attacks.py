from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, ClassVar
import os
import re
import time

from typing import Optional

def _resolve_hf_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """
    None / "" / "hf" / "default" => 不传 cache_dir，让 HF 用用户全局默认 cache
    其他 => 直接用用户给的路径
    """
    if cache_dir is None:
        return None
    s = str(cache_dir).strip()
    if s == "" or s.lower() in {"hf", "default", "global"}:
        return None
    return s

# -----------------------
# Utilities (rng-safe)
# -----------------------
def _rng_rand(rng) -> float:
    # numpy RandomState / Generator
    if hasattr(rng, "random") and callable(getattr(rng, "random")):
        return float(rng.random())
    if hasattr(rng, "rand") and callable(getattr(rng, "rand")):
        return float(rng.rand())
    # python.Random fallback
    import random
    return random.random()

def _rng_randint(rng, low: int, high: int) -> int:
    # returns int in [low, high)
    if hasattr(rng, "randint") and callable(getattr(rng, "randint")):
        return int(rng.randint(low, high))
    import random
    return random.randrange(low, high)

def _rng_choice(rng, seq: List[Any], p: Optional[List[float]] = None) -> Any:
    if hasattr(rng, "choice") and callable(getattr(rng, "choice")):
        # numpy: rng.choice supports p
        if p is not None:
            return rng.choice(seq, p=p)
        return rng.choice(seq)
    import random
    if p is None:
        return random.choice(seq)
    # weighted choice
    return random.choices(seq, weights=p, k=1)[0]

def _rng_sample_no_replace(rng, n: int, k: int) -> List[int]:
    # sample k indices from range(n) without replacement
    if k <= 0:
        return []
    k = min(k, n)
    if hasattr(rng, "choice") and callable(getattr(rng, "choice")):
        idx = rng.choice(n, size=k, replace=False)
        return [int(x) for x in idx.tolist()] if hasattr(idx, "tolist") else [int(x) for x in idx]
    import random
    return random.sample(list(range(n)), k)

def _dedup_strs(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------
# Tokenization helpers
# -----------------------
_WORD_OR_PUNCT = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def _split_words_ws(text: str) -> List[str]:
    return (text or "").split()

def _join_words_ws(words: List[str]) -> str:
    return " ".join(words)

def _tokenize_word_or_punct(text: str) -> List[str]:
    return _WORD_OR_PUNCT.findall(text or "")

def _detokenize_word_or_punct(tokens: List[str]) -> str:
    # join with spaces then fix common punctuation spacing
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r"\s+\"", "\"", s)
    s = re.sub(r"\"\s+", "\"", s)
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r"'\s+", "'", s)
    return s.strip()

# -----------------------
# Base style
# -----------------------
@dataclass
class NoAttack:
    name: str = "none"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        return []

# -----------------------
# Lightweight attacks (keep your existing ones)
# -----------------------
@dataclass
class WordDropAttack:
    p: float = 0.05
    min_words: int = 5
    n_variants: int = 1
    name: str = "word_drop"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        words = _split_words_ws(text)
        if len(words) <= self.min_words:
            return []
        outs: List[str] = []
        for _ in range(self.n_variants):
            keep = []
            for w in words:
                if _rng_rand(rng) < self.p:
                    continue
                keep.append(w)
            if len(keep) < self.min_words:
                keep = words[: self.min_words]
            out = _join_words_ws(keep)
            if out != text:
                outs.append(out)
        return _dedup_strs(outs)

@dataclass
class WordSwapAttack:
    p: float = 0.05
    n_variants: int = 1
    name: str = "word_swap"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        words = _split_words_ws(text)
        if len(words) < 8:
            return []
        outs: List[str] = []
        for _ in range(self.n_variants):
            w = words[:]
            i = 0
            while i < len(w) - 1:
                if _rng_rand(rng) < self.p:
                    w[i], w[i + 1] = w[i + 1], w[i]
                    i += 2
                else:
                    i += 1
            out = _join_words_ws(w)
            if out != text:
                outs.append(out)
        return _dedup_strs(outs)

@dataclass
class CharSwapAttack:
    p: float = 0.01
    n_variants: int = 1
    name: str = "char_swap"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if len(text) < 30:
            return []
        outs: List[str] = []
        for _ in range(self.n_variants):
            chars = list(text)
            i = 0
            while i < len(chars) - 1:
                if _rng_rand(rng) < self.p and chars[i].isalnum() and chars[i + 1].isalnum():
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 2
                else:
                    i += 1
            out = "".join(chars)
            if out != text:
                outs.append(out)
        return _dedup_strs(outs)

@dataclass
class PunctuationNoiseAttack:
    p: float = 0.05
    n_variants: int = 1
    name: str = "punct_noise"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if len(text) < 20:
            return []
        outs: List[str] = []
        punct = [",", ".", ";", ":", "!", "?"]
        for _ in range(self.n_variants):
            s = []
            for ch in text:
                s.append(ch)
                if ch.isalnum() and _rng_rand(rng) < self.p:
                    s.append(_rng_choice(rng, punct))
            out = "".join(s)
            if out != text:
                outs.append(out)
        return _dedup_strs(outs)

@dataclass
class DictionarySubstitutionAttack:
    mapping: Dict[str, str]
    p: float = 0.2
    n_variants: int = 1
    case_insensitive: bool = True
    name: str = "dict_subst"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if not self.mapping:
            return []
        toks = _tokenize_word_or_punct(text)
        outs: List[str] = []
        for _ in range(self.n_variants):
            out_toks = []
            for t in toks:
                key = t.lower() if self.case_insensitive else t
                if key in self.mapping and _rng_rand(rng) < self.p:
                    rep = str(self.mapping[key])
                    if t[:1].isupper() and rep[:1].islower():
                        rep = rep[:1].upper() + rep[1:]
                    out_toks.append(rep)
                else:
                    out_toks.append(t)
            out = _detokenize_word_or_punct(out_toks)
            if out != text:
                outs.append(out)
        return _dedup_strs(outs)

# -----------------------
# typo_* attacks (your original logic)
# -----------------------
@dataclass
class TypoAttack:
    """
    统一类别名：typo
    - mode 支持：mix/insert/delet/subst/trans
    - 也支持四元攻击前四字母：inse/dele/subs/tran
    - mode=typo 或不填 => mix
    """
    mode: str = "mix"
    pct_words_masked: float = 0.2
    n_variants: int = 1

    # 统一名称（满足你“typo 统称”的要求）
    name: str = field(default="typo", init=False)
    # 记录子类型（便于日志/保存 meta）
    variant: str = field(default="mix", init=False)

    # ✅ 关键：作为常量，不参与 dataclass 字段/__init__
    MIX_PROB: ClassVar[Dict[str, float]] = {
        "trans": 0.011,
        "delet": 0.23,
        "subst": 0.556,
        "insert": 0.203,
    }

    def __post_init__(self) -> None:
        m = (self.mode or "mix").strip().lower()
        alias = {
            "typo": "mix",
            "mix": "mix",

            # 前四字母元攻击
            "inse": "insert",
            "dele": "delet",
            "subs": "subst",
            "tran": "trans",

            # 全称
            "insertion": "insert",
            "insert": "insert",
            "deletion": "delet",
            "delete": "delet",
            "delet": "delet",
            "substitution": "subst",
            "subst": "subst",
            "transposition": "trans",
            "trans": "trans",
        }
        m = alias.get(m, m)
        if m not in ("mix", "insert", "delet", "subst", "trans"):
            raise ValueError(f"Unknown typo mode: {self.mode}")
        self.mode = m
        self.variant = m

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        words = (text or "").split()
        if not words:
            return []

        att_word_num = int(max(1, round(self.pct_words_masked * len(words))))
        att_index = _rng_sample_no_replace(rng, len(words), att_word_num)

        def trans(victim: List[str]) -> None:
            if len(victim) <= 1:
                return
            w_id = _rng_randint(rng, 0, len(victim) - 1)
            victim[w_id], victim[w_id + 1] = victim[w_id + 1], victim[w_id]

        def subst(victim: List[str]) -> None:
            if not victim:
                return
            w_id = _rng_randint(rng, 0, len(victim))
            ch = victim[w_id]
            if ch.islower():
                victim[w_id] = _rng_choice(rng, list("abcdefghijklmnopqrstuvwxyz"))
            elif ch.isupper():
                victim[w_id] = _rng_choice(rng, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

        def delet(victim: List[str]) -> None:
            if not victim:
                return
            w_id = _rng_randint(rng, 0, len(victim))
            del victim[w_id]

        def insert(victim: List[str]) -> None:
            w_id = _rng_randint(rng, 0, len(victim) + 1)
            victim.insert(w_id, _rng_choice(rng, list("abcdefghijklmnopqrstuvwxyz")))

        func_dict = {"trans": trans, "delet": delet, "subst": subst, "insert": insert}

        outs: List[str] = []
        for _ in range(self.n_variants):
            w = words[:]
            for att_id in att_index:
                victim = list(w[att_id])
                if len(victim) <= 1:
                    continue

                if self.mode == "mix":
                    keys = list(self.MIX_PROB.keys())
                    probs = [float(self.MIX_PROB[k]) for k in keys]  # sum=1.0
                    sel_mode = _rng_choice(rng, keys, p=probs)
                    func_dict[sel_mode](victim)
                else:
                    func_dict[self.mode](victim)

                w[att_id] = "".join(victim)

            out = " ".join(w)
            if out != text:
                outs.append(out)

        return _dedup_strs(outs)

# -----------------------
# form_* attacks (shift-u / zero-sp)
# -----------------------
@dataclass
class FormatAttack:
    """
    form_shift-u: sentence-level append vertical tabs
    form_zero-sp : word-level insert zero width space
    """
    mode: str = "zero-sp"
    pct_words_masked: float = 0.2
    n_variants: int = 1
    name: str = field(init=False, default="form")

    def __post_init__(self) -> None:
        m = (self.mode or "zero-sp").strip().lower()
        # keep naming compatible with your script
        self.mode = m
        self.name = f"form_{m}"

    def _sent_tokenize(self, text: str) -> List[str]:
        # prefer nltk
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            # fallback: naive
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            return [p for p in parts if p]

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = text or ""
        if not text.strip():
            return []

        outs: List[str] = []
        for _ in range(self.n_variants):
            if self.mode == "shift-u":
                sents = self._sent_tokenize(text)
                if not sents:
                    continue
                att_sent_num = self.pct_words_masked * len(sents)
                if self.pct_words_masked == 0:
                    att_sent_num = 1
                # allow pct>1 like your original code
                if self.pct_words_masked <= 1:
                    k = int(max(1, round(att_sent_num)))
                    idx = _rng_sample_no_replace(rng, len(sents), k)
                else:
                    k = int(round(att_sent_num))
                    idx = list(range(len(sents))) + _rng_sample_no_replace(rng, len(sents), max(0, k - len(sents)))
                for i in idx:
                    sents[i] = sents[i] + " \u000B\u000B "
                out = "".join(sents).strip()
            elif self.mode == "zero-sp":
                words = text.split()
                if not words:
                    continue
                att_word_num = int(max(1, round(self.pct_words_masked * len(words))))
                idx = _rng_sample_no_replace(rng, len(words), att_word_num)
                for i in idx:
                    victim = list(words[i])
                    if not victim:
                        continue
                    pos = _rng_randint(rng, 0, len(victim) + 1)
                    victim.insert(pos, "\u200B")
                    words[i] = "".join(victim)
                out = " ".join(words)
            else:
                raise ValueError(f"Unknown format mode: {self.mode}")

            if out != text:
                outs.append(out)

        return _dedup_strs(outs)

# -----------------------
# homo_* attacks (VIPER ECES / ICES)
# -----------------------
@dataclass
class ViperHomoglyphAttack:
    mode: str = "ECES"
    pct_words_masked: float = 0.2
    n_variants: int = 1
    name: str = "homo"
    variant: str = field(default="ECES", init=False)

    def __post_init__(self) -> None:
        m = (self.mode or "ECES").strip().upper()
        if m not in ("ECES", "ICES"):
            raise ValueError("mode must be ECES or ICES")
        self.mode = m
        self.variant = m

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        # keep your original behavior: direct VIPER call
        try:
            if self.mode == "ECES":
                from .VIPER.viper_eces import eces
                fn = eces
            else:
                from .VIPER.viper_ices import ices
                fn = ices
        except Exception as e:
            raise RuntimeError(
                "VIPER module not found. Please ensure VIPER/ is importable and provides "
                "VIPER.viper_eces.eces / VIPER.viper_ices.ices"
            ) from e

        outs: List[str] = []
        for _ in range(self.n_variants):
            out = fn(self.pct_words_masked, text)
            if isinstance(out, str) and out.strip() and out != text:
                outs.append(out)
        return _dedup_strs(outs)

# -----------------------
# word_subst_modelfree (WordNet synonyms)
# -----------------------
_PRONOUNS = [
    "I","me","you","he","him","she","her","it","we","us","they","them",
    "my","your","his","her","its","our","their","mine","yours","hers","ours","theirs",
    "this","that","these","those","who","whom","whose","which","what",
    "myself","yourself","himself","herself","itself","ourselves","yourselves","themselves"
]
_FUNC_WORDS = ["a","an","the","and","but","or","nor","for","yet","so","as","if","is","are","be","was","were","being","been"]
_STOP_WORDS = set([w.lower() for w in (_PRONOUNS + _FUNC_WORDS)])

def _get_synonyms_wordnet(word: str) -> List[str]:
    try:
        from nltk.corpus import wordnet
    except Exception as e:
        raise RuntimeError("NLTK wordnet is required for word_subst_modelfree. Install nltk and download wordnet.") from e

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

@dataclass
class SynonymSubstitutionAttack:
    pct_words_masked: float = 0.2
    n_variants: int = 1
    max_retries_per_subst: int = 5
    name: str = "syno"

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = text or ""
        if not text.strip():
            return []
        toks = _tokenize_word_or_punct(text)
        if len(toks) < 4:
            return []

        subst_num = int(max(1, round(self.pct_words_masked * len(toks))))
        outs: List[str] = []

        for _ in range(self.n_variants):
            words = toks[:]
            for _i in range(subst_num):
                retry = self.max_retries_per_subst
                while retry > 0:
                    idx = _rng_randint(rng, 0, len(words))
                    w = words[idx]
                    # skip punctuation
                    if re.fullmatch(r"[^\w\s]", w):
                        retry -= 1
                        continue
                    if w.lower() in _STOP_WORDS:
                        retry -= 1
                        continue
                    syns = _get_synonyms_wordnet(w)
                    if not syns:
                        retry -= 1
                        continue
                    # pick first like your code (but deterministic bias); here choose one
                    syn = syns[0].replace("_", " ")
                    if syn == w:
                        retry -= 1
                        continue
                    words[idx] = syn
                    break
            out = _detokenize_word_or_punct(words)
            if out != text and out.strip():
                outs.append(out)

        return _dedup_strs(outs)

# -----------------------
# ptb (T5 mask-filling perturbation)
# -----------------------
_EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")

def _count_masks(masked_text: str) -> int:
    return sum(1 for x in masked_text.split() if x.startswith("<extra_id_"))

def _tokenize_and_mask(
    text: str,
    *,
    span_length: int,
    pct_words_masked: float,
    buffer_size: int = 1,
    ceil_pct: bool = False,
    rng=None,
) -> str:
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"
    if not tokens:
        return text

    n_spans = pct_words_masked * len(tokens) / float(span_length + buffer_size * 2)
    if ceil_pct:
        import math
        n_spans = math.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    # keep your original constraints: avoid overlapping masks via buffer window
    while n_masks < n_spans and (len(tokens) - span_length) > 0:
        start = _rng_randint(rng, 0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    num_filled = 0
    for idx, tok in enumerate(tokens):
        if tok == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1

    return " ".join(tokens)

def _extract_fills(raw: str) -> List[str]:
    # strip pad/special tokens like your original
    raw = raw.replace("<pad>", "").replace("</s>", "").strip()
    fills = _EXTRA_ID_PATTERN.split(raw)[1:-1]
    fills = [f.strip() for f in fills]
    return fills

def _apply_fills(masked_text: str, fills: List[str]) -> str:
    toks = masked_text.split(" ")
    n_expected = _count_masks(masked_text)
    if len(fills) < n_expected:
        return ""
    for i in range(n_expected):
        key = f"<extra_id_{i}>"
        try:
            pos = toks.index(key)
        except ValueError:
            return ""
        toks[pos] = fills[i]
    return " ".join(toks).strip()

# after（推荐）
_HF_MODEL_CACHE: Dict[Tuple[str, str, str, Optional[str]], Any] = {}
_HF_TOK_CACHE: Dict[Tuple[str, Optional[str]], Any] = {}

def _get_torch_dtype(dtype: str):
    import torch
    d = (dtype or "").lower()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32

@dataclass
class PerturbationT5Attack:
    """
    对应你代码里的 ptb（mask filling perturbation）
    """
    pct_words_masked: float = 0.2
    span_length: int = 2
    buffer_size: int = 1
    mask_filling_model_name: str = "t5-large"
    mask_top_p: float = 1.0
    max_length: int = 400
    chunk_size: int = 20
    n_variants: int = 1
    device: str = "cuda"
    cache_dir: Optional[str] = None
    dtype: str = "bf16"  # bf16/fp16/fp32
    name: str = "span"
    FULL_NAME_SPAN = "Span Perturbation"

    def _load(self):
        import torch
        import transformers
        cd = self.cache_dir  # Optional[str]

        tok_key = (self.mask_filling_model_name, cd)
        if tok_key not in _HF_TOK_CACHE:
            _HF_TOK_CACHE[tok_key] = transformers.AutoTokenizer.from_pretrained(
                self.mask_filling_model_name,
                cache_dir=cd,
                use_fast=True,
            )
        tokenizer = _HF_TOK_CACHE[tok_key]

        model_key = (self.mask_filling_model_name, self.device, self.dtype, cd)
        if model_key not in _HF_MODEL_CACHE:
            torch_dtype = _get_torch_dtype(self.dtype)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.mask_filling_model_name,
                cache_dir=cd,
                torch_dtype=torch_dtype,
            )
            model.to(self.device)
            model.eval()
            _HF_MODEL_CACHE[model_key] = model

        return _HF_MODEL_CACHE[model_key], tokenizer

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = text or ""
        if not text.strip():
            return []
        mask_model, mask_tokenizer = self._load()

        outs: List[str] = []
        # n_variants 次独立 mask + fill
        for _ in range(self.n_variants):
            masked = _tokenize_and_mask(
                text,
                span_length=int(self.span_length),
                pct_words_masked=float(self.pct_words_masked),
                buffer_size=int(self.buffer_size),
                ceil_pct=False,
                rng=rng,
            )
            n_expected = _count_masks(masked)
            if n_expected <= 0:
                continue

            # stop_id = <extra_id_{max(n_expected)}>
            stop_id = mask_tokenizer.encode(f"<extra_id_{n_expected}>", add_special_tokens=False)[0]

            try:
                import torch
                toks = mask_tokenizer([masked], return_tensors="pt", padding=True).to(self.device)

                with torch.inference_mode():
                    gen = mask_model.generate(
                        **toks,
                        max_length=int(self.max_length),
                        do_sample=True,
                        top_p=float(self.mask_top_p),
                        num_return_sequences=1,
                        eos_token_id=int(stop_id),
                    )
                raw = mask_tokenizer.batch_decode(gen, skip_special_tokens=False)[0]
                fills = _extract_fills(raw)
                pert = _apply_fills(masked, fills)
            except Exception:
                pert = ""

            if pert and pert != text:
                outs.append(pert)

        return _dedup_strs(outs)

# -----------------------
# pegasus paraphrase (sentence-level)
# -----------------------
@dataclass
class PegasusParaphraseAttack:
    model_name: str = "tuner007/pegasus_paraphrase"
    top_p: float = 0.96
    temperature: float = 1.0
    no_repeat_ngram_size: int = 3
    max_length: int = 60      # per-sentence (like your original)
    sent_batch_size: int = 64
    n_variants: int = 1
    device: str = "cuda"
    cache_dir: Optional[str] = None
    dtype: str = "bf16"
    name: str = "para"
    backend: str = field(default="pegasus", init=False)
    def _sent_tokenize(self, text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(" ".join((text or "").split()))
        except Exception:
            parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
            return [p for p in parts if p]

    def _load(self):
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer

        cd = _resolve_hf_cache_dir(self.cache_dir)  # ✅ 统一解析：None => HF 全局用户 cache

        # tokenizer cache
        tok_key = (self.model_name, cd)
        if tok_key not in _HF_TOK_CACHE:
            _HF_TOK_CACHE[tok_key] = PegasusTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cd,  # cd=None -> 默认 ~/.cache/huggingface（或 HF_HOME 指定）
            )
        tokenizer = _HF_TOK_CACHE[tok_key]

        # model cache
        model_key = (self.model_name, self.device, self.dtype, cd)
        if model_key not in _HF_MODEL_CACHE:
            torch_dtype = _get_torch_dtype(self.dtype)
            model = PegasusForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=cd,
                torch_dtype=torch_dtype,
            ).to(self.device)
            model.eval()
            _HF_MODEL_CACHE[model_key] = model

        return _HF_MODEL_CACHE[model_key], tokenizer

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = text or ""
        if not text.strip():
            return []
        model, tok = self._load()

        sents = self._sent_tokenize(text)
        if not sents:
            return []

        outs: List[str] = []
        for _ in range(self.n_variants):
            # paraphrase all sentences, then join
            para_sents: List[str] = []
            # batched generation
            for i in range(0, len(sents), int(self.sent_batch_size)):
                batch_sents = sents[i : i + int(self.sent_batch_size)]
                batch = tok(
                    batch_sents,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    max_length=int(self.max_length),
                ).to(self.device)

                import torch
                with torch.inference_mode():
                    out_ids = model.generate(
                        **batch,
                        max_length=int(self.max_length),
                        temperature=float(self.temperature),
                        top_p=float(self.top_p),
                        no_repeat_ngram_size=int(self.no_repeat_ngram_size),
                        do_sample=True,
                    )
                para_sents.extend(tok.batch_decode(out_ids, skip_special_tokens=True))

            out = " ".join([x.strip() for x in para_sents if x.strip()]).strip()
            if out and out != text:
                outs.append(out)

        return _dedup_strs(outs)

# -----------------------
# DIPPER paraphrase (your heavy model wrapper)
# -----------------------
class _DipperParaphraser:
    def __init__(self, model: str, device: str = "cuda", dtype: str = "bf16") -> None:
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        torch_dtype = _get_torch_dtype(dtype)
        self.model = self.model.to(torch_dtype).to(device)
        self.model.eval()
        self.device = device

    def paraphrase(
        self,
        input_text: str,
        *,
        lex_diversity: int,
        order_diversity: int,
        prefix: str = "",
        sent_interval: int = 3,
        do_sample: bool = True,
        top_p: float = 0.96,
        top_k: Optional[int] = None,
        max_length: int = 512,
    ) -> str:
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(" ".join(input_text.split()))
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(input_text.split()))
            sentences = [s for s in sentences if s]

        assert lex_diversity in [0, 20, 40, 60, 80, 100]
        assert order_diversity in [0, 20, 40, 60, 80, 100]

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        prefix = " ".join((prefix or "").replace("\n", " ").split())
        output_text = ""

        import torch
        for sent_idx in range(0, len(sentences), sent_interval):
            curr = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr} </sent>"

            enc = self.tokenizer([final_input_text], return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}

            gen_kwargs = dict(
                do_sample=bool(do_sample),
                top_p=float(top_p),
                max_length=int(max_length),
            )
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)

            with torch.inference_mode():
                out_ids = self.model.generate(**enc, **gen_kwargs)

            out = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
            prefix = (prefix + " " + out).strip()
            output_text = (output_text + " " + out).strip()

        return output_text.strip()

@dataclass
class DipperParaphraseAttack:
    model_name: str = "kalpeshk2011/dipper-paraphraser-xxl"
    lex_diversity: int = 60
    order_diversity: int = 60
    n_variants: int = 1
    device: str = "cuda"
    dtype: str = "bf16"
    name: str = "para"

    _dp: Optional[_DipperParaphraser] = field(default=None, init=False, repr=False)

    def _load(self) -> _DipperParaphraser:
        if self._dp is None:
            self._dp = _DipperParaphraser(model=self.model_name, device=self.device, dtype=self.dtype)
        return self._dp

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = text or ""
        if not text.strip():
            return []
        dp = self._load()

        outs: List[str] = []
        for _ in range(self.n_variants):
            out = dp.paraphrase(
                text,
                lex_diversity=int(self.lex_diversity),
                order_diversity=int(self.order_diversity),
                prefix="",
                do_sample=True,
                top_p=0.96,
                top_k=None,
                max_length=512,
            )
            if out and out != text:
                outs.append(out)
        return _dedup_strs(outs)

# -----------------------
# Local HF prompted paraphrase (HF model + prompt)
# -----------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a ruthless paraphraser. You will aggressively and completely rewrite the user's text "
    "while preserving the core meaning and factual content. Keep the SAME LANGUAGE as the input. "
    "Do NOT add disclaimers, meta-comments, safety notes, citations, or new facts. "
    "Do NOT shorten excessively; keep roughly similar length unless instructed otherwise. "
    "Produce ONLY the rewritten text as plain text, with no surrounding quotes or markers."
)
DEFAULT_USER_INSTRUCTION = (
    "Paraphrase the following text aggressively (completely rewrite) while preserving its core meaning.\n"
    "Keep the SAME LANGUAGE as the input. Output only the rewritten text.\n"
    "Text:\n"
)

@dataclass
class HFPromptParaphraseAttack:
    """
    使用本地 HF 模型做“提示词释义”：
      - 支持 AutoModelForCausalLM 或 AutoModelForSeq2SeqLM
      - 对于 CausalLM：使用简单的指令拼接
    """
    model_name_or_path: str
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    do_sample: bool = True
    n_variants: int = 1
    device: str = "cuda"
    cache_dir: Optional[str] = None
    dtype: str = "bf16"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    name: str = "para"

    _model: Any = field(default=None, init=False, repr=False)
    _tok: Any = field(default=None, init=False, repr=False)
    _is_seq2seq: bool = field(default=False, init=False, repr=False)

    def _load(self):
        import torch
        import transformers

        if self._model is not None and self._tok is not None:
            return

        # tokenizer
        self._tok = transformers.AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            use_fast=True,
        )

        # try seq2seq first, else causal
        torch_dtype = _get_torch_dtype(self.dtype)
        try:
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
            )
            self._is_seq2seq = True
        except Exception:
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
            )
            self._is_seq2seq = False

        self._model.to(self.device)
        self._model.eval()

        # some causal tokenizers need pad token
        if getattr(self._tok, "pad_token_id", None) is None and getattr(self._tok, "eos_token_id", None) is not None:
            self._tok.pad_token = self._tok.eos_token

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        self._load()

        outs: List[str] = []
        import torch
        for _ in range(self.n_variants):
            prompt = f"{self.system_prompt}\n\n{DEFAULT_USER_INSTRUCTION}{text}\n"
            if self._is_seq2seq:
                enc = self._tok([prompt], return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.inference_mode():
                    out_ids = self._model.generate(
                        **enc,
                        max_new_tokens=int(self.max_new_tokens),
                        do_sample=bool(self.do_sample),
                        temperature=float(self.temperature),
                        top_p=float(self.top_p),
                    )
                out = self._tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            else:
                enc = self._tok([prompt], return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.inference_mode():
                    out_ids = self._model.generate(
                        **enc,
                        max_new_tokens=int(self.max_new_tokens),
                        do_sample=bool(self.do_sample),
                        temperature=float(self.temperature),
                        top_p=float(self.top_p),
                        pad_token_id=int(self._tok.pad_token_id),
                    )
                full = self._tok.batch_decode(out_ids, skip_special_tokens=True)[0]
                out = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

            if out and out != text:
                outs.append(out)

        return _dedup_strs(outs)

# -----------------------
# External API prompted paraphrase (OpenAI-compatible)
# -----------------------
@dataclass
class APIPromptParaphraseAttack:
    """
    外部 API 强释义：
      - OpenAI python>=1.x: OpenAI(api_key=..., base_url=...)
      - DeepSeek/OpenAI 兼容
    """
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""  # allow env injection
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.9
    max_tokens: int = 1024
    timeout: float = 60.0
    retries: int = 5
    sleep: float = 0.2
    n_variants: int = 1
    name: str = "para"

    _client: Any = field(default=None, init=False, repr=False)

    def _client_init(self):
        if self._client is not None:
            return
        key = self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not key:
            raise RuntimeError("APIPromptParaphraseAttack: API key empty. Set api_key or env DEEPSEEK_API_KEY/OPENAI_API_KEY.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("APIPromptParaphraseAttack requires `pip install openai` (OpenAI python>=1.x).") from e
        self._client = OpenAI(api_key=key, base_url=self.base_url)

    def _once(self, text: str) -> str:
        self._client_init()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{DEFAULT_USER_INSTRUCTION}\n{text}"},
            ],
            stream=False,
            temperature=float(self.temperature),
            max_tokens=int(self.max_tokens),
            timeout=float(self.timeout),
        )
        return (resp.choices[0].message.content or "").strip()

    def _with_retry(self, text: str) -> str:
        # match your script’s behavior
        self._client_init()
        delay = 1.0
        backoff = 1.5
        last_err: Optional[Exception] = None

        # import errors only when openai exists
        try:
            from openai import APIError, RateLimitError, APITimeoutError, InternalServerError  # type: ignore
            retryable = (RateLimitError, APITimeoutError, InternalServerError, APIError)
        except Exception:
            retryable = (Exception,)

        for _ in range(int(self.retries)):
            try:
                return self._once(text)
            except retryable as e:  # type: ignore
                last_err = e
                time.sleep(delay)
                delay *= backoff
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= backoff

        # fallback: keep original
        return text

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        outs: List[str] = []
        for _ in range(self.n_variants):
            out = self._with_retry(text)
            if self.sleep and self.sleep > 0:
                time.sleep(float(self.sleep))
            if out and out != text:
                outs.append(out)
        return _dedup_strs(outs)

# alias: chatgpt_para (for compatibility naming)
@dataclass
class ChatGPTParaphraseAttack(APIPromptParaphraseAttack):
    name: str = "para"

# -----------------------
# word_subst_modelbase (keep original behavior by delegating to your existing module)
# -----------------------
@dataclass
class WordSubstModelBaseAttack:
    """
    保留你原工程实现：尝试 import `attacks.word_subst_modelbase.generate_attack_with_lm_replacement`
    并调用其逻辑完成替换。

    你需要提供 replacement_model_name_or_path（例如本地 Llama 权重目录）。
    """
    replacement_model_name_or_path: str
    pct_words_masked: float = 0.2
    num_replacement_retry: int = 3
    n_variants: int = 1
    device: str = "cuda"
    dtype: str = "bf16"
    cache_dir: Optional[str] = None
    name: str = "syno"

    _model: Any = field(default=None, init=False, repr=False)
    _tok: Any = field(default=None, init=False, repr=False)

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if self._model is not None and self._tok is not None:
            return
        torch_dtype = _get_torch_dtype(self.dtype)
        cd = self.cache_dir
        self._tok = AutoTokenizer.from_pretrained(self.replacement_model_name_or_path, cache_dir=cd, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.replacement_model_name_or_path, cache_dir=cd, torch_dtype=torch_dtype
        ).to(self.device)
        self._model.eval()

    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            from attacks.word_subst_modelbase import generate_attack_with_lm_replacement  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "word_subst_modelbase requires your original implementation: "
                "`attacks.word_subst_modelbase.generate_attack_with_lm_replacement`. "
                "Please ensure it exists in PYTHONPATH."
            ) from e

        self._load()

        class _Args:
            # mimic your expected args fields
            test_ratio = float(self.pct_words_masked)
            num_replacement_retry = int(self.num_replacement_retry)
            attack_method = "llama_replacement"

        outs: List[str] = []
        for _ in range(self.n_variants):
            attacked, _subword_num = generate_attack_with_lm_replacement([text], _Args, self._model, self._tok)
            if attacked and isinstance(attacked, list) and attacked[0].strip() and attacked[0] != text:
                outs.append(attacked[0].strip())
        return _dedup_strs(outs)

# -----------------------
# Back-translation (Helsinki-NLP / MarianMT)
# -----------------------
def _normalize_lang(lang: str) -> str:
    if not lang:
        return ""
    lang = lang.strip().lower()
    # common normalizations
    if lang in ("zh-cn", "zh_cn", "zh-hans", "zh_hans"):
        return "zh"
    if lang in ("zh-tw", "zh_tw", "zh-hant", "zh_hant"):
        return "zh"
    return lang

def _guess_src_lang(meta: Optional[Dict[str, Any]]) -> str:
    if meta and isinstance(meta.get("lang", None), str) and meta["lang"].strip():
        return _normalize_lang(meta["lang"])
    # fallback: try langdetect if available
    try:
        from langdetect import detect  # type: ignore
        # NOTE: langdetect wants enough text; if too short, may fail
        return _normalize_lang(detect(meta.get("text", ""))) if meta and isinstance(meta.get("text", ""), str) else "en"
    except Exception:
        return "en"

@dataclass
class BackTranslationAttack:
    """
    back_translate:
      - source language: src_lang (None => infer from meta['lang'], else fallback 'en')
      - intermediate: pivot_lang (required)
      - rounds: n_rounds  (each round does src->pivot->src)
    """
    pivot_lang: str = "auto"      # ✅ 默认 auto
    pivot_for_en: str = "de"      # ✅ src=en -> de
    pivot_for_non_en: str = "en"  # ✅ src!=en -> en
    src_lang: Optional[str] = None
    n_rounds: int = 1
    num_beams: int = 5
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_length: int = 512
    n_variants: int = 1
    device: str = "cuda"
    cache_dir: Optional[str] = None
    dtype: str = "bf16"
    name: str = "back_trans"

    _fwd: Any = field(default=None, init=False, repr=False)
    _bwd: Any = field(default=None, init=False, repr=False)
    _tok_fwd: Any = field(default=None, init=False, repr=False)
    _tok_bwd: Any = field(default=None, init=False, repr=False)
    _pair: Tuple[str, str] = field(default=("", ""), init=False, repr=False)

    def _make_model_name(self, a: str, b: str) -> str:
        return f"Helsinki-NLP/opus-mt-{a}-{b}"

    def _load_pair(self, src: str, pivot: str):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        src = _normalize_lang(src)
        pivot = _normalize_lang(pivot)
        pair = (src, pivot)
        if self._pair == pair and self._fwd is not None and self._bwd is not None:
            return

        torch_dtype = _get_torch_dtype(self.dtype)

        fwd_name = self._make_model_name(src, pivot)
        bwd_name = self._make_model_name(pivot, src)

        cd = _resolve_hf_cache_dir(self.cache_dir)  # ✅ None => 用全局 HF cache

        tok_fwd_key = (fwd_name, cd)
        tok_bwd_key = (bwd_name, cd)
        if tok_fwd_key not in _HF_TOK_CACHE:
            _HF_TOK_CACHE[tok_fwd_key] = MarianTokenizer.from_pretrained(fwd_name, cache_dir=cd)
        if tok_bwd_key not in _HF_TOK_CACHE:
            _HF_TOK_CACHE[tok_bwd_key] = MarianTokenizer.from_pretrained(bwd_name, cache_dir=cd)

        self._tok_fwd = _HF_TOK_CACHE[tok_fwd_key]
        self._tok_bwd = _HF_TOK_CACHE[tok_bwd_key]

        fwd_key = (fwd_name, self.device, self.dtype, cd)
        bwd_key = (bwd_name, self.device, self.dtype, cd)
        if fwd_key not in _HF_MODEL_CACHE:
            _HF_MODEL_CACHE[fwd_key] = MarianMTModel.from_pretrained(
                fwd_name, cache_dir=cd, torch_dtype=torch_dtype
            ).to(self.device).eval()
        if bwd_key not in _HF_MODEL_CACHE:
            _HF_MODEL_CACHE[bwd_key] = MarianMTModel.from_pretrained(
                bwd_name, cache_dir=cd, torch_dtype=torch_dtype
            ).to(self.device).eval()

        self._fwd = _HF_MODEL_CACHE[fwd_key]
        self._bwd = _HF_MODEL_CACHE[bwd_key]
        self._pair = pair

    def _translate(self, model, tok, text: str) -> str:
        import torch
        enc = tok([text], return_tensors="pt", truncation=True, padding=True, max_length=int(self.max_length)).to(self.device)
        gen_kwargs = dict(
            max_length=int(self.max_length),
            num_beams=int(self.num_beams),
            do_sample=bool(self.do_sample),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
        )
        with torch.inference_mode():
            out_ids = model.generate(**enc, **gen_kwargs)
        return tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    
    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        def _cyrillic_ratio(s: str) -> float:
            if not s:
                return 0.0
            cyr = sum(1 for ch in s if 0x0400 <= ord(ch) <= 0x04FF)
            letters = sum(1 for ch in s if ch.isalpha())
            return (cyr / max(1, letters))
        
        text = (text or "").strip()
        if not text:
            return []

        src = _normalize_lang(self.src_lang) if self.src_lang else _guess_src_lang(meta)

        # ✅ 鲁棒保护：避免“英文被误判成 ru”
        # 你说“文本中没有俄语但识别为俄语”，最常见就是 langdetect/langid 在短文本/噪声文本上误报。
        if src == "ru" and _cyrillic_ratio(text) < 0.05:
            src = "en"

        # ✅ auto pivot：英文->德语；非英文->英语
        pivot_cfg = (self.pivot_lang or "auto").strip().lower()
        if pivot_cfg in ("auto", "default"):
            pivot = (self.pivot_for_en if src == "en" else self.pivot_for_non_en)
        else:
            pivot = pivot_cfg

        pivot = _normalize_lang(pivot)

        # （可选但很实用）避免 src==pivot 这种无意义 back-translation
        if pivot == src:
            pivot = "de" if src == "en" else "en"

        self._load_pair(src, pivot)

        outs: List[str] = []
        for _ in range(self.n_variants):
            cur = text
            for _r in range(int(max(1, self.n_rounds))):
                mid = self._translate(self._fwd, self._tok_fwd, cur)
                back = self._translate(self._bwd, self._tok_bwd, mid)
                cur = back
            if cur and cur != text:
                outs.append(cur)
        return _dedup_strs(outs)
