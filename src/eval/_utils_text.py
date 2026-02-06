from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _word_count(text: str) -> int:
    """
    Language-agnostic word count.
    - Prefer third-party `regex` for Unicode-aware counting.
    - Fallback: whitespace tokenization.
    """
    s = "" if text is None else str(text)
    try:
        import regex as re_u
        patt = re_u.compile(
            r"(?:\p{Han})|(?:\p{Hiragana}+)|(?:\p{Katakana}+)|(?:\p{Hangul}+)"
            r"|(?:[A-Za-z]+(?:[-'][A-Za-z]+)*)|(?:\d+)",
            re_u.UNICODE
        )
        return len(patt.findall(s))
    except Exception:
        return len([t for t in s.strip().split() if t])


def _extract_builder_samples(rec: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    for key in ("sample", "samples", "pair", "pairs"):
        v = rec.get(key, None)
        if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            return v, key
    return None, None


def _extract_text(obj: Dict[str, Any]) -> str:
    for k in ("text", "content", "response", "output", "generation", "gen", "final_text"):
        if k in obj and obj.get(k) is not None:
            return str(obj.get(k) or "").strip()
    return ""


def _extract_role(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("role", "source", "type"):
        if k in obj and obj.get(k) is not None:
            s = str(obj.get(k)).strip()
            return s if s else None
    return None


def _looks_like_builder_record(rec: Dict[str, Any]) -> bool:
    samples, _ = _extract_builder_samples(rec)
    if not (isinstance(samples, list) and len(samples) >= 1 and isinstance(samples[0], dict)):
        return False
    return bool(_extract_text(samples[0]))


def _infer_label_from_role(role: Any, default: int = 1) -> int:
    r = str(role or "").strip().lower()
    if r == "machine":
        return 1
    if r == "human":
        return 0
    return int(default)
