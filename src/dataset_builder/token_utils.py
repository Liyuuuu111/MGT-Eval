from __future__ import annotations
from typing import Optional, Protocol, Sequence, Any, Dict


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False, **kw: Any) -> Sequence[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True, **kw: Any) -> str: ...


def take_first_k_tokens(
    text: str,
    k: int,
    tokenizer: Optional[TokenizerLike],
    strategy: str = "auto",
) -> str:
    """
    返回“前 k 个token”对应的文本前缀。
    - 若 tokenizer 可用：用 tokenizer encode/decode 精确按 token 截断
    - 若不可用：按 whitespace token 截断（退化）
    """
    if not isinstance(text, str):
        text = str(text)

    if k is None or k <= 0:
        return text

    if tokenizer is not None and strategy in ("auto",) or strategy.startswith("hf:"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= k:
                return text
            prefix_ids = ids[:k]
            pref = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            return pref
        except Exception:
            # fall back
            pass

    # whitespace fallback
    toks = text.split()
    if len(toks) <= k:
        return text
    return " ".join(toks[:k])
