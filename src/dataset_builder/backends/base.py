from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from ..config import GenConfig


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    text: str  # either prompt+completion or completion_only depending on config
    meta: Dict[str, Any]


class LLMBackend(Protocol):
    name: str

    def generate(self, prompt: str, gen: GenConfig, system_prompt: Optional[str] = None) -> GenerationResult: ...
    def get_tokenizer(self) -> Any: ...
    def close(self) -> None: ...
