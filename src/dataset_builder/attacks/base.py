from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from ..config import GenConfig


class TextAttacker(Protocol):
    name: str
    def apply(self, text: str, rng, meta: Optional[Dict[str, Any]] = None) -> List[str]: ...


class RegenAttacker(Protocol):
    name: str
    def propose(self, base: GenConfig) -> List[GenConfig]: ...


@dataclass
class AttackBundle:
    text_attackers: List[TextAttacker]
    regen_attackers: List[RegenAttacker]
