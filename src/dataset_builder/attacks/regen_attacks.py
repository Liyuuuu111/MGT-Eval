from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ..config import GenConfig


@dataclass
class NoRegen:
    name: str = "none_regen"

    def propose(self, base: GenConfig) -> List[GenConfig]:
        return []


@dataclass
class TemperatureSweep:
    temps: List[float]
    name: str = "temp_sweep"

    def propose(self, base: GenConfig) -> List[GenConfig]:
        outs: List[GenConfig] = []
        for t in self.temps:
            if float(t) == float(base.temperature):
                continue
            g = GenConfig(**base.to_dict())
            g.temperature = float(t)
            g.do_sample = True
            outs.append(g)
        return outs


@dataclass
class TopPSweep:
    top_ps: List[float]
    name: str = "top_p_sweep"

    def propose(self, base: GenConfig) -> List[GenConfig]:
        outs: List[GenConfig] = []
        for p in self.top_ps:
            if float(p) == float(base.top_p):
                continue
            g = GenConfig(**base.to_dict())
            g.top_p = float(p)
            g.do_sample = True
            outs.append(g)
        return outs


@dataclass
class GreedyVsSample:
    """
    同一 prompt，生成一份 greedy（do_sample=False）和一份更随机的 sample（可改温度）
    """
    sample_temperature: float = 1.0
    name: str = "greedy_vs_sample"

    def propose(self, base: GenConfig) -> List[GenConfig]:
        outs: List[GenConfig] = []

        greedy = GenConfig(**base.to_dict())
        greedy.do_sample = False
        greedy.temperature = 1.0
        outs.append(greedy)

        sample = GenConfig(**base.to_dict())
        sample.do_sample = True
        sample.temperature = float(self.sample_temperature)
        outs.append(sample)

        # ： base
        uniq: List[GenConfig] = []
        seen = set()
        for g in outs:
            key = (g.do_sample, round(g.temperature, 4), round(g.top_p, 4), g.top_k, g.num_beams)
            if key not in seen:
                seen.add(key)
                if not (g.to_dict() == base.to_dict()):
                    uniq.append(g)
        return uniq
