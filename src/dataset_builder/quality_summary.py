# mgt_eval/dataset_builder/quality_summary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import math
from pathlib import Path


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _get_fre(q: Dict[str, Any]) -> Optional[float]:
    # readability.flesch_reading_ease
    r = q.get("readability", None)
    if isinstance(r, dict):
        return _safe_float(r.get("flesch_reading_ease", None))
    return None


def _get_ppl(q: Dict[str, Any]) -> Optional[float]:
    return _safe_float(q.get("ppl", None))


def _get_bert_f1(q: Dict[str, Any]) -> Optional[float]:
    b = q.get("bertscore", None)
    if isinstance(b, dict):
        return _safe_float(b.get("f1", None))
    return None


@dataclass
class RunningQualityStats:
    enable_dppl: bool = True
    enable_drea: bool = True
    enable_bert: bool = True

    # sums / counts over SAMPLES (not records)
    _sum_dppl: float = 0.0
    _cnt_dppl: int = 0

    _sum_dfre: float = 0.0
    _cnt_dfre: int = 0

    _sum_bert: float = 0.0
    _cnt_bert: int = 0

    _records: int = 0
    _samples: int = 0

    def update_from_quality(self, *, original_quality: Dict[str, Any], sample_qualities: List[Dict[str, Any]]) -> None:
        self._records += 1

        orig_ppl = _get_ppl(original_quality) if self.enable_dppl else None
        orig_fre = _get_fre(original_quality) if self.enable_drea else None

        for sq in sample_qualities:
            self._samples += 1

            if self.enable_dppl and orig_ppl is not None:
                sp = _get_ppl(sq)
                if sp is not None:
                    self._sum_dppl += (sp - orig_ppl)         # dPPL = sample - original
                    self._cnt_dppl += 1

            if self.enable_drea and orig_fre is not None:
                sf = _get_fre(sq)
                if sf is not None:
                    self._sum_dfre += (sf - orig_fre)         # dFRE = sample - original
                    self._cnt_dfre += 1

            if self.enable_bert:
                bf = _get_bert_f1(sq)
                if bf is not None:
                    self._sum_bert += bf
                    self._cnt_bert += 1

    def mean_dppl(self) -> Optional[float]:
        if not self.enable_dppl or self._cnt_dppl <= 0:
            return None
        return self._sum_dppl / float(self._cnt_dppl)

    def mean_dfre(self) -> Optional[float]:
        if not self.enable_drea or self._cnt_dfre <= 0:
            return None
        return self._sum_dfre / float(self._cnt_dfre)

    def mean_bert_f1(self) -> Optional[float]:
        if not self.enable_bert or self._cnt_bert <= 0:
            return None
        return self._sum_bert / float(self._cnt_bert)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "sample_average",
            "records_seen": int(self._records),
            "samples_seen": int(self._samples),
            "enable_dppl": bool(self.enable_dppl),
            "enable_drea": bool(self.enable_drea),
            "enable_bert": bool(self.enable_bert),
            "mean_delta_ppl": self.mean_dppl(),
            "mean_delta_flesch_reading_ease": self.mean_dfre(),
            "mean_bertscore_f1": self.mean_bert_f1(),
            "counts": {
                "dppl": int(self._cnt_dppl),
                "dfre": int(self._cnt_dfre),
                "bert_f1": int(self._cnt_bert),
            },
        }

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(p)


def compute_quality_means_from_jsonl(in_jsonl: str, out_json: Optional[str] = None) -> Dict[str, Any]:
    """
    离线从 builder 输出的 jsonl 重新统计均值（以 sample 为粒度）。
    只要 record 内 original[0].quality 与 sample[*].quality 存在即可工作。
    """
    stats = RunningQualityStats(enable_dppl=True, enable_drea=True, enable_bert=True)

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            orig_list = rec.get("original", [])
            samp_list = rec.get("sample", [])
            if not orig_list or not samp_list:
                continue

            oq = orig_list[0].get("quality", {})
            sqs = [s.get("quality", {}) for s in samp_list]
            stats.update_from_quality(original_quality=oq, sample_qualities=sqs)

    d = stats.to_dict()
    if out_json:
        stats.save(out_json)
    return d
