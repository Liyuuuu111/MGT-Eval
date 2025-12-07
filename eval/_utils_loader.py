from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..data_utils.load import load_dataset_unified
from ._utils_common import _norm_id, _read_jsonl
from ._utils_text import (
    _extract_builder_samples,
    _extract_text,
    _extract_role,
    _infer_label_from_role,
    _looks_like_builder_record,
)


def _load_examples_auto(
    dataset_spec: Union[str, Iterable[Dict[str, Any]]],
    *,
    sample_k: Optional[int],
    sample_seed: int,
    group_cols: Optional[Sequence[str]],
    builder_view: str = "flat",  # "flat" | "pre" | "post"
) -> List[Dict[str, Any]]:
    if not isinstance(dataset_spec, str):
        exs, _ = load_dataset_unified(
            dataset=dataset_spec,
            sample_k=sample_k,
            sample_seed=sample_seed,
            group_cols=group_cols,
        )
        return exs

    p = str(dataset_spec)
    if (p.endswith(".jsonl") or p.endswith(".json")) and os.path.exists(p):
        # detect builder quick
        try:
            with open(p, "r", encoding="utf-8") as f:
                head = []
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        head.append(json.loads(line))
            is_builder = any(isinstance(r, dict) and _looks_like_builder_record(r) for r in head)
        except Exception:
            is_builder = False

        if is_builder and builder_view in ("pre", "post"):
            recs = _read_jsonl(p)
            out: List[Dict[str, Any]] = []
            for i, r in enumerate(recs):
                if not isinstance(r, dict) or (not _looks_like_builder_record(r)):
                    continue
                samples, _k = _extract_builder_samples(r)
                if not samples:
                    continue
                if builder_view == "post" and len(samples) < 2:
                    continue

                obj = samples[0] if builder_view == "pre" else samples[-1]
                text = _extract_text(obj)
                if not text:
                    continue

                rid = (
                    _norm_id(r.get("id")) or _norm_id(r.get("record_id")) or _norm_id(r.get("qid"))
                    or _norm_id(r.get("question_id")) or str(i)
                )

                if "label" in r and isinstance(r.get("label"), (int, float, str)):
                    try:
                        lb_default = int(r.get("label"))
                    except Exception:
                        lb_default = 1
                else:
                    lb_default = 1

                role = _extract_role(obj)
                label = _infer_label_from_role(role, default=lb_default)

                ex: Dict[str, Any] = {"id": rid, "text": text, "label": int(label)}

                for k in ("lang", "source", "sub_source", "model"):
                    if k in r and r.get(k) is not None:
                        ex[k] = r.get(k)
                    elif isinstance(obj, dict) and (k in obj) and (obj.get(k) is not None):
                        ex[k] = obj.get(k)

                out.append(ex)

            return out

    # fallback to unified loader
    exs, _ = load_dataset_unified(
        dataset=dataset_spec,
        sample_k=sample_k,
        sample_seed=sample_seed,
        group_cols=group_cols,
    )
    return exs
