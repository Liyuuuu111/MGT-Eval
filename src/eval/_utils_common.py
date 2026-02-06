from __future__ import annotations

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Iterable


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _auto_run_dir(out_dir: Optional[str], detector_display_name: str) -> Path:
    """
    规则：
      - out_dir=None: results/runs_{detector}_{timestamp}/
      - out_dir!=None: 若最后一段目录名不含 \\d{8}-\\d{6}，自动追加 _{timestamp}
    """
    ts = _timestamp()
    if out_dir:
        p = Path(out_dir)
        tail = p.name
        if not re.search(r"\d{8}-\d{6}$", tail):
            p = p.with_name(f"{tail}_{ts}")
    else:
        p = Path(f"results/runs_{detector_display_name}_{ts}")

    p.mkdir(parents=True, exist_ok=True)
    (p / "metrics/curves").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    (p / "artifacts").mkdir(parents=True, exist_ok=True)
    return p


def _auto_prefix(
    dataset: Union[str, Iterable[Dict[str, Any]]],
    detector_display_name: str,
    model_name: Optional[str] = None,
) -> str:
    def _basename(p: str) -> str:
        p = p.rstrip("/").rstrip("\\")
        return os.path.basename(p) if p else "data"

    if isinstance(dataset, str):
        d = _basename(dataset.lower())
        for suf in (".jsonl", ".json"):
            if d.endswith(suf):
                d = d[: -len(suf)]
        d = d or "data"
    else:
        d = "data"

    if model_name and model_name != detector_display_name:
        return f"{d}__{detector_display_name}__{model_name}"
    return f"{d}__{detector_display_name}"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _norm_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None
