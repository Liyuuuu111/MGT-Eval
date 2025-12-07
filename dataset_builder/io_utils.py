from __future__ import annotations
import json
from typing import Any, Dict, Iterable, Optional
from pathlib import Path


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], append: bool = False) -> None:
    ensure_parent(path)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlWriter:
    def __init__(self, path: str, append: bool = False) -> None:
        ensure_parent(path)
        self.path = path
        self.f = open(path, "a" if append else "w", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()
