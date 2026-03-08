from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, Optional
from pathlib import Path


def normalize_path(path: str) -> str:
    """
    Normalize file path to ensure only '/' is used as path separator.
    This prevents '.' from being misinterpreted as a directory separator.

    Args:
        path: Input path string

    Returns:
        Normalized path with only '/' as separator
    """
    # Convert backslashes to forward slashes
    normalized = path.replace('\\', '/')

    # Use Path to handle the path correctly
    # Path will only treat '/' (and os.sep) as separators, not '.'
    p = Path(normalized)

    # Return the normalized string representation
    return str(p)


def ensure_parent(path: str) -> None:
    """
    Ensure parent directory exists for the given file path.
    Only '/' and os.sep are recognized as path separators, not '.'.

    Args:
        path: File path for which to create parent directory
    """
    # Normalize and get parent directory
    normalized_path = normalize_path(path)
    parent_dir = Path(normalized_path).parent

    # Create parent directory if it doesn't exist
    if parent_dir and str(parent_dir) != '.':
        parent_dir.mkdir(parents=True, exist_ok=True)


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
