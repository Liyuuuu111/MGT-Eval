"""Detector metadata loading service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MetadataService:
    """Loads detector metadata for frontend presentation."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.metadata_file = self.project_root / "src" / "detector_metadata" / "detectors.json"

    def get_detector_metadata(self) -> List[Dict[str, Any]]:
        from_file = self._load_from_file()
        if from_file:
            return from_file
        return self._load_from_cli_helpers()

    def _load_from_file(self) -> List[Dict[str, Any]]:
        if not self.metadata_file.exists():
            return []
        try:
            with self.metadata_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return []

        rows: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rows.append(item)
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if not isinstance(value, dict):
                    continue
                item = {"key": key, **value}
                rows.append(item)
        rows.sort(key=lambda x: str(x.get("key", "")).lower())
        return rows

    def _load_from_cli_helpers(self) -> List[Dict[str, Any]]:
        try:
            from src.cli_helpers import DETECTOR_METADATA  # type: ignore
        except Exception:
            return []

        rows: List[Dict[str, Any]] = []
        if isinstance(DETECTOR_METADATA, dict):
            for key, value in DETECTOR_METADATA.items():
                if not isinstance(value, dict):
                    continue
                rows.append(
                    {
                        "key": key,
                        "name": value.get("name"),
                        "type": value.get("type"),
                        "description": value.get("description"),
                        "paper": value.get("paper"),
                        "authors": value.get("authors"),
                        "link": value.get("link"),
                    }
                )
        rows.sort(key=lambda x: str(x.get("key", "")).lower())
        return rows


metadata_service = MetadataService()

