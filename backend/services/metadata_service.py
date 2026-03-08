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

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            return s or None
        if isinstance(value, dict):
            for key in ("en", "zh", "text", "default"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in value.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None
        return str(value).strip() or None

    @staticmethod
    def _normalize_i18n_text(value: Any) -> tuple[str | None, str | None]:
        """Extract English/Chinese localized text when available."""
        if isinstance(value, dict):
            en = value.get("en")
            zh = value.get("zh")
            en_s = str(en).strip() if isinstance(en, str) and en.strip() else None
            zh_s = str(zh).strip() if isinstance(zh, str) and zh.strip() else None
            return en_s, zh_s
        s = MetadataService._normalize_text(value)
        return s, None

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        desc_en, desc_zh = self._normalize_i18n_text(item.get("description"))
        description = desc_en or desc_zh
        out: Dict[str, Any] = {
            "key": str(item.get("key", "")).strip(),
            "name": self._normalize_text(item.get("name")),
            "type": self._normalize_text(item.get("type")),
            "description": description,
            "paper": self._normalize_text(item.get("paper")),
            "authors": self._normalize_text(item.get("authors")),
            "link": self._normalize_text(item.get("link")),
        }
        if desc_en is not None:
            out["description_en"] = desc_en
        if desc_zh is not None:
            out["description_zh"] = desc_zh
        # keep optional non-response-model fields if present (frontend may use these via raw fetches)
        venue = self._normalize_text(item.get("venue"))
        if venue is not None:
            out["venue"] = venue
        return out

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
                    normalized = self._normalize_item(item)
                    if normalized.get("key"):
                        rows.append(normalized)
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if not isinstance(value, dict):
                    continue
                item = {"key": key, **value}
                normalized = self._normalize_item(item)
                if normalized.get("key"):
                    rows.append(normalized)
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
                normalized = self._normalize_item(
                    {
                        "key": key,
                        "name": value.get("name"),
                        "type": value.get("type"),
                        "description": value.get("description"),
                        "paper": value.get("paper"),
                        "authors": value.get("authors"),
                        "link": value.get("link"),
                        "venue": value.get("venue"),
                    }
                )
                if normalized.get("key"):
                    rows.append(normalized)
        rows.sort(key=lambda x: str(x.get("key", "")).lower())
        return rows


metadata_service = MetadataService()
