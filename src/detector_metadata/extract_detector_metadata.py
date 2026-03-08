#!/usr/bin/env python3
"""Extract detector metadata for frontend display.

This script keeps CLI behavior unchanged. It only materializes metadata into
`src/detector_metadata/detectors.json` for API/frontend consumption.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_helpers import DETECTOR_METADATA  # noqa: E402


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _is_empty_value(value: Optional[str]) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"", "n/a", "none", "null", "unknown"}


def _pretty_name(key: str) -> str:
    return " ".join(part[:1].upper() + part[1:] for part in key.replace("-", "_").split("_"))


def _load_detector_keys(examples_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (template_key, detector_id_from_yaml_or_template_key)."""
    rows: List[Tuple[str, str]] = []
    for path in sorted(examples_dir.glob("*.yaml")):
        template_key = path.stem
        detector_id = template_key
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            detector_from_yaml = payload.get("detector")
            if detector_from_yaml:
                detector_id = str(detector_from_yaml).strip()
        except Exception:
            detector_id = template_key
        rows.append((template_key, detector_id))
    return rows


def _extract_str_node(node: ast.AST, module_consts: Dict[str, str]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    if isinstance(node, ast.Name):
        return _normalize_text(module_consts.get(node.id))
    if isinstance(node, ast.JoinedStr):
        chunks: List[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                chunks.append(part.value)
        text = "".join(chunks).strip()
        return text or None
    return None


def _collect_module_string_constants(tree: ast.Module) -> Dict[str, str]:
    constants: Dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = _extract_str_node(node.value, constants)
        if text:
            constants[target.id] = text
    return constants


def _registered_names(class_node: ast.ClassDef) -> List[str]:
    names: List[str] = []
    for dec in class_node.decorator_list:
        if isinstance(dec, ast.Call):
            func_name = None
            if isinstance(dec.func, ast.Name):
                func_name = dec.func.id
            elif isinstance(dec.func, ast.Attribute):
                func_name = dec.func.attr
            # Accept register, register_train, register_detect, etc.
            if not func_name or not func_name.startswith("register"):
                continue
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                names.append(str(dec.args[0].value).strip().lower())
    return names


def _collect_detector_citations(detectors_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    citations: Dict[str, Dict[str, Optional[str]]] = {}
    for py_file in detectors_root.rglob("*.py"):
        if py_file.name in {"__init__.py", "base.py", "registry.py"}:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception:
            continue

        module_consts = _collect_module_string_constants(tree)
        module_fallback = {
            "paper": _normalize_text(module_consts.get("CITATION_TITLE")),
            "authors": _normalize_text(module_consts.get("CITATION_AUTHORS")),
            "link": _normalize_text(module_consts.get("CITATION_LINK")),
        }

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            reg_names = _registered_names(node)
            if not reg_names:
                continue

            class_values: Dict[str, Optional[str]] = {"paper": None, "authors": None, "link": None}
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                    continue
                field_name = item.targets[0].id
                if field_name == "CITATION_TITLE":
                    class_values["paper"] = _normalize_text(_extract_str_node(item.value, module_consts))
                elif field_name == "CITATION_AUTHORS":
                    class_values["authors"] = _normalize_text(_extract_str_node(item.value, module_consts))
                elif field_name == "CITATION_LINK":
                    class_values["link"] = _normalize_text(_extract_str_node(item.value, module_consts))

            for reg_name in reg_names:
                existing = citations.get(reg_name, {"paper": None, "authors": None, "link": None})
                for field in ("paper", "authors", "link"):
                    if _is_empty_value(existing.get(field)):
                        existing[field] = class_values.get(field) or module_fallback.get(field)
                citations[reg_name] = existing

    return citations


def _build_detector_row(
    template_key: str,
    detector_id: str,
    citation_map: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    fallback = DETECTOR_METADATA.get(template_key, {})
    if not fallback and detector_id != template_key:
        fallback = DETECTOR_METADATA.get(detector_id, {})

    citation = citation_map.get(str(detector_id).lower(), {"paper": None, "authors": None, "link": None})

    name = _normalize_text(fallback.get("name")) or _pretty_name(template_key)
    dtype = _normalize_text(fallback.get("type")) or "Unknown"
    description = _normalize_text(fallback.get("description")) or "No description available."

    # Prioritize citation from source code over fallback
    paper = citation.get("paper")
    if _is_empty_value(paper):
        paper = _normalize_text(fallback.get("paper"))

    authors = citation.get("authors")
    if _is_empty_value(authors):
        authors = _normalize_text(fallback.get("authors"))

    # For link, ONLY use citation from source code (fallback never has link field)
    link = citation.get("link")
    if _is_empty_value(link):
        # Try fallback but it's usually not there
        link = _normalize_text(fallback.get("link"))

    return {
        "key": template_key,
        "detector": detector_id,
        "name": name,
        "type": dtype,
        "description": description,
        "paper": paper,
        "authors": authors,
        "link": link,
    }


def extract_metadata(output_path: Path, examples_dir: Path, detectors_root: Path) -> List[Dict[str, Any]]:
    detector_rows = _load_detector_keys(examples_dir)
    citation_map = _collect_detector_citations(detectors_root)
    payload = [
        _build_detector_row(template_key, detector_id, citation_map)
        for template_key, detector_id in detector_rows
    ]
    payload.sort(key=lambda row: str(row.get("key", "")).lower())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract detector metadata to JSON")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "src" / "detector_metadata" / "detectors.json"),
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default=str(PROJECT_ROOT / "examples" / "detect"),
        help="Path to detect example YAML templates",
    )
    parser.add_argument(
        "--detectors-root",
        type=str,
        default=str(PROJECT_ROOT / "src" / "detectors"),
        help="Path to source detector modules",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    examples_dir = Path(args.examples_dir).expanduser().resolve()
    detectors_root = Path(args.detectors_root).expanduser().resolve()

    rows = extract_metadata(
        output_path=output_path,
        examples_dir=examples_dir,
        detectors_root=detectors_root,
    )
    print(f"[detector-metadata] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
