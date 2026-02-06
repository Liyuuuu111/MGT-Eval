"""System Service for GPU and model detection"""

import subprocess
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import time
from datetime import datetime, timezone


class SystemService:
    """Service for system resource detection"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.hf_cache_dir = self._get_hf_cache_dir()
        self.default_calibration_dir = self.project_root / "calibration_results"

    def _get_hf_cache_dir(self) -> Path:
        """Get Hugging Face cache directory"""
        # Check environment variable first
        if "HF_HOME" in os.environ:
            return Path(os.environ["HF_HOME"]) / "hub"
        elif "HUGGINGFACE_HUB_CACHE" in os.environ:
            return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
        elif "TRANSFORMERS_CACHE" in os.environ:
            return Path(os.environ["TRANSFORMERS_CACHE"])
        else:
            # Default location
            return Path.home() / ".cache" / "huggingface" / "hub"

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """
        Detect available GPUs using nvidia-smi

        Returns:
            List of GPU information dictionaries
        """
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpus.append({
                                "id": int(parts[0]),
                                "name": parts[1],
                                "memory_total": f"{parts[2]} MB",
                                "memory_free": f"{parts[3]} MB",
                                "utilization": f"{parts[4]}%",
                                "available": int(parts[4]) < 90  # Consider available if < 90% utilized
                            })
                return gpus

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback: Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "id": i,
                        "name": props.name,
                        "memory_total": f"{props.total_memory // (1024**2)} MB",
                        "memory_free": "N/A",
                        "utilization": "N/A",
                        "available": True
                    })
                return gpus
        except ImportError:
            pass

        # No GPUs detected
        return []

    def detect_local_models(self, custom_dirs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Detect locally cached Hugging Face models

        Args:
            custom_dirs: Additional directories to scan for models

        Returns:
            List of model information dictionaries
        """
        models = []
        dirs_to_scan = [self.hf_cache_dir]
        local_scan_dirs = [Path.cwd()]

        if custom_dirs:
            dirs_to_scan.extend([Path(d) for d in custom_dirs])
            local_scan_dirs.extend([Path(d) for d in custom_dirs])

        for cache_dir in dirs_to_scan:
            if not cache_dir.exists():
                continue

            try:
                # Scan for model directories
                # HF cache structure: models--org--model_name
                for model_dir in cache_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("models--"):
                        # Extract model name from directory
                        # Format: models--organization--model-name
                        parts = model_dir.name.split("--")
                        if len(parts) >= 3:
                            org = parts[1]
                            model_name = "--".join(parts[2:])
                            full_name = f"{org}/{model_name}"

                            # Check if model has snapshots (actually downloaded)
                            snapshots_dir = model_dir / "snapshots"
                            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                                # Get model size
                                try:
                                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                                    size_str = self._format_size(size)
                                except:
                                    size_str = "Unknown"

                                models.append({
                                    "name": full_name,
                                    "path": str(model_dir),
                                    "size": size_str
                                })
            except Exception as e:
                print(f"Error scanning {cache_dir}: {e}")
                continue

        # Scan current working directory (and custom dirs) for local model folders
        for root_dir in local_scan_dirs:
            try:
                models.extend(self._scan_local_model_dirs(root_dir))
            except Exception as e:
                print(f"Error scanning local models in {root_dir}: {e}")
                continue

        # Deduplicate by (name, path)
        seen = set()
        unique = []
        for m in models:
            key = (m.get("name"), m.get("path"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(m)

        # Sort by name
        unique.sort(key=lambda x: x["name"])
        return unique

    def _scan_local_model_dirs(self, root_dir: Path, max_depth: int = 4) -> List[Dict[str, str]]:
        """Scan a local directory tree for model folders."""
        if not root_dir.exists():
            return []

        target_files = {
            "config.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "adapter_config.json",
            "adapter_model.bin",
        }

        models: List[Dict[str, str]] = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            try:
                rel_parts = Path(dirpath).relative_to(root_dir).parts
            except Exception:
                rel_parts = ()
            depth = len(rel_parts)
            if depth > max_depth:
                dirnames[:] = []
                continue

            if not any(fname in target_files for fname in filenames):
                continue

            model_path = Path(dirpath)
            name = self._format_local_model_name(model_path, root_dir)
            try:
                size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                size_str = self._format_size(size)
            except Exception:
                size_str = "Unknown"

            models.append({
                "name": name,
                "path": str(model_path),
                "size": size_str,
            })

        return models

    def _format_local_model_name(self, model_path: Path, root_dir: Path) -> str:
        """Format display name for local model folders."""
        try:
            rel = model_path.relative_to(root_dir)
            if rel.parts:
                return str(rel)
        except Exception:
            pass
        return str(model_path)

    def detect_calibrators(self, custom_dirs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Detect calibrator JSON files.

        Default scan directory: <project_root>/calibration_results
        """
        results: List[Dict[str, str]] = []
        seen_paths = set()

        dirs_to_scan: List[Path] = [self.default_calibration_dir]
        if custom_dirs:
            for raw in custom_dirs:
                if not raw:
                    continue
                p = Path(str(raw).strip()).expanduser()
                if not p.is_absolute():
                    p = (self.project_root / p).resolve()
                dirs_to_scan.append(p)

        unique_dirs: List[Path] = []
        seen_dirs = set()
        for d in dirs_to_scan:
            key = str(d.resolve()) if d.exists() else str(d)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            unique_dirs.append(d)

        for base_dir in unique_dirs:
            if not base_dir.exists():
                continue

            base_display = self._to_display_path(base_dir, is_dir=True)
            if base_display not in seen_paths:
                seen_paths.add(base_display)
                results.append({
                    "name": base_display,
                    "path": base_display,
                    "size": "Directory",
                })

            try:
                for current_root, dirnames, filenames in os.walk(base_dir):
                    rel_depth = len(Path(current_root).relative_to(base_dir).parts)
                    if rel_depth > 6:
                        dirnames[:] = []
                        continue

                    for filename in filenames:
                        if not filename.lower().endswith(".json"):
                            continue
                        full_path = Path(current_root) / filename
                        display_path = self._to_display_path(full_path, is_dir=False)
                        if display_path in seen_paths:
                            continue
                        seen_paths.add(display_path)

                        try:
                            size_str = self._format_size(full_path.stat().st_size)
                        except Exception:
                            size_str = "Unknown"

                        results.append({
                            "name": filename,
                            "path": display_path,
                            "size": size_str,
                        })
            except Exception as e:
                print(f"Error scanning calibrators in {base_dir}: {e}")
                continue

        results.sort(key=lambda item: item["path"])
        return results

    def get_hf_download_status(self) -> Dict[str, Any]:
        """Detect ongoing Hugging Face downloads from cache directory."""
        cache_dir = self.hf_cache_dir
        now = time.time()
        recent_threshold = 60 * 60  # 1 hour

        # Aggregate by logical download key (same file may have both .lock and .incomplete).
        downloads_by_key: Dict[str, Dict[str, Any]] = {}
        progress_suffixes = (".incomplete", ".tmp")
        all_suffixes = progress_suffixes + (".lock",)
        if not cache_dir.exists():
            return {
                "cache_dir": str(cache_dir),
                "active": False,
                "downloads": [],
                "total_downloaded_bytes": 0,
                "total_expected_bytes": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        def register_entry(entry: Path, stat: os.stat_result, suffix: str) -> None:
            base_key = str(entry)
            if base_key.endswith(suffix):
                base_key = base_key[: -len(suffix)]

            total_bytes = self._try_read_expected_size(entry)
            model_name = self._extract_model_name_from_path(entry)
            if not model_name:
                model_name = self._extract_model_name_from_metadata(entry)

            mtime_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            mtime_iso = mtime_dt.isoformat()
            size_bytes = int(stat.st_size) if suffix in progress_suffixes else 0
            is_progress_entry = suffix in progress_suffixes

            if base_key not in downloads_by_key:
                downloads_by_key[base_key] = {
                    "path": str(entry),
                    "size_bytes": size_bytes,
                    "total_bytes": total_bytes,
                    "percent": None,
                    "model": model_name,
                    "mtime": mtime_iso,
                    "_saw_progress_file": is_progress_entry,
                }
                return

            current = downloads_by_key[base_key]
            if is_progress_entry:
                if size_bytes >= int(current.get("size_bytes", 0)):
                    current["size_bytes"] = size_bytes
                    current["path"] = str(entry)
                current["_saw_progress_file"] = True
            elif not current.get("_saw_progress_file"):
                # Keep a visible path even before real bytes start flowing.
                current["path"] = str(entry)

            existing_total = current.get("total_bytes")
            if total_bytes and total_bytes > 0:
                if not existing_total or total_bytes > int(existing_total):
                    current["total_bytes"] = total_bytes

            if not current.get("model") and model_name:
                current["model"] = model_name

            existing_mtime = current.get("mtime")
            if (
                isinstance(existing_mtime, str)
                and existing_mtime
                and mtime_iso > existing_mtime
            ):
                current["mtime"] = mtime_iso

        def scan_dir(root_dir: Path) -> None:
            if not root_dir.exists() or not root_dir.is_dir():
                return
            try:
                for current_root, dirnames, filenames in os.walk(root_dir):
                    # Keep traversal bounded for frequent polling.
                    rel_parts = Path(current_root).relative_to(root_dir).parts
                    if len(rel_parts) > 5:
                        dirnames[:] = []
                        continue

                    for filename in filenames:
                        suffix = None
                        for cand in all_suffixes:
                            if filename.endswith(cand):
                                suffix = cand
                                break
                        if suffix is None:
                            continue
                        entry = Path(current_root) / filename
                        try:
                            stat = entry.stat()
                        except Exception:
                            continue
                        if now - stat.st_mtime > recent_threshold:
                            continue

                        register_entry(entry, stat, suffix)
            except Exception:
                return

        # Scan top-level cache download/temp directories (HF hub download manager)
        scan_dir(cache_dir / "downloads")
        scan_dir(cache_dir / "tmp")

        for model_root in cache_dir.iterdir():
            if not model_root.is_dir():
                continue
            if not (model_root.name.startswith("models--") or model_root.name.startswith("datasets--")):
                continue

            for subdir_name in ("blobs", "tmp"):
                scan_dir(model_root / subdir_name)

        downloads: List[Dict[str, Any]] = []
        for record in downloads_by_key.values():
            size_bytes = int(record.get("size_bytes", 0))
            total_bytes = record.get("total_bytes")
            percent = None
            if isinstance(total_bytes, int) and total_bytes > 0:
                percent = round(min(100.0, (size_bytes / total_bytes) * 100.0), 2)

            downloads.append(
                {
                    "path": str(record.get("path", "")),
                    "size_bytes": size_bytes,
                    "total_bytes": total_bytes,
                    "percent": percent,
                    "model": record.get("model"),
                    "mtime": record.get("mtime"),
                }
            )

        downloads.sort(key=lambda x: str(x.get("mtime") or ""), reverse=True)

        total_downloaded = sum(d["size_bytes"] for d in downloads)
        total_expected = None
        expected_list = [d["total_bytes"] for d in downloads if isinstance(d.get("total_bytes"), int)]
        if expected_list:
            total_expected = int(sum(expected_list))

        return {
            "cache_dir": str(cache_dir),
            "active": len(downloads) > 0,
            "downloads": downloads,
            "total_downloaded_bytes": total_downloaded,
            "total_expected_bytes": total_expected,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _try_read_expected_size(self, file_path: Path) -> Optional[int]:
        """Try to read expected size from huggingface metadata files."""
        base = str(file_path)
        for suffix in (".incomplete", ".lock", ".tmp"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        candidates = [
            base + ".json",
            base + ".metadata",
            base + ".size",
            base + ".meta",
        ]
        for cand in candidates:
            try:
                if not os.path.exists(cand):
                    continue
                with open(cand, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                parsed = self._parse_metadata_content(content)
                found = self._find_size_in_obj(parsed)
                if found is not None:
                    return found

                header_match = re.search(
                    r"content[-_ ]length[^0-9]{0,20}(\d{3,})",
                    content,
                    flags=re.IGNORECASE,
                )
                if header_match:
                    return int(header_match.group(1))

                explicit_match = re.search(
                    r"(?:expected|total|size|length)[^0-9]{0,20}(\d{3,})",
                    content,
                    flags=re.IGNORECASE,
                )
                if explicit_match:
                    return int(explicit_match.group(1))

                all_numbers = [int(x) for x in re.findall(r"(\d+)", content)]
                large_numbers = [x for x in all_numbers if x >= 1024]
                if large_numbers:
                    return max(large_numbers)
            except Exception:
                continue
        return None

    def _extract_model_name_from_metadata(self, file_path: Path) -> Optional[str]:
        """Try to infer model name/repo id from HF metadata files."""
        base = str(file_path)
        for suffix in (".incomplete", ".lock", ".tmp"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        metadata_candidates = [
            base + ".metadata",
            base + ".json",
            base + ".meta",
        ]
        for metadata_path in metadata_candidates:
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                parsed = self._parse_metadata_content(content)
                if not isinstance(parsed, dict):
                    continue
                for key in ("repo_id", "model_id", "model", "id", "name"):
                    value = parsed.get(key)
                    if isinstance(value, str) and "/" in value:
                        return value
                url = parsed.get("url")
                if isinstance(url, str):
                    match = re.search(r"/(?:models|datasets)/([^/]+/[^/]+)/", url)
                    if match:
                        return match.group(1)
            except Exception:
                continue
        return None

    def _parse_metadata_content(self, content: str) -> Optional[Any]:
        """Parse text metadata as JSON when possible."""
        if not content:
            return None
        try:
            return json.loads(content)
        except Exception:
            # Some metadata files are line-based key/value text.
            meta: Dict[str, Any] = {}
            for line in content.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                elif "=" in line:
                    key, value = line.split("=", 1)
                else:
                    continue
                key = key.strip().lower()
                value = value.strip()
                if not key:
                    continue
                meta[key] = value
            return meta if meta else None

    def _find_size_in_obj(self, obj: Any) -> Optional[int]:
        """Recursively extract expected size fields from parsed metadata."""
        if obj is None:
            return None

        size_keys = {
            "size",
            "size_bytes",
            "file_size",
            "content_length",
            "content-length",
            "length",
            "total",
            "total_bytes",
            "expected_size",
            "expected_bytes",
        }

        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).strip().lower()
                if key_lower in size_keys:
                    parsed = self._to_int(value)
                    if parsed is not None and parsed > 0:
                        return parsed
                nested = self._find_size_in_obj(value)
                if nested is not None:
                    return nested
            return None

        if isinstance(obj, list):
            for item in obj:
                nested = self._find_size_in_obj(item)
                if nested is not None:
                    return nested
            return None

        return None

    def _to_int(self, value: Any) -> Optional[int]:
        """Convert metadata scalar values to int when possible."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"(\d+)", value)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    return None
        return None

    def _extract_model_name_from_path(self, file_path: Path) -> Optional[str]:
        """Extract model name from cache path."""
        parts = file_path.parts
        for part in parts:
            if part.startswith("models--"):
                items = part.split("--")
                if len(items) >= 3:
                    org = items[1]
                    name = "--".join(items[2:])
                    return f"{org}/{name}"
            if part.startswith("datasets--"):
                items = part.split("--")
                if len(items) >= 3:
                    org = items[1]
                    name = "--".join(items[2:])
                    return f"{org}/{name}"
        return None

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _to_display_path(self, path: Path, is_dir: bool = False) -> str:
        """Prefer project-relative path for frontend display."""
        try:
            rel = path.resolve().relative_to(self.project_root.resolve())
            display = str(rel)
        except Exception:
            display = str(path.resolve() if path.exists() else path)
        if is_dir and not display.endswith("/"):
            display += "/"
        return display

    def get_recommended_gpu(self) -> Optional[int]:
        """
        Get recommended GPU based on availability

        Returns:
            GPU ID or None
        """
        gpus = self.detect_gpus()
        if not gpus:
            return None

        # Find GPU with most free memory
        available_gpus = [g for g in gpus if g.get("available", True)]
        if not available_gpus:
            return gpus[0]["id"]  # Return first GPU if none are "available"

        # Parse memory_free and find max
        best_gpu = None
        max_free = -1
        for gpu in available_gpus:
            memory_free_str = gpu.get("memory_free", "0 MB")
            try:
                # Extract number from "12345 MB"
                free_mb = int(memory_free_str.split()[0])
                if free_mb > max_free:
                    max_free = free_mb
                    best_gpu = gpu["id"]
            except:
                continue

        return best_gpu if best_gpu is not None else available_gpus[0]["id"]


# Global instance
system_service = SystemService()
