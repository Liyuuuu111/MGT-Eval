"""System Service for GPU and model detection"""

import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
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
        Detect available GPUs with robust fallback:
        1) NVML (pynvml) -> 2) nvidia-smi -> 3) torch.cuda

        Returns:
            List of GPU information dictionaries
        """
        metrics = self._detect_gpu_metrics()
        gpus: List[Dict[str, Any]] = []

        for item in metrics:
            memory_total_mb = self._to_float(item.get("memory_total_mb"))
            memory_free_mb = self._to_float(item.get("memory_free_mb"))
            memory_used_mb = self._to_float(item.get("memory_used_mb"))
            utilization = self._to_float(item.get("utilization"))

            if memory_free_mb is None and memory_total_mb is not None and memory_used_mb is not None:
                memory_free_mb = max(0.0, memory_total_mb - memory_used_mb)

            if utilization is None and memory_total_mb and memory_used_mb is not None and memory_total_mb > 0:
                utilization = min(100.0, max(0.0, (memory_used_mb / memory_total_mb) * 100.0))

            available = True
            if utilization is not None:
                available = utilization < 90.0
            elif memory_total_mb and memory_free_mb is not None and memory_total_mb > 0:
                available = (memory_free_mb / memory_total_mb) > 0.05

            gpus.append({
                "id": int(item.get("index", 0)),
                "name": str(item.get("name", "Unknown GPU")),
                "memory_total": f"{int(memory_total_mb)} MB" if memory_total_mb is not None else "N/A",
                "memory_free": f"{int(memory_free_mb)} MB" if memory_free_mb is not None else "N/A",
                "utilization": f"{int(utilization)}%" if utilization is not None else "N/A",
                "available": available,
            })

        return gpus

    def detect_gpu_monitor_stats(self) -> List[Dict[str, Any]]:
        """
        Detect GPU monitor stats for /system/monitor endpoint with the same fallback chain.
        Unknown fields are normalized to 0 to keep response schema stable.
        """
        metrics = self._detect_gpu_metrics()
        rows: List[Dict[str, Any]] = []

        for item in metrics:
            memory_total_mb = self._to_float(item.get("memory_total_mb")) or 0.0
            memory_used_mb = self._to_float(item.get("memory_used_mb"))
            memory_free_mb = self._to_float(item.get("memory_free_mb"))
            utilization = self._to_float(item.get("utilization"))
            temperature = self._to_float(item.get("temperature"))

            if memory_used_mb is None and memory_free_mb is not None and memory_total_mb > 0:
                memory_used_mb = max(0.0, memory_total_mb - memory_free_mb)
            if memory_used_mb is None:
                memory_used_mb = 0.0

            if utilization is None and memory_total_mb > 0:
                utilization = min(100.0, max(0.0, (memory_used_mb / memory_total_mb) * 100.0))
            if utilization is None:
                utilization = 0.0

            if temperature is None:
                temperature = 0.0

            rows.append({
                "index": int(item.get("index", 0)),
                "name": str(item.get("name", "Unknown GPU")),
                "utilization": float(round(utilization, 2)),
                "memory_used_mb": float(round(memory_used_mb, 2)),
                "memory_total_mb": float(round(memory_total_mb, 2)),
                "temperature": float(round(temperature, 2)),
            })

        return rows

    def _detect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """
        Internal unified GPU probe with fallback.
        Each row may contain:
        index, name, memory_total_mb, memory_free_mb, memory_used_mb, utilization, temperature, source
        """
        probes = (
            self._detect_gpu_metrics_via_nvml,
            self._detect_gpu_metrics_via_nvidia_smi,
            self._detect_gpu_metrics_via_torch,
        )
        for probe in probes:
            rows = probe()
            if rows:
                return rows
        return []

    def _detect_gpu_metrics_via_nvml(self) -> List[Dict[str, Any]]:
        """Preferred probe: pynvml (works even when nvidia-smi binary is absent)."""
        try:
            from pynvml import (
                NVML_TEMPERATURE_GPU,
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetMemoryInfo,
                nvmlDeviceGetName,
                nvmlDeviceGetTemperature,
                nvmlDeviceGetUtilizationRates,
                nvmlInit,
                nvmlShutdown,
            )
        except Exception:
            return []

        rows: List[Dict[str, Any]] = []
        initialized = False
        try:
            nvmlInit()
            initialized = True
            for index in range(int(nvmlDeviceGetCount())):
                handle = nvmlDeviceGetHandleByIndex(index)
                name = nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                mem = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                try:
                    temperature = float(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU))
                except Exception:
                    temperature = None

                rows.append({
                    "index": index,
                    "name": str(name),
                    "memory_total_mb": mem.total / (1024 ** 2),
                    "memory_used_mb": mem.used / (1024 ** 2),
                    "memory_free_mb": mem.free / (1024 ** 2),
                    "utilization": float(util.gpu),
                    "temperature": temperature,
                    "source": "nvml",
                })
        except Exception:
            rows = []
        finally:
            if initialized:
                try:
                    nvmlShutdown()
                except Exception:
                    pass
        return rows

    def _detect_gpu_metrics_via_nvidia_smi(self) -> List[Dict[str, Any]]:
        """Fallback probe: nvidia-smi command."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.total,memory.free,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
        except Exception:
            return []

        if result.returncode != 0:
            return []

        rows: List[Dict[str, Any]] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "utilization": self._to_float(parts[2]),
                "memory_total_mb": self._to_float(parts[3]),
                "memory_free_mb": self._to_float(parts[4]),
                "memory_used_mb": self._to_float(parts[5]),
                "temperature": self._to_float(parts[6]),
                "source": "nvidia-smi",
            })
        return rows

    def _detect_gpu_metrics_via_torch(self) -> List[Dict[str, Any]]:
        """Last-resort probe: torch.cuda (works when CUDA runtime is available)."""
        try:
            import torch
        except Exception:
            return []

        if not torch.cuda.is_available():
            return []

        rows: List[Dict[str, Any]] = []
        try:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_mb = float(props.total_memory) / (1024 ** 2)
                free_mb = None
                used_mb = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    free_mb = float(free_bytes) / (1024 ** 2)
                    used_mb = (float(total_bytes) - float(free_bytes)) / (1024 ** 2)
                    total_mb = float(total_bytes) / (1024 ** 2)
                except Exception:
                    # mem_get_info may be unavailable on some CUDA/runtime combinations.
                    pass

                rows.append({
                    "index": index,
                    "name": str(props.name),
                    "memory_total_mb": total_mb,
                    "memory_free_mb": free_mb,
                    "memory_used_mb": used_mb,
                    "utilization": None,
                    "temperature": None,
                    "source": "torch",
                })
        except Exception:
            return []

        return rows

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

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert loosely formatted numeric values (including 'N/A') to float."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            lowered = text.lower()
            if lowered in {"n/a", "na", "[n/a]", "none", "null", "unknown"}:
                return None
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                try:
                    return float(match.group(0))
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

    def get_calibrator_thresholds(self, raw_path: str) -> Dict[str, Any]:
        """Parse threshold presets from a calibrator file (or a directory of calibrators)."""
        target = self._resolve_user_path(raw_path)
        if not target.exists():
            raise FileNotFoundError(f"Calibrator path not found: {raw_path}")

        files: List[Path] = []
        selected_path = target
        if target.is_file():
            if target.suffix.lower() == ".json":
                files = [target]
        else:
            try:
                all_files = sorted(
                    [p for p in target.rglob("*.json") if p.is_file()],
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
                chosen = self._select_best_calibrator_json(all_files)
                if chosen is not None:
                    files = [chosen]
                    selected_path = chosen
                else:
                    files = []
            except Exception:
                files = []

        merged: List[Dict[str, Any]] = []
        default_threshold: Optional[float] = None
        seen: Dict[float, Tuple[int, Dict[str, Any]]] = {}

        for idx, file_path in enumerate(files):
            payload = self._read_json_file(file_path)
            if not isinstance(payload, dict):
                continue
            presets, suggested = self._extract_threshold_presets(payload, source_file=file_path.name)
            if default_threshold is None and suggested is not None:
                default_threshold = suggested

            for preset in presets:
                thr = float(preset["threshold"])
                key = round(thr, 12)
                priority = int(preset.get("_priority", 99))
                existing = seen.get(key)
                if existing is None or priority < existing[0]:
                    seen[key] = (priority, preset)

            # Guard against very large directories.
            if idx >= 199:
                break

        merged = [entry for _, entry in sorted(seen.values(), key=lambda item: self._threshold_sort_key(item[1]))]
        for item in merged:
            item.pop("_priority", None)
            item.pop("_fpr", None)

        return {
            "path": self._to_display_path(selected_path, is_dir=selected_path.is_dir()),
            "presets": merged,
            "default_threshold": default_threshold,
        }

    def _resolve_user_path(self, raw: str) -> Path:
        text = str(raw or "").strip()
        p = Path(text).expanduser()
        if not p.is_absolute():
            p = (self.project_root / p).resolve()
        else:
            p = p.resolve()
        return p

    def _read_json_file(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None

    def _select_best_calibrator_json(self, files: List[Path]) -> Optional[Path]:
        """Pick one best calibrator JSON from directory candidates."""
        if not files:
            return None

        def score_file(path: Path) -> Tuple[float, float, str]:
            name = path.name.lower()
            score = 0.0
            if any(token in name for token in ("best", "latest", "final")):
                score += 120.0
            if any(token in name for token in ("calibrator", "threshold", "decision", "boundary")):
                score += 100.0
            if any(token in name for token in ("tpr", "fpr", "roc", "pr_curve", "pr-curve")):
                score += 80.0
            if any(token in name for token in ("summary", "manifest", "metrics", "prediction", "predictions", "log", "asr")):
                score -= 180.0

            payload = self._read_json_file(path)
            if isinstance(payload, dict):
                meta = payload.get("meta")
                dev = meta.get("dev") if isinstance(meta, dict) else None
                decision = dev.get("decision") if isinstance(dev, dict) else None
                tpr_at_fpr = dev.get("tpr_at_fpr") if isinstance(dev, dict) else None
                calibrator = payload.get("calibrator")

                if isinstance(tpr_at_fpr, dict) and len(tpr_at_fpr) > 0:
                    score += 140.0
                if isinstance(decision, dict):
                    if self._to_float(decision.get("threshold")) is not None:
                        score += 90.0
                    thresholds = decision.get("thresholds")
                    if isinstance(thresholds, dict) and len(thresholds) > 0:
                        score += 90.0
                if isinstance(calibrator, dict):
                    if self._to_float(calibrator.get("beta0")) is not None and self._to_float(calibrator.get("beta1")) is not None:
                        score += 70.0
                    if isinstance(calibrator.get("name"), str):
                        score += 20.0

            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            return score, mtime, str(path)

        ranked = sorted(files, key=score_file, reverse=True)
        return ranked[0] if ranked else None

    def _extract_threshold_presets(
        self,
        payload: Dict[str, Any],
        source_file: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        presets: List[Dict[str, Any]] = []
        default_threshold: Optional[float] = None
        meta = payload.get("meta") if isinstance(payload, dict) else None
        dev = meta.get("dev") if isinstance(meta, dict) else None
        decision = dev.get("decision") if isinstance(dev, dict) else None
        dev_eval = meta.get("dev_eval") if isinstance(meta, dict) else None
        selected_threshold = dev.get("selected_threshold") if isinstance(dev, dict) else None

        # Pre-build threshold -> metrics map so different metadata branches can share metrics.
        threshold_metrics_map: Dict[float, Dict[str, Any]] = {}

        def _merge_threshold_metrics(threshold: Any, metric_payload: Any) -> None:
            thr = self._to_float(threshold)
            if thr is None:
                return
            metrics = self._extract_threshold_metrics(metric_payload)
            if not metrics:
                return
            key = round(float(thr), 12)
            existing = threshold_metrics_map.get(key, {})
            existing.update(metrics)
            threshold_metrics_map[key] = existing

        if isinstance(dev_eval, dict):
            _merge_threshold_metrics(default_threshold if default_threshold is not None else 0.5, dev_eval)
        if isinstance(selected_threshold, dict):
            _merge_threshold_metrics(selected_threshold.get("threshold"), selected_threshold)
            op = selected_threshold.get("operating_point")
            if isinstance(op, dict):
                _merge_threshold_metrics(op.get("threshold"), {"operating_point": op})

        def add_preset(key: str, threshold: Any, source: str, priority: int) -> None:
            thr = self._to_float(threshold)
            if thr is None:
                return
            fpr = self._extract_fpr_value(key)
            if fpr is not None:
                label = f"TPR@FPR<={fpr:g}: {thr:.6f}"
            else:
                label = f"{key}: {thr:.6f}"

            merged_metrics: Dict[str, Any] = {}
            known_metrics = threshold_metrics_map.get(round(float(thr), 12))
            if isinstance(known_metrics, dict):
                merged_metrics.update(known_metrics)
            if fpr is not None and merged_metrics.get("target_fpr") is None:
                merged_metrics["target_fpr"] = fpr

            presets.append({
                "key": str(key),
                "label": f"{source_file} | {label}",
                "threshold": float(thr),
                "source": f"{source_file}:{source}",
                **merged_metrics,
                "_priority": priority,
                "_fpr": fpr,
            })

        # 1) meta.dev.tpr_at_fpr
        tpr_at_fpr = dev.get("tpr_at_fpr") if isinstance(dev, dict) else None
        if isinstance(tpr_at_fpr, dict):
            for fpr_key, info in tpr_at_fpr.items():
                if not isinstance(info, dict):
                    continue
                fpr_value = self._to_float(fpr_key)
                info_metrics = self._extract_threshold_metrics(info)
                if fpr_value is not None and info_metrics.get("target_fpr") is None:
                    info_metrics["target_fpr"] = fpr_value
                _merge_threshold_metrics(info.get("threshold"), info_metrics)
                add_preset(f"tpr@fpr<={fpr_key}", info.get("threshold"), "meta.dev.tpr_at_fpr", 1)

        # 2) meta.dev.decision.thresholds
        thresholds = decision.get("thresholds") if isinstance(decision, dict) else None
        if isinstance(thresholds, dict):
            for name, value in thresholds.items():
                candidate = value
                if isinstance(value, dict):
                    if "threshold" in value:
                        candidate = value.get("threshold")
                    elif "thr" in value:
                        candidate = value.get("thr")
                    _merge_threshold_metrics(candidate, value)
                add_preset(str(name), candidate, "meta.dev.decision.thresholds", 2)

        # 3) meta.dev.decision.threshold
        if isinstance(decision, dict):
            dec_thr = self._to_float(decision.get("threshold"))
            if dec_thr is not None:
                default_threshold = float(dec_thr)
                _merge_threshold_metrics(dec_thr, decision)
                if isinstance(selected_threshold, dict):
                    sel_thr = self._to_float(selected_threshold.get("threshold"))
                    if sel_thr is not None and abs(float(sel_thr) - float(dec_thr)) <= 1e-9:
                        _merge_threshold_metrics(dec_thr, selected_threshold)
                add_preset("decision", dec_thr, "meta.dev.decision.threshold", 3)

        return presets, default_threshold

    def _extract_threshold_metrics(self, payload: Any) -> Dict[str, Any]:
        """Extract optional quality metrics for a threshold preset from mixed payload layouts."""
        if not isinstance(payload, dict):
            return {}

        metrics: Dict[str, Any] = {}

        def put_float(field: str, value: Any) -> None:
            v = self._to_float(value)
            if v is not None:
                metrics[field] = float(v)

        def put_int(field: str, value: Any) -> None:
            v = self._to_int(value)
            if v is not None:
                metrics[field] = int(v)

        # Flat keys
        for key in ("tpr", "fpr", "acc", "precision", "recall", "f1", "target_fpr"):
            put_float(key, payload.get(key))

        # Nested "metrics" object
        nested_metrics = payload.get("metrics")
        if isinstance(nested_metrics, dict):
            for key in ("acc", "precision", "recall", "f1", "tpr", "fpr"):
                put_float(key, nested_metrics.get(key))
            confusion = nested_metrics.get("confusion")
            if isinstance(confusion, dict):
                for key in ("tp", "tn", "fp", "fn"):
                    put_int(key, confusion.get(key))

        # Nested operating point for TPR@FPR modes
        operating_point = payload.get("operating_point")
        if isinstance(operating_point, dict):
            put_float("tpr", operating_point.get("tpr"))
            put_float("fpr", operating_point.get("fpr"))
            put_float("target_fpr", payload.get("target_fpr"))

        # Flat confusion object
        confusion = payload.get("confusion")
        if isinstance(confusion, dict):
            for key in ("tp", "tn", "fp", "fn"):
                put_int(key, confusion.get(key))

        return metrics

    def _extract_fpr_value(self, text: str) -> Optional[float]:
        if not text:
            return None
        lowered = str(text).strip().lower()
        match = re.search(r"fpr\s*<?=?\s*([0-9eE\.\-\+]+)", lowered)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        match = re.search(r"tpr@([0-9eE\.\-\+]+)", lowered)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def _threshold_sort_key(self, preset: Dict[str, Any]) -> Tuple[int, float, int, str]:
        fpr = preset.get("_fpr")
        priority = int(preset.get("_priority", 99))
        if isinstance(fpr, float):
            return (0, fpr, priority, str(preset.get("key", "")))
        return (1, 1e9, priority, str(preset.get("key", "")))

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
