"""Managed runtime file lifecycle service.

This service centralizes:
- dataset uploads for build/attack/train/detect,
- generated download token registration (TTL based),
- safe file serving and cleanup of expired files/tokens.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiofiles
from fastapi import UploadFile


def _now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _sanitize_name(file_name: str) -> str:
    raw = str(file_name or "").strip()
    if not raw:
        return "dataset.bin"
    base = os.path.basename(raw)
    safe_chars = []
    for ch in base:
        if ch.isalnum() or ch in {".", "-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._")
    return safe or "dataset.bin"


def _safe_phase(phase: Optional[str]) -> str:
    value = str(phase or "").strip().lower()
    if value in {"build", "attack", "train", "detect"}:
        return value
    return "build"


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


@dataclass
class ResolvedDownload:
    token: str
    file_name: str
    file_path: str
    expires_ts: int


class FileLifecycleService:
    """Persist and manage uploads/generated files with tokenized download URLs."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.runtime_root = self._resolve_runtime_root()
        self.upload_root = self.runtime_root / "uploads"
        self.generated_root = self.runtime_root / "generated"
        self.db_path = self.runtime_root / "file_lifecycle.db"

        self.output_ttl_hours = self._parse_int_env("MGT_EVAL_OUTPUT_TTL_HOURS", 72)
        self.upload_ttl_hours = self._parse_int_env("MGT_EVAL_UPLOAD_TTL_HOURS", 24)
        self.max_upload_size_mb = self._parse_int_env("MGT_EVAL_MAX_UPLOAD_SIZE_MB", 10)
        self.max_upload_size_bytes = int(self.max_upload_size_mb * 1024 * 1024)
        self.cleanup_interval_seconds = self._parse_int_env("MGT_EVAL_FILE_CLEANUP_INTERVAL_SECONDS", 600)

        self._lock = threading.RLock()
        self._job_managed_uploads: dict[str, set[str]] = {}
        self._init_storage()

    def _parse_int_env(self, key: str, default: int) -> int:
        raw = os.getenv(key, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except Exception:
            return default

    def _resolve_runtime_root(self) -> Path:
        env_value = os.getenv("MGT_EVAL_RUNTIME_FILE_ROOT", "").strip()
        if env_value:
            root = Path(env_value).expanduser()
        else:
            root = self.project_root / ".runtime" / "files"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_storage(self) -> None:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.generated_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS uploads (
                        upload_id TEXT PRIMARY KEY,
                        phase TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        stored_path TEXT NOT NULL UNIQUE,
                        file_size INTEGER NOT NULL,
                        created_ts INTEGER NOT NULL,
                        consumed_by_job TEXT,
                        consumed_ts INTEGER,
                        deleted_ts INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS generated_downloads (
                        token TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_ts INTEGER NOT NULL,
                        expires_ts INTEGER NOT NULL,
                        download_count INTEGER NOT NULL DEFAULT 0,
                        last_downloaded_ts INTEGER,
                        deleted_ts INTEGER
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_uploads_consumed ON uploads(consumed_by_job, deleted_ts, created_ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_downloads_job ON generated_downloads(job_id, deleted_ts, expires_ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_downloads_expire ON generated_downloads(expires_ts, deleted_ts)"
                )

    async def save_uploaded_dataset(self, file: UploadFile, phase: Optional[str] = None) -> Dict[str, Any]:
        phase_name = _safe_phase(phase)
        upload_id = uuid.uuid4().hex
        safe_name = _sanitize_name(getattr(file, "filename", "") or "")
        phase_dir = self.upload_root / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        stored_path = phase_dir / f"{upload_id}_{safe_name}"

        file_size = 0
        try:
            async with aiofiles.open(stored_path, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > self.max_upload_size_bytes:
                        raise ValueError(
                            f"File is too large. Maximum allowed size is {self.max_upload_size_mb}MB."
                        )
                    await out.write(chunk)
        except Exception:
            try:
                stored_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        finally:
            await file.close()

        created_ts = _now_ts()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO uploads (
                        upload_id, phase, file_name, stored_path, file_size, created_ts
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_id,
                        phase_name,
                        safe_name,
                        str(stored_path.resolve()),
                        int(file_size),
                        int(created_ts),
                    ),
                )

        return {
            "upload_id": upload_id,
            "file_name": safe_name,
            "file_size": int(file_size),
            "stored_path": str(stored_path.resolve()),
            "phase": phase_name,
        }

    def is_managed_upload_path(self, path_text: Any) -> bool:
        if not isinstance(path_text, str) or not path_text.strip():
            return False
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return _path_under(path, self.upload_root)

    def _mark_upload_consumed(self, path_text: str, job_id: str) -> None:
        path = Path(path_text).expanduser().resolve()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE uploads
                    SET consumed_by_job = ?, consumed_ts = COALESCE(consumed_ts, ?)
                    WHERE stored_path = ? AND deleted_ts IS NULL
                    """,
                    (job_id, _now_ts(), str(path)),
                )

    def _remember_job_upload(self, job_id: str, path_text: str) -> None:
        if job_id not in self._job_managed_uploads:
            self._job_managed_uploads[job_id] = set()
        self._job_managed_uploads[job_id].add(str(Path(path_text).expanduser().resolve()))

    def prepare_job_config(self, command: str, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = deepcopy(config) if isinstance(config, dict) else {}

        managed_fields: Iterable[str]
        if command in {"build", "attack"}:
            managed_fields = ("data",)
        elif command == "train":
            managed_fields = ("dataset_train", "dataset_test")
        elif command == "detect":
            managed_fields = ("data",)
        else:
            managed_fields = ()

        has_managed_input = False
        for key in managed_fields:
            raw = cfg.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            if self.is_managed_upload_path(raw):
                normalized = str(Path(raw).expanduser().resolve())
                cfg[key] = normalized
                self._mark_upload_consumed(normalized, job_id)
                self._remember_job_upload(job_id, normalized)
                has_managed_input = True

        if command in {"build", "attack"} and has_managed_input:
            out_raw = cfg.get("out")
            out_name = _sanitize_name(Path(str(out_raw)).name) if out_raw else "output.jsonl"
            out_dir = self.generated_root / command / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg["out"] = str((out_dir / out_name).resolve())

        return cfg

    def _cleanup_managed_uploads_for_job(self, job_id: str, fallback_config: Optional[Dict[str, Any]] = None) -> None:
        managed = set(self._job_managed_uploads.pop(job_id, set()))
        if fallback_config:
            for key in ("data", "dataset_train", "dataset_test"):
                raw = fallback_config.get(key) if isinstance(fallback_config, dict) else None
                if isinstance(raw, str) and self.is_managed_upload_path(raw):
                    managed.add(str(Path(raw).expanduser().resolve()))

        if not managed:
            return

        deleted_ts = _now_ts()
        with self._lock:
            with self._connect() as conn:
                for item in managed:
                    try:
                        p = Path(item).resolve()
                        if _path_under(p, self.upload_root) and p.exists():
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    conn.execute(
                        "UPDATE uploads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE stored_path = ?",
                        (deleted_ts, item),
                    )

    def _register_generated_file(self, job_id: str, file_path: Path, name: str) -> Optional[Dict[str, Any]]:
        if not file_path.exists() or not file_path.is_file():
            return None
        if not _path_under(file_path.resolve(), self.generated_root.resolve()):
            return None
        now = _now_ts()
        expires_ts = now + int(self.output_ttl_hours * 3600)
        token = secrets.token_urlsafe(32)
        size_bytes = int(file_path.stat().st_size)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO generated_downloads (
                        token, job_id, file_name, file_path, size_bytes, created_ts, expires_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token,
                        job_id,
                        str(name),
                        str(file_path.resolve()),
                        size_bytes,
                        now,
                        expires_ts,
                    ),
                )

        return {
            "name": str(name),
            "url": f"/api/files/generated/{token}",
            "expires_at": _ts_to_iso(expires_ts),
            "size_bytes": size_bytes,
        }

    def _register_build_attack_outputs(self, job_id: str, command: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if command not in {"build", "attack"}:
            return []
        out_raw = config.get("out") if isinstance(config, dict) else None
        if not isinstance(out_raw, str) or not out_raw.strip():
            return []
        out_path = Path(out_raw).expanduser()
        if not out_path.is_absolute():
            out_path = self.project_root / out_path
        out_path = out_path.resolve()

        candidates: List[Tuple[str, Path]] = [("output", out_path)]
        for split_name in ("train", "dev", "test"):
            split_path = Path(f"{out_path}.{split_name}.jsonl")
            candidates.append((split_name, split_path))

        downloads: List[Dict[str, Any]] = []
        for logical_name, p in candidates:
            if not p.exists():
                continue
            dl = self._register_generated_file(job_id, p, name=f"{logical_name}:{p.name}")
            if dl:
                downloads.append(dl)
        return downloads

    def finalize_job(
        self,
        *,
        command: str,
        job_id: str,
        config: Optional[Dict[str, Any]],
        success: bool,
    ) -> List[Dict[str, Any]]:
        cfg = config if isinstance(config, dict) else {}
        downloads: List[Dict[str, Any]] = []
        if success:
            downloads = self._register_build_attack_outputs(job_id, command, cfg)
        self._cleanup_managed_uploads_for_job(job_id, fallback_config=cfg)
        return downloads

    def get_job_downloads(self, job_id: str) -> List[Dict[str, Any]]:
        now = _now_ts()
        rows: List[sqlite3.Row] = []
        with self._lock:
            with self._connect() as conn:
                rows = list(
                    conn.execute(
                        """
                        SELECT token, file_name, file_path, size_bytes, expires_ts
                        FROM generated_downloads
                        WHERE job_id = ? AND deleted_ts IS NULL AND expires_ts > ?
                        ORDER BY created_ts ASC
                        """,
                        (job_id, now),
                    )
                )

        out: List[Dict[str, Any]] = []
        for row in rows:
            file_path = Path(str(row["file_path"])).expanduser()
            if not file_path.exists():
                self._mark_download_deleted(str(row["token"]))
                continue
            out.append(
                {
                    "name": str(row["file_name"]),
                    "url": f"/api/files/generated/{row['token']}",
                    "expires_at": _ts_to_iso(int(row["expires_ts"])),
                    "size_bytes": int(row["size_bytes"]),
                }
            )
        return out

    def _mark_download_deleted(self, token: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE generated_downloads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE token = ?",
                    (_now_ts(), token),
                )

    def resolve_download_token(self, token: str) -> Optional[ResolvedDownload]:
        now = _now_ts()
        token_text = str(token or "").strip()
        if not token_text:
            return None
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT token, file_name, file_path, expires_ts
                    FROM generated_downloads
                    WHERE token = ? AND deleted_ts IS NULL
                    """,
                    (token_text,),
                ).fetchone()
                if row is None:
                    return None

                expires_ts = int(row["expires_ts"])
                file_path = Path(str(row["file_path"])).expanduser().resolve()
                if expires_ts <= now:
                    self._safe_delete_generated_path(file_path)
                    conn.execute(
                        "UPDATE generated_downloads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE token = ?",
                        (now, token_text),
                    )
                    return None

                if (not file_path.exists()) or (not file_path.is_file()) or (not _path_under(file_path, self.generated_root)):
                    conn.execute(
                        "UPDATE generated_downloads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE token = ?",
                        (now, token_text),
                    )
                    return None

                conn.execute(
                    """
                    UPDATE generated_downloads
                    SET download_count = download_count + 1, last_downloaded_ts = ?
                    WHERE token = ?
                    """,
                    (now, token_text),
                )

                return ResolvedDownload(
                    token=token_text,
                    file_name=str(row["file_name"]),
                    file_path=str(file_path),
                    expires_ts=expires_ts,
                )

    def _safe_delete_generated_path(self, path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            return
        if not _path_under(resolved, self.generated_root):
            return
        try:
            resolved.unlink(missing_ok=True)
        except Exception:
            pass

    def cleanup_expired(self) -> Dict[str, int]:
        now = _now_ts()
        upload_cutoff = now - int(self.upload_ttl_hours * 3600)
        removed_uploads = 0
        removed_downloads = 0

        with self._lock:
            with self._connect() as conn:
                upload_rows = list(
                    conn.execute(
                        """
                        SELECT upload_id, stored_path, consumed_by_job, created_ts
                        FROM uploads
                        WHERE deleted_ts IS NULL
                        """
                    )
                )
                for row in upload_rows:
                    stored_path = Path(str(row["stored_path"])).expanduser()
                    consumed_by_job = row["consumed_by_job"]
                    created_ts = int(row["created_ts"])
                    should_delete = False
                    if not stored_path.exists():
                        should_delete = True
                    elif (not consumed_by_job) and created_ts <= upload_cutoff:
                        should_delete = True

                    if should_delete:
                        try:
                            if stored_path.exists() and _path_under(stored_path.resolve(), self.upload_root):
                                stored_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        conn.execute(
                            "UPDATE uploads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE upload_id = ?",
                            (now, str(row["upload_id"])),
                        )
                        removed_uploads += 1

                dl_rows = list(
                    conn.execute(
                        """
                        SELECT token, file_path, expires_ts
                        FROM generated_downloads
                        WHERE deleted_ts IS NULL
                        """
                    )
                )
                for row in dl_rows:
                    token = str(row["token"])
                    file_path = Path(str(row["file_path"])).expanduser()
                    expires_ts = int(row["expires_ts"])
                    should_delete = (expires_ts <= now) or (not file_path.exists())
                    if not should_delete:
                        continue
                    self._safe_delete_generated_path(file_path)
                    conn.execute(
                        "UPDATE generated_downloads SET deleted_ts = COALESCE(deleted_ts, ?) WHERE token = ?",
                        (now, token),
                    )
                    removed_downloads += 1

        return {
            "removed_uploads": removed_uploads,
            "removed_downloads": removed_downloads,
        }


file_lifecycle_service = FileLifecycleService()
