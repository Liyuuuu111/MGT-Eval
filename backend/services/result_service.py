"""Result service for resolving train/detect artifacts by job id."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.api.websocket.logs import manager as ws_manager
from backend.services.process_manager import Job, process_manager


class ResultService:
    """Resolve and summarize result artifacts for completed jobs."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        job = process_manager.get_job(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        logs = self._get_buffered_logs(job_id)
        parsed = self._extract_paths_from_logs(logs)

        run_dir = parsed.get("run_dir")
        auto_eval_dir = parsed.get("auto_eval_dir")
        if not run_dir:
            run_dir = self._fallback_run_dir(job)

        artifacts = self._resolve_artifacts(job.command, run_dir, auto_eval_dir)
        result = self._build_result_payload(job.command, artifacts)

        return {
            "job_id": job.id,
            "command": job.command,
            "status": job.status.value,
            "exit_code": job.exit_code,
            "artifacts": artifacts,
            "result": result,
        }

    def _get_buffered_logs(self, job_id: str) -> List[Dict[str, Any]]:
        buffer = ws_manager.message_buffer.get(job_id)
        if not buffer:
            return []
        return list(buffer)

    def _extract_paths_from_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        run_dir: Optional[str] = None
        auto_eval_dir: Optional[str] = None

        run_dir_patterns = [
            re.compile(r"run_dir\s*->\s*(.+)$"),
            re.compile(r"\bresults saved to:\s*(.+)$"),
            re.compile(r"\(multi-run\)\s*results saved to:\s*(.+)$"),
        ]
        auto_eval_pattern = re.compile(r"Auto-eval saved to:\s*(.+)$")

        for entry in logs:
            if entry.get("type") != "log":
                continue
            message = str(entry.get("message") or "").strip()
            if not message:
                continue

            for pattern in run_dir_patterns:
                match = pattern.search(message)
                if match:
                    run_dir = self._normalize_path(match.group(1))
                    break

            match_eval = auto_eval_pattern.search(message)
            if match_eval:
                auto_eval_dir = self._normalize_path(match_eval.group(1))

        return {
            "run_dir": run_dir,
            "auto_eval_dir": auto_eval_dir,
        }

    def _normalize_path(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        path_str = str(raw).strip().strip("'").strip('"')
        if not path_str:
            return None
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve())

    def _fallback_run_dir(self, job: Job) -> Optional[str]:
        cfg = job.config if isinstance(job.config, dict) else {}

        if job.command == "detect":
            out = cfg.get("out")
            if out:
                out_path = Path(self._normalize_path(str(out)) or "")
                candidate = self._find_latest_dir_with_file(out_path, "metrics/summary.json")
                if candidate:
                    return str(candidate)

        if job.command == "train":
            output_dir = cfg.get("output_dir")
            if output_dir:
                output_path = Path(self._normalize_path(str(output_dir)) or "")
                candidate = self._find_latest_dir_with_file(output_path, "train_summary.json")
                if candidate:
                    return str(candidate)

        return None

    def _find_latest_dir_with_file(self, root: Path, relative_file: str) -> Optional[Path]:
        if not root:
            return None
        if root.is_file():
            root = root.parent
        if not root.exists():
            return None

        candidates: List[Path] = []
        direct = root / relative_file
        if direct.exists():
            return root

        try:
            rel_depth = max(1, len(Path(relative_file).parts))
            for found in root.rglob(relative_file):
                if found.is_file():
                    run_parent = found.parents[rel_depth - 1]
                    candidates.append(run_parent)
        except Exception:
            return None

        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return candidates[0]

    def _resolve_artifacts(
        self,
        command: str,
        run_dir: Optional[str],
        auto_eval_dir: Optional[str],
    ) -> Dict[str, Optional[str]]:
        run_path = Path(run_dir) if run_dir else None
        eval_path = Path(auto_eval_dir) if auto_eval_dir else None

        artifacts: Dict[str, Optional[str]] = {
            "run_dir": str(run_path) if run_path and run_path.exists() else None,
            "summary_json": None,
            "train_summary_json": None,
            "manifest_json": None,
            "predictions_json": None,
            "eval_summary_json": None,
        }

        if run_path and run_path.exists():
            summary = run_path / "metrics" / "summary.json"
            train_summary = run_path / "train_summary.json"
            manifest = run_path / "run-manifest.json"
            predictions = run_path / "predictions.json"

            if summary.exists():
                artifacts["summary_json"] = str(summary)
            if train_summary.exists():
                artifacts["train_summary_json"] = str(train_summary)
            if manifest.exists():
                artifacts["manifest_json"] = str(manifest)
            if predictions.exists():
                artifacts["predictions_json"] = str(predictions)

        if command == "train":
            eval_summary = None
            if eval_path and eval_path.exists():
                candidate = eval_path / "metrics" / "summary.json"
                if candidate.exists():
                    eval_summary = candidate
            if eval_summary is None and run_path and run_path.exists():
                eval_summary = self._latest_file(run_path, "eval_test_*/metrics/summary.json")
            if eval_summary and eval_summary.exists():
                artifacts["eval_summary_json"] = str(eval_summary)

        return artifacts

    def _latest_file(self, root: Path, pattern: str) -> Optional[Path]:
        files = list(root.glob(pattern))
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return files[0]

    def _read_json(self, path: Optional[str]) -> Optional[Any]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _build_result_payload(self, command: str, artifacts: Dict[str, Optional[str]]) -> Dict[str, Any]:
        summary = self._read_json(artifacts.get("summary_json"))
        train_summary = self._read_json(artifacts.get("train_summary_json"))
        eval_summary = self._read_json(artifacts.get("eval_summary_json"))
        manifest = self._read_json(artifacts.get("manifest_json"))
        predictions = self._read_json(artifacts.get("predictions_json"))

        payload: Dict[str, Any] = {
            "command": command,
            "summary": summary,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "manifest": manifest,
        }

        if isinstance(predictions, list):
            payload["predictions_preview"] = predictions[:20]
            payload["predictions_count"] = len(predictions)
        else:
            payload["predictions_preview"] = []
            payload["predictions_count"] = 0

        return payload


result_service = ResultService()
