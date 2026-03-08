"""Demo inference service for single-text detector predictions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.services.executor import command_executor
from backend.services.yaml_service import YAMLService


class DemoService:
    """Run one-off detector inference for demo page."""

    def __init__(self):
        self.yaml_service = YAMLService()
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.tmp_root = self._resolve_tmp_root()
        # Store metadata for async jobs so results can be fetched later
        self._job_meta: Dict[str, Dict[str, Any]] = {}
        # Store logs for async jobs to extract run_dir
        self._job_logs: Dict[str, List[Tuple[str, str]]] = {}

    def _is_writable_dir(self, directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".mgt_eval_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _resolve_tmp_root(self) -> Path:
        env_keys = ("MGT_EVAL_DEMO_TMP_ROOT", "UNCOVERAI_DETECT_TMP_ROOT")
        for env_key in env_keys:
            raw = os.getenv(env_key, "").strip()
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if self._is_writable_dir(candidate):
                return candidate

        candidates = [
            Path(tempfile.gettempdir()) / "mgt_eval_backend_demo",
            Path.home() / ".mgt_eval" / "demo_tmp",
            self.project_root / ".runtime" / "demo_tmp",
        ]
        for candidate in candidates:
            if self._is_writable_dir(candidate):
                return candidate

        raise PermissionError(
            "No writable demo temp directory available. "
            "Set MGT_EVAL_DEMO_TMP_ROOT to a writable path."
        )

    def list_detectors(self) -> List[str]:
        return self.yaml_service.list_detectors("detect")

    def load_template(self, detector: str) -> Dict[str, Any]:
        return self.yaml_service.load_template("detect", detector)

    def prepare_demo_config(
        self,
        detector: str,
        text: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        hf_endpoint: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, Path, Path]:
        """Prepare config and temp files for demo execution.

        Returns (cfg, request_id, data_path, out_dir).
        """
        config_overrides = config_overrides or {}
        template = self.load_template(detector)
        cfg = dict(template)

        for key, value in config_overrides.items():
            if key in {"data", "out"}:
                continue
            cfg[key] = value

        cfg["detector"] = template.get("detector", detector)
        cfg["sample_k"] = 1
        cfg["save_curves"] = False

        # Disable attack dataset evaluation for demo (single-text detection)
        if "attack_dataset" in cfg:
            del cfg["attack_dataset"]
        if "attack_dataset_only" in cfg:
            del cfg["attack_dataset_only"]

        if hf_endpoint is not None:
            cfg["hf_endpoint"] = str(hf_endpoint).strip()

        request_id = uuid.uuid4().hex[:10]
        data_path = self.tmp_root / f"demo_input_{request_id}.jsonl"
        out_dir = self.tmp_root / f"demo_run_{request_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["data"] = str(data_path)
        cfg["out"] = str(out_dir)

        record = {"id": "demo-1", "text": text, "label": 0}
        data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        return cfg, request_id, data_path, out_dir

    def store_job_meta(self, job_id: str, cfg: Dict[str, Any], data_path: Path, out_dir: Path) -> None:
        """Store metadata for an async demo job."""
        self._job_meta[job_id] = {
            "cfg": cfg,
            "data_path": str(data_path),
            "out_dir": str(out_dir),
        }
        self._job_logs[job_id] = []

    def add_job_log(self, job_id: str, level: str, message: str) -> None:
        """Add a log entry for an async demo job."""
        if job_id in self._job_logs:
            self._job_logs[job_id].append((level, message))

    def parse_result(self, job_id: str) -> Dict[str, Any]:
        """Parse detection result from a completed demo job."""
        meta = self._job_meta.get(job_id)
        if not meta:
            raise KeyError(f"No demo metadata for job {job_id}")

        cfg = meta["cfg"]
        out_dir = Path(meta["out_dir"])

        # Clean up input file
        try:
            Path(meta["data_path"]).unlink(missing_ok=True)
        except Exception:
            pass

        # Try to extract run_dir from logs first, then fallback
        logs = self._job_logs.get(job_id, [])
        run_dir = self._extract_run_dir(logs) or self._resolve_fallback_run_dir(out_dir)
        summary_path = run_dir / "metrics" / "summary.json" if run_dir else None
        predictions_path = run_dir / "predictions.json" if run_dir else None

        summary = self._load_json(summary_path) if summary_path else None
        predictions = self._load_json(predictions_path) if predictions_path else None
        first_pred = predictions[0] if isinstance(predictions, list) and predictions else {}

        threshold = float(summary.get("threshold", cfg.get("threshold", 0.5))) if isinstance(summary, dict) else float(cfg.get("threshold", 0.5))
        ai_probability = self._extract_probability(first_pred, summary)
        pred_label = self._extract_pred_label(first_pred, ai_probability, threshold)
        label_text = "machine" if pred_label == 1 else "human"
        confidence = ai_probability if pred_label == 1 else (1.0 - ai_probability)
        confidence = max(0.0, min(1.0, float(confidence)))

        result = {
            "label": label_text,
            "confidence": confidence,
            "ai_probability": ai_probability,
            "threshold": threshold,
            "artifact_paths": {
                "run_dir": str(run_dir) if run_dir else None,
                "summary_json": str(summary_path) if summary_path and summary_path.exists() else None,
                "predictions_json": str(predictions_path) if predictions_path and predictions_path.exists() else None,
            },
        }

        # Clean up job metadata and logs after result is extracted
        self._job_meta.pop(job_id, None)
        self._job_logs.pop(job_id, None)

        return result

    async def predict(
        self,
        detector: str,
        text: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        hf_endpoint: Optional[str] = None,
        timeout_sec: int = 120,
    ) -> Dict[str, Any]:
        config_overrides = config_overrides or {}
        template = self.load_template(detector)
        cfg = dict(template)

        for key, value in config_overrides.items():
            if key in {"data", "out"}:
                continue
            cfg[key] = value

        cfg["detector"] = template.get("detector", detector)
        cfg["sample_k"] = 1
        cfg["save_curves"] = False

        # Disable attack dataset evaluation for demo (single-text detection)
        if "attack_dataset" in cfg:
            del cfg["attack_dataset"]
        if "attack_dataset_only" in cfg:
            del cfg["attack_dataset_only"]

        if hf_endpoint is not None:
            cfg["hf_endpoint"] = str(hf_endpoint).strip()

        request_id = uuid.uuid4().hex[:10]
        data_path = self.tmp_root / f"demo_input_{request_id}.jsonl"
        out_dir = self.tmp_root / f"demo_run_{request_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["data"] = str(data_path)
        cfg["out"] = str(out_dir)

        record = {"id": "demo-1", "text": text, "label": 0}
        data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        logs: List[Tuple[str, str]] = []

        async def log_callback(_job_id: str, message: str, level: str):
            logs.append((level, message))

        job_id = f"demo-{request_id}"
        try:
            success, exit_code = await asyncio.wait_for(
                command_executor.execute_command(job_id, "detect", cfg, log_callback),
                timeout=float(timeout_sec),
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Demo detect timed out after {timeout_sec}s.") from e
        finally:
            try:
                data_path.unlink(missing_ok=True)
            except Exception:
                pass

        if not success:
            tail = "\n".join([m for _, m in logs[-10:]])
            raise RuntimeError(
                f"Demo detect failed (exit_code={exit_code}). Recent logs:\n{tail}"
            )

        run_dir = self._extract_run_dir(logs) or self._resolve_fallback_run_dir(out_dir)
        summary_path = run_dir / "metrics" / "summary.json" if run_dir else None
        predictions_path = run_dir / "predictions.json" if run_dir else None

        summary = self._load_json(summary_path) if summary_path else None
        predictions = self._load_json(predictions_path) if predictions_path else None
        first_pred = predictions[0] if isinstance(predictions, list) and predictions else {}

        threshold = float(summary.get("threshold", cfg.get("threshold", 0.5))) if isinstance(summary, dict) else float(cfg.get("threshold", 0.5))
        ai_probability = self._extract_probability(first_pred, summary)
        pred_label = self._extract_pred_label(first_pred, ai_probability, threshold)
        label_text = "machine" if pred_label == 1 else "human"
        confidence = ai_probability if pred_label == 1 else (1.0 - ai_probability)
        confidence = max(0.0, min(1.0, float(confidence)))

        return {
            "label": label_text,
            "confidence": confidence,
            "ai_probability": ai_probability,
            "threshold": threshold,
            "artifact_paths": {
                "run_dir": str(run_dir) if run_dir else None,
                "summary_json": str(summary_path) if summary_path and summary_path.exists() else None,
                "predictions_json": str(predictions_path) if predictions_path and predictions_path.exists() else None,
            },
        }

    def _extract_run_dir(self, logs: List[Tuple[str, str]]) -> Optional[Path]:
        patterns = [
            re.compile(r"run_dir\s*->\s*(.+)$"),
            re.compile(r"\bresults saved to:\s*(.+)$"),
        ]
        for _level, message in logs:
            msg = str(message or "").strip()
            for pattern in patterns:
                match = pattern.search(msg)
                if match:
                    raw = match.group(1).strip().strip("'").strip('"')
                    path = Path(raw).expanduser()
                    if not path.is_absolute():
                        path = command_executor.project_root / path
                    if path.exists():
                        return path.resolve()
        return None

    def _resolve_fallback_run_dir(self, out_dir: Path) -> Optional[Path]:
        if not out_dir.exists():
            return None
        direct_summary = out_dir / "metrics" / "summary.json"
        if direct_summary.exists():
            return out_dir
        candidates = list(out_dir.rglob("metrics/summary.json"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return candidates[0].parents[1]

    def _load_json(self, path: Optional[Path]) -> Optional[Any]:
        if not path or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _extract_probability(self, prediction: Dict[str, Any], summary: Optional[Dict[str, Any]]) -> float:
        if isinstance(prediction, dict):
            prob = prediction.get("prob")
            if isinstance(prob, (int, float)):
                return float(max(0.0, min(1.0, prob)))
        if isinstance(summary, dict):
            metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
            # fallback: if only score exists in single-item eval, map by threshold center.
            score = metrics.get("score")
            if isinstance(score, (int, float)):
                return float(max(0.0, min(1.0, score)))
        return 0.5

    def _extract_pred_label(self, prediction: Dict[str, Any], prob: float, threshold: float) -> int:
        if isinstance(prediction, dict):
            pred = prediction.get("pred")
            if isinstance(pred, int):
                return 1 if pred == 1 else 0
            if isinstance(pred, float):
                return 1 if int(round(pred)) == 1 else 0
        return 1 if prob >= threshold else 0


demo_service = DemoService()
