"""Demo API endpoints."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.responses import DetectorListResponse, ExecuteResponse, TemplateResponse
from backend.services.demo_service import demo_service
from backend.services.process_manager import process_manager
from backend.services.executor import command_executor
from backend.api.websocket.logs import manager as ws_manager


router = APIRouter()


class DemoPredictRequest(BaseModel):
    detector: str
    text: str
    config: Dict[str, Any] = {}
    hf_endpoint: Optional[str] = None


class DemoPredictResponse(BaseModel):
    label: str
    confidence: float
    ai_probability: float
    threshold: float
    artifact_paths: Dict[str, Optional[str]]


class DemoExecuteRequest(BaseModel):
    detector: str
    text: str
    config: Dict[str, Any] = {}
    hf_endpoint: Optional[str] = None


@router.get("/detectors", response_model=DetectorListResponse)
async def list_demo_detectors():
    """List demo detectors from detect templates."""
    try:
        return DetectorListResponse(detectors=demo_service.list_detectors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/{detector_name}", response_model=TemplateResponse)
async def get_demo_template(detector_name: str):
    """Load demo template (reused from detect template)."""
    try:
        return TemplateResponse(template=demo_service.load_template(detector_name))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Detector template not found: {detector_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ExecuteResponse)
async def execute_demo(request: DemoExecuteRequest):
    """Execute demo detection asynchronously with log streaming."""
    try:
        if not request.text or not str(request.text).strip():
            raise HTTPException(status_code=400, detail="Text is required")

        cfg, request_id, data_path, out_dir = demo_service.prepare_demo_config(
            detector=request.detector,
            text=str(request.text),
            config_overrides=request.config or {},
            hf_endpoint=request.hf_endpoint,
        )

        job_id = process_manager.create_job("demo", cfg)
        demo_service.store_job_meta(job_id, cfg, data_path, out_dir)

        async def log_callback(jid: str, message: str, level: str):
            # Save log for later run_dir extraction
            demo_service.add_job_log(job_id, level, message)
            # Send to WebSocket
            await ws_manager.send_log(job_id, message, level)

        async def run_job():
            try:
                success, exit_code = await command_executor.execute_command(
                    job_id, "detect", cfg, log_callback
                )
                process_manager.complete_job(job_id, success, exit_code=exit_code)
                await ws_manager.send_completion(job_id, success, exit_code)
            except asyncio.CancelledError:
                return
            except Exception as e:
                process_manager.complete_job(job_id, False, error_message=str(e))
                await ws_manager.send_log(job_id, f"Job failed: {str(e)}", "error")
                await ws_manager.send_completion(job_id, False, 1)

        task = asyncio.create_task(run_job())
        process_manager.start_job(job_id, task)

        return ExecuteResponse(job_id=job_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{job_id}", response_model=DemoPredictResponse)
async def get_demo_result(job_id: str):
    """Parse and return demo prediction result after job completes."""
    try:
        result = demo_service.parse_result(job_id)
        return DemoPredictResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=DemoPredictResponse)
async def demo_predict(request: DemoPredictRequest):
    """Run single-text detector prediction for demo usage (synchronous)."""
    try:
        if not request.text or not str(request.text).strip():
            raise HTTPException(status_code=400, detail="Text is required")
        result = await demo_service.predict(
            detector=request.detector,
            text=str(request.text),
            config_overrides=request.config or {},
            hf_endpoint=request.hf_endpoint,
        )
        return DemoPredictResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
