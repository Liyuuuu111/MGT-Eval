"""System API endpoints for GPU and model detection"""

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
import psutil
from backend.services.system_service import system_service
from backend.services.process_manager import process_manager, JobStatus
from backend.api.websocket.logs import manager as ws_manager
from backend.services.result_service import result_service
from backend.services.metadata_service import metadata_service


router = APIRouter()


class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total: str
    memory_free: str
    utilization: str
    available: bool


class ModelInfo(BaseModel):
    name: str
    path: str
    size: str


class CalibratorInfo(BaseModel):
    name: str
    path: str
    size: str


class GPUListResponse(BaseModel):
    gpus: List[GPUInfo]
    recommended_gpu: Optional[int]


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class CalibratorListResponse(BaseModel):
    calibrators: List[CalibratorInfo]


class CalibratorThresholdPreset(BaseModel):
    key: str
    label: str
    threshold: float
    source: str
    tpr: Optional[float] = None
    fpr: Optional[float] = None
    target_fpr: Optional[float] = None
    acc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None


class CalibratorThresholdsResponse(BaseModel):
    path: str
    presets: List[CalibratorThresholdPreset]
    default_threshold: Optional[float] = None


class CancelJobResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class HFDownloadItem(BaseModel):
    path: str
    size_bytes: int
    total_bytes: Optional[int] = None
    percent: Optional[float] = None
    model: Optional[str] = None
    mtime: Optional[str] = None


class HFDownloadStatusResponse(BaseModel):
    cache_dir: str
    active: bool
    downloads: List[HFDownloadItem]
    total_downloaded_bytes: int
    total_expected_bytes: Optional[int] = None
    timestamp: str


class JobResultResponse(BaseModel):
    job_id: str
    command: str
    status: str
    exit_code: Optional[int] = None
    artifacts: Dict[str, Optional[str]]
    result: Dict[str, Any]


class DetectorMetadataItem(BaseModel):
    key: str
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    description_en: Optional[str] = None
    description_zh: Optional[str] = None
    paper: Optional[str] = None
    authors: Optional[str] = None
    venue: Optional[str] = None
    link: Optional[str] = None


class DetectorMetadataResponse(BaseModel):
    detectors: List[DetectorMetadataItem]


class GPUMonitorInfo(BaseModel):
    index: int
    name: str
    utilization: float
    memory_used_mb: float
    memory_total_mb: float
    temperature: float


class SystemMonitorResponse(BaseModel):
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpus: List[GPUMonitorInfo]


@router.get("/gpus", response_model=GPUListResponse)
async def get_gpus():
    """Get list of available GPUs"""
    try:
        gpus = system_service.detect_gpus()
        recommended = system_service.get_recommended_gpu()
        return GPUListResponse(gpus=gpus, recommended_gpu=recommended)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def get_local_models(custom_dirs: Optional[str] = None):
    """
    Get list of locally cached models

    Args:
        custom_dirs: Comma-separated list of custom directories to scan
    """
    try:
        dirs = None
        if custom_dirs:
            dirs = [d.strip() for d in custom_dirs.split(',')]

        models = system_service.detect_local_models(dirs)
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibrators", response_model=CalibratorListResponse)
async def get_calibrators(custom_dirs: Optional[str] = None):
    """
    Get list of detected calibrator files.

    Args:
        custom_dirs: Comma-separated list of additional directories to scan
    """
    try:
        dirs = None
        if custom_dirs:
            dirs = [d.strip() for d in custom_dirs.split(',') if d.strip()]

        calibrators = system_service.detect_calibrators(dirs)
        return CalibratorListResponse(calibrators=calibrators)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibrator-thresholds", response_model=CalibratorThresholdsResponse)
async def get_calibrator_thresholds(path: str):
    """Get threshold presets parsed from a calibrator file or directory."""
    try:
        return system_service.get_calibrator_thresholds(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Health check endpoint"""
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())


@router.get("/hf-downloads", response_model=HFDownloadStatusResponse)
async def get_hf_downloads():
    """Get Hugging Face download status from cache"""
    try:
        return system_service.get_hf_download_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job-result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Get resolved artifacts and summarized result by job id."""
    try:
        return result_service.get_job_result(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detector-metadata", response_model=DetectorMetadataResponse)
async def get_detector_metadata():
    """Get detector metadata extracted from source definitions."""
    try:
        rows = metadata_service.get_detector_metadata()
        return DetectorMetadataResponse(detectors=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{job_id}", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    """Cancel a running job"""
    job = process_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        return CancelJobResponse(status=job.status.value)

    process_manager.cancel_job(job_id)
    await ws_manager.send_completion(job_id, False, exit_code=-1)
    return CancelJobResponse(status="cancelled")


@router.get("/monitor", response_model=SystemMonitorResponse)
async def get_system_monitor():
    """Get real-time system resource usage (CPU, memory, GPU)"""
    try:
        # Get CPU and memory stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)

        # Get GPU stats using robust fallback in system_service.
        gpu_stats = system_service.detect_gpu_monitor_stats()
        gpus = [GPUMonitorInfo(**row) for row in gpu_stats]

        return SystemMonitorResponse(
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_percent=memory_percent,
            memory_used_gb=round(memory_used_gb, 2),
            memory_total_gb=round(memory_total_gb, 2),
            gpus=gpus,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
