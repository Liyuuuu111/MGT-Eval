"""Pydantic response models"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ExecuteResponse(BaseModel):
    """Response for command execution"""
    job_id: str
    status: str = "started"


class ValidateResponse(BaseModel):
    """Response for configuration validation"""
    valid: bool
    errors: List[str] = []


class TemplateResponse(BaseModel):
    """Response containing a YAML template"""
    template: Dict[str, Any]


class DetectorListResponse(BaseModel):
    """Response containing list of detectors"""
    detectors: List[str]


class AttacksResponse(BaseModel):
    """Response containing all attack types"""
    attacks: Dict[str, Any]


class FileUploadResponse(BaseModel):
    """Response for file upload"""
    file_path: str
    file_size: int


class DownloadFileResponse(BaseModel):
    """Response for file download"""
    file_path: str
    file_size: int


class DatasetUploadResponse(BaseModel):
    """Response for managed dataset upload."""
    upload_id: str
    file_name: str
    file_size: int
    stored_path: str
    phase: str


class JobStatusResponse(BaseModel):
    """Response for job status"""
    job_id: str
    command: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
