"""Pydantic request models"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class ExecuteRequest(BaseModel):
    """Generic request for executing a command"""
    config: Dict[str, Any]


class ValidateRequest(BaseModel):
    """Request for validating a configuration"""
    config: Dict[str, Any]


class DownloadFileRequest(BaseModel):
    """Request for downloading a file from URL"""
    url: str
    destination: Optional[str] = None
