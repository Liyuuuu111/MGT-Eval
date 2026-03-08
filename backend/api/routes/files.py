"""File API endpoints"""

from pathlib import Path

import aiofiles
import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.models.requests import DownloadFileRequest
from backend.models.responses import (
    DatasetUploadResponse,
    DownloadFileResponse,
    FileUploadResponse,
)
from backend.services.file_lifecycle_service import file_lifecycle_service

router = APIRouter()

# Auto-detect project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_SIZE_LABEL = "10MB"


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file"""
    file_path: Path | None = None
    try:
        # Create upload directory
        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_dir / file.filename
        file_size = 0
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_LABEL}.",
                    )
                await f.write(chunk)

        return FileUploadResponse(
            file_path=str(file_path),
            file_size=file_size
        )

    except HTTPException:
        try:
            if file_path is not None:
                file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        try:
            await file.close()
        except Exception:
            pass


@router.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset_file(
    file: UploadFile = File(...),
    phase: str | None = Form(default=None),
):
    """Upload dataset into managed runtime storage."""
    try:
        payload = await file_lifecycle_service.save_uploaded_dataset(file, phase=phase)
        return DatasetUploadResponse(**payload)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Managed dataset upload failed: {str(e)}")


@router.post("/download", response_model=DownloadFileResponse)
async def download_file(request: DownloadFileRequest):
    """Download a file from URL"""
    try:
        # Create download directory
        download_dir = PROJECT_ROOT / "data" / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = request.destination or request.url.split("/")[-1]
        if not filename:
            filename = "downloaded_file"

        file_path = download_dir / filename

        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(request.url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file: HTTP {response.status}"
                    )

                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

        file_size = file_path.stat().st_size

        return DownloadFileResponse(
            file_path=str(file_path),
            file_size=file_size
        )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/generated/{token}")
async def download_generated_file(token: str):
    """Download a generated file by short-lived token."""
    resolved = file_lifecycle_service.resolve_download_token(token)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Download link is invalid or expired")

    file_path = Path(resolved.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated file not found")

    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name,
    )
