"""File API endpoints"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.models.requests import DownloadFileRequest
from backend.models.responses import FileUploadResponse, DownloadFileResponse
from pathlib import Path
import aiofiles
import aiohttp
import os

router = APIRouter()

# Auto-detect project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file"""
    try:
        # Create upload directory
        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_dir / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        file_size = os.path.getsize(file_path)

        return FileUploadResponse(
            file_path=str(file_path),
            file_size=file_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


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

        file_size = os.path.getsize(file_path)

        return DownloadFileResponse(
            file_path=str(file_path),
            file_size=file_size
        )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
