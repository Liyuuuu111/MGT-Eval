"""FastAPI main application"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load env vars from project .env before importing services/routes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _is_writable_dir(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".mgt_eval_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _normalize_env_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return Path(value).expanduser()


def _ensure_writable_file_env(
    env_key: str,
    fallback_candidates: Iterable[Path],
) -> None:
    configured = _normalize_env_path(os.getenv(env_key))
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend([p.expanduser() for p in fallback_candidates])

    for candidate in candidates:
        if _is_writable_dir(candidate.parent):
            os.environ[env_key] = str(candidate)
            return


def _ensure_writable_dir_env(
    env_key: str,
    fallback_candidates: Iterable[Path],
) -> None:
    configured = _normalize_env_path(os.getenv(env_key))
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend([p.expanduser() for p in fallback_candidates])

    for candidate in candidates:
        if _is_writable_dir(candidate):
            os.environ[env_key] = str(candidate)
            return


def _configure_runtime_env() -> None:
    # Compatibility with UncoverAI deployments: honor PUBLIC_AUTH_DB_PATH when present.
    _ensure_writable_file_env(
        "PUBLIC_AUTH_DB_PATH",
        [
            Path.home() / ".uncoverai" / "uncoverai_public.db",
            PROJECT_ROOT / ".runtime" / "uncoverai_public.db",
        ],
    )

    # Backend temp yaml directory.
    _ensure_writable_dir_env(
        "MGT_EVAL_BACKEND_TMP_DIR",
        [
            Path(tempfile.gettempdir()) / "mgt_eval_backend",
            Path.home() / ".mgt_eval" / "backend_tmp",
            PROJECT_ROOT / ".runtime" / "backend_tmp",
        ],
    )
    if os.getenv("MGT_EVAL_BACKEND_TMP_DIR"):
        os.environ.setdefault("UNCOVERAI_BACKEND_TMP_DIR", os.environ["MGT_EVAL_BACKEND_TMP_DIR"])

    # Demo request temporary directory.
    _ensure_writable_dir_env(
        "MGT_EVAL_DEMO_TMP_ROOT",
        [
            Path(tempfile.gettempdir()) / "mgt_eval_backend_demo",
            Path.home() / ".mgt_eval" / "demo_tmp",
            PROJECT_ROOT / ".runtime" / "demo_tmp",
        ],
    )

    # Runtime model/cache directory.
    _ensure_writable_dir_env(
        "MGT_EVAL_RUNTIME_CACHE_ROOT",
        [
            Path(tempfile.gettempdir()) / "mgt_eval_cache",
            Path.home() / ".cache" / "mgt_eval_cache",
            PROJECT_ROOT / ".runtime" / "cache",
        ],
    )


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000", "http://localhost:5173"]


_configure_runtime_env()

# Import routers
from backend.api.routes import build, attack, train, detect, files, system, demo
from backend.api.websocket import logs

# Create FastAPI app
app = FastAPI(
    title="MGT Eval API",
    description="API for Machine Generated Text Evaluation",
    version="0.1.0"
)

# CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(build.router, prefix="/api/build", tags=["build"])
app.include_router(attack.router, prefix="/api/attack", tags=["attack"])
app.include_router(train.router, prefix="/api/train", tags=["train"])
app.include_router(detect.router, prefix="/api/detect", tags=["detect"])
app.include_router(demo.router, prefix="/api/demo", tags=["demo"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(logs.router, prefix="/api/ws")


# Serve static frontend files (production)
static_dir = Path(__file__).parent / "static"
if static_dir.exists() and (static_dir / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(static_dir / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # For SPA routing, always return index.html
        if not full_path.startswith("api") and not full_path.startswith("ws"):
            return FileResponse(static_dir / "index.html")
else:
    @app.get("/")
    async def root():
        return {
            "message": "MGT Eval API",
            "version": "0.1.0",
            "docs": "/docs"
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
