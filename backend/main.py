"""FastAPI main application"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
