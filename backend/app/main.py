"""NYELETI AI API.

    uvicorn backend.app.main:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import anomalies, areas, calendar, fires, harmonization, model, reports, trends
from backend.app.database import db
from ml import config

app = FastAPI(
    title="NYELETI AI",
    description="MODIS-VIIRS fire data harmonization and burning activity calendar, built on NASA FIRMS data.",
    version="1.0.0",
)

origins = os.environ.get("NYELETI_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)

for module in (areas, fires, calendar, harmonization, anomalies, trends, model, reports):
    app.include_router(module.router)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    ready = config.DB_PATH.exists()
    return {
        "status": "ok" if ready else "database missing",
        "database": str(config.DB_PATH.name),
        "active_model": db.metadata()["active_model"] if ready else None,
    }


# When the frontend has been built (npm run build) the API serves it too, so a
# single process is enough for a demo.
_dist = config.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (_dist / path).resolve()
        if path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
