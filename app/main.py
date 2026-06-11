# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : main.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.router import router as ai_router
from app.analyses.router import router as analyses_router
from app.core.config import get_cors_allow_origins, get_settings
from app.measurements.router import router as measurements_router
from app.segmentations.router import router as segmentations_router
from app.studies.router import router as studies_router
from app.workspace.router import router as workspace_router


settings = get_settings()
app = FastAPI(title="Hekia CT Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(settings),
    allow_origin_regex=settings.cors_allow_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    studies_router,
    ai_router,
    segmentations_router,
    analyses_router,
    measurements_router,
    workspace_router,
):
    app.include_router(router)
    app.include_router(router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return health()
