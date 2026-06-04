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
from app.measurements.router import router as measurements_router
from app.segmentations.router import router as segmentations_router
from app.studies.router import router as studies_router
from app.workspace.router import router as workspace_router


app = FastAPI(title="Hekia CT Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(studies_router)
app.include_router(ai_router)
app.include_router(segmentations_router)
app.include_router(analyses_router)
app.include_router(measurements_router)
app.include_router(workspace_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
