# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : router.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.studies.schemas import StudyImportResponse, StudyListResponse, StudyRead
from app.studies.service import get_study, import_study, list_studies


router = APIRouter(prefix="/studies", tags=["studies"])


@router.get("", response_model=StudyListResponse)
def list_studies_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyListResponse:
    return list_studies(settings)


@router.post("/import", response_model=StudyImportResponse)
async def import_study_endpoint(
    files: Annotated[list[UploadFile], File()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyImportResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    return await import_study(files, settings)


@router.get("/{study_id}", response_model=StudyRead)
def get_study_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyRead:
    study = get_study(study_id, settings)

    if study is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study not found.",
        )

    return study
