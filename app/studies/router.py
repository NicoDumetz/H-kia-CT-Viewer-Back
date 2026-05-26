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
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.studies.schemas import (
    StudyImportResponse,
    StudyListResponse,
    StudyPrepareResponse,
    StudyRead,
    StudyViewerRead,
    StudyVolumeRead,
)
from app.studies.service import (
    get_volume,
    get_study,
    get_study_file_path,
    get_study_viewer,
    import_study,
    list_studies,
    prepare_volume,
)
from app.studies.volume import VolumeError


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


@router.post("/{study_id}/prepare", response_model=StudyPrepareResponse)
def prepare_study_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyPrepareResponse:
    try:
        response = prepare_volume(study_id, settings)
    except VolumeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study not found.",
        )

    return StudyPrepareResponse.model_validate(response.model_dump())


@router.get("/{study_id}/volume", response_model=StudyVolumeRead)
def get_study_volume_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyVolumeRead:
    try:
        response = get_volume(study_id, settings)
    except VolumeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study not found.",
        )

    return response


@router.get("/{study_id}/viewer", response_model=StudyViewerRead)
def get_study_viewer_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyViewerRead:
    viewer = get_study_viewer(study_id, settings)

    if viewer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study not found.",
        )

    return viewer


@router.get("/{study_id}/files/{relative_path:path}", response_class=FileResponse)
def get_study_file_endpoint(
    study_id: str,
    relative_path: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    return build_study_file_response(study_id, relative_path, settings)


@router.head("/{study_id}/files/{relative_path:path}", response_class=FileResponse)
def head_study_file_endpoint(
    study_id: str,
    relative_path: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    return build_study_file_response(study_id, relative_path, settings)


def build_study_file_response(
    study_id: str,
    relative_path: str,
    settings: Settings,
) -> FileResponse:
    file_path = get_study_file_path(study_id, relative_path, settings)

    if file_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )

    return FileResponse(file_path)


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
