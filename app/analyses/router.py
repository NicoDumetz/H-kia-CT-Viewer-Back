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

from fastapi import APIRouter, Depends, HTTPException

from app.analyses.schemas import (
    AnalysisCreate,
    AnalysisListResponse,
    AnalysisRead,
    AnalysisResultRead,
)
from app.analyses.service import (
    AnalysisError,
    create_analysis,
    get_study_analysis,
    get_study_analysis_result,
    list_study_analyses,
)
from app.core.config import Settings, get_settings


router = APIRouter(tags=["analyses"])


@router.post("/studies/{study_id}/analyses", response_model=AnalysisRead)
def create_analysis_endpoint(
    study_id: str,
    payload: AnalysisCreate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisRead:
    try:
        return create_analysis(study_id, payload, settings)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/analyses", response_model=AnalysisListResponse)
def list_study_analyses_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisListResponse:
    try:
        return list_study_analyses(study_id, settings)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/analyses/{analysis_id}/result", response_model=AnalysisResultRead)
def get_study_analysis_result_endpoint(
    study_id: str,
    analysis_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisResultRead:
    try:
        return get_study_analysis_result(study_id, analysis_id, settings)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/analyses/{analysis_id}", response_model=AnalysisRead)
def get_study_analysis_endpoint(
    study_id: str,
    analysis_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisRead:
    try:
        return get_study_analysis(study_id, analysis_id, settings)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
