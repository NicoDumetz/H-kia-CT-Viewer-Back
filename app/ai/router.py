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

from app.ai.schemas import (
    AiModuleListResponse,
    AiRunCreate,
    AiRunListResponse,
    AiRunRead,
)
from app.ai.service import (
    AiRunError,
    create_ai_run,
    execute_ai_run,
    get_study_ai_run,
    list_available_ai_modules,
    list_study_ai_runs,
    simulate_ai_run,
)
from app.core.config import Settings, get_settings


router = APIRouter(tags=["ai"])


@router.get("/ai/modules", response_model=AiModuleListResponse)
def list_ai_modules_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiModuleListResponse:
    return list_available_ai_modules(settings)


@router.post("/studies/{study_id}/ai-runs", response_model=AiRunRead)
def create_ai_run_endpoint(
    study_id: str,
    payload: AiRunCreate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiRunRead:
    try:
        return create_ai_run(study_id, payload, settings)
    except AiRunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/ai-runs", response_model=AiRunListResponse)
def list_study_ai_runs_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiRunListResponse:
    try:
        return list_study_ai_runs(study_id, settings)
    except AiRunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/ai-runs/{run_id}", response_model=AiRunRead)
def get_study_ai_run_endpoint(
    study_id: str,
    run_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiRunRead:
    try:
        return get_study_ai_run(study_id, run_id, settings)
    except AiRunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/studies/{study_id}/ai-runs/{run_id}/simulate", response_model=AiRunRead)
def simulate_ai_run_endpoint(
    study_id: str,
    run_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiRunRead:
    try:
        return simulate_ai_run(study_id, run_id, settings)
    except AiRunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/studies/{study_id}/ai-runs/{run_id}/execute", response_model=AiRunRead)
def execute_ai_run_endpoint(
    study_id: str,
    run_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AiRunRead:
    try:
        return execute_ai_run(study_id, run_id, settings)
    except AiRunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
