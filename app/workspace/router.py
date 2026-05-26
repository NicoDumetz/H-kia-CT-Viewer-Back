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

from app.core.config import Settings, get_settings
from app.workspace.schemas import StudyWorkspaceRead
from app.workspace.service import WorkspaceError, get_study_workspace


router = APIRouter(tags=["workspace"])


@router.get("/studies/{study_id}/workspace", response_model=StudyWorkspaceRead)
def get_study_workspace_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StudyWorkspaceRead:
    try:
        return get_study_workspace(study_id, settings)
    except WorkspaceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
