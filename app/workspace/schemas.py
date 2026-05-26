# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : schemas.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.ai.schemas import AiModuleRead, AiRunRead
from app.analyses.schemas import AnalysisRead
from app.segmentations.schemas import SegmentationRead
from app.studies.schemas import InputType, StudyStatus, StudyVolumeRead


class WorkspaceStudyRead(BaseModel):
    id: str
    status: StudyStatus
    input_type: InputType
    files_count: int
    created_at: datetime
    updated_at: datetime


class WorkspaceVolumeRead(BaseModel):
    is_prepared: bool
    data: StudyVolumeRead | None


class WorkspaceAiRead(BaseModel):
    modules: list[AiModuleRead]
    runs: list[AiRunRead]


class WorkspaceCollectionRead(BaseModel):
    items: list[Any]
    latest: Any | None


class WorkspaceAvailableActionsRead(BaseModel):
    can_prepare_volume: bool
    can_create_ai_run: bool
    can_execute_ai: bool
    can_publish_segmentation: bool
    can_run_label_hu_statistics: bool


class StudyWorkspaceRead(BaseModel):
    study: WorkspaceStudyRead
    viewer: Any | None
    volume: WorkspaceVolumeRead
    ai: WorkspaceAiRead
    segmentations: WorkspaceCollectionRead
    analyses: WorkspaceCollectionRead
    available_actions: WorkspaceAvailableActionsRead
