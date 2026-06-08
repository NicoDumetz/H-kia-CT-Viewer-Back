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
from typing import Literal

from pydantic import BaseModel


AiRunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
AiModuleRunner = Literal["nnunet", "internal"]


class AiModuleNnunetConfig(BaseModel):
    dataset: str | None = None
    configuration: str | None = None
    fold: str | None = None
    checkpoint: str | None = None
    device: str | None = None


class AiModuleRead(BaseModel):
    id: str
    name: str
    task_type: str
    description: str
    input_type: str
    output_type: str
    is_available: bool
    availability_error: str | None = None
    runner: AiModuleRunner
    labels: list[str] | None = None


class AiModuleDefinition(AiModuleRead):
    nnunet: AiModuleNnunetConfig | None = None


class AiModuleListResponse(BaseModel):
    items: list[AiModuleRead]


class AiRunCreate(BaseModel):
    module_id: str


class AiRunInputRead(BaseModel):
    prepared_volume_path: str


class AiRunArtifactRead(BaseModel):
    type: str
    name: str
    relative_path: str
    url: str


class AiRunOutputRead(BaseModel):
    result_path: str
    artifacts: list[AiRunArtifactRead]


class AiRunRead(BaseModel):
    id: str
    study_id: str
    module_id: str
    module_name: str
    status: AiRunStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    input: AiRunInputRead
    output: AiRunOutputRead | None
    error: str | None
    error_detail: str | None = None


class AiRunListResponse(BaseModel):
    items: list[AiRunRead]
