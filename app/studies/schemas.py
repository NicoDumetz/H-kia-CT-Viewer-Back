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
from typing import Any, Literal

from pydantic import BaseModel, Field


InputType = Literal["dicom", "dicomdir", "nifti", "unknown"]


class StudyFileRead(BaseModel):
    filename: str
    relative_path: str
    size_bytes: int


class StudyListItem(BaseModel):
    id: str = Field(..., description="Study identifier")
    status: Literal["imported"]
    input_type: InputType
    files_count: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class StudyRead(StudyListItem):
    source_files: list[StudyFileRead]


class StudyListResponse(BaseModel):
    items: list[StudyListItem]


class StudyImportResponse(StudyRead):
    pass
