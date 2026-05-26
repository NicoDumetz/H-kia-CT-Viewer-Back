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


class ViewerFileRead(BaseModel):
    filename: str
    relative_path: str
    url: str


class ViewerNiftiRead(ViewerFileRead):
    metadata: dict[str, Any]


class ViewerDicomImageRead(ViewerFileRead):
    image_id: str
    instance_number: int | None
    slice_location: float | None
    image_position_patient: list[float] | None


class ViewerDicomSeriesRead(BaseModel):
    series_instance_uid: str | None
    study_instance_uid: str | None
    modality: str | None
    series_description: str | None
    protocol_name: str | None
    manufacturer: str | None
    files_count: int
    rows: int | None
    columns: int | None
    slice_thickness: float | None
    pixel_spacing: list[float] | None
    images: list[ViewerDicomImageRead]


class ViewerDicomRead(BaseModel):
    series: list[ViewerDicomSeriesRead]


class StudyViewerRead(BaseModel):
    study_id: str
    input_type: InputType
    status: Literal["ready"]
    nifti: ViewerNiftiRead | None
    dicom: ViewerDicomRead | None
