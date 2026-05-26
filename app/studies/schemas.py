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
from pydantic import ConfigDict


InputType = Literal["dicom", "dicomdir", "nifti", "unknown"]
StudyStatus = Literal["imported", "prepared"]


class StudyFileRead(BaseModel):
    filename: str
    relative_path: str
    size_bytes: int


class StudyListItem(BaseModel):
    id: str = Field(..., description="Study identifier")
    status: StudyStatus
    input_type: InputType
    files_count: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class StudyPreparedVolumeManifestRead(BaseModel):
    filename: str
    relative_path: str
    metadata_path: str


class StudyRead(StudyListItem):
    source_files: list[StudyFileRead]
    prepared_volume: StudyPreparedVolumeManifestRead | None = None


class StudyListResponse(BaseModel):
    items: list[StudyListItem]


class StudyImportResponse(StudyListItem):
    status: Literal["imported"]
    source_files: list[StudyFileRead]


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


class VolumeIntensityRead(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    p1: float
    p5: float
    p95: float
    p99: float


class VolumeMetadataRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    shape: list[int]
    spacing: list[float]
    origin: list[float] | None = None
    direction: list[float] | None = None
    intensity: VolumeIntensityRead


class PreparedVolumeRead(BaseModel):
    filename: str
    relative_path: str
    url: str
    metadata: VolumeMetadataRead


class StudyVolumeRead(BaseModel):
    study_id: str
    status: str
    volume: PreparedVolumeRead


class StudyPrepareResponse(StudyVolumeRead):
    pass
