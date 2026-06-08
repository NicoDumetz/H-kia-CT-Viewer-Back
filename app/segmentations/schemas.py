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


SegmentationSource = Literal["ai", "manual"]
SegmentationStatus = Literal["ready", "rejected", "failed"]


class SegmentationFileRead(BaseModel):
    filename: str
    relative_path: str
    url: str


class SegmentationBBoxRead(BaseModel):
    min: list[int]
    max: list[int]


class SegmentationLabelRead(BaseModel):
    id: int = 0
    label_id: int
    name: str
    group: str = "other"
    present: bool = True
    voxel_count: int
    volume_mm3: float
    color: str = "#CCCCCC"
    opacity: float = 0.45
    bbox_ijk: SegmentationBBoxRead
    center_ijk: list[float]


class SegmentationMetadataRead(BaseModel):
    shape: list[int]
    spacing: list[float]
    labels_count: int
    present_labels_count: int = 0
    labels_path: str | None = None
    labels_url: str | None = None
    labels: list[SegmentationLabelRead]


class SegmentationLabelsDocumentRead(BaseModel):
    segmentation_id: str
    source: SegmentationSource
    model_id: str | None = None
    labels_count: int
    present_labels_count: int
    labels: list[SegmentationLabelRead]


class SegmentationRead(BaseModel):
    id: str
    study_id: str
    source: SegmentationSource = "manual"
    source_run_id: str | None
    module_id: str
    module_name: str
    model_id: str | None = None
    status: SegmentationStatus
    created_at: datetime
    file: SegmentationFileRead
    labels_file: SegmentationFileRead | None = None
    metadata: SegmentationMetadataRead


class SegmentationListResponse(BaseModel):
    items: list[SegmentationRead]
