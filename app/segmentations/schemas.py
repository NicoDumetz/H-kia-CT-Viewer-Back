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


class SegmentationFileRead(BaseModel):
    filename: str
    relative_path: str
    url: str


class SegmentationBBoxRead(BaseModel):
    min: list[int]
    max: list[int]


class SegmentationLabelRead(BaseModel):
    label_id: int
    name: str
    voxel_count: int
    volume_mm3: float
    bbox_ijk: SegmentationBBoxRead
    center_ijk: list[float]


class SegmentationMetadataRead(BaseModel):
    shape: list[int]
    spacing: list[float]
    labels_count: int
    labels: list[SegmentationLabelRead]


class SegmentationRead(BaseModel):
    id: str
    study_id: str
    source_run_id: str
    module_id: str
    module_name: str
    status: Literal["ready"]
    created_at: datetime
    file: SegmentationFileRead
    metadata: SegmentationMetadataRead


class SegmentationListResponse(BaseModel):
    items: list[SegmentationRead]
