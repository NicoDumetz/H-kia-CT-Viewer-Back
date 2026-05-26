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


class AnalysisCreate(BaseModel):
    module_id: str
    segmentation_id: str
    label_ids: list[int] | None = None
    roi_mode: Literal["whole_label"]


class AnalysisInputRead(BaseModel):
    volume_path: str
    segmentation_id: str
    segmentation_path: str
    label_ids: list[int] | None
    roi_mode: str


class AnalysisArtifactRead(BaseModel):
    type: str
    name: str
    relative_path: str
    url: str


class AnalysisOutputRead(BaseModel):
    result_path: str
    artifacts: list[AnalysisArtifactRead]


class AnalysisRead(BaseModel):
    id: str
    study_id: str
    module_id: str
    status: Literal["succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    input: AnalysisInputRead
    output: AnalysisOutputRead | None
    error: str | None


class AnalysisListResponse(BaseModel):
    items: list[AnalysisRead]


class LabelHuStatsRead(BaseModel):
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None
    p1: float | None
    p5: float | None
    p25: float | None
    p75: float | None
    p95: float | None
    p99: float | None


class LabelHuResultRead(BaseModel):
    label_id: int
    name: str
    voxel_count: int
    volume_mm3: float
    hu: LabelHuStatsRead


class AnalysisResultRead(BaseModel):
    id: str
    study_id: str
    module_id: str
    segmentation_id: str
    roi_mode: str
    labels_count: int
    labels: list[LabelHuResultRead]
