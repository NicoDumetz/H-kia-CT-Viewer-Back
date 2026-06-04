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
# Created     : Thursday June 04 2026
#
# =============================================================

from typing import Literal

from pydantic import BaseModel


MeasurementPlane = Literal["axial", "sagittal", "coronal"]


class HuCircleMeasurementCreate(BaseModel):
    plane: MeasurementPlane
    center_world: list[float]
    edge_world: list[float] | None = None
    radius_mm: float | None = None


class HuCircleStatsRead(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p95: float


class HuCircleMeasurementRead(BaseModel):
    plane: MeasurementPlane
    center_world: list[float]
    radius_mm: float
    voxel_count: int
    area_mm2: float
    hu: HuCircleStatsRead
