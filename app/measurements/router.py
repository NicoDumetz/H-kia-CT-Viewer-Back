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
# Created     : Thursday June 04 2026
#
# =============================================================

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.measurements.schemas import HuCircleMeasurementCreate, HuCircleMeasurementRead
from app.measurements.service import MeasurementError, compute_hu_circle_measurement


router = APIRouter(tags=["measurements"])


@router.post(
    "/studies/{study_id}/measurements/hu-circle",
    response_model=HuCircleMeasurementRead,
)
def create_hu_circle_measurement_endpoint(
    study_id: str,
    payload: HuCircleMeasurementCreate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HuCircleMeasurementRead:
    try:
        return compute_hu_circle_measurement(study_id, payload, settings)
    except MeasurementError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
