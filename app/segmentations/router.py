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
# Created     : Tuesday May 26 2026
#
# =============================================================

from typing import Annotated

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import Settings, get_settings
from app.segmentations.labels import normalize_label_names
from app.segmentations.schemas import (
    SegmentationLabelsDocumentRead,
    SegmentationListResponse,
    SegmentationRead,
)
from app.segmentations.service import (
    SegmentationError,
    get_study_segmentation,
    get_study_segmentation_labels,
    list_study_segmentations,
    publish_run_segmentation,
    upload_manual_segmentation,
)


router = APIRouter(tags=["segmentations"])


@router.post(
    "/studies/{study_id}/ai-runs/{run_id}/publish-segmentation",
    response_model=SegmentationRead,
)
def publish_run_segmentation_endpoint(
    study_id: str,
    run_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SegmentationRead:
    try:
        return publish_run_segmentation(study_id, run_id, settings)
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/studies/{study_id}/segmentations/upload", response_model=SegmentationRead)
async def upload_manual_segmentation_endpoint(
    study_id: str,
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    name: Annotated[str | None, Form()] = None,
    source: Annotated[str, Form()] = "manual_upload",
) -> SegmentationRead:
    try:
        return await upload_manual_segmentation(study_id, file, name, source, settings)
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/studies/{study_id}/segmentations/manual", response_model=SegmentationRead)
async def upload_manual_mask_endpoint(
    study_id: str,
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    name: Annotated[str | None, Form()] = None,
    labels_json: Annotated[str | None, Form()] = None,
    labels: Annotated[UploadFile | None, File()] = None,
) -> SegmentationRead:
    try:
        labels_payload = await read_uploaded_labels(labels, labels_json)
        return await upload_manual_segmentation(
            study_id,
            file,
            name,
            "manual_upload",
            settings,
            labels_payload=labels_payload,
        )
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/segmentations", response_model=SegmentationListResponse)
def list_study_segmentations_endpoint(
    study_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SegmentationListResponse:
    try:
        return list_study_segmentations(study_id, settings)
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/studies/{study_id}/segmentations/{segmentation_id}", response_model=SegmentationRead)
def get_study_segmentation_endpoint(
    study_id: str,
    segmentation_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SegmentationRead:
    try:
        return get_study_segmentation(study_id, segmentation_id, settings)
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get(
    "/studies/{study_id}/segmentations/{segmentation_id}/labels",
    response_model=SegmentationLabelsDocumentRead,
)
def get_study_segmentation_labels_endpoint(
    study_id: str,
    segmentation_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SegmentationLabelsDocumentRead:
    try:
        return get_study_segmentation_labels(study_id, segmentation_id, settings)
    except SegmentationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


async def read_uploaded_labels(
    labels: UploadFile | None,
    labels_json: str | None,
) -> dict[int, str] | None:
    if labels is not None:
        content = await labels.read()
        await labels.seek(0)
        return parse_labels_json(content.decode("utf-8"))

    if labels_json:
        return parse_labels_json(labels_json)

    return None


def parse_labels_json(content: str) -> dict[int, str]:
    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SegmentationError(f"Invalid labels.json: {exc}") from exc

    return normalize_label_names(parsed)
