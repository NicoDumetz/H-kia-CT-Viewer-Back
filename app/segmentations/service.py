# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : service.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ai.schemas import AiRunArtifactRead, AiRunRead
from app.ai.storage import get_ai_run_dir, read_ai_run
from app.core.config import Settings
from app.segmentations.analysis import SegmentationAnalysisError, compute_segmentation_metadata
from app.segmentations.labels import load_label_names
from app.segmentations.schemas import (
    SegmentationFileRead,
    SegmentationListResponse,
    SegmentationRead,
)
from app.segmentations.storage import (
    SEGMENTATION_FILENAME,
    create_segmentation_dir,
    get_segmentation_dir,
    list_segmentation_dirs,
    read_segmentation_metadata,
    write_segmentation_metadata,
)
from app.studies.manifest import read_manifest
from app.studies.storage import get_study_dir


class SegmentationError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def publish_run_segmentation(
    study_id: str,
    run_id: str,
    settings: Settings,
) -> SegmentationRead:
    study_dir = get_existing_study_dir(settings, study_id)
    run = get_existing_run(settings, study_id, run_id)

    if run.status != "succeeded":
        raise SegmentationError("AI run must be succeeded before publishing segmentation")

    artifact = get_segmentation_artifact(run)
    source_path = study_dir / artifact.relative_path

    if not source_path.is_file():
        raise SegmentationError("Segmentation artifact file not found")

    segmentation_id = str(uuid4())
    segmentation_dir = create_segmentation_dir(settings.storage_root, study_id, segmentation_id)
    destination_path = segmentation_dir / SEGMENTATION_FILENAME
    relative_path = f"derived/segmentations/{segmentation_id}/{SEGMENTATION_FILENAME}"

    shutil.copy2(source_path, destination_path)

    try:
        metadata = compute_segmentation_metadata(destination_path, load_label_names(settings))
    except SegmentationAnalysisError as exc:
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        raise SegmentationError(exc.message) from exc

    segmentation = SegmentationRead(
        id=segmentation_id,
        study_id=study_id,
        source_run_id=run.id,
        module_id=run.module_id,
        module_name=run.module_name,
        status="ready",
        created_at=datetime.now(timezone.utc),
        file=SegmentationFileRead(
            filename=SEGMENTATION_FILENAME,
            relative_path=relative_path,
            url=f"/studies/{study_id}/files/{relative_path}",
        ),
        metadata=metadata,
    )

    write_segmentation_metadata(segmentation_dir, segmentation)
    return segmentation


def list_study_segmentations(study_id: str, settings: Settings) -> SegmentationListResponse:
    get_existing_study_dir(settings, study_id)

    segmentations = [
        segmentation
        for segmentation in (
            read_segmentation_metadata(path)
            for path in list_segmentation_dirs(settings.storage_root, study_id)
        )
        if segmentation is not None
    ]
    sorted_segmentations = sorted(
        segmentations,
        key=lambda segmentation: segmentation.created_at,
        reverse=True,
    )

    return SegmentationListResponse(items=sorted_segmentations)


def get_study_segmentation(
    study_id: str,
    segmentation_id: str,
    settings: Settings,
) -> SegmentationRead:
    get_existing_study_dir(settings, study_id)

    segmentation_dir = get_segmentation_dir(settings.storage_root, study_id, segmentation_id)
    segmentation = read_segmentation_metadata(segmentation_dir)

    if segmentation is None:
        raise SegmentationError("Segmentation not found.", status_code=404)

    return segmentation


def get_existing_study_dir(settings: Settings, study_id: str) -> Path:
    study_dir = get_study_dir(settings.storage_root, study_id)
    manifest = read_manifest(study_dir)

    if manifest is None:
        raise SegmentationError("Study not found.", status_code=404)

    return study_dir


def get_existing_run(settings: Settings, study_id: str, run_id: str) -> AiRunRead:
    run_dir = get_ai_run_dir(settings.storage_root, study_id, run_id)
    run = read_ai_run(run_dir)

    if run is None:
        raise SegmentationError("AI run not found.", status_code=404)

    return run


def get_segmentation_artifact(run: AiRunRead) -> AiRunArtifactRead:
    if run.output is None:
        raise SegmentationError("AI run has no output artifacts")

    for artifact in run.output.artifacts:
        if artifact.type == "nifti_segmentation":
            return artifact

    raise SegmentationError("AI run output does not contain a NIfTI segmentation artifact")
