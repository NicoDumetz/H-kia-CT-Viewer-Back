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

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ai.registry import get_module
from app.analyses.hu import HuAnalysisError, compute_label_hu_statistics
from app.analyses.schemas import (
    AnalysisArtifactRead,
    AnalysisCreate,
    AnalysisInputRead,
    AnalysisListResponse,
    AnalysisOutputRead,
    AnalysisRead,
    AnalysisResultRead,
)
from app.analyses.storage import (
    create_analysis_dir,
    get_analysis_dir,
    list_analysis_dirs,
    read_analysis,
    read_analysis_result,
    write_analysis,
    write_analysis_result,
)
from app.core.config import Settings
from app.segmentations.labels import load_label_names
from app.segmentations.storage import get_segmentation_dir, read_segmentation_metadata
from app.studies.manifest import read_manifest
from app.studies.storage import get_study_dir
from app.studies.volume import VOLUME_RELATIVE_PATH


SUPPORTED_MODULE_ID = "segmentation_label_hu_statistics"


class AnalysisError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def create_analysis(
    study_id: str,
    payload: AnalysisCreate,
    settings: Settings,
) -> AnalysisRead:
    study_dir = get_existing_study_dir(settings, study_id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH
    analysis_id = str(uuid4())
    result_relative_path = f"derived/analyses/{analysis_id}/result.json"
    now = datetime.now(timezone.utc)

    validate_analysis_request(payload)

    if not volume_path.is_file():
        raise AnalysisError("Prepared volume not found.", status_code=404)

    segmentation = get_existing_segmentation(settings, study_id, payload.segmentation_id)
    segmentation_path = study_dir / segmentation.file.relative_path

    if not segmentation_path.is_file():
        raise AnalysisError("Segmentation file not found.", status_code=404)

    try:
        result = compute_label_hu_statistics(
            analysis_id=analysis_id,
            study_id=study_id,
            module_id=payload.module_id,
            segmentation_id=payload.segmentation_id,
            roi_mode=payload.roi_mode,
            volume_path=volume_path,
            segmentation_path=segmentation_path,
            label_ids=payload.label_ids,
            label_names=load_label_names(settings),
        )
    except HuAnalysisError as exc:
        raise AnalysisError(exc.message) from exc

    analysis_dir = create_analysis_dir(settings.storage_root, study_id, analysis_id)
    output = AnalysisOutputRead(
        result_path=result_relative_path,
        artifacts=[
            AnalysisArtifactRead(
                type="json",
                name="result.json",
                relative_path=result_relative_path,
                url=f"/studies/{study_id}/files/{result_relative_path}",
            )
        ],
    )
    analysis = AnalysisRead(
        id=analysis_id,
        study_id=study_id,
        module_id=payload.module_id,
        status="succeeded",
        created_at=now,
        updated_at=now,
        input=AnalysisInputRead(
            volume_path=VOLUME_RELATIVE_PATH,
            segmentation_id=payload.segmentation_id,
            segmentation_path=segmentation.file.relative_path,
            label_ids=payload.label_ids,
            roi_mode=payload.roi_mode,
        ),
        output=output,
        error=None,
    )

    write_analysis_result(analysis_dir, result)
    write_analysis(analysis_dir, analysis)
    return analysis


def list_study_analyses(study_id: str, settings: Settings) -> AnalysisListResponse:
    get_existing_study_dir(settings, study_id)

    analyses = [
        analysis
        for analysis in (
            read_analysis(path)
            for path in list_analysis_dirs(settings.storage_root, study_id)
        )
        if analysis is not None
    ]
    sorted_analyses = sorted(analyses, key=lambda item: item.created_at, reverse=True)

    return AnalysisListResponse(items=sorted_analyses)


def get_study_analysis(
    study_id: str,
    analysis_id: str,
    settings: Settings,
) -> AnalysisRead:
    get_existing_study_dir(settings, study_id)

    analysis_dir = get_analysis_dir(settings.storage_root, study_id, analysis_id)
    analysis = read_analysis(analysis_dir)

    if analysis is None:
        raise AnalysisError("Analysis not found.", status_code=404)

    return analysis


def get_study_analysis_result(
    study_id: str,
    analysis_id: str,
    settings: Settings,
) -> AnalysisResultRead:
    get_study_analysis(study_id, analysis_id, settings)

    analysis_dir = get_analysis_dir(settings.storage_root, study_id, analysis_id)
    result = read_analysis_result(analysis_dir)

    if result is None:
        raise AnalysisError("Analysis result not found.", status_code=404)

    return result


def validate_analysis_request(payload: AnalysisCreate) -> None:
    module = get_module(payload.module_id)

    if module is None:
        raise AnalysisError("Unknown analysis module_id")

    if payload.module_id != SUPPORTED_MODULE_ID:
        raise AnalysisError("Analysis module is not supported by this endpoint")

    if payload.roi_mode != "whole_label":
        raise AnalysisError("Unsupported roi_mode")


def get_existing_study_dir(settings: Settings, study_id: str) -> Path:
    study_dir = get_study_dir(settings.storage_root, study_id)
    manifest = read_manifest(study_dir)

    if manifest is None:
        raise AnalysisError("Study not found.", status_code=404)

    return study_dir


def get_existing_segmentation(settings: Settings, study_id: str, segmentation_id: str):
    segmentation_dir = get_segmentation_dir(settings.storage_root, study_id, segmentation_id)
    segmentation = read_segmentation_metadata(segmentation_dir)

    if segmentation is None:
        raise AnalysisError("Segmentation not found.", status_code=404)

    return segmentation
