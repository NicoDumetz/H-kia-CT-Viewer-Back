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

from typing import Any

from app.ai.service import list_available_ai_modules, list_study_ai_runs
from app.analyses.service import list_study_analyses
from app.core.config import Settings
from app.segmentations.service import list_study_segmentations
from app.studies.service import get_study, get_study_viewer, get_volume
from app.studies.volume import VolumeError
from app.workspace.schemas import (
    StudyWorkspaceRead,
    WorkspaceAiRead,
    WorkspaceAvailableActionsRead,
    WorkspaceCollectionRead,
    WorkspaceStudyRead,
    WorkspaceVolumeRead,
)


class WorkspaceError(Exception):
    def __init__(self, message: str, status_code: int = 404) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def get_study_workspace(study_id: str, settings: Settings) -> StudyWorkspaceRead:
    try:
        study = get_study(study_id, settings)
    except Exception as exc:
        raise WorkspaceError("Study manifest could not be read.", status_code=422) from exc

    if study is None:
        raise WorkspaceError("Study not found.", status_code=404)

    viewer = safe_get_viewer(study_id, settings)
    volume = safe_get_volume(study_id, settings)
    modules = list_available_ai_modules(settings).items
    runs = list_study_ai_runs(study_id, settings).items
    segmentations = list_study_segmentations(study_id, settings).items
    analyses = list_study_analyses(study_id, settings).items
    actions = build_available_actions(
        input_type=study.input_type,
        volume_prepared=volume.is_prepared,
        modules=modules,
        runs=runs,
        segmentations=segmentations,
    )

    return StudyWorkspaceRead(
        study=WorkspaceStudyRead(
            id=study.id,
            status=study.status,
            input_type=study.input_type,
            files_count=study.files_count,
            created_at=study.created_at,
            updated_at=study.updated_at,
        ),
        viewer=viewer,
        volume=volume,
        ai=WorkspaceAiRead(modules=modules, runs=runs),
        segmentations=WorkspaceCollectionRead(
            items=segmentations,
            latest=get_latest_item(segmentations),
        ),
        analyses=WorkspaceCollectionRead(
            items=analyses,
            latest=get_latest_item(analyses),
        ),
        available_actions=actions,
    )


def safe_get_viewer(study_id: str, settings: Settings) -> Any | None:
    try:
        viewer = get_study_viewer(study_id, settings)
    except Exception:
        return {"error": "Viewer payload could not be built."}

    return viewer


def safe_get_volume(study_id: str, settings: Settings) -> WorkspaceVolumeRead:
    try:
        volume = get_volume(study_id, settings)
    except VolumeError as exc:
        if exc.status_code == 404:
            return WorkspaceVolumeRead(is_prepared=False, data=None)

        return WorkspaceVolumeRead(is_prepared=False, data=None)

    return WorkspaceVolumeRead(is_prepared=volume is not None, data=volume)


def get_latest_item(items: list[Any]) -> Any | None:
    if not items:
        return None

    return items[0]


def build_available_actions(
    input_type: str,
    volume_prepared: bool,
    modules: list[Any],
    runs: list[Any],
    segmentations: list[Any],
) -> WorkspaceAvailableActionsRead:
    return WorkspaceAvailableActionsRead(
        can_prepare_volume=(not volume_prepared and input_type in {"dicom", "dicomdir", "nifti"}),
        can_create_ai_run=volume_prepared,
        can_execute_ai=has_available_nnunet_module(modules),
        can_publish_segmentation=has_publishable_segmentation_run(runs),
        can_run_label_hu_statistics=(volume_prepared and bool(segmentations)),
    )


def has_available_nnunet_module(modules: list[Any]) -> bool:
    return any(module.runner == "nnunet" and module.is_available for module in modules)


def has_publishable_segmentation_run(runs: list[Any]) -> bool:
    for run in runs:
        if run.status == "succeeded" and run.output is not None:
            if has_nifti_segmentation_artifact(run.output.artifacts):
                return True

    return False


def has_nifti_segmentation_artifact(artifacts: list[Any]) -> bool:
    return any(artifact.type == "nifti_segmentation" for artifact in artifacts)
