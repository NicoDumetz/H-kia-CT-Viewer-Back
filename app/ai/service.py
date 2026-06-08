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

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np

from app.ai.nnunet_executor import (
    NnunetExecutionError,
    check_nnunet_available,
    run_nnunet_prediction,
)
from app.ai.registry import get_module, list_modules
from app.ai.schemas import (
    AiModuleDefinition,
    AiModuleListResponse,
    AiRunArtifactRead,
    AiRunCreate,
    AiRunInputRead,
    AiRunListResponse,
    AiRunOutputRead,
    AiRunRead,
)
from app.ai.storage import (
    LOGS_FILENAME,
    create_ai_run_dir,
    get_ai_run_dir,
    get_outputs_dir,
    list_ai_run_dirs,
    read_ai_run,
    write_ai_run,
)
from app.core.config import Settings
from app.studies.manifest import read_manifest
from app.studies.storage import get_study_dir
from app.studies.volume import VOLUME_RELATIVE_PATH


class AiRunError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def list_available_ai_modules(settings: Settings) -> AiModuleListResponse:
    return list_modules(settings)


def create_ai_run(
    study_id: str,
    payload: AiRunCreate,
    settings: Settings,
) -> AiRunRead:
    module = get_module(payload.module_id)
    study_dir = get_existing_study_dir(settings, study_id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH
    run_id = str(uuid4())
    now = datetime.now(timezone.utc)

    if module is None:
        raise AiRunError("Unknown AI module_id")

    if not volume_path.is_file():
        raise AiRunError("Prepared volume not found. Run preparation first.")

    run_dir = create_ai_run_dir(settings.storage_root, study_id, run_id)
    run = AiRunRead(
        id=run_id,
        study_id=study_id,
        module_id=module.id,
        module_name=module.name,
        status="pending",
        created_at=now,
        updated_at=now,
        started_at=None,
        finished_at=None,
        input=AiRunInputRead(prepared_volume_path=VOLUME_RELATIVE_PATH),
        output=None,
        error=None,
        error_detail=None,
    )

    write_ai_run(run_dir, run)
    return run


def list_study_ai_runs(study_id: str, settings: Settings) -> AiRunListResponse:
    get_existing_study_dir(settings, study_id)

    runs = [
        run
        for run in (
            read_ai_run(run_dir)
            for run_dir in list_ai_run_dirs(settings.storage_root, study_id)
        )
        if run is not None
    ]
    sorted_runs = sorted(runs, key=lambda run: run.created_at, reverse=True)

    return AiRunListResponse(items=sorted_runs)


def get_study_ai_run(study_id: str, run_id: str, settings: Settings) -> AiRunRead:
    get_existing_study_dir(settings, study_id)

    run_dir = get_ai_run_dir(settings.storage_root, study_id, run_id)
    run = read_ai_run(run_dir)

    if run is None:
        raise AiRunError("AI run not found.", status_code=404)

    return run


def simulate_ai_run(study_id: str, run_id: str, settings: Settings) -> AiRunRead:
    run = get_study_ai_run(study_id, run_id, settings)
    run_dir = get_ai_run_dir(settings.storage_root, study_id, run_id)
    running_at = datetime.now(timezone.utc)
    running_run = run.model_copy(
        update={
            "status": "running",
            "updated_at": running_at,
            "started_at": running_at,
            "error": None,
            "error_detail": None,
        }
    )

    write_ai_run(run_dir, running_run)
    write_simulation_logs(run_dir, running_run)

    output = write_simulation_output(run_dir, study_id, running_run, settings)
    finished_at = datetime.now(timezone.utc)
    finished_run = running_run.model_copy(
        update={
            "status": "succeeded",
            "updated_at": finished_at,
            "finished_at": finished_at,
            "output": output,
        }
    )

    write_ai_run(run_dir, finished_run)
    return finished_run


def execute_ai_run(study_id: str, run_id: str, settings: Settings) -> AiRunRead:
    run = get_study_ai_run(study_id, run_id, settings)
    module = get_module(run.module_id)
    study_dir = get_existing_study_dir(settings, study_id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH
    run_dir = get_ai_run_dir(settings.storage_root, study_id, run_id)
    available = False
    unavailable_reason = None

    if run.status != "pending":
        raise AiRunError("AI run is not pending and cannot be executed.", status_code=409)

    if module is None:
        raise AiRunError("Unknown AI module_id")

    if module.runner != "nnunet":
        raise AiRunError("AI module is not configured for nnU-Net execution")

    if not volume_path.is_file():
        raise AiRunError("Prepared volume not found. Run preparation first.")

    available, unavailable_reason = check_nnunet_available(module, settings)

    if not available:
        raise AiRunError(unavailable_reason or "nnU-Net execution is not available")

    return run_pending_nnunet(run, study_dir, run_dir, module, settings)


def run_pending_nnunet(
    run: AiRunRead,
    study_dir: Path,
    run_dir: Path,
    module: AiModuleDefinition,
    settings: Settings,
) -> AiRunRead:
    started_at = datetime.now(timezone.utc)
    running_run = run.model_copy(
        update={
            "status": "running",
            "started_at": run.started_at or started_at,
            "updated_at": started_at,
            "error": None,
            "error_detail": None,
        }
    )

    write_ai_run(run_dir, running_run)

    try:
        output = run_nnunet_prediction(running_run, study_dir, run_dir, module, settings)
        publish_succeeded_nnunet_segmentation(running_run, study_dir, output, module, settings)
    except NnunetExecutionError as exc:
        failed_at = datetime.now(timezone.utc)
        failed_run = running_run.model_copy(
            update={
                "status": "failed",
                "updated_at": failed_at,
                "finished_at": failed_at,
                "error": exc.message,
                "error_detail": exc.error_detail,
            }
        )
        write_ai_run(run_dir, failed_run)
        return failed_run

    finished_at = datetime.now(timezone.utc)
    succeeded_run = running_run.model_copy(
        update={
            "status": "succeeded",
            "updated_at": finished_at,
            "finished_at": finished_at,
            "output": output,
        }
    )

    write_ai_run(run_dir, succeeded_run)
    return succeeded_run


def publish_succeeded_nnunet_segmentation(
    run: AiRunRead,
    study_dir: Path,
    output: AiRunOutputRead,
    module: AiModuleDefinition,
    settings: Settings,
) -> None:
    from app.segmentations.labels import TOTALSEG117_MODEL_ID
    from app.segmentations.service import SegmentationError, create_segmentation_from_file

    source_path = study_dir / output.result_path

    try:
        create_segmentation_from_file(
            study_id=run.study_id,
            source_path=source_path,
            settings=settings,
            source_run_id=run.id,
            module_id=module.id,
            module_name=module.name,
            validate_against_volume=True,
            source="ai",
            model_id=TOTALSEG117_MODEL_ID,
            segmentation_id="totalseg117",
        )
    except SegmentationError as exc:
        raise NnunetExecutionError(f"Failed to publish nnU-Net segmentation: {exc.message}") from exc


def get_existing_study_dir(settings: Settings, study_id: str) -> Path:
    study_dir = get_study_dir(settings.storage_root, study_id)
    manifest = read_manifest(study_dir)

    if manifest is None:
        raise AiRunError("Study not found.", status_code=404)

    return study_dir


def write_simulation_logs(run_dir: Path, run: AiRunRead) -> Path:
    logs_path = run_dir / LOGS_FILENAME
    content = (
        "Simulated AI run started\n"
        f"run_id={run.id}\n"
        f"module_id={run.module_id}\n"
        "No real inference was executed\n"
    )

    logs_path.write_text(content, encoding="utf-8")
    return logs_path


def write_simulation_output(
    run_dir: Path,
    study_id: str,
    run: AiRunRead,
    settings: Settings,
) -> AiRunOutputRead:
    outputs_dir = get_outputs_dir(run_dir)
    result_path = outputs_dir / "result.json"
    segmentation_path = outputs_dir / "segmentation.nii.gz"
    result_relative_path = f"derived/ai-runs/{run.id}/outputs/result.json"
    segmentation_relative_path = f"derived/ai-runs/{run.id}/outputs/segmentation.nii.gz"
    result = {
        "message": "Simulated AI run completed",
        "module_id": run.module_id,
    }
    artifacts = [
        AiRunArtifactRead(
            type="json",
            name="result.json",
            relative_path=result_relative_path,
            url=f"/studies/{study_id}/files/{result_relative_path}",
        ),
        AiRunArtifactRead(
            type="nifti_segmentation",
            name="segmentation.nii.gz",
            relative_path=segmentation_relative_path,
            url=f"/studies/{study_id}/files/{segmentation_relative_path}",
        ),
    ]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_simulation_segmentation(segmentation_path, settings, study_id)

    return AiRunOutputRead(result_path=result_relative_path, artifacts=artifacts)


def write_simulation_segmentation(path: Path, settings: Settings, study_id: str) -> None:
    volume_path = get_study_dir(settings.storage_root, study_id) / VOLUME_RELATIVE_PATH
    image = nib.load(str(volume_path))
    array = np.zeros(image.shape, dtype=np.uint8)
    affine = image.affine

    if array.ndim >= 3:
        array[0:2, 0:2, 0:2] = 1
        array[-2:, -2:, -2:] = 2

    nib.save(nib.Nifti1Image(array, affine), str(path))
