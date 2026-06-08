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

import nibabel as nib
import numpy as np
from fastapi import UploadFile

from app.ai.schemas import AiRunArtifactRead, AiRunRead
from app.ai.storage import get_ai_run_dir, read_ai_run
from app.core.config import Settings
from app.segmentations.analysis import (
    SegmentationAnalysisError,
    compute_segmentation_metadata,
    validate_segmentation_array,
)
from app.segmentations.labels import (
    TOTALSEG117_LABELS,
    TOTALSEG117_MODEL_ID,
    load_label_names,
)
from app.segmentations.schemas import (
    SegmentationBBoxRead,
    SegmentationFileRead,
    SegmentationLabelRead,
    SegmentationLabelsDocumentRead,
    SegmentationListResponse,
    SegmentationRead,
    SegmentationSource,
)
from app.segmentations.storage import (
    LEGACY_SEGMENTATION_FILENAME,
    SEGMENTATION_LABELS_FILENAME,
    SEGMENTATION_FILENAME,
    create_segmentation_dir,
    find_segmentation_dir,
    get_canonical_segmentation_dir,
    list_segmentation_dirs,
    read_segmentation_labels,
    read_segmentation_metadata,
    write_segmentation_labels,
    write_segmentation_metadata,
)
from app.studies.manifest import read_manifest
from app.studies.storage import get_study_dir
from app.studies.volume import VOLUME_RELATIVE_PATH


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

    return create_segmentation_from_file(
        study_id=study_id,
        source_path=source_path,
        settings=settings,
        source_run_id=run.id,
        module_id=run.module_id,
        module_name=run.module_name,
        validate_against_volume=True,
        source="ai",
        model_id=TOTALSEG117_MODEL_ID,
        segmentation_id=get_ai_segmentation_id(run),
    )


async def upload_manual_segmentation(
    study_id: str,
    upload_file: UploadFile,
    name: str | None,
    source: str,
    settings: Settings,
    labels_payload: dict[int, str] | None = None,
) -> SegmentationRead:
    study_dir = get_existing_study_dir(settings, study_id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH
    segmentation_id = str(uuid4())
    segmentation_dir = create_segmentation_dir(
        settings.storage_root,
        study_id,
        segmentation_id,
        "manual",
    )
    upload_path = segmentation_dir / get_upload_temp_filename(upload_file)
    module_name = "Manual segmentation upload"

    if not volume_path.is_file():
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        raise SegmentationError("Prepared volume not found.", status_code=404)

    if not is_nifti_filename(upload_file.filename or ""):
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        raise SegmentationError("Uploaded segmentation must be a .nii or .nii.gz file")

    try:
        await save_upload_file(upload_file, upload_path)
        validate_segmentation_against_volume(upload_path, volume_path)
        return create_segmentation_from_file(
            study_id=study_id,
            source_path=upload_path,
            settings=settings,
            source_run_id=None,
            module_id=source or "manual_upload",
            module_name=module_name,
            validate_against_volume=False,
            source="manual",
            model_id=None,
            segmentation_id=segmentation_id,
            segmentation_dir=segmentation_dir,
            extra_label_names=labels_payload,
        )
    except SegmentationError:
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        raise SegmentationError(f"Failed to upload segmentation: {exc}") from exc


def create_segmentation_from_file(
    study_id: str,
    source_path: Path,
    settings: Settings,
    source_run_id: str | None,
    module_id: str,
    module_name: str,
    validate_against_volume: bool,
    source: SegmentationSource,
    model_id: str | None,
    segmentation_id: str | None = None,
    segmentation_dir: Path | None = None,
    extra_label_names: dict[int, str] | None = None,
) -> SegmentationRead:
    study_dir = get_existing_study_dir(settings, study_id)
    final_segmentation_id = segmentation_id or str(uuid4())
    final_segmentation_dir = segmentation_dir or prepare_segmentation_dir(
        settings,
        study_id,
        final_segmentation_id,
        source,
    )
    destination_path = final_segmentation_dir / SEGMENTATION_FILENAME
    relative_path = (
        f"derived/segmentations/{source}/{final_segmentation_id}/{SEGMENTATION_FILENAME}"
    )
    labels_relative_path = (
        f"derived/segmentations/{source}/{final_segmentation_id}/{SEGMENTATION_LABELS_FILENAME}"
    )

    if validate_against_volume:
        validate_segmentation_against_volume(source_path, study_dir / VOLUME_RELATIVE_PATH)

    save_nifti_as_gzip(source_path, destination_path)

    try:
        label_names = load_label_names(settings)

        if extra_label_names:
            label_names.update(extra_label_names)

        metadata = compute_segmentation_metadata(destination_path, label_names)
    except SegmentationAnalysisError as exc:
        shutil.rmtree(final_segmentation_dir, ignore_errors=True)
        raise SegmentationError(exc.message) from exc

    labels_document = build_labels_document(
        segmentation_id=final_segmentation_id,
        source=source,
        model_id=model_id,
        metadata=metadata,
    )
    labels_file = SegmentationFileRead(
        filename=SEGMENTATION_LABELS_FILENAME,
        relative_path=labels_relative_path,
        url=f"/studies/{study_id}/files/{labels_relative_path}",
    )
    metadata = metadata.model_copy(
        update={
            "labels_path": labels_relative_path,
            "labels_url": labels_file.url,
            "labels_count": labels_document.labels_count,
            "present_labels_count": labels_document.present_labels_count,
            "labels": labels_document.labels,
        }
    )
    segmentation = SegmentationRead(
        id=final_segmentation_id,
        study_id=study_id,
        source=source,
        source_run_id=source_run_id,
        module_id=module_id,
        module_name=module_name,
        model_id=model_id,
        status="ready",
        created_at=datetime.now(timezone.utc),
        file=SegmentationFileRead(
            filename=SEGMENTATION_FILENAME,
            relative_path=relative_path,
            url=f"/studies/{study_id}/files/{relative_path}",
        ),
        labels_file=labels_file,
        metadata=metadata,
    )

    write_segmentation_labels(final_segmentation_dir, labels_document)
    write_segmentation_metadata(final_segmentation_dir, segmentation)
    write_legacy_segmentation_files(study_id, final_segmentation_id, settings, segmentation)
    remove_temporary_upload(source_path, final_segmentation_dir)
    return segmentation


def prepare_segmentation_dir(
    settings: Settings,
    study_id: str,
    segmentation_id: str,
    source: SegmentationSource,
) -> Path:
    segmentation_dir = get_canonical_segmentation_dir(
        settings.storage_root,
        study_id,
        segmentation_id,
        source,
    )

    if segmentation_dir.exists():
        shutil.rmtree(segmentation_dir, ignore_errors=True)

    segmentation_dir.mkdir(parents=True, exist_ok=False)
    return segmentation_dir


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

    segmentation_dir = find_segmentation_dir(settings.storage_root, study_id, segmentation_id)

    if segmentation_dir is None:
        raise SegmentationError("Segmentation not found.", status_code=404)

    segmentation = read_segmentation_metadata(segmentation_dir)

    if segmentation is None:
        raise SegmentationError("Segmentation not found.", status_code=404)

    return segmentation


def get_study_segmentation_labels(
    study_id: str,
    segmentation_id: str,
    settings: Settings,
) -> SegmentationLabelsDocumentRead:
    get_existing_study_dir(settings, study_id)

    segmentation_dir = find_segmentation_dir(settings.storage_root, study_id, segmentation_id)

    if segmentation_dir is None:
        raise SegmentationError("Segmentation not found.", status_code=404)

    labels = read_segmentation_labels(segmentation_dir)

    if labels is None:
        raise SegmentationError("Segmentation labels not found.", status_code=404)

    return labels


def build_labels_document(
    segmentation_id: str,
    source: SegmentationSource,
    model_id: str | None,
    metadata,
) -> SegmentationLabelsDocumentRead:
    present_by_id = {label.label_id: label for label in metadata.labels}

    if model_id == TOTALSEG117_MODEL_ID:
        labels = [
            present_by_id.get(label_id) or build_absent_totalseg_label(label_id, name)
            for label_id, name in sorted(TOTALSEG117_LABELS.items())
        ]
    else:
        labels = metadata.labels

    return SegmentationLabelsDocumentRead(
        segmentation_id=segmentation_id,
        source=source,
        model_id=model_id,
        labels_count=len(labels),
        present_labels_count=len(present_by_id),
        labels=labels,
    )


def build_absent_totalseg_label(label_id: int, name: str) -> SegmentationLabelRead:
    from app.segmentations.labels import get_label_color, get_label_group

    return SegmentationLabelRead(
        id=label_id,
        label_id=label_id,
        name=name,
        group=get_label_group(name),
        present=False,
        voxel_count=0,
        volume_mm3=0.0,
        color=get_label_color(label_id),
        opacity=0.45,
        bbox_ijk=SegmentationBBoxRead(min=[], max=[]),
        center_ijk=[],
    )


def get_existing_study_dir(settings: Settings, study_id: str) -> Path:
    study_dir = get_study_dir(settings.storage_root, study_id)
    manifest = read_manifest(study_dir)

    if manifest is None:
        raise SegmentationError("Study not found.", status_code=404)

    return study_dir


def get_ai_segmentation_id(run: AiRunRead) -> str:
    if run.module_id == "ct_anatomy_segmentation_nnunet":
        return "totalseg117"

    return run.module_id


def write_legacy_segmentation_files(
    study_id: str,
    segmentation_id: str,
    settings: Settings,
    segmentation: SegmentationRead,
) -> None:
    canonical_dir = find_segmentation_dir(settings.storage_root, study_id, segmentation_id)

    if canonical_dir is None:
        return

    legacy_dir = get_canonical_segmentation_dir(
        settings.storage_root,
        study_id,
        segmentation_id,
        "",
    )
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_mask_relative_path = (
        f"derived/segmentations/{segmentation_id}/{LEGACY_SEGMENTATION_FILENAME}"
    )
    legacy_labels_relative_path = (
        f"derived/segmentations/{segmentation_id}/{SEGMENTATION_LABELS_FILENAME}"
    )
    legacy_labels_file = SegmentationFileRead(
        filename=SEGMENTATION_LABELS_FILENAME,
        relative_path=legacy_labels_relative_path,
        url=f"/studies/{study_id}/files/{legacy_labels_relative_path}",
    )
    legacy_metadata = segmentation.metadata.model_copy(
        update={
            "labels_path": legacy_labels_relative_path,
            "labels_url": legacy_labels_file.url,
        }
    )
    legacy_segmentation = segmentation.model_copy(
        update={
            "file": SegmentationFileRead(
                filename=LEGACY_SEGMENTATION_FILENAME,
                relative_path=legacy_mask_relative_path,
                url=f"/studies/{study_id}/files/{legacy_mask_relative_path}",
            ),
            "labels_file": legacy_labels_file,
            "metadata": legacy_metadata,
        }
    )

    shutil.copy2(canonical_dir / SEGMENTATION_FILENAME, legacy_dir / LEGACY_SEGMENTATION_FILENAME)
    write_segmentation_metadata(legacy_dir, legacy_segmentation)

    labels_path = canonical_dir / SEGMENTATION_LABELS_FILENAME

    if labels_path.is_file():
        shutil.copy2(labels_path, legacy_dir / SEGMENTATION_LABELS_FILENAME)


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


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    chunk_size = 1024 * 1024
    chunk = await upload_file.read(chunk_size)

    with destination.open("wb") as output:
        while chunk:
            output.write(chunk)
            chunk = await upload_file.read(chunk_size)

    await upload_file.seek(0)


def is_nifti_filename(filename: str) -> bool:
    return filename.lower().endswith((".nii", ".nii.gz"))


def get_upload_temp_filename(upload_file: UploadFile) -> str:
    filename = upload_file.filename or "upload.nii.gz"

    if filename.lower().endswith(".nii.gz"):
        return "upload.nii.gz"

    return "upload.nii"


def validate_segmentation_shape(segmentation_path: Path, volume_path: Path) -> None:
    validate_segmentation_against_volume(segmentation_path, volume_path, check_geometry=False)


def validate_segmentation_against_volume(
    segmentation_path: Path,
    volume_path: Path,
    check_geometry: bool = True,
) -> None:
    try:
        segmentation_image = nib.load(str(segmentation_path))
        volume_image = nib.load(str(volume_path))
        segmentation_array = np.asanyarray(segmentation_image.dataobj)
    except Exception as exc:
        raise SegmentationError(f"Failed to read NIfTI files: {exc}") from exc

    try:
        validate_segmentation_array(segmentation_array)
    except SegmentationAnalysisError as exc:
        raise SegmentationError(exc.message) from exc

    if segmentation_image.shape != volume_image.shape:
        raise SegmentationError(
            f"Segmentation shape {list(segmentation_image.shape)} does not match "
            f"volume shape {list(volume_image.shape)}"
        )

    if not check_geometry:
        return

    validate_segmentation_geometry(segmentation_image, volume_image)


def validate_segmentation_geometry(
    segmentation_image: nib.Nifti1Image,
    volume_image: nib.Nifti1Image,
) -> None:
    segmentation_spacing = segmentation_image.header.get_zooms()[:3]
    volume_spacing = volume_image.header.get_zooms()[:3]

    if not np.allclose(segmentation_spacing, volume_spacing, atol=1e-3, rtol=1e-3):
        raise SegmentationError(
            "Segmentation spacing is not compatible with the prepared CT volume"
        )

    if not np.allclose(segmentation_image.affine, volume_image.affine, atol=1e-3, rtol=1e-3):
        raise SegmentationError(
            "Segmentation orientation or affine is not compatible with the prepared CT volume"
        )


def save_nifti_as_gzip(source_path: Path, destination_path: Path) -> None:
    if source_path == destination_path:
        return

    try:
        image = nib.load(str(source_path))
        nib.save(image, str(destination_path))
    except Exception as exc:
        raise SegmentationError(f"Failed to save segmentation NIfTI: {exc}") from exc


def remove_temporary_upload(source_path: Path, segmentation_dir: Path) -> None:
    if source_path.parent == segmentation_dir and source_path.name != SEGMENTATION_FILENAME:
        source_path.unlink(missing_ok=True)
