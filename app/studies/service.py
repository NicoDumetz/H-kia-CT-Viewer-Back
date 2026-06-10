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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import pydicom
from fastapi import UploadFile

from app.core.config import Settings
from app.studies.detector import detect_file_type, detect_input_type
from app.studies.manifest import build_source_files, read_manifest, to_list_item, write_manifest
from app.studies.schemas import (
    InputType,
    StudyImportResponse,
    StudyListResponse,
    StudyRead,
    StudyViewerRead,
    StudyVolumeRead,
)
from app.studies.storage import (
    cleanup_expired_tmp_jobs as cleanup_expired_tmp_job_dirs,
    create_source_subdir,
    create_study_id,
    get_study_dir,
    list_study_dirs,
    resolve_study_file,
    save_upload_files,
)
from app.studies.viewer import build_study_viewer
from app.studies.volume import VolumeError, get_prepared_volume, prepare_study_volume


async def import_study(files: list[UploadFile], settings: Settings) -> StudyImportResponse:
    study = await create_imported_study(files, settings, "source")

    return StudyImportResponse.model_validate(study.model_dump())


async def upload_nifti_study(upload_file: UploadFile, settings: Settings) -> StudyVolumeRead:
    study = await create_imported_study([upload_file], settings, "source/original_upload")

    if study.input_type != "nifti":
        message = "Uploaded file must be a readable .nii or .nii.gz volume."
        mark_study_failed(study.id, settings, message)
        raise VolumeError(message)

    return prepare_volume_or_mark_failed(study, settings)


async def upload_dicom_study(
    files: list[UploadFile],
    settings: Settings,
) -> StudyVolumeRead:
    study = await create_imported_dicom_study(files, settings)

    if study.input_type not in {"dicom", "dicomdir"}:
        message = "No readable DICOM CT series found in upload."
        mark_study_failed(study.id, settings, message)
        raise VolumeError(message)

    return prepare_volume_or_mark_failed(study, settings)


async def create_imported_study(
    files: list[UploadFile],
    settings: Settings,
    source_relative_dir: str,
) -> StudyRead:
    study_id = create_study_id()
    source_dir = create_source_subdir(settings.storage_root, study_id, source_relative_dir)
    study_dir = get_study_dir(settings.storage_root, study_id)
    saved_paths = await save_upload_files(files, source_dir)
    input_type = detect_input_type(saved_paths)
    metadata = extract_metadata(saved_paths, input_type)
    now = datetime.now(timezone.utc)
    source_files = build_source_files(study_dir, saved_paths)

    study = StudyRead(
        id=study_id,
        status="imported",
        input_type=input_type,
        files_count=len(saved_paths),
        metadata=metadata,
        created_at=now,
        updated_at=now,
        source_files=source_files,
    )

    write_manifest(study_dir, study)
    return study


async def create_imported_dicom_study(
    files: list[UploadFile],
    settings: Settings,
) -> StudyRead:
    study_id = create_study_id()
    study_dir = get_study_dir(settings.storage_root, study_id)
    original_dir = create_source_subdir(
        settings.storage_root,
        study_id,
        "source/original_upload",
    )
    dicom_dir = study_dir / "source" / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=False)
    uploaded_paths = await save_upload_files(files, original_dir)
    dicom_paths = prepare_dicom_source_files(uploaded_paths, original_dir, dicom_dir)
    indexed_paths = uploaded_paths + dicom_paths
    input_type = detect_input_type(dicom_paths)
    metadata = extract_metadata(dicom_paths, input_type)
    now = datetime.now(timezone.utc)
    source_files = build_source_files(study_dir, indexed_paths)

    study = StudyRead(
        id=study_id,
        status="imported",
        input_type=input_type,
        files_count=len(indexed_paths),
        metadata=metadata,
        created_at=now,
        updated_at=now,
        source_files=source_files,
    )

    write_manifest(study_dir, study)
    return study


def prepare_dicom_source_files(
    uploaded_paths: list[Path],
    upload_root: Path,
    dicom_dir: Path,
) -> list[Path]:
    dicom_paths: list[Path] = []

    for path in uploaded_paths:
        if is_zip_file(path):
            dicom_paths.extend(extract_zip_file(path, dicom_dir))
        else:
            destination = dicom_dir / path.relative_to(upload_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            dicom_paths.append(destination)

    return dicom_paths


def is_zip_file(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def extract_zip_file(path: Path, destination_dir: Path) -> list[Path]:
    extracted_paths: list[Path] = []

    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                relative_path = safe_zip_member_path(member.filename)
                destination = destination_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

                extracted_paths.append(destination)
    except zipfile.BadZipFile as exc:
        raise VolumeError(f"Failed to read DICOM zip: {exc}") from exc

    return extracted_paths


def safe_zip_member_path(filename: str) -> Path:
    path = Path(filename)

    if path.is_absolute() or ".." in path.parts:
        raise VolumeError("DICOM zip contains an unsafe file path")

    return path


def list_studies(settings: Settings) -> StudyListResponse:
    study_dirs = list_study_dirs(settings.storage_root)
    studies = [study for study in (read_manifest(path) for path in study_dirs) if study]
    items = [to_list_item(study) for study in studies]
    sorted_items = sorted(items, key=lambda item: item.created_at, reverse=True)

    return StudyListResponse(items=sorted_items)


def get_study(study_id: str, settings: Settings) -> StudyRead | None:
    study_dir = get_study_dir(settings.storage_root, study_id)

    if not study_dir.is_dir():
        return None

    return read_manifest(study_dir)


def get_study_viewer(study_id: str, settings: Settings) -> StudyViewerRead | None:
    study = get_study(study_id, settings)

    if study is None:
        return None

    return build_study_viewer(study, settings)


def prepare_volume(study_id: str, settings: Settings) -> StudyVolumeRead | None:
    study = get_study(study_id, settings)

    if study is None:
        return None

    return prepare_volume_or_mark_failed(study, settings)


def prepare_volume_or_mark_failed(study: StudyRead, settings: Settings) -> StudyVolumeRead:
    try:
        return prepare_study_volume(study, settings)
    except VolumeError as exc:
        mark_study_failed(study.id, settings, exc.message)
        raise


def mark_study_failed(study_id: str, settings: Settings, message: str) -> None:
    study_dir = get_study_dir(settings.storage_root, study_id)
    study = read_manifest(study_dir)

    if study is None:
        return

    failed_study = study.model_copy(
        update={
            "status": "failed",
            "updated_at": datetime.now(timezone.utc),
            "error": message,
        }
    )

    write_manifest(study_dir, failed_study)


def get_volume(study_id: str, settings: Settings) -> StudyVolumeRead | None:
    study = get_study(study_id, settings)

    if study is None:
        return None

    return get_prepared_volume(study, settings)


def get_study_file_path(study_id: str, relative_path: str, settings: Settings) -> Path | None:
    return resolve_study_file(settings.storage_root, study_id, relative_path)


def cleanup_expired_tmp_jobs(settings: Settings) -> int:
    deleted = cleanup_expired_tmp_job_dirs(
        settings.storage_root,
        settings.storage_tmp_ttl_hours,
    )

    return len(deleted)


def extract_metadata(paths: list[Path], input_type: InputType) -> dict[str, Any]:
    path = find_metadata_source(paths, input_type)

    if path is None:
        return {}

    if input_type == "nifti":
        return extract_nifti_metadata(path)

    if input_type in {"dicom", "dicomdir"}:
        return extract_dicom_metadata(path)

    return {}


def find_metadata_source(paths: list[Path], input_type: InputType) -> Path | None:
    for path in paths:
        if detect_file_type(path) == input_type:
            return path

    if input_type == "dicomdir":
        for path in paths:
            if path.name.upper() == "DICOMDIR":
                return path

    return None


def extract_nifti_metadata(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    header = image.header
    spacing = header.get_zooms()[: len(image.shape)]

    return {
        "filename": path.name,
        "shape": list(image.shape),
        "spacing": [float(value) for value in spacing],
    }


def extract_dicom_metadata(path: Path) -> dict[str, Any]:
    dataset = pydicom.dcmread(path, stop_before_pixels=True)

    return {
        "filename": path.name,
        "modality": get_dicom_value(dataset, "Modality"),
        "study_instance_uid": get_dicom_value(dataset, "StudyInstanceUID"),
        "series_instance_uid": get_dicom_value(dataset, "SeriesInstanceUID"),
        "series_description": get_dicom_value(dataset, "SeriesDescription"),
        "protocol_name": get_dicom_value(dataset, "ProtocolName"),
        "rows": get_dicom_value(dataset, "Rows"),
        "columns": get_dicom_value(dataset, "Columns"),
        "manufacturer": get_dicom_value(dataset, "Manufacturer"),
        "slice_thickness": get_dicom_value(dataset, "SliceThickness"),
        "pixel_spacing": get_dicom_value(dataset, "PixelSpacing"),
    }


def get_dicom_value(dataset: pydicom.Dataset, field_name: str) -> Any:
    value = getattr(dataset, field_name, None)

    if value is None:
        return None

    if isinstance(value, pydicom.multival.MultiValue):
        return [convert_dicom_scalar(item) for item in value]

    return convert_dicom_scalar(value)


def convert_dicom_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace")

    if hasattr(value, "original_string"):
        return str(value)

    return value
