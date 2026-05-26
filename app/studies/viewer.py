# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : viewer.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import nibabel as nib
import pydicom

from app.core.config import Settings
from app.studies.schemas import (
    StudyRead,
    StudyViewerRead,
    ViewerDicomImageRead,
    ViewerDicomRead,
    ViewerDicomSeriesRead,
    ViewerNiftiRead,
)
from app.studies.storage import get_source_dir, get_study_dir


@dataclass(frozen=True)
class DicomEntry:
    path: Path
    dataset: pydicom.Dataset


def build_study_viewer(study: StudyRead, settings: Settings) -> StudyViewerRead:
    nifti = None
    dicom = None

    if study.input_type == "nifti":
        nifti = build_nifti_viewer(study, settings)

    if study.input_type in {"dicom", "dicomdir"}:
        dicom = build_dicom_viewer(study, settings)

    return StudyViewerRead(
        study_id=study.id,
        input_type=study.input_type,
        status="ready",
        nifti=nifti,
        dicom=dicom,
    )


def build_nifti_viewer(study: StudyRead, settings: Settings) -> ViewerNiftiRead | None:
    study_dir = get_study_dir(settings.storage_root, study.id)
    source_dir = get_source_dir(settings.storage_root, study.id)
    path = find_first_nifti_file(source_dir)

    if path is None:
        return None

    relative_path = path.relative_to(study_dir).as_posix()
    metadata = extract_viewer_nifti_metadata(path)

    return ViewerNiftiRead(
        filename=path.name,
        relative_path=relative_path,
        url=build_file_url(study.id, relative_path),
        metadata=metadata,
    )


def build_dicom_viewer(study: StudyRead, settings: Settings) -> ViewerDicomRead:
    source_dir = get_source_dir(settings.storage_root, study.id)
    entries = read_dicom_entries(source_dir)
    grouped_entries = group_dicom_entries(entries)
    series = [
        build_dicom_series(study, settings, series_entries)
        for series_entries in grouped_entries.values()
    ]

    return ViewerDicomRead(series=series)


def find_first_nifti_file(source_dir: Path) -> Path | None:
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())

    for path in paths:
        if is_nifti_file(path):
            return path

    return None


def is_nifti_file(path: Path) -> bool:
    suffixes = path.suffixes

    return path.suffix == ".nii" or suffixes[-2:] == [".nii", ".gz"]


def extract_viewer_nifti_metadata(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    spacing = image.header.get_zooms()[: len(image.shape)]

    return {
        "shape": list(image.shape),
        "spacing": [float(value) for value in spacing],
    }


def read_dicom_entries(source_dir: Path) -> list[DicomEntry]:
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    entries: list[DicomEntry] = []

    for path in paths:
        entry = read_dicom_entry(path)

        if entry is not None:
            entries.append(entry)

    return entries


def read_dicom_entry(path: Path) -> DicomEntry | None:
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None

    return DicomEntry(path=path, dataset=dataset)


def group_dicom_entries(entries: list[DicomEntry]) -> dict[str | None, list[DicomEntry]]:
    grouped_entries: dict[str | None, list[DicomEntry]] = {}

    for entry in entries:
        series_uid = get_dicom_str(entry.dataset, "SeriesInstanceUID")
        grouped_entries.setdefault(series_uid, []).append(entry)

    return grouped_entries


def build_dicom_series(
    study: StudyRead,
    settings: Settings,
    entries: list[DicomEntry],
) -> ViewerDicomSeriesRead:
    sorted_entries = sorted(entries, key=dicom_sort_key)
    first_dataset = sorted_entries[0].dataset
    images = [
        build_dicom_image(study, settings, entry)
        for entry in sorted_entries
    ]

    return ViewerDicomSeriesRead(
        series_instance_uid=get_dicom_str(first_dataset, "SeriesInstanceUID"),
        study_instance_uid=get_dicom_str(first_dataset, "StudyInstanceUID"),
        modality=get_dicom_str(first_dataset, "Modality"),
        series_description=get_dicom_str(first_dataset, "SeriesDescription"),
        protocol_name=get_dicom_str(first_dataset, "ProtocolName"),
        manufacturer=get_dicom_str(first_dataset, "Manufacturer"),
        files_count=len(sorted_entries),
        rows=get_dicom_int(first_dataset, "Rows"),
        columns=get_dicom_int(first_dataset, "Columns"),
        slice_thickness=get_dicom_float(first_dataset, "SliceThickness"),
        pixel_spacing=get_dicom_float_list(first_dataset, "PixelSpacing"),
        images=images,
    )


def build_dicom_image(
    study: StudyRead,
    settings: Settings,
    entry: DicomEntry,
) -> ViewerDicomImageRead:
    study_dir = get_study_dir(settings.storage_root, study.id)
    relative_path = entry.path.relative_to(study_dir).as_posix()
    url = build_file_url(study.id, relative_path)
    absolute_url = f"{settings.backend_public_url.rstrip('/')}{url}"

    return ViewerDicomImageRead(
        filename=entry.path.name,
        relative_path=relative_path,
        url=url,
        image_id=f"wadouri:{absolute_url}",
        instance_number=get_dicom_int(entry.dataset, "InstanceNumber"),
        slice_location=get_dicom_float(entry.dataset, "SliceLocation"),
        image_position_patient=get_dicom_float_list(entry.dataset, "ImagePositionPatient"),
    )


def dicom_sort_key(entry: DicomEntry) -> tuple[bool, int, str]:
    instance_number = get_dicom_int(entry.dataset, "InstanceNumber")

    return (instance_number is None, instance_number or 0, entry.path.name)


def build_file_url(study_id: str, relative_path: str) -> str:
    encoded_path = quote(relative_path, safe="/")

    return f"/studies/{study_id}/files/{encoded_path}"


def get_dicom_str(dataset: pydicom.Dataset, field_name: str) -> str | None:
    value = getattr(dataset, field_name, None)

    if value is None:
        return None

    return str(value)


def get_dicom_int(dataset: pydicom.Dataset, field_name: str) -> int | None:
    value = getattr(dataset, field_name, None)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_dicom_float(dataset: pydicom.Dataset, field_name: str) -> float | None:
    value = getattr(dataset, field_name, None)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_dicom_float_list(dataset: pydicom.Dataset, field_name: str) -> list[float] | None:
    value = getattr(dataset, field_name, None)

    if value is None:
        return None

    if not isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return None

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None
