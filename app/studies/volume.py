# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : volume.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

from app.core.config import Settings
from app.studies.manifest import write_manifest
from app.studies.schemas import (
    PreparedVolumeRead,
    StudyPreparedVolumeManifestRead,
    StudyRead,
    StudyVolumeRead,
    VolumeIntensityRead,
    VolumeMetadataRead,
)
from app.studies.storage import create_volume_dir, get_source_dir, get_study_dir


VOLUME_FILENAME = "ct.nii.gz"
VOLUME_RELATIVE_PATH = "derived/volume/ct.nii.gz"
METADATA_RELATIVE_PATH = "derived/volume/metadata.json"


class VolumeError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class DicomVolumeEntry:
    path: Path
    dataset: pydicom.Dataset


def prepare_study_volume(study: StudyRead, settings: Settings) -> StudyVolumeRead:
    if study.input_type == "nifti":
        return prepare_nifti_volume(study, settings)

    if study.input_type in {"dicom", "dicomdir"}:
        return prepare_dicom_volume(study, settings)

    raise VolumeError("Unknown input type cannot be prepared")


def get_prepared_volume(study: StudyRead, settings: Settings) -> StudyVolumeRead:
    study_dir = get_study_dir(settings.storage_root, study.id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH
    metadata_path = study_dir / METADATA_RELATIVE_PATH

    if not volume_path.is_file() or not metadata_path.is_file():
        raise VolumeError("Prepared volume not found. Run preparation first.", status_code=404)

    metadata = read_volume_metadata(metadata_path)
    volume = build_prepared_volume(study.id, metadata)

    return StudyVolumeRead(study_id=study.id, status="prepared", volume=volume)


def prepare_nifti_volume(study: StudyRead, settings: Settings) -> StudyVolumeRead:
    source_dir = get_source_dir(settings.storage_root, study.id)
    volume_dir = create_volume_dir(settings.storage_root, study.id)
    output_path = volume_dir / VOLUME_FILENAME
    source_path = find_first_nifti_file(source_dir)

    if source_path is None:
        raise VolumeError("No NIfTI source file found")

    try:
        image = nib.load(str(source_path))
        validate_nifti_volume(image)
        nib.save(image, str(output_path))
        metadata = compute_nifti_volume_metadata(image)
        metadata["source_type"] = "nifti"
    except Exception as exc:
        raise VolumeError(f"Failed to prepare NIfTI volume: {exc}") from exc

    return persist_prepared_volume(study, settings, metadata)


def validate_nifti_volume(image: nib.Nifti1Image) -> None:
    if len(image.shape) != 3:
        raise VolumeError("NIfTI CT volume must be a 3D image")


def prepare_dicom_volume(study: StudyRead, settings: Settings) -> StudyVolumeRead:
    source_dir = get_source_dir(settings.storage_root, study.id)
    volume_dir = create_volume_dir(settings.storage_root, study.id)
    output_path = volume_dir / VOLUME_FILENAME
    entries = read_dicom_entries(source_dir)

    if not entries and has_dicomdir_file(source_dir):
        raise VolumeError(
            "DICOMDIR alone does not contain pixel data. Upload associated DICOM files."
        )

    selected_entries = select_dicom_series(entries)
    logs_path = volume_dir / "prepare.log"

    if shutil.which("dcm2niix") is not None:
        metadata = convert_dicom_with_dcm2niix(source_dir, output_path, logs_path)
        metadata.update(get_selected_dicom_metadata(selected_entries))
        metadata["source_type"] = "dicom"
        return persist_prepared_volume(study, settings, metadata)

    sorted_entries = sort_dicom_entries(selected_entries)
    selected_paths = [str(entry.path) for entry in sorted_entries]
    spacing_diagnostic = compute_dicom_spacing_diagnostic(sorted_entries)

    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(selected_paths)
        image = execute_sitk_reader_without_warnings(reader)
        sitk.WriteImage(image, str(output_path))
        metadata = compute_sitk_volume_metadata(image)
        metadata.update(get_selected_dicom_metadata(selected_entries))
        metadata.update(spacing_diagnostic)
        metadata["source_type"] = "dicom"
        logs_path.write_text(
            build_simpleitk_prepare_log(spacing_diagnostic),
            encoding="utf-8",
        )
    except Exception as exc:
        logs_path.write_text(
            f"Prepared with SimpleITK. dcm2niix was not available.\nerror={exc}\n",
            encoding="utf-8",
        )
        raise VolumeError(f"Failed to prepare DICOM volume: {exc}") from exc

    return persist_prepared_volume(study, settings, metadata)


def convert_dicom_with_dcm2niix(
    source_dir: Path,
    output_path: Path,
    logs_path: Path,
) -> dict[str, Any]:
    temp_dir = output_path.parent / "dcm2niix"
    command = [
        "dcm2niix",
        "-z",
        "y",
        "-o",
        str(temp_dir),
        "-f",
        "ct",
        str(source_dir),
    ]

    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        write_prepare_log(logs_path, command, None, "", str(exc))
        raise VolumeError(f"Failed to start dcm2niix: {exc}") from exc

    write_prepare_log(logs_path, command, result.returncode, result.stdout, result.stderr)

    if result.returncode != 0:
        raise VolumeError(f"dcm2niix failed with return code {result.returncode}")

    nifti_path = find_largest_nifti_file(temp_dir)

    if nifti_path is None:
        raise VolumeError("dcm2niix did not produce a NIfTI volume")

    shutil.copy2(nifti_path, output_path)

    try:
        image = nib.load(str(output_path))
    except Exception as exc:
        raise VolumeError(f"Failed to read dcm2niix NIfTI output: {exc}") from exc

    return compute_nifti_volume_metadata(image)


def write_prepare_log(
    logs_path: Path,
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> Path:
    content = (
        f"command={' '.join(command)}\n"
        f"returncode={returncode}\n"
        "\nstdout:\n"
        f"{stdout}\n"
        "\nstderr:\n"
        f"{stderr}\n"
    )

    logs_path.write_text(content, encoding="utf-8")
    return logs_path


def find_largest_nifti_file(directory: Path) -> Path | None:
    paths = [path for path in directory.rglob("*") if path.is_file() and is_nifti_file(path)]

    if not paths:
        return None

    return max(paths, key=lambda path: path.stat().st_size)


def persist_prepared_volume(
    study: StudyRead,
    settings: Settings,
    metadata: dict[str, Any],
) -> StudyVolumeRead:
    study_dir = get_study_dir(settings.storage_root, study.id)
    metadata_path = study_dir / METADATA_RELATIVE_PATH
    prepared_at = datetime.now(timezone.utc)
    metadata = {
        **metadata,
        "prepared_at": prepared_at.isoformat(),
    }
    updated_study = study.model_copy(
        update={
            "status": "prepared",
            "updated_at": prepared_at,
            "error": None,
            "prepared_volume": StudyPreparedVolumeManifestRead(
                filename=VOLUME_FILENAME,
                relative_path=VOLUME_RELATIVE_PATH,
                metadata_path=METADATA_RELATIVE_PATH,
            ),
        }
    )
    volume_metadata = VolumeMetadataRead.model_validate(metadata)
    volume = build_prepared_volume(study.id, volume_metadata)

    metadata_path.write_text(
        json.dumps(volume_metadata.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    write_manifest(study_dir, updated_study)

    return StudyVolumeRead(study_id=study.id, status="prepared", volume=volume)


def get_selected_dicom_metadata(entries: list[DicomVolumeEntry]) -> dict[str, Any]:
    dataset = entries[0].dataset

    return {
        "selected_series_instance_uid": get_dicom_str(dataset, "SeriesInstanceUID"),
        "selected_series_description": get_dicom_str(dataset, "SeriesDescription"),
        "selected_protocol_name": get_dicom_str(dataset, "ProtocolName"),
        "selected_modality": get_dicom_str(dataset, "Modality"),
        "selected_files_count": len(entries),
    }


def compute_nifti_volume_metadata(image: nib.Nifti1Image) -> dict[str, Any]:
    data = np.asanyarray(image.dataobj)
    spacing = image.header.get_zooms()[: len(image.shape)]
    affine = image.affine
    origin = affine[:3, 3]
    direction = compute_nifti_direction(affine, spacing)
    intensity = compute_intensity_stats(data)

    return {
        "shape": list(image.shape),
        "spacing": [float(value) for value in spacing],
        "origin": [float(value) for value in origin],
        "direction": direction,
        "affine": [[float(value) for value in row] for row in affine.tolist()],
        "intensity": intensity,
    }


def compute_sitk_volume_metadata(image: sitk.Image) -> dict[str, Any]:
    data = sitk.GetArrayFromImage(image)
    intensity = compute_intensity_stats(data)

    return {
        "shape": [int(value) for value in image.GetSize()],
        "spacing": [float(value) for value in image.GetSpacing()],
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
        "affine": None,
        "intensity": intensity,
    }


def execute_sitk_reader_without_warnings(reader: sitk.ImageSeriesReader) -> sitk.Image:
    get_warning_display = getattr(sitk, "ProcessObject_GetGlobalWarningDisplay", None)
    set_warning_display = getattr(sitk, "ProcessObject_SetGlobalWarningDisplay", None)

    if get_warning_display is None or set_warning_display is None:
        return reader.Execute()

    warnings_were_enabled = get_warning_display()
    set_warning_display(False)

    try:
        return reader.Execute()
    finally:
        set_warning_display(warnings_were_enabled)


def build_simpleitk_prepare_log(spacing_diagnostic: dict[str, Any]) -> str:
    lines = ["Prepared with SimpleITK. dcm2niix was not available."]
    maximum_nonuniformity = spacing_diagnostic.get("dicom_maximum_nonuniformity_mm")

    if maximum_nonuniformity is not None and maximum_nonuniformity > 0:
        lines.append(
            "DICOM slice spacing is non-uniform or slices may be missing; "
            f"maximum_nonuniformity_mm={maximum_nonuniformity:.6f}"
        )

    return "\n".join(lines) + "\n"


def compute_intensity_stats(data: np.ndarray) -> VolumeIntensityRead:
    array = np.asarray(data)

    if array.size == 0:
        return VolumeIntensityRead(
            min=0.0,
            max=0.0,
            mean=0.0,
            median=0.0,
            p1=0.0,
            p5=0.0,
            p95=0.0,
            p99=0.0,
        )

    return VolumeIntensityRead(
        min=float(np.min(array)),
        max=float(np.max(array)),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        p1=float(np.percentile(array, 1)),
        p5=float(np.percentile(array, 5)),
        p95=float(np.percentile(array, 95)),
        p99=float(np.percentile(array, 99)),
    )


def compute_nifti_direction(affine: np.ndarray, spacing: tuple[float, ...]) -> list[float]:
    direction_matrix = affine[:3, :3].copy()

    for index, value in enumerate(spacing[:3]):
        if value:
            direction_matrix[:, index] = direction_matrix[:, index] / value

    return [float(value) for value in direction_matrix.reshape(-1)]


def find_first_nifti_file(source_dir: Path) -> Path | None:
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())

    for path in paths:
        if is_nifti_file(path):
            return path

    return None


def is_nifti_file(path: Path) -> bool:
    suffixes = path.suffixes

    return path.suffix == ".nii" or suffixes[-2:] == [".nii", ".gz"]


def read_dicom_entries(source_dir: Path) -> list[DicomVolumeEntry]:
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    entries: list[DicomVolumeEntry] = []

    for path in paths:
        if path.name.upper() == "DICOMDIR":
            continue

        entry = read_dicom_entry(path)

        if entry is not None:
            entries.append(entry)

    return entries


def has_dicomdir_file(source_dir: Path) -> bool:
    return any(
        path.is_file() and path.name.upper() == "DICOMDIR"
        for path in source_dir.rglob("*")
    )


def read_dicom_entry(path: Path) -> DicomVolumeEntry | None:
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None

    return DicomVolumeEntry(path=path, dataset=dataset)


def select_dicom_series(entries: list[DicomVolumeEntry]) -> list[DicomVolumeEntry]:
    grouped_entries = group_dicom_entries(entries)
    series = list(grouped_entries.values())

    if not series:
        raise VolumeError("No readable DICOM files found")

    return sorted(series, key=dicom_series_priority, reverse=True)[0]


def group_dicom_entries(
    entries: list[DicomVolumeEntry],
) -> dict[str | None, list[DicomVolumeEntry]]:
    grouped_entries: dict[str | None, list[DicomVolumeEntry]] = {}

    for entry in entries:
        series_uid = get_dicom_str(entry.dataset, "SeriesInstanceUID")
        grouped_entries.setdefault(series_uid, []).append(entry)

    return grouped_entries


def dicom_series_priority(entries: list[DicomVolumeEntry]) -> tuple[bool, int, bool]:
    dataset = entries[0].dataset
    modality = get_dicom_str(dataset, "Modality")
    files_count = len(entries)

    return (modality == "CT", files_count, files_count > 1)


def sort_dicom_entries(entries: list[DicomVolumeEntry]) -> list[DicomVolumeEntry]:
    normal = get_dicom_slice_normal(entries)

    if normal is not None:
        return sorted(entries, key=lambda entry: dicom_position_sort_key(entry, normal))

    return sorted(entries, key=dicom_instance_sort_key)


def dicom_position_sort_key(
    entry: DicomVolumeEntry,
    normal: tuple[float, float, float],
) -> tuple[bool, float, bool, int, str]:
    position = get_dicom_float_list(entry.dataset, "ImagePositionPatient")
    instance_number = get_dicom_int(entry.dataset, "InstanceNumber")

    if position is None or len(position) < 3:
        return (True, 0.0, instance_number is None, instance_number or 0, entry.path.name)

    projected_position = sum(position[index] * normal[index] for index in range(3))

    return (
        False,
        projected_position,
        instance_number is None,
        instance_number or 0,
        entry.path.name,
    )


def dicom_instance_sort_key(entry: DicomVolumeEntry) -> tuple[bool, int, str]:
    instance_number = get_dicom_int(entry.dataset, "InstanceNumber")

    return (instance_number is None, instance_number or 0, entry.path.name)


def compute_dicom_spacing_diagnostic(entries: list[DicomVolumeEntry]) -> dict[str, Any]:
    normal = get_dicom_slice_normal(entries)

    if normal is None:
        return {}

    positions = [
        get_projected_dicom_position(entry, normal)
        for entry in entries
    ]
    valid_positions = [position for position in positions if position is not None]

    if len(valid_positions) < 3:
        return {}

    spacings = [
        abs(valid_positions[index + 1] - valid_positions[index])
        for index in range(len(valid_positions) - 1)
    ]
    median_spacing = float(np.median(spacings))

    if median_spacing == 0:
        return {}

    maximum_nonuniformity = max(abs(spacing - median_spacing) for spacing in spacings)

    return {
        "dicom_slice_spacing_mm": median_spacing,
        "dicom_maximum_nonuniformity_mm": float(maximum_nonuniformity),
    }


def get_projected_dicom_position(
    entry: DicomVolumeEntry,
    normal: tuple[float, float, float],
) -> float | None:
    position = get_dicom_float_list(entry.dataset, "ImagePositionPatient")

    if position is None or len(position) < 3:
        return None

    return float(sum(position[index] * normal[index] for index in range(3)))


def get_dicom_slice_normal(entries: list[DicomVolumeEntry]) -> tuple[float, float, float] | None:
    for entry in entries:
        orientation = get_dicom_float_list(entry.dataset, "ImageOrientationPatient")

        if orientation is None or len(orientation) < 6:
            continue

        row = orientation[:3]
        column = orientation[3:6]
        normal = (
            row[1] * column[2] - row[2] * column[1],
            row[2] * column[0] - row[0] * column[2],
            row[0] * column[1] - row[1] * column[0],
        )
        length = math.sqrt(sum(value * value for value in normal))

        if length:
            return tuple(value / length for value in normal)

    return None


def build_prepared_volume(study_id: str, metadata: VolumeMetadataRead) -> PreparedVolumeRead:
    return PreparedVolumeRead(
        filename=VOLUME_FILENAME,
        relative_path=VOLUME_RELATIVE_PATH,
        url=build_file_url(study_id, VOLUME_RELATIVE_PATH),
        metadata=metadata,
    )


def read_volume_metadata(path: Path) -> VolumeMetadataRead:
    try:
        content = path.read_text(encoding="utf-8")
        return VolumeMetadataRead.model_validate_json(content)
    except Exception as exc:
        raise VolumeError(f"Failed to read prepared volume metadata: {exc}") from exc


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
