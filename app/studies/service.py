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

from pathlib import Path
from typing import Any

import nibabel as nib
import pydicom
from fastapi import UploadFile

from app.core.config import Settings
from app.studies.detector import detect_file_type, detect_input_type
from app.studies.schemas import InputType, StudyImportResponse
from app.studies.storage import create_source_dir, create_study_id, save_upload_files


async def import_study(files: list[UploadFile], settings: Settings) -> StudyImportResponse:
    study_id = create_study_id()
    source_dir = create_source_dir(settings.storage_root, study_id)
    saved_paths = await save_upload_files(files, source_dir)
    input_type = detect_input_type(saved_paths)
    metadata = extract_metadata(saved_paths, input_type)

    return StudyImportResponse(
        id=study_id,
        status="imported",
        input_type=input_type,
        files_count=len(saved_paths),
        metadata=metadata,
    )


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
