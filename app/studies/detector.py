# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : detector.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path

import pydicom

from app.studies.schemas import InputType


def detect_file_type(path: Path) -> InputType:
    filename = path.name
    suffixes = path.suffixes

    if filename.upper() == "DICOMDIR":
        return "dicomdir"

    if path.suffix == ".nii" or suffixes[-2:] == [".nii", ".gz"]:
        return "nifti"

    try:
        pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return "unknown"

    return "dicom"


def detect_input_type(paths: list[Path]) -> InputType:
    detected_types = [detect_file_type(path) for path in paths]
    recognized_types = {item for item in detected_types if item != "unknown"}

    if "dicomdir" in recognized_types:
        return "dicomdir"

    if len(recognized_types) == 1:
        return recognized_types.pop()

    return "unknown"
