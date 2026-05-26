# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : storage.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path

from app.segmentations.schemas import SegmentationRead
from app.studies.storage import get_study_dir


SEGMENTATION_FILENAME = "segmentation.nii.gz"
SEGMENTATION_METADATA_FILENAME = "metadata.json"


def get_segmentations_dir(storage_root: str, study_id: str) -> Path:
    return get_study_dir(storage_root, study_id) / "derived" / "segmentations"


def create_segmentation_dir(storage_root: str, study_id: str, segmentation_id: str) -> Path:
    segmentation_dir = get_segmentations_dir(storage_root, study_id) / segmentation_id
    segmentation_dir.mkdir(parents=True, exist_ok=False)
    return segmentation_dir


def get_segmentation_dir(storage_root: str, study_id: str, segmentation_id: str) -> Path:
    return get_segmentations_dir(storage_root, study_id) / segmentation_id


def list_segmentation_dirs(storage_root: str, study_id: str) -> list[Path]:
    segmentations_dir = get_segmentations_dir(storage_root, study_id)

    if not segmentations_dir.is_dir():
        return []

    return [path for path in segmentations_dir.iterdir() if path.is_dir()]


def write_segmentation_metadata(
    segmentation_dir: Path,
    segmentation: SegmentationRead,
) -> Path:
    metadata_path = segmentation_dir / SEGMENTATION_METADATA_FILENAME
    content = segmentation.model_dump_json(indent=2)

    metadata_path.write_text(content, encoding="utf-8")
    return metadata_path


def read_segmentation_metadata(segmentation_dir: Path) -> SegmentationRead | None:
    metadata_path = segmentation_dir / SEGMENTATION_METADATA_FILENAME

    if not metadata_path.is_file():
        return None

    return SegmentationRead.model_validate_json(metadata_path.read_text(encoding="utf-8"))
