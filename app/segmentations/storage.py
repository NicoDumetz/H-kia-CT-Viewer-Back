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

from app.segmentations.schemas import SegmentationLabelsDocumentRead, SegmentationRead
from app.studies.storage import get_study_dir


SEGMENTATION_FILENAME = "mask.nii.gz"
LEGACY_SEGMENTATION_FILENAME = "segmentation.nii.gz"
SEGMENTATION_METADATA_FILENAME = "metadata.json"
SEGMENTATION_LABELS_FILENAME = "labels.json"


def get_segmentations_dir(storage_root: str, study_id: str) -> Path:
    return get_study_dir(storage_root, study_id) / "derived" / "segmentations"


def get_source_segmentations_dir(storage_root: str, study_id: str, source: str) -> Path:
    return get_segmentations_dir(storage_root, study_id) / source


def create_segmentation_dir(
    storage_root: str,
    study_id: str,
    segmentation_id: str,
    source: str,
) -> Path:
    segmentation_dir = get_source_segmentations_dir(storage_root, study_id, source) / segmentation_id
    segmentation_dir.mkdir(parents=True, exist_ok=False)
    return segmentation_dir


def get_segmentation_dir(storage_root: str, study_id: str, segmentation_id: str) -> Path:
    return get_segmentations_dir(storage_root, study_id) / segmentation_id


def get_canonical_segmentation_dir(
    storage_root: str,
    study_id: str,
    segmentation_id: str,
    source: str,
) -> Path:
    return get_source_segmentations_dir(storage_root, study_id, source) / segmentation_id


def find_segmentation_dir(storage_root: str, study_id: str, segmentation_id: str) -> Path | None:
    segmentations_dir = get_segmentations_dir(storage_root, study_id)
    candidates = [
        segmentations_dir / "manual" / segmentation_id,
        segmentations_dir / "ai" / segmentation_id,
        segmentations_dir / segmentation_id,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def list_segmentation_dirs(storage_root: str, study_id: str) -> list[Path]:
    segmentations_dir = get_segmentations_dir(storage_root, study_id)

    if not segmentations_dir.is_dir():
        return []

    canonical_dirs: list[Path] = []

    for source in ("ai", "manual"):
        source_dir = segmentations_dir / source

        if not source_dir.is_dir():
            continue

        canonical_dirs.extend(path for path in source_dir.iterdir() if path.is_dir())
    canonical_ids = {path.name for path in canonical_dirs}
    legacy_dirs = [
        path
        for path in segmentations_dir.iterdir()
        if path.is_dir() and path.name not in {"ai", "manual"} and path.name not in canonical_ids
    ]

    return canonical_dirs + legacy_dirs


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


def write_segmentation_labels(
    segmentation_dir: Path,
    labels: SegmentationLabelsDocumentRead,
) -> Path:
    labels_path = segmentation_dir / SEGMENTATION_LABELS_FILENAME
    content = labels.model_dump_json(indent=2)

    labels_path.write_text(content, encoding="utf-8")
    return labels_path


def read_segmentation_labels(
    segmentation_dir: Path,
) -> SegmentationLabelsDocumentRead | None:
    labels_path = segmentation_dir / SEGMENTATION_LABELS_FILENAME

    if not labels_path.is_file():
        return None

    return SegmentationLabelsDocumentRead.model_validate_json(
        labels_path.read_text(encoding="utf-8")
    )
