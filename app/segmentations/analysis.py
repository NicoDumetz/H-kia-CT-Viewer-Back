# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : analysis.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from app.segmentations.labels import get_label_name
from app.segmentations.schemas import (
    SegmentationBBoxRead,
    SegmentationLabelRead,
    SegmentationMetadataRead,
)


class SegmentationAnalysisError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def load_segmentation_array(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise SegmentationAnalysisError(f"Failed to read segmentation NIfTI: {exc}") from exc

    return image, array


def compute_segmentation_metadata(
    path: Path,
    label_names: dict[int, str],
) -> SegmentationMetadataRead:
    image, array = load_segmentation_array(path)
    spacing = [float(value) for value in image.header.get_zooms()[: array.ndim]]
    labels = compute_label_stats(array, spacing, label_names)

    return SegmentationMetadataRead(
        shape=[int(value) for value in array.shape],
        spacing=spacing,
        labels_count=len(labels),
        labels=labels,
    )


def compute_label_stats(
    array: np.ndarray,
    spacing: list[float],
    label_names: dict[int, str],
) -> list[SegmentationLabelRead]:
    labels: list[SegmentationLabelRead] = []
    label_values = get_foreground_labels(array)
    voxel_volume = compute_voxel_volume(spacing)

    for label_value in label_values:
        label_id = int(label_value)
        coordinates = np.argwhere(array == label_value)
        voxel_count = int(coordinates.shape[0])
        labels.append(
            SegmentationLabelRead(
                label_id=label_id,
                name=get_label_name(label_id, label_names),
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count * voxel_volume),
                bbox_ijk=compute_bbox(coordinates),
                center_ijk=compute_center(coordinates),
            )
        )

    return labels


def get_foreground_labels(array: np.ndarray) -> list[Any]:
    labels = np.unique(array)
    foreground_labels = [value for value in labels if value != 0]

    return sorted(foreground_labels, key=lambda value: float(value))


def compute_voxel_volume(spacing: list[float]) -> float:
    values = spacing[:3] if len(spacing) >= 3 else spacing

    if not values:
        return 0.0

    return float(np.prod(values))


def compute_bbox(coordinates: np.ndarray) -> SegmentationBBoxRead:
    if coordinates.size == 0:
        return SegmentationBBoxRead(min=[], max=[])

    return SegmentationBBoxRead(
        min=[int(value) for value in coordinates.min(axis=0)],
        max=[int(value) for value in coordinates.max(axis=0)],
    )


def compute_center(coordinates: np.ndarray) -> list[float]:
    if coordinates.size == 0:
        return []

    return [float(value) for value in coordinates.mean(axis=0)]
