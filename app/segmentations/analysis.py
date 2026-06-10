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

import nibabel as nib
import numpy as np

from app.segmentations.labels import get_label_color, get_label_group, get_label_name
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
    validate_segmentation_array(array)
    spacing = [float(value) for value in image.header.get_zooms()[: array.ndim]]
    labels = compute_label_stats(array, spacing, label_names)

    return SegmentationMetadataRead(
        shape=[int(value) for value in array.shape],
        spacing=spacing,
        labels_count=len(labels),
        present_labels_count=len(labels),
        labels=labels,
    )


def validate_segmentation_array(array: np.ndarray) -> None:
    if array.ndim != 3:
        raise SegmentationAnalysisError("Segmentation mask must be a 3D NIfTI volume")

    if not contains_integer_labels(array):
        raise SegmentationAnalysisError("Segmentation mask must contain integer labels")


def contains_integer_labels(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.integer):
        return True

    finite = np.isfinite(array)

    if not bool(np.all(finite)):
        return False

    return bool(np.allclose(array, np.rint(array), atol=0.0, rtol=0.0))


def compute_label_stats(
    array: np.ndarray,
    spacing: list[float],
    label_names: dict[int, str],
) -> list[SegmentationLabelRead]:
    labels: list[SegmentationLabelRead] = []
    voxel_volume = compute_voxel_volume(spacing)
    foreground_coordinates = np.argwhere(array != 0)

    if foreground_coordinates.size == 0:
        return labels

    foreground_values = array[tuple(foreground_coordinates.T)]
    sort_order = np.argsort(foreground_values, kind="stable")
    sorted_values = foreground_values[sort_order]
    sorted_coordinates = foreground_coordinates[sort_order]
    label_values, first_indices, counts = np.unique(
        sorted_values,
        return_index=True,
        return_counts=True,
    )

    for label_value, first_index, count in zip(label_values, first_indices, counts):
        label_id = int(label_value)
        label_name = get_label_name(label_id, label_names)
        coordinates = sorted_coordinates[first_index : first_index + count]
        voxel_count = int(count)
        labels.append(
            SegmentationLabelRead(
                id=label_id,
                label_id=label_id,
                name=label_name,
                group=get_label_group(label_name),
                present=True,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count * voxel_volume),
                color=get_label_color(label_id),
                opacity=0.45,
                bbox_ijk=compute_bbox(coordinates),
                center_ijk=compute_center(coordinates),
            )
        )

    return labels


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
