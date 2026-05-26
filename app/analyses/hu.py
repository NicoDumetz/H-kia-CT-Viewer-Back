# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : hu.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path

import nibabel as nib
import numpy as np

from app.analyses.schemas import (
    AnalysisResultRead,
    LabelHuResultRead,
    LabelHuStatsRead,
)
from app.segmentations.labels import get_label_name


class HuAnalysisError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def compute_label_hu_statistics(
    analysis_id: str,
    study_id: str,
    module_id: str,
    segmentation_id: str,
    roi_mode: str,
    volume_path: Path,
    segmentation_path: Path,
    label_ids: list[int] | None,
    label_names: dict[int, str],
) -> AnalysisResultRead:
    volume_image, volume_array = load_nifti_array(volume_path, "CT volume")
    segmentation_image, segmentation_array = load_nifti_array(segmentation_path, "segmentation")
    spacing = [float(value) for value in volume_image.header.get_zooms()[: volume_array.ndim]]

    if volume_array.shape != segmentation_array.shape:
        raise HuAnalysisError("CT volume and segmentation shapes do not match")

    selected_labels = resolve_selected_labels(segmentation_array, label_ids)
    labels = [
        compute_single_label_stats(
            int(label_id),
            volume_array,
            segmentation_array,
            spacing,
            label_names,
        )
        for label_id in selected_labels
    ]

    return AnalysisResultRead(
        id=analysis_id,
        study_id=study_id,
        module_id=module_id,
        segmentation_id=segmentation_id,
        roi_mode=roi_mode,
        labels_count=len([label for label in labels if label.voxel_count > 0]),
        labels=labels,
    )


def load_nifti_array(path: Path, name: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise HuAnalysisError(f"Failed to read {name} NIfTI: {exc}") from exc

    return image, array


def resolve_selected_labels(
    segmentation_array: np.ndarray,
    label_ids: list[int] | None,
) -> list[int]:
    if label_ids is not None:
        return sorted({int(label_id) for label_id in label_ids if int(label_id) != 0})

    return [
        int(value)
        for value in sorted(np.unique(segmentation_array), key=lambda item: float(item))
        if int(value) != 0
    ]


def compute_single_label_stats(
    label_id: int,
    volume_array: np.ndarray,
    segmentation_array: np.ndarray,
    spacing: list[float],
    label_names: dict[int, str],
) -> LabelHuResultRead:
    mask = segmentation_array == label_id
    values = np.asarray(volume_array[mask])
    voxel_count = int(values.size)
    volume_mm3 = float(voxel_count * compute_voxel_volume(spacing))

    return LabelHuResultRead(
        label_id=label_id,
        name=get_label_name(label_id, label_names),
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        hu=compute_hu_stats(values),
    )


def compute_hu_stats(values: np.ndarray) -> LabelHuStatsRead:
    if values.size == 0:
        return LabelHuStatsRead(
            mean=None,
            median=None,
            std=None,
            min=None,
            max=None,
            p1=None,
            p5=None,
            p25=None,
            p75=None,
            p95=None,
            p99=None,
        )

    return LabelHuStatsRead(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p1=float(np.percentile(values, 1)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )


def compute_voxel_volume(spacing: list[float]) -> float:
    values = spacing[:3] if len(spacing) >= 3 else spacing

    if not values:
        return 0.0

    return float(np.prod(values))
