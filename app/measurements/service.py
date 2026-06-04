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
# Created     : Thursday June 04 2026
#
# =============================================================

from pathlib import Path

import nibabel as nib
import numpy as np

from app.core.config import Settings
from app.measurements.schemas import (
    HuCircleMeasurementCreate,
    HuCircleMeasurementRead,
    HuCircleStatsRead,
    MeasurementPlane,
)
from app.studies.manifest import read_manifest
from app.studies.storage import get_study_dir
from app.studies.volume import VOLUME_RELATIVE_PATH


class MeasurementError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


PLANE_AXES: dict[MeasurementPlane, tuple[int, int, int]] = {
    "sagittal": (0, 1, 2),
    "coronal": (1, 0, 2),
    "axial": (2, 0, 1),
}


def compute_hu_circle_measurement(
    study_id: str,
    payload: HuCircleMeasurementCreate,
    settings: Settings,
) -> HuCircleMeasurementRead:
    study_dir = get_existing_study_dir(settings, study_id)
    volume_path = study_dir / VOLUME_RELATIVE_PATH

    if not volume_path.is_file():
        raise MeasurementError("Prepared volume not found.", status_code=404)

    validate_world_point(payload.center_world, "center_world")

    if payload.edge_world is not None:
        validate_world_point(payload.edge_world, "edge_world")

    image, array = load_volume(volume_path)
    radius_mm = resolve_radius_mm(payload)

    if radius_mm <= 0:
        raise MeasurementError("radius_mm must be greater than 0.")

    values, voxel_count, area_mm2 = extract_circle_values(
        image=image,
        array=array,
        plane=payload.plane,
        center_world=payload.center_world,
        radius_mm=radius_mm,
    )

    if voxel_count == 0:
        raise MeasurementError("ROI does not intersect the prepared volume.")

    return HuCircleMeasurementRead(
        plane=payload.plane,
        center_world=[float(value) for value in payload.center_world],
        radius_mm=float(radius_mm),
        voxel_count=voxel_count,
        area_mm2=area_mm2,
        hu=compute_hu_stats(values),
    )


def get_existing_study_dir(settings: Settings, study_id: str) -> Path:
    study_dir = get_study_dir(settings.storage_root, study_id)
    manifest = read_manifest(study_dir)

    if manifest is None:
        raise MeasurementError("Study not found.", status_code=404)

    return study_dir


def load_volume(volume_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(volume_path))
        array = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise MeasurementError(f"Failed to read prepared volume: {exc}") from exc

    if array.ndim < 3:
        raise MeasurementError("Prepared volume must be 3D.")

    return image, array


def validate_world_point(point: list[float], field_name: str) -> None:
    if len(point) != 3:
        raise MeasurementError(f"{field_name} must contain 3 coordinates.")

    if not all(np.isfinite(float(value)) for value in point):
        raise MeasurementError(f"{field_name} must contain finite coordinates.")


def resolve_radius_mm(payload: HuCircleMeasurementCreate) -> float:
    if payload.radius_mm is not None:
        return float(payload.radius_mm)

    if payload.edge_world is None:
        raise MeasurementError("Either edge_world or radius_mm is required.")

    center = np.asarray(payload.center_world, dtype=float)
    edge = np.asarray(payload.edge_world, dtype=float)

    return float(np.linalg.norm(edge - center))


def extract_circle_values(
    image: nib.Nifti1Image,
    array: np.ndarray,
    plane: MeasurementPlane,
    center_world: list[float],
    radius_mm: float,
) -> tuple[np.ndarray, int, float]:
    fixed_axis, first_axis, second_axis = PLANE_AXES[plane]
    center_voxel = world_to_voxel(image.affine, center_world)
    fixed_index = int(round(float(center_voxel[fixed_axis])))
    shape = array.shape[:3]

    if fixed_index < 0 or fixed_index >= shape[fixed_axis]:
        raise MeasurementError("ROI center is outside the prepared volume.")

    spacing = get_voxel_spacing(image)
    first_indices = np.arange(shape[first_axis], dtype=float)
    second_indices = np.arange(shape[second_axis], dtype=float)
    first_grid, second_grid = np.meshgrid(first_indices, second_indices, indexing="ij")
    first_distance = (first_grid - float(center_voxel[first_axis])) * spacing[first_axis]
    second_distance = (second_grid - float(center_voxel[second_axis])) * spacing[second_axis]
    mask = (first_distance**2 + second_distance**2) <= radius_mm**2
    slice_values = take_plane_slice(array, fixed_axis, fixed_index)
    values = np.asarray(slice_values[mask], dtype=float)
    voxel_count = int(values.size)
    area_mm2 = float(voxel_count * spacing[first_axis] * spacing[second_axis])

    return values, voxel_count, area_mm2


def world_to_voxel(affine: np.ndarray, world: list[float]) -> np.ndarray:
    try:
        inverse_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise MeasurementError("Prepared volume affine is not invertible.") from exc

    homogeneous_world = np.asarray([world[0], world[1], world[2], 1.0], dtype=float)

    return inverse_affine @ homogeneous_world


def get_voxel_spacing(image: nib.Nifti1Image) -> list[float]:
    zooms = image.header.get_zooms()[:3]

    return [abs(float(value)) or 1.0 for value in zooms]


def take_plane_slice(array: np.ndarray, fixed_axis: int, fixed_index: int) -> np.ndarray:
    if fixed_axis == 0:
        return array[fixed_index, :, :]

    if fixed_axis == 1:
        return array[:, fixed_index, :]

    return array[:, :, fixed_index]


def compute_hu_stats(values: np.ndarray) -> HuCircleStatsRead:
    return HuCircleStatsRead(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p5=float(np.percentile(values, 5)),
        p95=float(np.percentile(values, 95)),
    )
