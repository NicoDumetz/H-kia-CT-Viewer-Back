from pathlib import Path

import nibabel as nib
import numpy as np


def test_create_and_read_label_hu_analyses(client, published_segmentation, create_hu_analysis):
    study_id = published_segmentation["study_id"]
    segmentation_id = published_segmentation["segmentation"]["id"]
    analysis_all = create_hu_analysis(study_id, segmentation_id)
    analysis_one = create_hu_analysis(study_id, segmentation_id, label_ids=[1])
    analysis_absent = create_hu_analysis(study_id, segmentation_id, label_ids=[999])
    list_response = client.get(f"/studies/{study_id}/analyses")
    get_response = client.get(f"/studies/{study_id}/analyses/{analysis_all['id']}")
    result_response = client.get(f"/studies/{study_id}/analyses/{analysis_absent['id']}/result")
    absent_label = result_response.json()["labels"][0]

    assert analysis_all["status"] == "succeeded"
    assert analysis_one["input"]["label_ids"] == [1]
    assert list_response.status_code == 200
    assert get_response.status_code == 200
    assert result_response.status_code == 200
    assert absent_label["label_id"] == 999
    assert absent_label["voxel_count"] == 0
    assert absent_label["hu"]["mean"] is None


def test_invalid_analysis_roi_mode_returns_422(client, published_segmentation):
    study_id = published_segmentation["study_id"]
    segmentation_id = published_segmentation["segmentation"]["id"]
    response = client.post(
        f"/studies/{study_id}/analyses",
        json={
            "module_id": "segmentation_label_hu_statistics",
            "segmentation_id": segmentation_id,
            "roi_mode": "trabecular",
        },
    )

    assert response.status_code == 422


def test_unknown_analysis_module_returns_422(client, published_segmentation):
    study_id = published_segmentation["study_id"]
    segmentation_id = published_segmentation["segmentation"]["id"]
    response = client.post(
        f"/studies/{study_id}/analyses",
        json={
            "module_id": "missing_module",
            "segmentation_id": segmentation_id,
            "roi_mode": "whole_label",
        },
    )

    assert response.status_code == 422


def test_missing_analysis_segmentation_returns_404(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/analyses",
        json={
            "module_id": "segmentation_label_hu_statistics",
            "segmentation_id": "missing-segmentation",
            "roi_mode": "whole_label",
        },
    )

    assert response.status_code == 404


def test_missing_analysis_returns_404(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.get(f"/studies/{study_id}/analyses/missing-analysis")

    assert response.status_code == 404


def test_analysis_shape_mismatch_returns_422(client, published_segmentation):
    study_id = published_segmentation["study_id"]
    segmentation_id = published_segmentation["segmentation"]["id"]
    segmentation_path = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived/segmentations"
        / segmentation_id
        / "segmentation.nii.gz"
    )
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    nib.save(image, str(segmentation_path))

    response = client.post(
        f"/studies/{study_id}/analyses",
        json={
            "module_id": "segmentation_label_hu_statistics",
            "segmentation_id": segmentation_id,
            "roi_mode": "whole_label",
        },
    )

    assert response.status_code == 422
